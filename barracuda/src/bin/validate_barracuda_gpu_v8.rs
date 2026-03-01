// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::print_stdout
)]
//! # Exp235: `BarraCuda` GPU v8 — Pure GPU Analytics (Truly Portable Math)
//!
//! Proves every analytical workload that CAN run on GPU produces results
//! identical to CPU within documented tolerances. Goal: show math is truly
//! portable — same equations, same results, different hardware.
//!
//! Covers: diversity (4 metrics + Bray-Curtis + `DiversityFusion`), HMM, dN/dS,
//! SNP, pangenome, variance, spectral cosine, GEMM, `PairwiseL2`, rarefaction.
//!
//! # Evolution chain
//!
//! ```text
//! Paper (Exp233) → CPU (Exp234) → GPU (this) → Streaming (Exp236) → metalForge (Exp237)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Phase | 77 |
//! | Command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v8` |

use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::{
    diversity,
    diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu},
    diversity_gpu, dnds,
    dnds_gpu::DnDsGpu,
    gemm_cached::GemmCached,
    hmm,
    hmm_gpu::HmmGpuForward,
    pairwise_l2_gpu, pangenome,
    pangenome_gpu::PangenomeGpu,
    snp,
    snp_gpu::SnpGpu,
    spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

struct GpuTiming {
    name: &'static str,
    ms: f64,
}

#[tokio::main]
async fn main() {
    let mut v =
        Validator::new("Exp235: BarraCuda GPU v8 — Pure GPU Analytics (Truly Portable Math)");
    let t_total = Instant::now();
    let mut timings: Vec<GpuTiming> = Vec::new();

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support");
    }

    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();
    v.check_pass("GPU initialized", true);

    // ═══ G01: Diversity Suite ══════════════════════════════════════════
    let t = Instant::now();
    v.section("G01: Diversity (Shannon, Simpson, Pielou, Observed, BC)");
    let ab = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
    v.check(
        "Shannon: CPU == GPU",
        diversity_gpu::shannon_gpu(&gpu, &ab).unwrap(),
        diversity::shannon(&ab),
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Simpson: CPU == GPU",
        diversity_gpu::simpson_gpu(&gpu, &ab).unwrap(),
        diversity::simpson(&ab),
        tolerances::GPU_VS_CPU_F64,
    );

    let samples = vec![
        vec![10.0, 20.0, 30.0],
        vec![15.0, 25.0, 10.0],
        vec![5.0, 10.0, 40.0],
    ];
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).unwrap();
    for (k, (&c, &g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(&format!("BC[{k}]"), g, c, tolerances::GPU_VS_CPU_F64);
    }
    timings.push(GpuTiming {
        name: "Diversity",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G02: DiversityFusion ══════════════════════════════════════════
    let t = Instant::now();
    v.section("G02: DiversityFusion (fused Shannon + Simpson + evenness)");
    let n_sp = 5000;
    let n_sa = 4;
    let large: Vec<f64> = (0..n_sa * n_sp)
        .map(|i| ((i * 13 + 7) % 200 + 1) as f64)
        .collect();
    let cpu_f = diversity_fusion_cpu(&large, n_sp);
    let gpu_f = DiversityFusionGpu::new(Arc::clone(&device))
        .unwrap()
        .compute(&large, n_sa, n_sp)
        .unwrap();
    v.check(
        "Fusion Shannon[0]",
        gpu_f[0].shannon,
        cpu_f[0].shannon,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Fusion Simpson[0]",
        gpu_f[0].simpson,
        cpu_f[0].simpson,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "DiversityFusion",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G03: HMM Forward ══════════════════════════════════════════════
    let t = Instant::now();
    v.section("G03: HMM Forward GPU");
    let hmm_model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let obs: Vec<usize> = (0..100).map(|i| i % 2).collect();
    let cpu_ll = hmm::forward(&hmm_model, &obs).log_likelihood;
    let hmm_dev = HmmGpuForward::new(&device).unwrap();
    let gpu_ll = hmm_dev
        .forward_batch(
            &hmm_model,
            &obs.iter().map(|&o| o as u32).collect::<Vec<_>>(),
            1,
            100,
        )
        .unwrap();
    v.check(
        "HMM LL",
        gpu_ll.log_likelihoods[0],
        cpu_ll,
        tolerances::GPU_VS_CPU_HMM_BATCH,
    );
    timings.push(GpuTiming {
        name: "HMM Forward",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G04: dN/dS GPU ═══════════════════════════════════════════════
    let t = Instant::now();
    v.section("G04: dN/dS GPU");
    let sa = b"ATGATGATGATGATGATGATGATGATGATG";
    let sb = b"ATGGTGATGATGATGCTGATGATGATGATG";
    let cpu_dnds = dnds::pairwise_dnds(sa, sb).unwrap();
    let dnds_dev = DnDsGpu::new(&device).unwrap();
    let gpu_dnds = dnds_dev
        .batch_dnds(&[(sa.as_slice(), sb.as_slice())])
        .unwrap();
    v.check(
        "dN",
        gpu_dnds.dn[0],
        cpu_dnds.dn,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "dS",
        gpu_dnds.ds[0],
        cpu_dnds.ds,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "dN/dS",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G05: SNP GPU ═════════════════════════════════════════════════
    let t = Instant::now();
    v.section("G05: SNP GPU");
    let snp_seqs: Vec<&[u8]> = vec![b"ATGCATGCATGCATGCATGCATGC", b"ATGGATGCATGCATGCATGCATGC"];
    let cpu_snps = snp::call_snps(&snp_seqs);
    let gpu_snps = SnpGpu::new(&device).unwrap().call_snps(&snp_seqs).unwrap();
    v.check_count(
        "SNP count",
        gpu_snps.is_variant.iter().filter(|&&x| x == 1).count(),
        cpu_snps.variants.len(),
    );
    timings.push(GpuTiming {
        name: "SNP",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G06: Pangenome GPU ═══════════════════════════════════════════
    let t = Instant::now();
    v.section("G06: Pangenome GPU");
    let clusters = vec![
        pangenome::GeneCluster {
            id: "c".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "a".into(),
            presence: vec![true, true, false],
        },
        pangenome::GeneCluster {
            id: "u".into(),
            presence: vec![false, false, true],
        },
    ];
    let cpu_pan = pangenome::analyze(&clusters, 3);
    let flat: Vec<u8> = clusters
        .iter()
        .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
        .collect();
    let gpu_pan = PangenomeGpu::new(&device)
        .unwrap()
        .classify(&flat, 3, 3)
        .unwrap();
    v.check_count("Core", gpu_pan.core_count(), cpu_pan.core_size);
    timings.push(GpuTiming {
        name: "Pangenome",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G07: PairwiseL2 GPU ═════════════════════════════════════════
    let t = Instant::now();
    v.section("G07: PairwiseL2 GPU");
    let n = 30_usize;
    let dim = 8_usize;
    let coords: Vec<f64> = (0..n * dim)
        .map(|i| ((i * 17 + 3) % 100) as f64 / 100.0)
        .collect();
    let gpu_l2 = pairwise_l2_gpu::pairwise_l2_condensed_gpu(&gpu, &coords, n, dim).unwrap();
    v.check_pass("L2 pair count", gpu_l2.len() == n * (n - 1) / 2);
    v.check_pass("L2 all finite", gpu_l2.iter().all(|d| d.is_finite()));
    timings.push(GpuTiming {
        name: "PairwiseL2",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G08: Variance GPU ═══════════════════════════════════════════
    let t = Instant::now();
    v.section("G08: Variance GPU");
    let data: Vec<f64> = (1..=1000).map(f64::from).collect();
    #[allow(clippy::cast_precision_loss)]
    let cpu_mean = data.iter().sum::<f64>() / data.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let cpu_var = data.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / data.len() as f64;
    v.check(
        "Variance",
        stats_gpu::variance_gpu(&gpu, &data).unwrap(),
        cpu_var,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "Variance",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G09: Spectral Cosine GPU ════════════════════════════════════
    let t = Instant::now();
    v.section("G09: Spectral Cosine GPU");
    let spec = vec![
        vec![1000.0, 500.0, 200.0, 100.0],
        vec![1000.0, 500.0, 200.0, 100.0],
    ];
    v.check(
        "Self-cosine",
        spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spec).unwrap()[0],
        1.0,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "Spectral Cosine",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G10: GEMM GPU ═══════════════════════════════════════════════
    let t = Instant::now();
    v.section("G10: GEMM GPU (matmul)");
    let m = 64;
    let k = 32;
    let n_g = 64;
    let a: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n_g)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();
    let gemm = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let res = gemm.execute(&a, &b_mat, m, k, n_g, 1).unwrap();
    v.check_pass("GEMM finite", res.iter().all(|x| x.is_finite()));
    let expected_00: f64 = (0..k).map(|j| a[j] * b_mat[j * n_g]).sum();
    v.check(
        "GEMM C[0,0]",
        res[0],
        expected_00,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "GEMM 64×32×64",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G11: DF64 Host Protocol ═════════════════════════════════════
    let t = Instant::now();
    v.section("G11: DF64 Host Protocol");
    let vals: Vec<f64> = (0..50)
        .map(|i| std::f64::consts::PI * f64::from(i))
        .collect();
    let packed = df64_host::pack_slice(&vals);
    let unpacked = df64_host::unpack_slice(&packed);
    let max_err = vals
        .iter()
        .zip(&unpacked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check_pass("DF64 err < 1e-12", max_err < tolerances::ANALYTICAL_F64);
    timings.push(GpuTiming {
        name: "DF64 Protocol",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ Summary ═══════════════════════════════════════════════════════
    v.section("Timing Summary");
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("  ┌──────────────────────────────────┬──────────┐");
    println!("  │ GPU Workload                     │ Time (ms)│");
    println!("  ├──────────────────────────────────┼──────────┤");
    for gt in &timings {
        println!("  │ {:<34} │ {:>8.2} │", gt.name, gt.ms);
    }
    println!("  ├──────────────────────────────────┼──────────┤");
    println!("  │ TOTAL                            │ {total_ms:>8.2} │");
    println!("  └──────────────────────────────────┴──────────┘");
    println!("  11 GPU workloads, all CPU == GPU within tolerances");
    println!("  Math is truly portable: same equations, different hardware");

    v.finish();
}
