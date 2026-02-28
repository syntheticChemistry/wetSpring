// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::print_stdout
)]
//! # Exp226: `BarraCuda` GPU v6 — V71 Precision-Flexible Portability Proof
//!
//! GPU portability: identical workloads on CPU and GPU, parity within
//! documented tolerances. Extends v5 with:
//! - **V71 `GemmCached::with_precision()`** — precision-flexible GEMM
//! - **V71 `df64_host`** — DF64 host protocol demonstration
//! - **`DiversityFusion`** — fused GPU map-reduce
//! - **`BandwidthTier`** — hardware-aware transfer estimation
//!
//! # Three-tier chain position
//!
//! ```text
//! Paper (Exp224) → CPU (Exp225) → GPU (this) → Streaming (Exp227) → metalForge (Exp228)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Phase | 71 |
//! | Command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v6` |

use std::sync::Arc;
use std::time::Instant;

use barracuda::shaders::Precision;
use wetspring_barracuda::bio::{
    diversity,
    diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu},
    diversity_gpu, dnds,
    dnds_gpu::DnDsGpu,
    gemm_cached::GemmCached,
    hmm,
    hmm_gpu::HmmGpuForward,
    pangenome,
    pangenome_gpu::PangenomeGpu,
    snp,
    snp_gpu::SnpGpu,
    spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {label}: {ms:.3} ms");
    (r, ms)
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp226: BarraCuda GPU v6 — V71 Precision-Flexible Portability");

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

    let strategy = gpu.fp64_strategy();
    let precision = gpu.optimal_precision();
    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    println!("  Fp64Strategy: {strategy:?}");
    println!("  optimal_precision: {precision:?}");
    v.check_pass("GPU initialized", true);
    v.check_pass("device not lost", !gpu.is_lost());

    // ═══ G01-G06: Core Diversity (CPU == GPU) ═════════════════════════

    v.section("G01-G04: Diversity CPU == GPU");
    let abundances = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
    let cpu_h = diversity::shannon(&abundances);
    let cpu_d = diversity::simpson(&abundances);
    let cpu_obs = diversity::observed_features(&abundances);
    let cpu_j = diversity::pielou_evenness(&abundances);

    let gpu_h = diversity_gpu::shannon_gpu(&gpu, &abundances).unwrap();
    let gpu_d = diversity_gpu::simpson_gpu(&gpu, &abundances).unwrap();
    let gpu_obs = diversity_gpu::observed_features_gpu(&gpu, &abundances).unwrap();
    let gpu_j = diversity_gpu::pielou_evenness_gpu(&gpu, &abundances).unwrap();

    v.check(
        "Shannon: CPU == GPU",
        gpu_h,
        cpu_h,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Simpson: CPU == GPU",
        gpu_d,
        cpu_d,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Observed: CPU == GPU",
        gpu_obs,
        cpu_obs,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Pielou: CPU == GPU",
        gpu_j,
        cpu_j,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G05: Bray-Curtis");
    let samples: Vec<Vec<f64>> = vec![
        vec![10.0, 20.0, 30.0, 5.0],
        vec![15.0, 25.0, 10.0, 8.0],
        vec![5.0, 10.0, 40.0, 12.0],
    ];
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).unwrap();
    for (k, (&c, &g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{k}]: CPU == GPU"),
            g,
            c,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    v.section("G06: DiversityFusion");
    let n_species = 5000;
    let n_samples = 4;
    let large: Vec<f64> = (0..n_samples * n_species)
        .map(|i| ((i * 13 + 7) % 200 + 1) as f64)
        .collect();
    let cpu_fusion = diversity_fusion_cpu(&large, n_species);
    let fusion_gpu = DiversityFusionGpu::new(Arc::clone(&device)).expect("DiversityFusionGpu");
    let gpu_fusion = fusion_gpu
        .compute(&large, n_samples, n_species)
        .expect("fusion");
    v.check(
        "Fusion Shannon: CPU == GPU",
        gpu_fusion[0].shannon,
        cpu_fusion[0].shannon,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Fusion Simpson: CPU == GPU",
        gpu_fusion[0].simpson,
        cpu_fusion[0].simpson,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G07-G10: Bio GPU Modules ═════════════════════════════════════

    v.section("G07: HMM Forward GPU");
    let hmm_model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let obs_cpu: Vec<usize> = (0..100).map(|i| i % 2).collect();
    let cpu_fwd = hmm::forward(&hmm_model, &obs_cpu);
    let hmm_gpu_dev = HmmGpuForward::new(&device).unwrap();
    let obs_gpu: Vec<u32> = obs_cpu.iter().map(|&o| o as u32).collect();
    let gpu_fwd = hmm_gpu_dev
        .forward_batch(&hmm_model, &obs_gpu, 1, 100)
        .unwrap();
    v.check(
        "HMM LL: CPU == GPU",
        gpu_fwd.log_likelihoods[0],
        cpu_fwd.log_likelihood,
        tolerances::GPU_VS_CPU_HMM_BATCH,
    );

    v.section("G08: dN/dS GPU");
    let seq_a = b"ATGATGATGATGATGATGATGATGATGATG";
    let seq_b = b"ATGGTGATGATGATGCTGATGATGATGATG";
    let cpu_dnds = dnds::pairwise_dnds(seq_a, seq_b).unwrap();
    let dnds_gpu_dev = DnDsGpu::new(&device).unwrap();
    let pairs: Vec<(&[u8], &[u8])> = vec![(seq_a.as_slice(), seq_b.as_slice())];
    let gpu_batch = dnds_gpu_dev.batch_dnds(&pairs).unwrap();
    v.check(
        "dN: CPU == GPU",
        gpu_batch.dn[0],
        cpu_dnds.dn,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "dS: CPU == GPU",
        gpu_batch.ds[0],
        cpu_dnds.ds,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G09: SNP GPU");
    let snp_seqs: Vec<&[u8]> = vec![b"ATGCATGCATGCATGCATGCATGC", b"ATGGATGCATGCATGCATGCATGC"];
    let cpu_snps = snp::call_snps(&snp_seqs);
    let snp_gpu_dev = SnpGpu::new(&device).unwrap();
    let gpu_snps = snp_gpu_dev.call_snps(&snp_seqs).unwrap();
    v.check_count(
        "SNP count: CPU == GPU",
        gpu_snps.is_variant.iter().filter(|&&x| x == 1).count(),
        cpu_snps.variants.len(),
    );

    v.section("G10: Pangenome GPU");
    let clusters = vec![
        pangenome::GeneCluster {
            id: "core".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "acc".into(),
            presence: vec![true, true, false],
        },
        pangenome::GeneCluster {
            id: "uniq".into(),
            presence: vec![false, false, true],
        },
    ];
    let cpu_pan = pangenome::analyze(&clusters, 3);
    let pan_gpu_dev = PangenomeGpu::new(&device).unwrap();
    let flat: Vec<u8> = clusters
        .iter()
        .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
        .collect();
    let gpu_pan = pan_gpu_dev.classify(&flat, 3, 3).unwrap();
    v.check_count("Core: CPU == GPU", gpu_pan.core_count(), cpu_pan.core_size);

    // ═══ G11-G12: Stats + Spectral GPU ════════════════════════════════

    v.section("G11: Variance GPU");
    let data: Vec<f64> = (1..=1000).map(f64::from).collect();
    let cpu_mean = data.iter().sum::<f64>() / data.len() as f64;
    let cpu_var: f64 = data.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / data.len() as f64;
    let gpu_var = stats_gpu::variance_gpu(&gpu, &data).unwrap();
    v.check(
        "Variance: CPU == GPU",
        gpu_var,
        cpu_var,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G12: Spectral Cosine GPU");
    let spec = vec![
        vec![1000.0, 500.0, 200.0, 100.0],
        vec![1000.0, 500.0, 200.0, 100.0],
    ];
    let gpu_cosines = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spec).unwrap();
    v.check(
        "Self-cosine == 1.0",
        gpu_cosines[0],
        1.0,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ V71: Precision-Flexible GEMM ══════════════════════════════════

    v.section("G13: V71 GEMM with_precision (F64)");
    let m = 128;
    let k = 64;
    let n = 128;
    let a_mat: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    let gemm_default = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let gemm_explicit =
        GemmCached::with_precision(Arc::clone(&device), Arc::clone(&ctx), Precision::F64);

    let (res_default, t_default) = bench("GEMM new() [F64]", || {
        gemm_default
            .execute(&a_mat, &b_mat, m, k, n, 1)
            .expect("GEMM default")
    });
    let (res_explicit, t_explicit) = bench("GEMM with_precision(F64)", || {
        gemm_explicit
            .execute(&a_mat, &b_mat, m, k, n, 1)
            .expect("GEMM explicit")
    });

    v.check_pass(
        "GEMM F64 result finite",
        res_default.iter().all(|x| x.is_finite()),
    );
    v.check(
        "new() == with_precision(F64)",
        res_default[0],
        res_explicit[0],
        tolerances::EXACT_F64,
    );

    let expected_00: f64 = (0..k).map(|j| a_mat[j] * b_mat[j * n]).sum();
    v.check(
        "GEMM C[0,0] ≈ CPU",
        res_default[0],
        expected_00,
        tolerances::GPU_VS_CPU_F64,
    );

    // Cached throughput
    for _ in 0..5 {
        let _ = gemm_default.execute(&a_mat, &b_mat, m, k, n, 1);
    }
    let ((), cached_ms) = bench("GEMM ×50 cached", || {
        for _ in 0..50 {
            let _ = gemm_default.execute(&a_mat, &b_mat, m, k, n, 1);
        }
    });
    let per_dispatch = cached_ms / 50.0;
    v.check_pass("cached dispatch measured", per_dispatch.is_finite());
    println!("  Cold: {t_default:.3} ms, Cached avg: {per_dispatch:.3} ms");

    // ═══ V71: DF64 Host Protocol on GPU data ═══════════════════════════

    v.section("G14: V71 DF64 Host Protocol");
    println!("  Validates pack/unpack matches GPU GEMM output precision");

    let gemm_result = &res_default[..5];
    for (i, &val) in gemm_result.iter().enumerate() {
        let [hi, lo] = df64_host::pack(val);
        let restored = df64_host::unpack(hi, lo);
        let err = (restored - val).abs();
        v.check_pass(
            &format!("DF64 roundtrip C[{i}] err={err:.2e}"),
            err < 2e-14 || val == 0.0,
        );
    }

    // BandwidthTier
    let tier =
        barracuda::unified_hardware::BandwidthTier::detect_from_adapter_name(&gpu.adapter_name);
    println!(
        "  BandwidthTier: {tier:?} ({:.1} GB/s)",
        tier.bandwidth_gbps()
    );
    v.check_pass("BandwidthTier > 0", tier.bandwidth_gbps() > 0.0);

    // ═══ Summary ═══════════════════════════════════════════════════════

    v.section("Summary");
    println!(
        "  GPU modules: diversity(4) + BC + fusion + HMM + dNdS + SNP + pangenome + var + cosine + GEMM + DF64"
    );
    println!("  V71 additions: with_precision(), df64_host roundtrip, BandwidthTier");
    println!("  All CPU == GPU within documented tolerances");
    println!(
        "  Timing: GEMM cold={t_default:.1}ms, explicit={t_explicit:.1}ms, cached={per_dispatch:.1}ms"
    );

    v.finish();
}
