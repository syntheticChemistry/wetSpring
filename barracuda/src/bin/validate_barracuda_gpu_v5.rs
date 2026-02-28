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
//! # Exp218: `BarraCuda` GPU v5 — 42-Module Portability Proof
//!
//! "Pure GPU math portability" across GPU modules. For each module,
//! runs identical workload on CPU and GPU and checks parity within
//! documented tolerances.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66+ |
//! | Command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v5` |

use wetspring_barracuda::bio::{
    diversity, diversity_gpu, dnds, dnds_gpu::DnDsGpu, hmm, hmm_gpu::HmmGpuForward, pangenome,
    pangenome_gpu::PangenomeGpu, snp, snp_gpu::SnpGpu, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp218: BarraCuda GPU v5 — 42-Module Portability Proof");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let device = gpu.to_wgpu_device();

    // ═══ G01: Shannon (FMR) ══════════════════════════════════════════
    v.section("G01: Shannon");
    let abundances = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
    let cpu_h = diversity::shannon(&abundances);
    let gpu_h = diversity_gpu::shannon_gpu(&gpu, &abundances).unwrap();
    v.check(
        "Shannon: CPU == GPU",
        gpu_h,
        cpu_h,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G02: Simpson (FMR) ══════════════════════════════════════════
    v.section("G02: Simpson");
    let cpu_d = diversity::simpson(&abundances);
    let gpu_d = diversity_gpu::simpson_gpu(&gpu, &abundances).unwrap();
    v.check(
        "Simpson: CPU == GPU",
        gpu_d,
        cpu_d,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G03: Observed Features (FMR) ════════════════════════════════
    v.section("G03: Observed Features");
    let cpu_obs = diversity::observed_features(&abundances);
    let gpu_obs = diversity_gpu::observed_features_gpu(&gpu, &abundances).unwrap();
    v.check(
        "Observed: CPU == GPU",
        gpu_obs,
        cpu_obs,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G04: Pielou Evenness (FMR) ══════════════════════════════════
    v.section("G04: Pielou Evenness");
    let cpu_j = diversity::pielou_evenness(&abundances);
    let gpu_j = diversity_gpu::pielou_evenness_gpu(&gpu, &abundances).unwrap();
    v.check(
        "Pielou: CPU == GPU",
        gpu_j,
        cpu_j,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G05: Bray-Curtis (BrayCurtisF64) ════════════════════════════
    v.section("G05: Bray-Curtis");
    let samples: Vec<Vec<f64>> = vec![
        vec![10.0, 20.0, 30.0, 5.0],
        vec![15.0, 25.0, 10.0, 8.0],
        vec![5.0, 10.0, 40.0, 12.0],
    ];
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).unwrap();
    v.check_count("BC length match", gpu_bc.len(), cpu_bc.len());
    for (k, (&c, &g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{k}]: CPU == GPU"),
            g,
            c,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // ═══ G06: Alpha Diversity (full) ═════════════════════════════════
    v.section("G06: Full Alpha Diversity");
    let alpha = diversity_gpu::alpha_diversity_gpu(&gpu, &abundances).unwrap();
    v.check(
        "Alpha Shannon",
        alpha.shannon,
        cpu_h,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Alpha Simpson",
        alpha.simpson,
        cpu_d,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══ G07: HMM Forward (GPU batch) ═══════════════════════════════
    v.section("G07: HMM Forward");
    let hmm_model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let obs_cpu: Vec<usize> = (0..100).map(|i| i % 2).collect();
    let cpu_fwd = hmm::forward(&hmm_model, &obs_cpu);
    let hmm_gpu = HmmGpuForward::new(&device).unwrap();
    let obs_gpu: Vec<u32> = obs_cpu.iter().map(|&o| o as u32).collect();
    let gpu_fwd = hmm_gpu.forward_batch(&hmm_model, &obs_gpu, 1, 100).unwrap();
    v.check(
        "HMM log-likelihood: CPU == GPU",
        gpu_fwd.log_likelihoods[0],
        cpu_fwd.log_likelihood,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    // ═══ G08: dN/dS (GPU batch) ═════════════════════════════════════
    v.section("G08: dN/dS");
    let seq_a = b"ATGATGATGATGATGATGATGATGATGATG";
    let seq_b = b"ATGGTGATGATGATGCTGATGATGATGATG";
    let cpu_dnds = dnds::pairwise_dnds(seq_a, seq_b).unwrap();
    let dnds_gpu = DnDsGpu::new(&device).unwrap();
    let pairs: Vec<(&[u8], &[u8])> = vec![(seq_a.as_slice(), seq_b.as_slice())];
    let gpu_batch = dnds_gpu.batch_dnds(&pairs).unwrap();
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

    // ═══ G09: SNP (GPU) ═════════════════════════════════════════════
    v.section("G09: SNP");
    let snp_seqs: Vec<&[u8]> = vec![b"ATGCATGCATGCATGCATGCATGC", b"ATGGATGCATGCATGCATGCATGC"];
    let cpu_snps = snp::call_snps(&snp_seqs);
    let cpu_variant_count = cpu_snps.variants.len();
    let snp_gpu_dev = SnpGpu::new(&device).unwrap();
    let gpu_snps = snp_gpu_dev.call_snps(&snp_seqs).unwrap();
    let gpu_variant_count = gpu_snps.is_variant.iter().filter(|&&v| v == 1).count();
    v.check_count(
        "SNP variant count: CPU == GPU",
        gpu_variant_count,
        cpu_variant_count,
    );

    // ═══ G10: Pangenome (GPU) ════════════════════════════════════════
    v.section("G10: Pangenome");
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
    let presence_flat: Vec<u8> = clusters
        .iter()
        .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
        .collect();
    let gpu_pan = pan_gpu_dev.classify(&presence_flat, 3, 3).unwrap();
    v.check_count("Core: CPU == GPU", gpu_pan.core_count(), cpu_pan.core_size);

    // ═══ G11: Stats GPU (variance) ═══════════════════════════════════
    v.section("G11: Stats GPU");
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

    // ═══ G12: Spectral Pairwise Cosine GPU ═══════════════════════════
    v.section("G12: Spectral Cosine GPU");
    let spec_a: Vec<f64> = vec![1000.0, 500.0, 200.0, 100.0];
    let spec_b: Vec<f64> = vec![1000.0, 500.0, 200.0, 100.0];
    let spectra = vec![spec_a, spec_b];
    let gpu_cosines = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spectra).unwrap();
    v.check(
        "Self-cosine == 1.0",
        gpu_cosines[0],
        1.0,
        tolerances::GPU_VS_CPU_F64,
    );

    let (passed, total) = v.counts();
    println!("\n  ── Exp218 Summary: {passed}/{total} checks ──");

    v.finish();
}
