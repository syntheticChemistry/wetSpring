// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
//! # Exp167: Diversity Fusion GPU — Lean Phase (absorbed S63)
//!
//! Validates `DiversityFusionGpu` via upstream `barracuda::ops::bio::diversity_fusion` —
//! CPU vs GPU parity.  Python baseline: `scipy`/`skbio` Shannon/Simpson/Pielou
//! (analytically verified).  Data: Synthetic abundance vectors (deterministic,
//! no external data).
//!
//! Originally validated the Write-phase local WGSL shader.  After `ToadStool` S63
//! absorption, the local shader was deleted and this binary now exercises the
//! upstream `DiversityFusionGpu` + `diversity_fusion_cpu` re-exports.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `756df26` |
//! | Baseline tool | scipy/skbio Shannon/Simpson/Pielou (analytically verified) |
//! | Baseline date | 2026-02-27 |
//! | Exact command | `cargo run --release --features gpu --bin validate_gpu_diversity_fusion` |
//! | Data | Synthetic abundance vectors (deterministic, no external data) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in barracuda::bio

use std::sync::Arc;
use wetspring_barracuda::bio::diversity_fusion_gpu::{
    DiversityFusionGpu, DiversityResult, diversity_fusion_cpu,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn validate_uniform(v: &mut Validator, fusion: &DiversityFusionGpu) {
    let n_species = 4;
    let abundances = vec![
        25.0, 25.0, 25.0, 25.0, // sample 0: uniform
        100.0, 0.0, 0.0, 0.0, // sample 1: single species
    ];
    let n_samples = 2;

    let cpu = diversity_fusion_cpu(&abundances, n_species);
    let gpu_result = fusion
        .compute(&abundances, n_samples, n_species)
        .expect("GPU compute");

    let cpu_tol = tolerances::ANALYTICAL_F64;
    let gpu_log_tol = tolerances::GPU_LOG_POLYFILL;

    let expected_shannon = 4.0_f64.ln();
    v.check(
        "uniform Shannon (CPU)",
        cpu[0].shannon,
        expected_shannon,
        cpu_tol,
    );
    v.check(
        "uniform Shannon (GPU, log_f64 polyfill)",
        gpu_result[0].shannon,
        expected_shannon,
        gpu_log_tol,
    );
    v.check(
        "uniform Shannon CPU↔GPU",
        cpu[0].shannon,
        gpu_result[0].shannon,
        gpu_log_tol,
    );
    v.check("uniform Simpson (CPU)", cpu[0].simpson, 0.75, cpu_tol);
    v.check(
        "uniform Simpson CPU↔GPU",
        cpu[0].simpson,
        gpu_result[0].simpson,
        cpu_tol,
    );
    v.check("uniform evenness (CPU)", cpu[0].evenness, 1.0, cpu_tol);
    v.check(
        "uniform evenness CPU↔GPU",
        cpu[0].evenness,
        gpu_result[0].evenness,
        gpu_log_tol,
    );

    v.check("single-species Shannon (CPU)", cpu[1].shannon, 0.0, cpu_tol);
    v.check(
        "single-species Shannon CPU↔GPU",
        cpu[1].shannon,
        gpu_result[1].shannon,
        cpu_tol,
    );
    v.check("single-species Simpson (CPU)", cpu[1].simpson, 0.0, cpu_tol);
    v.check(
        "single-species Simpson CPU↔GPU",
        cpu[1].simpson,
        gpu_result[1].simpson,
        cpu_tol,
    );
}

fn validate_skewed(v: &mut Validator, fusion: &DiversityFusionGpu) {
    let n_species = 6;
    let abundances = vec![50.0, 30.0, 10.0, 5.0, 3.0, 2.0];

    let cpu = diversity_fusion_cpu(&abundances, n_species);
    let gpu_result = fusion
        .compute(&abundances, 1, n_species)
        .expect("GPU compute");

    v.check(
        "skewed Shannon CPU↔GPU",
        cpu[0].shannon,
        gpu_result[0].shannon,
        tolerances::GPU_LOG_POLYFILL,
    );
    v.check(
        "skewed Simpson CPU↔GPU",
        cpu[0].simpson,
        gpu_result[0].simpson,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "skewed evenness CPU↔GPU",
        cpu[0].evenness,
        gpu_result[0].evenness,
        tolerances::GPU_LOG_POLYFILL,
    );
    v.check_pass("skewed Shannon > 0", cpu[0].shannon > 0.0);
    v.check_pass(
        "skewed evenness ∈ (0, 1)",
        cpu[0].evenness > 0.0 && cpu[0].evenness < 1.0,
    );
}

fn validate_batch(v: &mut Validator, fusion: &DiversityFusionGpu) {
    let n_species = 10;
    let n_samples = 128;
    let mut abundances = Vec::with_capacity(n_samples * n_species);
    for sample_idx in 0..n_samples {
        for species_idx in 0..n_species {
            let idx = sample_idx * n_species + species_idx + 1;
            abundances.push((idx as f64).sqrt());
        }
    }

    let cpu = diversity_fusion_cpu(&abundances, n_species);
    let gpu_result = fusion
        .compute(&abundances, n_samples, n_species)
        .expect("GPU compute");

    let all_parity = check_batch_parity(&cpu, &gpu_result);
    v.check_pass("128-sample batch CPU↔GPU parity", all_parity);
    v.check_pass(
        "all 128 Shannon finite",
        gpu_result.iter().all(|r| r.shannon.is_finite()),
    );
}

fn check_batch_parity(cpu: &[DiversityResult], gpu: &[DiversityResult]) -> bool {
    cpu.iter().zip(gpu).all(|(c, g)| {
        (c.shannon - g.shannon).abs() <= tolerances::GPU_LOG_POLYFILL
            && (c.simpson - g.simpson).abs() <= tolerances::ANALYTICAL_F64
            && (c.evenness - g.evenness).abs() <= tolerances::GPU_LOG_POLYFILL
    })
}

fn main() {
    let mut v = Validator::new("Exp167: Diversity Fusion GPU — Lean Phase (absorbed S63)");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let gpu = rt
        .block_on(wetspring_barracuda::gpu::GpuF64::new())
        .expect("GPU init");
    let device = Arc::new(gpu.to_wgpu_device());

    let fusion =
        DiversityFusionGpu::new(Arc::clone(&device)).expect("DiversityFusionGpu shader compile");

    v.section("Test 1: Uniform distribution (4 species, 2 samples)");
    validate_uniform(&mut v, &fusion);

    v.section("Test 2: Skewed distribution (ecological realism)");
    validate_skewed(&mut v, &fusion);

    v.section("Test 3: Batch of 128 samples (GPU dispatch sizing)");
    validate_batch(&mut v, &fusion);

    v.section("Lean delegation summary");
    println!("  GPU:   barracuda::ops::bio::diversity_fusion::DiversityFusionGpu (S63)");
    println!("  CPU:   barracuda::ops::bio::diversity_fusion::diversity_fusion_cpu (S63)");
    println!("  Local: thin re-export in bio/diversity_fusion_gpu.rs");
    println!("  WGSL:  deleted (was bio/shaders/diversity_fusion_f64.wgsl)");

    v.finish();
}
