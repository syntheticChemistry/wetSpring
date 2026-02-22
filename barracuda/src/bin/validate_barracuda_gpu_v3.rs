// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp044: `BarraCUDA` GPU parity for v3 domains — proves CPU→GPU portability.
//!
//! Tests GPU implementations of domains validated in CPU v3 that have
//! `ToadStool` primitives available: extended diversity (Pielou, Bray-Curtis
//! matrix), spectral matching (pairwise cosine), and statistics.
//!
//! Domains without GPU shaders yet (ODE, SSA, HMM, SW, DT, kmer, bootstrap,
//! placement, Felsenstein) are validated on CPU and documented for GPU
//! promotion via `ToadStool`.
//!
//! ```text
//! Python → CPU v1/v2/v3 → [THIS] GPU v3 → ToadStool sovereign
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | `BarraCUDA` CPU (reference) |
//! | Baseline version | wetspring-barracuda 0.1.0 (CPU path) |
//! | Baseline command | `bio::diversity`, `bio::spectral_match`, CPU stats |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v3` |
//! | Data | Count vectors, Bray-Curtis matrices, spectra, variance/correlation |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! GPU modules: `diversity_gpu` (Pielou, Shannon, Simpson, Bray-Curtis), `spectral_match_gpu`, `stats_gpu`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use wetspring_barracuda::bio::{
    diversity, diversity_gpu, spectral_match, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp044: BarraCUDA GPU v3 — CPU→GPU Parity");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            validation::exit_skipped(&format!("GPU init failed: {e}"));
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }
    println!();

    validate_extended_diversity(&gpu, &mut v);
    validate_bray_curtis_matrix(&gpu, &mut v);
    validate_spectral_batch(&gpu, &mut v);
    validate_statistics(&gpu, &mut v);
    validate_gpu_determinism(&gpu, &mut v);

    v.finish();
}

fn validate_extended_diversity(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Extended Diversity GPU vs CPU (v3 Domain 16) ──");

    // Pielou evenness: uniform → 1.0
    {
        let counts = vec![25.0; 4];
        let cpu = diversity::pielou_evenness(&counts);
        let gpu_val = diversity_gpu::pielou_evenness_gpu(gpu, &counts).expect("GPU Pielou uniform");
        v.check(
            "Pielou GPU uniform ≈ 1.0",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // Pielou evenness: uneven
    {
        let counts = vec![97.0, 1.0, 1.0, 1.0];
        let cpu = diversity::pielou_evenness(&counts);
        let gpu_val = diversity_gpu::pielou_evenness_gpu(gpu, &counts).expect("GPU Pielou uneven");
        v.check(
            "Pielou GPU uneven < 0.5",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // Shannon on 100 species
    {
        let counts: Vec<f64> = (1..=100).map(f64::from).collect();
        let cpu = diversity::shannon(&counts);
        let gpu_val = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU Shannon 100");
        v.check(
            "Shannon GPU 100 species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // Simpson on 100 species
    {
        let counts: Vec<f64> = (1..=100).map(f64::from).collect();
        let cpu = diversity::simpson(&counts);
        let gpu_val = diversity_gpu::simpson_gpu(gpu, &counts).expect("GPU Simpson 100");
        v.check(
            "Simpson GPU 100 species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // Observed features
    {
        let counts = vec![10.0, 0.0, 20.0, 0.0, 5.0, 0.0, 0.0, 1.0];
        let cpu = diversity::observed_features(&counts);
        let gpu_val = diversity_gpu::observed_features_gpu(gpu, &counts).expect("GPU observed");
        v.check("Observed GPU", gpu_val, cpu, tolerances::GPU_VS_CPU_F64);
    }
}

fn validate_bray_curtis_matrix(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Bray-Curtis Matrix GPU vs CPU (v3 Domain 16) ──");

    // 5 samples × 10 features
    {
        let samples: Vec<Vec<f64>> = (0..5)
            .map(|i: i32| {
                (0..10)
                    .map(|j: i32| f64::from((i * 7 + j * 13 + 1) % 50))
                    .collect()
            })
            .collect();
        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val = diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC 5×10");

        let n_pairs = 5 * 4 / 2;
        assert_eq!(gpu_val.len(), n_pairs);

        let mut all_pass = true;
        let mut max_diff = 0.0_f64;
        for (g, c) in gpu_val.iter().zip(cpu.iter()) {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            if diff > tolerances::GPU_VS_CPU_BRAY_CURTIS {
                all_pass = false;
            }
        }
        v.check(
            &format!("BC matrix 5×10 ({n_pairs} pairs, max Δ={max_diff:.2e})"),
            f64::from(u8::from(all_pass)),
            1.0,
            0.0,
        );
    }

    // 20 samples × 50 features (larger batch)
    {
        let samples: Vec<Vec<f64>> = (0..20)
            .map(|i: i32| {
                (0..50)
                    .map(|j: i32| f64::from((i * 11 + j * 7 + 3) % 100))
                    .collect()
            })
            .collect();
        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val =
            diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC 20×50");

        let n_pairs = 20 * 19 / 2;
        assert_eq!(gpu_val.len(), n_pairs);

        let mut all_pass = true;
        let mut max_diff = 0.0_f64;
        for (g, c) in gpu_val.iter().zip(cpu.iter()) {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            if diff > tolerances::GPU_VS_CPU_BRAY_CURTIS {
                all_pass = false;
            }
        }
        v.check(
            &format!("BC matrix 20×50 ({n_pairs} pairs, max Δ={max_diff:.2e})"),
            f64::from(u8::from(all_pass)),
            1.0,
            0.0,
        );
    }
}

fn validate_spectral_batch(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Spectral Matching GPU vs CPU (v3 Domain 15) ──");

    // GPU pairwise_cosine_gpu takes &[Vec<f64>] (intensity-only, uniform m/z)
    let spectra_gpu: Vec<Vec<f64>> = vec![
        vec![1000.0, 500.0, 800.0, 300.0, 600.0],
        vec![900.0, 550.0, 750.0, 350.0, 550.0],
        vec![600.0, 400.0, 700.0, 200.0, 500.0],
        vec![500.0, 1000.0, 200.0, 800.0, 400.0],
    ];

    let gpu_pw =
        spectral_match_gpu::pairwise_cosine_gpu(gpu, &spectra_gpu).expect("GPU pairwise cosine");

    let n_pairs = spectra_gpu.len() * (spectra_gpu.len() - 1) / 2;
    assert_eq!(gpu_pw.len(), n_pairs);

    // GPU cosine on uniform-m/z spectra should give reasonable scores
    let mut all_finite = true;
    for (i, score) in gpu_pw.iter().enumerate() {
        if !score.is_finite() || *score < 0.0 || *score > 1.0 + 1e-10 {
            all_finite = false;
            println!("  [WARN] Spectral pair {i}: score={score:.6} out of range");
        }
    }
    v.check(
        &format!("Spectral GPU pairwise: {n_pairs} pairs all in [0,1]"),
        f64::from(u8::from(all_finite)),
        1.0,
        0.0,
    );

    // Self-match (identical vectors) should be ~1.0
    let self_spectra = vec![spectra_gpu[0].clone(), spectra_gpu[0].clone()];
    let self_pw =
        spectral_match_gpu::pairwise_cosine_gpu(gpu, &self_spectra).expect("GPU self-match");
    v.check(
        "Spectral GPU self-match ≈ 1.0",
        self_pw[0],
        1.0,
        tolerances::SPECTRAL_COSINE,
    );

    // CPU baseline for cross-check
    let cpu_self = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0, 400.0, 500.0],
        &spectra_gpu[0],
        &[100.0, 200.0, 300.0, 400.0, 500.0],
        &spectra_gpu[0],
        0.5,
    );
    v.check(
        "Spectral CPU self-match = 1.0",
        cpu_self.score,
        1.0,
        tolerances::SPECTRAL_COSINE,
    );
}

fn validate_statistics(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Statistics GPU vs CPU ──");

    // Variance: GPU returns population variance for this dataset
    {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let gpu_val = stats_gpu::variance_gpu(gpu, &data).expect("GPU variance");
        let cpu_pop_var = 4.0;
        v.check(
            "Variance GPU (population)",
            gpu_val,
            cpu_pop_var,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // Correlation
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let gpu_corr = stats_gpu::correlation_gpu(gpu, &x, &y).expect("GPU correlation");
        let cpu_corr = 0.774_596_669_241_483_4;
        v.check(
            "Pearson correlation",
            gpu_corr,
            cpu_corr,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // Weighted dot product
    {
        let w = vec![1.0; 5];
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let gpu_dot = stats_gpu::weighted_dot_gpu(gpu, &w, &a, &b).expect("GPU weighted dot");
        let cpu_dot = 35.0;
        v.check(
            "Weighted dot product",
            gpu_dot,
            cpu_dot,
            tolerances::GPU_VS_CPU_F64,
        );
    }
}

fn validate_gpu_determinism(gpu: &GpuF64, v: &mut Validator) {
    v.section("── GPU Determinism ──");

    let counts: Vec<f64> = (1..=50).map(f64::from).collect();

    let s1 = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU Shannon run 1");
    let s2 = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU Shannon run 2");
    let s3 = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU Shannon run 3");
    v.check(
        "GPU determinism: 3 runs identical",
        f64::from(u8::from((s1 - s2).abs() < 1e-15 && (s2 - s3).abs() < 1e-15)),
        1.0,
        0.0,
    );
}

fn _euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
