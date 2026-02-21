// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU validation — compare all GPU results against CPU baselines.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | CPU diversity metrics (`bio::diversity`) |
//! | Baseline version | GPU parity validation |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --features gpu --bin validate_diversity_gpu` |
//! | Data | Synthetic (Shannon, Simpson, Bray-Curtis, `PCoA`, etc.) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Follows the hotSpring pattern: GPU result vs CPU result → harness check
//! → exit 0 (pass) / 1 (fail) / 2 (skip).
//!
//! Tests Tier A GPU promotion readiness:
//! - Shannon entropy (via `ToadStool`'s `FusedMapReduceF64`)
//! - Simpson diversity (via `ToadStool`'s `FusedMapReduceF64`)
//! - Bray-Curtis condensed distance matrix (via `ToadStool`'s `BrayCurtisF64`)
//! - `PCoA` ordination (via `ToadStool`'s `BatchedEighGpu`)
//! - Observed features, Pielou evenness, alpha diversity bundle
//! - Pairwise cosine similarity (via `ToadStool`'s `GemmF64` + `FusedMapReduceF64`)
//! - Variance/std dev (via `ToadStool`'s `VarianceF64`)
//! - Pearson correlation (via `ToadStool`'s `CorrelationF64`)
//! - Sample covariance (via `ToadStool`'s `CovarianceF64`)
//! - Weighted/plain dot product (via `ToadStool`'s `WeightedDotF64`)
//!
//! Run: `cargo run --features gpu --bin validate_diversity_gpu`

use wetspring_barracuda::bio::{
    diversity, diversity_gpu, pcoa, pcoa_gpu, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("wetSpring GPU Diversity Validation");

    // ── GPU init
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

    validate_shannon(&gpu, &mut v);
    validate_simpson(&gpu, &mut v);
    validate_bray_curtis(&gpu, &mut v);
    validate_pcoa(&gpu, &mut v);
    validate_alpha_diversity(&gpu, &mut v);
    validate_spectral_match(&gpu, &mut v);
    validate_stats_gpu(&mut v, &gpu);

    v.finish();
}

fn validate_shannon(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Shannon GPU vs CPU ──");

    // Uniform distribution (analytical: ln(4))
    {
        let counts = vec![25.0, 25.0, 25.0, 25.0];
        let cpu = diversity::shannon(&counts);
        let gpu_val = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU shannon uniform");
        v.check(
            "Shannon uniform (4 species)",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // Single species (should be 0)
    {
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        let cpu = diversity::shannon(&counts);
        let gpu_val = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU shannon single");
        v.check(
            "Shannon single species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // 100 uniform species
    {
        let counts = vec![10.0; 100];
        let cpu = diversity::shannon(&counts);
        let gpu_val = diversity_gpu::shannon_gpu(gpu, &counts).expect("GPU shannon 100");
        v.check(
            "Shannon 100 uniform species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
}

fn validate_simpson(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Simpson GPU vs CPU ──");

    // Uniform distribution (analytical: 0.75)
    {
        let counts = vec![25.0, 25.0, 25.0, 25.0];
        let cpu = diversity::simpson(&counts);
        let gpu_val = diversity_gpu::simpson_gpu(gpu, &counts).expect("GPU simpson uniform");
        v.check(
            "Simpson uniform (4 species)",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // Single species (should be 0)
    {
        let counts = vec![100.0, 0.0, 0.0];
        let cpu = diversity::simpson(&counts);
        let gpu_val = diversity_gpu::simpson_gpu(gpu, &counts).expect("GPU simpson single");
        v.check(
            "Simpson single species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // 10 uniform species (analytical: 0.9)
    {
        let counts = vec![50.0; 10];
        let cpu = diversity::simpson(&counts);
        let gpu_val = diversity_gpu::simpson_gpu(gpu, &counts).expect("GPU simpson 10");
        v.check(
            "Simpson 10 uniform species",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_F64,
        );
    }
}

fn validate_bray_curtis(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Bray-Curtis GPU vs CPU ──");

    // 3 samples (verify all pairs)
    {
        let samples = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 10.0, 25.0],
            vec![0.0, 50.0, 0.0],
        ];
        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val =
            diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC 3 samples");

        assert_eq!(gpu_val.len(), cpu.len(), "BC pair count mismatch");
        for (i, (&g, &c)) in gpu_val.iter().zip(cpu.iter()).enumerate() {
            v.check(
                &format!("Bray-Curtis pair {i}"),
                g,
                c,
                tolerances::GPU_VS_CPU_BRAY_CURTIS,
            );
        }
    }

    // Identical samples (should be 0)
    {
        let samples = vec![vec![10.0, 20.0, 30.0], vec![10.0, 20.0, 30.0]];
        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val =
            diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC identical");
        v.check(
            "Bray-Curtis identical samples",
            gpu_val[0],
            cpu[0],
            tolerances::GPU_VS_CPU_BRAY_CURTIS,
        );
    }

    // Completely different (should be 1)
    {
        let samples = vec![vec![10.0, 0.0, 0.0], vec![0.0, 0.0, 10.0]];
        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val =
            diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC disjoint");
        v.check(
            "Bray-Curtis disjoint samples",
            gpu_val[0],
            cpu[0],
            tolerances::GPU_VS_CPU_BRAY_CURTIS,
        );
    }

    // Larger matrix (10 samples × 50 features)
    {
        let n = 10;
        let d = 50;
        #[allow(clippy::cast_precision_loss)] // deterministic test data generation
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 13 + 1) % 100) as f64)
                    .collect()
            })
            .collect();

        let cpu = diversity::bray_curtis_condensed(&samples);
        let gpu_val =
            diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).expect("GPU BC 10×50");

        let n_pairs = n * (n - 1) / 2;
        assert_eq!(gpu_val.len(), n_pairs);

        let mut bc_passed = true;
        let mut max_diff = 0.0_f64;
        for (g, c) in gpu_val.iter().zip(cpu.iter()) {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            if diff > tolerances::GPU_VS_CPU_BRAY_CURTIS {
                bc_passed = false;
            }
        }

        v.check(
            &format!("Bray-Curtis 10×50 (max diff {max_diff:.2e}, {n_pairs} pairs)"),
            f64::from(u8::from(bc_passed)),
            1.0,
            0.0,
        );
    }
}

fn validate_pcoa(gpu: &GpuF64, v: &mut Validator) {
    v.section("── PCoA GPU vs CPU ──");

    // 4 samples with Bray-Curtis distances
    {
        let samples = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![15.0, 10.0, 25.0, 50.0],
            vec![0.0, 50.0, 0.0, 10.0],
            vec![5.0, 5.0, 45.0, 5.0],
        ];
        let condensed = diversity::bray_curtis_condensed(&samples);
        let n = samples.len();

        let cpu_result = pcoa::pcoa(&condensed, n, 2).expect("CPU PCoA");
        let gpu_result = pcoa_gpu::pcoa_gpu(gpu, &condensed, n, 2).expect("GPU PCoA");

        // Eigenvalues should match closely
        for (axis, (&cpu_ev, &gpu_ev)) in cpu_result
            .eigenvalues
            .iter()
            .zip(gpu_result.eigenvalues.iter())
            .enumerate()
        {
            v.check(
                &format!("PCoA eigenvalue axis {axis}"),
                gpu_ev,
                cpu_ev,
                tolerances::GPU_VS_CPU_F64,
            );
        }

        // Reconstructed distances should match (sign-invariant check)
        let mut dist_passed = true;
        let mut max_dist_diff = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let cpu_dist =
                    euclidean_dist(&cpu_result.coordinates[i], &cpu_result.coordinates[j]);
                let gpu_dist =
                    euclidean_dist(&gpu_result.coordinates[i], &gpu_result.coordinates[j]);
                let diff = (cpu_dist - gpu_dist).abs();
                max_dist_diff = max_dist_diff.max(diff);
                if diff > tolerances::GPU_VS_CPU_F64 {
                    dist_passed = false;
                }
            }
        }

        v.check(
            &format!("PCoA reconstructed distances (max diff {max_dist_diff:.2e})"),
            f64::from(u8::from(dist_passed)),
            1.0,
            0.0,
        );

        // Proportion explained should match CPU
        for (axis, (&cpu_p, &gpu_p)) in cpu_result
            .proportion_explained
            .iter()
            .zip(gpu_result.proportion_explained.iter())
            .enumerate()
        {
            v.check(
                &format!("PCoA proportion axis {axis}"),
                gpu_p,
                cpu_p,
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }
}

fn validate_alpha_diversity(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Alpha Diversity GPU vs CPU ──");

    // Observed features
    {
        let counts = vec![10.0, 0.0, 20.0, 0.0, 5.0];
        let cpu = diversity::observed_features(&counts);
        let gpu_val =
            diversity_gpu::observed_features_gpu(gpu, &counts).expect("GPU observed features");
        v.check(
            "Observed features",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // Pielou evenness (uniform)
    {
        let counts = vec![25.0; 4];
        let cpu = diversity::pielou_evenness(&counts);
        let gpu_val = diversity_gpu::pielou_evenness_gpu(gpu, &counts).expect("GPU Pielou uniform");
        v.check(
            "Pielou evenness uniform",
            gpu_val,
            cpu,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // Full alpha diversity bundle
    {
        let counts = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
        let cpu = diversity::alpha_diversity(&counts);
        let gpu_val =
            diversity_gpu::alpha_diversity_gpu(gpu, &counts).expect("GPU alpha diversity");
        v.check(
            "Alpha: observed",
            gpu_val.observed,
            cpu.observed,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            "Alpha: Shannon",
            gpu_val.shannon,
            cpu.shannon,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            "Alpha: Simpson",
            gpu_val.simpson,
            cpu.simpson,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            "Alpha: evenness",
            gpu_val.evenness,
            cpu.evenness,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
}

fn validate_spectral_match(gpu: &GpuF64, v: &mut Validator) {
    v.section("── Spectral Match GPU vs CPU ──");

    // 4 spectra of 5 bins each — pairwise cosine
    {
        let spectra = vec![
            vec![100.0, 50.0, 25.0, 10.0, 5.0],
            vec![100.0, 50.0, 25.0, 10.0, 5.0], // identical to first
            vec![5.0, 10.0, 25.0, 50.0, 100.0], // reversed
            vec![0.0, 0.0, 100.0, 0.0, 0.0],    // single peak
        ];

        // CPU baseline: direct vector cosine = dot(a,b) / (||a|| * ||b||)
        let n = spectra.len();
        let mut cpu_condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 1..n {
            for j in 0..i {
                let dot: f64 = spectra[i]
                    .iter()
                    .zip(spectra[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let norm_i: f64 = spectra[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_j: f64 = spectra[j].iter().map(|x| x * x).sum::<f64>().sqrt();
                let denom = norm_i * norm_j;
                let score = if denom > 0.0 { dot / denom } else { 0.0 };
                cpu_condensed.push(score);
            }
        }

        let gpu_val =
            spectral_match_gpu::pairwise_cosine_gpu(gpu, &spectra).expect("GPU pairwise cosine");

        assert_eq!(gpu_val.len(), cpu_condensed.len());

        for (i, (&g, &c)) in gpu_val.iter().zip(cpu_condensed.iter()).enumerate() {
            v.check(
                &format!("Cosine pair {i}"),
                g,
                c,
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }

    // Query vs library
    {
        let query = vec![100.0, 50.0, 25.0, 10.0, 5.0];
        let refs = vec![
            vec![100.0, 50.0, 25.0, 10.0, 5.0],
            vec![5.0, 10.0, 25.0, 50.0, 100.0],
        ];

        let gpu_scores =
            spectral_match_gpu::cosine_vs_library_gpu(gpu, &query, &refs).expect("GPU vs library");

        v.check(
            "Library match: identical",
            gpu_scores[0],
            1.0,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            "Library match: reversed < 1",
            f64::from(u8::from(gpu_scores[1] < 1.0)),
            1.0,
            0.0,
        );
    }
}

fn validate_stats_gpu(v: &mut Validator, gpu: &GpuF64) {
    v.section("Statistics GPU vs CPU");

    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let cpu_var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let cpu_sample_var = cpu_var * n / (n - 1.0);
    let cpu_std = cpu_sample_var.sqrt();

    let gpu_var = stats_gpu::variance_gpu(gpu, &data).expect("variance GPU");
    let gpu_sample_var = stats_gpu::sample_variance_gpu(gpu, &data).expect("sample variance GPU");
    let gpu_std = stats_gpu::std_dev_gpu(gpu, &data).expect("std dev GPU");

    v.check(
        "Population variance",
        gpu_var,
        cpu_var,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Sample variance",
        gpu_sample_var,
        cpu_sample_var,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Sample std dev",
        gpu_std,
        cpu_std,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];

    let x_mean = x.iter().sum::<f64>() / x.len() as f64;
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let cpu_cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>()
        / (x.len() as f64 - 1.0);
    let x_std =
        (x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt();
    let y_std =
        (y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (y.len() as f64 - 1.0)).sqrt();
    let cpu_corr = cpu_cov / (x_std * y_std);

    let gpu_cov = stats_gpu::covariance_gpu(gpu, &x, &y).expect("covariance GPU");
    let gpu_corr = stats_gpu::correlation_gpu(gpu, &x, &y).expect("correlation GPU");

    v.check(
        "Sample covariance",
        gpu_cov,
        cpu_cov,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Pearson correlation",
        gpu_corr,
        cpu_corr,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let a = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let b = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let cpu_wdot: f64 = w
        .iter()
        .zip(a.iter().zip(b.iter()))
        .map(|(&wi, (&ai, &bi))| wi * ai * bi)
        .sum();
    let gpu_wdot = stats_gpu::weighted_dot_gpu(gpu, &w, &a, &b).expect("weighted dot GPU");
    v.check(
        "Weighted dot product",
        gpu_wdot,
        cpu_wdot,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    let cpu_dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let gpu_dot = stats_gpu::dot_gpu(gpu, &x, &y).expect("dot GPU");
    v.check(
        "Dot product",
        gpu_dot,
        cpu_dot,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
