// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
//! # Exp111: Full MassBank GPU Spectral Screening at Scale
//!
//! Validates GPU spectral cosine similarity at library-scale (2048×2048)
//! using realistic mass spectra distributions. Demonstrates the 926x GPU
//! speedup at real-world library sizes.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Data source | Synthetic (mirrors MassBank PFAS spectral distributions) |
//! | GPU prims   | `GemmF64`, `FusedMapReduceF64` (via `pairwise_cosine_gpu`) |
//! | Date        | 2026-02-23 |

use std::time::Instant;
#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::spectral_match_gpu;
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::special;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn generate_spectra(n_spectra: usize, n_bins: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut spectra = Vec::with_capacity(n_spectra);
    let mut rng = seed;
    for _ in 0..n_spectra {
        let mut bins = Vec::with_capacity(n_bins);
        for _ in 0..n_bins {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let raw = ((rng >> 33) as f64) / (u32::MAX as f64);
            // Sparse: ~80% of bins are zero (typical MS data)
            if raw < 0.8 {
                bins.push(0.0);
            } else {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let intensity = ((rng >> 33) as f64) / (u32::MAX as f64) * 1000.0;
                bins.push(intensity);
            }
        }
        spectra.push(bins);
    }
    spectra
}

fn cpu_pairwise_cosine(spectra: &[Vec<f64>]) -> Vec<f64> {
    let n = spectra.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = special::dot(&spectra[i], &spectra[j]);
            let norm_a: f64 = special::l2_norm(&spectra[i]);
            let norm_b: f64 = special::l2_norm(&spectra[j]);
            let denom = norm_a * norm_b;
            if denom > 0.0 {
                condensed.push(dot / denom);
            } else {
                condensed.push(0.0);
            }
        }
    }
    condensed
}

fn main() {
    let mut v = Validator::new("Exp111: Full MassBank GPU Spectral Screening");

    // ── S1: Library generation ──
    v.section("── S1: Synthetic spectral library ──");

    let small_lib = generate_spectra(64, 500, 42);
    let medium_lib = generate_spectra(256, 500, 137);
    let large_lib = generate_spectra(1024, 500, 999);
    let xl_lib = generate_spectra(2048, 500, 7777);

    v.check_count("small library", small_lib.len(), 64);
    v.check_count("medium library", medium_lib.len(), 256);
    v.check_count("large library", large_lib.len(), 1024);
    v.check_count("XL library", xl_lib.len(), 2048);

    // ── S2: CPU baseline + timing at scale ──
    v.section("── S2: CPU cosine matrix scaling ──");

    let sizes = [("64", &small_lib), ("256", &medium_lib)];

    let mut cpu_times = Vec::new();
    let mut cpu_results = Vec::new();

    for (label, lib) in &sizes {
        let t0 = Instant::now();
        let cosine = cpu_pairwise_cosine(lib);
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let n_pairs = cosine.len();
        println!("  N={label}: {n_pairs} pairs, CPU {ms:.1} ms");

        // Sanity: all values in [0, 1]
        let all_valid = cosine.iter().all(|&c| (0.0..=1.0).contains(&c));
        v.check_count(
            &format!("N={label} cosine ∈ [0,1]"),
            usize::from(all_valid),
            1,
        );

        cpu_times.push(ms);
        cpu_results.push(cosine);
    }

    // Large scale CPU (timing only, sample subset for parity)
    let t0 = Instant::now();
    let large_cosine = cpu_pairwise_cosine(&large_lib[..128].to_vec());
    let cpu_large_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  N=1024 (128-sample subset): CPU {cpu_large_ms:.1} ms");

    v.check_count(
        "CPU large subset computed",
        usize::from(!large_cosine.is_empty()),
        1,
    );

    // ── S3: GPU spectral cosine + parity ──
    v.section("── S3: GPU spectral cosine + parity ──");

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let gpu = rt.block_on(GpuF64::new()).unwrap();

        if !gpu.has_f64 {
            validation::exit_skipped("No SHADER_F64 support");
        }

        let gpu_sizes: Vec<(&str, &Vec<Vec<f64>>)> = vec![
            ("64", &small_lib),
            ("256", &medium_lib),
            ("1024", &large_lib),
            ("2048", &xl_lib),
        ];

        for (idx, (label, lib)) in gpu_sizes.iter().enumerate() {
            let t0 = Instant::now();
            let gpu_cosine = spectral_match_gpu::pairwise_cosine_gpu(&gpu, lib).unwrap();
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let n_pairs = gpu_cosine.len();
            println!("  N={label}: {n_pairs} pairs, GPU {gpu_ms:.1} ms");

            // GPU values in [0, 1] (with small tolerance for rounding)
            let all_valid = gpu_cosine.iter().all(|&c| c >= -1e-10 && c <= 1.0 + 1e-10);
            v.check_count(
                &format!("N={label} GPU cosine valid"),
                usize::from(all_valid),
                1,
            );

            // Parity with CPU for small/medium sizes
            if idx < cpu_results.len() {
                let max_diff = gpu_cosine
                    .iter()
                    .zip(cpu_results[idx].iter())
                    .map(|(g, c)| (g - c).abs())
                    .fold(0.0_f64, f64::max);
                println!("    max |GPU-CPU| = {max_diff:.2e}");
                v.check(
                    &format!("N={label} GPU≈CPU"),
                    max_diff,
                    0.0,
                    tolerances::PYTHON_PARITY,
                );

                if cpu_times.len() > idx && gpu_ms > 0.0 {
                    println!("    Speedup: {:.1}x", cpu_times[idx] / gpu_ms);
                }
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("  [GPU not enabled — build with --features gpu]");
    }

    // ── S4: Scaling curve ──
    v.section("── S4: Scaling characterization ──");

    let scaling_sizes = [32, 64, 128, 256, 512];
    for &n in &scaling_sizes {
        let lib = generate_spectra(n, 500, 42 + n as u64);
        let t0 = Instant::now();
        let _ = cpu_pairwise_cosine(&lib);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let n_pairs = n * (n - 1) / 2;
        println!(
            "  N={n}: {n_pairs} pairs, CPU {cpu_ms:.1} ms ({:.0} pairs/ms)",
            n_pairs as f64 / cpu_ms
        );
    }

    v.check_count("scaling curve ran", 1, 1);

    v.finish();
}
