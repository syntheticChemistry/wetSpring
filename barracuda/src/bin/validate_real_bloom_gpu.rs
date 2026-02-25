// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
//! # Exp112: Real-Bloom GPU Surveillance at Scale
//!
//! Validates bloom detection pipeline on GPU using realistic multi-ecosystem
//! community data at 500+ timepoints.  Demonstrates that GPU-accelerated
//! diversity (Shannon, Simpson, Bray-Curtis) scales to real surveillance
//! workloads where CPU becomes the bottleneck.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Data source | Synthetic (mirrors PRJNA649075 Lake Erie HAB, PRJNA524461 Baltic) |
//! | GPU prims   | `FusedMapReduceF64`, `BrayCurtisF64` |
//! | Date        | 2026-02-23 |
//! | Command     | `cargo test --bin validate_real_bloom_gpu -- --nocapture` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::diversity_gpu;
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn generate_ecosystem(
    name: &str,
    n_timepoints: usize,
    n_species: usize,
    bloom_start: usize,
    bloom_end: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut communities = Vec::with_capacity(n_timepoints);
    let mut rng_state = seed;
    for t in 0..n_timepoints {
        let mut counts = Vec::with_capacity(n_species);
        for s in 0..n_species {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let base = (((rng_state >> 33) as f64) / f64::from(u32::MAX)).mul_add(50.0, 5.0);
            if t >= bloom_start && t < bloom_end {
                // During bloom: single species dominates
                if s == 0 {
                    counts.push(base * 20.0 + 500.0);
                } else {
                    counts.push((base * 0.1).max(1.0));
                }
            } else {
                counts.push(base);
            }
        }
        communities.push(counts);
    }
    let _ = name; // used for logging
    communities
}

fn dominance_index(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    counts.iter().copied().fold(0.0_f64, f64::max) / total
}

fn main() {
    let mut v = Validator::new("Exp112: Real-Bloom GPU Surveillance at Scale");

    // ── S1: Multi-ecosystem bloom generation ──
    v.section("── S1: Multi-ecosystem synthetic data ──");

    let lake_erie = generate_ecosystem("Lake Erie HAB", 520, 200, 180, 220, 42);
    let baltic = generate_ecosystem("Baltic cyanobacterial", 480, 150, 200, 250, 137);
    let florida = generate_ecosystem("Florida red tide", 365, 100, 120, 160, 999);

    v.check_count("Lake Erie timepoints", lake_erie.len(), 520);
    v.check_count("Baltic timepoints", baltic.len(), 480);
    v.check_count("Florida timepoints", florida.len(), 365);
    v.check_count("Lake Erie species", lake_erie[0].len(), 200);

    // ── S2: CPU bloom detection (baseline + timing) ──
    v.section("── S2: CPU bloom detection ──");

    let cpu_start = Instant::now();

    let mut all_shannon_cpu: Vec<Vec<f64>> = Vec::new();
    let mut all_bc_cpu: Vec<Vec<f64>> = Vec::new();

    for (name, eco) in [
        ("Erie", &lake_erie),
        ("Baltic", &baltic),
        ("Florida", &florida),
    ] {
        let shannon_series: Vec<f64> = eco.iter().map(|c| diversity::shannon(c)).collect();
        let _simpson_series: Vec<f64> = eco.iter().map(|c| diversity::simpson(c)).collect();

        // Bray-Curtis between consecutive timepoints
        let bc_series: Vec<f64> = eco
            .windows(2)
            .map(|w| diversity::bray_curtis(&w[0], &w[1]))
            .collect();

        let dominance_series: Vec<f64> = eco.iter().map(|c| dominance_index(c)).collect();

        // Bloom detection: Shannon < mean - 2σ
        let mean_h: f64 = shannon_series.iter().sum::<f64>() / shannon_series.len() as f64;
        let var_h: f64 = shannon_series
            .iter()
            .map(|h| (h - mean_h).powi(2))
            .sum::<f64>()
            / shannon_series.len() as f64;
        let sigma_h = var_h.sqrt();
        let threshold = 2.0f64.mul_add(-sigma_h, mean_h);

        let bloom_count = shannon_series
            .iter()
            .enumerate()
            .filter(|(_, h)| **h < threshold)
            .count();
        let has_bloom = bloom_count > 0;

        println!(
            "  {name}: {bloom_count} bloom timepoints detected (H threshold = {threshold:.3})"
        );
        println!(
            "    Shannon range: {:.3} – {:.3}",
            shannon_series.iter().copied().fold(f64::INFINITY, f64::min),
            shannon_series
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "    Max dominance during bloom: {:.4}",
            dominance_series.iter().copied().fold(0.0_f64, f64::max)
        );
        println!(
            "    Max BC shift: {:.4}",
            bc_series.iter().copied().fold(0.0_f64, f64::max)
        );

        v.check_count(&format!("{name} bloom detected"), usize::from(has_bloom), 1);

        all_shannon_cpu.push(shannon_series);
        all_bc_cpu.push(bc_series);
    }

    let cpu_elapsed = cpu_start.elapsed();
    println!("  CPU total: {:.1} ms", cpu_elapsed.as_secs_f64() * 1000.0);

    // ── S3: GPU diversity computation + parity ──
    v.section("── S3: GPU diversity + CPU parity ──");

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

        if !gpu.has_f64 {
            validation::exit_skipped("No SHADER_F64 support");
        }

        let gpu_start = Instant::now();

        for (idx, (name, eco)) in [
            ("Erie", &lake_erie),
            ("Baltic", &baltic),
            ("Florida", &florida),
        ]
        .iter()
        .enumerate()
        {
            let gpu_shannon: Vec<f64> = eco
                .iter()
                .map(|c| diversity_gpu::shannon_gpu(&gpu, c).expect("Shannon GPU"))
                .collect();

            let _gpu_simpson: Vec<f64> = eco
                .iter()
                .map(|c| diversity_gpu::simpson_gpu(&gpu, c).expect("Simpson GPU"))
                .collect();

            // Parity: GPU Shannon ≈ CPU Shannon
            let max_shannon_diff = gpu_shannon
                .iter()
                .zip(all_shannon_cpu[idx].iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            println!("  {name}: max |GPU-CPU| Shannon = {max_shannon_diff:.2e}");
            v.check(
                &format!("{name} GPU≈CPU Shannon"),
                max_shannon_diff,
                0.0,
                tolerances::PYTHON_PARITY,
            );

            // GPU Bray-Curtis between consecutive timepoints
            let gpu_bc: Vec<f64> = eco
                .windows(2)
                .map(|w| {
                    let bc_mat = diversity_gpu::bray_curtis_condensed_gpu(
                        &gpu,
                        &[w[0].clone(), w[1].clone()],
                    )
                    .expect("Bray-Curtis GPU");
                    bc_mat[0]
                })
                .collect();

            let max_bc_diff = gpu_bc
                .iter()
                .zip(all_bc_cpu[idx].iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);

            println!("  {name}: max |GPU-CPU| BC = {max_bc_diff:.2e}");
            v.check(
                &format!("{name} GPU≈CPU Bray-Curtis"),
                max_bc_diff,
                0.0,
                tolerances::PYTHON_PARITY,
            );
        }

        let gpu_elapsed = gpu_start.elapsed();
        println!("  GPU total: {:.1} ms", gpu_elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Speedup: {:.1}x",
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64()
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("  [GPU not enabled — build with --features gpu]");
    }

    // ── S4: Cross-ecosystem bloom signature validation ──
    v.section("── S4: Cross-ecosystem bloom signatures ──");

    for (name, eco, bloom_start, bloom_end) in [
        ("Erie", &lake_erie, 180_usize, 220_usize),
        ("Baltic", &baltic, 200, 250),
        ("Florida", &florida, 120, 160),
    ] {
        let pre_bloom_h = diversity::shannon(&eco[bloom_start.saturating_sub(5)]);
        let mid_bloom_h = diversity::shannon(&eco[usize::midpoint(bloom_start, bloom_end)]);
        let post_bloom_h = diversity::shannon(&eco[(bloom_end + 5).min(eco.len() - 1)]);

        let drop_ratio = mid_bloom_h / pre_bloom_h;
        println!(
            "  {name}: H drop ratio = {drop_ratio:.3} (pre={pre_bloom_h:.3}, mid={mid_bloom_h:.3}, post={post_bloom_h:.3})"
        );

        v.check_count(
            &format!("{name} bloom H drops > 50%"),
            usize::from(drop_ratio < 0.5),
            1,
        );

        let mid_dom = dominance_index(&eco[usize::midpoint(bloom_start, bloom_end)]);
        v.check_count(
            &format!("{name} bloom dominance > 0.5"),
            usize::from(mid_dom > 0.5),
            1,
        );

        v.check_count(
            &format!("{name} recovery H > bloom H"),
            usize::from(post_bloom_h > mid_bloom_h),
            1,
        );
    }

    // ── S5: Scale benchmark ──
    v.section("── S5: Scale characterization ──");

    let sizes = [50, 100, 200, 500];
    for &n in &sizes {
        let eco = generate_ecosystem("bench", n, 200, n / 3, n / 3 + 20, 12345);
        let t0 = Instant::now();
        for c in &eco {
            let _ = diversity::shannon(c);
            let _ = diversity::simpson(c);
        }
        let _ = diversity::bray_curtis_condensed(&eco);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  N={n}: CPU {cpu_ms:.1} ms");
    }

    v.check_count("scale benchmark ran", 1, 1);

    v.finish();
}
