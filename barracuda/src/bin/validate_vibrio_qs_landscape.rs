// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::field_reassign_with_default,
    reason = "validation harness: incremental struct field override for clarity"
)]
//! # Exp108: Vibrio QS Parameter Landscape via GPU ODE Sweep
//!
//! Sweeps 1024 QS parameter combinations through `OdeSweepGpu`, mapping the
//! bistability landscape across synthetic Vibrio-like parameter space.
//! Demonstrates GPU ODE sweep at population-genomics scale.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `5e6a00b` |
//! | Baseline type | Synthetic parameter sweep (GPU ODE) |
//! | Data source | Synthetic (mirrors ~12,000 Vibrio genomes on NCBI) |
//! | GPU prims | `BatchedOdeRK4F64` (via `OdeSweepGpu`) |
//! | Date | 2026-03-14 |
//! | Command | `cargo run --features gpu --bin validate_vibrio_qs_landscape` |
//! | Validation class | Synthetic — analytical known-values (GPU vs CPU parity) |

use std::time::Instant;
#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::ode_sweep_gpu::{N_PARAMS, N_VARS, OdeSweepConfig, OdeSweepGpu};
use wetspring_barracuda::bio::qs_biofilm::{self, QsBiofilmParams};
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::{self, Validator};

const N_BATCHES: usize = 1024;
const N_STEPS: u32 = 500;
const DT: f64 = tolerances::ODE_DT_SWEEP;

const fn params_to_flat_17(p: &QsBiofilmParams) -> [f64; N_PARAMS] {
    [
        p.mu_max,
        p.k_cap,
        p.death_rate,
        p.k_ai_prod,
        p.d_ai,
        p.k_hapr_max,
        p.k_hapr_ai,
        p.n_hapr,
        p.d_hapr,
        p.k_dgc_basal,
        p.k_dgc_rep,
        p.k_pde_basal,
        p.k_pde_act,
        p.k_bio_max,
        p.k_bio_cdg,
        p.n_bio,
        p.d_bio,
    ]
}

fn generate_parameter_landscape() -> Vec<QsBiofilmParams> {
    let mut params_vec = Vec::with_capacity(N_BATCHES);
    let _p = QsBiofilmParams::default();

    // Sweep mu_max (growth rate) and k_ai_prod (AI production) across Vibrio-like range
    let mu_steps = 32;
    let kai_steps = 32;

    for i in 0..mu_steps {
        for j in 0..kai_steps {
            let mut params = QsBiofilmParams::default();
            // mu_max: 0.2 – 1.2 (slow to fast growers)
            params.mu_max = 0.2 + f64::from(i) * 1.0 / (f64::from(mu_steps) - 1.0);
            // k_ai_prod: 1.0 – 10.0 (low to high AI production)
            params.k_ai_prod = 1.0 + f64::from(j) * 9.0 / (f64::from(kai_steps) - 1.0);
            // Vary hapR AI threshold for bistability detection
            params.k_hapr_ai = f64::from(i).mul_add(0.01, 0.3);
            params_vec.push(params);
        }
    }

    params_vec
}

fn classify_outcome(y_final: &[f64]) -> &'static str {
    let biofilm = y_final[4];
    let cells = y_final[0];
    if biofilm > 0.5 && cells > 0.1 {
        "biofilm"
    } else if biofilm < 0.1 && cells > 0.1 {
        "planktonic"
    } else if cells < 0.01 {
        "extinction"
    } else {
        "intermediate"
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single function"
)]
fn main() {
    let mut v = Validator::new("Exp108: Vibrio QS Parameter Landscape");

    // ── S1: Parameter space generation ──
    v.section("── S1: Parameter landscape ──");

    let params = generate_parameter_landscape();
    v.check_count("parameter sets", params.len(), N_BATCHES);

    let y0 = [0.01, 0.0, 0.0, 1.0, 0.0]; // N, A, H, C, B

    // ── S2: CPU baseline (sample subset + timing) ──
    v.section("── S2: CPU ODE integration ──");

    let cpu_start = Instant::now();
    let cpu_subset_size = 64;
    let mut cpu_results: Vec<[f64; 5]> = Vec::with_capacity(cpu_subset_size);

    for p in params.iter().take(cpu_subset_size) {
        let result = qs_biofilm::run_scenario(&y0, 5.0, DT, p);
        cpu_results.push([
            result.y_final[0],
            result.y_final[1],
            result.y_final[2],
            result.y_final[3],
            result.y_final[4],
        ]);
    }
    let cpu_elapsed = cpu_start.elapsed();
    println!(
        "  CPU ({cpu_subset_size} batches): {:.1} ms",
        cpu_elapsed.as_secs_f64() * 1000.0
    );

    let cpu_all_finite = cpu_results.iter().all(|r| r.iter().all(|x| x.is_finite()));
    v.check_count("CPU results finite", usize::from(cpu_all_finite), 1);

    // Classify CPU outcomes
    let mut cpu_classes = std::collections::HashMap::new();
    for r in &cpu_results {
        *cpu_classes.entry(classify_outcome(r)).or_insert(0_usize) += 1;
    }
    for (class, count) in &cpu_classes {
        println!("    {class}: {count}");
    }

    // ── S3: GPU full sweep ──
    v.section("── S3: GPU ODE sweep (1024 batches) ──");

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");

        if !gpu.has_f64 {
            validation::exit_skipped("No SHADER_F64 support");
        }

        let device = gpu.to_wgpu_device();
        let sweeper = OdeSweepGpu::new(device);

        let config = OdeSweepConfig {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "validation: bounded float→integer for index/count"
            )]
            n_batches: N_BATCHES as u32,
            n_steps: N_STEPS,
            h: DT,
            t0: 0.0,
            clamp_max: 100.0,
            clamp_min: 0.0,
        };

        let all_y0: Vec<f64> = (0..N_BATCHES).flat_map(|_| y0.iter().copied()).collect();
        let all_params: Vec<f64> = params.iter().flat_map(params_to_flat_17).collect();

        let gpu_start = Instant::now();
        let gpu_output = sweeper
            .integrate(&config, &all_y0, &all_params)
            .or_exit("ODE integrate");
        let gpu_elapsed = gpu_start.elapsed();

        println!(
            "  GPU ({N_BATCHES} batches): {:.1} ms",
            gpu_elapsed.as_secs_f64() * 1000.0
        );
        println!("  GPU output length: {}", gpu_output.len());

        v.check_count("GPU output size", gpu_output.len(), N_BATCHES * N_VARS);

        let gpu_all_finite = gpu_output.iter().all(|x| x.is_finite());
        v.check_count("GPU results finite", usize::from(gpu_all_finite), 1);

        // GPU vs CPU parity (first 64 batches)
        let mut max_diff = 0.0_f64;
        for (i, cpu_r) in cpu_results.iter().enumerate() {
            for var in 0..N_VARS {
                let gpu_val = gpu_output[i * N_VARS + var];
                let diff = (gpu_val - cpu_r[var]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        println!("  max |GPU-CPU| (64 batches): {max_diff:.4}");
        // Long-horizon ODE drift (500 steps × dt=0.01) accumulates; absolute
        // tolerance matches documented GPU ODE parity from Exp049.
        v.check_count(
            "GPU≈CPU parity < 2.0",
            usize::from(max_diff < tolerances::ODE_GPU_LANDSCAPE_PARITY),
            1,
        );

        // Classify GPU landscape
        let mut gpu_classes = std::collections::HashMap::new();
        for i in 0..N_BATCHES {
            let y = &gpu_output[i * N_VARS..(i + 1) * N_VARS];
            *gpu_classes.entry(classify_outcome(y)).or_insert(0_usize) += 1;
        }

        println!("  Landscape classification ({N_BATCHES} genomes):");
        for (class, count) in &gpu_classes {
            #[expect(
                clippy::cast_precision_loss,
                reason = "precision: bounded integer→f64 for validation metrics"
            )]
            let pct = (*count as f64) / (N_BATCHES as f64) * 100.0;
            println!("    {class}: {count} ({pct:.1}%)");
        }

        let has_biofilm = gpu_classes.get("biofilm").copied().unwrap_or(0) > 0;
        let has_multiple = gpu_classes.len() > 1;
        v.check_count(
            "landscape has biofilm outcomes",
            usize::from(has_biofilm),
            1,
        );
        v.check_count(
            "landscape has diverse outcomes",
            usize::from(has_multiple),
            1,
        );

        #[expect(
            clippy::cast_precision_loss,
            reason = "precision: bounded integer→f64 for validation metrics"
        )]
        let cpu_extrapolated_ms =
            cpu_elapsed.as_secs_f64() * 1000.0 * (N_BATCHES as f64 / cpu_subset_size as f64);
        println!("  Estimated CPU for {N_BATCHES}: {cpu_extrapolated_ms:.0} ms");
        println!(
            "  GPU speedup: {:.1}x (actual vs extrapolated)",
            cpu_extrapolated_ms / (gpu_elapsed.as_secs_f64() * 1000.0)
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("  [GPU not enabled — build with --features gpu]");
    }

    // ── S4: Bistability detection ──
    v.section("── S4: Bistability scan ──");

    let mut bistable_count = 0;
    let bistable_test_n = 32;
    for i in 0..bistable_test_n {
        let p = params[i * 32].clone();

        // Forward sweep: low initial biofilm
        let y0_low = [0.01, 0.0, 0.0, 1.0, 0.0];
        let r_low = qs_biofilm::run_scenario(&y0_low, 10.0, DT, &p);

        // Backward sweep: high initial biofilm
        let y0_high = [0.5, 0.5, 0.5, 0.5, 0.8];
        let r_high = qs_biofilm::run_scenario(&y0_high, 10.0, DT, &p);

        let b_low = r_low.y_final[4];
        let b_high = r_high.y_final[4];

        if (b_high - b_low).abs() > 0.3 {
            bistable_count += 1;
        }
    }

    println!("  Bistable parameter sets: {bistable_count}/{bistable_test_n}");
    v.check_count("bistability detected", usize::from(bistable_count > 0), 1);

    v.finish();
}
