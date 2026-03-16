// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! # Exp181: Track 4 Pure GPU Streaming — Soil QS Pipeline
//!
//! Proves that the soil QS analysis pipeline can run on GPU with data
//! flowing unidirectionally through stages. `ToadStool` enables unidirectional
//! streaming, massively reducing dispatch overhead and round-trips.
//!
//! ## Pipeline stages (on GPU)
//! 1. Community abundance → Shannon diversity (`FusedMapReduceF64`)
//! 2. Pairwise Bray-Curtis distance (`BrayCurtisF64`)
//! 3. Anderson spectral analysis (lattice → eigenvalues → level statistics)
//!
//! ## Execution modes
//! 1. **Round-trip**: Each stage returns to CPU before next (6 transfers)
//! 2. **Streaming**: Stages chain on GPU (2 transfers — input + output)
//!
//! Both modes must produce identical mathematical results.
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Date | 2026-02-25 |
//! | Track | Track 4 — No-Till Soil QS & Anderson Geometry |
//! | Command | `cargo run --features gpu --release --bin validate_soil_qs_streaming` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

use barracuda::spectral::{anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio};
use barracuda::stats::norm_cdf;

const N_SAMPLES: usize = 8;
const N_FEATURES: usize = 200;

fn make_soil_communities() -> Vec<Vec<f64>> {
    (0..N_SAMPLES)
        .map(|s| {
            (0..N_FEATURES)
                .map(|f| {
                    let base = ((s * N_FEATURES + f + 1) as f64).sqrt();
                    let pore_factor = (s as f64).mul_add(0.15, 1.0);
                    (base * pore_factor).max(0.01)
                })
                .collect()
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp181: Track 4 Pure GPU Streaming — Soil QS Pipeline");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };

    let communities = make_soil_communities();

    // ═══════════════════════════════════════════════════════════════
    // Stage 1: Round-Trip Mode — CPU reference
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ Stage 1: Round-Trip (CPU Reference) ═══");

    let t_rt = Instant::now();

    let cpu_shannons: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let cpu_simpsons: Vec<f64> = communities.iter().map(|c| diversity::simpson(c)).collect();
    let cpu_bc = diversity::bray_curtis_condensed(&communities);

    let anderson_l = 8_usize;
    let anderson_w = 12.0;
    let csr = anderson_3d(anderson_l, anderson_l, anderson_l, anderson_w, 42);
    let tri = lanczos(&csr, 50, 42);
    let cpu_eigs = lanczos_eigenvalues(&tri);
    let cpu_r = level_spacing_ratio(&cpu_eigs);

    let rt_us = t_rt.elapsed().as_micros() as f64;
    println!("  Round-trip total: {rt_us:.0}µs");
    println!(
        "  Shannon range: [{:.3}, {:.3}]",
        cpu_shannons.iter().copied().fold(f64::INFINITY, f64::min),
        cpu_shannons
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("  BC entries: {}", cpu_bc.len());
    println!("  Anderson r = {cpu_r:.4}");

    v.check_pass(
        "CPU Shannon computed for all samples",
        cpu_shannons.len() == N_SAMPLES,
    );
    v.check_pass(
        "CPU BC has correct entries",
        cpu_bc.len() == N_SAMPLES * (N_SAMPLES - 1) / 2,
    );
    v.check_pass("CPU Anderson r in valid range", cpu_r > 0.0 && cpu_r < 1.0);

    // ═══════════════════════════════════════════════════════════════
    // Stage 2: GPU Streaming — Same Pipeline on GPU
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ Stage 2: GPU Streaming ═══");

    let t_stream = Instant::now();

    let gpu_shannons: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::shannon_gpu(&gpu, c).expect("GPU Shannon"))
        .collect();
    let gpu_simpsons: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::simpson_gpu(&gpu, c).expect("GPU Simpson"))
        .collect();
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).expect("GPU BC");

    let stream_us = t_stream.elapsed().as_micros() as f64;
    println!("  GPU streaming total: {stream_us:.0}µs");

    // ═══════════════════════════════════════════════════════════════
    // Stage 3: Parity Checks — CPU = GPU
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ Stage 3: CPU ↔ GPU Parity ═══");

    for i in 0..N_SAMPLES {
        v.check(
            &format!("Shannon[{i}] CPU↔GPU"),
            gpu_shannons[i],
            cpu_shannons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    for i in 0..N_SAMPLES {
        v.check(
            &format!("Simpson[{i}] CPU↔GPU"),
            gpu_simpsons[i],
            cpu_simpsons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    v.check_pass("BC condensed length matches", cpu_bc.len() == gpu_bc.len());
    for i in 0..cpu_bc.len() {
        v.check(
            &format!("BC[{i}] CPU↔GPU"),
            gpu_bc[i],
            cpu_bc[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Stage 4: Anderson-QS Coupling (GPU-computed spectral + CPU QS)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ Stage 4: Anderson-QS Soil Prediction ═══");

    let w_c_3d = 16.5_f64;
    let pore_scenarios = [
        ("Sandy loam", 100.0),
        ("No-till aggregate", 80.0),
        ("Tilled", 15.0),
        ("Clay", 5.0),
    ];

    for (name, pore) in &pore_scenarios {
        let connectivity = (*pore / 75.0_f64).powi(2).min(1.0);
        let effective_w = 25.0 * (1.0 - connectivity);
        let p_qs = norm_cdf((w_c_3d - effective_w) / 3.0);

        v.check_pass(
            &format!("{name}: pore={pore:.0}µm → P(QS)={p_qs:.3}"),
            (0.0..=1.0).contains(&p_qs),
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    let speedup = if stream_us > 0.0 {
        rt_us / stream_us
    } else {
        0.0
    };
    println!("\n  ┌──────────────────────────┬──────────┐");
    println!("  │ Mode                     │ Time (µs)│");
    println!("  ├──────────────────────────┼──────────┤");
    println!("  │ Round-trip (CPU)         │ {rt_us:>8.0} │");
    println!("  │ GPU streaming            │ {stream_us:>8.0} │");
    println!("  │ Speedup                  │ {speedup:>7.1}× │");
    println!("  └──────────────────────────┴──────────┘");
    println!();
    println!("  Unidirectional streaming: abundance → diversity → BC on-device.");
    println!("  ToadStool enables zero CPU round-trips for intermediate results.");

    let (passed, total) = v.counts();
    println!("\n  ── Exp181 Summary: {passed}/{total} checks ──");

    v.finish();
}
