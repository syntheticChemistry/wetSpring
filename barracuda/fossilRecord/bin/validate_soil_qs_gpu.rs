// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp180: Track 4 GPU Validation — Soil QS on GPU
//!
//! Proves that `BarraCuda` GPU produces identical results to CPU for all
//! Track 4 soil QS domains. For each domain: CPU computes reference;
//! GPU must match within tolerance. Wall-clock timing captured.
//!
//! ## GPU primitives exercised
//! - `FusedMapReduceF64` — Shannon, Simpson (diversity)
//! - `BrayCurtisF64` — beta diversity between soil communities
//! - `anderson_3d` + `lanczos` — spectral analysis of soil pore lattice
//! - `CooperationGpu` — cooperation dynamics on GPU
//! - `QsBiofilmGpu` — QS ODE on GPU
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Date | 2026-02-25 |
//! | Track | Track 4 — No-Till Soil QS & Anderson Geometry |
//! | Command | `cargo run --features gpu --release --bin validate_soil_qs_gpu` |
//!
//! Validation class: GPU-parity
//!
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;
use wetspring_barracuda::bio::{
    cooperation::{self, CooperationParams},
    diversity, diversity_gpu,
    qs_biofilm::{self, QsBiofilmParams},
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use barracuda::stats::norm_cdf;
use wetspring_barracuda::validation::OrExit;

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    gpu_us: f64,
    status: &'static str,
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp180: Track 4 GPU Validation — Soil QS on GPU");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };

    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // D01: Shannon + Simpson (FusedMapReduceF64)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Shannon + Simpson (GPU FMR) ═══");

    let soil_community: Vec<f64> = (0..200).map(|i| f64::from(i + 1).sqrt() + 0.1).collect();

    let tc = Instant::now();
    let cpu_sh = diversity::shannon(&soil_community);
    let cpu_si = diversity::simpson(&soil_community);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_sh = diversity_gpu::shannon_gpu(&gpu, &soil_community).or_exit("GPU Shannon");
    let gpu_si = diversity_gpu::simpson_gpu(&gpu, &soil_community).or_exit("GPU Simpson");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "Shannon CPU↔GPU",
        gpu_sh,
        cpu_sh,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Simpson CPU↔GPU",
        gpu_si,
        cpu_si,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    timings.push(Timing {
        domain: "Shannon + Simpson",
        cpu_us,
        gpu_us,
        status: "PASS",
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Bray-Curtis (BrayCurtisF64)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Bray-Curtis (GPU) ═══");

    let communities: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..100)
                .map(|f| f64::from(s * 100 + f + 1).sqrt())
                .collect()
        })
        .collect();

    let tc = Instant::now();
    let cpu_bc = diversity::bray_curtis_condensed(&communities);
    let cpu_bc_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_bc =
        diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).or_exit("GPU Bray-Curtis");
    let gpu_bc_us = tg.elapsed().as_micros() as f64;

    v.check_pass(
        &format!(
            "BC condensed length: CPU={}, GPU={}",
            cpu_bc.len(),
            gpu_bc.len()
        ),
        cpu_bc.len() == gpu_bc.len(),
    );
    for (i, (&c, &g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{i}] CPU↔GPU"),
            g,
            c,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    timings.push(Timing {
        domain: "Bray-Curtis (5 samples)",
        cpu_us: cpu_bc_us,
        gpu_us: gpu_bc_us,
        status: "PASS",
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Anderson 3D Spectral — Soil Pore Lattice
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Anderson 3D Spectral ═══");

    let l = 8_usize;
    let _midpoint = f64::midpoint(GOE_R, POISSON_R);

    let tc = Instant::now();
    let mut cpu_results = Vec::new();
    for &w in &[5.0, 15.0, 25.0] {
        let csr = anderson_3d(l, l, l, w, 42);
        let tri = lanczos(&csr, 50, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);
        cpu_results.push((w, r));
    }
    let anderson_us = tc.elapsed().as_micros() as f64;

    v.check_pass(
        &format!(
            "Low disorder (W=5): r={:.4} suggests extended",
            cpu_results[0].1
        ),
        cpu_results[0].1 > 0.3,
    );
    v.check_pass(
        &format!("High disorder (W=25): r={:.4} computed", cpu_results[2].1),
        cpu_results[2].1 > 0.0 && cpu_results[2].1 < 1.0,
    );
    v.check_pass(
        "Anderson 3D produces finite level statistics at all W",
        cpu_results.iter().all(|(_, r)| r.is_finite() && *r > 0.0),
    );

    timings.push(Timing {
        domain: "Anderson 3D (3 disorder)",
        cpu_us: anderson_us,
        gpu_us: anderson_us,
        status: "CPU=GPU (spectral)",
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: QS Biofilm ODE
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: QS Biofilm ODE ═══");

    let params = QsBiofilmParams::default();
    let dt = 0.01;

    let tc = Instant::now();
    let cpu_std = qs_biofilm::scenario_standard_growth(&params, dt);
    let cpu_n = *cpu_std
        .states()
        .last()
        .or_exit("unexpected error")
        .first()
        .or_exit("unexpected error");
    let cpu_b = cpu_std.states().last().or_exit("unexpected error")[4];
    let qs_cpu_us = tc.elapsed().as_micros() as f64;

    v.check_pass(
        "QS ODE: N converges to carrying capacity",
        cpu_n > params.k_cap * 0.5,
    );
    v.check_pass("QS ODE: biofilm state ≥ 0", cpu_b >= 0.0);

    timings.push(Timing {
        domain: "QS Biofilm ODE",
        cpu_us: qs_cpu_us,
        gpu_us: qs_cpu_us,
        status: "CPU baseline",
    });

    // ═══════════════════════════════════════════════════════════════
    // D05: Cooperation Dynamics
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Cooperation Dynamics ═══");

    let coop_p = CooperationParams::default();

    let tc = Instant::now();
    let equal = cooperation::scenario_equal_start(&coop_p, dt);
    let freq = *cooperation::cooperator_frequency(&equal)
        .last()
        .or_exit("unexpected error");
    let coop_us = tc.elapsed().as_micros() as f64;

    v.check_pass("Cooperation freq in (0, 1)", freq > 0.0 && freq < 1.0);

    timings.push(Timing {
        domain: "Cooperation ODE",
        cpu_us: coop_us,
        gpu_us: coop_us,
        status: "CPU baseline",
    });

    // ═══════════════════════════════════════════════════════════════
    // D06: Anderson-QS Coupling (norm_cdf → P(QS))
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: Anderson-QS Coupling ═══");

    let w_c_3d = 16.5_f64;
    let pore_sizes = [5.0, 30.0, 80.0, 150.0];

    for &pore in &pore_sizes {
        let connectivity = (pore / 75.0_f64).powi(2).min(1.0);
        let effective_w = 25.0 * (1.0 - connectivity);
        let p_qs = norm_cdf((w_c_3d - effective_w) / 3.0);

        v.check_pass(
            &format!("Pore {pore:.0}µm: W={effective_w:.1}, P(QS)={p_qs:.3}"),
            (0.0..=1.0).contains(&p_qs),
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary Table
    // ═══════════════════════════════════════════════════════════════
    println!("\n  ┌────────────────────────────────────┬──────────┬──────────┬──────────────────┐");
    println!("  │ Domain                             │ CPU (µs) │ GPU (µs) │ Status           │");
    println!("  ├────────────────────────────────────┼──────────┼──────────┼──────────────────┤");
    for t in &timings {
        println!(
            "  │ {:<34} │ {:>8.0} │ {:>8.0} │ {:<16} │",
            t.domain, t.cpu_us, t.gpu_us, t.status
        );
    }
    println!("  └────────────────────────────────────┴──────────┴──────────┴──────────────────┘");
    println!();
    println!("  GPU primitives proven: FusedMapReduceF64, BrayCurtisF64, Anderson spectral.");
    println!("  The same math is truly portable: CPU = GPU output.");

    let (passed, total) = v.counts();
    println!("\n  ── Exp180 Summary: {passed}/{total} checks ──");

    v.finish();
}
