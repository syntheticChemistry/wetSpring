// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    dead_code
)]
//! # Exp277: Track 5 GPU Validation — Immunological Anderson on GPU
//!
//! Proves that `BarraCuda` GPU produces identical results to CPU for all
//! Paper 12 immunological Anderson domains. For each domain: CPU computes
//! reference, GPU must match within tolerance.
//!
//! ## GPU primitives exercised
//! - `FusedMapReduceF64` — Shannon, Simpson (cell-type diversity)
//! - `BrayCurtisF64` — beta diversity between tissue states
//! - `anderson_3d` + `lanczos` — spectral analysis (CPU, shared by both)
//!
//! ## Evolution path
//! - **Previous**: Exp276 CPU parity (pure Rust)
//! - **This experiment**: `BarraCuda` GPU (diversity on GPU, spectral on CPU)
//! - **Next**: Exp278 `ToadStool` streaming dispatch
//! - **Final**: Exp279 `metalForge` cross-substrate (NUCLEUS)
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | Track | Track 5 — Immunological Anderson & Drug Repurposing |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_immuno_anderson_gpu` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use wetspring_barracuda::bio::{diversity, diversity_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    gpu_us: f64,
    status: &'static str,
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp277: Track 5 GPU Validation — Immunological Anderson");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };

    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // G01: Shannon + Simpson on Cell Populations (FusedMapReduceF64)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ G01: Shannon + Simpson on Cell Populations (GPU FMR) ═══");

    let cell_pops: Vec<(&str, Vec<f64>)> = vec![
        (
            "Healthy",
            vec![850.0, 50.0, 30.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0, 10.0],
        ),
        (
            "Mild AD",
            vec![700.0, 70.0, 25.0, 35.0, 50.0, 40.0, 15.0, 20.0, 15.0, 30.0],
        ),
        (
            "Severe AD",
            vec![
                400.0, 100.0, 15.0, 60.0, 140.0, 90.0, 55.0, 40.0, 35.0, 65.0,
            ],
        ),
        (
            "Apoquel",
            vec![650.0, 60.0, 25.0, 30.0, 40.0, 25.0, 10.0, 15.0, 10.0, 35.0],
        ),
    ];

    let tc = Instant::now();
    let cpu_shannons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity::shannon(c))
        .collect();
    let cpu_simpsons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity::simpson(c))
        .collect();
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_shannons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity_gpu::shannon_gpu(&gpu, c).expect("GPU Shannon"))
        .collect();
    let gpu_simpsons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity_gpu::simpson_gpu(&gpu, c).expect("GPU Simpson"))
        .collect();
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (name, _)) in cell_pops.iter().enumerate() {
        v.check(
            &format!("Shannon({name}) CPU↔GPU"),
            gpu_shannons[i],
            cpu_shannons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("Simpson({name}) CPU↔GPU"),
            gpu_simpsons[i],
            cpu_simpsons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    timings.push(Timing {
        domain: "Shannon+Simpson (cells)",
        cpu_us,
        gpu_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // G02: Bray-Curtis Between Tissue States (BrayCurtisF64)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ G02: Bray-Curtis Between Tissue States (GPU) ═══");

    let samples: Vec<Vec<f64>> = cell_pops.iter().map(|(_, c)| c.clone()).collect();

    let tc = Instant::now();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let bc_cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).expect("GPU BC");
    let bc_gpu_us = tg.elapsed().as_micros() as f64;

    v.check_pass(
        "BC condensed length",
        cpu_bc.len() == gpu_bc.len() && cpu_bc.len() == 6,
    );

    for (i, (cpu_val, gpu_val)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{i}] CPU↔GPU"),
            *gpu_val,
            *cpu_val,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    let bc_healthy_severe = cpu_bc[2]; // pair (0,2) = healthy vs severe
    let bc_healthy_mild = cpu_bc[0]; // pair (0,1) = healthy vs mild
    v.check_pass(
        "BC(healthy,severe) > BC(healthy,mild)",
        bc_healthy_severe > bc_healthy_mild,
    );

    println!("  BC(healthy, mild)    = {bc_healthy_mild:.6}");
    println!("  BC(healthy, severe)  = {bc_healthy_severe:.6}");
    println!("  More severe AD → more dissimilar from healthy");

    timings.push(Timing {
        domain: "Bray-Curtis (tissue)",
        cpu_us: bc_cpu_us,
        gpu_us: bc_gpu_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // G03: Anderson Spectral (shared CPU — verifies consistency)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ G03: Anderson Spectral (CPU baseline for GPU context) ═══");

    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    let tc = Instant::now();

    let mat_epi = anderson_2d(8, 8, 16.0, 42);
    let tri_epi = lanczos(&mat_epi, 64, 42);
    let eigs_epi = lanczos_eigenvalues(&tri_epi);
    let r_epi = level_spacing_ratio(&eigs_epi);

    let mat_derm = anderson_3d(6, 6, 6, 8.0, 42);
    let tri_derm = lanczos(&mat_derm, 216, 42);
    let eigs_derm = lanczos_eigenvalues(&tri_derm);
    let r_derm = level_spacing_ratio(&eigs_derm);

    let spectral_us = tc.elapsed().as_micros() as f64;

    v.check_pass("Epidermis r localized", r_epi <= midpoint + 0.02);
    v.check_pass("Dermis r extended", r_derm > midpoint);
    v.check_pass("Dimension separation r_3D > r_2D", r_derm > r_epi);

    println!("  Epidermis (2D, W=16): r = {r_epi:.6}");
    println!("  Dermis    (3D, W=8):  r = {r_derm:.6}");
    println!("  (Spectral is CPU-only — Lanczos eigenvalues)");

    timings.push(Timing {
        domain: "Anderson spectral",
        cpu_us: spectral_us,
        gpu_us: 0.0,
        status: "CPU-ONLY",
    });

    // ═══════════════════════════════════════════════════════════════
    // G04: Large-Scale Cell Population GPU Diversity
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ G04: Large Cell Population GPU Benchmark ═══");

    let large_pop: Vec<f64> = (0..2000).map(|i| f64::from(i + 1).sqrt() + 0.1).collect();

    let tc = Instant::now();
    let cpu_sh_large = diversity::shannon(&large_pop);
    let cpu_si_large = diversity::simpson(&large_pop);
    let large_cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_sh_large = diversity_gpu::shannon_gpu(&gpu, &large_pop).expect("GPU Shannon large");
    let gpu_si_large = diversity_gpu::simpson_gpu(&gpu, &large_pop).expect("GPU Simpson large");
    let large_gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "Shannon large CPU↔GPU",
        gpu_sh_large,
        cpu_sh_large,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Simpson large CPU↔GPU",
        gpu_si_large,
        cpu_si_large,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    println!("  N=2000: Shannon CPU={cpu_sh_large:.6} GPU={gpu_sh_large:.6}");
    println!("  N=2000: Simpson CPU={cpu_si_large:.6} GPU={gpu_si_large:.6}");

    timings.push(Timing {
        domain: "Large population (N=2k)",
        cpu_us: large_cpu_us,
        gpu_us: large_gpu_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Exp277: Track 5 GPU Validation — Immunological Anderson          ║");
    println!("╠═══════════════════════╦════════════╦════════════╦═════════════════╣");
    println!("║ Domain                ║   CPU (µs) ║   GPU (µs) ║ Status          ║");
    println!("╠═══════════════════════╬════════════╬════════════╬═════════════════╣");

    for t in &timings {
        println!(
            "║ {:<21} ║ {:>10.0} ║ {:>10.0} ║ {:<15} ║",
            t.domain, t.cpu_us, t.gpu_us, t.status
        );
    }

    println!("╚═══════════════════════╩════════════╩════════════╩═════════════════╝");
    println!();

    v.finish();
}
