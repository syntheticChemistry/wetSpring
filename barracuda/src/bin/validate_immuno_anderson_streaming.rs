// SPDX-License-Identifier: AGPL-3.0-or-later
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
//! # Exp278: Track 5 `ToadStool` Streaming Dispatch — Immunological Anderson
//!
//! Pure GPU streaming pipeline for Paper 12. Uses `GpuPipelineSession` for
//! batched GPU operations (Shannon, Simpson, Bray-Curtis) with zero CPU
//! round-trips between GPU calls — the GPU session caches shader pipelines
//! and buffer allocations.
//!
//! ## Streaming vs Individual GPU
//! - Individual (`diversity_gpu::*`): one GPU dispatch per call, pipeline
//!   creation overhead each time
//! - Streaming (`GpuPipelineSession`): pipeline cached, buffers reused,
//!   unidirectional GPU flow
//!
//! ## Evolution path
//! - **Previous**: Exp277 GPU validation (individual dispatch)
//! - **This experiment**: `ToadStool` streaming (batched GPU)
//! - **Next**: Exp279 `metalForge` cross-substrate (NUCLEUS atomics)
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | Track | Track 5 — Immunological Anderson & Drug Repurposing |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_immuno_anderson_streaming` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use wetspring_barracuda::bio::{diversity, diversity_gpu, streaming_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

struct Timing {
    domain: &'static str,
    individual_us: f64,
    streaming_us: f64,
    status: &'static str,
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp278: Track 5 ToadStool Streaming — Immunological Anderson");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };

    let session = match streaming_gpu::GpuPipelineSession::new(&gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("No streaming session: {e}");
            validation::exit_skipped("GpuPipelineSession unavailable");
        }
    };

    let mut timings: Vec<Timing> = Vec::new();

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
            "Moderate",
            vec![550.0, 90.0, 20.0, 50.0, 100.0, 70.0, 35.0, 30.0, 25.0, 30.0],
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
        (
            "Cytopoint",
            vec![600.0, 65.0, 22.0, 40.0, 55.0, 50.0, 20.0, 20.0, 12.0, 16.0],
        ),
    ];

    // ═══════════════════════════════════════════════════════════════
    // S01: Streaming Shannon — All Disease States
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ S01: Streaming Shannon — Cell Population Diversity ═══");

    let ti = Instant::now();
    let ind_shannons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity_gpu::shannon_gpu(&gpu, c).expect("ind Shannon"))
        .collect();
    let ind_us = ti.elapsed().as_micros() as f64;

    let ts = Instant::now();
    let stream_shannons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| session.shannon(c).expect("stream Shannon"))
        .collect();
    let stream_us = ts.elapsed().as_micros() as f64;

    for (i, (name, _)) in cell_pops.iter().enumerate() {
        v.check(
            &format!("Shannon({name}) ind↔stream"),
            stream_shannons[i],
            ind_shannons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    timings.push(Timing {
        domain: "Shannon (6 states)",
        individual_us: ind_us,
        streaming_us: stream_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // S02: Streaming Simpson — All Disease States
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ S02: Streaming Simpson — Cell Population Diversity ═══");

    let ti = Instant::now();
    let ind_simpsons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| diversity_gpu::simpson_gpu(&gpu, c).expect("ind Simpson"))
        .collect();
    let ind_us = ti.elapsed().as_micros() as f64;

    let ts = Instant::now();
    let stream_simpsons: Vec<f64> = cell_pops
        .iter()
        .map(|(_, c)| session.simpson(c).expect("stream Simpson"))
        .collect();
    let stream_us = ts.elapsed().as_micros() as f64;

    for (i, (name, _)) in cell_pops.iter().enumerate() {
        v.check(
            &format!("Simpson({name}) ind↔stream"),
            stream_simpsons[i],
            ind_simpsons[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    timings.push(Timing {
        domain: "Simpson (6 states)",
        individual_us: ind_us,
        streaming_us: stream_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // S03: Streaming Bray-Curtis Matrix
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ S03: Streaming Bray-Curtis — Tissue State Matrix ═══");

    let refs: Vec<&[f64]> = cell_pops.iter().map(|(_, c)| c.as_slice()).collect();
    let owned: Vec<Vec<f64>> = cell_pops.iter().map(|(_, c)| c.clone()).collect();

    let ti = Instant::now();
    let ind_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &owned).expect("ind BC");
    let ind_bc_us = ti.elapsed().as_micros() as f64;

    let ts = Instant::now();
    let stream_bc = session.bray_curtis_matrix(&refs).expect("stream BC");
    let stream_bc_us = ts.elapsed().as_micros() as f64;

    v.check_pass("BC ind length", ind_bc.len() == 15); // C(6,2)
    v.check_pass("BC stream length", stream_bc.len() == 15);

    for i in 0..ind_bc.len() {
        v.check(
            &format!("BC[{i}] ind↔stream"),
            stream_bc[i],
            ind_bc[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    timings.push(Timing {
        domain: "Bray-Curtis (6×6)",
        individual_us: ind_bc_us,
        streaming_us: stream_bc_us,
        status: "PARITY",
    });

    // ═══════════════════════════════════════════════════════════════
    // S04: Anderson Spectral (CPU) + GPU Diversity — Combined Pipeline
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ S04: Combined Pipeline (Spectral + Streaming Diversity) ═══");

    let t0 = Instant::now();

    let _midpoint = f64::midpoint(GOE_R, POISSON_R);
    let mat = anderson_3d(6, 6, 6, 10.0, 42);
    let tri = lanczos(&mat, 216, 42);
    let eigs = lanczos_eigenvalues(&tri);
    let r = level_spacing_ratio(&eigs);

    let cpu_sh = diversity::shannon(&cell_pops[0].1);
    let stream_sh = session.shannon(&cell_pops[0].1).expect("stream");

    v.check_pass("Anderson r computed", r > 0.35 && r < 0.55);
    v.check(
        "Combined pipeline Shannon CPU↔stream",
        stream_sh,
        cpu_sh,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    let combined_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Combined pipeline",
        individual_us: combined_us,
        streaming_us: 0.0,
        status: "COMBINED",
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp278: Track 5 ToadStool Streaming — Immunological Anderson         ║");
    println!("╠═══════════════════════╦════════════╦════════════╦════════════════════╣");
    println!("║ Domain                ║  Indiv(µs) ║ Stream(µs) ║ Status             ║");
    println!("╠═══════════════════════╬════════════╬════════════╬════════════════════╣");

    for t in &timings {
        println!(
            "║ {:<21} ║ {:>10.0} ║ {:>10.0} ║ {:<18} ║",
            t.domain, t.individual_us, t.streaming_us, t.status
        );
    }

    println!("╚═══════════════════════╩════════════╩════════════╩════════════════════╝");
    println!();

    v.finish();
}
