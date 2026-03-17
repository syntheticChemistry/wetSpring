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
//! # Exp285: `ToadStool` Streaming — Gonzales Reproductions
//!
//! Pure GPU streaming pipeline for Gonzales paper reproductions. Uses
//! `GpuPipelineSession` for batched GPU operations with zero CPU round-trips
//! between GPU calls. Validates that streaming produces identical results to
//! individual GPU dispatch while reducing overhead.
//!
//! ## Streaming vs Individual GPU
//! - Individual (`diversity_gpu::*`): one GPU dispatch per call
//! - Streaming (`GpuPipelineSession`): pipeline cached, buffers reused,
//!   unidirectional GPU flow — what `ToadStool` absorbs for production
//!
//! ## Evolution chain
//! - **Previous**: Exp284 GPU validation
//! - **This**: `ToadStool` streaming (batched GPU)
//! - **Next**: Exp286 `metalForge` cross-substrate (NUCLEUS atomics)
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Validates | Exp280-282 streaming portability (Papers 53-56) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_gonzales_streaming` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::bio::{diversity, diversity_gpu, streaming_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::{self, Validator};

struct Timing {
    domain: &'static str,
    individual_us: f64,
    streaming_us: f64,
    checks: usize,
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp285: ToadStool Streaming — Gonzales Reproductions");

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
    let tol = tolerances::GPU_VS_CPU_TRANSCENDENTAL;

    // Gonzales cell populations from published data
    let populations: Vec<(&str, Vec<f64>)> = vec![
        ("Immune", vec![40.0, 25.0, 15.0, 10.0, 10.0]),
        ("Skin", vec![70.0, 15.0, 10.0, 5.0]),
        ("Neural", vec![50.0, 30.0, 20.0]),
        ("Receptor", vec![95.0, 60.0, 35.0, 12.0, 6.0, 27.0, 27.0]),
        ("IC50", vec![10.0, 36.0, 71.0, 80.0, 150.0, 249.0]),
    ];

    // ═══════════════════════════════════════════════════════════════
    // D01: Streaming Shannon
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Streaming Shannon ═══");

    let mut d1_checks = 0_usize;
    let mut d1_ind = 0.0_f64;
    let mut d1_str = 0.0_f64;

    for (name, pop) in &populations {
        let ti = Instant::now();
        let ind_val = diversity_gpu::shannon_gpu(&gpu, pop).or_exit("individual shannon");
        d1_ind += ti.elapsed().as_micros() as f64;

        let ts = Instant::now();
        let str_val = session.shannon(pop).or_exit("streaming shannon");
        d1_str += ts.elapsed().as_micros() as f64;

        let diff = (ind_val - str_val).abs();
        v.check_pass(
            &format!("{name} Shannon stream≈individual (diff={diff:.2e})"),
            diff < tol,
        );

        let cpu_val = diversity::shannon(pop);
        v.check_pass(
            &format!(
                "{name} Shannon stream≈CPU (diff={:.2e})",
                (cpu_val - str_val).abs()
            ),
            (cpu_val - str_val).abs() < tol,
        );

        d1_checks += 2;
    }

    timings.push(Timing {
        domain: "Streaming Shannon",
        individual_us: d1_ind,
        streaming_us: d1_str,
        checks: d1_checks,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Streaming Simpson
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Streaming Simpson ═══");

    let mut d2_checks = 0_usize;
    let mut d2_ind = 0.0_f64;
    let mut d2_str = 0.0_f64;

    for (name, pop) in &populations {
        let ti = Instant::now();
        let ind_val = diversity_gpu::simpson_gpu(&gpu, pop).or_exit("individual simpson");
        d2_ind += ti.elapsed().as_micros() as f64;

        let ts = Instant::now();
        let str_val = session.simpson(pop).or_exit("streaming simpson");
        d2_str += ts.elapsed().as_micros() as f64;

        let diff = (ind_val - str_val).abs();
        v.check_pass(
            &format!("{name} Simpson stream≈individual (diff={diff:.2e})"),
            diff < tol,
        );

        let cpu_val = diversity::simpson(pop);
        v.check_pass(
            &format!(
                "{name} Simpson stream≈CPU (diff={:.2e})",
                (cpu_val - str_val).abs()
            ),
            (cpu_val - str_val).abs() < tol,
        );

        d2_checks += 2;
    }

    timings.push(Timing {
        domain: "Streaming Simpson",
        individual_us: d2_ind,
        streaming_us: d2_str,
        checks: d2_checks,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Streaming Bray-Curtis
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Streaming Bray-Curtis ═══");

    let healthy = vec![5.0, 8.0, 3.0, 6.0, 4.0, 7.0];
    let ad_state = vec![45.0, 62.0, 38.0, 55.0, 72.0, 48.0];
    let mild_ad = vec![15.0, 20.0, 12.0, 18.0, 22.0, 16.0];

    let samples_vec = vec![healthy.clone(), mild_ad.clone(), ad_state.clone()];
    let sample_refs: Vec<&[f64]> = samples_vec.iter().map(Vec::as_slice).collect();

    let ti = Instant::now();
    let ind_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples_vec)
        .or_exit("individual BC condensed");
    let d3_ind = ti.elapsed().as_micros() as f64;

    let ts = Instant::now();
    let str_bc = session
        .bray_curtis_matrix(&sample_refs)
        .or_exit("streaming BC matrix");
    let d3_str = ts.elapsed().as_micros() as f64;

    let labels = ["healthy↔mild", "healthy↔AD", "mild↔AD"];
    let cpu_bcs = [
        diversity::bray_curtis(&healthy, &mild_ad),
        diversity::bray_curtis(&healthy, &ad_state),
        diversity::bray_curtis(&mild_ad, &ad_state),
    ];

    let mut d3_checks = 0_usize;
    for (i, label) in labels.iter().enumerate() {
        let diff_ind = (ind_bc[i] - str_bc[i]).abs();
        v.check_pass(
            &format!("{label} BC stream≈individual (diff={diff_ind:.2e})"),
            diff_ind < tol,
        );

        let diff_cpu = (cpu_bcs[i] - str_bc[i]).abs();
        v.check_pass(
            &format!("{label} BC stream≈CPU (diff={diff_cpu:.2e})"),
            diff_cpu < tol,
        );
        d3_checks += 2;
    }

    timings.push(Timing {
        domain: "Streaming BC",
        individual_us: d3_ind,
        streaming_us: d3_str,
        checks: d3_checks,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Full Pipeline Batch — All Metrics
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Full Pipeline Batch ═══");
    let t_batch = Instant::now();

    let all_pops: Vec<&Vec<f64>> = populations.iter().map(|(_, p)| p).collect();
    let mut batch_checks = 0_usize;

    for pop in &all_pops {
        let str_sh = session.shannon(pop).or_exit("batch shannon");
        let str_si = session.simpson(pop).or_exit("batch simpson");
        let cpu_sh = diversity::shannon(pop);
        let cpu_si = diversity::simpson(pop);

        v.check_pass(
            &format!("Batch Shannon parity (n={})", pop.len()),
            (str_sh - cpu_sh).abs() < tol,
        );
        v.check_pass(
            &format!("Batch Simpson parity (n={})", pop.len()),
            (str_si - cpu_si).abs() < tol,
        );
        batch_checks += 2;
    }

    // Large batch
    let large: Vec<f64> = (0..50_000)
        .map(|i| f64::from((i * 17 + 7) % 500) + 1.0)
        .collect();
    let str_large_sh = session.shannon(&large).or_exit("large batch shannon");
    let cpu_large_sh = diversity::shannon(&large);
    v.check_pass(
        &format!(
            "50K batch Shannon parity (diff={:.2e})",
            (str_large_sh - cpu_large_sh).abs()
        ),
        (str_large_sh - cpu_large_sh).abs() < tol,
    );
    batch_checks += 1;

    let d4_us = t_batch.elapsed().as_micros() as f64;

    timings.push(Timing {
        domain: "Full batch",
        individual_us: d4_us,
        streaming_us: d4_us,
        checks: batch_checks,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp285: ToadStool Streaming — Gonzales Reproductions               ║");
    println!("╠═════════════════════════╦════════════╦════════════╦════════════════╣");
    println!("║ Domain                  ║  Indiv(µs) ║ Stream(µs) ║ Checks         ║");
    println!("╠═════════════════════════╬════════════╬════════════╬════════════════╣");

    let mut total_checks = 0_usize;
    for t in &timings {
        println!(
            "║ {:<23} ║ {:>10.0} ║ {:>10.0} ║ {:>3}            ║",
            t.domain, t.individual_us, t.streaming_us, t.checks
        );
        total_checks += t.checks;
    }

    println!("╠═════════════════════════╬════════════╬════════════╬════════════════╣");
    println!(
        "║ TOTAL                   ║            ║            ║ {total_checks:>3}            ║"
    );
    println!("╚═════════════════════════╩════════════╩════════════╩════════════════╝");
    println!();

    v.finish();
}
