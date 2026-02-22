// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    unused_assignments
)]
//! Exp073: Compute Dispatch Overhead — Streaming vs Individual vs CPU
//!
//! Quantifies `ToadStool`'s dispatch overhead reduction by measuring the
//! same diversity workload (Shannon + Simpson + Observed) across three
//! strategies at multiple batch sizes.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCuda CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_dispatch_overhead_proof` |
//! | Data | Synthetic abundance vectors at [64, 256, 1024, 4096] |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, streaming_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

fn make_abundances(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64).mul_add(1.5, 0.5)).collect()
}

const BATCH_SIZES: &[usize] = &[64, 256, 1024, 4096];
const REPEATS: usize = 5;

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp073: Compute Dispatch Overhead — Streaming vs Individual");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let session = streaming_gpu::GpuPipelineSession::new(&gpu).unwrap();
    println!("  Session warmup: {:.1} ms", session.warmup_ms);

    let mut results: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut streaming_faster_count = 0_u32;

    for &n in BATCH_SIZES {
        v.section(&format!("═══ Batch N={n} ═══"));
        let abundances = make_abundances(n);

        // Strategy A: CPU
        let t_cpu = Instant::now();
        let mut cpu_shannon = 0.0;
        let mut cpu_simpson = 0.0;
        let mut cpu_observed = 0.0;
        for _ in 0..REPEATS {
            cpu_shannon = diversity::shannon(&abundances);
            cpu_simpson = diversity::simpson(&abundances);
            cpu_observed = diversity::observed_features(&abundances);
        }
        let cpu_us = t_cpu.elapsed().as_micros() as f64 / REPEATS as f64;

        // Strategy B: GPU Individual (new FMR each call)
        let t_ind = Instant::now();
        let mut ind_shannon = 0.0;
        let mut _ind_simpson = 0.0;
        let mut _ind_observed = 0.0;
        for _ in 0..REPEATS {
            let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device()).unwrap();
            ind_shannon = fmr.shannon_entropy(&abundances).unwrap();
            let dom = fmr.simpson_index(&abundances).unwrap();
            _ind_simpson = 1.0 - dom;
            let binary: Vec<f64> = abundances
                .iter()
                .map(|&c| if c > 0.0 { 1.0 } else { 0.0 })
                .collect();
            _ind_observed = fmr.sum(&binary).unwrap();
        }
        let ind_us = t_ind.elapsed().as_micros() as f64 / REPEATS as f64;

        // Strategy C: GPU Streaming (pre-warmed session)
        let t_stream = Instant::now();
        let mut stream_shannon = 0.0;
        let mut stream_simpson = 0.0;
        let mut stream_observed = 0.0;
        for _ in 0..REPEATS {
            stream_shannon = session.shannon(&abundances).unwrap();
            stream_simpson = session.simpson(&abundances).unwrap();
            stream_observed = session.observed_features(&abundances).unwrap();
        }
        let stream_us = t_stream.elapsed().as_micros() as f64 / REPEATS as f64;

        // Parity checks
        v.check(
            &format!("N={n}: Shannon CPU == GPU individual"),
            ind_shannon,
            cpu_shannon,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("N={n}: Shannon CPU == GPU streaming"),
            stream_shannon,
            cpu_shannon,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("N={n}: Simpson CPU == GPU streaming"),
            stream_simpson,
            cpu_simpson,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("N={n}: Observed CPU == GPU streaming"),
            stream_observed,
            cpu_observed,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("N={n}: Streaming overhead < Individual"),
            f64::from(u8::from(stream_us <= ind_us)),
            1.0,
            0.0,
        );

        if stream_us < ind_us {
            streaming_faster_count += 1;
        }

        results.push((n, cpu_us, ind_us, stream_us));

        println!(
            "  N={n:<5}: CPU {cpu_us:>8.0} µs | Indiv {ind_us:>8.0} µs | Stream {stream_us:>8.0} µs"
        );
    }

    // Overall: streaming should beat individual at least 3/4 of the time
    v.section("Overall Streaming Advantage");
    v.check(
        "Streaming faster than individual in ≥3/4 batch sizes",
        f64::from(u8::from(streaming_faster_count >= 3)),
        1.0,
        0.0,
    );

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Exp073 Dispatch Overhead Summary (avg over {REPEATS} repeats)             │");
    println!("├──────────┬────────────┬────────────┬────────────┬──────────────┤");
    println!("│ Batch N  │ CPU (µs)   │ Indiv (µs) │ Stream(µs) │ Stream/Ind   │");
    println!("├──────────┼────────────┼────────────┼────────────┼──────────────┤");
    for &(n, cpu, ind, stream) in &results {
        let ratio = if ind > 0.0 { stream / ind } else { f64::NAN };
        println!("│ {n:>8} │ {cpu:>10.0} │ {ind:>10.0} │ {stream:>10.0} │ {ratio:>11.3}x │");
    }
    println!("└──────────┴────────────┴────────────┴────────────┴──────────────┘");
    println!("  TensorContext stats: {}", session.ctx_stats());

    v.finish();
}
