// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
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
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | `BarraCuda` CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_dispatch_overhead_proof` |
//! | Data | Synthetic abundance vectors at [64, 256, 1024, 4096] |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, streaming_gpu};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator, test_data};

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

const BATCH_SIZES: &[usize] = &[64, 256, 1024, 4096];
const REPEATS: usize = 5;

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp073: Compute Dispatch Overhead — Streaming vs Individual");

    let gpu = validation::gpu_or_skip().await;

    let session = streaming_gpu::GpuPipelineSession::new(&gpu).expect("dispatch overhead");
    println!("  Session warmup: {:.1} ms", session.warmup_ms);

    let mut results: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut streaming_faster_count = 0_u32;

    for &n in BATCH_SIZES {
        v.section(&format!("═══ Batch N={n} ═══"));
        let abundances = test_data::make_abundances(n);

        // Strategy A: CPU — time REPEATS iterations, capture values on final run
        let t_cpu = Instant::now();
        for _ in 0..REPEATS - 1 {
            let _ = diversity::shannon(&abundances);
            let _ = diversity::simpson(&abundances);
            let _ = diversity::observed_features(&abundances);
        }
        let cpu_shannon = diversity::shannon(&abundances);
        let cpu_simpson = diversity::simpson(&abundances);
        let cpu_observed = diversity::observed_features(&abundances);
        let cpu_us = t_cpu.elapsed().as_micros() as f64 / REPEATS as f64;

        // Strategy B: GPU Individual (new FMR each call)
        let t_ind = Instant::now();
        for _ in 0..REPEATS - 1 {
            let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device()).expect("dispatch overhead");
            let _ = fmr.shannon_entropy(&abundances).expect("dispatch overhead");
            let _ = fmr.simpson_index(&abundances).expect("dispatch overhead");
            let binary: Vec<f64> = abundances
                .iter()
                .map(|&c| if c > 0.0 { 1.0 } else { 0.0 })
                .collect();
            let _ = fmr.sum(&binary).expect("dispatch overhead");
        }
        let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device()).expect("dispatch overhead");
        let ind_shannon = fmr.shannon_entropy(&abundances).expect("dispatch overhead");
        let ind_us = t_ind.elapsed().as_micros() as f64 / REPEATS as f64;

        // Strategy C: GPU Streaming (pre-warmed session)
        let t_stream = Instant::now();
        for _ in 0..REPEATS - 1 {
            let _ = session.shannon(&abundances).expect("dispatch overhead");
            let _ = session.simpson(&abundances).expect("dispatch overhead");
            let _ = session
                .observed_features(&abundances)
                .expect("dispatch overhead");
        }
        let stream_shannon = session.shannon(&abundances).expect("dispatch overhead");
        let stream_simpson = session.simpson(&abundances).expect("dispatch overhead");
        let stream_observed = session
            .observed_features(&abundances)
            .expect("dispatch overhead");
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
            tolerances::EXACT,
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
        tolerances::EXACT,
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
