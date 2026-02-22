// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names
)]
//! Exp072: `ToadStool` Unidirectional Streaming — Zero CPU Round-Trips
//!
//! Proves that chaining GPU stages via pre-warmed pipelines eliminates
//! per-stage dispatch overhead and delivers measurable throughput
//! improvement over individual GPU dispatches.
//!
//! Three paths tested:
//! - Path A: CPU baseline (sequential Rust math)
//! - Path B: GPU individual dispatch (new FMR/BrayCurtis each time)
//! - Path C: GPU streaming (`GpuPipelineSession`, pre-warmed pipelines)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCuda` CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_gpu_streaming_pipeline` |
//! | Data | Synthetic abundance/spectral vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu, spectral_match_gpu, streaming_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

fn cosine_cpu(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = na * nb;
    if denom > 0.0 {
        (dot / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn pairwise_cosine_cpu(spectra: &[Vec<f64>]) -> Vec<f64> {
    let n = spectra.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in 0..i {
            condensed.push(cosine_cpu(&spectra[i], &spectra[j]));
        }
    }
    condensed
}

fn make_abundances(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64).mul_add(2.5, 1.0)).collect()
}

fn make_samples(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|s| {
            (0..n_features)
                .map(|f| ((s * n_features + f + 1) as f64).sqrt())
                .collect()
        })
        .collect()
}

fn make_spectra(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|s| (0..64).map(|f| ((s * 64 + f + 1) as f64) * 0.01).collect())
        .collect()
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp072: ToadStool Streaming Pipeline — Zero CPU Round-Trips");

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

    // ═══════════════════════════════════════════════════════════════════
    // Test data
    // ═══════════════════════════════════════════════════════════════════

    let abundances = make_abundances(256);
    let samples = make_samples(8, 256);
    let spectra = make_spectra(4);

    // ═══════════════════════════════════════════════════════════════════
    // PATH A: CPU Baseline
    // ═══════════════════════════════════════════════════════════════════
    v.section("PATH A: CPU Baseline");

    let t_cpu = Instant::now();
    let cpu_shannon = diversity::shannon(&abundances);
    let cpu_simpson = diversity::simpson(&abundances);
    let cpu_observed = diversity::observed_features(&abundances);
    let cpu_bray = diversity::bray_curtis_condensed(&samples);
    let cpu_cosine = pairwise_cosine_cpu(&spectra);
    let cpu_total_us = t_cpu.elapsed().as_micros() as f64;

    v.check(
        "CPU: Shannon > 0",
        f64::from(u8::from(cpu_shannon > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "CPU: Simpson in (0,1)",
        f64::from(u8::from(cpu_simpson > 0.0 && cpu_simpson < 1.0)),
        1.0,
        0.0,
    );
    v.check("CPU: Observed = 256", cpu_observed, 256.0, 0.0);
    v.check(
        "CPU: Bray-Curtis len = 28",
        cpu_bray.len() as f64,
        28.0,
        0.0,
    );
    v.check("CPU: Cosine len = 6", cpu_cosine.len() as f64, 6.0, 0.0);

    println!("  CPU total: {cpu_total_us:.0} µs");

    // ═══════════════════════════════════════════════════════════════════
    // PATH B: GPU Individual Dispatch (new instances each time)
    // ═══════════════════════════════════════════════════════════════════
    v.section("PATH B: GPU Individual Dispatch");

    let t_ind = Instant::now();
    let ind_shannon = diversity_gpu::shannon_gpu(&gpu, &abundances).unwrap();
    let ind_simpson = diversity_gpu::simpson_gpu(&gpu, &abundances).unwrap();
    let ind_observed = diversity_gpu::observed_features_gpu(&gpu, &abundances).unwrap();
    let ind_bray = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).unwrap();
    let ind_cosine = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spectra).unwrap();
    let ind_total_us = t_ind.elapsed().as_micros() as f64;

    v.check(
        "Individual: Shannon == CPU",
        ind_shannon,
        cpu_shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Individual: Simpson == CPU",
        ind_simpson,
        cpu_simpson,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Individual: Observed == CPU",
        ind_observed,
        cpu_observed,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Individual: Bray-Curtis[0] == CPU",
        ind_bray[0],
        cpu_bray[0],
        tolerances::GPU_VS_CPU_BRAY_CURTIS,
    );
    v.check(
        "Individual: Cosine[0] == CPU",
        ind_cosine[0],
        cpu_cosine[0],
        tolerances::SPECTRAL_COSINE,
    );

    println!("  Individual dispatch total: {ind_total_us:.0} µs");

    // ═══════════════════════════════════════════════════════════════════
    // PATH C: GPU Streaming (pre-warmed GpuPipelineSession)
    // ═══════════════════════════════════════════════════════════════════
    v.section("PATH C: GPU Streaming (Pre-Warmed Session)");

    let session = streaming_gpu::GpuPipelineSession::new(&gpu).unwrap();
    println!("  Session warmup: {:.1} ms", session.warmup_ms);

    let t_stream = Instant::now();
    let stream_shannon = session.shannon(&abundances).unwrap();
    let stream_simpson = session.simpson(&abundances).unwrap();
    let stream_observed = session.observed_features(&abundances).unwrap();
    let stream_total_us = t_stream.elapsed().as_micros() as f64;

    v.check(
        "Streaming: Shannon == CPU",
        stream_shannon,
        cpu_shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Streaming: Simpson == CPU",
        stream_simpson,
        cpu_simpson,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Streaming: Observed == CPU",
        stream_observed,
        cpu_observed,
        tolerances::GPU_VS_CPU_F64,
    );

    println!("  Streaming dispatch total: {stream_total_us:.0} µs");
    println!("  TensorContext stats: {}", session.ctx_stats());

    // ═══════════════════════════════════════════════════════════════════
    // PARITY: all three paths agree
    // ═══════════════════════════════════════════════════════════════════
    v.section("Cross-Path Parity");

    v.check(
        "Shannon: Individual == Streaming",
        ind_shannon,
        stream_shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Simpson: Individual == Streaming",
        ind_simpson,
        stream_simpson,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Observed: Individual == Streaming",
        ind_observed,
        stream_observed,
        tolerances::GPU_VS_CPU_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // SCALING: repeat with larger workload to show streaming advantage
    // ═══════════════════════════════════════════════════════════════════
    v.section("Scaling: 10-Iteration Repeat");

    let big_abundances = make_abundances(2048);

    let t_ind_10 = Instant::now();
    for _ in 0..10 {
        let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device()).unwrap();
        let _ = fmr.shannon_entropy(&big_abundances).unwrap();
        let _ = fmr.simpson_index(&big_abundances).unwrap();
        let _ = fmr.sum(&big_abundances).unwrap();
    }
    let ind_10_us = t_ind_10.elapsed().as_micros() as f64;

    let t_stream_10 = Instant::now();
    for _ in 0..10 {
        let _ = session.shannon(&big_abundances).unwrap();
        let _ = session.simpson(&big_abundances).unwrap();
        let _ = session.observed_features(&big_abundances).unwrap();
    }
    let stream_10_us = t_stream_10.elapsed().as_micros() as f64;

    v.check(
        "10x: streaming total < individual total",
        f64::from(u8::from(stream_10_us < ind_10_us)),
        1.0,
        0.0,
    );

    println!("  10x Individual: {ind_10_us:.0} µs");
    println!("  10x Streaming:  {stream_10_us:.0} µs");
    let speedup = if stream_10_us > 0.0 {
        ind_10_us / stream_10_us
    } else {
        f64::INFINITY
    };
    println!("  Streaming speedup: {speedup:.2}x");

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("┌──────────────────────────────────────────────────────┐");
    println!("│ Exp072 Streaming Pipeline Summary                   │");
    println!("├──────────────────┬──────────────┬───────────────────┤");
    println!("│ Path             │ Time (µs)    │ Notes             │");
    println!("├──────────────────┼──────────────┼───────────────────┤");
    println!("│ A: CPU           │ {cpu_total_us:>12.0} │ baseline          │");
    println!("│ B: GPU indiv.    │ {ind_total_us:>12.0} │ new FMR each time │");
    println!("│ C: GPU streaming │ {stream_total_us:>12.0} │ pre-warmed FMR    │");
    println!("│ 10x Individual   │ {ind_10_us:>12.0} │ repeated dispatch │");
    println!("│ 10x Streaming    │ {stream_10_us:>12.0} │ session reuse     │");
    println!("├──────────────────┼──────────────┼───────────────────┤");
    println!("│ Speedup (10x)    │       {speedup:>5.2}x │ streaming/indiv.  │");
    println!("└──────────────────┴──────────────┴───────────────────┘");

    v.finish();
}
