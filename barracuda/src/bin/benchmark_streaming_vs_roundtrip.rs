// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
//! Exp091: Streaming vs Round-Trip Benchmark
//!
//! Quantifies the performance cost of CPU staging (round-trips) vs
//! `ToadStool`'s unidirectional streaming for multi-stage GPU pipelines.
//!
//! Measures wall-clock time for 3 patterns at increasing batch sizes:
//!
//! 1. **CPU baseline**: Pure CPU computation (no GPU at all)
//! 2. **Round-trip GPU**: Each stage dispatches independently (fresh pipelines)
//! 3. **Streaming GPU**: Pre-warmed `GpuPipelineSession` with cached pipelines
//!
//! Reports speedup ratios and per-dispatch overhead.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCuda` CPU + GPU |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_streaming_vs_roundtrip` |
//! | Data | Synthetic communities (variable batch × 256 features) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu, streaming_gpu, taxonomy};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

const N_FEATURES: usize = 256;

fn make_communities(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|s| {
            (0..N_FEATURES)
                .map(|f| ((s * N_FEATURES + f + 1) as f64).sqrt().max(0.01))
                .collect()
        })
        .collect()
}

fn make_sequences(n: usize) -> Vec<Vec<u8>> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..n)
        .map(|i| (0..80).map(|j| bases[(i * 80 + j) % 4]).collect())
        .collect()
}

fn training_refs() -> Vec<taxonomy::ReferenceSeq> {
    vec![
        taxonomy::ReferenceSeq {
            id: "ref_a".into(),
            sequence: b"ACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Alpha"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_b".into(),
            sequence: b"TGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Beta"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_c".into(),
            sequence: b"ATGATGATGATGATGATGATGATGATG".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Gamma"),
        },
    ]
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp091: Streaming vs Round-Trip Benchmark");

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
    println!("  Session warmup: {}", session.ctx_stats());

    let refs = training_refs();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, 4);
    let params = taxonomy::ClassifyParams::default();

    let t0 = Instant::now();

    println!("\n  ╔═══════════╦═════════════╦═════════════╦═════════════╦═══════════╦═══════════╗");
    println!("  ║ Batch     ║ CPU (µs)    ║ RT GPU (µs) ║ Stream (µs) ║ GPU/CPU   ║ Str/RT    ║");
    println!("  ╠═══════════╬═════════════╬═════════════╬═════════════╬═══════════╬═══════════╣");

    let batch_sizes = [1, 4, 16, 64, 128];
    let mut all_parity = true;

    for &n in &batch_sizes {
        let communities = make_communities(n);
        let sequences = make_sequences(n * 4);
        let seq_refs: Vec<&[u8]> = sequences.iter().map(Vec::as_slice).collect();

        // CPU baseline
        let t_cpu = Instant::now();
        let mut cpu_sh = Vec::with_capacity(n);
        for c in &communities {
            cpu_sh.push(diversity::shannon(c));
            let _ = diversity::simpson(c);
        }
        let _ = streaming_gpu::stream_classify_and_diversity_cpu(
            &classifier,
            &seq_refs,
            &communities[0],
            &params,
        );
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        // Round-trip GPU
        let t_rt = Instant::now();
        let mut rt_sh = Vec::with_capacity(n);
        for c in &communities {
            rt_sh.push(diversity_gpu::shannon_gpu(&gpu, c).unwrap());
            let _ = diversity_gpu::simpson_gpu(&gpu, c).unwrap();
        }
        let rt_us = t_rt.elapsed().as_micros() as f64;

        // Streaming GPU
        let t_st = Instant::now();
        let mut st_sh = Vec::with_capacity(n);
        for c in &communities {
            st_sh.push(session.shannon(c).unwrap());
            let _ = session.simpson(c).unwrap();
        }
        let st_result = session
            .stream_sample(&classifier, &seq_refs, &communities[0], &params)
            .unwrap();
        let st_us = t_st.elapsed().as_micros() as f64;

        let gpu_vs_cpu = if cpu_us > 0.0 { rt_us / cpu_us } else { 0.0 };
        let str_vs_rt = if rt_us > 0.0 { st_us / rt_us } else { 0.0 };

        println!(
            "  ║ {n:>4}×{N_FEATURES}  ║ {cpu_us:>9.0}   ║ {rt_us:>9.0}   ║ {st_us:>9.0}   ║ {gpu_vs_cpu:>7.2}×  ║ {str_vs_rt:>7.2}×  ║",
        );

        // Verify parity
        for i in 0..n {
            if (rt_sh[i] - cpu_sh[i]).abs() > tolerances::GPU_VS_CPU_TRANSCENDENTAL {
                all_parity = false;
            }
            if (st_sh[i] - cpu_sh[i]).abs() > tolerances::GPU_VS_CPU_TRANSCENDENTAL {
                all_parity = false;
            }
        }

        let _ = st_result;
    }

    println!("  ╚═══════════╩═════════════╩═════════════╩═════════════╩═══════════╩═══════════╝");

    v.check_pass(
        "All batch sizes: CPU ↔ RT GPU ↔ streaming parity",
        all_parity,
    );

    // Final parity check at largest batch
    let communities = make_communities(128);
    let rt_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).unwrap();
    let cpu_bc = diversity::bray_curtis_condensed(&communities);

    let bc_max_err = rt_bc
        .iter()
        .zip(cpu_bc.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);

    v.check(
        "Bray-Curtis: max CPU ↔ GPU error",
        bc_max_err,
        0.0,
        tolerances::GPU_VS_CPU_F64,
    );

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  [Total] {ms:.1} ms");
    v.finish();
}
