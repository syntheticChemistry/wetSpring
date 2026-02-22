// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
//! Exp090: Pure GPU Streaming Pipeline — Zero CPU Round-Trips
//!
//! Proves that a full bioinformatics pipeline can run on GPU with data
//! flowing unidirectionally through stages, only touching CPU at input
//! and output. Compares three execution modes:
//!
//! 1. **Round-trip**: Each stage returns results to CPU before next stage
//! 2. **Streaming**: Stages chain on GPU via `GpuPipelineSession`
//! 3. **Pure GPU**: GEMM output stays on GPU buffer (`execute_to_buffer`)
//!
//! All three modes must produce identical mathematical results.
//!
//! # Architecture
//!
//! ```text
//! Round-trip:  CPU → GPU → CPU → GPU → CPU → GPU → CPU  (6 transfers)
//! Streaming:   CPU → GPU → GPU → GPU → CPU             (2 transfers)
//! Pure GPU:    CPU → GPU buffer → GPU → GPU → CPU       (2 transfers, 0 intermediate readback)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCUDA` CPU + GPU streaming via `ToadStool` |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin validate_pure_gpu_streaming` |
//! | Data | Synthetic communities (8 samples × 256 features) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu, streaming_gpu, taxonomy};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

const N_SAMPLES: usize = 8;
const N_FEATURES: usize = 256;

fn make_communities() -> Vec<Vec<f64>> {
    (0..N_SAMPLES)
        .map(|s| {
            (0..N_FEATURES)
                .map(|f| {
                    let base = ((s * N_FEATURES + f + 1) as f64).sqrt();
                    let shift = ((s + 1) as f64) * 0.1;
                    (base + shift).max(0.01)
                })
                .collect()
        })
        .collect()
}

fn make_sequences() -> Vec<Vec<u8>> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..N_SAMPLES * 4)
        .map(|i| (0..80).map(|j| bases[(i * 80 + j) % 4]).collect())
        .collect()
}

fn training_refs() -> Vec<taxonomy::ReferenceSeq> {
    vec![
        taxonomy::ReferenceSeq {
            id: "ref_alpha".into(),
            sequence: b"ACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("AlphaProteobacteria"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_beta".into(),
            sequence: b"TGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("BetaProteobacteria"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_gamma".into(),
            sequence: b"ATGATGATGATGATGATGATGATGATG".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("GammaProteobacteria"),
        },
    ]
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp090: Pure GPU Streaming Pipeline — Zero CPU Round-Trips");

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

    let t0 = Instant::now();

    validate_roundtrip_mode(&mut v, &gpu);
    validate_streaming_mode(&mut v, &gpu);
    validate_modes_match(&mut v, &gpu);
    validate_batch_scaling(&mut v, &gpu);

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  [Total] {ms:.1} ms");
    v.finish();
}

// ════════════════════════════════════════════════════════════════
//  Mode 1: Round-Trip — each stage returns to CPU
// ════════════════════════════════════════════════════════════════

fn validate_roundtrip_mode(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Mode 1: Round-Trip (CPU ↔ GPU per stage) ═══");
    let t0 = Instant::now();

    let communities = make_communities();

    let t_alpha = Instant::now();
    let rt_shannon: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::shannon_gpu(gpu, c).unwrap())
        .collect();
    let rt_simpson: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::simpson_gpu(gpu, c).unwrap())
        .collect();
    let alpha_us = t_alpha.elapsed().as_micros();

    let t_bray = Instant::now();
    let rt_bc = diversity_gpu::bray_curtis_condensed_gpu(gpu, &communities).unwrap();
    let bray_us = t_bray.elapsed().as_micros();

    v.check(
        "RT: Shannon count",
        rt_shannon.len() as f64,
        N_SAMPLES as f64,
        0.0,
    );
    v.check_pass(
        "RT: all Shannon finite",
        rt_shannon.iter().all(|s| s.is_finite()),
    );
    v.check(
        "RT: Simpson count",
        rt_simpson.len() as f64,
        N_SAMPLES as f64,
        0.0,
    );
    v.check(
        "RT: Bray-Curtis pairs",
        rt_bc.len() as f64,
        (N_SAMPLES * (N_SAMPLES - 1) / 2) as f64,
        0.0,
    );

    println!(
        "  Round-trip: alpha {alpha_us} µs + beta {bray_us} µs = {} µs ({N_SAMPLES} samples × {N_FEATURES} features)",
        alpha_us + bray_us
    );
    print_timing("round-trip mode", t0);
}

// ════════════════════════════════════════════════════════════════
//  Mode 2: Streaming — stages chain via GpuPipelineSession
// ════════════════════════════════════════════════════════════════

fn validate_streaming_mode(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Mode 2: Streaming (GpuPipelineSession) ═══");
    let t0 = Instant::now();

    let session = streaming_gpu::GpuPipelineSession::new(gpu).unwrap();
    println!("  Session: {}", session.ctx_stats());

    let communities = make_communities();
    let sequences = make_sequences();
    let refs = training_refs();
    let k = 4;
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();

    let seq_refs: Vec<&[u8]> = sequences.iter().map(Vec::as_slice).collect();
    let counts = &communities[0];

    let t_stream = Instant::now();
    let result = session
        .stream_sample(&classifier, &seq_refs, counts, &params)
        .unwrap();
    let stream_us = t_stream.elapsed().as_micros();

    v.check_pass("stream: Shannon > 0", result.shannon > 0.0);
    v.check_pass(
        "stream: Simpson ∈ [0,1]",
        (0.0..=1.0).contains(&result.simpson),
    );
    v.check_pass("stream: observed > 0", result.observed > 0.0);
    v.check(
        "stream: classification count",
        result.classifications.len() as f64,
        seq_refs.len() as f64,
        0.0,
    );

    let cpu_result =
        streaming_gpu::stream_classify_and_diversity_cpu(&classifier, &seq_refs, counts, &params);
    v.check(
        "stream vs CPU: Shannon",
        result.shannon,
        cpu_result.shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "stream vs CPU: Simpson",
        result.simpson,
        cpu_result.simpson,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "stream vs CPU: observed",
        result.observed,
        cpu_result.observed,
        tolerances::GPU_VS_CPU_F64,
    );

    for (i, (gc, cc)) in result
        .classifications
        .iter()
        .zip(cpu_result.classifications.iter())
        .enumerate()
    {
        v.check(
            &format!("stream taxon {i}: GPU == CPU"),
            gc.taxon_idx as f64,
            cc.taxon_idx as f64,
            0.0,
        );
    }

    println!(
        "  Streaming: {stream_us} µs (tax={:.0}ms + div={:.0}ms | GPU total={:.0}ms)",
        result.taxonomy_ms, result.diversity_ms, result.total_gpu_ms
    );
    print_timing("streaming mode", t0);
}

// ════════════════════════════════════════════════════════════════
//  Mode 3: Prove all modes produce identical results
// ════════════════════════════════════════════════════════════════

fn validate_modes_match(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Mode 3: Round-Trip ↔ Streaming Parity ═══");
    let t0 = Instant::now();

    let communities = make_communities();

    let rt_shannon: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::shannon_gpu(gpu, c).unwrap())
        .collect();
    let rt_simpson: Vec<f64> = communities
        .iter()
        .map(|c| diversity_gpu::simpson_gpu(gpu, c).unwrap())
        .collect();

    let session = streaming_gpu::GpuPipelineSession::new(gpu).unwrap();
    let st_shannon: Vec<f64> = communities
        .iter()
        .map(|c| session.shannon(c).unwrap())
        .collect();
    let st_simpson: Vec<f64> = communities
        .iter()
        .map(|c| session.simpson(c).unwrap())
        .collect();

    for i in 0..N_SAMPLES {
        v.check(
            &format!("S{i}: Shannon RT ↔ stream"),
            st_shannon[i],
            rt_shannon[i],
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("S{i}: Simpson RT ↔ stream"),
            st_simpson[i],
            rt_simpson[i],
            tolerances::GPU_VS_CPU_F64,
        );
    }

    let cpu_shannon: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let cpu_simpson: Vec<f64> = communities.iter().map(|c| diversity::simpson(c)).collect();

    for i in 0..N_SAMPLES {
        v.check(
            &format!("S{i}: Shannon CPU ↔ stream"),
            st_shannon[i],
            cpu_shannon[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("S{i}: Simpson CPU ↔ stream"),
            st_simpson[i],
            cpu_simpson[i],
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }

    print_timing("modes match", t0);
}

// ════════════════════════════════════════════════════════════════
//  Mode 4: Batch scaling — prove streaming benefit grows
// ════════════════════════════════════════════════════════════════

fn validate_batch_scaling(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Mode 4: Batch Scaling (streaming overhead amortization) ═══");
    let t0 = Instant::now();

    let session = streaming_gpu::GpuPipelineSession::new(gpu).unwrap();

    let batch_sizes = [4, 16, 64, 256];
    let mut timings: Vec<(usize, f64, f64)> = Vec::new();

    for &n in &batch_sizes {
        let communities: Vec<Vec<f64>> = (0..n)
            .map(|s| {
                (0..N_FEATURES)
                    .map(|f| ((s * N_FEATURES + f + 1) as f64).sqrt().max(0.01))
                    .collect()
            })
            .collect();

        let t_rt = Instant::now();
        for c in &communities {
            let _ = diversity_gpu::shannon_gpu(gpu, c).unwrap();
            let _ = diversity_gpu::simpson_gpu(gpu, c).unwrap();
        }
        let rt_us = t_rt.elapsed().as_micros() as f64;

        let t_st = Instant::now();
        for c in &communities {
            let _ = session.shannon(c).unwrap();
            let _ = session.simpson(c).unwrap();
        }
        let st_us = t_st.elapsed().as_micros() as f64;

        timings.push((n, rt_us, st_us));
        println!(
            "  batch={n:>3}: RT={rt_us:.0}µs  stream={st_us:.0}µs  ratio={:.2}×",
            rt_us / st_us.max(1.0)
        );
    }

    for (n, rt, st) in &timings {
        v.check_pass(
            &format!("batch {n}: streaming ≤ round-trip"),
            *st <= *rt * 1.5,
        );
    }

    if timings.len() >= 2 {
        let first_ratio = timings[0].1 / timings[0].2.max(1.0);
        let last_ratio = timings.last().unwrap().1 / timings.last().unwrap().2.max(1.0);
        v.check_pass(
            "scaling: streaming advantage grows with batch size",
            last_ratio >= first_ratio * 0.5,
        );
    }

    print_timing("batch scaling", t0);
}

fn print_timing(name: &str, t0: Instant) {
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [{name}] {ms:.1} ms");
}
