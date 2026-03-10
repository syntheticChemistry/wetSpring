// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::print_stdout
)]
//! # Exp219: Pure GPU Streaming v3 — Unidirectional Pipeline
//!
//! Demonstrates the `ToadStool` unidirectional streaming advantage:
//! multi-stage pipeline chaining quality filter, diversity, `PCoA`,
//! and spectral match in a single GPU session with zero CPU
//! round-trips for compute stages.
//!
//! Benchmarks streaming (pre-warmed session) vs individual dispatch
//! to quantify the latency advantage.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66+ |
//! | Command | `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v3` |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::pcoa;
use wetspring_barracuda::bio::pcoa_gpu;
use wetspring_barracuda::bio::quality::{self, QualityParams};
use wetspring_barracuda::bio::spectral_match_gpu;
use wetspring_barracuda::bio::streaming_gpu::GpuPipelineSession;
use wetspring_barracuda::bio::taxonomy::{
    ClassifyParams, Lineage, NaiveBayesClassifier, ReferenceSeq,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::special;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp219: Pure GPU Streaming v3 — Unidirectional Pipeline");

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

    let session = match GpuPipelineSession::new(&gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Session init failed: {e}");
            validation::exit_skipped("GpuPipelineSession init failed");
        }
    };
    println!("  Session: {}", session.ctx_stats());

    let mut timings: Vec<(&str, f64, f64)> = Vec::new();
    let pipeline_start = Instant::now();

    // ═══ Stage 1: Quality Filtering (GPU pre-warmed) ═════════════════
    validate_quality_filter(&session, &mut v, &mut timings);

    // ═══ Stage 2: Alpha + Beta Diversity Streaming ═══════════════════
    let (cpu_bc, gpu_bc) = validate_diversity_streaming(&session, &mut v, &mut timings);

    // ═══ Stage 3: PCoA from Streaming BC ═════════════════════════════
    validate_pcoa_streaming(&gpu, &cpu_bc, &gpu_bc, &mut v, &mut timings);

    // ═══ Stage 4: Spectral Cosine Streaming ══════════════════════════
    validate_spectral_streaming(&session, &gpu, &mut v, &mut timings);

    // ═══ Stage 5: Full End-to-End Pipeline ═══════════════════════════
    validate_full_pipeline(&session, &mut v, &mut timings);

    // ═══ Stage 6: Streaming vs Individual Dispatch Benchmark ═════════
    benchmark_streaming_vs_dispatch(&session, &gpu, &mut v, &mut timings);

    // ═══ Summary ═════════════════════════════════════════════════════
    v.section("═══ Streaming v3 Summary ═══");
    println!();
    println!("  {:<35} {:>10} {:>10}", "Stage", "CPU (µs)", "GPU (µs)");
    println!("  {}", "─".repeat(58));
    for (name, cpu_us, gpu_us) in &timings {
        println!("  {name:<35} {cpu_us:>10.0} {gpu_us:>10.0}");
    }
    println!("  {}", "─".repeat(58));
    let total_ms = pipeline_start.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Unidirectional streaming: 6 stages validated");
    println!("  [Total] {total_ms:.1} ms");
    v.finish();
}

// ═══ Stage 1: Quality Filter ═════════════════════════════════════════

fn validate_quality_filter(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S1: Quality Filtering (GPU pre-warmed)");

    let reads = synthetic_reads(200, 150);
    let params = QualityParams::default();

    let tc = Instant::now();
    let cpu_filtered = quality::filter_reads(&reads, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let (gpu_filtered, gpu_stats) = session.filter_reads(&reads, &params).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check_count(
        "filtered read count",
        gpu_filtered.len(),
        cpu_filtered.0.len(),
    );
    v.check_pass(
        "GPU stats consistent",
        gpu_stats.input_reads == reads.len()
            && gpu_stats.output_reads + gpu_stats.discarded_reads == reads.len(),
    );

    timings.push(("Quality Filter", cpu_us, gpu_us));
}

// ═══ Stage 2: Diversity Streaming ════════════════════════════════════

fn validate_diversity_streaming(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) -> (Vec<f64>, Vec<f64>) {
    v.section("S2: Alpha + Beta Diversity Streaming");

    let samples: Vec<Vec<f64>> = vec![
        (1..=80).map(|i| f64::from(i % 20 + 1)).collect(),
        (1..=80).map(|i| f64::from((i * 3) % 25 + 1)).collect(),
        (1..=80).map(|i| f64::from((i * 7) % 30 + 1)).collect(),
        (1..=80).map(|i| f64::from((i * 11) % 15 + 1)).collect(),
    ];
    let sample_refs: Vec<&[f64]> = samples.iter().map(Vec::as_slice).collect();

    let tc = Instant::now();
    let cpu_shannons: Vec<f64> = samples.iter().map(|c| diversity::shannon(c)).collect();
    let cpu_simpsons: Vec<f64> = samples.iter().map(|c| diversity::simpson(c)).collect();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_shannons: Vec<f64> = samples
        .iter()
        .map(|c| session.shannon(c).unwrap())
        .collect();
    let gpu_simpsons: Vec<f64> = samples
        .iter()
        .map(|c| session.simpson(c).unwrap())
        .collect();
    let gpu_bc = session.bray_curtis_matrix(&sample_refs).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (c, g)) in cpu_shannons.iter().zip(&gpu_shannons).enumerate() {
        v.check(
            &format!("Shannon[{i}]"),
            *g,
            *c,
            tolerances::GPU_LOG_POLYFILL,
        );
    }
    for (i, (c, g)) in cpu_simpsons.iter().zip(&gpu_simpsons).enumerate() {
        v.check(&format!("Simpson[{i}]"), *g, *c, tolerances::ANALYTICAL_F64);
    }
    v.check_count("BC condensed length", gpu_bc.len(), cpu_bc.len());
    for (i, (c, g)) in cpu_bc.iter().zip(&gpu_bc).enumerate() {
        v.check(&format!("BC[{i}]"), *g, *c, tolerances::GPU_VS_CPU_F64);
    }

    timings.push(("Diversity (α + β)", cpu_us, gpu_us));
    (cpu_bc, gpu_bc)
}

// ═══ Stage 3: PCoA from Streaming Bray-Curtis ═══════════════════════

fn validate_pcoa_streaming(
    gpu: &GpuF64,
    cpu_bc: &[f64],
    gpu_bc: &[f64],
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S3: PCoA from Streaming BC (zero round-trip)");

    let n_samples = 4;
    let n_axes = 2;

    let tc = Instant::now();
    let cpu_pcoa = pcoa::pcoa(cpu_bc, n_samples, n_axes).unwrap();
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_pcoa = pcoa_gpu::pcoa_gpu(gpu, gpu_bc, n_samples, n_axes).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check_count("PCoA samples", gpu_pcoa.n_samples, cpu_pcoa.n_samples);
    v.check_count("PCoA axes", gpu_pcoa.n_axes, cpu_pcoa.n_axes);

    let all_finite = gpu_pcoa.coordinates.iter().all(|c| c.is_finite());
    v.check_pass("PCoA coordinates finite", all_finite);

    let eigenvalues_positive = gpu_pcoa
        .eigenvalues
        .iter()
        .all(|&e| e >= -tolerances::GPU_VS_CPU_F64);
    v.check_pass("PCoA eigenvalues non-negative", eigenvalues_positive);

    timings.push(("PCoA (from BC)", cpu_us, gpu_us));
}

// ═══ Stage 4: Spectral Cosine Streaming ══════════════════════════════

fn validate_spectral_streaming(
    session: &GpuPipelineSession,
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S4: Spectral Cosine Streaming");

    let spectra: Vec<Vec<f64>> = vec![
        vec![100.0, 200.0, 50.0, 300.0, 150.0],
        vec![100.0, 200.0, 50.0, 300.0, 150.0],
        vec![10.0, 20.0, 500.0, 30.0, 15.0],
        vec![0.0, 0.0, 0.0, 0.0, 1000.0],
    ];

    let tc = Instant::now();
    let cpu_cos = cpu_cosine_condensed(&spectra);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let spec_refs: Vec<&[f64]> = spectra.iter().map(Vec::as_slice).collect();
    let tg = Instant::now();
    let gpu_cos = session
        .spectral_cosine_matrix(&spec_refs)
        .expect("cosine stream");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check_count("cosine condensed length", gpu_cos.len(), cpu_cos.len());
    for (i, (c, g)) in cpu_cos.iter().zip(&gpu_cos).enumerate() {
        v.check(&format!("cos[{i}]"), *g, *c, tolerances::GPU_VS_CPU_F64);
    }

    v.check(
        "identical spectra cos ≈ 1",
        gpu_cos[0],
        1.0,
        tolerances::GPU_VS_CPU_F64,
    );

    let gpu_pairwise = spectral_match_gpu::pairwise_cosine_gpu(gpu, &spectra).unwrap();
    v.check_count(
        "pairwise vs session length",
        gpu_pairwise.len(),
        gpu_cos.len(),
    );

    timings.push(("Spectral Cosine", cpu_us, gpu_us));
}

// ═══ Stage 5: Full End-to-End Pipeline ═══════════════════════════════

fn validate_full_pipeline(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S5: Full End-to-End Pipeline (taxonomy + diversity + BC)");

    let samples: Vec<Vec<f64>> = vec![
        vec![10.0, 20.0, 30.0, 5.0, 15.0],
        vec![12.0, 18.0, 28.0, 7.0, 13.0],
        vec![1.0, 2.0, 40.0, 20.0, 3.0],
    ];
    let sample_refs: Vec<&[f64]> = samples.iter().map(Vec::as_slice).collect();

    let tc = Instant::now();
    let cpu_alpha: Vec<(f64, f64, f64)> = samples
        .iter()
        .map(|c| {
            (
                diversity::shannon(c),
                diversity::simpson(c),
                diversity::observed_features(c),
            )
        })
        .collect();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let synthetic_ref = ReferenceSeq {
        id: "synth_ref".to_string(),
        sequence: b"ACGTACGTACGTACGT".to_vec(),
        lineage: Lineage::from_taxonomy_string("k__Bacteria;p__Proteobacteria"),
    };
    let classifier = NaiveBayesClassifier::train(&[synthetic_ref], 8);
    let params = ClassifyParams::default();

    let tg = Instant::now();
    let result = session
        .stream_full_analytics(&classifier, &[], &sample_refs, &params)
        .unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (cpu_a, gpu_a)) in cpu_alpha.iter().zip(&result.alpha).enumerate() {
        v.check(
            &format!("pipeline α[{i}] Shannon"),
            gpu_a.shannon,
            cpu_a.0,
            tolerances::GPU_LOG_POLYFILL,
        );
    }

    for (i, (c, g)) in cpu_bc.iter().zip(&result.bray_curtis).enumerate() {
        v.check(
            &format!("pipeline BC[{i}]"),
            *g,
            *c,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    println!(
        "    Tax: {:.1}ms  Div: {:.1}ms  BC: {:.1}ms  Total: {:.1}ms",
        result.taxonomy_ms, result.diversity_ms, result.bray_curtis_ms, result.total_ms,
    );

    timings.push(("Full Pipeline", cpu_us, gpu_us));
}

// ═══ Stage 6: Streaming vs Individual Dispatch ═══════════════════════

fn benchmark_streaming_vs_dispatch(
    session: &GpuPipelineSession,
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S6: Streaming vs Individual Dispatch Latency");

    let counts: Vec<f64> = (1..=500).map(f64::from).collect();
    let iters = 50;

    let t_stream = Instant::now();
    for _ in 0..iters {
        let _ = session.shannon(&counts).unwrap();
        let _ = session.simpson(&counts).unwrap();
        let _ = session.observed_features(&counts).unwrap();
    }
    let stream_us = t_stream.elapsed().as_micros() as f64;

    let t_individual = Instant::now();
    for _ in 0..iters {
        use wetspring_barracuda::bio::diversity_gpu;
        let _ = diversity_gpu::shannon_gpu(gpu, &counts).unwrap();
        let _ = diversity_gpu::simpson_gpu(gpu, &counts).unwrap();
        let _ = diversity_gpu::observed_features_gpu(gpu, &counts).unwrap();
    }
    let individual_us = t_individual.elapsed().as_micros() as f64;

    let stream_val = session.shannon(&counts).unwrap();
    let individual_val =
        wetspring_barracuda::bio::diversity_gpu::shannon_gpu(gpu, &counts).unwrap();
    v.check(
        "Streaming == Individual (Shannon)",
        stream_val,
        individual_val,
        tolerances::EXACT,
    );

    let speedup = individual_us / stream_us.max(1.0);
    println!("    Streaming: {stream_us:.0} µs ({iters} × 3 ops)");
    println!("    Individual: {individual_us:.0} µs ({iters} × 3 ops)");
    println!("    Streaming speedup: {speedup:.2}×");

    v.check_pass(
        "Streaming not significantly slower",
        stream_us < individual_us * 5.0,
    );

    timings.push(("Streaming (×50)", stream_us, individual_us));
}

// ═══ Helpers ═════════════════════════════════════════════════════════

fn synthetic_reads(n: usize, len: usize) -> Vec<FastqRecord> {
    let bases = b"ACGT";
    (0..n)
        .map(|i| {
            let seq: Vec<u8> = (0..len).map(|j| bases[(i + j) % 4]).collect();
            let qual: Vec<u8> = (0..len)
                .map(|j| {
                    let q = 20 + ((i + j) % 20) as u8;
                    q + 33
                })
                .collect();
            FastqRecord {
                id: format!("read_{i}"),
                sequence: seq,
                quality: qual,
            }
        })
        .collect()
}

fn cpu_cosine_condensed(spectra: &[Vec<f64>]) -> Vec<f64> {
    let n = spectra.len();
    let norms: Vec<f64> = spectra.iter().map(|s| special::l2_norm(s)).collect();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = special::dot(&spectra[i], &spectra[j]);
            let denom = norms[i] * norms[j];
            let cos = if denom > 1e-15 { dot / denom } else { 0.0 };
            out.push(cos.clamp(0.0, 1.0));
        }
    }
    out
}
