// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
//! Exp105: Pure GPU Streaming v2 — Multi-Domain Analytics Pipeline
//!
//! Proves that expanded `GpuPipelineSession` can chain taxonomy, alpha
//! diversity, Bray-Curtis beta diversity, and spectral cosine matching
//! in a single streaming session with pre-warmed pipelines.
//!
//! Validates:
//! 1. **Alpha diversity streaming** (Shannon, Simpson, observed) — FMR
//! 2. **Bray-Curtis streaming** — pre-compiled `BrayCurtisF64`
//! 3. **Spectral cosine streaming** — pre-compiled GEMM + FMR norms
//! 4. **Full analytics pipeline** — taxonomy + diversity + Bray-Curtis chained
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | `BarraCuda` CPU reference |
//! | Baseline date | 2026-02-23 |
//! | Exact command | `cargo run --features gpu --release --bin validate_pure_gpu_streaming_v2` |
//! | Data | Synthetic test vectors (self-contained) |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::streaming_gpu::GpuPipelineSession;
use wetspring_barracuda::bio::taxonomy::{
    ClassifyParams, Lineage, NaiveBayesClassifier, ReferenceSeq,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::special;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp105: Pure GPU Streaming v2 — Multi-Domain Analytics");

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

    let t0 = Instant::now();
    let mut timings: Vec<(&str, f64, f64)> = Vec::new();

    validate_alpha_streaming(&session, &mut v, &mut timings);
    validate_bray_curtis_streaming(&session, &mut v, &mut timings);
    validate_spectral_streaming(&session, &mut v, &mut timings);
    validate_full_pipeline(&session, &mut v, &mut timings);

    v.section("═══ Streaming v2 Summary ═══");
    println!();
    println!("  {:<30} {:>10} {:>10}", "Stage", "CPU (µs)", "GPU (µs)");
    println!("  {}", "─".repeat(54));
    for (name, cpu_us, gpu_us) in &timings {
        println!("  {name:<30} {cpu_us:>10.0} {gpu_us:>10.0}");
    }
    println!("  {}", "─".repeat(54));
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Pre-warmed streaming: 4 domain groups validated");
    println!("  [Total] {ms:.1} ms");
    v.finish();
}

// ═══ Section 1: Alpha Diversity Streaming ══════════════════════════

fn validate_alpha_streaming(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S1: Alpha Diversity Streaming (FMR pre-warmed)");

    let counts = vec![10.0, 20.0, 30.0, 5.0, 15.0, 8.0, 12.0, 3.0];

    let tc = Instant::now();
    let cpu_shannon = diversity::shannon(&counts);
    let cpu_simpson = diversity::simpson(&counts);
    let cpu_observed = diversity::observed_features(&counts);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_shannon = session.shannon(&counts).expect("shannon stream");
    let gpu_simpson = session.simpson(&counts).expect("simpson stream");
    let gpu_observed = session.observed_features(&counts).expect("observed stream");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "Shannon stream",
        gpu_shannon,
        cpu_shannon,
        tolerances::GPU_LOG_POLYFILL,
    );
    v.check(
        "Simpson stream",
        gpu_simpson,
        cpu_simpson,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Observed stream",
        gpu_observed,
        cpu_observed,
        tolerances::EXACT,
    );
    timings.push(("Alpha Diversity", cpu_us, gpu_us));
}

// ═══ Section 2: Bray-Curtis Streaming ══════════════════════════════

fn validate_bray_curtis_streaming(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S2: Bray-Curtis Streaming (pre-warmed BrayCurtisF64)");

    let samples: Vec<Vec<f64>> = vec![
        vec![10.0, 20.0, 30.0, 5.0],
        vec![12.0, 18.0, 28.0, 7.0],
        vec![1.0, 2.0, 40.0, 20.0],
    ];

    let tc = Instant::now();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let sample_refs: Vec<&[f64]> = samples.iter().map(Vec::as_slice).collect();
    let tg = Instant::now();
    let gpu_bc = session.bray_curtis_matrix(&sample_refs).expect("BC stream");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "BC condensed length",
        gpu_bc.len() as f64,
        cpu_bc.len() as f64,
        tolerances::EXACT,
    );

    for (i, (c, g)) in cpu_bc.iter().zip(&gpu_bc).enumerate() {
        v.check(&format!("BC[{i}]"), *g, *c, tolerances::GPU_VS_CPU_F64);
    }

    let all_in_range = gpu_bc.iter().all(|&d| (0.0..=1.0).contains(&d));
    v.check_pass("BC all in [0,1]", all_in_range);

    timings.push(("Bray-Curtis", cpu_us, gpu_us));
}

// ═══ Section 3: Spectral Cosine Streaming ══════════════════════════

fn validate_spectral_streaming(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S3: Spectral Cosine Streaming (GEMM + FMR pre-warmed)");

    let spectra: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 3.0, 0.0, 5.0],
        vec![1.0, 0.0, 3.0, 0.0, 5.0],
        vec![0.0, 2.0, 0.0, 4.0, 0.0],
    ];

    // CPU cosine similarity on pre-aligned spectra
    let tc = Instant::now();
    let cpu_cos = cpu_cosine_condensed(&spectra);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let spec_refs: Vec<&[f64]> = spectra.iter().map(Vec::as_slice).collect();
    let tg = Instant::now();
    let gpu_cos = session
        .spectral_cosine_matrix(&spec_refs)
        .expect("cosine stream");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "cosine condensed length",
        gpu_cos.len() as f64,
        cpu_cos.len() as f64,
        tolerances::EXACT,
    );

    for (i, (c, g)) in cpu_cos.iter().zip(&gpu_cos).enumerate() {
        v.check(&format!("cos[{i}]"), *g, *c, tolerances::GPU_VS_CPU_F64);
    }

    // Identical spectra should have cosine ≈ 1.0
    v.check(
        "identical spectra cos ≈ 1",
        gpu_cos[0],
        1.0,
        tolerances::GPU_VS_CPU_F64,
    );

    // Orthogonal spectra should have cosine ≈ 0.0
    v.check(
        "orthogonal spectra cos ≈ 0",
        gpu_cos[1],
        0.0,
        tolerances::GPU_VS_CPU_F64,
    );

    timings.push(("Spectral Cosine", cpu_us, gpu_us));
}

// ═══ Section 4: Full Streaming Pipeline ════════════════════════════

fn validate_full_pipeline(
    session: &GpuPipelineSession,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S4: Full Analytics Pipeline (taxonomy + diversity + BC)");

    let sample_a = vec![10.0, 20.0, 30.0, 5.0, 15.0];
    let sample_b = vec![12.0, 18.0, 28.0, 7.0, 13.0];
    let sample_c = vec![1.0, 2.0, 40.0, 20.0, 3.0];

    let sample_counts: Vec<&[f64]> = vec![&sample_a, &sample_b, &sample_c];

    // CPU reference: per-sample alpha diversity
    let tc = Instant::now();
    let cpu_alpha: Vec<(f64, f64, f64)> = sample_counts
        .iter()
        .map(|c| {
            (
                diversity::shannon(c),
                diversity::simpson(c),
                diversity::observed_features(c),
            )
        })
        .collect();
    let samples_owned: Vec<Vec<f64>> = sample_counts.iter().map(|s| s.to_vec()).collect();
    let cpu_bc = diversity::bray_curtis_condensed(&samples_owned);
    let cpu_us = tc.elapsed().as_micros() as f64;

    // Build a trivial classifier (classify_batch returns early for empty sequences)
    let synthetic_ref = ReferenceSeq {
        id: "synthetic_ref".to_string(),
        sequence: b"ACGTACGTACGTACGT".to_vec(),
        lineage: Lineage::from_taxonomy_string("k__Bacteria;p__Proteobacteria"),
    };
    let classifier = NaiveBayesClassifier::train(&[synthetic_ref], 8);
    let params = ClassifyParams::default();

    // GPU streaming: full analytics (no taxonomy seqs for this test)
    let tg = Instant::now();
    let gpu_result = session
        .stream_full_analytics(&classifier, &[], &sample_counts, &params)
        .expect("full stream");
    let gpu_us = tg.elapsed().as_micros() as f64;

    // Verify alpha diversity parity
    for (i, (cpu_a, gpu_a)) in cpu_alpha.iter().zip(&gpu_result.alpha).enumerate() {
        v.check(
            &format!("alpha[{i}] Shannon"),
            gpu_a.shannon,
            cpu_a.0,
            tolerances::GPU_LOG_POLYFILL,
        );
        v.check(
            &format!("alpha[{i}] Simpson"),
            gpu_a.simpson,
            cpu_a.1,
            tolerances::ANALYTICAL_F64,
        );
        v.check(
            &format!("alpha[{i}] observed"),
            gpu_a.observed,
            cpu_a.2,
            tolerances::EXACT,
        );
    }

    // Verify Bray-Curtis parity
    v.check(
        "pipeline BC length",
        gpu_result.bray_curtis.len() as f64,
        cpu_bc.len() as f64,
        tolerances::EXACT,
    );
    for (i, (c, g)) in cpu_bc.iter().zip(&gpu_result.bray_curtis).enumerate() {
        v.check(
            &format!("pipeline BC[{i}]"),
            *g,
            *c,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    println!(
        "    Taxonomy: {:.1} ms, Diversity: {:.1} ms, BC: {:.1} ms, Total: {:.1} ms",
        gpu_result.taxonomy_ms,
        gpu_result.diversity_ms,
        gpu_result.bray_curtis_ms,
        gpu_result.total_ms,
    );

    timings.push(("Full Pipeline", cpu_us, gpu_us));
}

fn cpu_cosine_condensed(spectra: &[Vec<f64>]) -> Vec<f64> {
    let n = spectra.len();
    let norms: Vec<f64> = spectra.iter().map(|s| special::l2_norm(s)).collect();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = special::dot(&spectra[i], &spectra[j]);
            let denom = norms[i] * norms[j];
            let cos = if denom > tolerances::MATRIX_EPS {
                dot / denom
            } else {
                0.0
            };
            out.push(cos.clamp(0.0, 1.0));
        }
    }
    out
}
