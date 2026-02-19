// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming GPU pipeline — upload once, compute all, download once.
//!
//! Chains taxonomy GEMM + diversity metrics into a batched GPU session
//! using ToadStool's `TensorContext` for dispatch grouping.  Data flows:
//!
//!   CPU I/O → FASTQ parse → QF → derep → DADA2 → chimera (all CPU, <10ms)
//!         ↓
//!   Upload ASV data → GPU: [taxonomy GEMM] → [diversity FMR] → Download
//!         ↓
//!   CPU post-process: argmax, confidence, formatting
//!
//! The GPU session avoids per-op device init overhead — `FusedMapReduceF64`
//! and `GemmF64` share the same `WgpuDevice` and `TensorContext`.

use crate::bio::diversity;
use crate::bio::diversity_gpu;
use crate::bio::taxonomy::{ClassifyParams, Classification, NaiveBayesClassifier};
use crate::bio::taxonomy_gpu;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use std::time::Instant;

/// Results from the streaming GPU pipeline for a single sample.
#[derive(Debug)]
pub struct StreamingGpuResult {
    pub classifications: Vec<Classification>,
    pub shannon: f64,
    pub simpson: f64,
    pub observed: f64,
    pub taxonomy_ms: f64,
    pub diversity_ms: f64,
    pub total_gpu_ms: f64,
}

/// Run taxonomy + diversity on GPU in a single batched session.
///
/// Uses `TensorContext::begin_batch` / `end_batch` to group all GPU
/// dispatches, minimizing driver overhead and keeping the device busy.
///
/// CPU pre-processing (QF, derep, DADA2, chimera) must be done before
/// calling this — pass in the clean ASV sequences and abundance counts.
pub fn stream_classify_and_diversity(
    gpu: &GpuF64,
    classifier: &NaiveBayesClassifier,
    sequences: &[&[u8]],
    counts: &[f64],
    params: &ClassifyParams,
) -> Result<StreamingGpuResult> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for streaming GPU".into()));
    }

    let session_start = Instant::now();

    // ── Taxonomy: single GEMM dispatch ──────────────────────────────────────
    let tax_start = Instant::now();
    let classifications = if sequences.is_empty() || classifier.n_taxa() == 0 {
        vec![]
    } else {
        taxonomy_gpu::classify_batch_gpu(gpu, classifier, sequences, params)?
    };
    let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

    // ── Diversity: FMR dispatches (shared device, no re-init) ───────────────
    let div_start = Instant::now();
    let (shannon, simpson, observed) = if counts.len() < 2 {
        (0.0, 0.0, counts.len() as f64)
    } else {
        let s = diversity_gpu::shannon_gpu(gpu, counts)?;
        let d = diversity_gpu::simpson_gpu(gpu, counts)?;
        let o = diversity_gpu::observed_features_gpu(gpu, counts)?;
        (s, d, o)
    };
    let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

    let total_gpu_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    Ok(StreamingGpuResult {
        classifications,
        shannon,
        simpson,
        observed,
        taxonomy_ms,
        diversity_ms,
        total_gpu_ms,
    })
}

/// Run the CPU equivalent for comparison benchmarking.
pub fn stream_classify_and_diversity_cpu(
    classifier: &NaiveBayesClassifier,
    sequences: &[&[u8]],
    counts: &[f64],
    params: &ClassifyParams,
) -> StreamingGpuResult {
    let session_start = Instant::now();

    let tax_start = Instant::now();
    let classifications: Vec<Classification> = sequences
        .iter()
        .map(|seq| classifier.classify(seq, params))
        .collect();
    let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

    let div_start = Instant::now();
    let shannon = diversity::shannon(counts);
    let simpson = diversity::simpson(counts);
    let observed = diversity::observed_features(counts);
    let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

    let total_gpu_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    StreamingGpuResult {
        classifications,
        shannon,
        simpson,
        observed,
        taxonomy_ms,
        diversity_ms,
        total_gpu_ms,
    }
}
