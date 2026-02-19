// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unidirectional streaming GPU pipeline — ToadStool infrastructure wired.
//!
//! # Architecture (v2 — ToadStool wiring)
//!
//! ```text
//! GpuPipelineSession::new(gpu)
//!   ├── TensorContext          ← buffer pool + bind group cache + batching
//!   ├── GemmCached             ← pre-compiled GEMM pipeline (local extension)
//!   ├── FusedMapReduceF64      ← pre-compiled FMR pipeline (ToadStool)
//!   └── warmup dispatches      ← prime driver caches
//!
//! session.stream_sample(...)
//!   ├── GemmCached::execute()  ← cached pipeline, per-call buffers only
//!   ├── FMR::shannon()         ← reuses compiled pipeline
//!   ├── FMR::simpson()         ← reuses compiled pipeline
//!   └── FMR::observed()        ← reuses compiled pipeline
//! ```
//!
//! # ToadStool systems wired
//!
//! - `TensorContext`: buffer pool, bind group cache, batch dispatch grouping
//! - `GemmCached`: local extension — pre-compiled GEMM pipeline (ToadStool absorbs)
//! - `FusedMapReduceF64`: pre-compiled at init, reused across all calls
//! - `GLOBAL_CACHE`: pipeline cache (used by TensorContext internally)
//!
//! # Local extensions (for ToadStool absorption)
//!
//! - `GemmCached`: caches shader + pipeline + BGL at init (GemmF64 recreates per call)
//! - `execute_to_buffer()`: returns GPU buffer without readback (chaining primitive)

use crate::bio::diversity;
use crate::bio::gemm_cached::GemmCached;
use crate::bio::taxonomy::{
    extract_kmers, ClassifyParams, Classification, NaiveBayesClassifier, TaxRank,
};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::device::TensorContext;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// Pre-warmed GPU pipeline session with ToadStool infrastructure.
///
/// Wires TensorContext (pool + cache + batching) and GemmCached (pre-compiled
/// pipeline) into the streaming architecture. All per-sample dispatches reuse
/// compiled pipelines — only buffer allocation and data transfer per call.
pub struct GpuPipelineSession {
    ctx: Arc<TensorContext>,
    fmr: FusedMapReduceF64,
    gemm: GemmCached,
    pub warmup_ms: f64,
}

impl GpuPipelineSession {
    /// Create a pre-warmed session with ToadStool infrastructure.
    ///
    /// - TensorContext: buffer pool, bind group cache
    /// - GemmCached: pre-compiled GEMM shader + pipeline
    /// - FMR: pre-compiled map-reduce shader + pipeline
    /// - Warmup dispatches: prime driver caches
    pub fn new(gpu: &GpuF64) -> Result<Self> {
        if !gpu.has_f64 {
            return Err(Error::Gpu("SHADER_F64 required".into()));
        }

        let warmup_start = Instant::now();
        let device = gpu.to_wgpu_device();
        let ctx = gpu.tensor_context().clone();

        let fmr = FusedMapReduceF64::new(device.clone())
            .map_err(|e| Error::Gpu(format!("FMR init: {e}")))?;

        let gemm = GemmCached::new(device.clone(), ctx.clone());

        // Prime driver caches with tiny dispatches
        let _ = fmr.sum(&[1.0, 2.0, 3.0]);
        let _ = gemm.execute(&[1.0; 4], &[1.0; 4], 2, 2, 2, 1);

        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            ctx,
            fmr,
            gemm,
            warmup_ms,
        })
    }

    /// TensorContext stats: buffer pool reuse, bind group cache, batched ops.
    pub fn ctx_stats(&self) -> String {
        self.ctx.stats().to_string()
    }

    // ── Diversity: pre-warmed FMR (no shader recompilation) ─────────────

    /// Shannon entropy via pre-warmed FMR.
    pub fn shannon(&self, counts: &[f64]) -> Result<f64> {
        if counts.is_empty() {
            return Ok(0.0);
        }
        self.fmr
            .shannon_entropy(counts)
            .map_err(|e| Error::Gpu(format!("Shannon: {e}")))
    }

    /// Simpson diversity (1 - dominance) via pre-warmed FMR.
    pub fn simpson(&self, counts: &[f64]) -> Result<f64> {
        if counts.is_empty() {
            return Ok(0.0);
        }
        let dom = self
            .fmr
            .simpson_index(counts)
            .map_err(|e| Error::Gpu(format!("Simpson: {e}")))?;
        Ok(1.0 - dom)
    }

    /// Observed features via pre-warmed FMR.
    pub fn observed_features(&self, counts: &[f64]) -> Result<f64> {
        if counts.is_empty() {
            return Ok(0.0);
        }
        let binary: Vec<f64> = counts
            .iter()
            .map(|&c| if c > 0.0 { 1.0 } else { 0.0 })
            .collect();
        self.fmr
            .sum(&binary)
            .map_err(|e| Error::Gpu(format!("observed: {e}")))
    }

    // ── Taxonomy: cached GEMM pipeline ──────────────────────────────────

    /// Batch-classify via compact GEMM using the cached pipeline.
    pub fn classify_batch(
        &self,
        classifier: &NaiveBayesClassifier,
        sequences: &[&[u8]],
        params: &ClassifyParams,
    ) -> Result<Vec<Classification>> {
        if sequences.is_empty() || classifier.n_taxa() == 0 {
            return Ok(vec![]);
        }

        let n_queries = sequences.len();
        let n_taxa = classifier.n_taxa();
        let ks = classifier.kmer_space();

        if n_queries * n_taxa < 100 {
            return Ok(sequences
                .iter()
                .map(|seq| classifier.classify(seq, params))
                .collect());
        }

        let k = (ks as f64).log(4.0).round() as usize;
        let query_kmer_lists: Vec<Vec<u64>> = sequences
            .iter()
            .map(|seq| extract_kmers(seq, k))
            .collect();

        let mut active_set = HashSet::new();
        for kmers in &query_kmer_lists {
            for &kmer in kmers {
                active_set.insert(kmer as usize);
            }
        }
        let mut active_indices: Vec<usize> = active_set.into_iter().collect();
        active_indices.sort_unstable();
        let n_active = active_indices.len();

        let mut kmer_to_col = vec![0_usize; ks];
        for (col, &ki) in active_indices.iter().enumerate() {
            kmer_to_col[ki] = col;
        }

        let n_boot = params.bootstrap_n;
        let rows_per_query = 1 + n_boot;
        let total_rows = n_queries * rows_per_query;

        let mut q_compact = vec![0.0_f64; total_rows * n_active];
        for (qi, kmers) in query_kmer_lists.iter().enumerate() {
            let base_row = qi * rows_per_query;
            let row_start = base_row * n_active;
            for &kmer in kmers {
                q_compact[row_start + kmer_to_col[kmer as usize]] += 1.0;
            }
            let n_sample = (kmers.len() * 2 / 3).max(1);
            let mut seed: u64 = 42;
            for bi in 0..n_boot {
                let boot_start = (base_row + 1 + bi) * n_active;
                for _ in 0..n_sample {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let idx = (seed >> 33) as usize % kmers.len();
                    q_compact[boot_start + kmer_to_col[kmers[idx] as usize]] += 1.0;
                }
            }
        }

        let dense = classifier.dense_log_probs();
        let mut t_compact = vec![0.0_f64; n_active * n_taxa];
        for (col, &ki) in active_indices.iter().enumerate() {
            for ti in 0..n_taxa {
                t_compact[col * n_taxa + ti] = dense[ti * ks + ki];
            }
        }

        // GemmCached: reuses pre-compiled pipeline (no shader recompilation)
        let scores = self
            .gemm
            .execute(&q_compact, &t_compact, total_rows, n_active, n_taxa, 1)?;

        let log_priors = classifier.log_priors();
        let taxon_labels = classifier.taxon_labels();
        let mut results = Vec::with_capacity(n_queries);

        for qi in 0..n_queries {
            let base_row = qi * rows_per_query;
            let full_row = &scores[base_row * n_taxa..(base_row + 1) * n_taxa];
            let best_taxon = argmax_with_priors(full_row, log_priors);

            let full_lineage = &taxon_labels[best_taxon];
            let n_ranks = TaxRank::all().len();
            let mut rank_votes = vec![0_usize; n_ranks];

            for bi in 0..n_boot {
                let boot_row = base_row + 1 + bi;
                let boot_scores = &scores[boot_row * n_taxa..(boot_row + 1) * n_taxa];
                let boot_taxon = argmax_with_priors(boot_scores, log_priors);
                let boot_lineage = &taxon_labels[boot_taxon];

                for (ri, rank) in TaxRank::all().iter().enumerate() {
                    let full_at = full_lineage.at_rank(*rank);
                    let boot_at = boot_lineage.at_rank(*rank);
                    if full_at.is_some() && full_at == boot_at {
                        rank_votes[ri] += 1;
                    }
                }
            }

            let confidence: Vec<f64> = rank_votes
                .iter()
                .map(|&v| v as f64 / n_boot as f64)
                .collect();

            results.push(Classification {
                lineage: taxon_labels[best_taxon].clone(),
                confidence,
                taxon_idx: best_taxon,
            });
        }

        Ok(results)
    }

    // ── Streaming session: taxonomy + diversity in one call ──────────────

    /// Run taxonomy + diversity on GPU in a single streaming session.
    ///
    /// All pipelines are pre-compiled — this call is pure buffer + compute.
    pub fn stream_sample(
        &self,
        classifier: &NaiveBayesClassifier,
        sequences: &[&[u8]],
        counts: &[f64],
        params: &ClassifyParams,
    ) -> Result<StreamingGpuResult> {
        let session_start = Instant::now();

        let tax_start = Instant::now();
        let classifications = self.classify_batch(classifier, sequences, params)?;
        let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

        let div_start = Instant::now();
        let (shannon, simpson, observed) = if counts.len() < 2 {
            (0.0, 0.0, counts.len() as f64)
        } else {
            (
                self.shannon(counts)?,
                self.simpson(counts)?,
                self.observed_features(counts)?,
            )
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
}

/// Results from the streaming GPU pipeline.
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

/// CPU equivalent for benchmarking comparison.
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

fn argmax_with_priors(scores: &[f64], log_priors: &[f64]) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, (&s, &lp)) in scores.iter().zip(log_priors.iter()).enumerate() {
        let total = s + lp;
        if total > best {
            best = total;
            best_idx = i;
        }
    }
    best_idx
}
