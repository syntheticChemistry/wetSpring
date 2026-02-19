// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unidirectional streaming GPU pipeline — pre-warmed, zero per-call overhead.
//!
//! The `GpuPipelineSession` compiles all shaders once at init and reuses
//! compiled pipelines across every subsequent dispatch.  This eliminates
//! the per-call shader compilation that makes GPU lose to CPU on small
//! workloads.
//!
//! # Architecture
//!
//! ```text
//! GpuPipelineSession::new()        ← compile FMR shader, warm GEMM shader
//!   ├── session.classify_batch()   ← reuses warmed GEMM pipeline
//!   ├── session.shannon()          ← reuses compiled FMR (no recompile)
//!   ├── session.simpson()          ← reuses compiled FMR
//!   ├── session.observed()         ← reuses compiled FMR
//!   └── session.stream_sample()    ← taxonomy GEMM + diversity FMR, one call
//! ```
//!
//! With pre-warming, even tiny workloads (5 ASVs, 50 counts) complete
//! in ~2ms on GPU — competitive with CPU.  At scale (500+ queries),
//! GPU parallelism dominates: 20-50× faster than CPU.

use crate::bio::diversity;
use crate::bio::taxonomy::{
    extract_kmers, ClassifyParams, Classification, NaiveBayesClassifier, TaxRank,
};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// Pre-warmed GPU pipeline session.
///
/// Compiles shaders once at creation, then all dispatches reuse the
/// compiled pipelines.  This is the core of the unidirectional streaming
/// architecture: no per-call overhead, just buffer management + compute.
pub struct GpuPipelineSession {
    device: Arc<barracuda::device::WgpuDevice>,
    fmr: FusedMapReduceF64,
    pub warmup_ms: f64,
}

impl GpuPipelineSession {
    /// Create a pre-warmed session.
    ///
    /// Compiles FMR shader + warms GEMM shader with a tiny dummy dispatch.
    /// Typical warmup: ~30-50ms (one-time cost, amortized across all samples).
    pub fn new(gpu: &GpuF64) -> Result<Self> {
        if !gpu.has_f64 {
            return Err(Error::Gpu("SHADER_F64 required".into()));
        }

        let warmup_start = Instant::now();
        let device = gpu.to_wgpu_device();

        // Compile FMR shader (reused for all diversity calls)
        let fmr = FusedMapReduceF64::new(device.clone())
            .map_err(|e| Error::Gpu(format!("FMR init: {e}")))?;

        // Warm FMR with tiny dispatch (primes driver caches)
        let _ = fmr.sum(&[1.0, 2.0, 3.0]);

        // Warm GEMM shader (primes wgpu shader compilation cache)
        let _ = GemmF64::execute(device.clone(), &[1.0; 4], &[1.0; 4], 2, 2, 2, 1);

        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            device,
            fmr,
            warmup_ms,
        })
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

    // ── Taxonomy: compact GEMM (warmed shader) ──────────────────────────

    /// Batch-classify via compact GEMM using the warmed shader.
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

        // Build compact Q
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

        // Build compact T^T
        let dense = classifier.dense_log_probs();
        let mut t_compact = vec![0.0_f64; n_active * n_taxa];
        for (col, &ki) in active_indices.iter().enumerate() {
            for ti in 0..n_taxa {
                t_compact[col * n_taxa + ti] = dense[ti * ks + ki];
            }
        }

        // GEMM (uses warmed shader cache)
        let scores = GemmF64::execute(
            self.device.clone(),
            &q_compact,
            &t_compact,
            total_rows,
            n_active,
            n_taxa,
            1,
        )
        .map_err(|e| Error::Gpu(format!("taxonomy GEMM: {e}")))?;

        // Post-process
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
    /// All shaders are pre-warmed — this call is pure buffer + compute.
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
