// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unidirectional streaming GPU pipeline — full-stage GPU coverage.
//!
//! # Architecture (v3 — full pipeline GPU)
//!
//! ```text
//! GpuPipelineSession::new(gpu)
//!   ├── TensorContext          ← buffer pool + bind group cache + batching
//!   ├── QualityFilterCached    ← pre-compiled QF shader (local extension)
//!   ├── Dada2Gpu              ← pre-compiled DADA2 E-step shader (local extension)
//!   ├── GemmCached             ← pre-compiled GEMM pipeline (local extension)
//!   ├── FusedMapReduceF64      ← pre-compiled FMR pipeline (barraCuda)
//!   └── warmup dispatches      ← prime driver caches
//!
//! session.filter_reads_gpu(reads, params)
//!   └── QualityFilterCached::execute()  ← per-read parallel trimming
//!
//! session.denoise_gpu(uniques, params)
//!   └── Dada2Gpu::batch_log_p_error()  ← E-step on GPU, EM control on CPU
//!
//! session.stream_sample(...)
//!   ├── GemmCached::execute()  ← cached pipeline, per-call buffers only
//!   ├── FMR::shannon()         ← reuses compiled pipeline
//!   ├── FMR::simpson()         ← reuses compiled pipeline
//!   └── FMR::observed()        ← reuses compiled pipeline
//! ```
//!
//! # Pipeline stages on GPU
//!
//! | Stage         | Primitive            | Status          |
//! |---------------|----------------------|-----------------|
//! | Quality filter| `QualityFilterCached`  | GPU (per-read)  |
//! | Dereplication | —                    | CPU (hash)      |
//! | DADA2 denoise | `Dada2Gpu`             | GPU E-step      |
//! | Chimera       | —                    | CPU (k-mer)     |
//! | Taxonomy      | `GemmCached`           | GPU (GEMM)      |
//! | Diversity     | `FusedMapReduceF64`    | GPU/CPU (FMR)   |
//!
//! # Local extensions (for barraCuda absorption)
//!
//! - `QualityFilterCached`: per-read parallel quality trimming WGSL shader
//! - `Dada2Gpu`: batch pair-wise `log_p_error` with precomputed log-err table
//! - `GemmCached`: caches shader + pipeline + BGL at init
//! - `execute_to_buffer()`: returns GPU buffer without readback (chaining)

use crate::bio::dada2::{Asv, Dada2Params, Dada2Stats};
use crate::bio::dada2_gpu::{self, Dada2Gpu};
use crate::bio::derep::UniqueSequence;
use crate::bio::diversity;
use crate::bio::gemm_cached::GemmCached;
use crate::bio::quality::{FilterStats, QualityParams};
use crate::bio::quality_gpu::QualityFilterCached;
use crate::bio::taxonomy::{
    Classification, ClassifyParams, NaiveBayesClassifier, TaxRank, argmax_with_priors,
    extract_kmers,
};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;
use barracuda::ops::bray_curtis_f64::BrayCurtisF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// Pre-warmed GPU pipeline session with full-stage coverage.
///
/// Holds pre-compiled pipelines for all GPU-accelerated stages:
/// quality filter, DADA2 E-step, taxonomy GEMM, and diversity FMR.
/// Quality and DADA2 now delegate to barraCuda absorbed primitives.
pub struct GpuPipelineSession {
    qf: QualityFilterCached,
    dada2: Dada2Gpu,
    fmr: FusedMapReduceF64,
    gemm: GemmCached,
    bc: BrayCurtisF64,
    /// GPU warmup time in milliseconds.
    pub warmup_ms: f64,
}

impl GpuPipelineSession {
    /// Create a pre-warmed session with all GPU pipelines compiled.
    ///
    /// # Errors
    ///
    /// Returns an error if the device lacks `SHADER_F64` or pipeline compilation fails.
    pub fn new(gpu: &GpuF64) -> Result<Self> {
        if !gpu.has_f64 {
            return Err(Error::Gpu("SHADER_F64 required".into()));
        }

        let warmup_start = Instant::now();
        let device = gpu.to_wgpu_device();
        // Arc bump, O(1)
        let ctx = Arc::clone(gpu.tensor_context());

        let qf = QualityFilterCached::new(Arc::clone(&device))?;
        let dada2 = Dada2Gpu::new(Arc::clone(&device))?;
        let fmr = FusedMapReduceF64::new(Arc::clone(&device))
            .map_err(|e| Error::Gpu(format!("FMR init: {e}")))?;
        let gemm = GemmCached::new(Arc::clone(&device), ctx);
        let bc = BrayCurtisF64::new(device)
            .map_err(|e| Error::Gpu(format!("BrayCurtisF64 init: {e}")))?;

        // Prime driver caches with tiny dispatches
        let _ = fmr.sum(&[1.0, 2.0, 3.0]);
        let _ = gemm.execute(&[1.0; 4], &[1.0; 4], 2, 2, 2, 1);

        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            qf,
            dada2,
            fmr,
            gemm,
            bc,
            warmup_ms,
        })
    }

    /// Pipeline session info string.
    #[must_use]
    pub fn ctx_stats(&self) -> String {
        format!("warmup={:.1}ms", self.warmup_ms)
    }

    // ── Quality filter: real GPU dispatch ────────────────────────────────

    /// GPU-accelerated quality filtering — real per-read parallel trimming.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    pub fn filter_reads(
        &self,
        reads: &[FastqRecord],
        params: &QualityParams,
    ) -> Result<(Vec<FastqRecord>, FilterStats)> {
        let trim_results = self.qf.execute(reads, params)?;
        let mut output = Vec::with_capacity(reads.len());
        let mut stats = FilterStats {
            input_reads: reads.len(),
            output_reads: 0,
            discarded_reads: 0,
            leading_bases_trimmed: 0,
            trailing_bases_trimmed: 0,
            window_bases_trimmed: 0,
            adapter_bases_trimmed: 0,
        };

        for (record, trim) in reads.iter().zip(trim_results.iter()) {
            if let Some((start, end)) = trim {
                stats.leading_bases_trimmed += *start as u64;
                stats.trailing_bases_trimmed += (record.quality.len() - end) as u64;
                output.push(crate::bio::quality::apply_trim(record, *start, *end));
                stats.output_reads += 1;
            } else {
                stats.discarded_reads += 1;
            }
        }

        Ok((output, stats))
    }

    // ── DADA2: GPU E-step ────────────────────────────────────────────────

    /// GPU-accelerated DADA2 denoising — E-step on GPU, EM control on CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails or the device lacks f64 support.
    pub fn denoise(
        &self,
        seqs: &[UniqueSequence],
        params: &Dada2Params,
    ) -> Result<(Vec<Asv>, Dada2Stats)> {
        dada2_gpu::denoise_gpu(&self.dada2, seqs, params)
    }

    // ── Diversity: pre-warmed FMR (no shader recompilation) ─────────────

    /// Shannon entropy via pre-warmed FMR.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn shannon(&self, counts: &[f64]) -> Result<f64> {
        if counts.is_empty() {
            return Ok(0.0);
        }
        self.fmr
            .shannon_entropy(counts)
            .map_err(|e| Error::Gpu(format!("Shannon: {e}")))
    }

    /// Simpson diversity (1 - dominance) via pre-warmed FMR.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails or GEMM readback fails.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
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
        let query_kmer_lists: Vec<Vec<u64>> =
            sequences.iter().map(|seq| extract_kmers(seq, k)).collect();

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
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
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
                // ownership transfer: Classification owns lineage, classifier retains taxon_labels
                lineage: taxon_labels[best_taxon].clone(),
                confidence,
                taxon_idx: best_taxon,
            });
        }

        Ok(results)
    }

    // ── Bray-Curtis: pre-warmed pipeline ───────────────────────────────

    /// All-pairs Bray-Curtis distance matrix via pre-warmed pipeline.
    ///
    /// Returns condensed upper-triangle: `n*(n-1)/2` distances.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    pub fn bray_curtis_matrix(&self, samples: &[&[f64]]) -> Result<Vec<f64>> {
        let n = samples.len();
        if n < 2 {
            return Ok(vec![]);
        }
        let d = samples[0].len();
        let flat: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();
        self.bc
            .condensed_distance_matrix(&flat, n, d)
            .map_err(|e| Error::Gpu(format!("Bray-Curtis stream: {e}")))
    }

    // ── Spectral cosine: GEMM + FMR norms ───────────────────────────────

    /// Pairwise cosine similarity matrix via pre-warmed GEMM + FMR.
    ///
    /// Returns condensed upper-triangle: `n*(n-1)/2` similarities.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn spectral_cosine_matrix(&self, spectra: &[&[f64]]) -> Result<Vec<f64>> {
        let n = spectra.len();
        if n < 2 {
            return Ok(vec![]);
        }
        let d = spectra[0].len();
        let flat: Vec<f64> = spectra.iter().flat_map(|s| s.iter().copied()).collect();

        // Transpose for GEMM: A[N×D] × A^T[D×N] = dot_products[N×N]
        let mut flat_t = vec![0.0_f64; d * n];
        for i in 0..n {
            for j in 0..d {
                flat_t[j * n + i] = flat[i * d + j];
            }
        }

        let dot_matrix = self
            .gemm
            .execute(&flat, &flat_t, n, d, n, 1)
            .map_err(|e| Error::Gpu(format!("spectral GEMM: {e}")))?;

        // Norms via FMR sum_of_squares (pre-warmed)
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let sq_sum = self
                .fmr
                .sum_of_squares(&flat[i * d..(i + 1) * d])
                .map_err(|e| Error::Gpu(format!("norm FMR: {e}")))?;
            norms.push(sq_sum.sqrt());
        }

        // Condensed cosine similarity
        let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let denom = norms[i] * norms[j];
                let cos = if denom > crate::tolerances::MATRIX_EPS {
                    dot_matrix[i * n + j] / denom
                } else {
                    0.0
                };
                condensed.push(cos.clamp(0.0, 1.0));
            }
        }
        Ok(condensed)
    }

    // ── Streaming session: taxonomy + diversity in one call ──────────────

    /// Run taxonomy + diversity on GPU in a single streaming session.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails for taxonomy or diversity.
    #[expect(clippy::cast_precision_loss)]
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

    // ── Extended streaming: full analytics pipeline ────────────────────

    /// Run taxonomy + diversity + Bray-Curtis in a single streaming session.
    ///
    /// Chains: GEMM taxonomy → FMR diversity → `BrayCurtisF64` beta diversity.
    /// All use pre-warmed pipelines — zero shader recompilation.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails for any stage.
    pub fn stream_full_analytics(
        &self,
        classifier: &NaiveBayesClassifier,
        sequences: &[&[u8]],
        sample_counts: &[&[f64]],
        params: &ClassifyParams,
    ) -> Result<FullStreamingResult> {
        let session_start = Instant::now();

        // Stage 1: taxonomy
        let tax_start = Instant::now();
        let classifications = self.classify_batch(classifier, sequences, params)?;
        let taxonomy_ms = tax_start.elapsed().as_secs_f64() * 1000.0;

        // Stage 2: per-sample alpha diversity
        let div_start = Instant::now();
        let mut alpha = Vec::with_capacity(sample_counts.len());
        for counts in sample_counts {
            let shannon = if counts.len() < 2 {
                0.0
            } else {
                self.shannon(counts)?
            };
            let simpson = if counts.len() < 2 {
                0.0
            } else {
                self.simpson(counts)?
            };
            let observed = if counts.is_empty() {
                0.0
            } else {
                self.observed_features(counts)?
            };
            alpha.push(AlphaDiversity {
                shannon,
                simpson,
                observed,
            });
        }
        let diversity_ms = div_start.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: Bray-Curtis beta diversity (if ≥ 2 samples)
        let bc_start = Instant::now();
        let bray_curtis = if sample_counts.len() >= 2 {
            self.bray_curtis_matrix(sample_counts)?
        } else {
            vec![]
        };
        let bray_curtis_ms = bc_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = session_start.elapsed().as_secs_f64() * 1000.0;

        Ok(FullStreamingResult {
            classifications,
            alpha,
            bray_curtis,
            taxonomy_ms,
            diversity_ms,
            bray_curtis_ms,
            total_ms,
        })
    }
}

/// Per-sample alpha diversity metrics.
#[derive(Debug)]
pub struct AlphaDiversity {
    /// Shannon entropy.
    pub shannon: f64,
    /// Simpson diversity (1 - dominance).
    pub simpson: f64,
    /// Observed species count.
    pub observed: f64,
}

/// Results from the full streaming analytics pipeline.
#[derive(Debug)]
pub struct FullStreamingResult {
    /// Per-read taxonomy classifications.
    pub classifications: Vec<Classification>,
    /// Per-sample alpha diversity.
    pub alpha: Vec<AlphaDiversity>,
    /// Condensed Bray-Curtis distance matrix.
    pub bray_curtis: Vec<f64>,
    /// Taxonomy stage time in milliseconds.
    pub taxonomy_ms: f64,
    /// Diversity stage time in milliseconds.
    pub diversity_ms: f64,
    /// Bray-Curtis stage time in milliseconds.
    pub bray_curtis_ms: f64,
    /// Total pipeline time in milliseconds.
    pub total_ms: f64,
}

/// Results from the streaming GPU pipeline.
#[derive(Debug)]
pub struct StreamingGpuResult {
    /// Per-read taxonomy classifications.
    pub classifications: Vec<Classification>,
    /// Shannon diversity index.
    pub shannon: f64,
    /// Simpson diversity index.
    pub simpson: f64,
    /// Observed species count.
    pub observed: f64,
    /// Taxonomy classification time in milliseconds.
    pub taxonomy_ms: f64,
    /// Diversity computation time in milliseconds.
    pub diversity_ms: f64,
    /// Total GPU pipeline time in milliseconds.
    pub total_gpu_ms: f64,
}

/// CPU equivalent for benchmarking comparison.
#[must_use]
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

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        fn _assert_session(_: &GpuPipelineSession) {}
        fn _assert_alpha(_: &AlphaDiversity) {}
        fn _assert_streaming_result(_: &StreamingGpuResult) {}
        fn _assert_full_result(_: &FullStreamingResult) {}
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let session = GpuPipelineSession::new(&gpu).expect("GpuPipelineSession::new");
        let counts = vec![1.0, 2.0, 3.0];
        let result = session.shannon(&counts);
        assert!(result.is_ok(), "shannon should succeed with valid input");
    }
}
