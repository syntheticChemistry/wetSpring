// SPDX-License-Identifier: AGPL-3.0-or-later
//! Individual pipeline stage dispatch methods for `GpuPipelineSession`.
//!
//! Each method corresponds to a single GPU-accelerated stage:
//! quality filter, DADA2 denoise, diversity metrics, taxonomy GEMM,
//! Bray-Curtis beta diversity, and spectral cosine similarity.

use super::GpuPipelineSession;
use crate::bio::dada2::{Asv, Dada2Params, Dada2Stats};
use crate::bio::dada2_gpu;
use crate::bio::derep::UniqueSequence;
use crate::bio::quality::{FilterStats, QualityParams};
use crate::bio::taxonomy::{
    Classification, ClassifyParams, NaiveBayesClassifier, TaxRank, argmax_with_priors,
    extract_kmers,
};
use crate::error::{Error, Result};
use crate::io::fastq::FastqRecord;
use std::collections::HashSet;

impl GpuPipelineSession {
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

        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let sq_sum = self
                .fmr
                .sum_of_squares(&flat[i * d..(i + 1) * d])
                .map_err(|e| Error::Gpu(format!("norm FMR: {e}")))?;
            norms.push(sq_sum.sqrt());
        }

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
}
