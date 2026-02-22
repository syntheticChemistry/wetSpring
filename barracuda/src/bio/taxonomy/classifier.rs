// SPDX-License-Identifier: AGPL-3.0-or-later
//! Naive Bayes taxonomy classifier — training, classification, and NPU quantization.
//!
//! Implements the RDP-style naive Bayes classifier (Wang et al. 2007) used by
//! QIIME2's `feature-classifier classify-sklearn` and DADA2's `assignTaxonomy`.

use std::collections::{HashMap, HashSet};

use super::kmers::extract_kmers;
use super::types::{Classification, ClassifyParams, Lineage, NpuWeights, ReferenceSeq, TaxRank};

/// A trained naive Bayes classifier.
///
/// Stores log-probabilities in a flat `n_taxa` × `kmer_space` array for
/// O(1) lookup (no `HashMap` in the scoring hot path). For `k`=8, `kmer_space`
/// = 4^8 = 65,536 entries per taxon.
#[derive(Debug)]
pub struct NaiveBayesClassifier {
    k: usize,
    dense_log_probs: Vec<f64>,
    kmer_space: usize,
    log_priors: Vec<f64>,
    taxon_labels: Vec<Lineage>,
    taxon_priors: Vec<f64>,
    n_kmers_total: usize,
}

impl NaiveBayesClassifier {
    /// Train a classifier from reference sequences.
    ///
    /// Groups reference sequences by genus-level lineage, extracts k-mers,
    /// and precomputes a flat log-probability table for O(1) scoring.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn train(refs: &[ReferenceSeq], k: usize) -> Self {
        let kmer_space = 1_usize << (2 * k);

        let mut taxon_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in refs.iter().enumerate() {
            let key = r.lineage.to_string_at_rank(TaxRank::Genus);
            taxon_map.entry(key).or_default().push(i);
        }

        let mut taxon_labels = Vec::new();
        let mut taxon_counts = Vec::new();
        let mut all_kmers = HashSet::new();

        let mut taxon_sparse: Vec<HashMap<u64, usize>> = Vec::new();
        for (taxon_key, ref_indices) in &taxon_map {
            taxon_labels.push(Lineage::from_taxonomy_string(taxon_key));
            taxon_counts.push(ref_indices.len());

            let mut kmer_presence: HashMap<u64, usize> = HashMap::new();
            for &ri in ref_indices {
                let kmers = extract_kmers(&refs[ri].sequence, k);
                for kmer in kmers {
                    all_kmers.insert(kmer);
                    *kmer_presence.entry(kmer).or_insert(0) += 1;
                }
            }
            taxon_sparse.push(kmer_presence);
        }

        let n_kmers_total = all_kmers.len();
        let n_taxa = taxon_labels.len();
        #[allow(clippy::cast_precision_loss)]
        let default_log_p = (0.5 / (n_kmers_total.max(1) as f64 + 1.0)).max(1e-300).ln();

        let mut dense_log_probs = vec![default_log_p; n_taxa * kmer_space];
        for (ti, (sparse, count)) in taxon_sparse.iter().zip(taxon_counts.iter()).enumerate() {
            let n_refs = *count as f64;
            let row_start = ti * kmer_space;
            for (&kmer, &presence) in sparse {
                let p = (presence as f64 + 0.5) / (n_refs + 1.0);
                dense_log_probs[row_start + kmer as usize] = p.max(1e-300).ln();
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let total_refs: f64 = taxon_counts.iter().sum::<usize>() as f64;
        let taxon_priors: Vec<f64> = taxon_counts
            .iter()
            .map(|&c| c as f64 / total_refs)
            .collect();
        let log_priors: Vec<f64> = taxon_priors.iter().map(|&p| p.max(1e-300).ln()).collect();

        Self {
            k,
            dense_log_probs,
            kmer_space,
            log_priors,
            taxon_labels,
            taxon_priors,
            n_kmers_total,
        }
    }

    /// Access the dense log-probability table for GPU GEMM dispatch.
    /// Layout: `n_taxa` × `kmer_space`, row-major.
    #[must_use]
    pub fn dense_log_probs(&self) -> &[f64] {
        &self.dense_log_probs
    }

    /// K-mer space size (4^k).
    #[must_use]
    pub const fn kmer_space(&self) -> usize {
        self.kmer_space
    }

    /// Log-prior per taxon.
    #[must_use]
    pub fn log_priors(&self) -> &[f64] {
        &self.log_priors
    }

    /// Taxon labels.
    #[must_use]
    pub fn taxon_labels(&self) -> &[Lineage] {
        &self.taxon_labels
    }

    /// Number of taxa in the classifier.
    #[must_use]
    pub fn n_taxa(&self) -> usize {
        self.taxon_labels.len()
    }

    /// Prior probability per taxon (fraction of training sequences).
    #[must_use]
    pub fn taxon_priors(&self) -> &[f64] {
        &self.taxon_priors
    }

    /// Total distinct k-mers observed across all training sequences.
    #[must_use]
    pub const fn n_kmers_total(&self) -> usize {
        self.n_kmers_total
    }

    /// Classify a query sequence.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn classify(&self, sequence: &[u8], params: &ClassifyParams) -> Classification {
        let query_kmers: Vec<u64> = extract_kmers(sequence, self.k);

        if query_kmers.is_empty() || self.taxon_labels.is_empty() {
            return Classification {
                lineage: Lineage {
                    ranks: vec!["Unclassified".to_string()],
                },
                confidence: vec![0.0; 7],
                taxon_idx: 0,
            };
        }

        let best_taxon = self.score_all_kmers(&query_kmers);
        let confidence = self.bootstrap_confidence(&query_kmers, params.bootstrap_n, best_taxon);

        Classification {
            lineage: self.taxon_labels[best_taxon].clone(),
            confidence,
            taxon_idx: best_taxon,
        }
    }

    /// Score using all query k-mers and return best taxon index.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn score_all_kmers(&self, query_kmers: &[u64]) -> usize {
        let n_taxa = self.taxon_labels.len();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for ti in 0..n_taxa {
            let row = ti * self.kmer_space;
            let mut log_score = self.log_priors[ti];
            for &kmer in query_kmers {
                log_score += self.dense_log_probs[row + kmer as usize];
            }
            if log_score > best_score {
                best_score = log_score;
                best_idx = ti;
            }
        }

        best_idx
    }

    /// Bootstrap confidence estimation.
    ///
    /// Repeatedly classifies random subsets of k-mers and counts how often
    /// each rank agrees with the full classification.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn bootstrap_confidence(
        &self,
        query_kmers: &[u64],
        n_boot: usize,
        full_taxon: usize,
    ) -> Vec<f64> {
        let n_ranks = TaxRank::all().len();
        let mut rank_votes = vec![0_usize; n_ranks];
        let n_sample = (query_kmers.len() * 2 / 3).max(1);
        let full_lineage = &self.taxon_labels[full_taxon];

        let mut seed: u64 = 42;

        for _ in 0..n_boot {
            let subset: Vec<u64> = (0..n_sample)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let idx = (seed >> 33) as usize % query_kmers.len();
                    query_kmers[idx]
                })
                .collect();

            let boot_taxon = self.score_all_kmers(&subset);
            let boot_lineage = &self.taxon_labels[boot_taxon];

            for (ri, rank) in TaxRank::all().iter().enumerate() {
                let full_at = full_lineage.at_rank(*rank);
                let boot_at = boot_lineage.at_rank(*rank);
                if full_at.is_some() && full_at == boot_at {
                    rank_votes[ri] += 1;
                }
            }
        }

        rank_votes
            .iter()
            .map(|&v| v as f64 / n_boot as f64)
            .collect()
    }

    /// Quantize log-probability table to int8 for NPU inference.
    ///
    /// Maps the f64 log-probability range to `[-128, 127]` using affine
    /// quantization: `q = round((x - zero_point) / scale)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn to_int8_weights(&self) -> NpuWeights {
        if self.dense_log_probs.is_empty() {
            return NpuWeights {
                weights_i8: Vec::new(),
                priors_i8: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
                n_taxa: 0,
                kmer_space: self.kmer_space,
            };
        }

        let min_val = self
            .dense_log_probs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_val = self
            .dense_log_probs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let zero_point = min_val;

        let weights_i8: Vec<i8> = self
            .dense_log_probs
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i64 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();

        let priors_i8: Vec<i8> = self
            .log_priors
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i64 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();

        NpuWeights {
            weights_i8,
            priors_i8,
            scale,
            zero_point,
            n_taxa: self.taxon_labels.len(),
            kmer_space: self.kmer_space,
        }
    }

    /// Classify using int8-quantized weights (NPU path).
    ///
    /// Produces the same argmax as full-precision for well-separated taxa.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    #[must_use]
    pub fn classify_quantized(&self, sequence: &[u8]) -> usize {
        let query_kmers = extract_kmers(sequence, self.k);
        if query_kmers.is_empty() || self.taxon_labels.is_empty() {
            return 0;
        }

        let npu = self.to_int8_weights();
        let n_taxa = npu.n_taxa;
        let ks = npu.kmer_space;

        let mut best_score = i64::MIN;
        let mut best_idx = 0;

        for ti in 0..n_taxa {
            let row = ti * ks;
            let mut acc = i64::from(npu.priors_i8[ti]);
            for &kmer in &query_kmers {
                acc += i64::from(npu.weights_i8[row + kmer as usize]);
            }
            if acc > best_score {
                best_score = acc;
                best_idx = ti;
            }
        }

        best_idx
    }
}
