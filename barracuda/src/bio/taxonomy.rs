// SPDX-License-Identifier: AGPL-3.0-or-later
//! Naive Bayes taxonomy classification for 16S rRNA sequences.
//!
//! Implements the RDP-style naive Bayes classifier (Wang et al. 2007) used by
//! QIIME2's `feature-classifier classify-sklearn` and DADA2's `assignTaxonomy`.
//!
//! # Algorithm
//!
//! 1. **Training**: Build a k-mer frequency model from a reference database
//!    (e.g., SILVA 138). For each taxon at each rank, record which k-mers
//!    are present and their conditional probabilities.
//!
//! 2. **Classification**: For each query sequence, extract k-mers and compute
//!    the posterior probability for each taxon using Bayes' theorem with the
//!    naive (conditional independence) assumption.
//!
//! 3. **Bootstrapping**: Repeat classification on random subsets of the query's
//!    k-mers to estimate confidence at each taxonomic rank.
//!
//! # References
//!
//! - Wang et al. "Naive Bayesian Classifier for Rapid Assignment of rRNA
//!   Sequences into the New Bacterial Taxonomy." Applied and Environmental
//!   Microbiology 73, 5261–5267 (2007).
//! - SILVA 138 SSU reference database.

use std::collections::HashMap;

const DEFAULT_K: usize = 8;
const DEFAULT_BOOTSTRAP_N: usize = 100;
const DEFAULT_MIN_CONFIDENCE: f64 = 0.8;

/// Taxonomic ranks used in 16S classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaxRank {
    Kingdom,
    Phylum,
    Class,
    Order,
    Family,
    Genus,
    Species,
}

impl TaxRank {
    pub fn all() -> &'static [TaxRank] {
        &[
            TaxRank::Kingdom,
            TaxRank::Phylum,
            TaxRank::Class,
            TaxRank::Order,
            TaxRank::Family,
            TaxRank::Genus,
            TaxRank::Species,
        ]
    }

    pub fn depth(self) -> usize {
        match self {
            TaxRank::Kingdom => 0,
            TaxRank::Phylum => 1,
            TaxRank::Class => 2,
            TaxRank::Order => 3,
            TaxRank::Family => 4,
            TaxRank::Genus => 5,
            TaxRank::Species => 6,
        }
    }
}

/// A taxonomic lineage (one entry per rank from kingdom to species).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lineage {
    pub ranks: Vec<String>,
}

impl Lineage {
    /// Parse a semicolon-delimited taxonomy string (e.g., SILVA format).
    /// "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;..."
    pub fn from_taxonomy_string(s: &str) -> Self {
        let ranks: Vec<String> = s
            .split(';')
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect();
        Lineage { ranks }
    }

    pub fn at_rank(&self, rank: TaxRank) -> Option<&str> {
        self.ranks.get(rank.depth()).map(String::as_str)
    }

    pub fn to_string_at_rank(&self, rank: TaxRank) -> String {
        let depth = rank.depth() + 1;
        self.ranks[..depth.min(self.ranks.len())].join(";")
    }
}

/// A reference sequence for training the classifier.
#[derive(Debug, Clone)]
pub struct ReferenceSeq {
    pub id: String,
    pub sequence: Vec<u8>,
    pub lineage: Lineage,
}

/// A trained naive Bayes classifier.
#[derive(Debug)]
pub struct NaiveBayesClassifier {
    k: usize,
    /// For each genus (or lowest-rank taxon): taxon string -> kmer presence probabilities
    /// taxon_kmer_probs[taxon_id][kmer] = P(kmer | taxon)
    taxon_kmer_probs: Vec<HashMap<u64, f64>>,
    /// Taxon labels (full lineage string)
    taxon_labels: Vec<Lineage>,
    /// Prior probabilities: P(taxon)
    taxon_priors: Vec<f64>,
    /// Total number of unique k-mers seen in training
    n_kmers_total: usize,
}

/// Classification result for a single query.
#[derive(Debug, Clone)]
pub struct Classification {
    /// Assigned lineage.
    pub lineage: Lineage,
    /// Bootstrap confidence at each rank (0.0 to 1.0).
    pub confidence: Vec<f64>,
    /// Index of the matched taxon in the classifier.
    pub taxon_idx: usize,
}

/// Parameters for classification.
#[derive(Debug, Clone)]
pub struct ClassifyParams {
    /// K-mer size (must match training). Default: 8.
    pub k: usize,
    /// Number of bootstrap iterations. Default: 100.
    pub bootstrap_n: usize,
    /// Minimum bootstrap confidence to report a rank. Default: 0.8.
    pub min_confidence: f64,
}

impl Default for ClassifyParams {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            bootstrap_n: DEFAULT_BOOTSTRAP_N,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }
}

impl NaiveBayesClassifier {
    /// Train a classifier from reference sequences.
    ///
    /// Groups reference sequences by genus-level lineage, extracts k-mers,
    /// and computes conditional probabilities with Laplace smoothing.
    #[allow(clippy::cast_precision_loss)]
    pub fn train(refs: &[ReferenceSeq], k: usize) -> Self {
        // Group references by genus-level lineage
        let mut taxon_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in refs.iter().enumerate() {
            let key = r.lineage.to_string_at_rank(TaxRank::Genus);
            taxon_map.entry(key).or_default().push(i);
        }

        let mut taxon_labels = Vec::new();
        let mut taxon_kmer_probs = Vec::new();
        let mut taxon_counts = Vec::new();
        let mut all_kmers: HashMap<u64, bool> = HashMap::new();

        for (taxon_key, ref_indices) in &taxon_map {
            taxon_labels.push(Lineage::from_taxonomy_string(taxon_key));
            taxon_counts.push(ref_indices.len());

            // Collect all k-mers present in any reference in this taxon
            let mut kmer_presence: HashMap<u64, usize> = HashMap::new();
            for &ri in ref_indices {
                let kmers = extract_kmers(&refs[ri].sequence, k);
                for kmer in kmers {
                    all_kmers.insert(kmer, true);
                    *kmer_presence.entry(kmer).or_insert(0) += 1;
                }
            }

            // Convert to probabilities with Laplace smoothing
            let n_refs = ref_indices.len() as f64;
            let probs: HashMap<u64, f64> = kmer_presence
                .into_iter()
                .map(|(kmer, count)| {
                    let p = (count as f64 + 0.5) / (n_refs + 1.0);
                    (kmer, p)
                })
                .collect();

            taxon_kmer_probs.push(probs);
        }

        let total_refs: f64 = taxon_counts.iter().sum::<usize>() as f64;
        let taxon_priors: Vec<f64> = taxon_counts
            .iter()
            .map(|&c| c as f64 / total_refs)
            .collect();

        NaiveBayesClassifier {
            k,
            taxon_kmer_probs,
            taxon_labels,
            taxon_priors,
            n_kmers_total: all_kmers.len(),
        }
    }

    /// Classify a query sequence.
    #[allow(clippy::cast_precision_loss)]
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

        // Full classification
        let best_taxon = self.score_all_kmers(&query_kmers);

        // Bootstrap confidence
        let confidence = self.bootstrap_confidence(&query_kmers, params.bootstrap_n, best_taxon);

        Classification {
            lineage: self.taxon_labels[best_taxon].clone(),
            confidence,
            taxon_idx: best_taxon,
        }
    }

    /// Score using all query k-mers and return best taxon index.
    fn score_all_kmers(&self, query_kmers: &[u64]) -> usize {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        let default_p = 0.5 / (self.n_kmers_total.max(1) as f64 + 1.0);

        for (ti, probs) in self.taxon_kmer_probs.iter().enumerate() {
            let mut log_score = self.taxon_priors[ti].max(1e-300).ln();
            for kmer in query_kmers {
                let p = probs.get(kmer).copied().unwrap_or(default_p);
                log_score += p.max(1e-300).ln();
            }
            if log_score > best_score {
                best_score = log_score;
                best_idx = ti;
            }
        }

        best_idx
    }

    /// Bootstrap confidence estimation.
    /// Repeatedly classify random subsets of k-mers and count how often
    /// each rank agrees with the full classification.
    #[allow(clippy::cast_precision_loss)]
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

        // Simple deterministic pseudo-random for reproducibility
        let mut seed: u64 = 42;

        for _ in 0..n_boot {
            // Sample k-mers with replacement
            let subset: Vec<u64> = (0..n_sample)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let idx = (seed >> 33) as usize % query_kmers.len();
                    query_kmers[idx]
                })
                .collect();

            let boot_taxon = self.score_all_kmers(&subset);
            let boot_lineage = &self.taxon_labels[boot_taxon];

            // Count rank agreement
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

    /// Number of taxa in the classifier.
    pub fn n_taxa(&self) -> usize {
        self.taxon_labels.len()
    }
}

/// Extract all k-mers from a DNA sequence (no canonicalization — direction matters for taxonomy).
fn extract_kmers(seq: &[u8], k: usize) -> Vec<u64> {
    if seq.len() < k || k == 0 || k > 32 {
        return vec![];
    }

    let mut kmers = Vec::with_capacity(seq.len() - k + 1);
    let mask = if k == 32 { u64::MAX } else { (1_u64 << (2 * k)) - 1 };
    let mut kmer = 0_u64;
    let mut valid_len = 0_usize;

    for &base in seq.iter() {
        if let Some(encoded) = encode_base(base) {
            kmer = ((kmer << 2) | encoded) & mask;
            valid_len += 1;
        } else {
            valid_len = 0;
            kmer = 0;
        }

        if valid_len >= k {
            kmers.push(kmer);
        }
    }

    kmers
}

fn encode_base(b: u8) -> Option<u64> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

/// Parse a FASTA reference database into `ReferenceSeq` records.
/// Expects SILVA-style headers: `>ID taxonomy_string`
pub fn parse_reference_fasta(contents: &str) -> Vec<ReferenceSeq> {
    let mut refs = Vec::new();
    let mut current_id = String::new();
    let mut current_tax = String::new();
    let mut current_seq = Vec::new();

    for line in contents.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !current_id.is_empty() && !current_seq.is_empty() {
                refs.push(ReferenceSeq {
                    id: current_id.clone(),
                    sequence: current_seq.clone(),
                    lineage: Lineage::from_taxonomy_string(&current_tax),
                });
            }
            // Parse header: ID followed by taxonomy (space-separated)
            let parts: Vec<&str> = header.splitn(2, |c: char| c.is_whitespace()).collect();
            current_id = parts[0].to_string();
            current_tax = parts.get(1).unwrap_or(&"").to_string();
            current_seq.clear();
        } else {
            current_seq.extend(
                line.trim()
                    .bytes()
                    .filter(|b| b.is_ascii_alphabetic())
                    .map(|b| b.to_ascii_uppercase()),
            );
        }
    }

    if !current_id.is_empty() && !current_seq.is_empty() {
        refs.push(ReferenceSeq {
            id: current_id,
            sequence: current_seq,
            lineage: Lineage::from_taxonomy_string(&current_tax),
        });
    }

    refs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ref(id: &str, seq: &[u8], tax: &str) -> ReferenceSeq {
        ReferenceSeq {
            id: id.to_string(),
            sequence: seq.to_vec(),
            lineage: Lineage::from_taxonomy_string(tax),
        }
    }

    #[test]
    fn lineage_parsing() {
        let l = Lineage::from_taxonomy_string("d__Bacteria;p__Firmicutes;c__Bacilli");
        assert_eq!(l.ranks.len(), 3);
        assert_eq!(l.at_rank(TaxRank::Kingdom), Some("d__Bacteria"));
        assert_eq!(l.at_rank(TaxRank::Phylum), Some("p__Firmicutes"));
        assert_eq!(l.at_rank(TaxRank::Class), Some("c__Bacilli"));
        assert_eq!(l.at_rank(TaxRank::Order), None);
    }

    #[test]
    fn extract_kmers_basic() {
        let seq = b"ACGT";
        let kmers = extract_kmers(seq, 2);
        // AC=0b0001=1, CG=0b0110=6, GT=0b1011=11? Let me compute:
        // A=0, C=1, G=2, T=3
        // AC = (0<<2)|1 = 1
        // CG = (1<<2)|2 = 6
        // GT = (2<<2)|3 = 11
        assert_eq!(kmers.len(), 3);
        assert_eq!(kmers[0], 1); // AC
        assert_eq!(kmers[1], 6); // CG
        assert_eq!(kmers[2], 11); // GT
    }

    #[test]
    fn extract_kmers_with_ambiguous() {
        let seq = b"ACNGT";
        let kmers = extract_kmers(seq, 2);
        // AC valid, CN skip, NG skip, GT valid (but valid_len resets at N)
        // After N: G starts new, GT is second valid → valid_len=2 at GT
        assert_eq!(kmers.len(), 2); // AC and GT
    }

    #[test]
    fn train_and_classify() {
        let refs = vec![
            make_ref("r1", b"AAAAAAAAACCCCCCCCC", "Bacteria;Firmicutes;Bacilli"),
            make_ref("r2", b"AAAAAAAAACCCCCCCCC", "Bacteria;Firmicutes;Bacilli"),
            make_ref("r3", b"GGGGGGGGGTTTTTTTTTT", "Bacteria;Proteobacteria;Gamma"),
            make_ref("r4", b"GGGGGGGGGTTTTTTTTTT", "Bacteria;Proteobacteria;Gamma"),
        ];

        let classifier = NaiveBayesClassifier::train(&refs, 4);
        assert_eq!(classifier.n_taxa(), 2);

        // Query similar to Firmicutes
        let result = classifier.classify(b"AAAAAAAAACCCCCCCCC", &ClassifyParams::default());
        assert!(result.lineage.at_rank(TaxRank::Phylum).unwrap().contains("Firmicutes"));

        // Query similar to Proteobacteria
        let result = classifier.classify(b"GGGGGGGGGTTTTTTTTTT", &ClassifyParams::default());
        assert!(result
            .lineage
            .at_rank(TaxRank::Phylum)
            .unwrap()
            .contains("Proteobacteria"));
    }

    #[test]
    fn bootstrap_confidence_high_for_clear_match() {
        let refs = vec![
            make_ref("r1", b"ACGTACGTACGTACGT", "Bac;Firm;Bac;Lac;Lac;Lacto"),
            make_ref("r2", b"ACGTACGTACGTACGT", "Bac;Firm;Bac;Lac;Lac;Lacto"),
            make_ref("r3", b"TTTTTTTTTTTTTTTT", "Bac;Prot;Gam;Ent;Ent;Esch"),
            make_ref("r4", b"TTTTTTTTTTTTTTTT", "Bac;Prot;Gam;Ent;Ent;Esch"),
        ];

        let classifier = NaiveBayesClassifier::train(&refs, 4);
        let result = classifier.classify(b"ACGTACGTACGTACGT", &ClassifyParams::default());

        // Kingdom confidence should be 1.0 (both taxa are Bac)
        assert!(result.confidence[0] >= 0.8);
        // Phylum confidence should be high for a clear match
        assert!(result.confidence[1] >= 0.8);
    }

    #[test]
    fn empty_input() {
        let classifier = NaiveBayesClassifier::train(&[], 8);
        assert_eq!(classifier.n_taxa(), 0);

        let result = classifier.classify(b"ACGT", &ClassifyParams::default());
        assert_eq!(result.lineage.ranks[0], "Unclassified");
    }

    #[test]
    fn parse_reference_fasta_basic() {
        let fasta = ">seq1 Bacteria;Firmicutes;Bacilli\nACGTACGT\n>seq2 Bacteria;Proteobacteria\nGCTAGCTA\n";
        let refs = parse_reference_fasta(fasta);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "seq1");
        assert_eq!(refs[0].sequence, b"ACGTACGT");
        assert_eq!(refs[0].lineage.ranks[0], "Bacteria");
        assert_eq!(refs[1].lineage.ranks[1], "Proteobacteria");
    }

    #[test]
    fn classify_returns_confidence_per_rank() {
        let refs = vec![
            make_ref("r1", b"ACGTACGTACGTACGT", "K;P;C;O;F;G;S"),
        ];
        let classifier = NaiveBayesClassifier::train(&refs, 4);
        let result = classifier.classify(b"ACGTACGTACGTACGT", &ClassifyParams::default());
        assert_eq!(result.confidence.len(), 7);
    }
}
