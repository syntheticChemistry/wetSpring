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
///
/// Follows the standard hierarchy from kingdom down to species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaxRank {
    /// Domain/kingdom (e.g., Bacteria, Archaea).
    Kingdom,
    /// Phylum-level classification.
    Phylum,
    /// Class-level classification.
    Class,
    /// Order-level classification.
    Order,
    /// Family-level classification.
    Family,
    /// Genus-level classification.
    Genus,
    /// Species-level classification.
    Species,
}

impl TaxRank {
    /// Return all ranks from kingdom to species, in order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Kingdom,
            Self::Phylum,
            Self::Class,
            Self::Order,
            Self::Family,
            Self::Genus,
            Self::Species,
        ]
    }

    /// Zero-based index for this rank (Kingdom=0, Species=6). Use with `ranks.get(depth)`.
    #[must_use]
    pub const fn depth(self) -> usize {
        match self {
            Self::Kingdom => 0,
            Self::Phylum => 1,
            Self::Class => 2,
            Self::Order => 3,
            Self::Family => 4,
            Self::Genus => 5,
            Self::Species => 6,
        }
    }
}

/// A taxonomic lineage (one entry per rank from kingdom to species).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lineage {
    /// Taxon names at each rank (index matches `TaxRank::depth()`).
    pub ranks: Vec<String>,
}

impl Lineage {
    /// Parse a semicolon-delimited taxonomy string (e.g., SILVA format).
    /// `d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;...`
    #[must_use]
    pub fn from_taxonomy_string(s: &str) -> Self {
        let ranks: Vec<String> = s
            .split(';')
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .collect();
        Self { ranks }
    }

    /// Lookup taxon name at a specific rank. Returns `None` if rank exceeds lineage depth.
    #[must_use]
    pub fn at_rank(&self, rank: TaxRank) -> Option<&str> {
        self.ranks.get(rank.depth()).map(String::as_str)
    }

    /// Format lineage up to (and including) the given rank, semicolon-separated.
    #[must_use]
    pub fn to_string_at_rank(&self, rank: TaxRank) -> String {
        let depth = rank.depth() + 1;
        self.ranks[..depth.min(self.ranks.len())].join(";")
    }
}

/// A reference sequence for training the classifier.
#[derive(Debug, Clone)]
pub struct ReferenceSeq {
    /// Accession or identifier from the FASTA header.
    pub id: String,
    /// DNA sequence as bytes (A,C,G,T in uppercase).
    pub sequence: Vec<u8>,
    /// Taxonomic lineage parsed from header or taxonomy file.
    pub lineage: Lineage,
}

/// A trained naive Bayes classifier.
///
/// Stores log-probabilities in a flat `n_taxa` × `kmer_space` array for
/// O(1) lookup (no `HashMap` in the scoring hot path). For `k`=8, `kmer_space`
/// = 4^8 = 65,536 entries per taxon.
#[derive(Debug)]
pub struct NaiveBayesClassifier {
    k: usize,
    /// Flat log-probability table: `dense_log_probs[taxon * kmer_space + kmer]`
    /// = log P(kmer | taxon). Pre-filled with `default_log_p` for unseen k-mers.
    dense_log_probs: Vec<f64>,
    /// Number of possible k-mers: 4^k
    kmer_space: usize,
    /// Log-prior per taxon: ln P(taxon)
    log_priors: Vec<f64>,
    /// Taxon labels (full lineage string)
    taxon_labels: Vec<Lineage>,
    /// Prior probabilities: P(taxon) — kept for model introspection
    #[allow(dead_code)]
    taxon_priors: Vec<f64>,
    /// Total number of unique k-mers seen in training — kept for model introspection
    #[allow(dead_code)]
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
    /// and precomputes a flat log-probability table for O(1) scoring.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn train(refs: &[ReferenceSeq], k: usize) -> Self {
        let kmer_space = 1_usize << (2 * k); // 4^k

        let mut taxon_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in refs.iter().enumerate() {
            let key = r.lineage.to_string_at_rank(TaxRank::Genus);
            taxon_map.entry(key).or_default().push(i);
        }

        let mut taxon_labels = Vec::new();
        let mut taxon_counts = Vec::new();
        let mut all_kmers: HashMap<u64, bool> = HashMap::new();

        // First pass: collect all taxon data
        let mut taxon_sparse: Vec<HashMap<u64, usize>> = Vec::new();
        for (taxon_key, ref_indices) in &taxon_map {
            taxon_labels.push(Lineage::from_taxonomy_string(taxon_key));
            taxon_counts.push(ref_indices.len());

            let mut kmer_presence: HashMap<u64, usize> = HashMap::new();
            for &ri in ref_indices {
                let kmers = extract_kmers(&refs[ri].sequence, k);
                for kmer in kmers {
                    all_kmers.insert(kmer, true);
                    *kmer_presence.entry(kmer).or_insert(0) += 1;
                }
            }
            taxon_sparse.push(kmer_presence);
        }

        let n_kmers_total = all_kmers.len();
        let n_taxa = taxon_labels.len();
        let default_log_p = (0.5 / (n_kmers_total.max(1) as f64 + 1.0)).max(1e-300).ln();

        // Build dense log-probability table: n_taxa × kmer_space
        let mut dense_log_probs = vec![default_log_p; n_taxa * kmer_space];
        for (ti, (sparse, count)) in taxon_sparse.iter().zip(taxon_counts.iter()).enumerate() {
            let n_refs = *count as f64;
            let row_start = ti * kmer_space;
            for (&kmer, &presence) in sparse {
                let p = (presence as f64 + 0.5) / (n_refs + 1.0);
                dense_log_probs[row_start + kmer as usize] = p.max(1e-300).ln();
            }
        }

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
    /// Uses flat array indexing — no `HashMap` lookups.
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
    /// Repeatedly classify random subsets of k-mers and count how often
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
    #[must_use]
    pub fn n_taxa(&self) -> usize {
        self.taxon_labels.len()
    }
}

/// Extract all k-mers from a DNA sequence (no canonicalization — direction matters for taxonomy).
#[must_use]
pub fn extract_kmers(seq: &[u8], k: usize) -> Vec<u64> {
    if seq.len() < k || k == 0 || k > 32 {
        return vec![];
    }

    let mut kmers = Vec::with_capacity(seq.len() - k + 1);
    let mask = if k == 32 {
        u64::MAX
    } else {
        (1_u64 << (2 * k)) - 1
    };
    let mut kmer = 0_u64;
    let mut valid_len = 0_usize;

    for &base in seq {
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

const fn encode_base(b: u8) -> Option<u64> {
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
#[must_use]
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
                    .filter(u8::is_ascii_alphabetic)
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
#[allow(clippy::expect_used, clippy::unwrap_used)]
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
            make_ref(
                "r3",
                b"GGGGGGGGGTTTTTTTTTT",
                "Bacteria;Proteobacteria;Gamma",
            ),
            make_ref(
                "r4",
                b"GGGGGGGGGTTTTTTTTTT",
                "Bacteria;Proteobacteria;Gamma",
            ),
        ];

        let classifier = NaiveBayesClassifier::train(&refs, 4);
        assert_eq!(classifier.n_taxa(), 2);

        // Query similar to Firmicutes
        let result = classifier.classify(b"AAAAAAAAACCCCCCCCC", &ClassifyParams::default());
        assert!(result
            .lineage
            .at_rank(TaxRank::Phylum)
            .unwrap()
            .contains("Firmicutes"));

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
        let refs = vec![make_ref("r1", b"ACGTACGTACGTACGT", "K;P;C;O;F;G;S")];
        let classifier = NaiveBayesClassifier::train(&refs, 4);
        let result = classifier.classify(b"ACGTACGTACGTACGT", &ClassifyParams::default());
        assert_eq!(result.confidence.len(), 7);
    }
}
