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
//! # Module structure
//!
//! - [`types`] — Domain types: ranks, lineages, reference sequences, parameters
//! - [`kmers`] — K-mer extraction and FASTA parsing
//! - [`NaiveBayesClassifier`] — Naive Bayes training, classification, NPU quantization
//!
//! # References
//!
//! - Wang et al. "Naive Bayesian Classifier for Rapid Assignment of rRNA
//!   Sequences into the New Bacterial Taxonomy." Applied and Environmental
//!   Microbiology 73, 5261–5267 (2007).
//! - SILVA 138 SSU reference database.

mod classifier;
pub mod kmers;
pub mod types;

pub use classifier::NaiveBayesClassifier;
pub use kmers::{extract_kmers, parse_reference_fasta};
pub use types::{Classification, ClassifyParams, Lineage, NpuWeights, ReferenceSeq, TaxRank};

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
        assert_eq!(kmers.len(), 3);
        assert_eq!(kmers[0], 1); // AC
        assert_eq!(kmers[1], 6); // CG
        assert_eq!(kmers[2], 11); // GT
    }

    #[test]
    fn extract_kmers_with_ambiguous() {
        let seq = b"ACNGT";
        let kmers = extract_kmers(seq, 2);
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

        let result = classifier.classify(b"AAAAAAAAACCCCCCCCC", &ClassifyParams::default());
        assert!(
            result
                .lineage
                .at_rank(TaxRank::Phylum)
                .unwrap()
                .contains("Firmicutes")
        );

        let result = classifier.classify(b"GGGGGGGGGTTTTTTTTTT", &ClassifyParams::default());
        assert!(
            result
                .lineage
                .at_rank(TaxRank::Phylum)
                .unwrap()
                .contains("Proteobacteria")
        );
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

        assert!(result.confidence[0] >= 0.8);
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

    #[test]
    fn int8_quantization_round_trip() {
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
        let npu = classifier.to_int8_weights();

        assert_eq!(npu.n_taxa, 2);
        assert_eq!(npu.kmer_space, 256);
        assert_eq!(npu.weights_i8.len(), 2 * 256);
        assert_eq!(npu.priors_i8.len(), 2);
        assert!(npu.scale > 0.0);
    }

    #[test]
    fn quantized_classification_parity() {
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

        let full_result = classifier.classify(b"AAAAAAAAACCCCCCCCC", &ClassifyParams::default());
        let quant_idx = classifier.classify_quantized(b"AAAAAAAAACCCCCCCCC");
        assert_eq!(
            full_result.taxon_idx, quant_idx,
            "int8 argmax must match f64 for Firmicutes query"
        );

        let full_result2 = classifier.classify(b"GGGGGGGGGTTTTTTTTTT", &ClassifyParams::default());
        let quant_idx2 = classifier.classify_quantized(b"GGGGGGGGGTTTTTTTTTT");
        assert_eq!(
            full_result2.taxon_idx, quant_idx2,
            "int8 argmax must match f64 for Proteobacteria query"
        );
    }

    #[test]
    fn accessor_taxon_priors_and_kmer_count() {
        let refs = vec![
            make_ref("r1", b"AAAAAAAAACCCCCCCCC", "Bacteria;Firmicutes;Bacilli"),
            make_ref("r2", b"AAAAAAAAACCCCCCCCC", "Bacteria;Firmicutes;Bacilli"),
            make_ref("r3", b"AAAAAAAAACCCCCCCCC", "Bacteria;Firmicutes;Bacilli"),
            make_ref(
                "r4",
                b"GGGGGGGGGTTTTTTTTTT",
                "Bacteria;Proteobacteria;Gamma",
            ),
        ];

        let classifier = NaiveBayesClassifier::train(&refs, 4);
        let priors = classifier.taxon_priors();
        assert_eq!(priors.len(), 2);
        let total: f64 = priors.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "priors must sum to 1.0");
        let max_prior = priors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_prior > 0.5, "majority taxon should have prior > 0.5");

        assert!(classifier.n_kmers_total() > 0);
        assert_eq!(classifier.kmer_space(), 256); // 4^4
        assert!(!classifier.dense_log_probs().is_empty());
        assert_eq!(classifier.log_priors().len(), 2);
        assert_eq!(classifier.taxon_labels().len(), 2);
    }

    #[test]
    fn empty_classifier_int8() {
        let classifier = NaiveBayesClassifier::train(&[], 4);
        let npu = classifier.to_int8_weights();
        assert!(npu.weights_i8.is_empty());
        assert!(npu.priors_i8.is_empty());
        assert_eq!(npu.n_taxa, 0);

        let idx = classifier.classify_quantized(b"ACGT");
        assert_eq!(idx, 0);
    }

    #[test]
    fn npu_buffer_sizes() {
        let refs = vec![
            make_ref("r1", b"AAAAAAAAACCCCCCCCC", "Bac;Firm;Bac"),
            make_ref("r2", b"GGGGGGGGGTTTTTTTTTT", "Bac;Prot;Gam"),
            make_ref("r3", b"ACACACACACACACAC", "Bac;Bact;Del"),
        ];

        let classifier = NaiveBayesClassifier::train(&refs, 8);
        let npu = classifier.to_int8_weights();
        assert_eq!(npu.kmer_space, 65_536);
        assert_eq!(
            npu.weights_i8.len(),
            npu.n_taxa * 65_536,
            "weight buffer size = n_taxa × 4^k"
        );
    }
}
