// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference-free chimera detection for amplicon sequences.
//!
//! Implements a UCHIME-style algorithm (Edgar et al. 2011) for detecting
//! chimeric sequences formed during PCR amplification. Chimeras are
//! artificial sequences composed of two or more parent sequences joined
//! at crossover points.
//!
//! # Algorithm
//!
//! 1. Sort sequences by abundance (descending).
//! 2. For each query, use k-mer sketch to find the top-K most similar
//!    parents from the pool of more-abundant sequences.
//! 3. For the top parent pairs, compute chimera scores using prefix-sum
//!    match vectors for O(1) crossover evaluation.
//! 4. If the best chimera score exceeds the threshold, flag the sequence.
//!
//! # Performance
//!
//! Uses k-mer sketching (8-mers) for parent candidate selection instead
//! of exhaustive all-pairs. For N ASVs with K candidate parents per query
//! and L base sequence length:
//!
//! - Previous: O(N × P² × L) where P ≈ N → O(N³L)
//! - Current:  O(N × K² × L) where K = min(8, P) → O(N × 64 × L)
//!
//! # References
//!
//! - Edgar et al. "UCHIME improves sensitivity and speed of chimera
//!   detection." Bioinformatics 27, 2194–2200 (2011).
//! - DADA2 `removeBimeraDenovo` (Callahan et al. 2016).

mod detection;
mod kmer_sketch;

#[cfg(feature = "gpu")]
pub(crate) use detection::test_chimera_fast;
pub use detection::{detect_chimeras, remove_chimeras};

/// Result of chimera detection for a single sequence.
#[derive(Debug, Clone)]
pub struct ChimeraResult {
    /// Index of the query sequence.
    pub query_idx: usize,
    /// Whether this sequence is flagged as chimeric.
    pub is_chimera: bool,
    /// Chimera score (higher = more chimeric).
    pub score: f64,
    /// Index of the left parent (in the input list), if chimeric.
    pub left_parent: Option<usize>,
    /// Index of the right parent (in the input list), if chimeric.
    pub right_parent: Option<usize>,
    /// Crossover position (0-based index in the alignment).
    pub crossover: Option<usize>,
}

/// Parameters for chimera detection.
#[derive(Debug, Clone)]
pub struct ChimeraParams {
    /// Minimum chimera score to flag a sequence. Default: 2.0.
    pub min_score: f64,
    /// Minimum fold-abundance of a parent relative to the query.
    /// A parent must be at least this many times more abundant. Default: 2.0.
    pub min_parent_fold: f64,
    /// Minimum number of differences between a query and each parent
    /// in their respective non-matching segments. Default: 3.
    pub min_diffs: usize,
}

const DEFAULT_MIN_SCORE: f64 = 2.0;
const DEFAULT_MIN_PARENT_ABUNDANCE: f64 = 2.0;

/// Minimum segment length for crossover evaluation.
pub(super) const MIN_SEGMENT_LEN: usize = 3;

/// Manual impl intentional: all fields use non-zero defaults (UCHIME-style).
impl Default for ChimeraParams {
    fn default() -> Self {
        Self {
            min_score: DEFAULT_MIN_SCORE,
            min_parent_fold: DEFAULT_MIN_PARENT_ABUNDANCE,
            min_diffs: MIN_SEGMENT_LEN,
        }
    }
}

/// Statistics from chimera detection.
#[derive(Debug, Clone)]
pub struct ChimeraStats {
    /// Total sequences evaluated.
    pub input_sequences: usize,
    /// Number of chimeras detected.
    pub chimeras_found: usize,
    /// Number of non-chimeric sequences retained.
    pub retained: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::dada2::Asv;

    fn make_asv(seq: &[u8], abundance: usize) -> Asv {
        Asv {
            sequence: seq.to_vec(),
            abundance,
            n_members: 1,
        }
    }

    #[test]
    fn no_chimera_in_small_set() {
        let asvs = vec![make_asv(b"AAAAAAAAAA", 100)];
        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 0);
        assert_eq!(stats.retained, 1);
        assert!(!results[0].is_chimera);
    }

    #[test]
    fn two_distinct_sequences_no_chimera() {
        let asvs = vec![make_asv(b"AAAAAAAAAA", 1000), make_asv(b"CCCCCCCCCC", 500)];
        let (_, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 0);
        assert_eq!(stats.retained, 2);
    }

    #[test]
    fn obvious_chimera_detected() {
        let parent_a = b"AAAAAGGGGG";
        let parent_b = b"CCCCCTTTTTT";
        let chimera = b"AAAAATTTTT";

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 800),
            make_asv(chimera, 10),
        ];

        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(
            results[2].is_chimera,
            "chimera not detected, score={}",
            results[2].score
        );
        assert_eq!(stats.chimeras_found, 1);
    }

    #[test]
    fn real_looking_chimera() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        let asvs = vec![
            make_asv(parent_a, 5000),
            make_asv(parent_b, 3000),
            make_asv(chimera, 50),
        ];

        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(results[2].is_chimera, "score={}", results[2].score);
        assert_eq!(stats.chimeras_found, 1);
    }

    #[test]
    fn non_chimeric_variant_not_flagged() {
        let parent_a = b"AAAAAAAAAA";
        let parent_b = b"CCCCCCCCCC";
        let variant = b"AAAAAAAAAT";

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 800),
            make_asv(variant, 50),
        ];

        let (results, _) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(!results[2].is_chimera);
    }

    #[test]
    fn remove_chimeras_filters_correctly() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        let asvs = vec![
            make_asv(parent_a, 5000),
            make_asv(parent_b, 3000),
            make_asv(chimera, 50),
        ];

        let (filtered, stats) = remove_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 1);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn empty_input() {
        let (results, stats) = detect_chimeras(&[], &ChimeraParams::default());
        assert!(results.is_empty());
        assert_eq!(stats.chimeras_found, 0);
    }

    #[test]
    fn min_parent_fold_filter() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        let asvs = vec![
            make_asv(parent_a, 100),
            make_asv(parent_b, 100),
            make_asv(chimera, 90),
        ];

        let params = ChimeraParams {
            min_parent_fold: 2.0,
            ..ChimeraParams::default()
        };
        let (results, _) = detect_chimeras(&asvs, &params);
        assert!(!results[2].is_chimera);
    }

    #[test]
    fn default_params_values() {
        let p = ChimeraParams::default();
        assert!((p.min_score - DEFAULT_MIN_SCORE).abs() < f64::EPSILON);
        assert!((p.min_parent_fold - DEFAULT_MIN_PARENT_ABUNDANCE).abs() < f64::EPSILON);
        assert_eq!(p.min_diffs, MIN_SEGMENT_LEN);
    }

    #[test]
    fn high_min_score_suppresses_detection() {
        let parent_a = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let parent_b = b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";
        let chimera = b"AAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCC";

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 1000),
            make_asv(chimera, 10),
        ];

        let strict = ChimeraParams {
            min_score: 100.0,
            ..ChimeraParams::default()
        };
        let (results, _) = detect_chimeras(&asvs, &strict);
        assert!(
            !results[2].is_chimera,
            "high min_score should suppress detection"
        );
    }
}
