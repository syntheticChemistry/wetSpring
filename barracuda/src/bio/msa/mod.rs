// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive multiple sequence alignment (MSA).
//!
//! Implements guide-tree-based progressive alignment in the style of
//! MAFFT / MUSCLE / `ClustalW`. Builds a pairwise distance matrix via
//! Smith-Waterman scores, constructs a neighbor-joining guide tree,
//! then aligns sequences progressively from leaves to root.
//!
//! # Algorithm
//!
//! 1. All-vs-all pairwise scoring via [`crate::bio::alignment::score_batch`]
//! 2. Score → distance conversion (`d = max_score − score`)
//! 3. Neighbor-joining guide tree via [`super::neighbor_joining::neighbor_joining`]
//! 4. Progressive alignment following the guide tree topology
//!    - Leaf–leaf: standard Smith-Waterman → global extension
//!    - Profile–leaf / profile–profile: majority-consensus scoring
//!
//! # Limitations
//!
//! - No iterative refinement (MUSCLE stage 3). This is progressive only.
//! - Profile–profile alignment uses majority consensus, not full DP.
//! - Suited for moderate sequence counts (< 1000). For larger sets,
//!   delegate to GPU-accelerated pairwise scoring.
//!
//! # References
//!
//! - Katoh et al. 2002, *NAR* 30:3059-3066 (MAFFT)
//! - Edgar 2004, *NAR* 32:1792-1797 (MUSCLE)
//! - Liu et al. 2009, *Science* 324:1561-1564 (`SATé`)

mod alignment;
mod scoring;

use crate::bio::alignment::{self as pairwise, ScoringParams};

/// Result of multiple sequence alignment.
#[derive(Debug, Clone)]
pub struct MsaResult {
    /// Aligned sequences (same length, with `-` gap characters).
    pub aligned: Vec<Vec<u8>>,
    /// Original sequence labels in input order.
    pub labels: Vec<String>,
    /// Alignment length (columns).
    pub alignment_length: usize,
    /// Guide tree in Newick format.
    pub guide_tree: String,
}

/// Configuration for MSA.
#[derive(Debug, Clone, Default)]
pub struct MsaParams {
    /// Scoring parameters for pairwise alignment.
    pub scoring: ScoringParams,
}

/// Perform progressive multiple sequence alignment.
///
/// # Arguments
///
/// * `sequences` — Input sequences (unaligned, variable length).
/// * `labels` — Label for each sequence (same length as `sequences`).
/// * `params` — MSA parameters.
///
/// # Returns
///
/// [`MsaResult`] with all sequences aligned to equal length.
///
/// # Panics
///
/// Panics if `sequences.len() != labels.len()` or fewer than 2 sequences.
#[must_use]
pub fn align_multiple(
    sequences: &[&[u8]],
    labels: &[impl AsRef<str>],
    params: &MsaParams,
) -> MsaResult {
    let n = sequences.len();
    assert_eq!(n, labels.len(), "sequences and labels must match");
    assert!(n >= 2, "need at least 2 sequences for MSA");

    if n == 2 {
        return alignment::align_pair(sequences, labels, &params.scoring);
    }

    // 1. Pairwise scoring
    let scores = pairwise::pairwise_scores(sequences, &params.scoring);

    // 2. Convert scores to distances
    let dist = scoring::scores_to_distances(&scores, n);

    // 3. Build guide tree
    let str_labels: Vec<String> = labels.iter().map(|l| l.as_ref().to_owned()).collect();
    let nj = super::neighbor_joining::neighbor_joining(&dist, &str_labels);

    // 4. Progressive alignment following NJ join order
    //    Parse the Newick tree to determine join order, then merge
    //    alignments from leaves up.
    let aligned = alignment::progressive_align(sequences, &nj.newick, &params.scoring);

    let alignment_length = aligned.first().map_or(0, Vec::len);

    MsaResult {
        aligned,
        labels: str_labels,
        alignment_length,
        guide_tree: nj.newick,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::alignment::ScoringParams;

    #[test]
    fn msa_identical_sequences() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGT", b"ACGTACGT"];
        let labels = ["S1", "S2", "S3"];
        let result = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(result.aligned.len(), 3);
        assert_eq!(result.labels, vec!["S1", "S2", "S3"]);
        // All rows should be identical and equal length
        let len = result.alignment_length;
        for row in &result.aligned {
            assert_eq!(row.len(), len);
        }
    }

    #[test]
    fn msa_two_sequences() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACTT"];
        let labels = ["A", "B"];
        let result = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(result.aligned.len(), 2);
        assert_eq!(result.aligned[0].len(), result.aligned[1].len());
    }

    #[test]
    fn msa_different_lengths() {
        let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACGTACGT", b"ACGT"];
        let labels = ["short1", "long", "short2"];
        let result = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(result.aligned.len(), 3);
        let len = result.alignment_length;
        for row in &result.aligned {
            assert_eq!(row.len(), len, "all rows must be same length");
        }
    }

    #[test]
    fn msa_with_gaps() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGACGT", b"ACGTACGT"];
        let labels = ["A", "B", "C"];
        let result = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(result.aligned.len(), 3);
        let len = result.alignment_length;
        for row in &result.aligned {
            assert_eq!(row.len(), len);
        }
        // B is shorter → should have at least one gap
        let has_gap = result.aligned[1].contains(&b'-');
        assert!(has_gap, "shorter sequence should have gaps");
    }

    #[test]
    fn msa_four_sequences() {
        let seqs: Vec<&[u8]> = vec![
            b"ACGTACGTACGT",
            b"ACGTACTTACGT",
            b"TGCATGCATGCA",
            b"ACGTACGTACTT",
        ];
        let labels = ["S1", "S2", "S3", "S4"];
        let result = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(result.aligned.len(), 4);
        let len = result.alignment_length;
        for row in &result.aligned {
            assert_eq!(row.len(), len);
        }
        assert!(!result.guide_tree.is_empty());
    }

    #[test]
    fn msa_deterministic() {
        let seqs: Vec<&[u8]> = vec![b"AAACCC", b"AAAGGG", b"CCCGGG"];
        let labels = ["X", "Y", "Z"];
        let r1 = align_multiple(&seqs, &labels, &MsaParams::default());
        let r2 = align_multiple(&seqs, &labels, &MsaParams::default());
        assert_eq!(r1.aligned, r2.aligned);
        assert_eq!(r1.guide_tree, r2.guide_tree);
    }

    #[test]
    fn consensus_basic() {
        let profile = vec![b"ACGT".to_vec(), b"ACGT".to_vec(), b"ACTT".to_vec()];
        let cons = alignment::consensus(&profile);
        assert_eq!(cons, b"ACGT"); // majority at position 2 is G (2 vs 1 T)
    }

    #[test]
    fn extract_leaves_basic() {
        let newick = "(A:0.1,B:0.2);";
        let leaves = alignment::extract_newick_leaves(newick);
        assert_eq!(leaves, vec!["A", "B"]);
    }

    #[test]
    fn extract_leaves_nested() {
        let newick = "((A:0.1,B:0.2):0.3,C:0.4);";
        let leaves = alignment::extract_newick_leaves(newick);
        assert_eq!(leaves, vec!["A", "B", "C"]);
    }

    #[test]
    fn column_map_gapped() {
        let gapped = b"AC--GT";
        let map = alignment::build_column_map(gapped);
        assert_eq!(map, vec![Some(0), Some(1), None, None, Some(2), Some(3)]);
    }

    #[test]
    fn extend_to_global_covers_full() {
        let s1 = b"XXXACGTXXX";
        let s2 = b"ACGT";
        let local = crate::bio::alignment::smith_waterman(s1, s2, &ScoringParams::default());
        let (a, b) = alignment::extend_to_global(s1, s2, &local);
        assert_eq!(a.len(), b.len());
        let a_ungapped: Vec<u8> = a.iter().filter(|&&c| c != b'-').copied().collect();
        assert_eq!(a_ungapped, s1.to_vec());
    }
}
