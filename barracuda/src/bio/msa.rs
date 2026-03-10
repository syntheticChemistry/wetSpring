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
//! 1. All-vs-all pairwise scoring via [`alignment::score_batch`]
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

use super::alignment::{self, AlignmentResult, ScoringParams};

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
        return align_pair(sequences, labels, params);
    }

    // 1. Pairwise scoring
    let scores = alignment::pairwise_scores(sequences, &params.scoring);

    // 2. Convert scores to distances
    let max_score = scores.iter().copied().max().unwrap_or(0);
    let mut dist = vec![0.0_f64; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = f64::from(max_score - scores[idx]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
            idx += 1;
        }
    }

    // 3. Build guide tree
    let str_labels: Vec<String> = labels.iter().map(|l| l.as_ref().to_owned()).collect();
    let nj = super::neighbor_joining::neighbor_joining(&dist, &str_labels);

    // 4. Progressive alignment following NJ join order
    //    Parse the Newick tree to determine join order, then merge
    //    alignments from leaves up.
    let aligned = progressive_align(sequences, &nj.newick, params);

    let alignment_length = aligned.first().map_or(0, Vec::len);

    MsaResult {
        aligned,
        labels: str_labels,
        alignment_length,
        guide_tree: nj.newick,
    }
}

/// Two-sequence case: global extension of Smith-Waterman alignment.
fn align_pair(sequences: &[&[u8]], labels: &[impl AsRef<str>], params: &MsaParams) -> MsaResult {
    let r = alignment::smith_waterman(sequences[0], sequences[1], &params.scoring);
    let (a, b) = extend_to_global(sequences[0], sequences[1], &r);

    let alignment_length = a.len();
    let str_labels: Vec<String> = labels.iter().map(|l| l.as_ref().to_owned()).collect();

    MsaResult {
        aligned: vec![a, b],
        labels: str_labels,
        alignment_length,
        guide_tree: String::new(),
    }
}

/// Extend a local alignment to cover full sequences (global-like).
fn extend_to_global(seq1: &[u8], seq2: &[u8], local: &AlignmentResult) -> (Vec<u8>, Vec<u8>) {
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();

    let qs = local.query_start;
    let ts = local.target_start;

    // Leading unaligned portions
    let lead = qs.max(ts);
    for i in 0..lead {
        a1.push(if i >= lead - qs {
            seq1[i - (lead - qs)]
        } else {
            b'-'
        });
        a2.push(if i >= lead - ts {
            seq2[i - (lead - ts)]
        } else {
            b'-'
        });
    }

    // Core local alignment
    a1.extend_from_slice(&local.aligned_query);
    a2.extend_from_slice(&local.aligned_target);

    // Trailing unaligned portions
    let qe = qs + local.aligned_query.iter().filter(|&&c| c != b'-').count();
    let te = ts + local.aligned_target.iter().filter(|&&c| c != b'-').count();
    let q_tail = seq1.len().saturating_sub(qe);
    let t_tail = seq2.len().saturating_sub(te);
    let trail = q_tail.max(t_tail);
    for i in 0..trail {
        a1.push(if i < q_tail { seq1[qe + i] } else { b'-' });
        a2.push(if i < t_tail { seq2[te + i] } else { b'-' });
    }

    (a1, a2)
}

/// Progressive alignment via Newick-guided merge.
///
/// Parses the guide tree to determine which sequences to merge first,
/// then iteratively aligns pairs/profiles.
fn progressive_align(sequences: &[&[u8]], newick: &str, params: &MsaParams) -> Vec<Vec<u8>> {
    // Parse join order from the Newick tree
    let leaf_names: Vec<String> = (0..sequences.len()).map(|i| format!("{i}")).collect();
    let join_order = parse_join_order(newick, &leaf_names);

    // Start with each sequence as its own profile (single-row alignment)
    let mut profiles: Vec<Option<Vec<Vec<u8>>>> =
        sequences.iter().map(|s| Some(vec![s.to_vec()])).collect();

    for (left, right) in &join_order {
        let lp = profiles[*left].take();
        let rp = profiles[*right].take();
        if let (Some(l), Some(r)) = (lp, rp) {
            let merged = merge_profiles(&l, &r, params);
            profiles[*left] = Some(merged);
        }
    }

    // Find the surviving profile
    if let Some(profile) = profiles.iter().flatten().next() {
        return profile.clone();
    }

    // Fallback: shouldn't happen, but return unaligned with gap padding
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    sequences
        .iter()
        .map(|s| {
            let mut v = s.to_vec();
            v.resize(max_len, b'-');
            v
        })
        .collect()
}

/// Parse join order from Newick string.
///
/// Returns pairs of profile indices to merge. We use leaf labels "0", "1", ...
/// to map back to sequence indices. For internal nodes, we track which profile
/// index holds the merged result.
fn parse_join_order(newick: &str, leaf_names: &[String]) -> Vec<(usize, usize)> {
    let n = leaf_names.len();
    if n <= 1 {
        return vec![];
    }

    // Extract leaf labels from the Newick string in the order they appear
    let leaves = extract_newick_leaves(newick);

    // If we can't parse enough leaves, fall back to sequential merging
    if leaves.len() < n {
        return sequential_join_order(n);
    }

    // Map leaf labels to indices
    let label_to_idx: std::collections::HashMap<&str, usize> = leaf_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Build join pairs from the Newick tree structure
    build_joins_from_newick(newick, &label_to_idx)
}

/// Extract leaf names from a Newick string.
fn extract_newick_leaves(newick: &str) -> Vec<String> {
    let mut leaves = Vec::new();
    let mut current = String::new();
    let mut in_branch_length = false;

    for ch in newick.chars() {
        match ch {
            '(' | ',' | ')' | ';' => {
                if !in_branch_length && !current.is_empty() {
                    leaves.push(current.clone());
                }
                current.clear();
                in_branch_length = false;
            }
            ':' => {
                if !current.is_empty() {
                    leaves.push(current.clone());
                    current.clear();
                }
                in_branch_length = true;
            }
            _ if in_branch_length => {}
            _ => current.push(ch),
        }
    }
    if !current.is_empty() && !in_branch_length {
        leaves.push(current);
    }
    leaves
}

/// Build join pairs by parsing Newick parentheses.
fn build_joins_from_newick(
    newick: &str,
    label_to_idx: &std::collections::HashMap<&str, usize>,
) -> Vec<(usize, usize)> {
    let mut joins = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut current = String::new();
    let mut in_branch_length = false;

    for ch in newick.chars() {
        match ch {
            '(' => {
                current.clear();
                in_branch_length = false;
            }
            ':' => {
                if !current.is_empty() {
                    if let Some(&idx) = label_to_idx.get(current.as_str()) {
                        stack.push(idx);
                    }
                    current.clear();
                }
                in_branch_length = true;
            }
            ',' => {
                if !in_branch_length && !current.is_empty() {
                    if let Some(&idx) = label_to_idx.get(current.as_str()) {
                        stack.push(idx);
                    }
                    current.clear();
                }
                in_branch_length = false;
            }
            ')' => {
                if !in_branch_length && !current.is_empty() {
                    if let Some(&idx) = label_to_idx.get(current.as_str()) {
                        stack.push(idx);
                    }
                    current.clear();
                }
                in_branch_length = false;
                // Join the top two items on the stack
                if stack.len() >= 2 {
                    let right = stack.pop().unwrap_or(0);
                    let left = stack.pop().unwrap_or(0);
                    joins.push((left, right));
                    stack.push(left); // merged result lives at left's index
                }
            }
            ';' => {
                if !in_branch_length && !current.is_empty() {
                    if let Some(&idx) = label_to_idx.get(current.as_str()) {
                        stack.push(idx);
                    }
                }
                current.clear();
                in_branch_length = false;
            }
            _ if in_branch_length => {}
            _ => current.push(ch),
        }
    }

    if joins.is_empty() {
        return sequential_join_order(label_to_idx.len());
    }
    joins
}

/// Fallback: merge sequences sequentially (0+1, result+2, result+3, ...).
fn sequential_join_order(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (0, i)).collect()
}

/// Merge two profiles (sets of aligned sequences) by aligning their consensus.
fn merge_profiles(left: &[Vec<u8>], right: &[Vec<u8>], params: &MsaParams) -> Vec<Vec<u8>> {
    let lcons = consensus(left);
    let rcons = consensus(right);

    let r = alignment::smith_waterman(&lcons, &rcons, &params.scoring);
    let (aligned_l, aligned_r) = extend_to_global(&lcons, &rcons, &r);

    // Build column mapping: for each position in the aligned consensus,
    // determine source positions in the original profiles
    let l_map = build_column_map(&aligned_l);
    let r_map = build_column_map(&aligned_r);

    let out_len = aligned_l.len();

    let mut result = Vec::with_capacity(left.len() + right.len());

    for seq in left {
        let mut out = Vec::with_capacity(out_len);
        for &src_idx in &l_map {
            match src_idx {
                Some(i) if i < seq.len() => out.push(seq[i]),
                _ => out.push(b'-'),
            }
        }
        result.push(out);
    }

    for seq in right {
        let mut out = Vec::with_capacity(out_len);
        for &src_idx in &r_map {
            match src_idx {
                Some(i) if i < seq.len() => out.push(seq[i]),
                _ => out.push(b'-'),
            }
        }
        result.push(out);
    }

    result
}

/// Build a column map: for each position in the gapped alignment,
/// return `Some(original_index)` or `None` if it's a gap.
fn build_column_map(gapped: &[u8]) -> Vec<Option<usize>> {
    let mut map = Vec::with_capacity(gapped.len());
    let mut src = 0;
    for &ch in gapped {
        if ch == b'-' {
            map.push(None);
        } else {
            map.push(Some(src));
            src += 1;
        }
    }
    map
}

/// Compute majority-rule consensus of a profile.
fn consensus(profile: &[Vec<u8>]) -> Vec<u8> {
    if profile.is_empty() {
        return Vec::new();
    }
    let len = profile[0].len();
    (0..len)
        .map(|col| {
            let mut counts = [0u32; 5]; // A=0, C=1, G=2, T=3, gap=4
            for seq in profile {
                if col < seq.len() {
                    let idx = match seq[col].to_ascii_uppercase() {
                        b'A' => 0,
                        b'C' => 1,
                        b'G' => 2,
                        b'T' | b'U' => 3,
                        _ => 4,
                    };
                    counts[idx] += 1;
                }
            }
            let best = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| *c)
                .map_or(4, |(i, _)| i);
            match best {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => b'-',
            }
        })
        .filter(|&c| c != b'-')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cons = consensus(&profile);
        assert_eq!(cons, b"ACGT"); // majority at position 2 is G (2 vs 1 T)
    }

    #[test]
    fn extract_leaves_basic() {
        let newick = "(A:0.1,B:0.2);";
        let leaves = extract_newick_leaves(newick);
        assert_eq!(leaves, vec!["A", "B"]);
    }

    #[test]
    fn extract_leaves_nested() {
        let newick = "((A:0.1,B:0.2):0.3,C:0.4);";
        let leaves = extract_newick_leaves(newick);
        assert_eq!(leaves, vec!["A", "B", "C"]);
    }

    #[test]
    fn column_map_gapped() {
        let gapped = b"AC--GT";
        let map = build_column_map(gapped);
        assert_eq!(map, vec![Some(0), Some(1), None, None, Some(2), Some(3)]);
    }

    #[test]
    fn extend_to_global_covers_full() {
        let s1 = b"XXXACGTXXX";
        let s2 = b"ACGT";
        let local = alignment::smith_waterman(s1, s2, &ScoringParams::default());
        let (a, b) = extend_to_global(s1, s2, &local);
        assert_eq!(a.len(), b.len());
        let a_ungapped: Vec<u8> = a.iter().filter(|&&c| c != b'-').copied().collect();
        assert_eq!(a_ungapped, s1.to_vec());
    }
}
