// SPDX-License-Identifier: AGPL-3.0-or-later
//! MSA alignment algorithm: progressive alignment, profile merging, guide tree parsing.
//!
//! Implements guide-tree-based progressive alignment: pairwise scoring, score-to-distance
//! conversion, neighbor-joining guide tree, then progressive merge from leaves to root.

use crate::bio::alignment::{self, AlignmentResult, ScoringParams};

/// Two-sequence case: global extension of Smith-Waterman alignment.
pub(super) fn align_pair(
    sequences: &[&[u8]],
    labels: &[impl AsRef<str>],
    scoring: &ScoringParams,
) -> super::MsaResult {
    let r = alignment::smith_waterman(sequences[0], sequences[1], scoring);
    let (a, b) = extend_to_global(sequences[0], sequences[1], &r);

    let alignment_length = a.len();
    let str_labels: Vec<String> = labels.iter().map(|l| l.as_ref().to_owned()).collect();

    super::MsaResult {
        aligned: vec![a, b],
        labels: str_labels,
        alignment_length,
        guide_tree: String::new(),
    }
}

/// Extend a local alignment to cover full sequences (global-like).
pub(super) fn extend_to_global(
    seq1: &[u8],
    seq2: &[u8],
    local: &AlignmentResult,
) -> (Vec<u8>, Vec<u8>) {
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
pub(super) fn progressive_align(
    sequences: &[&[u8]],
    newick: &str,
    scoring: &ScoringParams,
) -> Vec<Vec<u8>> {
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
            let merged = merge_profiles(&l, &r, scoring);
            profiles[*left] = Some(merged);
        }
    }

    // Find the surviving profile (take ownership to avoid clone)
    if let Some(profile) = profiles.into_iter().flatten().next() {
        return profile;
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
pub(super) fn parse_join_order(newick: &str, leaf_names: &[String]) -> Vec<(usize, usize)> {
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
pub(super) fn extract_newick_leaves(newick: &str) -> Vec<String> {
    let mut leaves = Vec::new();
    let mut current = String::new();
    let mut in_branch_length = false;

    for ch in newick.chars() {
        match ch {
            '(' | ',' | ')' | ';' => {
                if !in_branch_length && !current.is_empty() {
                    leaves.push(std::mem::take(&mut current));
                }
                current.clear();
                in_branch_length = false;
            }
            ':' => {
                if !current.is_empty() {
                    leaves.push(std::mem::take(&mut current));
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
pub(super) fn build_joins_from_newick(
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
pub(super) fn sequential_join_order(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (0, i)).collect()
}

/// Merge two profiles (sets of aligned sequences) by aligning their consensus.
pub(super) fn merge_profiles(
    left: &[Vec<u8>],
    right: &[Vec<u8>],
    scoring: &ScoringParams,
) -> Vec<Vec<u8>> {
    let lcons = consensus(left);
    let rcons = consensus(right);

    let r = alignment::smith_waterman(&lcons, &rcons, scoring);
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
pub(super) fn build_column_map(gapped: &[u8]) -> Vec<Option<usize>> {
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
pub(super) fn consensus(profile: &[Vec<u8>]) -> Vec<u8> {
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
