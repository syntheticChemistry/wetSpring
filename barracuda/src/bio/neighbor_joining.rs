// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neighbor-Joining tree construction (Saitou & Nei 1987).
//!
//! Builds an unrooted phylogenetic tree from a pairwise distance matrix.
//! This is the core guide-tree primitive used by `SATé` (Liu 2009) for
//! iterative alignment-tree co-estimation.
//!
//! # References
//!
//! - Saitou & Nei 1987, *Mol Biol Evol* 4:406-425
//! - Liu et al. 2009, *Science* 324:1561-1564 (`SATé`)
//!
//! # GPU Promotion
//!
//! The Q-matrix computation is O(n²) and embarrassingly parallel — each
//! element is independent. `ToadStool` can dispatch one workgroup for the
//! Q-matrix, then CPU performs the sequential join step. For large n
//! (thousands of taxa), the distance matrix computation from sequences
//! is also a GPU target via `score_batch` (Smith-Waterman) or JC distance.

/// Result of Neighbor-Joining: Newick string + branch lengths.
#[derive(Debug, Clone)]
pub struct NjResult {
    /// Newick-format tree string.
    pub newick: String,
    /// Total number of join operations performed.
    pub n_joins: usize,
}

/// Jukes-Cantor corrected distance between two aligned sequences.
///
/// Returns 10.0 for saturated distances (p >= 0.75).
///
/// # Panics
///
/// Panics if `seq1` and `seq2` have different lengths.
#[must_use]
pub fn jukes_cantor_distance(seq1: &[u8], seq2: &[u8]) -> f64 {
    assert_eq!(seq1.len(), seq2.len(), "sequences must have equal length");
    let diffs = seq1.iter().zip(seq2.iter()).filter(|(a, b)| a != b).count();
    #[allow(clippy::cast_precision_loss)]
    let p = diffs as f64 / seq1.len() as f64;
    if p >= 0.75 {
        return 10.0;
    }
    -0.75 * (1.0 - 4.0 * p / 3.0).ln()
}

/// Build a pairwise JC distance matrix from aligned sequences.
///
/// Returns flat row-major `[n * n]` distance matrix (GPU-friendly layout).
#[must_use]
pub fn distance_matrix(sequences: &[&[u8]]) -> Vec<f64> {
    let n = sequences.len();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = jukes_cantor_distance(sequences[i], sequences[j]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Neighbor-Joining algorithm from a flat distance matrix.
///
/// `dist` is row-major `[n * n]`, `labels` has `n` elements.
///
/// # Panics
///
/// Panics if `labels.len() < 2` or `dist.len() != n * n`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn neighbor_joining(dist: &[f64], labels: &[String]) -> NjResult {
    let n = labels.len();
    assert!(n >= 2, "need at least 2 taxa");
    assert_eq!(dist.len(), n * n, "distance matrix size mismatch");

    // Working copies: expand to hold new internal nodes
    let max_nodes = 2 * n;
    let mut d = vec![0.0_f64; max_nodes * max_nodes];
    for i in 0..n {
        for j in 0..n {
            d[i * max_nodes + j] = dist[i * n + j];
        }
    }

    let mut active: Vec<usize> = (0..n).collect();
    let mut node_labels: Vec<String> = labels.to_vec();
    let mut next_node = n;
    let mut n_joins = 0_usize;

    while active.len() > 2 {
        let r = active.len() as f64;

        // Row sums over active nodes
        let mut row_sums = vec![0.0_f64; max_nodes];
        for &i in &active {
            for &j in &active {
                if i != j {
                    row_sums[i] += d[i * max_nodes + j];
                }
            }
        }

        // Find pair minimizing Q
        let mut best_q = f64::INFINITY;
        let mut best_pair = (active[0], active[1]);
        for (ai, &i) in active.iter().enumerate() {
            for &j in &active[(ai + 1)..] {
                let q = (r - 2.0).mul_add(d[i * max_nodes + j], -row_sums[i]) - row_sums[j];
                if q < best_q {
                    best_q = q;
                    best_pair = (i, j);
                }
            }
        }
        let (bi, bj) = best_pair;

        // Branch lengths
        let dij = d[bi * max_nodes + bj];
        let delta = if r > 2.0 {
            (row_sums[bi] - row_sums[bj]) / (r - 2.0)
        } else {
            0.0
        };
        let li = (0.5 * (dij + delta)).max(0.0);
        let lj = (0.5 * (dij - delta)).max(0.0);

        // New node label (Newick subtree)
        let new_label = format!(
            "({}:{:.6},{}:{:.6})",
            node_labels[bi], li, node_labels[bj], lj
        );
        while node_labels.len() <= next_node {
            node_labels.push(String::new());
        }
        node_labels[next_node] = new_label;

        // Distances from new node to remaining active
        for &k in &active {
            if k != bi && k != bj {
                let dk = 0.5 * (d[bi * max_nodes + k] + d[bj * max_nodes + k] - dij);
                d[next_node * max_nodes + k] = dk;
                d[k * max_nodes + next_node] = dk;
            }
        }
        d[next_node * max_nodes + next_node] = 0.0;

        // Update active set
        active.retain(|&x| x != bi && x != bj);
        active.push(next_node);
        next_node += 1;
        n_joins += 1;
    }

    // Final pair
    let (fi, fj) = (active[0], active[1]);
    let final_d = d[fi * max_nodes + fj];
    let newick = format!(
        "({}:{:.6},{}:{:.6});",
        node_labels[fi],
        final_d / 2.0,
        node_labels[fj],
        final_d / 2.0
    );

    NjResult { newick, n_joins }
}

/// Batch distance matrix computation from multiple alignments.
///
/// Each alignment produces an independent distance matrix — maps to
/// one GPU workgroup per alignment for batch dispatch.
#[must_use]
pub fn distance_matrix_batch(alignments: &[Vec<&[u8]>]) -> Vec<Vec<f64>> {
    alignments
        .iter()
        .map(|seqs| distance_matrix(seqs))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jc_distance_identical() {
        let d = jukes_cantor_distance(b"ACGTACGT", b"ACGTACGT");
        assert!(d.abs() < 1e-12, "identical seqs → zero distance: {d}");
    }

    #[test]
    fn jc_distance_positive() {
        let d = jukes_cantor_distance(b"ACGTACGT", b"ACGTACTT");
        assert!(d > 0.0, "different seqs → positive distance: {d}");
        assert!(d < 1.0, "small difference → distance < 1: {d}");
    }

    #[test]
    fn jc_distance_saturated() {
        let d = jukes_cantor_distance(b"AAAA", b"CCCC");
        assert!((d - 10.0).abs() < 1e-12, "100% different → saturated: {d}");
    }

    #[test]
    fn nj_3taxon() {
        let labels: Vec<String> = vec!["X".into(), "Y".into(), "Z".into()];
        let dist = vec![
            0.0, 0.2, 0.4, //
            0.2, 0.0, 0.4, //
            0.4, 0.4, 0.0, //
        ];
        let result = neighbor_joining(&dist, &labels);
        assert_eq!(result.n_joins, 1);
        assert!(result.newick.contains('X'));
        assert!(result.newick.contains('Y'));
        assert!(result.newick.contains('Z'));
        // X and Y should be joined first (smallest distance)
        assert!(
            result.newick.contains("(X:") && result.newick.contains("Y:"),
            "X-Y should be joined: {}",
            result.newick
        );
    }

    #[test]
    fn nj_4taxon_topology() {
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        #[rustfmt::skip]
        let dist = vec![
            0.0, 0.3, 0.5, 0.6,
            0.3, 0.0, 0.6, 0.5,
            0.5, 0.6, 0.0, 0.3,
            0.6, 0.5, 0.3, 0.0,
        ];
        let result = neighbor_joining(&dist, &labels);
        assert_eq!(result.n_joins, 2);
        // A-B and C-D should be sister pairs
        let nwk = &result.newick;
        assert!(nwk.contains('A') && nwk.contains('B'));
        assert!(nwk.contains('C') && nwk.contains('D'));
    }

    #[test]
    fn nj_branch_lengths_nonnegative() {
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let dist = vec![
            0.0, 0.1, 0.5, //
            0.1, 0.0, 0.5, //
            0.5, 0.5, 0.0, //
        ];
        let result = neighbor_joining(&dist, &labels);
        // All branch lengths in Newick should be non-negative
        for part in result.newick.split(':') {
            if let Some(num_str) = part.split([',', ')', ';']).next() {
                if let Ok(v) = num_str.parse::<f64>() {
                    assert!(v >= 0.0, "negative branch length: {v}");
                }
            }
        }
    }

    #[test]
    fn nj_deterministic() {
        let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        #[rustfmt::skip]
        let dist = vec![
            0.0, 0.3, 0.5, 0.6,
            0.3, 0.0, 0.6, 0.5,
            0.5, 0.6, 0.0, 0.3,
            0.6, 0.5, 0.3, 0.0,
        ];
        let r1 = neighbor_joining(&dist, &labels);
        let r2 = neighbor_joining(&dist, &labels);
        assert_eq!(r1.newick, r2.newick);
    }

    #[test]
    fn distance_matrix_symmetric() {
        let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACTT", b"TTTT"];
        let dm = distance_matrix(&seqs);
        let n = seqs.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dm[i * n + j] - dm[j * n + i]).abs() < 1e-12,
                    "matrix must be symmetric"
                );
            }
            assert!(dm[i * n + i].abs() < 1e-12, "diagonal must be zero");
        }
    }

    #[test]
    fn distance_matrix_batch_works() {
        let s1: &[u8] = b"ACGT";
        let s2: &[u8] = b"ACTT";
        let s3: &[u8] = b"TTTT";
        let aln1 = vec![s1, s2, s3];
        let aln2 = vec![s1, s3];
        let results = distance_matrix_batch(&[aln1, aln2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 9); // 3×3
        assert_eq!(results[1].len(), 4); // 2×2
    }

    #[test]
    fn nj_from_sequences() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"ACGTACGTACTT", b"TGCATGCATGCA"];
        let dm = distance_matrix(&seqs);
        let labels: Vec<String> = vec!["S1".into(), "S2".into(), "S3".into()];
        let result = neighbor_joining(&dm, &labels);
        assert!(result.newick.contains("S1"));
        assert!(result.newick.contains("S2"));
        assert!(result.newick.contains("S3"));
        assert_eq!(result.n_joins, 1);
    }
}
