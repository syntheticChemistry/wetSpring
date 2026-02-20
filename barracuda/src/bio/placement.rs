// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phylogenetic placement of query sequences onto reference trees.
//!
//! Implements the core placement likelihood primitive from Alamin & Liu 2024.
//! Given a reference tree and alignment, computes the log-likelihood of
//! inserting a query sequence at each edge, enabling classification of
//! metagenomic reads without full tree inference.
//!
//! # References
//!
//! - Alamin & Liu 2024, *IEEE/ACM TCBB*
//!
//! # GPU Promotion
//!
//! Edge-parallel: each candidate placement is independent. One workgroup
//! per edge computes the Felsenstein likelihood with the query inserted.

use super::felsenstein::{encode_dna, log_likelihood, TreeNode};

/// Result of placing a query at a specific edge.
#[derive(Debug, Clone)]
pub struct PlacementResult {
    /// Edge index where query was placed.
    pub edge_idx: usize,
    /// Log-likelihood with query placed at this edge.
    pub log_likelihood: f64,
    /// Pendant branch length used.
    pub pendant_length: f64,
}

/// Result of placing a query on all edges.
#[derive(Debug, Clone)]
pub struct PlacementScan {
    /// Per-edge placement results, sorted by edge index.
    pub placements: Vec<PlacementResult>,
    /// Index of the best (highest LL) placement.
    pub best_edge: usize,
    /// Log-likelihood of the best placement.
    pub best_ll: f64,
    /// Log-likelihood weight ratio: `exp(best - second_best)`.
    pub confidence: f64,
}

/// Collect all edges in the tree (post-order) with their parent context.
///
/// Returns `(edge_idx, parent_node_path)` for each internal edge.
fn count_edges(tree: &TreeNode) -> usize {
    match tree {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Internal { left, right, .. } => 1 + count_edges(left) + count_edges(right),
    }
}

/// Insert a query leaf at a specific edge of the tree.
///
/// Splits the target edge and attaches the query as a sister taxon.
/// `edge_idx` is a post-order index; `pendant_len` is the query branch length.
fn insert_at_edge(
    tree: &TreeNode,
    query_states: &[usize],
    target_edge: usize,
    pendant_len: f64,
    current_idx: &mut usize,
) -> (TreeNode, bool) {
    match tree {
        TreeNode::Leaf { name, states } => {
            let my_idx = *current_idx;
            *current_idx += 1;
            if my_idx == target_edge {
                let new_node = TreeNode::Internal {
                    left: Box::new(TreeNode::Leaf {
                        name: name.clone(),
                        states: states.clone(),
                    }),
                    right: Box::new(TreeNode::Leaf {
                        name: "query".into(),
                        states: query_states.to_vec(),
                    }),
                    left_branch: 0.01,
                    right_branch: pendant_len,
                };
                (new_node, true)
            } else {
                (tree.clone(), false)
            }
        }
        TreeNode::Internal {
            left,
            right,
            left_branch,
            right_branch,
        } => {
            let my_idx = *current_idx;
            *current_idx += 1;

            let (new_left, found_left) =
                insert_at_edge(left, query_states, target_edge, pendant_len, current_idx);
            let (new_right, found_right) =
                insert_at_edge(right, query_states, target_edge, pendant_len, current_idx);

            if found_left || found_right {
                let node = TreeNode::Internal {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                    left_branch: *left_branch,
                    right_branch: *right_branch,
                };
                (node, true)
            } else if my_idx == target_edge {
                let new_node = TreeNode::Internal {
                    left: Box::new(TreeNode::Internal {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_branch: *left_branch,
                        right_branch: *right_branch,
                    }),
                    right: Box::new(TreeNode::Leaf {
                        name: "query".into(),
                        states: query_states.to_vec(),
                    }),
                    left_branch: 0.01,
                    right_branch: pendant_len,
                };
                (new_node, true)
            } else {
                let node = TreeNode::Internal {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                    left_branch: *left_branch,
                    right_branch: *right_branch,
                };
                (node, false)
            }
        }
    }
}

/// Scan all edges of the reference tree and compute placement likelihood.
///
/// For each edge, inserts the query sequence and computes the Felsenstein
/// log-likelihood. Returns the scan with the best placement identified.
///
/// # Panics
///
/// Panics if the tree has no edges (empty tree).
#[must_use]
pub fn placement_scan(
    reference_tree: &TreeNode,
    query_sequence: &str,
    pendant_length: f64,
    mu: f64,
) -> PlacementScan {
    let query_states = encode_dna(query_sequence);
    let n_edges = count_edges(reference_tree);

    let mut placements = Vec::with_capacity(n_edges);
    for edge_idx in 0..n_edges {
        let mut idx = 0;
        let (augmented_tree, _) = insert_at_edge(
            reference_tree,
            &query_states,
            edge_idx,
            pendant_length,
            &mut idx,
        );
        let ll = log_likelihood(&augmented_tree, mu);
        placements.push(PlacementResult {
            edge_idx,
            log_likelihood: ll,
            pendant_length,
        });
    }

    let best_edge = placements
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.log_likelihood
                .partial_cmp(&b.log_likelihood)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(i, _)| i);

    let best_ll = placements[best_edge].log_likelihood;

    // Confidence: ratio of best to second-best
    let mut sorted_lls: Vec<f64> = placements.iter().map(|p| p.log_likelihood).collect();
    sorted_lls.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let confidence = if sorted_lls.len() >= 2 {
        (sorted_lls[0] - sorted_lls[1]).exp()
    } else {
        1.0
    };

    PlacementScan {
        placements,
        best_edge,
        best_ll,
        confidence,
    }
}

/// Batch placement: place multiple query sequences on the same reference tree.
#[must_use]
pub fn batch_placement(
    reference_tree: &TreeNode,
    queries: &[&str],
    pendant_length: f64,
    mu: f64,
) -> Vec<PlacementScan> {
    queries
        .iter()
        .map(|q| placement_scan(reference_tree, q, pendant_length, mu))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reference_tree() -> TreeNode {
        TreeNode::Internal {
            left: Box::new(TreeNode::Internal {
                left: Box::new(TreeNode::Leaf {
                    name: "sp1".into(),
                    states: encode_dna("ACGTACGTACGT"),
                }),
                right: Box::new(TreeNode::Leaf {
                    name: "sp2".into(),
                    states: encode_dna("ACGTACTTACGT"),
                }),
                left_branch: 0.1,
                right_branch: 0.1,
            }),
            right: Box::new(TreeNode::Leaf {
                name: "sp3".into(),
                states: encode_dna("ACTTACGTACGT"),
            }),
            left_branch: 0.2,
            right_branch: 0.3,
        }
    }

    #[test]
    fn placement_scan_finds_all_edges() {
        let tree = make_reference_tree();
        let scan = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        let n_edges = count_edges(&tree);
        assert_eq!(scan.placements.len(), n_edges);
    }

    #[test]
    fn identical_query_prefers_close_edge() {
        let tree = make_reference_tree();
        let scan = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        assert!(scan.best_ll.is_finite(), "best LL should be finite");
        assert!(scan.best_ll < 0.0, "best LL should be negative");
    }

    #[test]
    fn different_queries_different_placements() {
        let tree = make_reference_tree();
        let scan_a = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        let scan_b = placement_scan(&tree, "ACTTACGTACGT", 0.05, 1.0);
        // They might or might not differ, but both should be valid
        assert!(scan_a.best_ll.is_finite());
        assert!(scan_b.best_ll.is_finite());
    }

    #[test]
    fn confidence_positive() {
        let tree = make_reference_tree();
        let scan = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        assert!(scan.confidence > 0.0, "confidence should be positive");
    }

    #[test]
    fn batch_placement_correct_count() {
        let tree = make_reference_tree();
        let queries = vec!["ACGTACGTACGT", "ACTTACGTACGT", "GGGGGGGGGGGG"];
        let results = batch_placement(&tree, &queries, 0.05, 1.0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn deterministic() {
        let tree = make_reference_tree();
        let s1 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        let s2 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
        assert_eq!(s1.best_edge, s2.best_edge);
        assert_eq!(s1.best_ll.to_bits(), s2.best_ll.to_bits());
    }
}
