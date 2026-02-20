// SPDX-License-Identifier: AGPL-3.0-or-later
//! Robinson-Foulds symmetric distance between phylogenetic trees.
//!
//! Computes the unweighted RF distance (Robinson & Foulds 1981) by comparing
//! the bipartition (split) sets induced by each tree's internal edges.
//!
//! # Algorithm
//!
//! 1. For each internal edge, compute the bipartition it induces (the set of
//!    leaf labels on one side).
//! 2. Represent each bipartition as the smaller side's sorted leaf set.
//! 3. RF = |symmetric difference of the two bipartition sets|.
//!
//! # References
//!
//! - Robinson & Foulds 1981. "Comparison of phylogenetic trees."
//!   *Mathematical Biosciences* 53:131-147.
//! - Validated against dendropy (Python) — see `scripts/rf_distance_baseline.py`.

use std::collections::HashSet;

use super::unifrac::PhyloTree;

/// Compute the unweighted Robinson-Foulds distance between two Newick trees.
///
/// Both trees must share the same leaf label set. Returns the number of
/// bipartitions present in one tree but not the other (symmetric difference).
///
/// # Panics
///
/// Does not panic; returns 0 for degenerate cases (≤3 leaves in unrooted
/// representation, where all binary trees have the same topology).
#[must_use]
pub fn rf_distance(tree_a: &PhyloTree, tree_b: &PhyloTree) -> usize {
    let splits_a = bipartitions(tree_a);
    let splits_b = bipartitions(tree_b);
    splits_a.symmetric_difference(&splits_b).count()
}

/// Normalized RF distance: RF / `max_rf`.
///
/// For unrooted binary trees with n leaves, max RF = 2(n-3).
/// Returns 0.0 for degenerate cases (n ≤ 3).
#[must_use]
#[allow(clippy::cast_precision_loss)] // leaf counts are small
pub fn rf_distance_normalized(tree_a: &PhyloTree, tree_b: &PhyloTree) -> f64 {
    let n = leaf_count(tree_a);
    if n <= 3 {
        return 0.0;
    }
    let max_rf = 2 * (n - 3);
    let rf = rf_distance(tree_a, tree_b);
    rf as f64 / max_rf as f64
}

/// Extract the set of bipartitions induced by internal edges of a tree.
///
/// Each bipartition is represented as the sorted smaller side's leaf labels,
/// joined into a canonical string. Trivial splits (single leaf vs rest) are
/// excluded, as they are shared by all trees with the same leaf set.
fn bipartitions(tree: &PhyloTree) -> HashSet<String> {
    let mut splits = HashSet::new();
    let all_leaves = collect_leaves(tree, tree.root);
    let n = all_leaves.len();

    for (idx, node) in tree.nodes.iter().enumerate() {
        if idx == tree.root || node.children.is_empty() {
            continue;
        }
        let subtree_leaves = collect_leaves(tree, idx);
        let size = subtree_leaves.len();

        // Skip trivial splits (single leaf or all-but-one)
        if size <= 1 || size >= n - 1 {
            continue;
        }

        // Canonical form: use the smaller side; ties broken lexicographically
        // so both children of the root map to the same canonical key.
        let complement: Vec<&str> = all_leaves
            .iter()
            .filter(|l| !subtree_leaves.contains(*l))
            .map(String::as_str)
            .collect();

        let mut sub_sorted: Vec<&str> = subtree_leaves.iter().map(String::as_str).collect();
        sub_sorted.sort_unstable();
        let mut comp_sorted = complement;
        comp_sorted.sort_unstable();

        let key = match sub_sorted.len().cmp(&comp_sorted.len()) {
            std::cmp::Ordering::Less => sub_sorted.join("|"),
            std::cmp::Ordering::Greater => comp_sorted.join("|"),
            std::cmp::Ordering::Equal => {
                let sub_key = sub_sorted.join("|");
                let comp_key = comp_sorted.join("|");
                if sub_key <= comp_key {
                    sub_key
                } else {
                    comp_key
                }
            }
        };

        splits.insert(key);
    }
    splits
}

/// Collect all leaf labels under a given node.
fn collect_leaves(tree: &PhyloTree, node_idx: usize) -> Vec<String> {
    let node = &tree.nodes[node_idx];
    if node.children.is_empty() {
        if node.label.is_empty() {
            return vec![];
        }
        return vec![node.label.clone()];
    }
    let mut leaves = Vec::new();
    for &child in &node.children {
        leaves.extend(collect_leaves(tree, child));
    }
    leaves
}

/// Count leaves in a tree.
fn leaf_count(tree: &PhyloTree) -> usize {
    tree.nodes
        .iter()
        .filter(|n| n.children.is_empty() && !n.label.is_empty())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(nwk: &str) -> PhyloTree {
        PhyloTree::from_newick(nwk)
    }

    // ── Analytical cases ─────────────────────────────────────────

    #[test]
    fn identical_4leaf_rf_zero() {
        let t1 = tree("((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);");
        let t2 = tree("((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);");
        assert_eq!(rf_distance(&t1, &t2), 0);
    }

    #[test]
    fn single_nni_4leaf_rf_two() {
        let t1 = tree("((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);");
        let t2 = tree("((A:0.1,C:0.3):0.3,(B:0.2,D:0.4):0.5);");
        assert_eq!(rf_distance(&t1, &t2), 2);
    }

    #[test]
    fn identical_5leaf() {
        let t1 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        let t2 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        assert_eq!(rf_distance(&t1, &t2), 0);
    }

    #[test]
    fn rearranged_5leaf_rf_two() {
        let t1 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        let t2 = tree("(((A:0.1,C:0.3):0.2,B:0.1):0.1,(D:0.2,E:0.3):0.4);");
        assert_eq!(rf_distance(&t1, &t2), 2);
    }

    #[test]
    fn fully_different_5leaf_rf_four() {
        let t1 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        let t2 = tree("(((A:0.1,D:0.2):0.2,E:0.3):0.1,(B:0.1,C:0.3):0.4);");
        assert_eq!(rf_distance(&t1, &t2), 4);
    }

    #[test]
    fn caterpillar_vs_balanced_6leaf() {
        let t1 = tree("(((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1);");
        let t2 = tree("((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);");
        // dendropy confirmed: RF = 2
        assert_eq!(rf_distance(&t1, &t2), 2);
    }

    // ── Degenerate cases ─────────────────────────────────────────

    #[test]
    fn two_leaf_always_zero() {
        let t1 = tree("(A:0.5,B:0.5);");
        let t2 = tree("(A:0.5,B:0.5);");
        assert_eq!(rf_distance(&t1, &t2), 0);
    }

    #[test]
    fn three_leaf_unrooted_always_zero() {
        let t1 = tree("((A:0.1,B:0.2):0.3,C:0.4);");
        let t2 = tree("((A:0.1,C:0.4):0.3,B:0.2);");
        // Unrooted 3-leaf has only 1 topology (star)
        assert_eq!(rf_distance(&t1, &t2), 0);
    }

    // ── Normalized ───────────────────────────────────────────────

    #[test]
    fn normalized_identical_is_zero() {
        let t1 = tree("((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);");
        assert!((rf_distance_normalized(&t1, &t1) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_max_5leaf_is_one() {
        let t1 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        let t2 = tree("(((A:0.1,D:0.2):0.2,E:0.3):0.1,(B:0.1,C:0.3):0.4);");
        assert!((rf_distance_normalized(&t1, &t2) - 1.0).abs() < 1e-12);
    }

    // ── Determinism ──────────────────────────────────────────────

    #[test]
    fn deterministic_across_runs() {
        let t1 = tree("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
        let t2 = tree("(((A:0.1,C:0.3):0.2,B:0.1):0.1,(D:0.2,E:0.3):0.4);");
        let d1 = rf_distance(&t1, &t2);
        let d2 = rf_distance(&t1, &t2);
        assert_eq!(d1, d2);
    }

    // ── Symmetry ─────────────────────────────────────────────────

    #[test]
    fn symmetric() {
        let t1 = tree("((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);");
        let t2 = tree("((A:0.1,C:0.3):0.3,(B:0.2,D:0.4):0.5);");
        assert_eq!(rf_distance(&t1, &t2), rf_distance(&t2, &t1));
    }
}
