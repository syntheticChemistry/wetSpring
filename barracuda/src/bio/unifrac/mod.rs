// SPDX-License-Identifier: AGPL-3.0-or-later
//! `UniFrac` distance — phylogeny-weighted beta diversity.
//!
//! Implements unweighted and weighted `UniFrac` metrics (Lozupone & Knight 2005,
//! Lozupone et al. 2007) for comparing microbial communities using phylogenetic
//! information.
//!
//! # Module structure
//!
//! - [`tree`] — Phylogenetic tree types and Newick parser
//! - [`flat_tree`] — GPU-compatible CSR flat tree layout
//! - [`distance`] — Unweighted/weighted `UniFrac` computation and distance matrices
//!
//! # References
//!
//! - Lozupone & Knight. "`UniFrac`: a new phylogenetic method for comparing
//!   microbial communities." Applied and Environmental Microbiology 71,
//!   8228–8235 (2005).
//! - Lozupone et al. "Quantitative and qualitative beta diversity measures lead
//!   to different insights into factors that structure microbial communities."
//!   Applied and Environmental Microbiology 73, 1576–1585 (2007).

pub mod distance;
pub mod flat_tree;
pub mod tree;

pub use distance::{
    AbundanceTable, UnifracDistanceMatrix, to_sample_matrix, unifrac_distance_matrix,
    unweighted_unifrac, weighted_unifrac,
};
pub use flat_tree::FlatTree;
pub use tree::{PhyloTree, TreeNode};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn simple_tree() -> PhyloTree {
        PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6)")
    }

    #[test]
    fn newick_parsing() {
        let tree = simple_tree();
        assert_eq!(tree.n_leaves(), 4);
        assert!(tree.leaf_idx("A").is_some());
        assert!(tree.leaf_idx("B").is_some());
        assert!(tree.leaf_idx("C").is_some());
        assert!(tree.leaf_idx("D").is_some());
        assert!(tree.total_branch_length() > 0.0);
    }

    #[test]
    fn identical_samples_distance_zero() {
        let tree = simple_tree();
        let mut sample: HashMap<String, f64> = HashMap::new();
        sample.insert("A".to_string(), 10.0);
        sample.insert("B".to_string(), 20.0);

        let d = unweighted_unifrac(&tree, &sample, &sample);
        assert!(
            (d - 0.0).abs() < 1e-10,
            "identical samples should have distance 0, got {d}"
        );

        let d = weighted_unifrac(&tree, &sample, &sample);
        assert!(
            (d - 0.0).abs() < 1e-10,
            "identical samples should have distance 0, got {d}"
        );
    }

    #[test]
    fn disjoint_samples_high_distance() {
        let tree = simple_tree();
        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("C".to_string(), 10.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        assert!(
            d > 0.5,
            "disjoint samples should have high distance, got {d}"
        );
        assert!(d <= 1.0);
    }

    #[test]
    fn weighted_unifrac_abundance_sensitive() {
        let tree = simple_tree();

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 100.0);
        sa.insert("B".to_string(), 1.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("A".to_string(), 1.0);
        sb.insert("B".to_string(), 100.0);

        let d = weighted_unifrac(&tree, &sa, &sb);
        assert!(d > 0.0, "different abundance should give non-zero distance");
        assert!(d <= 1.0);
    }

    #[test]
    fn empty_samples() {
        let tree = simple_tree();
        let empty: HashMap<String, f64> = HashMap::new();
        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);

        assert!((unweighted_unifrac(&tree, &empty, &empty) - 0.0).abs() < 1e-10);

        let d = weighted_unifrac(&tree, &sa, &empty);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "one empty should give distance 1.0, got {d}"
        );
    }

    #[test]
    fn distance_matrix_symmetric() {
        let tree = simple_tree();
        let mut samples: AbundanceTable = HashMap::new();

        let mut s1: HashMap<String, f64> = HashMap::new();
        s1.insert("A".to_string(), 10.0);
        s1.insert("B".to_string(), 5.0);
        samples.insert("sample1".to_string(), s1);

        let mut s2: HashMap<String, f64> = HashMap::new();
        s2.insert("C".to_string(), 10.0);
        s2.insert("D".to_string(), 5.0);
        samples.insert("sample2".to_string(), s2);

        let dm = unifrac_distance_matrix(&tree, &samples, false);
        assert_eq!(dm.sample_ids.len(), 2);
        assert_eq!(dm.condensed.len(), 1);
        assert!(dm.condensed[0] > 0.0, "distinct samples should have d > 0");
    }

    #[test]
    fn unweighted_unifrac_partial_overlap() {
        let tree = simple_tree();

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);
        sa.insert("B".to_string(), 10.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("B".to_string(), 10.0);
        sb.insert("C".to_string(), 10.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        assert!(d > 0.0, "partial overlap should give non-zero distance");
        assert!(
            d < 1.0,
            "partial overlap should give less than max distance"
        );
    }

    #[test]
    fn star_tree() {
        let tree = PhyloTree::from_newick("(A:1,B:1,C:1)");
        assert_eq!(tree.n_leaves(), 3);

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 1.0);
        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("B".to_string(), 1.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "completely disjoint on star tree should be 1.0, got {d}"
        );
    }

    #[test]
    fn flat_tree_round_trip() {
        let tree = simple_tree();
        let flat = tree.to_flat_tree();
        assert_eq!(flat.n_nodes as usize, tree.nodes.len());
        assert_eq!(flat.leaf_indices.len(), 4);
        assert_eq!(flat.leaf_labels.len(), 4);

        let restored = flat.to_phylo_tree();
        assert_eq!(restored.n_leaves(), tree.n_leaves());
        for label in &["A", "B", "C", "D"] {
            assert_eq!(
                restored.leaf_idx(label),
                tree.leaf_idx(label),
                "leaf {label} index mismatch"
            );
        }
        assert!(
            (restored.total_branch_length() - tree.total_branch_length()).abs() < 1e-12,
            "branch length mismatch"
        );
    }

    #[test]
    fn flat_tree_unifrac_parity() {
        let tree = simple_tree();
        let flat = tree.to_flat_tree();
        let restored = flat.to_phylo_tree();

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);
        sa.insert("B".to_string(), 5.0);
        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("C".to_string(), 10.0);
        sb.insert("D".to_string(), 5.0);

        let d_orig = unweighted_unifrac(&tree, &sa, &sb);
        let d_flat = unweighted_unifrac(&restored, &sa, &sb);
        assert!(
            (d_orig - d_flat).abs() < 1e-12,
            "unweighted UniFrac mismatch: {d_orig} vs {d_flat}"
        );

        let d_orig_w = weighted_unifrac(&tree, &sa, &sb);
        let d_flat_w = weighted_unifrac(&restored, &sa, &sb);
        assert!(
            (d_orig_w - d_flat_w).abs() < 1e-12,
            "weighted UniFrac mismatch: {d_orig_w} vs {d_flat_w}"
        );
    }

    #[test]
    fn sample_matrix_layout() {
        let tree = simple_tree();
        let flat = tree.to_flat_tree();

        let mut samples: AbundanceTable = HashMap::new();
        let mut s1: HashMap<String, f64> = HashMap::new();
        s1.insert("A".to_string(), 10.0);
        s1.insert("C".to_string(), 5.0);
        samples.insert("sample1".to_string(), s1);
        let mut s2: HashMap<String, f64> = HashMap::new();
        s2.insert("B".to_string(), 8.0);
        s2.insert("D".to_string(), 3.0);
        samples.insert("sample2".to_string(), s2);

        let (matrix, n_samples, n_leaves) = to_sample_matrix(&flat, &samples);
        assert_eq!(n_samples, 2);
        assert_eq!(n_leaves, 4);
        assert_eq!(matrix.len(), 8);
        let total: f64 = matrix.iter().sum();
        assert!((total - 26.0).abs() < 1e-10, "total abundance mismatch");
    }

    #[test]
    fn flat_tree_csr_consistency() {
        let tree = simple_tree();
        let flat = tree.to_flat_tree();

        for i in 0..flat.n_nodes as usize {
            let nc = flat.n_children[i] as usize;
            let off = flat.children_offset[i] as usize;
            assert!(
                off + nc <= flat.children_flat.len(),
                "CSR overflow at node {i}"
            );
            for &child in &flat.children_flat[off..off + nc] {
                assert_eq!(
                    flat.parent[child as usize], i as u32,
                    "parent mismatch for child {child}"
                );
            }
        }
    }
}
