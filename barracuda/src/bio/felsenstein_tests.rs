// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;

fn make_test_tree() -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: encode_dna("ACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: encode_dna("ACGT"),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: encode_dna("ACGT"),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    }
}

#[test]
fn jc69_transition_probabilities() {
    let p_same = jc69_prob(0, 0, 0.0, 1.0);
    assert!(
        (p_same - 1.0).abs() < crate::tolerances::ANALYTICAL_F64,
        "zero branch → identity"
    );
    let p_diff = jc69_prob(0, 1, 0.0, 1.0);
    assert!(
        p_diff.abs() < crate::tolerances::ANALYTICAL_F64,
        "zero branch → no change"
    );

    let p_large = jc69_prob(0, 0, 1000.0, 1.0);
    assert!((p_large - 0.25).abs() < 1e-6, "long branch → uniform");
}

#[test]
fn transition_matrix_rows_sum_to_one() {
    let mat = transition_matrix(0.5, 1.0);
    for row in &mat {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < crate::tolerances::ANALYTICAL_F64,
            "row sum={sum}"
        );
    }
}

#[test]
fn identical_sequences_higher_likelihood() {
    let tree_identical = make_test_tree();

    let tree_different = TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: encode_dna("AAAA"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: encode_dna("CCCC"),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: encode_dna("GGGG"),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    };

    let ll_same = log_likelihood(&tree_identical, 1.0);
    let ll_diff = log_likelihood(&tree_different, 1.0);
    assert!(
        ll_same > ll_diff,
        "identical seqs should have higher likelihood: {ll_same} vs {ll_diff}"
    );
}

#[test]
fn per_site_sums_to_total() {
    let tree = make_test_tree();
    let total = log_likelihood(&tree, 1.0);
    let per_site = site_log_likelihoods(&tree, 1.0);
    let sum: f64 = per_site.iter().sum();
    assert!(
        (total - sum).abs() < crate::tolerances::ANALYTICAL_F64,
        "per-site sum should equal total: {sum} vs {total}"
    );
}

#[test]
fn longer_branch_lower_likelihood_for_identical() {
    let short = TreeNode::Internal {
        left: Box::new(TreeNode::Leaf {
            name: "A".into(),
            states: encode_dna("ACGT"),
        }),
        right: Box::new(TreeNode::Leaf {
            name: "B".into(),
            states: encode_dna("ACGT"),
        }),
        left_branch: 0.01,
        right_branch: 0.01,
    };
    let long = TreeNode::Internal {
        left: Box::new(TreeNode::Leaf {
            name: "A".into(),
            states: encode_dna("ACGT"),
        }),
        right: Box::new(TreeNode::Leaf {
            name: "B".into(),
            states: encode_dna("ACGT"),
        }),
        left_branch: 2.0,
        right_branch: 2.0,
    };
    let ll_short = log_likelihood(&short, 1.0);
    let ll_long = log_likelihood(&long, 1.0);
    assert!(
        ll_short > ll_long,
        "short branches + identical seqs → higher LL: {ll_short} vs {ll_long}"
    );
}

#[test]
fn log_likelihood_finite() {
    let tree = make_test_tree();
    let ll = log_likelihood(&tree, 1.0);
    assert!(ll.is_finite(), "log-likelihood should be finite");
    assert!(ll < 0.0, "log-likelihood should be negative");
}

#[test]
fn deterministic() {
    let tree = make_test_tree();
    let ll1 = log_likelihood(&tree, 1.0);
    let ll2 = log_likelihood(&tree, 1.0);
    assert_eq!(ll1.to_bits(), ll2.to_bits());
}

// ─── FlatTree tests ─────────────────────────────────────────────

#[test]
fn flat_tree_matches_recursive() {
    let tree = make_test_tree();
    let flat = FlatTree::from_tree(&tree, 1.0);
    let ll_recursive = log_likelihood(&tree, 1.0);
    let ll_flat = flat.log_likelihood();
    assert!(
        (ll_recursive - ll_flat).abs() < crate::tolerances::ANALYTICAL_F64,
        "flat should match recursive: {ll_flat} vs {ll_recursive}"
    );
}

#[test]
fn flat_tree_per_site_matches() {
    let tree = make_test_tree();
    let flat = FlatTree::from_tree(&tree, 1.0);
    let recursive = site_log_likelihoods(&tree, 1.0);
    let flat_sites = flat.site_log_likelihoods();
    assert_eq!(recursive.len(), flat_sites.len());
    for (r, f) in recursive.iter().zip(&flat_sites) {
        assert!(
            (r - f).abs() < crate::tolerances::ANALYTICAL_F64,
            "site LL mismatch: recursive={r} flat={f}"
        );
    }
}

#[test]
fn flat_tree_dimensions() {
    let tree = make_test_tree();
    let flat = FlatTree::from_tree(&tree, 1.0);
    assert_eq!(flat.n_leaves, 3);
    assert_eq!(flat.n_internal, 2);
    assert_eq!(flat.n_sites, 4);
    assert_eq!(flat.leaf_states.len(), 4 * 3);
    assert_eq!(flat.trans_left.len(), 2 * 16);
}

#[test]
fn flat_tree_deterministic() {
    let tree = make_test_tree();
    let flat = FlatTree::from_tree(&tree, 1.0);
    let ll1 = flat.log_likelihood();
    let ll2 = flat.log_likelihood();
    assert_eq!(ll1.to_bits(), ll2.to_bits());
}

#[test]
fn flat_tree_longer_alignment() {
    let tree = TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: encode_dna("ACGTACGTACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: encode_dna("ACGTACTTACGT"),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: encode_dna("ACGTACGTACTT"),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    };
    let flat = FlatTree::from_tree(&tree, 1.0);
    let ll_r = log_likelihood(&tree, 1.0);
    let ll_f = flat.log_likelihood();
    assert!(
        (ll_r - ll_f).abs() < crate::tolerances::ANALYTICAL_F64,
        "12-site flat must match recursive: {ll_f} vs {ll_r}"
    );
}
