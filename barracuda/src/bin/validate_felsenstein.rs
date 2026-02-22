// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Felsenstein pruning phylogenetic likelihood (Exp029).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Algorithm | Felsenstein 1981, *J Mol Evol* 17:368-376 |
//! | Baseline script | `scripts/felsenstein_pruning_baseline.py` |
//! | Baseline commit | `e4358c5` |
//! | Date | 2026-02-21 |
//! | Exact command | `python3 scripts/felsenstein_pruning_baseline.py` |

use wetspring_barracuda::bio::felsenstein::{
    encode_dna, jc69_prob, log_likelihood, site_log_likelihoods, transition_matrix, TreeNode,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn make_tree_identical() -> TreeNode {
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

fn make_tree_different() -> TreeNode {
    TreeNode::Internal {
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
    }
}

fn make_tree_16s() -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "sp1".into(),
                states: encode_dna("ACGTACGTACGTACGTACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "sp2".into(),
                states: encode_dna("ACGTACTTACGTACGTACGT"),
            }),
            left_branch: 0.05,
            right_branch: 0.05,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "sp3".into(),
            states: encode_dna("ACGTACGTACTTACGTACGT"),
        }),
        left_branch: 0.1,
        right_branch: 0.15,
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp029: Felsenstein Pruning Phylogenetic Likelihood");

    // ── JC69 transition model ───────────────────────────────────────
    v.section("── JC69 transition probabilities ──");
    v.check(
        "P(same, t=0) = 1",
        jc69_prob(0, 0, 0.0, 1.0),
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "P(diff, t=0) = 0",
        jc69_prob(0, 1, 0.0, 1.0),
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "P(same, t=∞) = 0.25",
        jc69_prob(0, 0, 1000.0, 1.0),
        0.25,
        tolerances::JC69_PROBABILITY,
    );

    let mat = transition_matrix(0.5, 1.0);
    for (i, row) in mat.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        v.check(
            &format!("Row {i} sums to 1"),
            sum,
            1.0,
            tolerances::ANALYTICAL_F64,
        );
    }

    // ── Identical sequences tree ────────────────────────────────────
    v.section("── Identical sequences ((A,B):0.2, C:0.3) ──");
    let tree = make_tree_identical();
    let ll = log_likelihood(&tree, 1.0);
    v.check(
        "LL matches Python",
        ll,
        -8.144_952_041_080_28,
        tolerances::PHYLO_LIKELIHOOD,
    );
    let sll = site_log_likelihoods(&tree, 1.0);
    v.check(
        "Per-site sum = total",
        sll.iter().sum::<f64>(),
        ll,
        tolerances::ANALYTICAL_F64,
    );

    // ── Different sequences tree ────────────────────────────────────
    v.section("── Different sequences (AAAA, CCCC, GGGG) ──");
    let tree = make_tree_different();
    let ll = log_likelihood(&tree, 1.0);
    v.check(
        "LL matches Python",
        ll,
        -25.053_907_628_517_04,
        tolerances::PHYLO_LIKELIHOOD,
    );

    // ── Identical > different likelihood ────────────────────────────
    v.check(
        "Identical LL > different LL",
        f64::from(u8::from(log_likelihood(&make_tree_identical(), 1.0) > ll)),
        1.0,
        0.0,
    );

    // ── 16S fragment (20bp) ─────────────────────────────────────────
    v.section("── 16S fragment (20bp, 3 taxa) ──");
    let tree = make_tree_16s();
    let ll = log_likelihood(&tree, 1.0);
    v.check(
        "LL matches Python",
        ll,
        -40.881_169_027_599_25,
        tolerances::PHYLO_LIKELIHOOD,
    );
    let sll = site_log_likelihoods(&tree, 1.0);
    v.check(
        "Site 7 (mismatch)",
        sll[6],
        -5.713_413_379_154_645,
        tolerances::PHYLO_LIKELIHOOD,
    );
    v.check(
        "Site 11 (mismatch)",
        sll[10],
        -4.128_643_256_260_973,
        tolerances::PHYLO_LIKELIHOOD,
    );

    // ── Branch length effect ────────────────────────────────────────
    v.section("── Branch length effect ──");
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
    v.check(
        "Short branches → higher LL",
        f64::from(u8::from(ll_short > ll_long)),
        1.0,
        0.0,
    );

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let ll1 = log_likelihood(&make_tree_identical(), 1.0);
    let ll2 = log_likelihood(&make_tree_identical(), 1.0);
    v.check(
        "Deterministic (bit-exact)",
        f64::from(u8::from(ll1.to_bits() == ll2.to_bits())),
        1.0,
        0.0,
    );

    v.finish();
}
