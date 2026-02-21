// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation binary: Robinson-Foulds tree distance (Exp021).
//!
//! Compares Rust RF distance against dendropy (Python) ground truth
//! on synthetic Newick trees with known analytical distances.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline script | `scripts/rf_distance_baseline.py` |
//! | Baseline output | `experiments/results/021_rf_baseline/rf_python_baseline.json` |
//! | Python library | dendropy 5.0.8 |
//! | Reference | Robinson & Foulds 1981, Math Biosci 53:131-147 |
//! | Date | 2026-02-19 |
//! | Exact command | `python3 scripts/rf_distance_baseline.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |

use wetspring_barracuda::bio::robinson_foulds::{rf_distance, rf_distance_normalized};
use wetspring_barracuda::bio::unifrac::PhyloTree;
use wetspring_barracuda::validation::Validator;

/// Test cases matching `scripts/rf_distance_baseline.py` exactly.
/// `(name, newick_a, newick_b, expected_rf_from_dendropy)`
const CASES: &[(&str, &str, &str, usize)] = &[
    (
        "identical_4leaf",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        0,
    ),
    (
        "single_nni_4leaf",
        "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        "((A:0.1,C:0.3):0.3,(B:0.2,D:0.4):0.5);",
        2,
    ),
    (
        "identical_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        0,
    ),
    (
        "rearranged_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,C:0.3):0.2,B:0.1):0.1,(D:0.2,E:0.3):0.4);",
        2,
    ),
    (
        "fully_different_5leaf",
        "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        "(((A:0.1,D:0.2):0.2,E:0.3):0.1,(B:0.1,C:0.3):0.4);",
        4,
    ),
    (
        "identical_6leaf",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        0,
    ),
    (
        "caterpillar_vs_balanced_6leaf",
        "(((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1);",
        "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        2,
    ),
    ("two_leaf", "(A:0.5,B:0.5);", "(A:0.5,B:0.5);", 0),
    (
        "three_leaf_identical",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        0,
    ),
    (
        "three_leaf_rearranged",
        "((A:0.1,B:0.2):0.3,C:0.4);",
        "((A:0.1,C:0.4):0.3,B:0.2);",
        0,
    ),
];

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("validate_rf_distance (Exp021: Robinson-Foulds)");

    // ── Python-matched RF distances ─────────────────────────────
    v.section("── RF distance vs dendropy baseline ──");

    for &(name, nwk_a, nwk_b, expected) in CASES {
        let tree_a = PhyloTree::from_newick(nwk_a);
        let tree_b = PhyloTree::from_newick(nwk_b);
        let actual = rf_distance(&tree_a, &tree_b);
        v.check_count(&format!("RF({name})"), actual, expected);
    }

    // ── Symmetry ────────────────────────────────────────────────
    v.section("── Symmetry checks ──");

    for &(name, nwk_a, nwk_b, _) in CASES {
        let tree_a = PhyloTree::from_newick(nwk_a);
        let tree_b = PhyloTree::from_newick(nwk_b);
        let d_ab = rf_distance(&tree_a, &tree_b);
        let d_ba = rf_distance(&tree_b, &tree_a);
        v.check_count(&format!("symmetric({name})"), d_ab, d_ba);
    }

    // ── Normalized RF ───────────────────────────────────────────
    v.section("── Normalized RF ──");

    let t5a = PhyloTree::from_newick("(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);");
    let t5b = PhyloTree::from_newick("(((A:0.1,D:0.2):0.2,E:0.3):0.1,(B:0.1,C:0.3):0.4);");
    v.check(
        "normalized_identical",
        rf_distance_normalized(&t5a, &t5a),
        0.0,
        wetspring_barracuda::tolerances::ANALYTICAL_F64,
    );
    v.check(
        "normalized_max_5leaf",
        rf_distance_normalized(&t5a, &t5b),
        1.0,
        wetspring_barracuda::tolerances::ANALYTICAL_F64,
    );

    // ── Determinism ─────────────────────────────────────────────
    v.section("── Determinism ──");

    let tree_a = PhyloTree::from_newick(CASES[1].1);
    let tree_b = PhyloTree::from_newick(CASES[1].2);
    let d1 = rf_distance(&tree_a, &tree_b);
    let d2 = rf_distance(&tree_a, &tree_b);
    v.check_count("deterministic_rerun", d1, d2);

    v.finish();
}
