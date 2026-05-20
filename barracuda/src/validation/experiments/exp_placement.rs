// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Alamin & Liu 2024 phylogenetic placement (Exp032).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | `alamin2024_placement.py` |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 `scripts/alamin2024_placement.py` |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/alamin2024_placement.py` |
//! | Data | reference tree, 12bp query sequences |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: Python-parity
//!
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use crate::bio::felsenstein::{TreeNode, encode_dna};
use crate::bio::placement::{batch_placement, placement_scan};
use crate::tolerances;
use crate::validation::Validator;

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

#[expect(clippy::cast_precision_loss)]
/// Run the `validate_placement` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let tree = make_reference_tree();

    v.section("── Close to sp1 ──");
    let scan = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    {
        v.check(
            "Edges scanned",
            scan.placements.len() as f64,
            5.0,
            tolerances::EXACT,
        );
        v.check(
            "Best edge matches Python",
            scan.best_edge as f64,
            2.0,
            tolerances::EXACT,
        );
    }
    v.check(
        "Best LL matches Python",
        scan.best_ll,
        -29.977_219_041_460_447,
        tolerances::PHYLO_LIKELIHOOD,
    );
    v.check(
        "Confidence > 0",
        f64::from(u8::from(scan.confidence > 0.0)),
        1.0,
        tolerances::EXACT,
    );

    v.section("── Close to sp3 ──");
    let scan = placement_scan(&tree, "ACTTACGTACGT", 0.05, 1.0);
    {
        v.check(
            "Best edge matches Python",
            scan.best_edge as f64,
            4.0,
            tolerances::EXACT,
        );
    }
    v.check(
        "Best LL matches Python",
        scan.best_ll,
        -29.976_782_512_790_9,
        tolerances::PHYLO_LIKELIHOOD,
    );

    v.section("── Divergent query ──");
    let scan = placement_scan(&tree, "GGGGGGGGGGGG", 0.05, 1.0);
    {
        v.check(
            "Best edge for divergent",
            scan.best_edge as f64,
            0.0,
            tolerances::EXACT,
        );
    }
    v.check(
        "Divergent LL matches Python",
        scan.best_ll,
        -62.894_574_771_654_01,
        tolerances::PHYLO_LIKELIHOOD,
    );

    v.section("── Batch placement ──");
    let queries = vec!["ACGTACGTACGT", "ACTTACGTACGT", "GGGGGGGGGGGG"];
    let results = batch_placement(&tree, &queries, 0.05, 1.0);
    {
        v.check(
            "Batch: 3 queries",
            results.len() as f64,
            3.0,
            tolerances::EXACT,
        );
    }
    v.check(
        "Batch[0] best LL consistent",
        results[0].best_ll,
        -29.977_219_041_460_447,
        tolerances::PHYLO_LIKELIHOOD,
    );

    v.section("── Determinism ──");
    let s1 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    let s2 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    {
        v.check(
            "Deterministic edge",
            f64::from(u8::from(s1.best_edge == s2.best_edge)),
            1.0,
            tolerances::EXACT,
        );
    }
    v.check(
        "Deterministic LL",
        f64::from(u8::from(s1.best_ll.to_bits() == s2.best_ll.to_bits())),
        1.0,
        tolerances::EXACT,
    );

}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_placement");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "placement",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Rust,
        provenance_crate: "validate_placement",
        provenance_date: "2026-05-20",
        description: "Validation: Alamin & Liu 2024 phylogenetic placement (Exp032)",
    },
    run: |v, _ctx| run_as_scenario(v),
};
