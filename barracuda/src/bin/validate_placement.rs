// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Alamin & Liu 2024 phylogenetic placement (Exp032).

use wetspring_barracuda::bio::felsenstein::{encode_dna, TreeNode};
use wetspring_barracuda::bio::placement::{batch_placement, placement_scan};
use wetspring_barracuda::validation::Validator;

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

fn main() {
    let mut v = Validator::new("Exp032: Alamin & Liu 2024 Phylogenetic Placement");
    let tree = make_reference_tree();

    v.section("── Close to sp1 ──");
    let scan = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("Edges scanned", scan.placements.len() as f64, 5.0, 0.0);
        v.check("Best edge matches Python", scan.best_edge as f64, 2.0, 0.0);
    }
    v.check(
        "Best LL matches Python",
        scan.best_ll,
        -29.977_219_041_460_447,
        1e-4,
    );
    v.check(
        "Confidence > 0",
        f64::from(u8::from(scan.confidence > 0.0)),
        1.0,
        0.0,
    );

    v.section("── Close to sp3 ──");
    let scan = placement_scan(&tree, "ACTTACGTACGT", 0.05, 1.0);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("Best edge matches Python", scan.best_edge as f64, 4.0, 0.0);
    }
    v.check(
        "Best LL matches Python",
        scan.best_ll,
        -29.976_782_512_790_9,
        1e-4,
    );

    v.section("── Divergent query ──");
    let scan = placement_scan(&tree, "GGGGGGGGGGGG", 0.05, 1.0);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("Best edge for divergent", scan.best_edge as f64, 0.0, 0.0);
    }
    v.check(
        "Divergent LL matches Python",
        scan.best_ll,
        -62.894_574_771_654_01,
        1e-4,
    );

    v.section("── Batch placement ──");
    let queries = vec!["ACGTACGTACGT", "ACTTACGTACGT", "GGGGGGGGGGGG"];
    let results = batch_placement(&tree, &queries, 0.05, 1.0);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("Batch: 3 queries", results.len() as f64, 3.0, 0.0);
    }
    v.check(
        "Batch[0] best LL consistent",
        results[0].best_ll,
        -29.977_219_041_460_447,
        1e-4,
    );

    v.section("── Determinism ──");
    let s1 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    let s2 = placement_scan(&tree, "ACGTACGTACGT", 0.05, 1.0);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check(
            "Deterministic edge",
            f64::from(u8::from(s1.best_edge == s2.best_edge)),
            1.0,
            0.0,
        );
    }
    v.check(
        "Deterministic LL",
        f64::from(u8::from(s1.best_ll.to_bits() == s2.best_ll.to_bits())),
        1.0,
        0.0,
    );

    v.finish();
}
