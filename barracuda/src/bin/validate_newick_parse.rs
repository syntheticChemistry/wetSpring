// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Newick tree parsing — Exp019 Phase 1.
//!
//! Parses the same set of Newick trees used in the Python/dendropy baseline
//! and validates leaf counts, total branch lengths, and leaf label sets.
//! This proves our `PhyloTree::from_newick` parser is correct before using
//! it for phylogenetic computations (RF distance, `UniFrac`, etc.).
//!
//! Follows the `hotSpring` pattern: hardcoded expected values from
//! `newick_parse_python_baseline.json`, explicit pass/fail, exit code 0/1.

use wetspring_barracuda::bio::unifrac::PhyloTree;
use wetspring_barracuda::validation::Validator;

struct TestCase {
    name: &'static str,
    newick: &'static str,
    expected_leaves: usize,
    expected_branch_length: f64,
    expected_labels: &'static [&'static str],
}

const CASES: &[TestCase] = &[
    TestCase {
        name: "simple_4leaf",
        newick: "((A:0.1,B:0.2):0.3,(C:0.3,D:0.4):0.5);",
        expected_leaves: 4,
        expected_branch_length: 1.8,
        expected_labels: &["A", "B", "C", "D"],
    },
    TestCase {
        name: "simple_5leaf",
        newick: "(((A:0.1,B:0.1):0.2,C:0.3):0.1,(D:0.2,E:0.3):0.4);",
        expected_leaves: 5,
        expected_branch_length: 1.7,
        expected_labels: &["A", "B", "C", "D", "E"],
    },
    TestCase {
        name: "balanced_6leaf",
        newick: "((A:0.1,(B:0.1,C:0.1):0.2):0.1,(D:0.1,(E:0.1,F:0.1):0.2):0.1);",
        expected_leaves: 6,
        expected_branch_length: 1.2,
        expected_labels: &["A", "B", "C", "D", "E", "F"],
    },
    TestCase {
        name: "caterpillar_6leaf",
        newick: "(((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1);",
        expected_leaves: 6,
        expected_branch_length: 1.0,
        expected_labels: &["A", "B", "C", "D", "E", "F"],
    },
    TestCase {
        name: "trivial_2leaf",
        newick: "(A:0.5,B:0.5);",
        expected_leaves: 2,
        expected_branch_length: 1.0,
        expected_labels: &["A", "B"],
    },
    TestCase {
        name: "trivial_3leaf",
        newick: "((A:0.1,B:0.2):0.3,C:0.4);",
        expected_leaves: 3,
        expected_branch_length: 1.0,
        expected_labels: &["A", "B", "C"],
    },
    TestCase {
        name: "unequal_branch_7leaf",
        newick: "(((A:0.01,B:0.99):0.5,(C:0.5,D:0.5):0.01):0.1,(E:0.3,(F:0.7,G:0.2):0.4):0.6);",
        expected_leaves: 7,
        expected_branch_length: 4.81,
        expected_labels: &["A", "B", "C", "D", "E", "F", "G"],
    },
    TestCase {
        name: "star_4leaf",
        newick: "(A:0.1,B:0.2,C:0.3,D:0.4);",
        expected_leaves: 4,
        expected_branch_length: 1.0,
        expected_labels: &["A", "B", "C", "D"],
    },
    TestCase {
        name: "deep_caterpillar_8leaf",
        newick:
            "((((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1):0.1,F:0.1):0.1,(G:0.1,H:0.1):0.1);",
        expected_leaves: 8,
        expected_branch_length: 1.4,
        expected_labels: &["A", "B", "C", "D", "E", "F", "G", "H"],
    },
    TestCase {
        name: "zero_length_branches",
        newick: "((A:0.0,B:0.0):0.0,(C:0.0,D:0.0):0.0);",
        expected_leaves: 4,
        expected_branch_length: 0.0,
        expected_labels: &["A", "B", "C", "D"],
    },
];

fn main() {
    let mut v = Validator::new("Exp019 Phase 1: Newick Parsing Validation");

    for tc in CASES {
        v.section(&format!("── {} ──", tc.name));

        let tree = PhyloTree::from_newick(tc.newick);

        v.check_count(
            &format!("{}: leaf count", tc.name),
            tree.n_leaves(),
            tc.expected_leaves,
        );

        v.check(
            &format!("{}: total branch length", tc.name),
            tree.total_branch_length(),
            tc.expected_branch_length,
            1e-10,
        );

        let mut rust_labels: Vec<&str> = tree
            .nodes
            .iter()
            .filter(|n| n.children.is_empty() && !n.label.is_empty())
            .map(|n| n.label.as_str())
            .collect();
        rust_labels.sort_unstable();

        let mut expected_labels: Vec<&str> = tc.expected_labels.to_vec();
        expected_labels.sort_unstable();

        let labels_match = rust_labels == expected_labels;
        v.check(
            &format!("{}: leaf labels match", tc.name),
            f64::from(u8::from(labels_match)),
            1.0,
            0.0,
        );
    }

    v.finish();
}
