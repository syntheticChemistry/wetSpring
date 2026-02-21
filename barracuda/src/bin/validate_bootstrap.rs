// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Wang 2021 RAWR bootstrap resampling (Exp031).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | wang2021_rawr_bootstrap.py |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 scripts/wang2021_rawr_bootstrap.py |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/wang2021_rawr_bootstrap.py` |
//! | Data | 3-taxon alignment, 100 replicates |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_barracuda::bio::bootstrap::{
    bootstrap_likelihoods, bootstrap_support, resample_columns, Alignment,
};
use wetspring_barracuda::bio::felsenstein::{encode_dna, log_likelihood, TreeNode};
use wetspring_barracuda::bio::gillespie::Lcg64;
use wetspring_barracuda::validation::Validator;

fn make_alignment() -> Alignment {
    let rows = vec![
        encode_dna("ACGTACGTACGT"),
        encode_dna("ACGTACTTACGT"),
        encode_dna("ACGTACGTACTT"),
    ];
    Alignment::from_rows(&rows)
}

fn make_tree(aln: &Alignment) -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: aln.columns.iter().map(|c| c[0]).collect(),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: aln.columns.iter().map(|c| c[1]).collect(),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: aln.columns.iter().map(|c| c[2]).collect(),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    }
}

fn main() {
    let mut v = Validator::new("Exp031: Wang 2021 RAWR Bootstrap Resampling");
    let aln = make_alignment();
    let tree = make_tree(&aln);

    v.section("── Original likelihood ──");
    let orig_ll = log_likelihood(&tree, 1.0);
    v.check(
        "Original LL finite",
        f64::from(u8::from(orig_ll.is_finite())),
        1.0,
        0.0,
    );
    v.check(
        "Original LL negative",
        f64::from(u8::from(orig_ll < 0.0)),
        1.0,
        0.0,
    );

    v.section("── Resampling preserves dimensions ──");
    let mut rng = Lcg64::new(42);
    let rep = resample_columns(&aln, &mut rng);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check(
            "Replicate n_taxa",
            rep.n_taxa as f64,
            aln.n_taxa as f64,
            0.0,
        );
        v.check(
            "Replicate n_sites",
            rep.n_sites as f64,
            aln.n_sites as f64,
            0.0,
        );
    }

    v.section("── Bootstrap likelihoods ──");
    let lls = bootstrap_likelihoods(&tree, &aln, 100, 1.0, 42);
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("100 replicates", lls.len() as f64, 100.0, 0.0);
    }
    let all_finite = lls.iter().all(|ll| ll.is_finite());
    let all_neg = lls.iter().all(|ll| *ll < 0.0);
    v.check("All LLs finite", f64::from(u8::from(all_finite)), 1.0, 0.0);
    v.check("All LLs negative", f64::from(u8::from(all_neg)), 1.0, 0.0);

    let mean: f64 = lls.iter().sum::<f64>() / 100.0;
    v.check("Mean LL near original", (mean - orig_ll).abs(), 0.0, 5.0);

    v.section("── Bootstrap support ──");
    let alt_tree = TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: aln.columns.iter().map(|c| c[0]).collect(),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "C".into(),
                states: aln.columns.iter().map(|c| c[2]).collect(),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "B".into(),
            states: aln.columns.iter().map(|c| c[1]).collect(),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    };
    let support = bootstrap_support(&tree, &alt_tree, &aln, 50, 1.0, 42);
    v.check(
        "Support ∈ [0,1]",
        f64::from(u8::from((0.0..=1.0).contains(&support))),
        1.0,
        0.0,
    );

    v.section("── Determinism ──");
    let lls1 = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
    let lls2 = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
    let bits_match = lls1
        .iter()
        .zip(&lls2)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    v.check(
        "Deterministic (bit-exact)",
        f64::from(u8::from(bits_match)),
        1.0,
        0.0,
    );

    v.finish();
}
