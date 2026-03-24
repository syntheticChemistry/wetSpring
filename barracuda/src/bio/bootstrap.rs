// SPDX-License-Identifier: AGPL-3.0-or-later
//! RAWR bootstrap resampling for phylogenetic confidence.
//!
//! Implements the core "Resampling Aligned Weighted Reads" primitive from
//! Wang et al. 2021. This generates bootstrap replicate alignments by
//! resampling columns (sites) with replacement, then computes support
//! values for tree bipartitions.
//!
//! # References
//!
//! - Wang et al. 2021, *Bioinformatics* (ISMB) 37:i111-i119
//!
//! # GPU Promotion
//!
//! Column resampling and per-replicate likelihood are embarrassingly
//! parallel. Each replicate can be processed independently.

use super::felsenstein::{TreeNode, log_likelihood};
use super::gillespie::Lcg64;

/// A multiple sequence alignment stored column-major.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Number of taxa (sequences).
    pub n_taxa: usize,
    /// Number of sites (columns).
    pub n_sites: usize,
    /// Column-major data: `columns[site][taxon]` = state index.
    pub columns: Vec<Vec<usize>>,
}

impl Alignment {
    /// Create from row-major sequences (each row = one taxon).
    #[must_use]
    pub fn from_rows(rows: &[Vec<usize>]) -> Self {
        let n_taxa = rows.len();
        let n_sites = rows.first().map_or(0, Vec::len);
        let columns: Vec<Vec<usize>> = (0..n_sites)
            .map(|site| rows.iter().map(|row| row[site]).collect())
            .collect();
        Self {
            n_taxa,
            n_sites,
            columns,
        }
    }
}

/// Generate a bootstrap replicate by resampling columns with replacement.
#[must_use]
pub fn resample_columns(alignment: &Alignment, rng: &mut Lcg64) -> Alignment {
    let n = alignment.n_sites;
    let mut new_columns = Vec::with_capacity(n);
    for _ in 0..n {
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "precision: bootstrap sample sizes well below 2^53"
        )]
        let idx = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        new_columns.push(alignment.columns[idx].clone());
    }
    Alignment {
        n_taxa: alignment.n_taxa,
        n_sites: n,
        columns: new_columns,
    }
}

/// Build a tree with the replicate alignment (for Felsenstein likelihood).
///
/// This takes a reference tree topology and replaces leaf sequences with
/// the resampled alignment. Returns the log-likelihood under JC69.
#[must_use]
pub fn replicate_log_likelihood(tree: &TreeNode, replicate: &Alignment, mu: f64) -> f64 {
    let remapped = remap_tree(tree, replicate);
    log_likelihood(&remapped, mu)
}

/// Replace leaf states in a tree with columns from an alignment.
fn remap_tree(tree: &TreeNode, aln: &Alignment) -> TreeNode {
    remap_node(tree, aln, &mut 0)
}

fn remap_node(node: &TreeNode, aln: &Alignment, leaf_idx: &mut usize) -> TreeNode {
    match node {
        TreeNode::Leaf { name, .. } => {
            let idx = *leaf_idx;
            *leaf_idx += 1;
            let states: Vec<usize> = aln.columns.iter().map(|col| col[idx]).collect();
            TreeNode::Leaf {
                name: name.clone(),
                states,
            }
        }
        TreeNode::Internal {
            left,
            right,
            left_branch,
            right_branch,
        } => TreeNode::Internal {
            left: Box::new(remap_node(left, aln, leaf_idx)),
            right: Box::new(remap_node(right, aln, leaf_idx)),
            left_branch: *left_branch,
            right_branch: *right_branch,
        },
    }
}

/// Run N bootstrap replicates and return per-replicate log-likelihoods.
///
/// This is the core RAWR primitive. For phylogenetic support, these
/// likelihoods would be compared across candidate trees.
#[must_use]
pub fn bootstrap_likelihoods(
    tree: &TreeNode,
    alignment: &Alignment,
    n_reps: usize,
    mu: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Lcg64::new(seed);
    (0..n_reps)
        .map(|_| {
            let rep = resample_columns(alignment, &mut rng);
            replicate_log_likelihood(tree, &rep, mu)
        })
        .collect()
}

/// Bootstrap support: fraction of replicates where `tree_a` has higher
/// likelihood than `tree_b`.
#[must_use]
pub fn bootstrap_support(
    tree_a: &TreeNode,
    tree_b: &TreeNode,
    alignment: &Alignment,
    n_reps: usize,
    mu: f64,
    seed: u64,
) -> f64 {
    let mut rng = Lcg64::new(seed);
    let mut a_wins = 0_usize;
    for _ in 0..n_reps {
        let rep = resample_columns(alignment, &mut rng);
        let ll_a = replicate_log_likelihood(tree_a, &rep, mu);
        let ll_b = replicate_log_likelihood(tree_b, &rep, mu);
        if ll_a > ll_b {
            a_wins += 1;
        }
    }
    #[expect(clippy::cast_precision_loss)]
    {
        a_wins as f64 / n_reps as f64
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: assertions use expect for clarity"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "test module: usize→f64 for sample mean computation"
)]
mod tests {
    use super::*;
    use crate::bio::felsenstein::encode_dna;

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

    #[test]
    fn resample_preserves_dimensions() {
        let aln = make_alignment();
        let mut rng = Lcg64::new(42);
        let rep = resample_columns(&aln, &mut rng);
        assert_eq!(rep.n_taxa, aln.n_taxa);
        assert_eq!(rep.n_sites, aln.n_sites);
        assert_eq!(rep.columns.len(), aln.columns.len());
    }

    #[test]
    fn bootstrap_likelihoods_correct_count() {
        let aln = make_alignment();
        let tree = make_tree(&aln);
        let lls = bootstrap_likelihoods(&tree, &aln, 10, 1.0, 42);
        assert_eq!(lls.len(), 10);
        for ll in &lls {
            assert!(ll.is_finite(), "LL should be finite: {ll}");
            assert!(*ll < 0.0, "LL should be negative: {ll}");
        }
    }

    #[test]
    fn bootstrap_support_in_range() {
        let aln = make_alignment();
        let tree_a = make_tree(&aln);
        let tree_b = TreeNode::Internal {
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
        let support = bootstrap_support(&tree_a, &tree_b, &aln, 50, 1.0, 42);
        assert!(
            (0.0..=1.0).contains(&support),
            "support should be in [0,1]: {support}"
        );
    }

    #[test]
    fn deterministic_bootstrap() {
        let aln = make_alignment();
        let tree = make_tree(&aln);
        let lls1 = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
        let lls2 = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
        for (a, b) in lls1.iter().zip(&lls2) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn bitwise_deterministic_with_seed() {
        let aln = make_alignment();
        let tree = make_tree(&aln);
        let lls_a = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
        let lls_b = bootstrap_likelihoods(&tree, &aln, 20, 1.0, 42);
        assert_eq!(lls_a.len(), lls_b.len(), "bitwise determinism violated");
        for (a, b) in lls_a.iter().zip(&lls_b) {
            assert_eq!(a.to_bits(), b.to_bits(), "bitwise determinism violated");
        }

        let data: Vec<f64> = (0..30).map(|i| f64::from(i * 3 + 7)).collect();
        let ci_a = barracuda::stats::bootstrap_ci(&data, barracuda::stats::mean, 200, 0.95, 42)
            .expect("bootstrap_ci");
        let ci_b = barracuda::stats::bootstrap_ci(&data, barracuda::stats::mean, 200, 0.95, 42)
            .expect("bootstrap_ci");
        assert_eq!(ci_a.estimate.to_bits(), ci_b.estimate.to_bits(), "bitwise determinism violated");
        assert_eq!(ci_a.lower.to_bits(), ci_b.lower.to_bits(), "bitwise determinism violated");
        assert_eq!(ci_a.upper.to_bits(), ci_b.upper.to_bits(), "bitwise determinism violated");
        assert_eq!(
            ci_a.confidence.to_bits(),
            ci_b.confidence.to_bits(),
            "bitwise determinism violated"
        );
        assert_eq!(
            ci_a.std_error.to_bits(),
            ci_b.std_error.to_bits(),
            "bitwise determinism violated"
        );
        assert_eq!(ci_a.n_bootstrap, ci_b.n_bootstrap, "bitwise determinism violated");
        assert_eq!(ci_a.distribution.len(), ci_b.distribution.len(), "bitwise determinism violated");
        for (a, b) in ci_a.distribution.iter().zip(&ci_b.distribution) {
            assert_eq!(a.to_bits(), b.to_bits(), "bitwise determinism violated");
        }
    }

    mod prop {
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]

            #[test]
            fn bootstrap_ci_contains_mean(
                data in proptest::collection::vec(0.1..100.0_f64, 20..60),
                seed in 0u64..1000,
            ) {
                let sample_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
                let ci = barracuda::stats::bootstrap_ci(
                    &data,
                    barracuda::stats::mean,
                    200,
                    0.95,
                    seed,
                )
                .expect("bootstrap_ci");
                let eps = crate::tolerances::ANALYTICAL_LOOSE;
                prop_assert!(
                    ci.lower - eps <= sample_mean && sample_mean <= ci.upper + eps,
                    "CI [{:.4}, {:.4}] does not contain sample mean {:.4}",
                    ci.lower,
                    ci.upper,
                    sample_mean
                );
            }
        }
    }
}
