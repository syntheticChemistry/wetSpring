// SPDX-License-Identifier: AGPL-3.0-or-later
//! `UniFrac` distance computation — unweighted and weighted variants.

use std::collections::HashMap;

use super::flat_tree::FlatTree;
use super::tree::PhyloTree;

/// An abundance table: `sample_id` -> (`leaf_label` -> `count`).
pub type AbundanceTable = HashMap<String, HashMap<String, f64>>;

/// Pairwise `UniFrac` distance matrix result.
#[derive(Debug, Clone)]
pub struct UnifracDistanceMatrix {
    /// Sample identifiers in order.
    pub sample_ids: Vec<String>,
    /// Condensed upper-triangle distances: N*(N-1)/2 values, row-major lower
    /// triangle order matching [`super::super::diversity::bray_curtis_condensed`].
    pub condensed: Vec<f64>,
}

/// Build a dense sample × leaf matrix for GPU dispatch.
///
/// Returns `(matrix, n_samples, n_leaves)` where `matrix[s * n_leaves + l]` =
/// abundance of leaf `l` in sample `s`. Leaf ordering matches `flat_tree.leaf_labels`.
#[must_use]
pub fn to_sample_matrix(
    flat_tree: &FlatTree,
    samples: &AbundanceTable,
) -> (Vec<f64>, usize, usize) {
    let sample_ids: Vec<&String> = samples.keys().collect();
    let n_samples = sample_ids.len();
    let n_leaves = flat_tree.leaf_labels.len();
    let mut matrix = vec![0.0_f64; n_samples * n_leaves];

    for (si, sid) in sample_ids.iter().enumerate() {
        if let Some(abundances) = samples.get(*sid) {
            for (li, label) in flat_tree.leaf_labels.iter().enumerate() {
                if let Some(&val) = abundances.get(label) {
                    matrix[si * n_leaves + li] = val;
                }
            }
        }
    }

    (matrix, n_samples, n_leaves)
}

/// Compute the unweighted `UniFrac` distance between two samples.
///
/// `UniFrac_u` = (`unique` `branch_length`) / (total observed `branch_length`)
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn unweighted_unifrac<S>(
    tree: &PhyloTree,
    sample_a: &HashMap<String, f64, S>,
    sample_b: &HashMap<String, f64, S>,
) -> f64
where
    S: std::hash::BuildHasher,
{
    let n = tree.nodes.len();
    let mut has_a = vec![false; n];
    let mut has_b = vec![false; n];

    for (label, idx) in &tree.leaf_index {
        let in_a = sample_a.get(label).is_some_and(|&v| v > 0.0);
        let in_b = sample_b.get(label).is_some_and(|&v| v > 0.0);
        has_a[*idx] = in_a;
        has_b[*idx] = in_b;
    }

    for i in (0..n).rev() {
        for &child in &tree.nodes[i].children {
            if has_a[child] {
                has_a[i] = true;
            }
            if has_b[child] {
                has_b[i] = true;
            }
        }
    }

    let mut unique_length = 0.0_f64;
    let mut total_length = 0.0_f64;

    for i in 0..n {
        if i == tree.root {
            continue;
        }
        let bl = tree.nodes[i].branch_length;
        if has_a[i] || has_b[i] {
            total_length += bl;
            if has_a[i] != has_b[i] {
                unique_length += bl;
            }
        }
    }

    if total_length > 0.0 {
        unique_length / total_length
    } else {
        0.0
    }
}

/// Compute the weighted `UniFrac` distance between two samples.
///
/// `UniFrac_w` = Σ_`branches` |`p_A` - `p_B`| × `branch_length` /
/// Σ_`branches` max(`p_A`, `p_B`) × `branch_length`
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn weighted_unifrac<S>(
    tree: &PhyloTree,
    sample_a: &HashMap<String, f64, S>,
    sample_b: &HashMap<String, f64, S>,
) -> f64
where
    S: std::hash::BuildHasher,
{
    let total_a: f64 = sample_a.values().sum();
    let total_b: f64 = sample_b.values().sum();

    if total_a == 0.0 || total_b == 0.0 {
        return if total_a == 0.0 && total_b == 0.0 {
            0.0
        } else {
            1.0
        };
    }

    let n = tree.nodes.len();
    let mut prop_a = vec![0.0_f64; n];
    let mut prop_b = vec![0.0_f64; n];

    for (label, &idx) in &tree.leaf_index {
        prop_a[idx] = sample_a.get(label).copied().unwrap_or(0.0) / total_a;
        prop_b[idx] = sample_b.get(label).copied().unwrap_or(0.0) / total_b;
    }

    for i in (0..n).rev() {
        for &child in &tree.nodes[i].children {
            prop_a[i] += prop_a[child];
            prop_b[i] += prop_b[child];
        }
    }

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    for i in 0..n {
        if i == tree.root {
            continue;
        }
        let bl = tree.nodes[i].branch_length;
        let diff = (prop_a[i] - prop_b[i]).abs();
        numerator += bl * diff;
        denominator += bl * prop_a[i].max(prop_b[i]);
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute a pairwise `UniFrac` distance matrix for multiple samples.
///
/// Returns condensed form (upper triangle only) for direct use with
/// [`super::super::pcoa::pcoa`].
#[must_use]
pub fn unifrac_distance_matrix(
    tree: &PhyloTree,
    samples: &AbundanceTable,
    weighted: bool,
) -> UnifracDistanceMatrix {
    let sample_ids: Vec<String> = samples.keys().cloned().collect();
    let n = sample_ids.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);

    for i in 1..n {
        for j in 0..i {
            let sa = &samples[&sample_ids[i]];
            let sb = &samples[&sample_ids[j]];
            let dist = if weighted {
                weighted_unifrac(tree, sa, sb)
            } else {
                unweighted_unifrac(tree, sa, sb)
            };
            condensed.push(dist);
        }
    }

    UnifracDistanceMatrix {
        sample_ids,
        condensed,
    }
}
