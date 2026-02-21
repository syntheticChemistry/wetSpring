// SPDX-License-Identifier: AGPL-3.0-or-later
//! Molecular clock rate estimation for phylogenomics.
//!
//! Provides strict and uncorrelated relaxed clock models for estimating
//! divergence times from branch lengths and fossil calibration constraints.
//! Used in Mateos & Anderson 2023 (sulfur enzymes) and Boden & Anderson 2024
//! (phosphorus enzymes) for dating enzyme evolution through geological time.
//!
//! # Models
//!
//! - **Strict clock**: single substitution rate across all branches
//! - **Relaxed clock (uncorrelated lognormal)**: branch rates drawn from
//!   `LogNormal(ln(mean_rate), sigma)` — the model used by `PhyloBayes`
//!
//! # References
//!
//! - Zuckerkandl & Pauling (1965) — molecular clock hypothesis
//! - Drummond et al. (2006) `PLoS` Biol 4:e88 — relaxed molecular clocks

/// A calibration constraint from fossil or geological evidence.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// Node identifier (e.g., most recent common ancestor of a clade).
    pub node_id: usize,
    /// Minimum age in millions of years (Ma).
    pub min_age_ma: f64,
    /// Maximum age in millions of years (Ma).
    pub max_age_ma: f64,
}

/// Result of a strict-clock rate estimation.
#[derive(Debug, Clone)]
pub struct StrictClockResult {
    /// Estimated substitution rate (substitutions / site / Ma).
    pub rate: f64,
    /// Estimated node ages (indexed by node ID).
    pub node_ages: Vec<f64>,
    /// Whether all calibration constraints are satisfied.
    pub calibrations_satisfied: bool,
}

/// Estimate a strict molecular clock rate from a root age and total tree height.
///
/// Given a rooted tree with total branch-length height `tree_height`
/// (substitutions per site) and a root age `root_age_ma` (millions of years),
/// the strict clock rate is simply `tree_height / root_age_ma`.
///
/// Node ages are computed by subtracting each node's distance-from-root
/// from the root age, scaled by the clock rate.
///
/// # Errors
///
/// Returns `None` if `root_age_ma` is zero or negative.
#[must_use]
pub fn strict_clock(
    branch_lengths: &[f64],
    parent_indices: &[Option<usize>],
    root_age_ma: f64,
    calibrations: &[CalibrationPoint],
) -> Option<StrictClockResult> {
    if root_age_ma <= 0.0 {
        return None;
    }

    let n = branch_lengths.len();
    let tree_height = root_tree_height(branch_lengths, parent_indices);

    if tree_height <= 0.0 {
        return None;
    }

    let rate = tree_height / root_age_ma;

    let mut dist_from_root = vec![0.0; n];
    for i in 0..n {
        if let Some(p) = parent_indices[i] {
            dist_from_root[i] = dist_from_root[p] + branch_lengths[i];
        }
    }

    let node_ages: Vec<f64> = dist_from_root
        .iter()
        .map(|&d| root_age_ma - d / rate)
        .collect();

    let calibrations_satisfied = calibrations.iter().all(|cal| {
        cal.node_id < n
            && node_ages[cal.node_id] >= cal.min_age_ma
            && node_ages[cal.node_id] <= cal.max_age_ma
    });

    Some(StrictClockResult {
        rate,
        node_ages,
        calibrations_satisfied,
    })
}

/// Estimate branch rates under an uncorrelated lognormal relaxed clock.
///
/// Each branch has an independent rate drawn from `LogNormal(ln(mean_rate), sigma)`.
/// Given observed branch lengths and estimated divergence times, the per-branch
/// rate is `branch_length / (parent_age - child_age)`.
#[must_use]
pub fn relaxed_clock_rates(
    branch_lengths: &[f64],
    node_ages: &[f64],
    parent_indices: &[Option<usize>],
) -> Vec<f64> {
    let n = branch_lengths.len();
    let mut rates = vec![0.0; n];

    for i in 0..n {
        if let Some(p) = parent_indices[i] {
            let time_span = node_ages[p] - node_ages[i];
            if time_span > 0.0 {
                rates[i] = branch_lengths[i] / time_span;
            }
        }
    }
    rates
}

/// Coefficient of variation of branch rates — measures clock-likeness.
///
/// CV near 0 → strict clock; CV > 0.5 → substantial rate variation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rate_variation_cv(rates: &[f64]) -> f64 {
    let positive: Vec<f64> = rates.iter().copied().filter(|&r| r > 0.0).collect();
    if positive.len() < 2 {
        return 0.0;
    }
    let n = positive.len() as f64;
    let mean = positive.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = positive.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean
}

/// Height of the tree from root (index 0) to deepest leaf.
fn root_tree_height(branch_lengths: &[f64], parent_indices: &[Option<usize>]) -> f64 {
    let n = branch_lengths.len();
    let mut dist_from_root = vec![0.0; n];
    for i in 0..n {
        if let Some(p) = parent_indices[i] {
            dist_from_root[i] = dist_from_root[p] + branch_lengths[i];
        }
    }
    dist_from_root.iter().copied().fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> (Vec<f64>, Vec<Option<usize>>) {
        // Tree: root(0) -> A(1, bl=0.1), root(0) -> B(2, bl=0.2)
        //       A(1) -> C(3, bl=0.05), A(1) -> D(4, bl=0.05)
        let branch_lengths = vec![0.0, 0.1, 0.2, 0.05, 0.05];
        let parents = vec![None, Some(0), Some(0), Some(1), Some(1)];
        (branch_lengths, parents)
    }

    #[test]
    fn strict_clock_basic() {
        let (bl, parents) = simple_tree();
        let result = strict_clock(&bl, &parents, 100.0, &[]).unwrap();
        assert!(result.rate > 0.0);
        assert!((result.node_ages[0] - 100.0).abs() < 1e-10);
        assert!(result.node_ages[1] < 100.0);
        assert!(result.node_ages[3] < result.node_ages[1]);
        assert!(result.calibrations_satisfied);
    }

    #[test]
    fn strict_clock_calibration_satisfied() {
        let (bl, parents) = simple_tree();
        let cals = vec![CalibrationPoint {
            node_id: 0,
            min_age_ma: 80.0,
            max_age_ma: 120.0,
        }];
        let result = strict_clock(&bl, &parents, 100.0, &cals).unwrap();
        assert!(result.calibrations_satisfied);
    }

    #[test]
    fn strict_clock_calibration_violated() {
        let (bl, parents) = simple_tree();
        let cals = vec![CalibrationPoint {
            node_id: 0,
            min_age_ma: 200.0,
            max_age_ma: 300.0,
        }];
        let result = strict_clock(&bl, &parents, 100.0, &cals).unwrap();
        assert!(!result.calibrations_satisfied);
    }

    #[test]
    fn strict_clock_zero_age() {
        let (bl, parents) = simple_tree();
        assert!(strict_clock(&bl, &parents, 0.0, &[]).is_none());
    }

    #[test]
    fn relaxed_rates() {
        let (bl, parents) = simple_tree();
        let result = strict_clock(&bl, &parents, 100.0, &[]).unwrap();
        let rates = relaxed_clock_rates(&bl, &result.node_ages, &parents);
        assert!(rates.iter().filter(|&&r| r > 0.0).count() >= 2);
    }

    #[test]
    fn cv_strict_is_zero() {
        let rates = vec![0.5, 0.5, 0.5, 0.5];
        let cv = rate_variation_cv(&rates);
        assert!(cv.abs() < 1e-12);
    }

    #[test]
    fn cv_variable_positive() {
        let rates = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let cv = rate_variation_cv(&rates);
        assert!(cv > 0.0);
    }

    #[test]
    fn relaxed_rates_on_strict_tree() {
        let (bl, parents) = simple_tree();
        let result = strict_clock(&bl, &parents, 100.0, &[]).unwrap();
        let rates = relaxed_clock_rates(&bl, &result.node_ages, &parents);
        let positive_rates: Vec<f64> = rates.iter().copied().filter(|&r| r > 0.0).collect();
        let cv = rate_variation_cv(&positive_rates);
        assert!(cv < 1e-10, "Strict tree should have CV ≈ 0, got {cv}");
    }
}
