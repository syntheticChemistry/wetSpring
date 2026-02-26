// SPDX-License-Identifier: AGPL-3.0-or-later
//! Alpha and beta diversity metrics for ecological community analysis.
//!
//! Delegates to `barracuda::stats::diversity` (ToadStool S64 absorption).
//! `chao1` uses `crate::tolerances::CHAO1_COUNT_HALFWIDTH` for singleton/doubleton
//! detection — delegates to upstream which uses the same 0.5 halfwidth.
//!
//! These are the core metrics from `QIIME2`/`skbio` used in Exp002.
//! Pure math — GPU targets via `BarraCuda` reduce shaders in `diversity_gpu`.

/// Observed features: count of non-zero entries.
///
/// Delegates to `barracuda::stats::diversity::observed_features` (S64).
#[inline]
#[must_use]
pub fn observed_features(counts: &[f64]) -> f64 {
    barracuda::stats::observed_features(counts)
}

/// Shannon entropy: H = -sum(p\_i \* ln(p\_i)).
///
/// Uses natural log (base e), matching skbio default.
/// Delegates to `barracuda::stats::diversity::shannon` (S64).
///
/// ```
/// use wetspring_barracuda::bio::diversity;
///
/// let uniform = vec![25.0; 4];
/// let h = diversity::shannon(&uniform);
/// assert!((h - 4.0_f64.ln()).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn shannon(counts: &[f64]) -> f64 {
    barracuda::stats::shannon(counts)
}

/// Simpson's diversity index: 1 - sum(p\_i^2).
///
/// Higher values = more diverse (0 to 1).
/// Delegates to `barracuda::stats::diversity::simpson` (S64).
///
/// ```
/// use wetspring_barracuda::bio::diversity;
///
/// let uniform = vec![100.0; 10];
/// let d = diversity::simpson(&uniform);
/// assert!((d - 0.9).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn simpson(counts: &[f64]) -> f64 {
    barracuda::stats::simpson(counts)
}

/// Chao1 richness estimator.
///
/// Delegates to `barracuda::stats::diversity::chao1` (S64).
#[inline]
#[must_use]
pub fn chao1(counts: &[f64]) -> f64 {
    barracuda::stats::chao1(counts)
}

/// Bray-Curtis dissimilarity between two samples.
///
/// `BC` = sum(|a\_i - b\_i|) / sum(a\_i + b\_i).
/// Range \[0, 1\]: 0 = identical, 1 = completely different.
/// Delegates to `barracuda::stats::diversity::bray_curtis` (S64).
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::bio::diversity;
///
/// let a = vec![10.0, 20.0, 30.0];
/// let b = vec![10.0, 20.0, 30.0];
/// assert!(diversity::bray_curtis(&a, &b).abs() < 1e-10);
///
/// let c = vec![10.0, 0.0, 0.0];
/// let d = vec![0.0, 0.0, 10.0];
/// assert!((diversity::bray_curtis(&c, &d) - 1.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[must_use]
pub fn bray_curtis(a: &[f64], b: &[f64]) -> f64 {
    barracuda::stats::bray_curtis(a, b)
}

/// Condensed Bray-Curtis distance matrix (lower triangle, row-major).
///
/// Delegates to `barracuda::stats::diversity::bray_curtis_condensed` (S64).
#[must_use]
pub fn bray_curtis_condensed(samples: &[Vec<f64>]) -> Vec<f64> {
    barracuda::stats::bray_curtis_condensed(samples)
}

/// Look up the index into a condensed distance matrix for pair (i, j).
///
/// Delegates to `barracuda::stats::diversity::condensed_index` (S64).
#[inline]
#[must_use]
pub fn condensed_index(i: usize, j: usize) -> usize {
    barracuda::stats::condensed_index(i, j)
}

/// Compute Bray-Curtis distance matrix for multiple samples.
/// Returns a flat symmetric matrix (row-major, N x N).
///
/// Delegates to `barracuda::stats::diversity::bray_curtis_matrix` (S64).
#[must_use]
pub fn bray_curtis_matrix(samples: &[Vec<f64>]) -> Vec<f64> {
    barracuda::stats::bray_curtis_matrix(samples)
}

/// Pielou's evenness index: J' = H / ln(S).
///
/// Delegates to `barracuda::stats::diversity::pielou_evenness` (S64).
#[inline]
#[must_use]
pub fn pielou_evenness(counts: &[f64]) -> f64 {
    barracuda::stats::pielou_evenness(counts)
}

/// Compute a rarefaction curve for a single sample.
///
/// Delegates to `barracuda::stats::diversity::rarefaction_curve` (S64).
#[must_use]
pub fn rarefaction_curve(counts: &[f64], depths: &[f64]) -> Vec<f64> {
    barracuda::stats::rarefaction_curve(counts, depths)
}

/// Alpha diversity summary for a single sample.
pub use barracuda::stats::AlphaDiversity;

/// Compute all alpha diversity metrics for a sample.
///
/// Delegates to `barracuda::stats::diversity::alpha_diversity` (S64).
#[must_use]
pub fn alpha_diversity(counts: &[f64]) -> AlphaDiversity {
    barracuda::stats::alpha_diversity(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_uniform() {
        // 4 equally abundant species: H = ln(4)
        let counts = vec![25.0, 25.0, 25.0, 25.0];
        let h = shannon(&counts);
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_single_species() {
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        assert!(shannon(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simpson_uniform() {
        // 4 equally abundant: Simpson = 1 - 4*(0.25)^2 = 0.75
        let counts = vec![25.0, 25.0, 25.0, 25.0];
        assert!((simpson(&counts) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_single() {
        let counts = vec![100.0, 0.0, 0.0];
        assert!(simpson(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn test_observed() {
        let counts = vec![10.0, 0.0, 5.0, 0.0, 1.0];
        assert!((observed_features(&counts) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chao1_no_singletons() {
        let counts = vec![10.0, 20.0, 30.0];
        assert!((chao1(&counts) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bray_curtis_identical() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![10.0, 20.0, 30.0];
        assert!(bray_curtis(&a, &b).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bray_curtis_different() {
        let a = vec![10.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 10.0];
        assert!((bray_curtis(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bray_curtis_symmetry() {
        let a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
        let b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
        let bc_ab = bray_curtis(&a, &b);
        let bc_ba = bray_curtis(&b, &a);
        assert!((bc_ab - bc_ba).abs() < 1e-15);
    }

    #[test]
    fn test_bray_curtis_matrix_diagonal() {
        let samples = vec![vec![10.0, 20.0], vec![15.0, 25.0]];
        let dm = bray_curtis_matrix(&samples);
        // Diagonal should be zero
        assert!(dm[0].abs() < f64::EPSILON);
        assert!(dm[3].abs() < f64::EPSILON);
        // Off-diagonal should be symmetric
        assert!((dm[1] - dm[2]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pielou_uniform() {
        // Perfectly even community: J' = 1.0
        let counts = vec![25.0; 4];
        assert!((pielou_evenness(&counts) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pielou_single_species() {
        // Only one species: S ≤ 1, return 0
        let counts = vec![100.0, 0.0, 0.0];
        assert!((pielou_evenness(&counts)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pielou_uneven() {
        // Very uneven: J' < 1
        let counts = vec![99.0, 1.0];
        let j = pielou_evenness(&counts);
        assert!(j > 0.0 && j < 1.0);
    }

    #[test]
    fn test_rarefaction_at_full_depth() {
        // At full depth, expected species = observed species
        let counts = vec![10.0, 20.0, 30.0, 5.0];
        let total: f64 = counts.iter().sum();
        let curve = rarefaction_curve(&counts, &[total]);
        assert!((curve[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rarefaction_at_zero() {
        let counts = vec![10.0, 20.0, 30.0];
        let curve = rarefaction_curve(&counts, &[0.0]);
        assert!(curve[0].abs() < f64::EPSILON);
    }

    #[test]
    fn test_rarefaction_monotonic() {
        // Expected species should increase with depth
        let counts = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
        let depths: Vec<f64> = (1..=120).map(f64::from).collect();
        let curve = rarefaction_curve(&counts, &depths);

        for i in 1..curve.len() {
            assert!(
                curve[i] >= curve[i - 1] - 1e-10,
                "rarefaction curve not monotonic at depth {}",
                depths[i]
            );
        }
    }

    #[test]
    fn test_rarefaction_empty() {
        let counts = vec![0.0, 0.0, 0.0];
        let curve = rarefaction_curve(&counts, &[10.0, 20.0]);
        assert!(curve.iter().all(|&x| x.abs() < f64::EPSILON));
    }

    #[test]
    fn test_rarefaction_single_species() {
        // Only one species: always observe 1 at any depth > 0
        let counts = vec![100.0, 0.0, 0.0];
        let curve = rarefaction_curve(&counts, &[1.0, 50.0, 100.0]);
        for &val in &curve {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_alpha_diversity_uniform() {
        let counts = vec![25.0; 4];
        let ad = alpha_diversity(&counts);
        assert!((ad.observed - 4.0).abs() < f64::EPSILON);
        assert!((ad.shannon - 4.0_f64.ln()).abs() < 1e-10);
        assert!((ad.simpson - 0.75).abs() < 1e-10);
        assert!((ad.evenness - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_condensed_matrix() {
        let samples = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 10.0, 25.0],
            vec![0.0, 50.0, 0.0],
        ];
        let condensed = bray_curtis_condensed(&samples);
        // 3 samples: 3*(3-1)/2 = 3 pairs
        assert_eq!(condensed.len(), 3);

        // Check against full matrix
        let full = bray_curtis_matrix(&samples);
        assert!((condensed[condensed_index(1, 0)] - full[3]).abs() < f64::EPSILON);
        assert!((condensed[condensed_index(2, 0)] - full[6]).abs() < f64::EPSILON);
        assert!((condensed[condensed_index(2, 1)] - full[7]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_condensed_index_symmetric() {
        assert_eq!(condensed_index(2, 0), condensed_index(0, 2));
        assert_eq!(condensed_index(3, 1), condensed_index(1, 3));
    }

    #[test]
    fn test_chao1_with_singletons() {
        // 3 observed + 2 singletons + 1 doubleton
        let counts = vec![10.0, 20.0, 30.0, 1.0, 1.0, 2.0];
        let c = chao1(&counts);
        // Chao1 = 6 + 2*(2-1)/(2*(1+1)) = 6 + 2/4 = 6.5
        assert!((c - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_all_zeros() {
        // All-zero counts: total=0, should return 0 (not NaN/panic)
        let counts = vec![0.0, 0.0, 0.0];
        assert!(shannon(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simpson_all_zeros() {
        let counts = vec![0.0, 0.0, 0.0];
        assert!(simpson(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bray_curtis_all_zeros() {
        // Both samples all-zero: BC should be 0 (no species to differ on)
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert!(bray_curtis(&a, &b).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chao1_singletons_no_doubletons() {
        // Singletons but zero doubletons: uses bias-corrected formula
        // Chao1 = S_obs + f1*(f1-1) / (2*(f2+1))
        let counts = vec![10.0, 20.0, 1.0, 1.0, 1.0];
        let c = chao1(&counts);
        // S_obs=5, f1=3, f2=0 → Chao1 = 5 + 3*2 / (2*1) = 5 + 3 = 8
        assert!((c - 8.0).abs() < 1e-10);
    }
}
