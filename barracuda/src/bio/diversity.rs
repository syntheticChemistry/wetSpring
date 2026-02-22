// SPDX-License-Identifier: AGPL-3.0-or-later
//! Alpha and beta diversity metrics for ecological community analysis.
//!
//! Implements Shannon entropy, Simpson index, Chao1 richness estimator,
//! observed features, Pielou's evenness, rarefaction curves, and
//! Bray-Curtis dissimilarity.
//!
//! These are the core metrics from QIIME2/skbio used in Exp002.
//! Pure math — future GPU targets via `BarraCUDA` reduce shaders.

/// Observed features: count of non-zero entries.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)] // ecological counts always fit in f64
pub fn observed_features(counts: &[f64]) -> f64 {
    counts.iter().filter(|&&c| c > 0.0).count() as f64
}

/// Shannon entropy: H = -sum(p\_i \* ln(p\_i)).
///
/// Uses natural log (base e), matching skbio default.
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
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Simpson's diversity index: 1 - sum(p\_i^2).
///
/// Higher values = more diverse (0 to 1).
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
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut sum_p2 = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            sum_p2 += p * p;
        }
    }
    1.0 - sum_p2
}

/// Chao1 richness estimator.
///
/// `Chao1` = `S_obs` + f1\*(f1-1) / (2\*(f2+1))
/// where f1 = singletons, f2 = doubletons.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn chao1(counts: &[f64]) -> f64 {
    let s_obs = observed_features(counts);
    let f1 = counts.iter().filter(|&&c| (c - 1.0).abs() < 0.5).count() as f64;
    let f2 = counts.iter().filter(|&&c| (c - 2.0).abs() < 0.5).count() as f64;

    if f2 > 0.0 {
        s_obs + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))
    } else if f1 > 0.0 {
        s_obs + (f1 * (f1 - 1.0)) / 2.0
    } else {
        s_obs
    }
}

/// Bray-Curtis dissimilarity between two samples.
///
/// `BC` = sum(|a\_i - b\_i|) / sum(a\_i + b\_i).
/// Range \[0, 1\]: 0 = identical, 1 = completely different.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[must_use]
pub fn bray_curtis(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Samples must have same length");

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        numerator += (ai - bi).abs();
        denominator += ai + bi;
    }

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Condensed Bray-Curtis distance matrix (lower triangle, row-major).
///
/// For N samples, returns N*(N-1)/2 values in the order:
/// (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
///
/// Use [`condensed_index`] to look up a pair.
#[must_use]
pub fn bray_curtis_condensed(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);

    for i in 1..n {
        for j in 0..i {
            condensed.push(bray_curtis(&samples[i], &samples[j]));
        }
    }

    condensed
}

/// Look up the index into a condensed distance matrix for pair (i, j).
///
/// # Panics
///
/// Panics if `i == j`.
#[inline]
#[must_use]
pub fn condensed_index(i: usize, j: usize) -> usize {
    assert_ne!(i, j, "diagonal entries are always zero");
    let (a, b) = if i > j { (i, j) } else { (j, i) };
    a * (a - 1) / 2 + b
}

/// Compute Bray-Curtis distance matrix for multiple samples.
/// Returns a flat symmetric matrix (row-major, N x N).
///
/// For large N, prefer [`bray_curtis_condensed`] which uses half the memory.
#[must_use]
pub fn bray_curtis_matrix(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let mut matrix = vec![0.0; n * n];
    let condensed = bray_curtis_condensed(samples);

    for i in 1..n {
        for j in 0..i {
            let idx = condensed_index(i, j);
            matrix[i * n + j] = condensed[idx];
            matrix[j * n + i] = condensed[idx];
        }
    }

    matrix
}

/// Pielou's evenness index: J' = H / ln(S).
///
/// Ratio of observed Shannon entropy to maximum possible entropy.
/// Range \[0, 1\]: 0 = completely uneven, 1 = perfectly even.
///
/// Returns 0.0 when S ≤ 1 (undefined).
#[inline]
#[must_use]
pub fn pielou_evenness(counts: &[f64]) -> f64 {
    let s = observed_features(counts);
    if s <= 1.0 {
        return 0.0;
    }
    let h = shannon(counts);
    h / s.ln()
}

/// Compute a rarefaction curve for a single sample.
///
/// Subsamples the community at each depth in `depths` and computes
/// the expected number of observed species. Uses the hypergeometric
/// formula (exact, no randomness):
///
/// E\[S\_n\] = S - Σ C(N-N\_i, n) / C(N, n)
///
/// where N = total count, N\_i = count of species i, n = subsample depth.
///
/// Falls back to the combinatorial identity in log-space for numerical stability.
///
/// # Arguments
///
/// * `counts` — Species abundance vector.
/// * `depths` — Sequence of subsample depths (integers as `f64`).
///
/// # Returns
///
/// Vector of expected observed species at each depth, in the same order.
#[must_use]
pub fn rarefaction_curve(counts: &[f64], depths: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return vec![0.0; depths.len()];
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let total_n = total as u64;

    depths
        .iter()
        .map(|&depth| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = depth.min(total) as u64;
            if n == 0 {
                return 0.0;
            }
            if n >= total_n {
                return observed_features(counts);
            }

            let mut expected = 0.0;
            for &c in counts {
                if c <= 0.0 {
                    continue;
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ni = c as u64;
                // P(species absent at depth n) = C(N-Ni, n) / C(N, n)
                // In log-space: Σ ln(N-Ni-k+1) - Σ ln(N-k+1) for k=1..n
                // = Σ_{k=0}^{n-1} [ln(N-Ni-k) - ln(N-k)]
                let absent_log = log_hypergeometric_absent(total_n, ni, n);
                expected += 1.0 - absent_log.exp();
            }
            expected
        })
        .collect()
}

/// log(C(N-Ni, n) / C(N, n)) computed in log-space.
fn log_hypergeometric_absent(big_n: u64, ni: u64, n: u64) -> f64 {
    if ni >= big_n {
        return f64::NEG_INFINITY; // species is the whole community
    }
    let remainder = big_n - ni;
    if n > remainder {
        return f64::NEG_INFINITY; // can't draw n without this species
    }

    let mut log_ratio = 0.0_f64;
    for k in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let lnum = ((remainder - k) as f64).ln();
        #[allow(clippy::cast_precision_loss)]
        let lden = ((big_n - k) as f64).ln();
        log_ratio += lnum - lden;
    }
    log_ratio
}

/// Alpha diversity summary for a single sample.
#[derive(Debug, Clone)]
pub struct AlphaDiversity {
    /// Number of non-zero features.
    pub observed: f64,
    /// Shannon entropy (natural log).
    pub shannon: f64,
    /// Simpson diversity index.
    pub simpson: f64,
    /// Chao1 richness estimate.
    pub chao1: f64,
    /// Pielou's evenness.
    pub evenness: f64,
}

/// Compute all alpha diversity metrics for a sample.
#[must_use]
pub fn alpha_diversity(counts: &[f64]) -> AlphaDiversity {
    AlphaDiversity {
        observed: observed_features(counts),
        shannon: shannon(counts),
        simpson: simpson(counts),
        chao1: chao1(counts),
        evenness: pielou_evenness(counts),
    }
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
