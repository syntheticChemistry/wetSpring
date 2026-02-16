//! Alpha and beta diversity metrics for ecological community analysis.
//!
//! Implements Shannon entropy, Simpson index, Chao1 richness estimator,
//! observed features, and Bray-Curtis dissimilarity.
//!
//! These are the core metrics from QIIME2/skbio used in Exp002.
//! Pure math — future GPU targets via barracuda reduce shaders.

/// Observed features: count of non-zero entries.
pub fn observed_features(counts: &[f64]) -> f64 {
    counts.iter().filter(|&&c| c > 0.0).count() as f64
}

/// Shannon entropy: H = -Σ p_i * ln(p_i)
/// Uses natural log (base e), matching skbio default.
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

/// Simpson's diversity index: 1 - Σ p_i²
/// Higher values = more diverse (0 to 1).
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
/// Chao1 = S_obs + f1*(f1-1) / (2*(f2+1))
/// where f1 = singletons, f2 = doubletons.
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
/// BC = Σ |a_i - b_i| / Σ (a_i + b_i)
/// Range [0, 1]: 0 = identical, 1 = completely different.
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

/// Compute Bray-Curtis distance matrix for multiple samples.
/// Returns a flat lower-triangular matrix (row-major).
pub fn bray_curtis_matrix(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let mut matrix = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..i {
            let bc = bray_curtis(&samples[i], &samples[j]);
            matrix[i * n + j] = bc;
            matrix[j * n + i] = bc;
        }
    }

    matrix
}

/// Alpha diversity summary for a single sample.
#[derive(Debug, Clone)]
pub struct AlphaDiversity {
    pub observed: f64,
    pub shannon: f64,
    pub simpson: f64,
    pub chao1: f64,
}

/// Compute all alpha diversity metrics for a sample.
pub fn alpha_diversity(counts: &[f64]) -> AlphaDiversity {
    AlphaDiversity {
        observed: observed_features(counts),
        shannon: shannon(counts),
        simpson: simpson(counts),
        chao1: chao1(counts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_uniform() {
        // 4 equally abundant species: H = ln(4) ≈ 1.386
        let counts = vec![25.0, 25.0, 25.0, 25.0];
        let h = shannon(&counts);
        assert!((h - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_single_species() {
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        assert_eq!(shannon(&counts), 0.0);
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
        assert_eq!(simpson(&counts), 0.0);
    }

    #[test]
    fn test_observed() {
        let counts = vec![10.0, 0.0, 5.0, 0.0, 1.0];
        assert_eq!(observed_features(&counts), 3.0);
    }

    #[test]
    fn test_chao1_no_singletons() {
        let counts = vec![10.0, 20.0, 30.0];
        assert_eq!(chao1(&counts), 3.0); // No singletons, Chao1 = S_obs
    }

    #[test]
    fn test_bray_curtis_identical() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![10.0, 20.0, 30.0];
        assert_eq!(bray_curtis(&a, &b), 0.0);
    }

    #[test]
    fn test_bray_curtis_different() {
        let a = vec![10.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 10.0];
        assert_eq!(bray_curtis(&a, &b), 1.0);
    }
}
