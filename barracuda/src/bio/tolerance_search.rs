// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tolerance-based m/z matching for mass spectrometry.
//!
//! Binary search with ppm tolerance — the core algorithm behind
//! PFAS suspect screening (`FindPFAS`) and compound identification.
//!
//! Key operations:
//! - [`find_within_ppm`]: find all entries within +/- ppm of a query
//! - [`find_within_da`]: find all entries within +/- Da of a query
//! - [`screen_pfas_fragments`]: screen MS2 spectra for PFAS fragment patterns

/// Find all indices in a sorted array within +/- ppm tolerance of a query.
///
/// NaN values in `sorted_mz` are safely skipped (they never match any range).
#[must_use]
pub fn find_within_ppm(sorted_mz: &[f64], query: f64, ppm: f64) -> Vec<usize> {
    let tol = query * ppm * 1e-6;
    let lo = query - tol;
    let hi = query + tol;
    find_in_range(sorted_mz, lo, hi)
}

/// Find all indices in a sorted array within +/- absolute Da tolerance.
///
/// NaN values in `sorted_mz` are safely skipped (they never match any range).
#[must_use]
pub fn find_within_da(sorted_mz: &[f64], query: f64, tol_da: f64) -> Vec<usize> {
    let lo = query - tol_da;
    let hi = query + tol_da;
    find_in_range(sorted_mz, lo, hi)
}

/// Binary search + linear scan for values in `[lo, hi]`.
///
/// NaN-safe: `partial_cmp` returns `None` for NaN, which we treat as
/// `Greater` (NaN sorts to the end, matching IEEE 754 totalOrder semantics
/// used by most sort implementations).
fn find_in_range(sorted_mz: &[f64], lo: f64, hi: f64) -> Vec<usize> {
    let start = sorted_mz
        .binary_search_by(|v| v.partial_cmp(&lo).unwrap_or(std::cmp::Ordering::Greater))
        .unwrap_or_else(|i| i);

    let mut matches = Vec::new();
    for (i, &mz) in sorted_mz.iter().enumerate().skip(start) {
        if mz > hi {
            break;
        }
        if mz >= lo {
            matches.push(i);
        }
    }
    matches
}

/// PFAS-characteristic fragment mass differences.
pub struct PfasFragments {
    /// CF2 mass difference (49.99681 Da).
    pub cf2: f64,
    /// C2F4 mass difference (99.99361 Da).
    pub c2f4: f64,
    /// HF mass difference (20.00623 Da).
    pub hf: f64,
}

impl Default for PfasFragments {
    fn default() -> Self {
        Self {
            cf2: 49.99681,
            c2f4: 99.99361,
            hf: 20.00623,
        }
    }
}

/// Result of fragment difference screening on a single spectrum.
#[derive(Debug, Clone)]
pub struct FragmentScreenResult {
    /// Precursor m/z of the screened spectrum.
    pub precursor_mz: f64,
    /// Retention time (minutes).
    pub rt: f64,
    /// Number of CF2 fragment differences found.
    pub cf2_count: usize,
    /// Number of C2F4 fragment differences found.
    pub c2f4_count: usize,
    /// Number of HF fragment differences found.
    pub hf_count: usize,
    /// Total number of characteristic differences.
    pub total_diffs: usize,
}

/// Screen an MS2 spectrum for PFAS-characteristic fragment differences.
///
/// For each pair of fragment ions, check if the mass difference matches
/// CF2 (49.997 Da), C2F4 (99.994 Da), or HF (20.006 Da).
/// This is the core algorithm from `FindPFAS`.
#[must_use]
pub fn screen_pfas_fragments(
    mz_array: &[f64],
    intensity_array: &[f64],
    precursor_mz: f64,
    rt: f64,
    tol_da: f64,
    min_intensity_pct: f64,
) -> Option<FragmentScreenResult> {
    let frags = PfasFragments::default();

    // Filter by relative intensity
    let max_i = intensity_array.iter().copied().fold(0.0_f64, f64::max);
    if max_i <= 0.0 || mz_array.len() < 2 {
        return None;
    }

    let threshold = max_i * min_intensity_pct / 100.0;
    let mzs: Vec<f64> = mz_array
        .iter()
        .zip(intensity_array.iter())
        .filter(|(_, &i)| i >= threshold)
        .map(|(&m, _)| m)
        .collect();

    if mzs.len() < 2 {
        return None;
    }

    let n = mzs.len();
    let mut cf2_count = 0_usize;
    let mut c2f4_count = 0_usize;
    let mut hf_count = 0_usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (mzs[j] - mzs[i]).abs();
            if (diff - frags.cf2).abs() <= tol_da {
                cf2_count += 1;
            }
            if (diff - frags.c2f4).abs() <= tol_da {
                c2f4_count += 1;
            }
            if (diff - frags.hf).abs() <= tol_da {
                hf_count += 1;
            }
        }
    }

    let total = cf2_count + c2f4_count + hf_count;
    if total > 0 {
        Some(FragmentScreenResult {
            precursor_mz,
            rt,
            cf2_count,
            c2f4_count,
            hf_count,
            total_diffs: total,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_within_ppm() {
        let mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let matches = find_within_ppm(&mz, 300.0, 10.0);
        // 10 ppm of 300 = 0.003 Da
        assert_eq!(matches, vec![2]); // Only exact match at index 2
    }

    #[test]
    fn test_find_within_da() {
        let mz = vec![100.0, 149.99, 150.0, 150.01, 200.0];
        let matches = find_within_da(&mz, 150.0, 0.02);
        assert_eq!(matches, vec![1, 2, 3]); // 149.99, 150.0, 150.01
    }

    #[test]
    fn test_find_empty_array() {
        let matches = find_within_ppm(&[], 300.0, 10.0);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_pfas_fragments() {
        let frags = PfasFragments::default();
        assert!((frags.cf2 - 49.99681).abs() < 1e-6);
        assert!((frags.c2f4 - 99.99361).abs() < 1e-6);
        assert!((frags.hf - 20.00623).abs() < 1e-6);
    }

    #[test]
    fn test_screen_with_cf2_pair() {
        // Two peaks separated by CF2 mass
        let mz = vec![100.0, 149.99681];
        let intensity = vec![1000.0, 800.0];
        let result = screen_pfas_fragments(&mz, &intensity, 200.0, 5.0, 0.01, 5.0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.cf2_count, 1);
    }

    #[test]
    fn test_screen_no_match() {
        // Two peaks NOT separated by any PFAS characteristic mass
        let mz = vec![100.0, 110.0];
        let intensity = vec![1000.0, 800.0];
        let result = screen_pfas_fragments(&mz, &intensity, 200.0, 5.0, 0.001, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_screen_below_intensity_threshold() {
        let mz = vec![100.0, 149.99681];
        let intensity = vec![1000.0, 1.0]; // second peak is < 5% of max
        let result = screen_pfas_fragments(&mz, &intensity, 200.0, 5.0, 0.01, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_screen_too_few_peaks() {
        let mz = vec![100.0];
        let intensity = vec![1000.0];
        let result = screen_pfas_fragments(&mz, &intensity, 200.0, 5.0, 0.01, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_nan_safe_ppm() {
        // NaN values in the array should not panic — they are safely skipped
        let mz = vec![100.0, f64::NAN, 300.0, 400.0, 500.0];
        let matches = find_within_ppm(&mz, 300.0, 10.0);
        assert!(matches.contains(&2)); // 300.0 should match
        assert!(!matches.contains(&1)); // NaN should not match
    }

    #[test]
    fn test_nan_safe_da() {
        let mz = vec![100.0, f64::NAN, 150.0];
        let matches = find_within_da(&mz, 150.0, 0.02);
        assert!(matches.contains(&2)); // 150.0 should match
        assert!(!matches.contains(&1)); // NaN should not match
    }

    #[test]
    fn test_screen_all_fragment_types() {
        // Peaks separated by CF2, C2F4, and HF simultaneously
        let mz = vec![100.0, 120.00623, 149.99681, 199.99361];
        let intensity = vec![1000.0, 900.0, 800.0, 700.0];
        let result = screen_pfas_fragments(&mz, &intensity, 300.0, 10.0, 0.01, 5.0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.cf2_count > 0);
        assert!(r.hf_count > 0);
        assert!(r.c2f4_count > 0);
        assert!(r.total_diffs == r.cf2_count + r.c2f4_count + r.hf_count);
    }
}
