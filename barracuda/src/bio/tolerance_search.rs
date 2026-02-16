//! Tolerance-based m/z matching for mass spectrometry.
//!
//! Binary search with ppm tolerance — the core algorithm behind
//! PFAS suspect screening (FindPFAS) and compound identification.
//!
//! Key operations:
//! - `find_within_ppm`: find all entries within ± ppm of a query
//! - `fragment_differences`: find characteristic mass differences in spectra
//! - `pfas_screen`: screen MS2 spectra for PFAS fragment patterns

/// Find all indices in a sorted array within ± ppm tolerance of a query.
pub fn find_within_ppm(sorted_mz: &[f64], query: f64, ppm: f64) -> Vec<usize> {
    let tol = query * ppm * 1e-6;
    let lo = query - tol;
    let hi = query + tol;

    // Binary search for lower bound
    let start = match sorted_mz.binary_search_by(|v| v.partial_cmp(&lo).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };

    let mut matches = Vec::new();
    for i in start..sorted_mz.len() {
        if sorted_mz[i] > hi {
            break;
        }
        matches.push(i);
    }
    matches
}

/// Find all indices in a sorted array within ± absolute tolerance of a query.
pub fn find_within_da(sorted_mz: &[f64], query: f64, tol_da: f64) -> Vec<usize> {
    let lo = query - tol_da;
    let hi = query + tol_da;

    let start = match sorted_mz.binary_search_by(|v| v.partial_cmp(&lo).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };

    let mut matches = Vec::new();
    for i in start..sorted_mz.len() {
        if sorted_mz[i] > hi {
            break;
        }
        matches.push(i);
    }
    matches
}

/// PFAS-characteristic fragment mass differences.
pub struct PfasFragments {
    /// CF2 mass difference (49.99681 Da)
    pub cf2: f64,
    /// C2F4 mass difference (99.99361 Da)
    pub c2f4: f64,
    /// HF mass difference (20.00623 Da)
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
    pub precursor_mz: f64,
    pub rt: f64,
    pub cf2_count: usize,
    pub c2f4_count: usize,
    pub hf_count: usize,
    pub total_diffs: usize,
}

/// Screen an MS2 spectrum for PFAS-characteristic fragment differences.
///
/// For each pair of fragment ions, check if the mass difference matches
/// CF2 (49.997 Da), C2F4 (99.994 Da), or HF (20.006 Da).
/// This is the core algorithm from FindPFAS.
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
    let max_i = intensity_array.iter().cloned().fold(0.0f64, f64::max);
    if max_i <= 0.0 || mz_array.len() < 2 {
        return None;
    }

    let threshold = max_i * min_intensity_pct / 100.0;
    let filtered: Vec<(f64, f64)> = mz_array
        .iter()
        .zip(intensity_array.iter())
        .filter(|(_, &i)| i >= threshold)
        .map(|(&m, &i)| (m, i))
        .collect();

    if filtered.len() < 2 {
        return None;
    }

    let mzs: Vec<f64> = filtered.iter().map(|&(m, _)| m).collect();
    let n = mzs.len();

    let mut cf2_count = 0usize;
    let mut c2f4_count = 0usize;
    let mut hf_count = 0usize;

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
    fn test_pfas_fragments() {
        let frags = PfasFragments::default();
        assert!((frags.cf2 - 49.99681).abs() < 1e-6);
        assert!((frags.c2f4 - 99.99361).abs() < 1e-6);
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
}
