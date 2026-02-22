// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D signal processing for chromatographic peak detection.
//!
//! Replaces `scipy.signal.find_peaks` for asari-style LC-MS feature extraction.
//! Detects local maxima with configurable prominence, width, and height filters.
//!
//! # Algorithm
//!
//! 1. Identify all local maxima (y\[i\] > y\[i-1\] AND y\[i\] > y\[i+1\])
//! 2. Compute prominence for each peak (height above the higher of the two
//!    surrounding valleys)
//! 3. Compute peak width at half-prominence via linear interpolation
//! 4. Filter by user-specified thresholds
//!
//! # References
//!
//! - `scipy.signal.find_peaks` (`SciPy` 1.12)
//! - Li, S. et al. "Trackable and scalable LC-MS metabolomics data processing
//!   using asari." Nature Communications 14, 4113 (2023).

/// Detected peak with its properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    /// Index of the peak in the input array.
    pub index: usize,
    /// Height (value at the peak index).
    pub height: f64,
    /// Prominence: height above the higher surrounding valley.
    pub prominence: f64,
    /// Left valley index (base of prominence calculation).
    pub left_base: usize,
    /// Right valley index (base of prominence calculation).
    pub right_base: usize,
    /// Width at half-prominence (in index units, may be fractional).
    pub width: f64,
    /// Left interpolated position of the half-prominence crossing.
    pub left_ips: f64,
    /// Right interpolated position of the half-prominence crossing.
    pub right_ips: f64,
}

/// Configuration for peak detection.
#[derive(Debug, Clone)]
pub struct PeakParams {
    /// Minimum peak height (absolute). Default: no filter.
    pub min_height: Option<f64>,
    /// Minimum prominence. Default: no filter.
    pub min_prominence: Option<f64>,
    /// Minimum width at half-prominence (index units). Default: no filter.
    pub min_width: Option<f64>,
    /// Maximum width at half-prominence (index units). Default: no filter.
    pub max_width: Option<f64>,
    /// Minimum horizontal distance between peaks (index units). Default: 1.
    pub distance: usize,
}

impl Default for PeakParams {
    fn default() -> Self {
        Self {
            min_height: None,
            min_prominence: None,
            min_width: None,
            max_width: None,
            distance: 1,
        }
    }
}

/// Detect peaks in a 1D signal with configurable filtering.
///
/// Equivalent to `scipy.signal.find_peaks` with `height`, `prominence`,
/// `width`, and `distance` parameters.
///
/// # Arguments
///
/// * `data` — 1D signal (e.g., chromatogram intensities).
/// * `params` — Peak detection parameters.
///
/// # Returns
///
/// Vector of detected [`Peak`] structs, sorted by index.
#[must_use]
pub fn find_peaks(data: &[f64], params: &PeakParams) -> Vec<Peak> {
    if data.len() < 3 {
        return vec![];
    }

    // 1. Find all local maxima
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            candidates.push(i);
        }
    }

    // Also handle flat-topped peaks: find runs where consecutive values are equal
    // and both edges are strictly lower. Pick the midpoint.
    #[allow(clippy::float_cmp)] // exact equality is intentional for plateau detection
    let eq = |a: f64, b: f64| a == b;

    let mut i = 1;
    while i < data.len() - 1 {
        if eq(data[i], data[i - 1]) {
            i += 1;
            continue;
        }
        if data[i] > data[i - 1] {
            let start = i;
            while i + 1 < data.len() && eq(data[i + 1], data[i]) {
                i += 1;
            }
            if i + 1 < data.len() && data[i + 1] < data[i] {
                let mid = usize::midpoint(start, i);
                if !candidates.contains(&mid) {
                    candidates.push(mid);
                }
            }
        }
        i += 1;
    }

    candidates.sort_unstable();
    candidates.dedup();

    // 2. Height filter
    if let Some(min_h) = params.min_height {
        candidates.retain(|&idx| data[idx] >= min_h);
    }

    // 3. Compute prominence and width for each candidate
    let mut peaks: Vec<Peak> = candidates
        .iter()
        .map(|&idx| compute_peak_properties(data, idx))
        .collect();

    // 4. Prominence filter
    if let Some(min_p) = params.min_prominence {
        peaks.retain(|p| p.prominence >= min_p);
    }

    // 5. Width filter
    if let Some(min_w) = params.min_width {
        peaks.retain(|p| p.width >= min_w);
    }
    if let Some(max_w) = params.max_width {
        peaks.retain(|p| p.width <= max_w);
    }

    // 6. Distance filter (keep highest peak within each window)
    if params.distance > 1 {
        peaks = filter_by_distance(peaks, data, params.distance);
    }

    peaks
}

/// Compute prominence and width for a single peak.
fn compute_peak_properties(data: &[f64], peak_idx: usize) -> Peak {
    let height = data[peak_idx];

    // Find left valley: scan left to find minimum before a higher peak or boundary
    let mut left_min = height;
    let mut left_base = peak_idx;
    for i in (0..peak_idx).rev() {
        if data[i] < left_min {
            left_min = data[i];
            left_base = i;
        }
        if data[i] > height {
            break;
        }
    }

    // Find right valley: scan right to find minimum before a higher peak or boundary
    let mut right_min = height;
    let mut right_base = peak_idx;
    for (i, &val) in data.iter().enumerate().skip(peak_idx + 1) {
        if val < right_min {
            right_min = val;
            right_base = i;
        }
        if val > height {
            break;
        }
    }

    let reference = left_min.max(right_min);
    let prominence = height - reference;

    // Width at half-prominence
    let half_height = height - prominence / 2.0;
    let left_ips = interpolate_crossing(data, peak_idx, half_height, true);
    let right_ips = interpolate_crossing(data, peak_idx, half_height, false);
    let width = right_ips - left_ips;

    Peak {
        index: peak_idx,
        height,
        prominence,
        left_base,
        right_base,
        width,
        left_ips,
        right_ips,
    }
}

/// Linearly interpolate where the signal crosses a threshold.
#[allow(clippy::cast_precision_loss)] // indices are tiny relative to f64 range
fn interpolate_crossing(data: &[f64], peak_idx: usize, threshold: f64, go_left: bool) -> f64 {
    if go_left {
        for i in (1..=peak_idx).rev() {
            if data[i - 1] <= threshold && data[i] > threshold {
                let frac = (threshold - data[i - 1]) / (data[i] - data[i - 1]);
                return (i - 1) as f64 + frac;
            }
        }
        0.0
    } else {
        for i in peak_idx..data.len().saturating_sub(1) {
            if data[i] > threshold && data[i + 1] <= threshold {
                let frac = (data[i] - threshold) / (data[i] - data[i + 1]);
                return i as f64 + frac;
            }
        }
        (data.len() - 1) as f64
    }
}

/// Keep only the highest peak within each `distance` window.
fn filter_by_distance(mut peaks: Vec<Peak>, data: &[f64], distance: usize) -> Vec<Peak> {
    // Sort by height descending (highest priority first)
    peaks.sort_by(|a, b| {
        data[b.index]
            .partial_cmp(&data[a.index])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![true; peaks.len()];
    for i in 0..peaks.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..peaks.len() {
            if !keep[j] {
                continue;
            }
            let dist = peaks[i].index.abs_diff(peaks[j].index);
            if dist < distance {
                keep[j] = false;
            }
        }
    }

    let mut result: Vec<Peak> = peaks
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| p)
        .collect();

    result.sort_by_key(|p| p.index);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_peak() {
        let data = [0.0, 1.0, 3.0, 1.0, 0.0];
        let peaks = find_peaks(&data, &PeakParams::default());
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 2);
        assert!((peaks[0].height - 3.0).abs() < f64::EPSILON);
        assert!((peaks[0].prominence - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn two_peaks() {
        let data = [0.0, 2.0, 0.0, 3.0, 0.0];
        let peaks = find_peaks(&data, &PeakParams::default());
        assert_eq!(peaks.len(), 2);
        assert_eq!(peaks[0].index, 1);
        assert_eq!(peaks[1].index, 3);
    }

    #[test]
    fn height_filter() {
        let data = [0.0, 1.0, 0.0, 5.0, 0.0];
        let params = PeakParams {
            min_height: Some(2.0),
            ..PeakParams::default()
        };
        let peaks = find_peaks(&data, &params);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 3);
    }

    #[test]
    fn prominence_filter() {
        // Peak at 3 has low prominence because the valley between is shallow
        let data = [0.0, 5.0, 4.0, 4.5, 0.0];
        let params = PeakParams {
            min_prominence: Some(2.0),
            ..PeakParams::default()
        };
        let peaks = find_peaks(&data, &params);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 1);
    }

    #[test]
    fn distance_filter() {
        let data = [0.0, 3.0, 1.0, 2.0, 0.0];
        let params = PeakParams {
            distance: 3,
            ..PeakParams::default()
        };
        let peaks = find_peaks(&data, &params);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 1); // higher peak wins
    }

    #[test]
    fn width_filter() {
        // Narrow peak
        let data = [0.0, 0.0, 5.0, 0.0, 0.0];
        let params = PeakParams {
            min_width: Some(2.0),
            ..PeakParams::default()
        };
        let peaks = find_peaks(&data, &params);
        assert_eq!(peaks.len(), 0); // too narrow
    }

    #[test]
    fn empty_and_short_data() {
        assert!(find_peaks(&[], &PeakParams::default()).is_empty());
        assert!(find_peaks(&[1.0], &PeakParams::default()).is_empty());
        assert!(find_peaks(&[1.0, 2.0], &PeakParams::default()).is_empty());
    }

    #[test]
    fn monotonic_no_peaks() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(find_peaks(&data, &PeakParams::default()).is_empty());
    }

    #[test]
    fn gaussian_peak_width() {
        // Gaussian-like peak with known width
        let n = 101;
        let sigma = 10.0;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let x = (f64::from(i) - 50.0) / sigma;
                (-0.5 * x * x).exp()
            })
            .collect();

        let peaks = find_peaks(&data, &PeakParams::default());
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].index, 50);
        assert!((peaks[0].height - 1.0).abs() < 1e-10);
        // FWHM of Gaussian = 2*sqrt(2*ln(2))*sigma ≈ 23.55
        // Width at half-prominence ≈ FWHM
        assert!(peaks[0].width > 20.0);
        assert!(peaks[0].width < 30.0);
    }

    #[test]
    fn chromatographic_peaks() {
        // Simulate LC-MS chromatogram with 3 peaks of varying size
        let n = 200;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let x = f64::from(i);
                let p1 = 100.0 * (-0.5 * ((x - 40.0) / 5.0).powi(2)).exp();
                let p2 = 500.0 * (-0.5 * ((x - 100.0) / 8.0).powi(2)).exp();
                let p3 = 200.0 * (-0.5 * ((x - 160.0) / 6.0).powi(2)).exp();
                p1 + p2 + p3 + 10.0 // baseline
            })
            .collect();

        let params = PeakParams {
            min_height: Some(50.0),
            min_prominence: Some(30.0),
            ..PeakParams::default()
        };
        let peaks = find_peaks(&data, &params);
        assert_eq!(peaks.len(), 3);
        // Peaks should be near 40, 100, 160
        #[allow(clippy::cast_precision_loss)]
        {
            assert!((peaks[0].index as f64 - 40.0).abs() < 2.0);
            assert!((peaks[1].index as f64 - 100.0).abs() < 2.0);
            assert!((peaks[2].index as f64 - 160.0).abs() < 2.0);
        }
        // Heights should approximate the peak amplitudes + baseline
        assert!(peaks[1].height > peaks[2].height);
        assert!(peaks[2].height > peaks[0].height);
    }

    #[test]
    fn flat_top_peak() {
        let data = [0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0];
        let peaks = find_peaks(&data, &PeakParams::default());
        assert_eq!(peaks.len(), 1);
        assert!(peaks[0].index >= 2 && peaks[0].index <= 4);
    }

    #[test]
    fn peak_at_boundary_not_detected() {
        // Peaks at index 0 or last index are not local maxima
        let data = [5.0, 3.0, 1.0, 3.0, 5.0];
        let peaks = find_peaks(&data, &PeakParams::default());
        // Only interior peak at index 3 if it qualifies — actually index 3 has data[3]=3 > data[2]=1 but data[3]=3 < data[4]=5
        // So no peaks
        assert!(peaks.is_empty());
    }
}
