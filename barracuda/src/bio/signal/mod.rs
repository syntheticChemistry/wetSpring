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

mod peak_detect;
mod prominence;
mod smoothing;

pub use peak_detect::{find_peaks, find_peaks_with_area, integrate_peak};
pub use smoothing::{savitzky_golay, savitzky_golay_coefficients};

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

/// Manual impl intentional: `distance: 1` is non-zero (scipy.signal default).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

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
        assert!((peaks[0].height - 1.0).abs() < tolerances::PYTHON_PARITY);
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
        #[expect(clippy::cast_precision_loss)]
        {
            assert!((peaks[0].index as f64 - 40.0).abs() < crate::tolerances::PEAK_INDEX_PROXIMITY);
            assert!(
                (peaks[1].index as f64 - 100.0).abs() < crate::tolerances::PEAK_INDEX_PROXIMITY
            );
            assert!(
                (peaks[2].index as f64 - 160.0).abs() < crate::tolerances::PEAK_INDEX_PROXIMITY
            );
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

    #[test]
    fn integrate_peak_triangle() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 10.0, 0.0, 0.0];
        let peaks = find_peaks(&y, &PeakParams::default());
        assert_eq!(peaks.len(), 1);
        let area = integrate_peak(&x, &y, &peaks[0]);
        // Triangle: base=2 (index 1..3), height=10 → area = 10.0
        assert!((area - 10.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn integrate_peak_gaussian() {
        let n = 201;
        let sigma = 10.0;
        let x: Vec<f64> = (0..n).map(f64::from).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = (f64::from(i) - 100.0) / sigma;
                1000.0 * (-0.5 * t * t).exp()
            })
            .collect();

        let peaks = find_peaks(&y, &PeakParams::default());
        assert_eq!(peaks.len(), 1);
        let area = integrate_peak(&x, &y, &peaks[0]);
        // Gaussian integral ≈ amplitude × σ × √(2π) ≈ 1000 × 10 × 2.5066 ≈ 25066
        // We integrate over left_base..right_base which captures most of it
        assert!(area > 20000.0, "area should be substantial: {area}");
        assert!(area < 30000.0, "area should be bounded: {area}");
    }

    #[test]
    fn integrate_peak_invalid_returns_zero() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 5.0, 0.0];
        let peak = Peak {
            index: 1,
            height: 5.0,
            prominence: 5.0,
            left_base: 2,
            right_base: 0, // invalid: left > right
            width: 1.0,
            left_ips: 0.5,
            right_ips: 1.5,
        };
        assert!(integrate_peak(&x, &y, &peak).abs() < f64::EPSILON);
    }

    #[test]
    fn find_peaks_with_area_basic() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 10.0, 0.0, 0.0];
        let results = find_peaks_with_area(&x, &y, &PeakParams::default());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.index, 2);
        assert!((results[0].1 - 10.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }
}
