// SPDX-License-Identifier: AGPL-3.0-or-later
//! Feature table construction from LC-MS data.
//!
//! Connects the EIC extraction ([`super::eic`]), peak detection
//! ([`super::signal`]), and integration pipelines into a complete
//! asari-style feature extraction workflow.
//!
//! # Pipeline
//!
//! ```text
//! mzML spectra → mass track detection → EIC extraction
//!     → peak detection → integration → feature table
//! ```
//!
//! # References
//!
//! - Li, S. et al. "Trackable and scalable LC-MS metabolomics data processing
//!   using asari." Nature Communications 14, 4113 (2023).

use crate::io::mzml::MzmlSpectrum;

use super::eic;
use super::signal::{self, PeakParams};

/// A single detected feature (m/z × RT × intensity).
#[derive(Debug, Clone)]
pub struct Feature {
    /// Target m/z value (centroid of mass track).
    pub mz: f64,
    /// Retention time at peak apex (minutes).
    pub rt_apex: f64,
    /// Retention time at peak start (minutes, interpolated).
    pub rt_start: f64,
    /// Retention time at peak end (minutes, interpolated).
    pub rt_end: f64,
    /// Peak height (maximum intensity).
    pub height: f64,
    /// Integrated peak area (trapezoidal, intensity × minutes).
    pub area: f64,
    /// Signal-to-noise ratio estimate.
    pub snr: f64,
    /// Peak width at half-height (minutes).
    pub width_fwhm: f64,
}

/// Configuration for feature extraction.
#[derive(Debug, Clone)]
pub struct FeatureParams {
    /// EIC extraction tolerance (ppm).
    pub eic_ppm: f64,
    /// Mass track minimum scans.
    pub min_scans: usize,
    /// Peak detection parameters.
    pub peak_params: PeakParams,
    /// Minimum peak height for feature acceptance.
    pub min_height: f64,
    /// Minimum signal-to-noise ratio.
    pub min_snr: f64,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            eic_ppm: 5.0,
            min_scans: 3,
            peak_params: PeakParams {
                min_prominence: Some(100.0),
                ..PeakParams::default()
            },
            min_height: 1000.0,
            min_snr: 3.0,
        }
    }
}

/// Feature table: collection of features extracted from a sample.
#[derive(Debug, Clone)]
pub struct FeatureTable {
    /// Detected features, sorted by m/z.
    pub features: Vec<Feature>,
    /// Number of mass tracks evaluated.
    pub mass_tracks_evaluated: usize,
    /// Number of EICs with at least one peak.
    pub eics_with_peaks: usize,
}

/// Extract features from parsed mzML spectra.
///
/// Performs the complete asari-style pipeline:
/// 1. Detect mass tracks from MS1 spectra
/// 2. Extract EIC for each mass track
/// 3. Detect peaks in each EIC
/// 4. Integrate and filter features
///
/// # Arguments
///
/// * `spectra` — Parsed mzML spectra (MS1 + MS2 mixed is OK; MS2 are filtered).
/// * `params` — Feature extraction parameters.
///
/// # Returns
///
/// [`FeatureTable`] with detected features.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // index → f64 for RT interpolation; right_ips/left_ips are small
#[must_use]
pub fn extract_features(spectra: &[MzmlSpectrum], params: &FeatureParams) -> FeatureTable {
    // 1. Detect mass tracks
    let mass_tracks = eic::detect_mass_tracks(spectra, params.eic_ppm, params.min_scans);
    let n_tracks = mass_tracks.len();

    if mass_tracks.is_empty() {
        return FeatureTable {
            features: vec![],
            mass_tracks_evaluated: 0,
            eics_with_peaks: 0,
        };
    }

    // 2. Extract EICs for all mass tracks
    let eics = eic::extract_eics(spectra, &mass_tracks, params.eic_ppm);

    // 3. Detect peaks and build features
    let mut features = Vec::new();
    let mut eics_with_peaks = 0;

    for chromatogram in &eics {
        if chromatogram.intensity.is_empty() {
            continue;
        }

        let peaks = signal::find_peaks(&chromatogram.intensity, &params.peak_params);

        if peaks.is_empty() {
            continue;
        }
        eics_with_peaks += 1;

        for peak in &peaks {
            if peak.height < params.min_height {
                continue;
            }

            // Estimate noise as median of non-peak intensities
            let noise = estimate_noise(&chromatogram.intensity, peak.left_base, peak.right_base);
            let snr = if noise > 0.0 {
                peak.height / noise
            } else {
                f64::INFINITY
            };

            if snr < params.min_snr {
                continue;
            }

            // Integration
            let area = eic::integrate_peak(
                &chromatogram.rt,
                &chromatogram.intensity,
                peak.left_base,
                peak.right_base,
            );

            // RT values
            let rt_apex = chromatogram.rt[peak.index];
            let rt_start = if peak.left_ips >= 0.0 {
                interpolate_rt(&chromatogram.rt, peak.left_ips)
            } else {
                chromatogram.rt[peak.left_base]
            };
            let rt_end = if peak.right_ips < chromatogram.rt.len() as f64 {
                interpolate_rt(&chromatogram.rt, peak.right_ips)
            } else {
                chromatogram.rt[peak.right_base]
            };

            // Width at half-height in RT units
            let width_fwhm = rt_end - rt_start;

            features.push(Feature {
                mz: chromatogram.target_mz,
                rt_apex,
                rt_start,
                rt_end,
                height: peak.height,
                area,
                snr,
                width_fwhm: width_fwhm.max(0.0),
            });
        }
    }

    // Sort features by m/z
    features.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(std::cmp::Ordering::Equal));

    FeatureTable {
        features,
        mass_tracks_evaluated: n_tracks,
        eics_with_peaks,
    }
}

/// Estimate noise level as median of intensities outside the peak region.
fn estimate_noise(intensity: &[f64], peak_start: usize, peak_end: usize) -> f64 {
    let mut non_peak: Vec<f64> = intensity
        .iter()
        .enumerate()
        .filter(|&(i, _)| i < peak_start || i > peak_end)
        .map(|(_, &v)| v)
        .filter(|&v| v > 0.0)
        .collect();

    if non_peak.is_empty() {
        return 0.0;
    }

    non_peak.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    non_peak[non_peak.len() / 2]
}

/// Interpolate RT at a fractional index position.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn interpolate_rt(rt: &[f64], fractional_idx: f64) -> f64 {
    if rt.is_empty() {
        return 0.0;
    }
    let idx = fractional_idx.floor() as usize;
    if idx >= rt.len() - 1 {
        return rt[rt.len() - 1];
    }
    let frac = fractional_idx - idx as f64;
    frac.mul_add(rt[idx + 1] - rt[idx], rt[idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::mzml::MzmlSpectrum;

    fn make_ms1(rt: f64, mz: &[f64], intensity: &[f64]) -> MzmlSpectrum {
        MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: rt,
            tic: intensity.iter().sum(),
            base_peak_mz: mz.first().copied().unwrap_or(0.0),
            base_peak_intensity: intensity.iter().copied().fold(0.0_f64, f64::max),
            lowest_mz: mz.first().copied().unwrap_or(0.0),
            highest_mz: mz.last().copied().unwrap_or(0.0),
            mz_array: mz.to_vec(),
            intensity_array: intensity.to_vec(),
        }
    }

    #[test]
    fn extract_features_empty() {
        let ft = extract_features(&[], &FeatureParams::default());
        assert!(ft.features.is_empty());
    }

    #[test]
    fn extract_features_gaussian_peak() {
        // Simulate a single compound at m/z 200.0 with a Gaussian chromatographic peak
        let n_scans = 50;
        let target_mz = 200.0;
        let peak_rt = 5.0; // peak at RT 5.0 min
        let sigma = 0.3; // ~0.7 min FWHM

        let spectra: Vec<MzmlSpectrum> = (0..n_scans)
            .map(|i| {
                let rt = f64::from(i).mul_add(0.1, 3.0); // 3.0 to 7.9 min
                let x = (rt - peak_rt) / sigma;
                let int = 50000.0f64.mul_add((-0.5 * x * x).exp(), 100.0); // baseline 100
                make_ms1(rt, &[target_mz], &[int])
            })
            .collect();

        let params = FeatureParams {
            eic_ppm: 10.0,
            min_scans: 2,
            peak_params: PeakParams {
                min_prominence: Some(500.0),
                ..PeakParams::default()
            },
            min_height: 1000.0,
            min_snr: 2.0,
        };

        let ft = extract_features(&spectra, &params);
        assert_eq!(
            ft.features.len(),
            1,
            "expected 1 feature, got {}",
            ft.features.len()
        );

        let f = &ft.features[0];
        assert!((f.mz - target_mz).abs() < 1.0);
        assert!((f.rt_apex - peak_rt).abs() < 0.2);
        assert!(f.height > 40000.0);
        assert!(f.area > 0.0);
        assert!(f.snr > 3.0);
    }

    #[test]
    fn extract_features_two_compounds() {
        let n_scans = 100;
        let spectra: Vec<MzmlSpectrum> = (0..n_scans)
            .map(|i| {
                let rt = f64::from(i).mul_add(0.1, 1.0);
                let x1 = (rt - 4.0) / 0.3;
                let x2 = (rt - 7.0) / 0.4;
                let int1 = 30000.0 * (-0.5 * x1 * x1).exp();
                let int2 = 50000.0 * (-0.5 * x2 * x2).exp();
                make_ms1(rt, &[150.0, 300.0], &[int1 + 50.0, int2 + 50.0])
            })
            .collect();

        let params = FeatureParams {
            eic_ppm: 10.0,
            min_scans: 2,
            peak_params: PeakParams {
                min_prominence: Some(500.0),
                ..PeakParams::default()
            },
            min_height: 1000.0,
            min_snr: 2.0,
        };

        let ft = extract_features(&spectra, &params);
        assert_eq!(
            ft.features.len(),
            2,
            "expected 2 features, got {}",
            ft.features.len()
        );

        // Should be sorted by m/z
        assert!(ft.features[0].mz < ft.features[1].mz);
    }

    #[test]
    fn noise_estimation() {
        let data = vec![10.0, 10.0, 10.0, 1000.0, 10.0, 10.0, 10.0];
        let noise = estimate_noise(&data, 3, 3);
        assert!((noise - 10.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_rt_exact_index() {
        let rt = vec![1.0, 2.0, 3.0, 4.0];
        assert!((interpolate_rt(&rt, 2.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn interpolate_rt_fractional() {
        let rt = vec![1.0, 2.0, 3.0, 4.0];
        assert!((interpolate_rt(&rt, 1.5) - 2.5).abs() < 1e-10);
    }
}
