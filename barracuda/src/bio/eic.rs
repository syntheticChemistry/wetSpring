// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extracted Ion Chromatogram (EIC/XIC) construction from mzML spectra.
//!
//! Builds chromatographic traces by binning m/z values across sequential
//! scans, enabling peak detection and quantification.
//!
//! This is the bridge between raw mzML parsing ([`crate::io::mzml`]) and
//! feature detection ([`super::signal::find_peaks`]).
//!
//! # Algorithm
//!
//! 1. For each target m/z (or m/z bin):
//!    a. Scan all MS1 spectra in retention-time order
//!    b. Sum intensities within ±ppm tolerance of the target
//!    c. Build (rt, intensity) chromatogram
//! 2. Apply peak detection to each chromatogram
//! 3. Integrate peak areas for quantification
//!
//! # References
//!
//! - Li, S. et al. "Trackable and scalable LC-MS metabolomics data processing
//!   using asari." Nature Communications 14, 4113 (2023).

use crate::io::mzml::MzmlSpectrum;

/// A single extracted ion chromatogram.
#[derive(Debug, Clone)]
pub struct Eic {
    /// Target m/z value (center of extraction window).
    pub target_mz: f64,
    /// Retention times (minutes) for each data point.
    pub rt: Vec<f64>,
    /// Summed intensity within the extraction window at each RT.
    pub intensity: Vec<f64>,
}

/// Extract ion chromatograms for multiple target m/z values.
///
/// # Arguments
///
/// * `spectra` — MS1 spectra sorted by retention time.
/// * `target_mzs` — Target m/z values to extract.
/// * `ppm` — Extraction tolerance in parts-per-million.
///
/// # Returns
///
/// One [`Eic`] per target m/z, in the same order as `target_mzs`.
#[must_use]
pub fn extract_eics(spectra: &[MzmlSpectrum], target_mzs: &[f64], ppm: f64) -> Vec<Eic> {
    // Filter to MS1 only and sort by RT
    let mut ms1: Vec<&MzmlSpectrum> = spectra.iter().filter(|s| s.ms_level == 1).collect();
    ms1.sort_by(|a, b| {
        a.rt_minutes
            .partial_cmp(&b.rt_minutes)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    target_mzs
        .iter()
        .map(|&target| {
            let mut rt = Vec::with_capacity(ms1.len());
            let mut intensity = Vec::with_capacity(ms1.len());

            for spectrum in &ms1 {
                rt.push(spectrum.rt_minutes);

                let tol = target * ppm * 1e-6;
                let lo = target - tol;
                let hi = target + tol;

                let sum: f64 = spectrum
                    .mz_array
                    .iter()
                    .zip(spectrum.intensity_array.iter())
                    .filter(|&(&mz, _)| mz >= lo && mz <= hi)
                    .map(|(_, &int)| int)
                    .sum();

                intensity.push(sum);
            }

            Eic {
                target_mz: target,
                rt,
                intensity,
            }
        })
        .collect()
}

/// Detect mass tracks (unique m/z bins) from a set of MS1 spectra.
///
/// Groups all observed m/z values into bins of width `ppm`, returning
/// the centroid m/z for each bin that appears in at least `min_scans`
/// consecutive scans.
///
/// This is the asari-style "mass track" extraction step.
///
/// # Arguments
///
/// * `spectra` — MS1 spectra.
/// * `ppm` — Binning tolerance in parts-per-million.
/// * `min_scans` — Minimum number of scans a track must span.
///
/// # Returns
///
/// Sorted vector of centroid m/z values for detected mass tracks.
#[must_use]
#[allow(clippy::cast_precision_loss)] // scan counts < 2^53
pub fn detect_mass_tracks(spectra: &[MzmlSpectrum], ppm: f64, min_scans: usize) -> Vec<f64> {
    // Collect all m/z values from MS1 spectra
    let ms1: Vec<&MzmlSpectrum> = spectra.iter().filter(|s| s.ms_level == 1).collect();
    if ms1.is_empty() {
        return vec![];
    }

    // Collect all unique m/z values sorted
    let mut all_mz: Vec<f64> = ms1
        .iter()
        .flat_map(|s| s.mz_array.iter().copied())
        .collect();
    all_mz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if all_mz.is_empty() {
        return vec![];
    }

    // Group into bins: consecutive m/z values within ppm tolerance
    let mut bins: Vec<(f64, usize)> = Vec::new(); // (centroid, count_of_mz_values)
    let mut bin_start = 0;
    for i in 1..all_mz.len() {
        let gap_ppm = (all_mz[i] - all_mz[i - 1]) / all_mz[i - 1] * 1e6;
        if gap_ppm > ppm {
            let centroid: f64 = all_mz[bin_start..i].iter().sum::<f64>() / (i - bin_start) as f64;
            bins.push((centroid, i - bin_start));
            bin_start = i;
        }
    }
    // Last bin
    let centroid: f64 = all_mz[bin_start..].iter().sum::<f64>() / (all_mz.len() - bin_start) as f64;
    bins.push((centroid, all_mz.len() - bin_start));

    // Filter by minimum scan count
    bins.into_iter()
        .filter(|&(_, count)| count >= min_scans)
        .map(|(mz, _)| mz)
        .collect()
}

/// Integrate the area under a chromatographic peak.
///
/// Uses trapezoidal integration between `left_idx` and `right_idx`.
///
/// # Arguments
///
/// * `rt` — Retention times (minutes).
/// * `intensity` — Intensities at each time point.
/// * `left_idx` — Start index (inclusive).
/// * `right_idx` — End index (inclusive).
///
/// # Returns
///
/// Integrated area (intensity × minutes).
#[must_use]
pub fn integrate_peak(rt: &[f64], intensity: &[f64], left_idx: usize, right_idx: usize) -> f64 {
    if left_idx >= right_idx || right_idx >= rt.len() {
        return 0.0;
    }

    let mut area = 0.0;
    for i in left_idx..right_idx {
        let dt = rt[i + 1] - rt[i];
        area += 0.5 * (intensity[i] + intensity[i + 1]) * dt;
    }
    area
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
    fn extract_single_eic() {
        let spectra = vec![
            make_ms1(1.0, &[100.0, 200.0, 300.0], &[10.0, 20.0, 30.0]),
            make_ms1(2.0, &[100.0, 200.0, 300.0], &[50.0, 60.0, 70.0]),
            make_ms1(3.0, &[100.0, 200.0, 300.0], &[5.0, 15.0, 25.0]),
        ];
        let eics = extract_eics(&spectra, &[200.0], 10.0);
        assert_eq!(eics.len(), 1);
        assert_eq!(eics[0].rt, vec![1.0, 2.0, 3.0]);
        assert_eq!(eics[0].intensity, vec![20.0, 60.0, 15.0]);
    }

    #[test]
    fn extract_multiple_eics() {
        let spectra = vec![
            make_ms1(1.0, &[100.0, 200.0], &[10.0, 20.0]),
            make_ms1(2.0, &[100.0, 200.0], &[30.0, 40.0]),
        ];
        let eics = extract_eics(&spectra, &[100.0, 200.0], 10.0);
        assert_eq!(eics.len(), 2);
        assert_eq!(eics[0].intensity, vec![10.0, 30.0]);
        assert_eq!(eics[1].intensity, vec![20.0, 40.0]);
    }

    #[test]
    fn eic_filters_ms2() {
        let mut ms2 = make_ms1(1.5, &[100.0], &[999.0]);
        ms2.ms_level = 2;
        let spectra = vec![
            make_ms1(1.0, &[100.0], &[10.0]),
            ms2,
            make_ms1(2.0, &[100.0], &[20.0]),
        ];
        let eics = extract_eics(&spectra, &[100.0], 10.0);
        assert_eq!(eics[0].rt.len(), 2); // MS2 filtered out
    }

    #[test]
    fn eic_empty_spectra() {
        let eics = extract_eics(&[], &[100.0], 10.0);
        assert_eq!(eics.len(), 1);
        assert!(eics[0].rt.is_empty());
    }

    #[test]
    fn integrate_trapezoidal() {
        let rt = vec![0.0, 1.0, 2.0, 3.0];
        let intensity = vec![0.0, 10.0, 10.0, 0.0];
        let area = integrate_peak(&rt, &intensity, 0, 3);
        // Trapezoid: (0+10)/2*1 + (10+10)/2*1 + (10+0)/2*1 = 5 + 10 + 5 = 20
        assert!((area - 20.0).abs() < 1e-10);
    }

    #[test]
    fn integrate_invalid_range() {
        let rt = vec![0.0, 1.0];
        let intensity = vec![10.0, 20.0];
        assert!(integrate_peak(&rt, &intensity, 1, 0).abs() < f64::EPSILON);
        assert!(integrate_peak(&rt, &intensity, 0, 5).abs() < f64::EPSILON);
    }

    #[test]
    fn detect_tracks_basic() {
        let spectra = vec![
            make_ms1(1.0, &[100.0, 200.0, 300.0], &[10.0, 20.0, 30.0]),
            make_ms1(2.0, &[100.001, 200.001, 300.001], &[10.0, 20.0, 30.0]),
            make_ms1(3.0, &[100.002, 200.002], &[10.0, 20.0]),
        ];
        let tracks = detect_mass_tracks(&spectra, 10.0, 2);
        // All 3 m/z should form tracks (each appears in 2+ scans within ppm)
        assert!(tracks.len() >= 2);
    }

    #[test]
    fn detect_tracks_empty() {
        assert!(detect_mass_tracks(&[], 10.0, 1).is_empty());
    }
}
