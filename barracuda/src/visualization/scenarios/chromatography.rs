// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chromatography scenario builders for `petalTongue`.
//!
//! Builds TIC, EIC, and feature-detection visualizations from
//! synthetic LC-MS data. Demonstrates the full analytical chemistry
//! pipeline: chromatogram → peak detection → integration → quantitation.

use crate::bio::calibration;
use crate::bio::eic;
use crate::bio::signal::{self, PeakParams};
use crate::cast::usize_f64;
use crate::io::mzml::MzmlSpectrum;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, edge, gauge, node, scaffold, timeseries};

/// Build a synthetic TIC chromatogram with peak annotations.
///
/// Generates a Gaussian-peak chromatogram, detects peaks, integrates
/// areas, and wraps everything in `petalTongue` scenario format.
#[must_use]
pub fn chromatogram_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "LC-MS Chromatography",
        "TIC/EIC chromatogram with peak detection and integration",
    );

    // Synthetic chromatogram: 3 Gaussian peaks on a baseline
    let n_points = 200;
    let rt: Vec<f64> = (0..n_points).map(|i| f64::from(i) * 0.1).collect();
    let intensity: Vec<f64> = rt
        .iter()
        .map(|&t| {
            let p1 = 50_000.0 * (-0.5 * ((t - 4.0) / 0.3).powi(2)).exp();
            let p2 = 120_000.0 * (-0.5 * ((t - 10.0) / 0.5).powi(2)).exp();
            let p3 = 30_000.0 * (-0.5 * ((t - 16.0) / 0.4).powi(2)).exp();
            p1 + p2 + p3 + 500.0 // baseline
        })
        .collect();

    // Peak detection
    let params = PeakParams {
        min_height: Some(5000.0),
        min_prominence: Some(3000.0),
        ..PeakParams::default()
    };
    let peaks_with_area = signal::find_peaks_with_area(&rt, &intensity, &params);

    // TIC chromatogram node
    let mut tic_node = node(
        "tic_chromatogram",
        "Total Ion Chromatogram",
        "compute",
        &["science.chromatography"],
    );
    tic_node.data_channels.push(timeseries(
        "tic",
        "TIC",
        "Retention Time (min)",
        "Intensity",
        "AU",
        &rt,
        &intensity,
    ));
    s.nodes.push(tic_node);

    // Peak detection node
    let mut peak_node = node(
        "peak_detection",
        "Peak Detection",
        "compute",
        &["science.signal"],
    );

    #[expect(clippy::cast_precision_loss)] // Precision: peak count bounded
    let peak_count = peaks_with_area.len() as f64;

    let peak_names: Vec<String> = peaks_with_area
        .iter()
        .enumerate()
        .map(|(i, (p, _))| format!("Peak {} (RT {:.1})", i + 1, rt[p.index]))
        .collect();
    let peak_heights: Vec<f64> = peaks_with_area.iter().map(|(p, _)| p.height).collect();
    let peak_areas: Vec<f64> = peaks_with_area.iter().map(|(_, a)| *a).collect();

    peak_node.data_channels.push(bar(
        "peak_heights",
        "Peak Heights",
        &peak_names,
        &peak_heights,
        "AU",
    ));
    peak_node.data_channels.push(bar(
        "peak_areas",
        "Integrated Areas",
        &peak_names,
        &peak_areas,
        "AU·min",
    ));
    peak_node.data_channels.push(gauge(
        "peaks_detected",
        "Peaks Detected",
        peak_count,
        0.0,
        10.0,
        "count",
        [1.0, 5.0],
        [5.0, 10.0],
    ));
    s.nodes.push(peak_node);

    let edges = vec![edge(
        "tic_chromatogram",
        "peak_detection",
        "raw signal → peaks",
    )];
    (s, edges)
}

/// Build a calibration + quantitation scenario.
///
/// Fits a calibration curve from standards, then quantifies unknowns.
#[must_use]
pub fn quantitation_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Analytical Quantitation",
        "Calibration curve fitting and unknown quantitation",
    );

    // Standards: 5-point calibration
    let std_conc = [0.0, 10.0, 50.0, 100.0, 500.0];
    let std_area = [0.0, 1050.0, 5200.0, 10100.0, 50500.0];

    let curve = calibration::fit_calibration(&std_conc, &std_area, "ng/L", "AU·min");

    let mut cal_node = node(
        "calibration",
        "Calibration Curve",
        "compute",
        &["science.calibration"],
    );
    cal_node.data_channels.push(timeseries(
        "cal_curve",
        "Standard Curve",
        "Concentration (ng/L)",
        "Peak Area (AU·min)",
        "AU·min",
        &std_conc,
        &std_area,
    ));
    if let Some(ref c) = curve {
        cal_node.data_channels.push(gauge(
            "r_squared",
            "R²",
            c.r_squared,
            0.0,
            1.0,
            "",
            [0.99, 1.0],
            [0.95, 0.99],
        ));
    }
    s.nodes.push(cal_node);

    // Unknowns
    let unknown_areas = [2500.0, 7800.0, 25000.0];
    let mut quant_node = node(
        "quantitation",
        "Unknown Quantitation",
        "compute",
        &["science.quantitation"],
    );

    if let Some(ref c) = curve {
        let results = calibration::quantify_batch(c, &unknown_areas);
        let labels: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, _)| format!("Unknown {}", i + 1))
            .collect();
        let concs: Vec<f64> = results.iter().map(|r| r.concentration).collect();
        quant_node.data_channels.push(bar(
            "concentrations",
            "Predicted Concentrations",
            &labels,
            &concs,
            "ng/L",
        ));
    }
    s.nodes.push(quant_node);

    let edges = vec![edge("calibration", "quantitation", "calibration → predict")];
    (s, edges)
}

/// Build a combined EIC + feature detection scenario from synthetic spectra.
///
/// Demonstrates the `asari`-style pipeline: spectra → EICs → peaks → features.
#[must_use]
pub fn eic_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "EIC Feature Detection",
        "Extracted ion chromatogram construction and peak detection",
    );

    // Synthetic MS1 spectra with two compounds
    let n_scans: usize = 60;
    let target_mz = 200.0;
    let spectra: Vec<MzmlSpectrum> = (0..n_scans)
        .map(|i| {
            let rt = usize_f64(i).mul_add(0.2, 1.0);
            let x = (rt - 6.0) / 0.5;
            let int = 80_000.0f64.mul_add((-0.5 * x * x).exp(), 200.0);
            MzmlSpectrum {
                index: i,
                ms_level: 1,
                rt_minutes: rt,
                tic: int,
                base_peak_mz: target_mz,
                base_peak_intensity: int,
                lowest_mz: target_mz,
                highest_mz: target_mz,
                mz_array: vec![target_mz],
                intensity_array: vec![int],
            }
        })
        .collect();

    let eics = eic::extract_eics(&spectra, &[target_mz], 10.0);

    let mut eic_node = node(
        "eic_extraction",
        "Extracted Ion Chromatogram",
        "compute",
        &["science.eic"],
    );
    if let Some(e) = eics.first() {
        eic_node.data_channels.push(timeseries(
            "eic_200",
            "EIC m/z 200.0",
            "Retention Time (min)",
            "Intensity",
            "AU",
            &e.rt,
            &e.intensity,
        ));

        let peaks = signal::find_peaks(
            &e.intensity,
            &PeakParams {
                min_height: Some(1000.0),
                ..PeakParams::default()
            },
        );

        #[expect(clippy::cast_precision_loss)] // Precision: peak count bounded
        let n_peaks = peaks.len() as f64;
        eic_node.data_channels.push(gauge(
            "eic_peaks",
            "Detected Peaks",
            n_peaks,
            0.0,
            5.0,
            "count",
            [1.0, 3.0],
            [3.0, 5.0],
        ));
    }
    s.nodes.push(eic_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromatogram_builds() {
        let (s, edges) = chromatogram_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
        assert!(s.nodes[1].data_channels.len() >= 2);
    }

    #[test]
    fn quantitation_builds() {
        let (s, edges) = quantitation_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn eic_builds() {
        let (s, _) = eic_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }
}
