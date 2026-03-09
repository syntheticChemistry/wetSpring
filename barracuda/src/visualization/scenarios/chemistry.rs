// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chemistry scenario: EIC chromatograms, KMD plots, peak annotations.

use crate::bio::{kmd, signal};
use crate::io::mzml::MzmlSpectrum;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{edge, node, scaffold, scatter, timeseries};

/// Build a chemistry scenario from parsed mzML spectra and m/z targets.
///
/// Produces EIC chromatograms with peak annotations, KMD scatter plots
/// for PFAS homologue detection.
#[must_use]
pub fn chemistry_scenario(
    spectra: &[MzmlSpectrum],
    targets: &[f64],
    ppm: f64,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Analytical Chemistry",
        "EIC chromatograms, peak detection, and Kendrick mass defect analysis",
    );

    let eics = crate::bio::eic::extract_eics(spectra, targets, ppm);

    let mut eic_node = node(
        "eic",
        "Extracted Ion Chromatograms",
        "compute",
        &["science.eic"],
    );

    for (i, eic) in eics.iter().enumerate() {
        eic_node.data_channels.push(timeseries(
            &format!("eic_{i}"),
            &format!("EIC m/z {:.4}", eic.target_mz),
            "Retention time (min)",
            "Intensity",
            "counts",
            &eic.rt,
            &eic.intensity,
        ));
    }

    s.nodes.push(eic_node);

    let mut peak_node = node(
        "peaks",
        "Peak Detection",
        "compute",
        &["science.peak_detect"],
    );

    if let Some(first_eic) = eics.first() {
        let params = signal::PeakParams::default();
        let peaks = signal::find_peaks(&first_eic.intensity, &params);

        let peak_rts: Vec<f64> = peaks.iter().map(|p| first_eic.rt[p.index]).collect();
        let peak_heights: Vec<f64> = peaks.iter().map(|p| p.height).collect();
        let peak_labels: Vec<String> = peaks
            .iter()
            .enumerate()
            .map(|(i, p)| format!("P{}: h={:.0}", i + 1, p.height))
            .collect();

        peak_node.data_channels.push(scatter(
            "peak_positions",
            "Detected Peaks",
            &peak_rts,
            &peak_heights,
            &peak_labels,
            "Retention time (min)",
            "Height",
            "counts",
        ));
    }

    s.nodes.push(peak_node);

    let all_mz: Vec<f64> = spectra
        .iter()
        .flat_map(|sp| sp.mz_array.iter().copied())
        .collect();
    if !all_mz.is_empty() {
        let mut kmd_node = node("kmd", "Kendrick Mass Defect", "compute", &["science.kmd"]);

        let kmd_results =
            kmd::kendrick_mass_defect(&all_mz, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);

        let nominal: Vec<f64> = kmd_results.iter().map(|r| r.nominal_km).collect();
        let defect: Vec<f64> = kmd_results.iter().map(|r| r.kmd).collect();

        kmd_node.data_channels.push(scatter(
            "kmd_plot",
            "KMD Plot (CF₂ basis)",
            &nominal,
            &defect,
            &[] as &[String],
            "Nominal Kendrick Mass",
            "Kendrick Mass Defect",
            "KMD",
        ));

        s.nodes.push(kmd_node);
    }

    let mut edges = vec![edge("eic", "peaks", "chromatogram → peak detection")];
    if !all_mz.is_empty() {
        edges.push(edge("eic", "kmd", "m/z → KMD analysis"));
    }

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chemistry_empty_spectra() {
        let (scenario, edges) = chemistry_scenario(&[], &[200.0], 10.0);
        assert!(!scenario.nodes.is_empty());
        assert!(!edges.is_empty());
    }
}
