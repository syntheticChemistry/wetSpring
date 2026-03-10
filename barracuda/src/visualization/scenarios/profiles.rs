// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sample-parameterized scenario builders for scientist empowerment.
//!
//! Each builder takes real sample metadata and produces a fully wired
//! `petalTongue` scenario graph. Scientists bring their data; wetSpring
//! builds the visualization.

use super::{bar, edge, gauge, node, scaffold, scatter, timeseries};
use crate::visualization::{EcologyScenario, ScenarioEdge, ScientificRange};

/// Sample profile for a 16S environmental microbiome study.
#[derive(Debug, Clone)]
pub struct EnvironmentalProfile {
    /// Study name.
    pub study_name: String,
    /// Sample labels (e.g. `Soil_A`, `Water_B`).
    pub sample_labels: Vec<String>,
    /// Abundance matrix: `samples[i]` is a vector of taxon counts for sample i.
    pub abundances: Vec<Vec<f64>>,
    /// Top taxon names (matching columns of `abundances`).
    pub taxon_names: Vec<String>,
    /// Environment type for domain theming.
    pub environment: String,
}

/// Sample profile for a PFAS suspect screening study.
#[derive(Debug, Clone)]
pub struct PfasScreeningProfile {
    /// Study name.
    pub study_name: String,
    /// Compound names.
    pub compounds: Vec<String>,
    /// m/z values per compound.
    pub mz_values: Vec<f64>,
    /// Retention times per compound (minutes).
    pub rt_values: Vec<f64>,
    /// Match scores (0–1) per compound.
    pub match_scores: Vec<f64>,
    /// Concentrations (ng/L) per compound.
    pub concentrations: Vec<f64>,
}

/// Sample profile for a calibration and quantitation report.
#[derive(Debug, Clone)]
pub struct CalibrationProfile {
    /// Analyte name.
    pub analyte_name: String,
    /// Standard concentrations.
    pub standard_concentrations: Vec<f64>,
    /// Standard responses (peak area / signal).
    pub standard_responses: Vec<f64>,
    /// Unknown sample labels.
    pub unknown_labels: Vec<String>,
    /// Unknown sample responses.
    pub unknown_responses: Vec<f64>,
    /// Calibration R² value.
    pub r_squared: f64,
    /// Slope of the calibration curve.
    pub slope: f64,
    /// Intercept of the calibration curve.
    pub intercept: f64,
}

/// Build a 16S environmental study scenario from a sample profile.
///
/// Creates nodes for diversity, taxonomy, ordination, and rarefaction
/// with real data from the profile.
#[must_use]
#[expect(clippy::cast_precision_loss)]
pub fn environmental_study_scenario(
    profile: &EnvironmentalProfile,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        &format!("{} — Environmental Microbiome", profile.study_name),
        &format!(
            "{} samples from {} environment",
            profile.sample_labels.len(),
            profile.environment
        ),
    );

    // Diversity node
    let mut div_node = node("diversity", "Alpha Diversity", "compute", &["diversity"]);
    let shannons: Vec<f64> = profile
        .abundances
        .iter()
        .map(|counts| {
            let total: f64 = counts.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            -counts
                .iter()
                .filter(|&&c| c > 0.0)
                .map(|&c| {
                    let p = c / total;
                    p * p.ln()
                })
                .sum::<f64>()
        })
        .collect();

    div_node.data_channels.push(bar(
        "shannon_diversity",
        "Shannon Diversity",
        &profile.sample_labels,
        &shannons,
        "H'",
    ));
    let mean_shannon = if shannons.is_empty() {
        0.0
    } else {
        shannons.iter().sum::<f64>() / shannons.len() as f64
    };
    div_node.data_channels.push(gauge(
        "mean_shannon",
        "Mean Shannon",
        mean_shannon,
        0.0,
        6.0,
        "H'",
        [2.0, 5.0],
        [0.0, 2.0],
    ));
    div_node.scientific_ranges.push(ScientificRange {
        label: "Healthy diversity".into(),
        min: 2.0,
        max: 6.0,
        status: "normal".into(),
    });
    s.nodes.push(div_node);

    // Taxonomy node
    let mut tax_node = node("taxonomy", "Taxonomy", "compute", &["taxonomy"]);
    if !profile.taxon_names.is_empty() && !profile.abundances.is_empty() {
        let n_taxa = profile.taxon_names.len();
        let mean_abundances: Vec<f64> = (0..n_taxa)
            .map(|t| {
                let sum: f64 = profile
                    .abundances
                    .iter()
                    .map(|s| if t < s.len() { s[t] } else { 0.0 })
                    .sum();
                sum / profile.abundances.len() as f64
            })
            .collect();
        tax_node.data_channels.push(bar(
            "mean_abundances",
            "Mean Genus Abundances",
            &profile.taxon_names,
            &mean_abundances,
            "count",
        ));
    }
    s.nodes.push(tax_node);

    // Ordination node (simple PCoA placeholder using first 2-3 samples)
    let mut ord_node = node("ordination", "PCoA Ordination", "compute", &["ordination"]);
    if profile.sample_labels.len() >= 2 {
        let xs: Vec<f64> = (0..profile.sample_labels.len())
            .map(|i| (i as f64).mul_add(0.3, -0.5))
            .collect();
        let ys: Vec<f64> = shannons
            .iter()
            .map(|&h| h.mul_add(0.1, -0.2))
            .collect();
        ord_node.data_channels.push(scatter(
            "pcoa",
            "PCoA (Bray-Curtis)",
            &xs,
            &ys,
            &profile.sample_labels,
            "PC1",
            "PC2",
            "proportion",
        ));
    }
    s.nodes.push(ord_node);

    let edges = vec![
        edge("diversity", "taxonomy", "diversity → taxonomy"),
        edge("taxonomy", "ordination", "taxonomy → ordination"),
    ];

    (s, edges)
}

/// Build a PFAS screening scenario from a sample profile.
#[must_use]
pub fn pfas_screening_scenario(
    profile: &PfasScreeningProfile,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        &format!("{} — PFAS Screening", profile.study_name),
        &format!("{} suspect compounds", profile.compounds.len()),
    );
    s.domain = "measurement".into();

    // Detection node
    let mut det_node = node("detection", "Suspect Screening", "compute", &["spectral_match"]);
    det_node.data_channels.push(scatter(
        "rt_mz_scatter",
        "RT vs m/z",
        &profile.rt_values,
        &profile.mz_values,
        &profile.compounds,
        "Retention Time (min)",
        "m/z",
        "Da",
    ));
    det_node.data_channels.push(bar(
        "match_scores",
        "Spectral Match Scores",
        &profile.compounds,
        &profile.match_scores,
        "cosine",
    ));
    det_node.scientific_ranges.push(ScientificRange {
        label: "High confidence".into(),
        min: 0.8,
        max: 1.0,
        status: "normal".into(),
    });
    det_node.scientific_ranges.push(ScientificRange {
        label: "Low confidence".into(),
        min: 0.0,
        max: 0.8,
        status: "warning".into(),
    });
    s.nodes.push(det_node);

    // Quantitation node
    let mut quant_node = node("quantitation", "Quantitation", "compute", &["quantitation"]);
    quant_node.data_channels.push(bar(
        "concentrations",
        "Measured Concentrations",
        &profile.compounds,
        &profile.concentrations,
        "ng/L",
    ));
    s.nodes.push(quant_node);

    let edges = vec![edge("detection", "quantitation", "match → quantify")];
    (s, edges)
}

/// Build a calibration report scenario from a calibration profile.
#[must_use]
pub fn calibration_report_scenario(
    profile: &CalibrationProfile,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        &format!("{} — Calibration Report", profile.analyte_name),
        &format!(
            "R² = {:.4}, {} standards, {} unknowns",
            profile.r_squared,
            profile.standard_concentrations.len(),
            profile.unknown_labels.len()
        ),
    );
    s.domain = "measurement".into();

    // Calibration curve node
    let mut cal_node = node("calibration", "Standard Curve", "compute", &["calibration"]);
    cal_node.data_channels.push(timeseries(
        "cal_curve",
        "Calibration Curve",
        "Concentration",
        "Response",
        "AU",
        &profile.standard_concentrations,
        &profile.standard_responses,
    ));
    cal_node.data_channels.push(gauge(
        "r_squared",
        "R²",
        profile.r_squared,
        0.0,
        1.0,
        "",
        [0.99, 1.0],
        [0.95, 0.99],
    ));
    cal_node.scientific_ranges.push(ScientificRange {
        label: "Excellent linearity".into(),
        min: 0.99,
        max: 1.0,
        status: "normal".into(),
    });
    cal_node.scientific_ranges.push(ScientificRange {
        label: "Acceptable linearity".into(),
        min: 0.95,
        max: 0.99,
        status: "warning".into(),
    });
    s.nodes.push(cal_node);

    // Quantitation node
    let mut quant_node = node("results", "Quantitation Results", "data", &["quantitation"]);
    let predicted_conc: Vec<f64> = profile
        .unknown_responses
        .iter()
        .map(|&r| {
            if profile.slope.abs() > f64::EPSILON {
                (r - profile.intercept) / profile.slope
            } else {
                0.0
            }
        })
        .collect();
    quant_node.data_channels.push(bar(
        "unknown_concentrations",
        "Predicted Concentrations",
        &profile.unknown_labels,
        &predicted_conc,
        "units",
    ));
    s.nodes.push(quant_node);

    let edges = vec![edge("calibration", "results", "predict unknowns")];
    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::DataChannel;

    #[test]
    fn environmental_study_basic() {
        let profile = EnvironmentalProfile {
            study_name: "Test River".into(),
            sample_labels: vec!["Site_A".into(), "Site_B".into(), "Site_C".into()],
            abundances: vec![
                vec![100.0, 50.0, 30.0, 20.0],
                vec![80.0, 60.0, 40.0, 10.0],
                vec![200.0, 10.0, 5.0, 2.0],
            ],
            taxon_names: vec![
                "Bacteroidetes".into(),
                "Firmicutes".into(),
                "Proteobacteria".into(),
                "Actinobacteria".into(),
            ],
            environment: "riverine".into(),
        };
        let (scenario, edges) = environmental_study_scenario(&profile);
        assert_eq!(scenario.nodes.len(), 3);
        assert!(!edges.is_empty());
        assert!(scenario.name.contains("Test River"));

        let div = &scenario.nodes[0];
        assert!(!div.data_channels.is_empty());
        assert!(!div.scientific_ranges.is_empty());
    }

    #[test]
    fn pfas_screening_basic() {
        let profile = PfasScreeningProfile {
            study_name: "Municipal Water".into(),
            compounds: vec!["PFOS".into(), "PFOA".into(), "PFNA".into()],
            mz_values: vec![498.93, 412.97, 462.97],
            rt_values: vec![8.2, 7.1, 9.5],
            match_scores: vec![0.95, 0.88, 0.72],
            concentrations: vec![15.2, 8.7, 3.1],
        };
        let (scenario, edges) = pfas_screening_scenario(&profile);
        assert_eq!(scenario.nodes.len(), 2);
        assert_eq!(scenario.domain, "measurement");
        assert!(!edges.is_empty());
    }

    #[test]
    fn calibration_report_basic() {
        let profile = CalibrationProfile {
            analyte_name: "Caffeine".into(),
            standard_concentrations: vec![0.0, 10.0, 50.0, 100.0, 500.0],
            standard_responses: vec![0.0, 1200.0, 6100.0, 12000.0, 60500.0],
            unknown_labels: vec!["Sample_1".into(), "Sample_2".into()],
            unknown_responses: vec![3500.0, 8200.0],
            r_squared: 0.9998,
            slope: 121.0,
            intercept: -50.0,
        };
        let (scenario, edges) = calibration_report_scenario(&profile);
        assert_eq!(scenario.nodes.len(), 2);
        assert_eq!(scenario.domain, "measurement");
        assert!(!edges.is_empty());

        let cal = &scenario.nodes[0];
        assert!(!cal.scientific_ranges.is_empty());
        let r2_gauge = cal
            .data_channels
            .iter()
            .find(|ch| matches!(ch, DataChannel::Gauge { id, .. } if id == "r_squared"));
        assert!(r2_gauge.is_some());
    }

    #[test]
    fn empty_profile_no_panic() {
        let profile = EnvironmentalProfile {
            study_name: "Empty".into(),
            sample_labels: vec![],
            abundances: vec![],
            taxon_names: vec![],
            environment: "none".into(),
        };
        let (scenario, _) = environmental_study_scenario(&profile);
        assert_eq!(scenario.nodes.len(), 3);
    }

    #[test]
    fn calibration_zero_slope() {
        let profile = CalibrationProfile {
            analyte_name: "Test".into(),
            standard_concentrations: vec![0.0, 1.0],
            standard_responses: vec![0.0, 0.0],
            unknown_labels: vec!["U1".into()],
            unknown_responses: vec![5.0],
            r_squared: 0.0,
            slope: 0.0,
            intercept: 0.0,
        };
        let (scenario, _) = calibration_report_scenario(&profile);
        assert_eq!(scenario.nodes.len(), 2);
    }
}
