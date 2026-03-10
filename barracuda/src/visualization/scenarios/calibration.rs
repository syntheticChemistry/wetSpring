// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibration curve scenario: standard curve, R², predicted unknowns.
//!
//! Wraps [`crate::bio::calibration`] math into a petalTongue-ready scenario
//! with actionable ranges for linearity and recovery.

use crate::bio::calibration;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, gauge, node, scaffold, timeseries};

/// Build a calibration scenario from standard concentrations and responses.
///
/// Fits a linear calibration curve, computes R², and predicts unknown
/// concentrations from their responses.
///
/// # Errors
///
/// Returns `Err` if the calibration fit fails (e.g. < 2 points).
pub fn calibration_scenario(
    analyte: &str,
    std_conc: &[f64],
    std_resp: &[f64],
    unknown_labels: &[String],
    unknown_resp: &[f64],
) -> crate::error::Result<(EcologyScenario, Vec<ScenarioEdge>)> {
    let fit = calibration::fit_calibration(std_conc, std_resp, "conc", "AU").ok_or_else(|| {
        crate::error::Error::InvalidInput("calibration fit failed: need >= 2 points".into())
    })?;

    let mut s = scaffold(
        &format!("{analyte} — Calibration"),
        &format!(
            "R² = {:.4}, slope = {:.4}, intercept = {:.4}",
            fit.r_squared, fit.slope, fit.intercept
        ),
    );
    s.domain = "measurement".into();

    // Standard curve node
    let mut cal_node = node("calibration", "Standard Curve", "compute", &["calibration"]);
    cal_node.data_channels.push(timeseries(
        "std_curve",
        "Calibration Curve",
        "Concentration",
        "Response",
        "AU",
        std_conc,
        std_resp,
    ));
    cal_node.data_channels.push(gauge(
        "r_squared",
        "R²",
        fit.r_squared,
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

    // Quantitation results node
    let mut results_node = node("results", "Predicted Unknowns", "data", &["quantitation"]);
    let predicted: Vec<f64> = unknown_resp
        .iter()
        .map(|&r| fit.predict(r).map_or(0.0, |q| q.concentration))
        .collect();
    results_node.data_channels.push(bar(
        "predicted",
        "Predicted Concentrations",
        unknown_labels,
        &predicted,
        "conc",
    ));
    s.nodes.push(results_node);

    Ok((
        s,
        vec![super::edge("calibration", "results", "predict unknowns")],
    ))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn calibration_scenario_basic() {
        let conc = [0.0, 10.0, 50.0, 100.0];
        let resp = [0.0, 120.0, 600.0, 1200.0];
        let labels = vec!["U1".into(), "U2".into()];
        let unk = [350.0, 800.0];
        let (scenario, edges) =
            calibration_scenario("Caffeine", &conc, &resp, &labels, &unk).unwrap();
        assert_eq!(scenario.nodes.len(), 2);
        assert_eq!(scenario.domain, "measurement");
        assert!(!edges.is_empty());
        assert!(!scenario.nodes[0].scientific_ranges.is_empty());
    }

    #[test]
    fn calibration_scenario_insufficient_points() {
        let result = calibration_scenario("X", &[1.0], &[2.0], &[], &[]);
        assert!(result.is_err());
    }
}
