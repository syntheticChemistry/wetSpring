// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hormesis dose-response petalTongue scenarios.
//!
//! Biphasic dose-response visualization: sweeps showing hormetic zone,
//! Anderson W mapping, dose-to-disorder curves. Paper 14 math.

use crate::bio::hormesis;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{edge, gauge, node, scaffold, timeseries};

/// Hormesis biphasic response scenario.
///
/// Sweeps a dose range showing the hormetic zone, stimulation/survival
/// components, and Anderson disorder mapping.
#[must_use]
pub fn hormesis_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Hormesis Dose-Response",
        "Biphasic dose-response with Anderson disorder mapping (Paper 14)",
    );

    let hp = hormesis::HormesisParams::new(0.3, 10.0, 2.0, 100.0, 2.0)
        .expect("default hormesis params are valid");

    let n_points: u32 = 100;
    let dose_max = 200.0;
    let doses: Vec<f64> = (0..n_points)
        .map(|i| dose_max * f64::from(i) / f64::from(n_points - 1))
        .collect();

    let points = hormesis::sweep(&doses, &hp);

    let responses: Vec<f64> = points.iter().map(|p| p.response).collect();
    let stimulations: Vec<f64> = points.iter().map(|p| p.stimulation).collect();
    let survivals: Vec<f64> = points.iter().map(|p| p.survival).collect();

    let w_baseline = 16.5;
    let sensitivity = 0.1;
    let disorder_values: Vec<f64> = doses
        .iter()
        .map(|&d| hormesis::dose_to_disorder(d, w_baseline, sensitivity, 1.0))
        .collect();

    let mut response_node = node(
        "hormesis_response",
        "Biphasic Response",
        "compute",
        &["science.anderson.hormesis"],
    );

    response_node.data_channels.push(timeseries(
        "composite_response",
        "Composite Response (stimulation × survival)",
        "Dose",
        "Response",
        "fold-change",
        &doses,
        &responses,
    ));

    response_node.data_channels.push(timeseries(
        "stimulation",
        "Stimulation Component",
        "Dose",
        "Stimulation",
        "fold-change",
        &doses,
        &stimulations,
    ));

    response_node.data_channels.push(timeseries(
        "survival",
        "Survival Component",
        "Dose",
        "Survival",
        "fraction",
        &doses,
        &survivals,
    ));

    if let Some((peak_dose, peak_response)) = hormesis::find_peak(&doses, &hp) {
        response_node.data_channels.push(gauge(
            "peak_response",
            "Peak Hormetic Response",
            peak_response,
            0.0,
            2.0,
            "fold-change",
            [1.0, 1.5],
            [0.5, 1.0],
        ));

        response_node.scientific_ranges.push(ScientificRange {
            label: format!("Peak at dose = {peak_dose:.1}"),
            min: peak_dose * 0.8,
            max: peak_dose * 1.2,
            status: "normal".into(),
        });
    }

    s.nodes.push(response_node);

    let mut disorder_node = node(
        "hormesis_disorder",
        "Dose → Anderson Disorder",
        "compute",
        &["science.anderson.hormesis"],
    );

    disorder_node.data_channels.push(timeseries(
        "dose_to_w",
        "Dose → Disorder W Mapping",
        "Dose",
        "Anderson W",
        "W",
        &doses,
        &disorder_values,
    ));

    disorder_node.data_channels.push(gauge(
        "w_baseline",
        "Baseline Disorder W",
        w_baseline,
        0.0,
        30.0,
        "W",
        [10.0, 20.0],
        [20.0, 25.0],
    ));

    disorder_node.scientific_ranges.push(ScientificRange {
        label: "W_c ≈ 16.26 (localization threshold)".into(),
        min: 15.0,
        max: 17.5,
        status: "warning".into(),
    });

    s.nodes.push(disorder_node);

    let edges = vec![edge(
        "hormesis_response",
        "hormesis_disorder",
        "response → disorder mapping",
    )];

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hormesis_builds() {
        let (s, edges) = hormesis_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(s.nodes[0].data_channels.len() >= 3);
    }
}
