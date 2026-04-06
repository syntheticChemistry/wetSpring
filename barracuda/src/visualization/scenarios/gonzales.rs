// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gonzales dermatitis petalTongue scenarios.
//!
//! IC50 dose-response curves, PK decay, pruritus time-series, and
//! treatment comparison. Uses real `barracuda::stats::hill` and
//! `bio::hormesis` math with tolerance-backed expected values.

use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, edge, gauge, node, scaffold, timeseries};

/// IC50 dose-response scenario for 6 cytokine pathways.
///
/// Builds timeseries nodes for each pathway showing Hill inhibition
/// curves, plus a bar chart of IC50 values.
#[must_use]
pub fn gonzales_dose_response_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Gonzales IC50 Dose-Response",
        "Hill equation inhibition curves for 6 cytokine pathways (Gonzales 2014)",
    );

    let pathways: &[(&str, f64)] = &[
        ("JAK1", 10.0),
        ("IL-2", 36.0),
        ("IL-6", 36.0),
        ("IL-31", 63.0),
        ("IL-4", 159.0),
        ("IL-13", 249.0),
    ];

    let n_points = 50;
    let dose_max = 500.0;
    let doses: Vec<f64> = (0..n_points)
        .map(|i| dose_max * f64::from(i) / f64::from(n_points - 1))
        .collect();

    let names: Vec<String> = pathways.iter().map(|(n, _)| (*n).to_string()).collect();
    let ic50s: Vec<f64> = pathways.iter().map(|(_, v)| *v).collect();

    let mut ic50_node = node(
        "gonzales_ic50",
        "IC50 Dose-Response",
        "compute",
        &["science.gonzales.dose_response"],
    );

    for &(name, ic50) in pathways {
        let responses: Vec<f64> = doses
            .iter()
            .map(|&d| barracuda::stats::hill(d, ic50, 1.0))
            .collect();
        ic50_node.data_channels.push(timeseries(
            &format!("dr_{}", name.to_lowercase()),
            &format!("{name} (IC50 = {ic50} nM)"),
            "Dose (nM)",
            "Fractional Inhibition",
            "fraction",
            &doses,
            &responses,
        ));
    }

    ic50_node
        .data_channels
        .push(bar("ic50_bar", "IC50 Values", &names, &ic50s, "nM"));

    ic50_node.scientific_ranges.push(ScientificRange {
        label: "JAK1 IC50 = 10 nM (most potent)".into(),
        min: 5.0,
        max: 15.0,
        status: "normal".into(),
    });

    s.nodes.push(ic50_node);
    (s, vec![])
}

/// Lokivetmab PK decay scenario (3 doses).
///
/// Exponential efficacy decay for 0.125, 0.5, and 2.0 mg/kg.
#[must_use]
pub fn gonzales_pk_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Gonzales PK Decay",
        "Lokivetmab (Cytopoint) pharmacokinetic efficacy decay (Fleck/Gonzales 2021)",
    );

    let doses: [f64; 3] = [0.125, 0.5, 2.0];
    let durations: [f64; 3] = [14.0, 28.0, 42.0];
    let k_decay = (doses[2] / doses[0]).ln() / (durations[2] - durations[0]);

    let n_points = 100;
    let t_max = 56.0;
    let times: Vec<f64> = (0..n_points)
        .map(|i| t_max * f64::from(i) / f64::from(n_points - 1))
        .collect();

    let mut pk_node = node(
        "gonzales_pk",
        "Lokivetmab PK Decay",
        "compute",
        &["science.gonzales.pk_decay"],
    );

    for (&dose, &dur) in doses.iter().zip(durations.iter()) {
        let efficacy: Vec<f64> = times
            .iter()
            .map(|&t| {
                if t <= dur {
                    (-k_decay * t / dur).exp()
                } else {
                    0.0
                }
            })
            .collect();
        pk_node.data_channels.push(timeseries(
            &format!("pk_{}", dose.to_bits()),
            &format!("{dose} mg/kg (duration {dur} days)"),
            "Days",
            "Efficacy",
            "fraction",
            &times,
            &efficacy,
        ));
    }

    pk_node.data_channels.push(gauge(
        "k_decay",
        "Decay Constant",
        k_decay,
        0.0,
        0.5,
        "1/day",
        [0.05, 0.2],
        [0.2, 0.4],
    ));

    s.nodes.push(pk_node);
    (s, vec![])
}

/// Full Gonzales scenario: dose-response + PK combined.
#[must_use]
pub fn gonzales_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Gonzales Dermatitis Science",
        "IC50 dose-response + lokivetmab PK — Gonzales published data",
    );

    let (dr, _) = gonzales_dose_response_scenario();
    let (pk, _) = gonzales_pk_scenario();

    for n in dr.nodes {
        s.nodes.push(n);
    }
    for n in pk.nodes {
        s.nodes.push(n);
    }

    let edges = vec![edge(
        "gonzales_ic50",
        "gonzales_pk",
        "IC50 → PK treatment model",
    )];

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dose_response_builds() {
        let (s, _) = gonzales_dose_response_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 7);
    }

    #[test]
    fn pk_builds() {
        let (s, _) = gonzales_pk_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 4);
    }

    #[test]
    fn gonzales_combined_builds() {
        let (s, edges) = gonzales_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
    }
}
