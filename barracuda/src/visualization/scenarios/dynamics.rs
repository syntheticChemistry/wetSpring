// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dynamics scenario: ODE time series for QS biofilm and other models.

use crate::bio::ode::OdeResult;
use crate::bio::qs_biofilm::{self, QsBiofilmParams};
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{edge, node, scaffold, timeseries};

/// Build a QS biofilm ODE dynamics scenario.
///
/// Integrates the Waters 2008 QS model with default parameters and
/// produces time series for each state variable (Cell, AHL, `HapR`,
/// c-di-GMP, Biofilm).
#[must_use]
pub fn qs_biofilm_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring QS Biofilm Dynamics",
        "Waters 2008 quorum sensing ODE — Cell, AHL, HapR, c-di-GMP, biofilm",
    );

    let params = QsBiofilmParams::default();
    let y0: [f64; 5] = [0.01, 0.0, 0.0, 0.0, 0.0];
    let result = qs_biofilm::run_scenario(&y0, 50.0, crate::tolerances::ODE_DEFAULT_DT, &params);

    let var_names = ["Cell", "AHL", "HapR", "c-di-GMP", "Biofilm"];

    let mut ode_node = node("qs_ode", "QS Biofilm ODE", "compute", &["science.qs_model"]);

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("qs_{}", name.to_lowercase().replace('-', "_")),
            &format!("{name} Concentration"),
            "Time (h)",
            "Concentration",
            "AU",
            &result.t,
            &y_vals,
        ));
    }

    s.nodes.push(ode_node);
    (s, vec![])
}

/// Extract time series for a single state variable from an [`OdeResult`].
fn extract_variable(result: &OdeResult, var_idx: usize) -> Vec<f64> {
    result
        .t
        .iter()
        .enumerate()
        .map(|(step, _)| result.y[step * result.n_vars + var_idx])
        .collect()
}

/// Build a combined dynamics scenario with multiple ODE models.
///
/// Includes QS biofilm and a bistable switch node.
#[must_use]
pub fn dynamics_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring ODE Dynamics",
        "Multi-model ODE integration — QS biofilm, bistable, capacitor",
    );

    let (qs, _) = qs_biofilm_scenario();
    for n in qs.nodes {
        s.nodes.push(n);
    }

    let bistable_node = node(
        "bistable",
        "Bistable Switch",
        "compute",
        &["science.bistable"],
    );
    s.nodes.push(bistable_node);

    let edges = vec![edge("qs_ode", "bistable", "QS → bistable coupling")];
    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qs_biofilm_produces_timeseries() {
        let (scenario, _) = qs_biofilm_scenario();
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 5);
    }

    #[test]
    fn dynamics_combines_models() {
        let (scenario, edges) = dynamics_scenario();
        assert!(scenario.nodes.len() >= 2);
        assert!(!edges.is_empty());
    }
}
