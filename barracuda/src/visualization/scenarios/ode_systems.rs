// SPDX-License-Identifier: AGPL-3.0-or-later
//! ODE system scenario builders: phage defense, bistable switch,
//! cooperation, multi-signal QS, and phenotypic capacitor.

use crate::bio::bistable::{self, BistableParams};
use crate::bio::capacitor::{self, CapacitorParams};
use crate::bio::cooperation::{self, CooperationParams};
use crate::bio::multi_signal::{self, MultiSignalParams};
use crate::bio::ode::OdeResult;
use crate::bio::phage_defense::{self, PhageDefenseParams};
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{gauge, node, scaffold, timeseries};

fn extract_variable(result: &OdeResult, var_idx: usize) -> Vec<f64> {
    result
        .t
        .iter()
        .enumerate()
        .map(|(step, _)| result.y[step * result.n_vars + var_idx])
        .collect()
}

/// Phage defense ODE scenario.
///
/// Runs the phage-bacteria-CRISPR-defense model and visualises all four
/// state variables as time series.
#[must_use]
pub fn phage_defense_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Phage Defense Dynamics",
        "Bacteria-phage-CRISPR-defense ODE system (4 state variables)",
    );

    let params = PhageDefenseParams::default();
    let result = phage_defense::scenario_phage_attack(&params, 0.01);

    let var_names = ["Bacteria", "Phage", "CRISPR", "Defense"];
    let mut ode_node = node(
        "phage_defense",
        "Phage Defense ODE",
        "compute",
        &["science.phage_defense"],
    );

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("pd_{}", name.to_lowercase()),
            &format!("{name} Dynamics"),
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

/// Bistable switch ODE scenario.
///
/// Integrates the QS bistable switch model (5 variables) and reports
/// the final biofilm state as a gauge.
#[must_use]
pub fn bistable_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Bistable Switch",
        "QS-driven bistable switch ODE — 5 state variables with hysteresis",
    );

    let params = BistableParams::default();
    let y0 = [0.01, 0.0, 0.0, 0.0, 0.0];
    let result = bistable::run_bistable(&y0, 50.0, 0.01, &params);

    let var_names = ["Cell", "AHL", "HapR", "c-di-GMP", "Biofilm"];
    let mut ode_node = node(
        "bistable",
        "Bistable Switch",
        "compute",
        &["science.bistable"],
    );

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("bs_{}", name.to_lowercase().replace('-', "_")),
            name,
            "Time (h)",
            "Concentration",
            "AU",
            &result.t,
            &y_vals,
        ));
    }

    let final_biofilm = result.y_final[4];
    ode_node.data_channels.push(gauge(
        "steady_state",
        "Steady-State Biofilm",
        final_biofilm,
        0.0,
        1.0,
        "AU",
        [0.0, 0.3],
        [0.3, 0.7],
    ));
    s.nodes.push(ode_node);
    (s, vec![])
}

/// Cooperation ODE scenario.
///
/// Models cooperator/cheater dynamics with AI signalling (4 variables)
/// and tracks cooperator frequency over time.
#[must_use]
pub fn cooperation_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Cooperation Dynamics",
        "Cooperator-cheater ODE with AI signalling and public-goods benefit",
    );

    let params = CooperationParams::default();
    let result = cooperation::scenario_equal_start(&params, 0.01);

    let var_names = ["Cooperators", "Cheaters", "AI", "Biofilm"];
    let mut ode_node = node(
        "cooperation",
        "Cooperation ODE",
        "compute",
        &["science.cooperation"],
    );

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("coop_{}", name.to_lowercase()),
            name,
            "Time (h)",
            "Concentration",
            "AU",
            &result.t,
            &y_vals,
        ));
    }

    let coop_freq = cooperation::cooperator_frequency(&result);
    let final_freq = coop_freq.last().copied().unwrap_or(0.5);
    ode_node.data_channels.push(gauge(
        "cooperator_fraction",
        "Cooperator Fraction",
        final_freq,
        0.0,
        1.0,
        "fraction",
        [0.4, 1.0],
        [0.1, 0.4],
    ));
    s.nodes.push(ode_node);
    (s, vec![])
}

/// Multi-signal QS scenario.
///
/// Dual quorum-sensing circuit (CAI-1 + AI-2) with 7 state variables.
#[must_use]
pub fn multi_signal_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Multi-Signal QS",
        "Dual QS circuit (CAI-1 + AI-2) — 7 state variables",
    );

    let params = MultiSignalParams::default();
    let result = multi_signal::scenario_wild_type(&params, 0.01);

    let var_names = ["Cell", "CAI-1", "AI-2", "CqsS", "LuxPQ", "HapR", "Biofilm"];
    let mut ode_node = node(
        "multi_signal",
        "Multi-Signal QS",
        "compute",
        &["science.multi_signal"],
    );

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("ms_{}", name.to_lowercase().replace('-', "_")),
            name,
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

/// Phenotypic capacitor ODE scenario.
///
/// Stress-responsive phenotypic switching model (6 variables) with
/// capacitor charge gauge.
#[must_use]
pub fn capacitor_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Phenotypic Capacitor",
        "Stress-responsive phenotypic switching ODE — 6 state variables",
    );

    let params = CapacitorParams::default();
    let result = capacitor::scenario_stress(&params, 0.01);

    let var_names = ["Cell", "AHL", "HapR", "c-di-GMP", "Biofilm", "VpsR"];
    let mut ode_node = node(
        "capacitor",
        "Phenotypic Capacitor",
        "compute",
        &["science.capacitor"],
    );

    for (i, name) in var_names.iter().enumerate() {
        let y_vals = extract_variable(&result, i);
        ode_node.data_channels.push(timeseries(
            &format!("cap_{}", name.to_lowercase().replace('-', "_")),
            name,
            "Time (h)",
            "Concentration",
            "AU",
            &result.t,
            &y_vals,
        ));
    }

    let final_cdg = result.y_final[3];
    ode_node.data_channels.push(gauge(
        "capacitor_charge",
        "Capacitor Charge (c-di-GMP)",
        final_cdg,
        0.0,
        1.0,
        "AU",
        [0.0, 0.3],
        [0.3, 0.7],
    ));
    s.nodes.push(ode_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phage_defense_builds() {
        let (s, _) = phage_defense_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 4);
    }

    #[test]
    fn bistable_builds() {
        let (s, _) = bistable_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 6);
    }

    #[test]
    fn cooperation_builds() {
        let (s, _) = cooperation_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 5);
    }

    #[test]
    fn multi_signal_builds() {
        let (s, _) = multi_signal_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 7);
    }

    #[test]
    fn capacitor_builds() {
        let (s, _) = capacitor_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 7);
    }
}
