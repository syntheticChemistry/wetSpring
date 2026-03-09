// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson localization scenario: spectral analysis, level spacing, W(t).

use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{gauge, node, scaffold, timeseries};

/// Build an Anderson spectral analysis scenario.
///
/// Visualizes level spacing ratio, spectral regime, and dynamic W(t)
/// disorder curves for the soil QS ↔ Anderson localization analogy.
#[must_use]
pub fn anderson_scenario(
    level_spacing_ratio: f64,
    w_values: &[f64],
    t_values: &[f64],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Anderson Spectral",
        "Anderson localization diagnostics — level spacing, regime, W(t) curves",
    );

    let regime = if level_spacing_ratio > 0.45 {
        "extended (GOE)"
    } else if level_spacing_ratio < 0.40 {
        "localized (Poisson)"
    } else {
        "critical"
    };

    let mut spectral_node = node(
        "anderson",
        "Anderson Spectral",
        "compute",
        &["science.anderson"],
    );

    spectral_node.data_channels.push(gauge(
        "level_spacing",
        "Level Spacing Ratio ⟨r⟩",
        level_spacing_ratio,
        0.0,
        1.0,
        "ratio",
        [0.386, 0.530],
        [0.35, 0.386],
    ));

    spectral_node.scientific_ranges.push(ScientificRange {
        label: format!("Regime: {regime}"),
        min: 0.386,
        max: 0.530,
        status: "normal".into(),
    });

    if !t_values.is_empty() && !w_values.is_empty() {
        spectral_node.data_channels.push(timeseries(
            "w_t_curve",
            "Dynamic W(t) Disorder",
            "Time",
            "Disorder W",
            "W",
            t_values,
            w_values,
        ));
    }

    s.nodes.push(spectral_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anderson_goe_regime() {
        let (scenario, _) = anderson_scenario(0.53, &[10.0, 12.0, 14.0], &[0.0, 1.0, 2.0]);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn anderson_poisson_regime() {
        let (scenario, _) = anderson_scenario(0.38, &[], &[]);
        assert_eq!(scenario.nodes[0].data_channels.len(), 1);
    }
}
