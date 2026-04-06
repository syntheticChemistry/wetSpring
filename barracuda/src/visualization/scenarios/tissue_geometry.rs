// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tissue geometry petalTongue scenarios.
//!
//! 2D epidermis / 3D dermis lattice, barrier promotion sweep,
//! cell-type heterogeneity, and species comparison for the Anderson
//! localization framework applied to atopic dermatitis (Paper 12).

use crate::bio::diversity;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, edge, gauge, node, scaffold, timeseries};

/// Tissue lattice scenario: AD severity profiles with Anderson disorder.
///
/// Shows how cell-type heterogeneity maps to Anderson disorder across
/// a gradient from healthy to chronic lesion.
#[must_use]
pub fn tissue_lattice_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Tissue Geometry Lattice",
        "Anderson disorder mapping for atopic dermatitis tissue profiles (Paper 12)",
    );

    let profiles: &[(&str, f64, &[f64])] = &[
        ("Healthy", 0.85, &[60.0, 20.0, 10.0, 5.0, 3.0, 2.0]),
        ("Mild AD", 0.72, &[45.0, 25.0, 15.0, 8.0, 5.0, 2.0]),
        ("Moderate AD", 0.58, &[30.0, 25.0, 20.0, 12.0, 8.0, 5.0]),
        ("Severe AD", 0.40, &[20.0, 22.0, 20.0, 18.0, 12.0, 8.0]),
        (
            "Barrier Breach",
            0.25,
            &[15.0, 18.0, 22.0, 20.0, 15.0, 10.0],
        ),
        (
            "Chronic Lesion",
            0.15,
            &[10.0, 15.0, 20.0, 22.0, 18.0, 15.0],
        ),
    ];

    let base_disorder = 10.0;

    let profile_names: Vec<String> = profiles.iter().map(|(n, _, _)| (*n).to_string()).collect();
    let shannons: Vec<f64> = profiles.iter().map(|(_, _, c)| diversity::shannon(c)).collect();
    let pielous: Vec<f64> = profiles
        .iter()
        .map(|(_, _, c)| diversity::pielou_evenness(c))
        .collect();
    let disorder_w: Vec<f64> = profiles
        .iter()
        .map(|(_, ev, _)| base_disorder * (1.0 - ev))
        .collect();

    let mut lattice_node = node(
        "tissue_lattice",
        "Tissue Lattice Profiles",
        "compute",
        &["science.gonzales.tissue_lattice"],
    );

    lattice_node.data_channels.push(bar(
        "shannon_profiles",
        "Shannon Diversity by Profile",
        &profile_names,
        &shannons,
        "bits",
    ));

    lattice_node.data_channels.push(bar(
        "pielou_profiles",
        "Pielou Evenness by Profile",
        &profile_names,
        &pielous,
        "J'",
    ));

    lattice_node.data_channels.push(bar(
        "disorder_profiles",
        "Anderson Disorder W by Profile",
        &profile_names,
        &disorder_w,
        "W",
    ));

    lattice_node.scientific_ranges.push(ScientificRange {
        label: "Localization threshold W_c ≈ 16.26".into(),
        min: 15.0,
        max: 17.5,
        status: "warning".into(),
    });

    s.nodes.push(lattice_node);

    let mut dimension_node = node(
        "tissue_dimension",
        "Tissue Dimensionality",
        "info",
        &["science.gonzales.tissue_lattice"],
    );

    dimension_node.data_channels.push(gauge(
        "epidermis_d",
        "Epidermis (effective dimension)",
        2.0,
        1.0,
        3.0,
        "d_eff",
        [1.8, 2.2],
        [1.5, 1.8],
    ));

    dimension_node.data_channels.push(gauge(
        "dermis_d",
        "Dermis (effective dimension)",
        3.0,
        1.0,
        4.0,
        "d_eff",
        [2.8, 3.2],
        [2.5, 2.8],
    ));

    s.nodes.push(dimension_node);

    let edges = vec![edge(
        "tissue_lattice",
        "tissue_dimension",
        "profile → dimension analysis",
    )];

    (s, edges)
}

/// Barrier promotion sweep: how W changes as barrier function degrades.
#[must_use]
pub fn barrier_promotion_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Barrier Promotion Sweep",
        "Anderson disorder vs. barrier function — dimensional promotion in skin tissue",
    );

    let n_points = 50;
    let evenness_values: Vec<f64> = (0..n_points)
        .map(|i| 1.0 - f64::from(i) / f64::from(n_points - 1))
        .collect();
    let base_w = 10.0;
    let disorder_values: Vec<f64> = evenness_values
        .iter()
        .map(|&e| base_w * (1.0 - e))
        .collect();

    let mut sweep_node = node(
        "barrier_sweep",
        "Barrier Promotion Sweep",
        "compute",
        &["science.gonzales.tissue_lattice"],
    );

    sweep_node.data_channels.push(timeseries(
        "barrier_w",
        "Disorder W vs. Barrier Evenness",
        "Pielou Evenness (J')",
        "Anderson Disorder W",
        "W",
        &evenness_values,
        &disorder_values,
    ));

    s.nodes.push(sweep_node);
    (s, vec![])
}

/// Full tissue geometry scenario.
#[must_use]
pub fn tissue_geometry_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Tissue Geometry Explorer",
        "AD tissue profiles + barrier sweep + dimensionality analysis",
    );

    let (lattice, mut lattice_edges) = tissue_lattice_scenario();
    let (barrier, _) = barrier_promotion_scenario();

    for n in lattice.nodes {
        s.nodes.push(n);
    }
    for n in barrier.nodes {
        s.nodes.push(n);
    }

    lattice_edges.push(edge(
        "tissue_dimension",
        "barrier_sweep",
        "dimension → barrier sweep",
    ));

    (s, lattice_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tissue_lattice_builds() {
        let (s, edges) = tissue_lattice_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn barrier_promotion_builds() {
        let (s, _) = barrier_promotion_scenario();
        assert_eq!(s.nodes.len(), 1);
    }

    #[test]
    fn tissue_geometry_combined_builds() {
        let (s, edges) = tissue_geometry_scenario();
        assert_eq!(s.nodes.len(), 3);
        assert!(edges.len() >= 2);
    }
}
