// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-species tissue geometry petalTongue scenarios.
//!
//! Side-by-side comparison of dog, cat, human, horse, and mouse tissue
//! morphometry: epidermal thickness, follicle density, effective dimension,
//! Anderson disorder, predicted AD severity (Paper 12 extension).

use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, edge, node, scaffold, scatter};

/// Cross-species tissue comparison scenario.
///
/// Bar charts for morphometry and scatter for dimension vs. disorder.
#[must_use]
pub fn cross_species_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Cross-Species Tissue Geometry",
        "Tissue morphometry and AD severity prediction across 5 species (Paper 12)",
    );

    let species = ["Dog", "Cat", "Human", "Horse", "Mouse"];
    let epidermis_um: [f64; 5] = [15.0, 10.0, 50.0, 35.0, 8.0];
    let follicle_density: [f64; 5] = [500.0, 800.0, 100.0, 300.0, 900.0];
    let d_eff: [f64; 5] = [2.5, 2.1, 2.8, 2.3, 2.0];
    let ic50_nm: [f64; 5] = [10.0, 36.0, 15.0, 63.0, 100.0];
    let barrier_w: Vec<f64> = ic50_nm.iter().map(|&v| v.ln() * 4.0).collect();

    let species_str: Vec<String> = species.iter().map(|s| (*s).to_string()).collect();

    let mut morphometry_node = node(
        "cross_species_morphometry",
        "Tissue Morphometry",
        "data",
        &["science.anderson.cross_species"],
    );

    morphometry_node.data_channels.push(bar(
        "epidermis",
        "Epidermal Thickness",
        &species_str,
        &epidermis_um,
        "μm",
    ));

    morphometry_node.data_channels.push(bar(
        "follicles",
        "Follicle Density",
        &species_str,
        &follicle_density,
        "/cm²",
    ));

    morphometry_node.data_channels.push(bar(
        "dimension",
        "Effective Dimension d_eff",
        &species_str,
        &d_eff,
        "d",
    ));

    morphometry_node.scientific_ranges.push(ScientificRange {
        label: "Dog: highest AD susceptibility (thin epidermis, dense follicles)".into(),
        min: 2.0,
        max: 2.5,
        status: "warning".into(),
    });

    s.nodes.push(morphometry_node);

    let mut anderson_node = node(
        "cross_species_anderson",
        "Anderson Disorder by Species",
        "compute",
        &["science.anderson.cross_species"],
    );

    anderson_node.data_channels.push(bar(
        "barrier_w",
        "Barrier Disorder W",
        &species_str,
        &barrier_w,
        "W",
    ));

    anderson_node.data_channels.push(scatter(
        "d_eff_vs_w",
        "Effective Dimension vs. Barrier Disorder",
        &d_eff,
        &barrier_w,
        &species_str,
        "d_eff",
        "Barrier W",
        "dimensionless",
    ));

    anderson_node.scientific_ranges.push(ScientificRange {
        label: "W_c ≈ 16.26 — localization threshold".into(),
        min: 15.0,
        max: 17.5,
        status: "warning".into(),
    });

    s.nodes.push(anderson_node);

    let edges = vec![edge(
        "cross_species_morphometry",
        "cross_species_anderson",
        "morphometry → Anderson analysis",
    )];

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_species_builds() {
        let (s, edges) = cross_species_scenario();
        assert_eq!(s.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(s.nodes[0].data_channels.len() >= 3);
    }
}
