// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-domain `petalTongue` scenario builders.
//!
//! Each builder calls real wetSpring / `barraCuda` math and wraps outputs
//! in [`DataChannel`] / [`ScenarioNode`] / [`EcologyScenario`] so
//! `petalTongue` can render them directly.

pub mod anderson;
pub mod basecalling;
pub mod benchmarks;
pub mod calibration;
pub mod chemistry;
pub mod chromatography;
pub mod composite;
pub mod dynamics;
pub mod ecology;
pub mod hmm;
pub mod lcms;
pub mod ml_models;
pub mod msa;
pub mod neighbor_joining;
pub mod nmf;
pub mod ode_systems;
pub mod ordination;
pub mod pangenome;
pub mod phylogenetics;
pub mod pipeline;
pub mod popgen;
pub mod profiles;
pub mod rarefaction;
pub mod similarity;
pub mod spectroscopy;
pub mod stochastic;
pub mod streaming_pipeline;

pub use anderson::anderson_scenario;
pub use basecalling::basecalling_scenario;
pub use benchmarks::benchmark_scenario;
pub use calibration::calibration_scenario;
pub use chemistry::chemistry_scenario;
pub use chromatography::{chromatogram_scenario, eic_scenario, quantitation_scenario};
pub use dynamics::{dynamics_scenario, qs_biofilm_scenario};
pub use ecology::ecology_scenario;
pub use hmm::hmm_scenario;
pub use lcms::{pfas_overview_scenario, spectral_match_scenario, tolerance_search_scenario};
pub use ml_models::{decision_tree_scenario, esn_scenario, random_forest_scenario};
pub use msa::msa_scenario;
pub use neighbor_joining::neighbor_joining_scenario;
pub use nmf::nmf_scenario;
pub use ode_systems::{
    bistable_scenario, capacitor_scenario, cooperation_scenario, multi_signal_scenario,
    phage_defense_scenario,
};
pub use ordination::ordination_scenario;
pub use pangenome::pangenome_scenario;
pub use phylogenetics::{
    dnds_scenario, felsenstein_scenario, molecular_clock_scenario, placement_scenario,
    reconciliation_scenario, unifrac_scenario,
};
pub use pipeline::{
    dada2_scenario, pipeline_overview_scenario, quality_scenario, taxonomy_scenario,
};
pub use popgen::{kmer_spectrum_scenario, population_genomics_scenario, snp_scenario};
pub use profiles::{
    CalibrationProfile, EnvironmentalProfile, PfasScreeningProfile, calibration_report_scenario,
    environmental_study_scenario, pfas_screening_scenario,
};
pub use rarefaction::rarefaction_scenario;
pub use similarity::similarity_scenario;
pub use spectroscopy::{spectroscopy_scenario, spectroscopy_scenario_from_data};
pub use stochastic::stochastic_scenario;

pub use composite::{
    full_16s_scenario, full_ecology_scenario, full_pfas_scenario, full_qs_scenario,
};

use super::types::{DataChannel, EcologyScenario, ScenarioEdge, ScenarioNode};

// ── Helper constructors ─────────────────────────────────────────────────

/// Scaffold a scenario with default `petalTongue` metadata.
fn scaffold(name: &str, description: &str) -> EcologyScenario {
    EcologyScenario {
        name: name.into(),
        description: description.into(),
        version: "1.0.0".into(),
        mode: "live-ecosystem".into(),
        domain: "ecology".into(),
        nodes: vec![],
        edges: vec![],
    }
}

/// Create a scenario node.
fn node(id: &str, name: &str, node_type: &str, capabilities: &[&str]) -> ScenarioNode {
    ScenarioNode {
        id: id.into(),
        name: name.into(),
        node_type: node_type.into(),
        family: crate::PRIMAL_NAME.into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: capabilities.iter().map(|s| (*s).into()).collect(),
        data_channels: vec![],
        scientific_ranges: vec![],
    }
}

/// Create a directed edge.
fn edge(from: &str, to: &str, label: &str) -> ScenarioEdge {
    ScenarioEdge {
        from: from.into(),
        to: to.into(),
        edge_type: "data_flow".into(),
        label: label.into(),
    }
}

/// Create a `TimeSeries` channel.
fn timeseries(
    id: &str,
    label: &str,
    x_label: &str,
    y_label: &str,
    unit: &str,
    x_values: &[f64],
    y_values: &[f64],
) -> DataChannel {
    DataChannel::TimeSeries {
        id: id.into(),
        label: label.into(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        unit: unit.into(),
        x_values: x_values.to_vec(),
        y_values: y_values.to_vec(),
    }
}

/// Create a `Bar` channel.
fn bar(
    id: &str,
    label: &str,
    categories: &[impl AsRef<str>],
    values: &[f64],
    unit: &str,
) -> DataChannel {
    DataChannel::Bar {
        id: id.into(),
        label: label.into(),
        categories: categories.iter().map(|s| s.as_ref().into()).collect(),
        values: values.to_vec(),
        unit: unit.into(),
    }
}

/// Create a `Gauge` channel.
#[expect(clippy::too_many_arguments, reason = "mirrors petalTongue schema")]
fn gauge(
    id: &str,
    label: &str,
    value: f64,
    min: f64,
    max: f64,
    unit: &str,
    normal_range: [f64; 2],
    warning_range: [f64; 2],
) -> DataChannel {
    DataChannel::Gauge {
        id: id.into(),
        label: label.into(),
        value,
        min,
        max,
        unit: unit.into(),
        normal_range,
        warning_range,
    }
}

/// Create a `Heatmap` channel.
fn heatmap(
    id: &str,
    label: &str,
    x_labels: &[String],
    y_labels: &[String],
    values: &[f64],
    unit: &str,
) -> DataChannel {
    DataChannel::Heatmap {
        id: id.into(),
        label: label.into(),
        x_labels: x_labels.to_vec(),
        y_labels: y_labels.to_vec(),
        values: values.to_vec(),
        unit: unit.into(),
    }
}

/// Create a `Scatter` channel.
#[expect(clippy::too_many_arguments, reason = "mirrors petalTongue schema")]
fn scatter(
    id: &str,
    label: &str,
    x: &[f64],
    y: &[f64],
    point_labels: &[impl AsRef<str>],
    x_label: &str,
    y_label: &str,
    unit: &str,
) -> DataChannel {
    DataChannel::Scatter {
        id: id.into(),
        label: label.into(),
        x: x.to_vec(),
        y: y.to_vec(),
        point_labels: point_labels.iter().map(|s| s.as_ref().into()).collect(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        unit: unit.into(),
    }
}

/// Create a `Scatter3D` channel.
#[expect(clippy::too_many_arguments, reason = "mirrors petalTongue schema")]
fn scatter3d(
    id: &str,
    label: &str,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    point_labels: &[impl AsRef<str>],
    x_label: &str,
    y_label: &str,
    z_label: &str,
    unit: &str,
) -> DataChannel {
    DataChannel::Scatter3D {
        id: id.into(),
        label: label.into(),
        x: x.to_vec(),
        y: y.to_vec(),
        z: z.to_vec(),
        point_labels: point_labels.iter().map(|s| s.as_ref().into()).collect(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        z_label: z_label.into(),
        unit: unit.into(),
    }
}

/// Create a `FieldMap` channel.
#[expect(
    dead_code,
    reason = "spatial field scenarios (soil pore geometry, kriging output) not yet implemented; \
              helper completes the DataChannel constructor set and is tested via visualization::tests"
)]
pub(crate) fn fieldmap(
    id: &str,
    label: &str,
    grid_x: &[f64],
    grid_y: &[f64],
    values: &[f64],
    unit: &str,
) -> DataChannel {
    DataChannel::FieldMap {
        id: id.into(),
        label: label.into(),
        grid_x: grid_x.to_vec(),
        grid_y: grid_y.to_vec(),
        values: values.to_vec(),
        unit: unit.into(),
    }
}

/// Create a `Spectrum` channel.
pub(crate) fn spectrum(
    id: &str,
    label: &str,
    unit: &str,
    frequencies: &[f64],
    amplitudes: &[f64],
) -> DataChannel {
    DataChannel::Spectrum {
        id: id.into(),
        label: label.into(),
        unit: unit.into(),
        frequencies: frequencies.to_vec(),
        amplitudes: amplitudes.to_vec(),
    }
}

/// Create a `Distribution` channel.
fn distribution(
    id: &str,
    label: &str,
    unit: &str,
    values: &[f64],
    mean: f64,
    std: f64,
) -> DataChannel {
    DataChannel::Distribution {
        id: id.into(),
        label: label.into(),
        unit: unit.into(),
        values: values.to_vec(),
        mean,
        std,
    }
}

/// Build a combined all-domains scenario.
///
/// Merges ecology and dynamics scenarios into a single graph.
#[must_use]
pub fn full_pipeline_scenario(
    samples: &[Vec<f64>],
    sample_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Full Pipeline",
        "Combined ecology + dynamics + ordination scenario",
    );

    let (eco, mut eco_edges) = ecology_scenario(samples, sample_labels);
    let (dyn_s, mut dyn_edges) = dynamics_scenario();

    for n in eco.nodes {
        s.nodes.push(n);
    }
    for n in dyn_s.nodes {
        s.nodes.push(n);
    }

    let mut edges = Vec::new();
    edges.append(&mut eco_edges);
    edges.append(&mut dyn_edges);
    edges.push(edge("diversity", "qs_ode", "diversity → QS dynamics"));

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_sets_defaults() {
        let s = scaffold("test", "a test");
        assert_eq!(s.name, "test");
        assert_eq!(s.domain, "ecology");
        assert!(s.nodes.is_empty());
    }

    #[test]
    fn node_creates_wetspring_family() {
        let n = node("x", "X", "compute", &["cap"]);
        assert_eq!(n.family, "wetspring");
        assert_eq!(n.health, 100);
    }

    #[test]
    fn full_pipeline_combines() {
        let samples = vec![vec![10.0, 20.0, 30.0]];
        let labels = vec!["S1".into()];
        let (scenario, edges) = full_pipeline_scenario(&samples, &labels);
        assert!(scenario.nodes.len() >= 2);
        assert!(!edges.is_empty());
    }
}
