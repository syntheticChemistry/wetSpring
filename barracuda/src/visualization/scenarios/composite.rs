// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composite full-pipeline scenarios that merge multiple domain tracks
//! into unified views, following healthSpring's `full_study()` pattern.

use crate::bio::diversity;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{edge, scaffold};

/// Full 16S rRNA analysis pipeline scenario.
///
/// Composes quality → DADA2 → taxonomy → diversity into a single graph
/// with data-flow edges.
#[must_use]
pub fn full_16s_scenario(
    samples: &[Vec<f64>],
    sample_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Full 16S Pipeline",
        "Quality → DADA2 → Taxonomy → Diversity",
    );

    let (qual, _) = super::quality_scenario();
    let (dada2, _) = super::dada2_scenario();
    let (tax, _) = super::taxonomy_scenario();
    let (eco, _) = super::ecology_scenario(samples, sample_labels);

    for n in qual.nodes {
        s.nodes.push(n);
    }
    for n in dada2.nodes {
        s.nodes.push(n);
    }
    for n in tax.nodes {
        s.nodes.push(n);
    }
    for n in eco.nodes {
        s.nodes.push(n);
    }

    if samples.len() > 1 {
        let bc_matrix = diversity::bray_curtis_matrix(samples);
        if let Ok((ord, _)) = super::ordination_scenario(&bc_matrix, samples.len(), sample_labels) {
            for n in ord.nodes {
                s.nodes.push(n);
            }
        }
    }

    let edges = vec![
        edge("quality", "dada2", "filtered reads"),
        edge("dada2", "taxonomy", "ASVs"),
        edge("taxonomy", "diversity", "classified OTUs"),
    ];
    (s, edges)
}

/// Full PFAS detection pipeline scenario.
///
/// Composes spectral match → tolerance search → classification into a
/// unified analytical chemistry pipeline.
#[must_use]
pub fn full_pfas_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Full PFAS Pipeline",
        "Spectral Match → Tolerance Search → Classification",
    );

    let (pfas, pfas_edges) = super::pfas_overview_scenario();

    for n in pfas.nodes {
        s.nodes.push(n);
    }

    (s, pfas_edges)
}

/// Full quorum-sensing model suite scenario.
///
/// Combines all 5 ODE models into a comprehensive QS biology dashboard.
#[must_use]
pub fn full_qs_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Full QS Model Suite",
        "All ODE models — phage defense, bistable, cooperation, multi-signal, capacitor",
    );

    let (phage, _) = super::phage_defense_scenario();
    let (bistable, _) = super::bistable_scenario();
    let (coop, _) = super::cooperation_scenario();
    let (multi, _) = super::multi_signal_scenario();
    let (cap, _) = super::capacitor_scenario();

    for n in phage.nodes {
        s.nodes.push(n);
    }
    for n in bistable.nodes {
        s.nodes.push(n);
    }
    for n in coop.nodes {
        s.nodes.push(n);
    }
    for n in multi.nodes {
        s.nodes.push(n);
    }
    for n in cap.nodes {
        s.nodes.push(n);
    }

    let edges = vec![
        edge("phage_defense", "bistable", "defense → switch coupling"),
        edge("bistable", "cooperation", "switch state"),
        edge("cooperation", "multi_signal", "cooperator density"),
        edge("multi_signal", "capacitor", "QS signal levels"),
    ];
    (s, edges)
}

/// Full ecology dashboard scenario — the "scientist dashboard".
///
/// Merges every domain track into a single comprehensive view: 16S pipeline,
/// ODE models, phylogenetics, LC-MS, population genomics, and ML models.
#[must_use]
pub fn full_ecology_scenario(
    samples: &[Vec<f64>],
    sample_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Scientist Dashboard",
        "Full wetSpring ecology — all compute domains in one view",
    );

    let (pipeline_16s, mut edges_16s) = full_16s_scenario(samples, sample_labels);
    let (qs, mut edges_qs) = full_qs_scenario();
    let (phylo, _) = super::felsenstein_scenario();
    let (dnds, _) = super::dnds_scenario();
    let (kmer, _) = super::kmer_spectrum_scenario();
    let (lcms, mut edges_lcms) = full_pfas_scenario();
    let (ml_dt, _) = super::decision_tree_scenario();
    let (ml_esn, _) = super::esn_scenario();

    for n in pipeline_16s.nodes {
        s.nodes.push(n);
    }
    for n in qs.nodes {
        s.nodes.push(n);
    }
    for n in phylo.nodes {
        s.nodes.push(n);
    }
    for n in dnds.nodes {
        s.nodes.push(n);
    }
    for n in kmer.nodes {
        s.nodes.push(n);
    }
    for n in lcms.nodes {
        s.nodes.push(n);
    }
    for n in ml_dt.nodes {
        s.nodes.push(n);
    }
    for n in ml_esn.nodes {
        s.nodes.push(n);
    }

    let mut all_edges = Vec::new();
    all_edges.append(&mut edges_16s);
    all_edges.append(&mut edges_qs);
    all_edges.append(&mut edges_lcms);
    all_edges.push(edge("diversity", "phage_defense", "ecology → QS dynamics"));
    all_edges.push(edge(
        "taxonomy",
        "felsenstein",
        "classified → phylogenetics",
    ));
    all_edges.push(edge(
        "kmer_spectrum",
        "decision_tree",
        "k-mer features → ML",
    ));

    (s, all_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_16s_builds() {
        let samples = vec![vec![10.0, 20.0, 30.0], vec![15.0, 25.0, 5.0]];
        let labels = vec!["S1".into(), "S2".into()];
        let (s, edges) = full_16s_scenario(&samples, &labels);
        assert!(s.nodes.len() >= 4);
        assert!(!edges.is_empty());
    }

    #[test]
    fn full_pfas_builds() {
        let (s, edges) = full_pfas_scenario();
        assert!(s.nodes.len() >= 3);
        assert!(!edges.is_empty());
    }

    #[test]
    fn full_qs_builds() {
        let (s, edges) = full_qs_scenario();
        assert!(s.nodes.len() >= 5);
        assert!(!edges.is_empty());
    }

    #[test]
    fn full_ecology_builds() {
        let samples = vec![vec![10.0, 20.0, 30.0]];
        let labels = vec!["S1".into()];
        let (s, edges) = full_ecology_scenario(&samples, &labels);
        assert!(s.nodes.len() >= 10);
        assert!(!edges.is_empty());
    }
}
