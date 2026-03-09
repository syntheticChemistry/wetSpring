// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ecology scenario: diversity metrics, rarefaction, Bray-Curtis heatmap.

use crate::bio::diversity;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, edge, gauge, heatmap, node, scaffold, timeseries};

/// Build an ecology scenario from community abundance data.
///
/// Computes Shannon, Simpson, Chao1, Pielou evenness, rarefaction curve,
/// and (if multiple samples) Bray-Curtis distance matrix using live
/// `barraCuda` math.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "species counts ≤ millions")]
pub fn ecology_scenario(
    samples: &[Vec<f64>],
    sample_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Ecology",
        "Alpha/beta diversity, rarefaction, and Bray-Curtis dissimilarity",
    );

    let mut diversity_node = node(
        "diversity",
        "Alpha Diversity",
        "compute",
        &["science.diversity"],
    );

    if let Some(counts) = samples.first() {
        let shannon = diversity::shannon(counts);
        let simpson = diversity::simpson(counts);
        let pielou = diversity::pielou_evenness(counts);
        let chao1 = diversity::chao1(counts);
        let obs = diversity::observed_features(counts);

        diversity_node.data_channels.push(bar(
            "alpha_metrics",
            "Alpha Diversity Metrics",
            &["Shannon H'", "Simpson D", "Pielou J", "Chao1"],
            &[shannon, simpson, pielou, chao1],
            "index",
        ));

        diversity_node.data_channels.push(gauge(
            "observed_features",
            "Observed Features",
            obs,
            0.0,
            counts.len() as f64,
            "species",
            [0.0, counts.len() as f64],
            [0.0, 0.0],
        ));

        let step = (counts.len() / 20).max(1);
        let depths: Vec<f64> = (1..=counts.len()).step_by(step).map(|d| d as f64).collect();
        let curve = diversity::rarefaction_curve(counts, &depths);
        diversity_node.data_channels.push(timeseries(
            "rarefaction",
            "Rarefaction Curve",
            "Sequencing depth",
            "Observed species",
            "species",
            &depths,
            &curve,
        ));

        diversity_node.scientific_ranges.push(ScientificRange {
            label: "High diversity".into(),
            min: 3.0,
            max: 6.0,
            status: "normal".into(),
        });
    }

    s.nodes.push(diversity_node);

    if samples.len() > 1 {
        let mut beta_node = node(
            "beta_diversity",
            "Beta Diversity",
            "compute",
            &["science.beta_diversity"],
        );

        let bc_matrix = diversity::bray_curtis_matrix(samples);
        beta_node.data_channels.push(heatmap(
            "bray_curtis",
            "Bray-Curtis Dissimilarity",
            sample_labels,
            sample_labels,
            &bc_matrix,
            "BC index",
        ));

        s.nodes.push(beta_node);
    }

    let edges = if samples.len() > 1 {
        vec![edge(
            "diversity",
            "beta_diversity",
            "alpha → beta diversity",
        )]
    } else {
        vec![]
    };

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecology_single_sample() {
        let counts = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (scenario, edges) = ecology_scenario(&[counts], &["S1".into()]);
        assert_eq!(scenario.nodes.len(), 1);
        assert!(edges.is_empty());
        assert!(!scenario.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn ecology_multiple_samples() {
        let s1 = vec![10.0, 20.0, 30.0];
        let s2 = vec![15.0, 25.0, 5.0];
        let labels = vec!["Sample1".into(), "Sample2".into()];
        let (scenario, edges) = ecology_scenario(&[s1, s2], &labels);
        assert_eq!(scenario.nodes.len(), 2);
        assert_eq!(edges.len(), 1);
    }
}
