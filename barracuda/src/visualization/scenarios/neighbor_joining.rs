// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neighbor-joining tree scenario: distance heatmap, tree metrics.
//!
//! Wraps [`crate::bio::neighbor_joining`] into a petalTongue-ready scenario
//! with pairwise distance heatmap and tree statistics.

use crate::bio::neighbor_joining;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{gauge, heatmap, node, scaffold};

/// Build a neighbor-joining tree scenario from a full distance matrix.
///
/// The distance matrix must be n×n (full square) and `labels` must have length n.
///
/// Constructs the NJ tree and visualizes:
/// - **Heatmap**: pairwise distance matrix
/// - **Gauge**: number of join operations
#[must_use]
#[expect(clippy::cast_precision_loss)]
pub fn neighbor_joining_scenario(
    distance_matrix: &[f64],
    labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let n = labels.len();

    let tree = neighbor_joining::neighbor_joining(distance_matrix, labels);

    let mut s = scaffold(
        "wetSpring Neighbor-Joining Tree",
        &format!("{n} taxa, {} joins", tree.n_joins),
    );

    let mut nj_node = node("nj_tree", "Neighbor-Joining Tree", "compute", &["phylo.nj"]);

    // Distance matrix heatmap
    nj_node.data_channels.push(heatmap(
        "distance_matrix",
        "Pairwise Distances",
        labels,
        labels,
        distance_matrix,
        "distance",
    ));

    // Joins gauge
    nj_node.data_channels.push(gauge(
        "n_joins",
        "Join Operations",
        tree.n_joins as f64,
        0.0,
        n as f64,
        "joins",
        [0.0, n as f64],
        [n as f64, n as f64 * 2.0],
    ));

    nj_node.scientific_ranges.push(ScientificRange {
        label: "Normal tree size".into(),
        min: 0.0,
        max: n as f64,
        status: "normal".into(),
    });

    s.nodes.push(nj_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nj_scenario_basic() {
        // 3×3 distance matrix (symmetric)
        let dm = vec![0.0, 0.2, 0.3, 0.2, 0.0, 0.4, 0.3, 0.4, 0.0];
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let (scenario, _) = neighbor_joining_scenario(&dm, &labels);
        assert_eq!(scenario.nodes.len(), 1);
        let n = &scenario.nodes[0];
        assert!(n.data_channels.len() >= 2);
        assert!(!n.scientific_ranges.is_empty());
    }

    #[test]
    fn nj_scenario_four_taxa() {
        let dm = vec![
            0.0, 0.1, 0.3, 0.25, 0.1, 0.0, 0.35, 0.2, 0.3, 0.35, 0.0, 0.15, 0.25, 0.2, 0.15, 0.0,
        ];
        let labels = vec!["X".into(), "Y".into(), "Z".into(), "W".into()];
        let (scenario, _) = neighbor_joining_scenario(&dm, &labels);
        assert_eq!(scenario.nodes.len(), 1);
    }
}
