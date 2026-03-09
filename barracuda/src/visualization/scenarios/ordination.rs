// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ordination scenario: `PCoA` 2D scatter and eigenvalue scree plot.

use crate::bio::pcoa;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, node, scaffold, scatter};

/// Build a `PCoA` ordination scenario from a distance matrix.
///
/// Runs Jacobi eigendecomposition via [`pcoa::pcoa`], produces a 2D scatter
/// (PC1 vs PC2) and a scree bar chart of proportion explained.
///
/// # Errors
///
/// Returns `Err` if `PCoA` fails (e.g. non-square matrix).
pub fn ordination_scenario(
    distance_matrix: &[f64],
    n_samples: usize,
    sample_labels: &[String],
) -> crate::error::Result<(EcologyScenario, Vec<ScenarioEdge>)> {
    let n_axes = 3.min(n_samples.saturating_sub(1));
    let result = pcoa::pcoa(distance_matrix, n_samples, n_axes)?;

    let mut s = scaffold(
        "wetSpring PCoA Ordination",
        "Principal Coordinates Analysis — sample ordination in reduced dimensions",
    );

    let mut pcoa_node = node("pcoa", "PCoA Ordination", "compute", &["science.pcoa"]);

    let x: Vec<f64> = (0..n_samples)
        .map(|i| result.coordinates[i * n_axes])
        .collect();
    let y: Vec<f64> = (0..n_samples)
        .map(|i| {
            if n_axes > 1 {
                result.coordinates[i * n_axes + 1]
            } else {
                0.0
            }
        })
        .collect();

    let x_label = format!(
        "PC1 ({:.1}%)",
        result.proportion_explained.first().unwrap_or(&0.0) * 100.0
    );
    let y_label = format!(
        "PC2 ({:.1}%)",
        result.proportion_explained.get(1).unwrap_or(&0.0) * 100.0
    );

    pcoa_node.data_channels.push(scatter(
        "pcoa_scatter",
        "PCoA Ordination",
        &x,
        &y,
        sample_labels,
        &x_label,
        &y_label,
        "coordinate",
    ));

    let axis_labels: Vec<String> = (1..=n_axes).map(|i| format!("PC{i}")).collect();
    let pct_explained: Vec<f64> = result
        .proportion_explained
        .iter()
        .map(|p| p * 100.0)
        .collect();
    pcoa_node.data_channels.push(bar(
        "scree",
        "Variance Explained",
        &axis_labels,
        &pct_explained,
        "%",
    ));

    s.nodes.push(pcoa_node);
    Ok((s, vec![]))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn ordination_from_distance_matrix() {
        let dm = vec![0.5, 0.8, 0.6]; // condensed upper-triangle for 3 samples
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let (scenario, _) = ordination_scenario(&dm, 3, &labels).unwrap();
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 2);
    }
}
