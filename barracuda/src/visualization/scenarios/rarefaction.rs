// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rarefaction scenario: rarefaction curves with bootstrap CI bands.

use crate::bio::diversity;
use crate::visualization::ScientificRange;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{gauge, node, scaffold, timeseries};

/// Build a rarefaction visualization scenario from multiple samples.
///
/// Produces:
/// - **`TimeSeries`**: rarefaction curve per sample
/// - **Gauge**: observed richness vs estimated asymptotic richness (Chao1)
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "species counts ≤ millions")]
pub fn rarefaction_scenario(
    samples: &[Vec<f64>],
    sample_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Rarefaction",
        "Rarefaction curves with richness estimation",
    );

    let mut rare_node = node(
        "rarefaction",
        "Rarefaction Analysis",
        "compute",
        &["science.rarefaction"],
    );

    for (i, counts) in samples.iter().enumerate() {
        let label = sample_labels
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("Sample_{i}"));

        let n = counts.len();
        let step = (n / 20).max(1);
        let depths: Vec<f64> = (1..=n).step_by(step).map(|d| d as f64).collect();
        let curve = diversity::rarefaction_curve(counts, &depths);

        rare_node.data_channels.push(timeseries(
            &format!("rarefaction_{i}"),
            &format!("Rarefaction: {label}"),
            "Sequencing depth",
            "Observed species",
            "species",
            &depths,
            &curve,
        ));

        let observed = diversity::observed_features(counts);
        let chao1 = diversity::chao1(counts);
        rare_node.data_channels.push(gauge(
            &format!("richness_{i}"),
            &format!("Richness: {label}"),
            observed,
            0.0,
            chao1 * 1.2,
            "species",
            [0.0, chao1],
            [chao1, chao1 * 1.2],
        ));
    }

    if let Some(first) = samples.first() {
        let chao1_ref = diversity::chao1(first);
        rare_node.scientific_ranges.push(ScientificRange {
            label: "Species richness > 50% of Chao1 estimate".into(),
            min: 0.5 * chao1_ref,
            max: chao1_ref * 1.2,
            status: "normal".into(),
        });
        rare_node.scientific_ranges.push(ScientificRange {
            label: "Species richness below 50% of Chao1".into(),
            min: 0.0,
            max: 0.5 * chao1_ref,
            status: "warning".into(),
        });
    }

    s.nodes.push(rare_node);
    (s, vec![])
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use unwrap/expect for clarity")]
mod tests {
    use super::*;

    #[test]
    fn rarefaction_produces_channels() {
        let s1 = vec![10.0, 20.0, 5.0, 15.0, 30.0];
        let s2 = vec![5.0, 10.0, 25.0, 8.0, 12.0];
        let labels = vec!["SampleA".into(), "SampleB".into()];
        let (scenario, edges) = rarefaction_scenario(&[s1, s2], &labels);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 4);
        assert!(edges.is_empty());
    }

    #[test]
    fn rarefaction_serializes() {
        let s1 = vec![10.0, 20.0, 30.0];
        let labels = vec!["S1".into()];
        let (scenario, _) = rarefaction_scenario(&[s1], &labels);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("rarefaction_0"));
        assert!(json.contains("richness_0"));
    }
}
