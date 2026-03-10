// SPDX-License-Identifier: AGPL-3.0-or-later
//! NMF scenario: W/H factor heatmaps and component loading bars.

use barracuda::prelude::BarracudaError;

use crate::bio::nmf::{NmfConfig, NmfResult};
use crate::visualization::ScientificRange;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, heatmap, node, scaffold};

/// Build an NMF decomposition visualization from a pre-computed result.
///
/// Produces:
/// - **Heatmap**: W matrix (samples × components)
/// - **Heatmap**: H matrix (components × features)
/// - **Bar**: top feature loadings per component
#[must_use]
pub fn nmf_scenario(
    result: &NmfResult,
    sample_labels: &[String],
    feature_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring NMF Decomposition",
        "Non-negative Matrix Factorization — W and H factor matrices",
    );

    let mut nmf_node = node("nmf", "NMF Analysis", "compute", &["science.nmf"]);

    let comp_labels: Vec<String> = (0..result.k).map(|c| format!("Component_{c}")).collect();

    nmf_node.data_channels.push(heatmap(
        "w_matrix",
        "W Matrix (Samples × Components)",
        &comp_labels,
        sample_labels,
        &result.w,
        "loading",
    ));

    nmf_node.data_channels.push(heatmap(
        "h_matrix",
        "H Matrix (Components × Features)",
        feature_labels,
        &comp_labels,
        &result.h,
        "loading",
    ));

    for c in 0..result.k {
        let row_start = c * result.n;
        let row_end = row_start + result.n;
        let h_row = &result.h[row_start..row_end];

        let mut indices: Vec<usize> = (0..result.n).collect();
        indices.sort_by(|&a, &b| {
            h_row[b]
                .partial_cmp(&h_row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_n = result.n.min(10);

        let top_labels: Vec<&str> = indices[..top_n]
            .iter()
            .map(|&i| feature_labels.get(i).map_or("?", String::as_str))
            .collect();
        let top_values: Vec<f64> = indices[..top_n].iter().map(|&i| h_row[i]).collect();

        nmf_node.data_channels.push(bar(
            &format!("top_features_comp_{c}"),
            &format!("Top Features: Component {c}"),
            &top_labels,
            &top_values,
            "loading",
        ));
    }

    nmf_node.scientific_ranges.push(ScientificRange {
        label: "Reconstruction error < 0.1".into(),
        min: 0.0,
        max: 0.1,
        status: "normal".into(),
    });
    nmf_node.scientific_ranges.push(ScientificRange {
        label: "Reconstruction error 0.1–0.5".into(),
        min: 0.1,
        max: 0.5,
        status: "warning".into(),
    });

    s.nodes.push(nmf_node);
    (s, vec![])
}

/// Convenience: run NMF and build the scenario.
///
/// # Errors
///
/// Returns `barracuda::BarracudaError` if NMF fails.
pub fn nmf_scenario_from_data(
    data: &[f64],
    m: usize,
    n: usize,
    config: &NmfConfig,
    sample_labels: &[String],
    feature_labels: &[String],
) -> Result<(EcologyScenario, Vec<ScenarioEdge>), BarracudaError> {
    let result = crate::bio::nmf::nmf(data, m, n, config)?;
    Ok(nmf_scenario(&result, sample_labels, feature_labels))
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use unwrap/expect for clarity")]
mod tests {
    use super::*;
    use crate::bio::nmf::{NmfConfig, NmfObjective};
    use crate::tolerances;

    #[test]
    fn nmf_scenario_produces_channels() {
        let data = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 0.5, 0.3, 1.0, 0.8, 0.2, 0.1];
        let config = NmfConfig {
            rank: 2,
            max_iter: 50,
            tol: tolerances::NMF_CONVERGENCE,
            objective: NmfObjective::KlDivergence,
            seed: 42,
        };
        let samples = vec!["S1".into(), "S2".into(), "S3".into(), "S4".into()];
        let features = vec!["F1".into(), "F2".into(), "F3".into()];
        let (scenario, edges) =
            nmf_scenario_from_data(&data, 4, 3, &config, &samples, &features).expect("nmf");
        assert_eq!(scenario.nodes.len(), 1);
        assert!(scenario.nodes[0].data_channels.len() >= 4);
        assert!(edges.is_empty());
    }

    #[test]
    fn nmf_serializes() {
        let result = NmfResult {
            w: vec![0.5, 0.3, 0.8, 0.1],
            h: vec![0.6, 0.4, 0.2, 0.9, 0.1, 0.7],
            m: 2,
            k: 2,
            n: 3,
            errors: vec![1.0, 0.5, 0.1],
        };
        let samples = vec!["A".into(), "B".into()];
        let features = vec!["X".into(), "Y".into(), "Z".into()];
        let (scenario, _) = nmf_scenario(&result, &samples, &features);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("w_matrix"));
        assert!(json.contains("h_matrix"));
        assert!(json.contains("top_features_comp_"));
    }
}
