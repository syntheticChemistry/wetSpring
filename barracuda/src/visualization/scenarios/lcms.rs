// SPDX-License-Identifier: AGPL-3.0-or-later
//! LC-MS / PFAS scenario builders: spectral matching, tolerance search,
//! and combined PFAS overview.

use crate::bio::spectral_match;
use crate::bio::tolerance_search;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, edge, gauge, node, scaffold, scatter};

/// Spectral matching scenario.
///
/// Runs cosine similarity between a query and reference spectrum,
/// visualising match scores and peak correspondence.
#[must_use]
pub fn spectral_match_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Spectral Matching",
        "Cosine similarity between query and reference MS2 spectra",
    );

    let query_mz = vec![100.05, 150.08, 200.12, 250.15, 300.18];
    let query_int = vec![1000.0, 500.0, 800.0, 300.0, 600.0];
    let ref_mz = vec![100.06, 150.07, 200.13, 275.20, 300.17];
    let ref_int = vec![900.0, 600.0, 750.0, 400.0, 550.0];

    let result = spectral_match::cosine_similarity(&query_mz, &query_int, &ref_mz, &ref_int, 0.1);

    let mut match_node = node(
        "spectral_match",
        "Spectral Matching",
        "compute",
        &["science.spectral_match"],
    );

    match_node.data_channels.push(scatter(
        "cosine_vs_mz",
        "Cosine Score vs m/z",
        &query_mz,
        &query_int,
        &query_mz
            .iter()
            .map(|m| format!("{m:.2}"))
            .collect::<Vec<_>>(),
        "m/z",
        "Intensity",
        "AU",
    ));

    let peak_labels: Vec<String> = (0..result.matched_peaks)
        .map(|i| format!("peak_{i}"))
        .collect();
    let peak_scores: Vec<f64> = vec![result.score; result.matched_peaks];
    match_node.data_channels.push(bar(
        "top_matches",
        "Matched Peaks",
        &peak_labels,
        &peak_scores,
        "cosine",
    ));

    match_node.data_channels.push(gauge(
        "match_score",
        "Cosine Match Score",
        result.score,
        0.0,
        1.0,
        "cosine",
        [0.7, 1.0],
        [0.4, 0.7],
    ));
    s.nodes.push(match_node);
    (s, vec![])
}

/// Tolerance search / PFAS fragment screening scenario.
///
/// Screens an m/z array for PFAS-indicative fragments and visualises
/// hit counts by compound class.
#[must_use]
pub fn tolerance_search_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Tolerance Search",
        "PFAS fragment screening with EPA threshold compliance",
    );

    let fragments = tolerance_search::PfasFragments::default();
    let mz_array = vec![
        50.0,
        fragments.cf2 - 0.001,
        fragments.cf2 + 0.001,
        fragments.c2f4 - 0.002,
        fragments.c2f4 + 0.001,
        fragments.hf - 0.001,
        fragments.hf + 0.002,
        100.0,
        150.0,
        200.0,
    ];
    let mut sorted_mz = mz_array;
    sorted_mz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let cf2_hits = tolerance_search::find_within_da(&sorted_mz, fragments.cf2, 0.01);
    let c2f4_hits = tolerance_search::find_within_da(&sorted_mz, fragments.c2f4, 0.01);
    let hf_hits = tolerance_search::find_within_da(&sorted_mz, fragments.hf, 0.01);

    let mut tol_node = node(
        "tolerance_search",
        "Tolerance Search",
        "compute",
        &["science.tolerance_search"],
    );

    #[expect(clippy::cast_precision_loss)] // hit counts are tiny (< 100)
    let hit_counts = [
        cf2_hits.len() as f64,
        c2f4_hits.len() as f64,
        hf_hits.len() as f64,
    ];
    tol_node.data_channels.push(bar(
        "hit_counts",
        "Fragment Hit Counts",
        &["CF₂⁺", "C₂F₄⁺", "HF⁺"],
        &hit_counts,
        "hits",
    ));

    let epa_ppt_threshold = 70.0;
    tol_node.data_channels.push(gauge(
        "epa_threshold",
        "EPA Advisory (ppt)",
        epa_ppt_threshold,
        0.0,
        200.0,
        "ppt",
        [0.0, 70.0],
        [70.0, 200.0],
    ));
    s.nodes.push(tol_node);
    (s, vec![])
}

/// Combined PFAS overview scenario.
///
/// Merges spectral matching and tolerance search into a single decision
/// pipeline with classification and confidence gauges.
#[must_use]
pub fn pfas_overview_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "PFAS Detection Overview",
        "Combined spectral match → tolerance search → classification pipeline",
    );

    let (sm, _) = spectral_match_scenario();
    let (ts, _) = tolerance_search_scenario();

    for n in sm.nodes {
        s.nodes.push(n);
    }
    for n in ts.nodes {
        s.nodes.push(n);
    }

    let mut class_node = node(
        "pfas_classification",
        "PFAS Classification",
        "compute",
        &["science.decision_tree"],
    );
    class_node.data_channels.push(bar(
        "classifications",
        "PFAS Classifications",
        &["PFOS", "PFOA", "GenX", "Other PFAS", "Non-PFAS"],
        &[3.0, 2.0, 1.0, 4.0, 15.0],
        "count",
    ));
    class_node.data_channels.push(gauge(
        "pfas_confidence",
        "Classification Confidence",
        0.87,
        0.0,
        1.0,
        "ratio",
        [0.8, 1.0],
        [0.5, 0.8],
    ));
    s.nodes.push(class_node);

    let edges = vec![
        edge("spectral_match", "tolerance_search", "candidate m/z"),
        edge("tolerance_search", "pfas_classification", "PFAS fragments"),
    ];
    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_match_builds() {
        let (s, _) = spectral_match_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(s.nodes[0].data_channels.len() >= 2);
    }

    #[test]
    fn tolerance_search_builds() {
        let (s, _) = tolerance_search_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn pfas_overview_builds() {
        let (s, edges) = pfas_overview_scenario();
        assert!(s.nodes.len() >= 3);
        assert_eq!(edges.len(), 2);
    }
}
