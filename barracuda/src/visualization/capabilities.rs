// SPDX-License-Identifier: AGPL-3.0-or-later
//! Songbird capability announcement for wetSpring visualization.
//!
//! Declares which visualization capabilities this spring can produce so that
//! `petalTongue` and `biomeOS` can discover and route rendering requests
//! automatically. No compile-time `petalTongue` dependency — discovery is
//! via Songbird JSON-RPC at runtime.

use serde::Serialize;

/// All visualization capabilities advertised by wetSpring.
pub const VISUALIZATION_CAPABILITIES: &[&str] = &[
    "visualization.ecology.diversity",
    "visualization.ecology.ordination",
    "visualization.ecology.dynamics",
    "visualization.ecology.chemistry",
    "visualization.ecology.pangenome",
    "visualization.ecology.hmm",
    "visualization.ecology.stochastic",
    "visualization.ecology.similarity",
    "visualization.ecology.rarefaction",
    "visualization.ecology.nmf",
    "visualization.ecology.streaming",
    "visualization.ecology.benchmarks",
    "visualization.ecology.anderson",
    "visualization.metalforge.inventory",
    "visualization.metalforge.dispatch",
    "visualization.metalforge.nucleus",
];

/// Songbird-compatible capability announcement payload.
#[derive(Debug, Clone, Serialize)]
pub struct VisualizationAnnouncement {
    /// Primal name.
    pub primal: String,
    /// Visualization domain theme.
    pub domain: String,
    /// Schema version.
    pub version: String,
    /// List of visualization capabilities.
    pub capabilities: Vec<String>,
    /// Supported data channel types.
    pub channel_types: Vec<String>,
    /// Whether streaming (`visualization.render.stream`) is supported.
    pub supports_streaming: bool,
}

/// Build the visualization announcement for Songbird registration.
#[must_use]
pub fn announcement() -> VisualizationAnnouncement {
    VisualizationAnnouncement {
        primal: "wetspring".into(),
        domain: "ecology".into(),
        version: "1.0.0".into(),
        capabilities: VISUALIZATION_CAPABILITIES
            .iter()
            .map(|s| (*s).into())
            .collect(),
        channel_types: vec![
            "timeseries".into(),
            "distribution".into(),
            "bar".into(),
            "gauge".into(),
            "heatmap".into(),
            "scatter".into(),
            "spectrum".into(),
        ],
        supports_streaming: true,
    }
}

/// Serialize the announcement to JSON for Songbird registration.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails.
pub fn announcement_json() -> serde_json::Result<String> {
    serde_json::to_string_pretty(&announcement())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap/expect for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn announcement_has_all_fields() {
        let a = announcement();
        assert_eq!(a.primal, "wetspring");
        assert_eq!(a.domain, "ecology");
        assert!(a.supports_streaming);
        assert!(a.capabilities.len() >= 10);
        assert!(a.channel_types.contains(&"spectrum".into()));
    }

    #[test]
    fn announcement_json_roundtrip() {
        let json = announcement_json().expect("serialize");
        assert!(json.contains("\"primal\": \"wetspring\""));
        assert!(json.contains("visualization.ecology.diversity"));
        assert!(json.contains("\"supports_streaming\": true"));
    }

    #[test]
    fn capabilities_constant_not_empty() {
        assert!(!VISUALIZATION_CAPABILITIES.is_empty());
        for cap in VISUALIZATION_CAPABILITIES {
            assert!(
                cap.starts_with("visualization."),
                "capability should start with 'visualization.': {cap}"
            );
        }
    }
}
