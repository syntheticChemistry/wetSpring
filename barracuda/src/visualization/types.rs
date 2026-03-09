// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wire types for the `petalTongue`-compatible scenario schema.
//!
//! These types serialize to JSON that `petalTongue` can consume directly
//! via `--scenario <path>` or JSON-RPC `visualization.render`. No
//! `petalTongue` crate dependency — integration is via JSON only.

use serde::Serialize;

/// A typed data channel attached to a scenario node.
///
/// Each variant maps to a `petalTongue` `DataBinding` channel type.
/// The `channel_type` tag selects the renderer on the `petalTongue` side.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "channel_type")]
pub enum DataChannel {
    /// Line chart for time series, rarefaction curves, ODE trajectories.
    #[serde(rename = "timeseries")]
    TimeSeries {
        /// Unique channel identifier within the scenario.
        id: String,
        /// Human-readable label.
        label: String,
        /// X-axis label (e.g. "Time (h)", "Sequencing depth").
        x_label: String,
        /// Y-axis label (e.g. "Observed species", "Concentration").
        y_label: String,
        /// Unit for y-axis values.
        unit: String,
        /// X-axis data points.
        x_values: Vec<f64>,
        /// Y-axis data points (same length as `x_values`).
        y_values: Vec<f64>,
    },
    /// Histogram or density plot for distributions.
    #[serde(rename = "distribution")]
    Distribution {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// Unit for values.
        unit: String,
        /// Raw sample values.
        values: Vec<f64>,
        /// Population mean.
        mean: f64,
        /// Population standard deviation.
        std: f64,
    },
    /// Categorical bar chart for diversity metrics, genus abundances.
    #[serde(rename = "bar")]
    Bar {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// Category labels (x-axis).
        categories: Vec<String>,
        /// Values per category (same length as `categories`).
        values: Vec<f64>,
        /// Unit for values.
        unit: String,
    },
    /// Progress/dial for bounded scalar values.
    #[serde(rename = "gauge")]
    Gauge {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// Current value.
        value: f64,
        /// Minimum bound.
        min: f64,
        /// Maximum bound.
        max: f64,
        /// Unit for value.
        unit: String,
        /// Normal (healthy) range `[lo, hi]`.
        normal_range: [f64; 2],
        /// Warning range `[lo, hi]`.
        warning_range: [f64; 2],
    },
    /// 2D matrix visualization (distance matrix, correlation, abundance grid).
    #[serde(rename = "heatmap")]
    Heatmap {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// Column labels.
        x_labels: Vec<String>,
        /// Row labels.
        y_labels: Vec<String>,
        /// Row-major flattened values (`x_labels.len() * y_labels.len()`).
        values: Vec<f64>,
        /// Unit for cell values.
        unit: String,
    },
    /// 2D scatter plot (`PCoA` ordination, KMD plots).
    #[serde(rename = "scatter")]
    Scatter {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// X coordinates.
        x: Vec<f64>,
        /// Y coordinates.
        y: Vec<f64>,
        /// Per-point labels (optional, empty if unlabeled).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        point_labels: Vec<String>,
        /// X-axis label.
        x_label: String,
        /// Y-axis label.
        y_label: String,
        /// Unit for axes.
        unit: String,
    },
    /// Power spectrum / FFT amplitude plot (Anderson spectral, HRV, signal).
    #[serde(rename = "spectrum")]
    Spectrum {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// Unit for amplitude values.
        unit: String,
        /// Frequency axis values.
        frequencies: Vec<f64>,
        /// Amplitude axis values (same length as `frequencies`).
        amplitudes: Vec<f64>,
    },
    /// Spatial field map for gridded 2D data (environmental sampling, spatial ecology).
    ///
    /// Per wateringHole `VISUALIZATION_INTEGRATION_GUIDE` v2.0: `grid_x`, `grid_y`,
    /// `values` in row-major order. Enables spatial ecology, ET0 maps, Richards PDE.
    #[serde(rename = "fieldmap")]
    FieldMap {
        /// Unique channel identifier.
        id: String,
        /// Human-readable label.
        label: String,
        /// X-axis grid coordinates.
        grid_x: Vec<f64>,
        /// Y-axis grid coordinates.
        grid_y: Vec<f64>,
        /// Row-major values (`grid_x.len() * grid_y.len()`).
        values: Vec<f64>,
        /// Unit for cell values.
        unit: String,
    },
}

/// Reference range for threshold coloring on gauge/chart overlays.
#[derive(Debug, Clone, Serialize)]
pub struct ScientificRange {
    /// Human-readable label (e.g. "Optimal diversity").
    pub label: String,
    /// Lower bound of the range.
    pub min: f64,
    /// Upper bound of the range.
    pub max: f64,
    /// Severity or status label (`"normal"`, `"warning"`, `"critical"`).
    pub status: String,
}

/// A node in the scenario graph.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioNode {
    /// Unique node identifier.
    pub id: String,
    /// Human-readable node name.
    pub name: String,
    /// Node category (`"compute"`, `"data"`, `"pipeline"`).
    #[serde(rename = "type")]
    pub node_type: String,
    /// Primal family (`"wetspring"`).
    pub family: String,
    /// Node status (`"healthy"`, `"degraded"`).
    pub status: String,
    /// Health score 0–100.
    pub health: u8,
    /// Confidence score 0–100.
    pub confidence: u8,
    /// IPC capabilities this node advertises.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Data channels attached to this node.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub data_channels: Vec<DataChannel>,
    /// Reference ranges for gauge/chart thresholds.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scientific_ranges: Vec<ScientificRange>,
}

/// An edge connecting two scenario nodes.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Edge category (`"data_flow"`, `"validation"`).
    pub edge_type: String,
    /// Human-readable edge label.
    pub label: String,
}

/// Complete scenario — `petalTongue`-compatible wire format.
#[derive(Debug, Clone, Serialize)]
pub struct EcologyScenario {
    /// Scenario title.
    pub name: String,
    /// Scenario description.
    pub description: String,
    /// Schema version.
    pub version: String,
    /// Render mode (`"live-ecosystem"`, `"static"`).
    pub mode: String,
    /// `petalTongue` domain theme (`"ecology"`, `"measurement"`).
    pub domain: String,
    /// Graph nodes.
    pub nodes: Vec<ScenarioNode>,
    /// Graph edges.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<ScenarioEdge>,
}

/// UI configuration for domain-themed rendering.
///
/// Follows healthSpring's `UiConfig` pattern — sent alongside the scenario
/// via `push_render_with_config()` to control petalTongue panel layout,
/// theme, and initial viewport.
#[derive(Debug, Clone, Serialize)]
pub struct UiConfig {
    /// Theme name (`"ecology-dark"`, `"metagenomics"`, `"default"`).
    pub theme: String,
    /// Initial zoom level (`"fit"`, `"100%"`, `"auto"`).
    pub initial_zoom: String,
    /// Panel visibility.
    pub show_panels: ShowPanels,
}

/// Panel visibility configuration for `petalTongue` layout.
#[derive(Debug, Clone, Serialize)]
pub struct ShowPanels {
    /// Show left sidebar (node list, search).
    pub left_sidebar: bool,
    /// Show data inspector panel.
    pub data_inspector: bool,
    /// Show pipeline progress panel.
    pub pipeline_progress: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "ecology-dark".into(),
            initial_zoom: "fit".into(),
            show_panels: ShowPanels {
                left_sidebar: true,
                data_inspector: true,
                pipeline_progress: false,
            },
        }
    }
}
