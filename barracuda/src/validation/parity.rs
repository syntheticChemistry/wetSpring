// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-tier parity results — structured three-layer proof.
//!
//! Layers:
//! - L1: Python baseline (breseq output)
//! - L2: Sovereign Rust pipeline (local execution)
//! - L3: Primal composition (same pipeline via IPC)
//!
//! The [`ParityResult`] is the contract lithoSpore consumes for
//! USB artifact validation.

use serde::{Deserialize, Serialize};

/// One clone's parity comparison across tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneParity {
    /// Clone name (e.g. `"REL1164M"`).
    pub clone_name: String,
    /// SRA accession (e.g. `"SRR032370"`).
    pub accession: String,
    /// LTEE generation number.
    pub generation: u32,
    /// L1: Python/breseq variant count (`None` if baseline absent).
    pub l1_variants: Option<u32>,
    /// L2: Sovereign Rust variant count.
    pub l2_variants: u32,
    /// L3: Primal-composed variant count (`None` if not yet run).
    pub l3_variants: Option<u32>,
    /// Positional overlap between L1 and L2 variant calls.
    pub l1_l2_position_matches: Option<u32>,
    /// L1 calls not found in L2 (false negatives from L2 perspective).
    pub l1_only: Option<u32>,
    /// L2 calls not found in L1 (false positives from L1 perspective).
    pub l2_only: Option<u32>,
    /// Wall-clock time for this clone in seconds.
    pub wall_seconds: f64,
    /// Free-form annotation for exceptional results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Aggregate parity result for a complete dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityResult {
    /// Dataset identifier (e.g. `"barrick_2009_sovereign_resequencing"`).
    pub dataset_id: String,
    /// Producing spring name.
    pub spring: String,
    /// Spring version at time of production.
    pub spring_version: String,
    /// Reference genome identifier.
    pub reference: String,
    /// L1 baseline tool name (e.g. `"breseq"`).
    pub l1_tool: String,
    /// L1 baseline tool version.
    pub l1_tool_version: String,
    /// L2 sovereign tool name.
    pub l2_tool: String,
    /// L2 sovereign tool version.
    pub l2_tool_version: String,
    /// Caller configuration snapshot for reproducibility.
    pub caller_config: serde_json::Value,
    /// Per-clone parity comparisons.
    pub clones: Vec<CloneParity>,
    /// Aggregate statistics.
    pub summary: ParitySummary,
    /// ISO 8601 timestamp.
    pub timestamp: String,
}

/// Aggregate parity statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParitySummary {
    /// Total clones in the dataset.
    pub clones_total: u32,
    /// Clones with an L1 baseline available.
    pub clones_with_baseline: u32,
    /// Sum of L1 variants across all clones with baselines.
    pub total_l1_variants: u32,
    /// Sum of L2 variants across all clones.
    pub total_l2_variants: u32,
    /// Sum of position matches across all clones.
    pub total_position_matches: u32,
    /// Total wall-clock seconds for all clones.
    pub total_wall_seconds: f64,
    /// Ratio of total L2 to total L1 variants.
    pub l2_over_l1_ratio: f64,
    /// Human-readable assessment of parity quality.
    pub assessment: String,
}

impl ParityResult {
    /// Serialize to pretty JSON for lithoSpore consumption.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}
