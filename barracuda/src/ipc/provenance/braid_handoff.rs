// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ferment transcript braid export — portable provenance handoffs for downstream consumers.
//!
//! Implements the wire format defined in the lithoSpore ferment transcript braid
//! handoff contract (`LITHOSPORE_FERMENT_TRANSCRIPT_BRAID_HANDOFF_MAY17_2026.md`).
//!
//! A ferment transcript is the provenance record of upstream computation: the spring
//! does the fermentation (processing raw data into validated results), and the
//! guideStone carries the transcript. The handoff JSON is self-describing — a USB
//! artifact can document the provenance chain airgapped and verify it when reconnected.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

/// Ferment transcript braid — the portable provenance handoff for downstream consumers.
///
/// Wire format per the lithoSpore contract:
///
/// ```json
/// {
///   "dataset_id": "tenaillon_2016_genomes",
///   "spring": "wetSpring",
///   "spring_version": "0.3.0",
///   "braid_id": "braid-abc123...",
///   "dag_session_id": "dag-def456...",
///   "dag_merkle_root": "789abc...",
///   "spine_id": "spine-xyz...",
///   "computation": { ... },
///   "summary_blake3": "...",
///   "timestamp": "2026-05-17T12:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FermentTranscriptBraid {
    /// Identifier for the dataset this braid covers.
    pub dataset_id: String,
    /// The spring that produced the computation.
    pub spring: String,
    /// Spring version at time of computation.
    pub spring_version: String,
    /// sweetGrass braid ID (attribution record).
    pub braid_id: String,
    /// rhizoCrypt DAG session ID (computation DAG).
    pub dag_session_id: String,
    /// Merkle root covering the full computation DAG.
    pub dag_merkle_root: String,
    /// loamSpine ledger entry ID (permanent commit).
    pub spine_id: String,
    /// Computation metadata (tool, version, inputs, outputs, timing).
    pub computation: ComputationMetadata,
    /// BLAKE3 hash of the summary statistics handed off.
    pub summary_blake3: String,
    /// ISO 8601 timestamp of braid creation.
    pub timestamp: String,
}

/// Metadata about the upstream computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetadata {
    /// Computational tool used (e.g., `"breseq"`).
    pub tool: String,
    /// Tool version (e.g., `"0.38.1"`).
    pub tool_version: String,
    /// Input data accession or identifier.
    pub input_accession: String,
    /// BLAKE3 hash of the input data.
    pub input_blake3: String,
    /// BLAKE3 hash of the output data.
    pub output_blake3: String,
    /// Wall-clock time for the computation in seconds.
    pub wall_time_seconds: u64,
    /// Number of items processed (e.g., genome count).
    pub node_count: u64,
}

impl FermentTranscriptBraid {
    /// Convert to JSON wire format.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({"error": "serialize_failed"}))
    }

    /// Construct from a completed provenance session result.
    ///
    /// Extracts `braid_id`, `dag_session_id`, `merkle_root`, and `spine_id`
    /// from the `complete_session()` result JSON, combining with the provided
    /// dataset and computation metadata.
    #[must_use]
    pub fn from_session_result(
        dataset_id: &str,
        session_result: &Value,
        computation: ComputationMetadata,
        summary_blake3: &str,
    ) -> Self {
        let session_id = session_result["session_id"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let braid_id = session_result["braid_id"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let merkle_root = session_result["merkle_root"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let commit_id = session_result["commit_id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Self {
            dataset_id: dataset_id.to_string(),
            spring: crate::primal_names::SELF_DISPLAY.to_string(),
            spring_version: env!("CARGO_PKG_VERSION").to_string(),
            braid_id,
            dag_session_id: session_id,
            dag_merkle_root: merkle_root,
            spine_id: commit_id,
            computation,
            summary_blake3: summary_blake3.to_string(),
            timestamp: now_iso8601(),
        }
    }
}

/// Handle `provenance.export_braid` — export a ferment transcript braid
/// from a completed provenance session.
///
/// Params:
/// - `session_id` (required): ID of a completed provenance session.
/// - `dataset_id` (required): Dataset identifier for the braid.
/// - `computation` (optional): Computation metadata object.
/// - `summary_blake3` (optional): BLAKE3 of the summary stats.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `session_id` or `dataset_id` is missing.
pub fn handle_export_braid(params: &Value) -> Result<Value, RpcError> {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: session_id"))?;

    let dataset_id = params
        .get("dataset_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: dataset_id"))?;

    let session_result = super::complete_session(session_id);

    let computation = params
        .get("computation")
        .and_then(|v| serde_json::from_value::<ComputationMetadata>(v.clone()).ok())
        .unwrap_or_else(|| ComputationMetadata {
            tool: String::new(),
            tool_version: String::new(),
            input_accession: String::new(),
            input_blake3: String::new(),
            output_blake3: String::new(),
            wall_time_seconds: 0,
            node_count: 0,
        });

    let summary_blake3 = params
        .get("summary_blake3")
        .and_then(Value::as_str)
        .unwrap_or("");

    let braid =
        FermentTranscriptBraid::from_session_result(dataset_id, &session_result, computation, summary_blake3);

    Ok(json!({
        "braid": braid.to_json(),
        "session_result": session_result,
    }))
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{secs}")
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn ferment_braid_serialization_roundtrip() {
        let braid = FermentTranscriptBraid {
            dataset_id: "tenaillon_2016_genomes".to_string(),
            spring: "wetSpring".to_string(),
            spring_version: "0.1.0".to_string(),
            braid_id: "braid-abc123".to_string(),
            dag_session_id: "dag-def456".to_string(),
            dag_merkle_root: "789abc".to_string(),
            spine_id: "spine-xyz".to_string(),
            computation: ComputationMetadata {
                tool: "breseq".to_string(),
                tool_version: "0.38.1".to_string(),
                input_accession: "PRJNA294072".to_string(),
                input_blake3: "aabbcc".to_string(),
                output_blake3: "ddeeff".to_string(),
                wall_time_seconds: 86400,
                node_count: 264,
            },
            summary_blake3: "112233".to_string(),
            timestamp: "2026-05-17T12:00:00Z".to_string(),
        };

        let json = braid.to_json();
        assert_eq!(json["dataset_id"], "tenaillon_2016_genomes");
        assert_eq!(json["spring"], "wetSpring");
        assert_eq!(json["braid_id"], "braid-abc123");
        assert_eq!(json["computation"]["tool"], "breseq");
        assert_eq!(json["computation"]["node_count"], 264);

        let roundtrip: FermentTranscriptBraid = serde_json::from_value(json).unwrap();
        assert_eq!(roundtrip.dataset_id, "tenaillon_2016_genomes");
        assert_eq!(roundtrip.computation.wall_time_seconds, 86400);
    }

    #[test]
    fn from_session_result_extracts_fields() {
        let session_result = json!({
            "session_id": "dag-session-001",
            "braid_id": "braid-created-001",
            "merkle_root": "merkle-root-abc",
            "commit_id": "spine-entry-xyz",
            "provenance": "complete",
        });

        let computation = ComputationMetadata {
            tool: "breseq".to_string(),
            tool_version: "0.38.1".to_string(),
            input_accession: "PRJNA294072".to_string(),
            input_blake3: "input-hash".to_string(),
            output_blake3: "output-hash".to_string(),
            wall_time_seconds: 3600,
            node_count: 19,
        };

        let braid = FermentTranscriptBraid::from_session_result(
            "barrick_2009_mutations",
            &session_result,
            computation,
            "summary-hash-abc",
        );

        assert_eq!(braid.dataset_id, "barrick_2009_mutations");
        assert_eq!(braid.spring, "wetSpring");
        assert_eq!(braid.braid_id, "braid-created-001");
        assert_eq!(braid.dag_session_id, "dag-session-001");
        assert_eq!(braid.dag_merkle_root, "merkle-root-abc");
        assert_eq!(braid.spine_id, "spine-entry-xyz");
        assert_eq!(braid.summary_blake3, "summary-hash-abc");
    }

    #[test]
    fn handle_export_braid_requires_session_id() {
        let err = handle_export_braid(&json!({"dataset_id": "test"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn handle_export_braid_requires_dataset_id() {
        let err = handle_export_braid(&json!({"session_id": "test"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn handle_export_braid_degrades_gracefully() {
        let result = handle_export_braid(&json!({
            "session_id": "local-test-123",
            "dataset_id": "test_dataset",
        }))
        .unwrap();

        assert!(result.get("braid").is_some());
        assert!(result.get("session_result").is_some());
        assert_eq!(result["braid"]["dataset_id"], "test_dataset");
        assert_eq!(result["braid"]["spring"], "wetSpring");
    }
}
