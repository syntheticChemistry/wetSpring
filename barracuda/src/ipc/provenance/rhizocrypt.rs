// SPDX-License-Identifier: AGPL-3.0-or-later
//! rhizoCrypt DAG session operations — ephemeral derivation graph tracking.
//!
//! Wraps the `dag.session.create`, `dag.event.append`, and `dag.dehydrate`
//! capability calls routed through biomeOS Neural API. Gracefully degrades
//! when rhizoCrypt is unreachable (standalone mode).

use serde_json::{Value, json};

use super::{ProvenanceResult, capability_call, local_session_id, neural_api_socket};

/// Begin a provenance-tracked experiment session via rhizoCrypt.
///
/// Returns a local fallback ID if the trio is unavailable.
#[must_use]
pub fn begin_session(experiment_name: &str) -> ProvenanceResult {
    let Some(socket) = neural_api_socket() else {
        return ProvenanceResult {
            id: local_session_id(),
            available: false,
            data: json!({"provenance": "unavailable"}),
        };
    };

    let args = json!({
        "metadata": {"type": "experiment", "name": experiment_name},
        "session_type": {"Experiment": {"spring_id": crate::PRIMAL_NAME}},
        "description": experiment_name,
    });

    capability_call(&socket, "dag", "session.create", &args).map_or_else(
        |_| ProvenanceResult {
            id: local_session_id(),
            available: false,
            data: json!({"provenance": "unavailable"}),
        },
        |result| {
            let session_id = super::json_str_or(&result, "session_id", "unknown").to_string();
            ProvenanceResult {
                id: session_id.clone(),
                available: true,
                data: json!({"session_id": session_id}),
            }
        },
    )
}

/// Record an experiment step in the rhizoCrypt DAG.
#[must_use]
pub fn record_step(session_id: &str, step: &Value) -> ProvenanceResult {
    let Some(socket) = neural_api_socket() else {
        return ProvenanceResult {
            id: "unavailable".to_string(),
            available: false,
            data: json!({"provenance": "unavailable"}),
        };
    };

    let args = json!({"session_id": session_id, "event": step});

    capability_call(&socket, "dag", "event.append", &args).map_or_else(
        |_| ProvenanceResult {
            id: "unavailable".to_string(),
            available: false,
            data: json!({"provenance": "unavailable"}),
        },
        |result| {
            let vertex_id = result
                .get("vertex_id")
                .or_else(|| result.get("id"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            ProvenanceResult {
                id: vertex_id.clone(),
                available: true,
                data: json!({"vertex_id": vertex_id}),
            }
        },
    )
}

/// Dehydrate a session DAG into a Merkle root via rhizoCrypt.
///
/// Returns the dehydration result or an error. Used by `complete_session`
/// as Phase 1 of the three-phase provenance completion.
pub(super) fn dehydrate(
    socket: &std::path::Path,
    session_id: &str,
) -> Result<Value, crate::error::Error> {
    capability_call(
        socket,
        "dag",
        "dehydrate",
        &json!({"session_id": session_id}),
    )
}

/// Compute a Merkle root over a subset of vertices without closing the session.
///
/// rhizoCrypt S69 `dag.partial_dehydrate`: produces a cryptographically valid
/// root for sealed vertices while leaving the session open for further appends.
/// Pass empty `vertex_ids` to dehydrate all current vertices.
///
/// This is the "aglet" pattern — seal the finished nodes, leave open branches.
pub fn partial_dehydrate(
    session_id: &str,
    vertex_ids: &[String],
) -> Option<Value> {
    let socket = neural_api_socket()?;
    capability_call(
        &socket,
        "dag",
        "partial_dehydrate",
        &json!({
            "session_id": session_id,
            "vertex_ids": vertex_ids,
        }),
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_session_degrades_gracefully() {
        let result = begin_session("test_experiment");
        let prefix = format!("local-{}-", crate::PRIMAL_NAME);
        assert!(result.id.starts_with(&prefix));
        assert!(!result.available);
    }

    #[test]
    fn record_step_degrades_gracefully() {
        let result = record_step("fake-session", &json!({"step": "diversity"}));
        assert_eq!(result.id, "unavailable");
        assert!(!result.available);
    }
}
