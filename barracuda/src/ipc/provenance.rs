// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance trio integration for wetSpring experiment sessions.
//!
//! Follows the `SPRING_PROVENANCE_TRIO_INTEGRATION_PATTERN` from wateringHole:
//! rhizoCrypt (ephemeral DAG) → loamSpine (permanent commit) → sweetGrass
//! (W3C PROV-O attribution).
//!
//! All trio interaction goes through biomeOS `capability.call` over a Unix
//! socket — zero compile-time coupling to trio crates. Domain logic never
//! fails when provenance is unavailable (graceful degradation).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

const RPC_TIMEOUT: Duration = Duration::from_secs(10);

/// Shorthand for extracting a string from a JSON value, falling back to a default.
fn json_str_or<'a>(val: &'a Value, key: &str, default: &'a str) -> &'a str {
    val.get(key).and_then(Value::as_str).unwrap_or(default)
}

/// Result of a provenance operation, including availability status.
#[derive(Debug)]
pub struct ProvenanceResult {
    /// Session or vertex ID (local fallback if trio unavailable).
    pub id: String,
    /// Whether the trio was actually reachable.
    pub available: bool,
    /// Raw response data.
    pub data: Value,
}

/// Discover the biomeOS Neural API socket.
///
/// Priority: `NEURAL_API_SOCKET` → `BIOMEOS_SOCKET_DIR` →
/// `$XDG_RUNTIME_DIR/biomeos/` → `std::env::temp_dir()` (platform-specific temp).
#[must_use]
pub fn neural_api_socket() -> Option<PathBuf> {
    let family_id = super::discover::family_id();

    let sock_name = format!("neural-api-{family_id}.sock");

    let candidates: [Option<PathBuf>; 4] = [
        std::env::var("NEURAL_API_SOCKET").ok().map(PathBuf::from),
        std::env::var("BIOMEOS_SOCKET_DIR")
            .ok()
            .map(|d| PathBuf::from(d).join(&sock_name)),
        std::env::var("XDG_RUNTIME_DIR").ok().map(|d| {
            PathBuf::from(d)
                .join(super::primal_names::BIOMEOS)
                .join(&sock_name)
        }),
        Some(std::env::temp_dir().join(&sock_name)),
    ];

    candidates.into_iter().flatten().find(|p| p.exists())
}

/// Send a `capability.call` JSON-RPC request to the Neural API.
fn capability_call(
    socket_path: &Path,
    capability: &str,
    operation: &str,
    args: &Value,
) -> Result<Value, crate::error::Error> {
    use crate::error::IpcError;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": capability,
            "operation": operation,
            "args": args,
        },
        "id": 1,
    });

    let payload =
        serde_json::to_string(&request).map_err(|e| IpcError::Codec(format!("serialize: {e}")))?;

    let stream = std::os::unix::net::UnixStream::connect(socket_path)
        .map_err(|e| IpcError::Connect(format!("{}: {e}", socket_path.display())))?;
    stream.set_read_timeout(Some(RPC_TIMEOUT)).ok();
    stream.set_write_timeout(Some(RPC_TIMEOUT)).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(payload.as_bytes())
        .map_err(|e| IpcError::Transport(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| IpcError::Transport(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| IpcError::Transport(format!("flush: {e}")))?;
    drop(writer);

    stream
        .shutdown(std::net::Shutdown::Write)
        .map_err(|e| IpcError::Transport(format!("shutdown: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| IpcError::Transport(format!("read: {e}")))?;

    let parsed: Value =
        serde_json::from_str(line.trim()).map_err(|e| IpcError::Codec(format!("parse: {e}")))?;

    if let Some(err) = parsed.get("error") {
        let code = err.get("code").and_then(Value::as_i64).unwrap_or(-32000);
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        return Err(IpcError::RpcReject {
            code,
            message: msg.to_string(),
        }
        .into());
    }

    parsed
        .get("result")
        .cloned()
        .ok_or_else(|| IpcError::EmptyResponse.into())
}

fn local_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("local-{}-{ts}", crate::PRIMAL_NAME)
}

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
            let session_id = json_str_or(&result, "session_id", "unknown").to_string();
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

/// Complete an experiment: dehydrate → commit → attribute.
///
/// Three-phase completion following the wateringHole provenance trio pattern.
/// Each phase degrades gracefully if the downstream primal is unavailable.
pub fn complete_session(session_id: &str) -> Value {
    let Some(socket) = neural_api_socket() else {
        return json!({
            "provenance": "unavailable",
            "session_id": session_id,
        });
    };

    // Phase 1: Dehydrate (rhizoCrypt)
    let Ok(dehydration) = capability_call(
        &socket,
        "dag",
        "dehydrate",
        &json!({"session_id": session_id}),
    ) else {
        return json!({
            "provenance": "unavailable",
            "session_id": session_id,
        });
    };

    let merkle_root = json_str_or(&dehydration, "merkle_root", "").to_string();

    // Phase 2: Commit (loamSpine — session.commit)
    let Ok(commit_result) = capability_call(
        &socket,
        "session",
        "commit",
        &json!({"summary": dehydration, "content_hash": merkle_root}),
    ) else {
        return json!({
            "provenance": "partial",
            "session_id": session_id,
            "dehydrated": true,
            "committed": false,
            "merkle_root": merkle_root,
        });
    };

    let commit_id = commit_result
        .get("commit_id")
        .or_else(|| commit_result.get("entry_id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // Phase 3: Attribute (sweetGrass) — best-effort
    let braid_args = json!({
        "commit_ref": commit_id,
        "agents": [{
            "did": format!("did:key:{}", crate::PRIMAL_NAME),
            "role": "author",
            "contribution": 1.0,
        }],
    });
    let braid_result = capability_call(&socket, "braid", "create", &braid_args);

    let braid_id = braid_result
        .ok()
        .and_then(|r| {
            let id = r.get("braid_id").or_else(|| r.get("id"));
            id.and_then(Value::as_str).map(str::to_string)
        })
        .unwrap_or_default();

    json!({
        "provenance": "complete",
        "session_id": session_id,
        "merkle_root": merkle_root,
        "commit_id": commit_id,
        "braid_id": braid_id,
    })
}

// --- IPC Handlers ---

/// Handle `provenance.begin` — start a provenance-tracked session.
///
/// # Errors
///
/// Always succeeds (degrades gracefully when trio is unavailable).
pub fn handle_provenance_begin(params: &Value) -> Result<Value, RpcError> {
    let name = params
        .get("experiment")
        .or_else(|| params.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("unnamed_experiment");

    let result = begin_session(name);
    let status = if result.available {
        "available"
    } else {
        "unavailable"
    };
    Ok(json!({
        "session_id": result.id,
        "provenance": status,
        "data": result.data,
    }))
}

/// Handle `provenance.record` — record a step in the DAG.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `session_id` is missing.
pub fn handle_provenance_record(params: &Value) -> Result<Value, RpcError> {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: session_id"))?;

    let step = params
        .get("step")
        .or_else(|| params.get("event"))
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = record_step(session_id, &step);
    let status = if result.available {
        "available"
    } else {
        "unavailable"
    };
    Ok(json!({
        "vertex_id": result.id,
        "provenance": status,
        "data": result.data,
    }))
}

/// Handle `provenance.complete` — dehydrate → commit → attribute.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `session_id` is missing.
pub fn handle_provenance_complete(params: &Value) -> Result<Value, RpcError> {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: session_id"))?;

    Ok(complete_session(session_id))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn neural_api_socket_does_not_panic() {
        let _ = neural_api_socket();
    }

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

    #[test]
    fn complete_session_degrades_gracefully() {
        let result = complete_session("fake-session");
        assert_eq!(result["provenance"], "unavailable");
    }

    #[test]
    fn handle_begin_without_trio() {
        let result = handle_provenance_begin(&json!({"experiment": "test"})).unwrap();
        assert_eq!(result["provenance"], "unavailable");
        assert!(result["session_id"].as_str().unwrap().starts_with("local-"));
    }

    #[test]
    fn handle_record_requires_session_id() {
        let err = handle_provenance_record(&json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn handle_complete_requires_session_id() {
        let err = handle_provenance_complete(&json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn handle_record_without_trio() {
        let result =
            handle_provenance_record(&json!({"session_id": "local-1", "step": {"a": 1}})).unwrap();
        assert_eq!(result["provenance"], "unavailable");
    }

    #[test]
    fn handle_complete_without_trio() {
        let result = handle_provenance_complete(&json!({"session_id": "local-1"})).unwrap();
        assert_eq!(result["provenance"], "unavailable");
    }
}
