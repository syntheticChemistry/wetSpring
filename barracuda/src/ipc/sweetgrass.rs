// SPDX-License-Identifier: AGPL-3.0-or-later
//! sweetGrass braid IPC client — experiment provenance hand-off.
//!
//! Emits `braid.create` followed by `braid.commit` to the sweetGrass primal when
//! its socket is discovered at runtime. Standalone wetSpring runs skip this path
//! without failing.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::primal_names::SWEETGRASS;

const RPC_TIMEOUT: Duration = Duration::from_secs(10);

/// Parameters for creating a new provenance braid on sweetGrass.
#[derive(Debug, Clone, Serialize)]
pub struct BraidRequest {
    /// Session identifier (often matches wetSpring provenance session id).
    pub session_id: String,
    /// Human-readable experiment or run description.
    pub description: String,
    /// Role of the caller in the ecosystem graph (e.g. `"wetspring"`).
    pub agent_role: String,
}

/// Parameters for committing content to an existing braid.
///
/// When using [`record_experiment_provenance`], `braid_id` is overwritten with the
/// identifier returned from `braid.create` before `braid.commit` is sent.
#[derive(Debug, Clone, Serialize)]
pub struct BraidCommitRequest {
    /// Braid identifier returned from `braid.create` (may be ignored by [`record_experiment_provenance`]).
    pub braid_id: String,
    /// Content-address hash of the committed artifact (hex or multihash string).
    pub content_hash: String,
    /// Free-form JSON metadata (tool versions, inputs, etc.).
    pub metadata: Value,
}

/// Discover the sweetGrass Unix socket using the standard biomeOS cascade.
///
/// Returns `None` when sweetGrass is not present (standalone mode).
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    super::discover::discover_socket(&super::discover::socket_env_var(SWEETGRASS), SWEETGRASS)
}

/// Record experiment provenance on sweetGrass via `braid.create` then `braid.commit`.
///
/// If sweetGrass is unreachable or either RPC fails, logs at `warn` and returns —
/// callers should treat this as best-effort telemetry, not a hard error.
pub fn record_experiment_provenance(create: &BraidRequest, commit: &BraidCommitRequest) {
    let Some(socket) = discover_socket() else {
        tracing::debug!(
            primal = SWEETGRASS,
            "sweetGrass socket not found; skipping braid IPC"
        );
        return;
    };

    if let Err(e) = record_experiment_provenance_to(&socket, create, commit) {
        tracing::warn!(
            error = %e,
            primal = SWEETGRASS,
            "sweetGrass braid provenance IPC failed; continuing without remote braid"
        );
    }
}

fn record_experiment_provenance_to(
    socket: &Path,
    create: &BraidRequest,
    commit: &BraidCommitRequest,
) -> Result<(), String> {
    let create_params =
        serde_json::to_value(create).map_err(|e| format!("serialize braid.create params: {e}"))?;
    let create_line = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "braid.create",
        "params": create_params,
        "id": 1,
    });
    let create_req =
        serde_json::to_string(&create_line).map_err(|e| format!("encode braid.create: {e}"))?;

    let create_resp = rpc_call(socket, &create_req)?;
    let create_val: Value = serde_json::from_str(create_resp.trim())
        .map_err(|e| format!("parse braid.create response: {e}"))?;

    if let Some(err) = create_val.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("braid.create RPC error");
        return Err(msg.to_string());
    }

    let braid_id = create_val
        .get("result")
        .and_then(|r| r.get("braid_id"))
        .and_then(Value::as_str)
        .ok_or_else(|| "braid.create result missing braid_id".to_string())?;

    let mut commit_body = commit.clone();
    commit_body.braid_id = braid_id.to_string();

    let commit_params = serde_json::to_value(&commit_body)
        .map_err(|e| format!("serialize braid.commit params: {e}"))?;
    let commit_line = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "braid.commit",
        "params": commit_params,
        "id": 2,
    });
    let commit_req =
        serde_json::to_string(&commit_line).map_err(|e| format!("encode braid.commit: {e}"))?;

    let commit_resp = rpc_call(socket, &commit_req)?;
    let commit_val: Value = serde_json::from_str(commit_resp.trim())
        .map_err(|e| format!("parse braid.commit response: {e}"))?;

    if let Some(err) = commit_val.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("braid.commit RPC error");
        return Err(msg.to_string());
    }

    tracing::debug!(braid_id = %braid_id, "sweetGrass braid.create + braid.commit completed");
    Ok(())
}

fn rpc_call(socket: &Path, request: &str) -> Result<String, String> {
    let stream =
        UnixStream::connect(socket).map_err(|e| format!("connect {}: {e}", socket.display()))?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| format!("set read timeout: {e}"))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| format!("set write timeout: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response from sweetGrass".to_string());
    }

    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }
}
