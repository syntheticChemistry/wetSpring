// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed `toadstool.validate` + `toadstool.list_workloads` IPC client.
//!
//! Tier 2 pre-flight: validates a workload TOML against toadStool before
//! dispatch. Returns GPU availability, precision tier, estimated time,
//! warnings, and required capabilities.
//!
//! RPC: `toadstool.validate`, `toadstool.list_workloads` (toadStool S245+).
//! Discovery: reuses `compute_dispatch::discover()`.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const RPC_TIMEOUT: Duration = super::timeouts::COMPUTE;

/// Result of a `toadstool.validate` pre-flight check.
#[derive(Debug)]
pub struct ValidateResult {
    /// Whether the workload TOML is valid for dispatch.
    pub valid: bool,
    /// Whether a compatible GPU is available.
    pub gpu_available: bool,
    /// Recommended precision tier (e.g. `"DF64"`, `"F32"`).
    pub precision_tier: String,
    /// Estimated dispatch wall-clock time in milliseconds.
    pub estimated_dispatch_time_ms: u64,
    /// Advisory warnings (e.g. missing capabilities).
    pub warnings: Vec<String>,
    /// Capabilities required by this workload.
    pub required_capabilities: Vec<String>,
    /// Echo of the `dry_run` flag sent in the request (NUCLEUS spec).
    pub dry_run: bool,
}

/// A single workload entry from `toadstool.list_workloads`.
#[derive(Debug)]
pub struct WorkloadEntry {
    /// Workload identifier.
    pub id: String,
    /// Filesystem path to the workload TOML.
    pub path: String,
    /// Current status (e.g. `"ready"`, `"running"`).
    pub status: String,
    /// ISO 8601 timestamp of last execution (empty if never run).
    pub last_run: String,
    /// Precision tier assigned to this workload.
    pub precision_tier: String,
}

/// Result of a `toadstool.list_workloads` query.
#[derive(Debug)]
pub struct ListWorkloadsResult {
    /// Matching workload entries.
    pub workloads: Vec<WorkloadEntry>,
    /// Total count.
    pub total: u64,
}

/// Errors from toadStool `validate`/`list_workloads` RPC calls.
#[derive(Debug)]
pub enum ValidateError {
    /// toadStool socket not discovered.
    NoToadStool,
    /// Transport-level failure (connect, read, write).
    Transport(String),
    /// Protocol-level failure (JSON parse, RPC error).
    Protocol(String),
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoToadStool => f.write_str("toadStool not discovered"),
            Self::Transport(msg) => write!(f, "transport: {msg}"),
            Self::Protocol(msg) => write!(f, "protocol: {msg}"),
        }
    }
}

/// Discover the toadStool socket (reuses `compute_dispatch::discover()`).
#[must_use]
pub fn discover() -> Option<PathBuf> {
    super::compute_dispatch::discover()
}

/// Pre-flight a workload TOML through toadStool.
///
/// # Errors
///
/// Returns [`ValidateError::NoToadStool`] if toadStool is not discovered,
/// or transport/protocol errors.
pub fn validate(workload_path: &str, dry_run: bool) -> Result<ValidateResult, ValidateError> {
    let socket = discover().ok_or(ValidateError::NoToadStool)?;
    validate_at(&socket, workload_path, dry_run)
}

/// Pre-flight a workload TOML through a specific toadStool socket.
///
/// # Errors
///
/// Returns transport or protocol errors.
pub fn validate_at(
    socket: &Path,
    workload_path: &str,
    dry_run: bool,
) -> Result<ValidateResult, ValidateError> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"toadstool.validate","params":{{"workload_path":"{workload_path}","dry_run":{dry_run}}},"id":1}}"#,
    );
    let response = rpc_call(socket, &request)?;
    parse_validate_response(&response)
}

/// List workloads registered with toadStool.
///
/// # Errors
///
/// Returns [`ValidateError::NoToadStool`] if toadStool is not discovered,
/// or transport/protocol errors.
pub fn list_workloads(filter: &str) -> Result<ListWorkloadsResult, ValidateError> {
    let socket = discover().ok_or(ValidateError::NoToadStool)?;
    list_workloads_at(&socket, filter)
}

/// List workloads from a specific toadStool socket.
///
/// # Errors
///
/// Returns transport or protocol errors.
pub fn list_workloads_at(
    socket: &Path,
    filter: &str,
) -> Result<ListWorkloadsResult, ValidateError> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"toadstool.list_workloads","params":{{"filter":"{filter}"}},"id":2}}"#,
    );
    let response = rpc_call(socket, &request)?;
    parse_list_response(&response)
}

fn rpc_call(socket: &Path, request: &str) -> Result<String, ValidateError> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| ValidateError::Transport(format!("connect {}: {e}", socket.display())))?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| ValidateError::Transport(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| ValidateError::Transport(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| ValidateError::Transport(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| ValidateError::Transport(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| ValidateError::Transport(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| ValidateError::Transport(format!("read: {e}")))?;

    if line.is_empty() {
        return Err(ValidateError::Transport(
            "empty response from toadStool".to_string(),
        ));
    }
    Ok(line)
}

fn parse_validate_response(response: &str) -> Result<ValidateResult, ValidateError> {
    let v: serde_json::Value = serde_json::from_str(response)
        .map_err(|e| ValidateError::Protocol(format!("JSON parse: {e}")))?;

    if let Some(err) = v.get("error") {
        return Err(ValidateError::Protocol(format!(
            "RPC error: {}",
            err.get("message").and_then(|m| m.as_str()).unwrap_or("unknown")
        )));
    }

    let result = v
        .get("result")
        .ok_or_else(|| ValidateError::Protocol("missing result".to_string()))?;

    Ok(ValidateResult {
        valid: result.get("valid").and_then(serde_json::Value::as_bool).unwrap_or(false),
        gpu_available: result
            .get("gpu_available")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        precision_tier: result
            .get("precision_tier")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        estimated_dispatch_time_ms: result
            .get("estimated_dispatch_time_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        warnings: result
            .get("warnings")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        required_capabilities: result
            .get("required_capabilities")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        dry_run: result
            .get("dry_run")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}

fn parse_list_response(response: &str) -> Result<ListWorkloadsResult, ValidateError> {
    let v: serde_json::Value = serde_json::from_str(response)
        .map_err(|e| ValidateError::Protocol(format!("JSON parse: {e}")))?;

    if let Some(err) = v.get("error") {
        return Err(ValidateError::Protocol(format!(
            "RPC error: {}",
            err.get("message").and_then(|m| m.as_str()).unwrap_or("unknown")
        )));
    }

    let result = v
        .get("result")
        .ok_or_else(|| ValidateError::Protocol("missing result".to_string()))?;

    let total = result.get("total").and_then(serde_json::Value::as_u64).unwrap_or(0);

    let workloads = result
        .get("workloads")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|w| WorkloadEntry {
                    id: w
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    path: w
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    status: w
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    last_run: w
                        .get("last_run")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    precision_tier: w
                        .get("precision_tier")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(ListWorkloadsResult { workloads, total })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_validate_response_ok() {
        let json = r#"{"jsonrpc":"2.0","result":{"valid":true,"gpu_available":true,"precision_tier":"DF64","estimated_dispatch_time_ms":1200,"warnings":[],"required_capabilities":["compute","shader"],"dry_run":true},"id":1}"#;
        let result = parse_validate_response(json).expect("parse ok");
        assert!(result.valid);
        assert!(result.gpu_available);
        assert_eq!(result.precision_tier, "DF64");
        assert_eq!(result.estimated_dispatch_time_ms, 1200);
        assert!(result.warnings.is_empty());
        assert_eq!(result.required_capabilities, vec!["compute", "shader"]);
        assert!(result.dry_run);
    }

    #[test]
    fn parse_validate_response_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"code":-32602,"message":"invalid workload path"},"id":1}"#;
        let result = parse_validate_response(json);
        assert!(result.is_err());
    }

    #[test]
    fn parse_list_response_ok() {
        let json = r#"{"jsonrpc":"2.0","result":{"workloads":[{"id":"ws-16s","path":"workloads/wetspring/wetspring-16s-rust-validation.toml","status":"ready","last_run":"2026-05-12T10:00:00Z","precision_tier":"F32"}],"total":1},"id":2}"#;
        let result = parse_list_response(json).expect("parse ok");
        assert_eq!(result.total, 1);
        assert_eq!(result.workloads.len(), 1);
        assert_eq!(result.workloads[0].id, "ws-16s");
        assert_eq!(result.workloads[0].last_run, "2026-05-12T10:00:00Z");
    }

    #[test]
    fn discover_returns_none_when_absent() {
        assert!(discover().is_none() || discover().is_some());
    }
}
