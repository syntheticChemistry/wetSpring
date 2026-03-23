// SPDX-License-Identifier: AGPL-3.0-or-later
//! toadStool `compute.performance_surface.report` IPC stub.
//!
//! Sends measured throughput samples so toadStool can tune routing. Uses the same
//! socket discovery as other compute clients; missing toadStool is a no-op.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::primal_names::TOADSTOOL;

const RPC_TIMEOUT: Duration = Duration::from_secs(10);

/// One measured throughput tuple for performance-surface reporting.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSurfaceSample {
    /// Operation or kernel identifier (e.g. `"science.diversity"`).
    pub op: String,
    /// Unit of work counted (e.g. `"samples"`, `"bases"`).
    pub unit: String,
    /// Precision label (e.g. `"f64"`, `"df64"`).
    pub precision: String,
    /// Observed throughput in operations per second.
    pub throughput_ops_per_sec: f64,
    /// Relative tolerance or confidence bound for this measurement.
    pub tolerance: f64,
}

/// Discover the toadStool Unix socket for compute RPCs.
///
/// Returns `None` in standalone mode when no socket is present.
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    super::discover::discover_socket(
        &super::discover::socket_env_var(TOADSTOOL),
        TOADSTOOL,
    )
}

/// Report measured performance samples to toadStool (`compute.performance_surface.report`).
///
/// If toadStool is unavailable or the RPC fails, logs and returns without error.
pub fn report_performance_surface(samples: &[PerformanceSurfaceSample]) {
    if samples.is_empty() {
        tracing::debug!("performance_surface.report skipped (empty samples)");
        return;
    }

    let Some(socket) = discover_socket() else {
        tracing::debug!(primal = TOADSTOOL, "toadStool socket not found; skipping performance_surface.report");
        return;
    };

    if let Err(e) = report_performance_surface_to(&socket, samples) {
        tracing::warn!(
            error = %e,
            primal = TOADSTOOL,
            "toadStool performance_surface.report failed; continuing without remote routing hints"
        );
    }
}

fn report_performance_surface_to(
    socket: &Path,
    samples: &[PerformanceSurfaceSample],
) -> Result<(), String> {
    let params = serde_json::json!({ "samples": samples });
    let line = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "compute.performance_surface.report",
        "params": params,
        "id": 1,
    });
    let request = serde_json::to_string(&line).map_err(|e| format!("encode request: {e}"))?;

    let response = rpc_call(socket, &request)?;
    let val: serde_json::Value = serde_json::from_str(response.trim())
        .map_err(|e| format!("parse response: {e}"))?;

    if let Some(err) = val.get("error") {
        let msg = err
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("compute.performance_surface.report RPC error");
        return Err(msg.to_string());
    }

    tracing::debug!(
        count = samples.len(),
        "toadStool compute.performance_surface.report completed"
    );
    Ok(())
}

fn rpc_call(socket: &Path, request: &str) -> Result<String, String> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| format!("connect {}: {e}", socket.display()))?;

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
        return Err("empty response from toadStool".to_string());
    }

    Ok(line)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn report_empty_samples_is_noop() {
        report_performance_surface(&[]);
    }
}
