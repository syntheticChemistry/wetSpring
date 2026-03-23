// SPDX-License-Identifier: AGPL-3.0-or-later
//! toadStool `compute.performance_surface.*` IPC client.
//!
//! Reports measured throughput samples and queries routing hints so wetSpring can
//! align with toadStool’s capability-aware dispatch. Uses the same socket
//! discovery as other compute clients; missing toadStool degrades gracefully.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::primal_names::TOADSTOOL;

use super::transport;

/// One measured throughput tuple for performance-surface reporting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Routing hint returned from toadStool (`compute.performance_surface.query`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceSurfaceHint {
    /// Operation or kernel this hint applies to.
    pub op: String,
    /// Recommended numeric precision label (e.g. `"f64"`, `"df64"`).
    pub recommended_precision: String,
    /// Estimated throughput (ops/s) for that choice.
    pub estimated_throughput: f64,
}

#[derive(Deserialize)]
struct HintsEnvelope {
    hints: Vec<PerformanceSurfaceHint>,
}

/// Discover the toadStool Unix socket for compute RPCs.
///
/// Returns `None` in standalone mode when no socket is present.
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    super::discover::discover_socket(&super::discover::socket_env_var(TOADSTOOL), TOADSTOOL)
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
        tracing::debug!(
            primal = TOADSTOOL,
            "toadStool socket not found; skipping performance_surface.report"
        );
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

/// Query routing hints from toadStool (`compute.performance_surface.query`).
///
/// Returns `None` if toadStool is not reachable, the RPC fails, or the response
/// cannot be parsed. Callers should fall back to local defaults.
#[must_use]
pub fn query_performance_surface() -> Option<Vec<PerformanceSurfaceHint>> {
    let Some(socket) = discover_socket() else {
        tracing::debug!(
            primal = TOADSTOOL,
            "toadStool socket not found; skipping performance_surface.query"
        );
        return None;
    };

    match query_performance_surface_to(&socket) {
        Ok(hints) => Some(hints),
        Err(e) => {
            tracing::warn!(
                error = %e,
                primal = TOADSTOOL,
                "toadStool performance_surface.query failed; continuing without remote routing hints"
            );
            None
        }
    }
}

fn jsonrpc_request_line(
    method: &str,
    params: &serde_json::Value,
    id: i64,
) -> Result<String, String> {
    let line = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": id,
    });
    serde_json::to_string(&line).map_err(|e| format!("encode request: {e}"))
}

fn report_performance_surface_to(
    socket: &Path,
    samples: &[PerformanceSurfaceSample],
) -> Result<(), String> {
    let params = serde_json::json!({ "samples": samples });
    let request = jsonrpc_request_line("compute.performance_surface.report", &params, 1)?;

    let response = transport::unix_jsonrpc_line(socket, &request)?;
    let val: serde_json::Value =
        serde_json::from_str(response.trim()).map_err(|e| format!("parse response: {e}"))?;
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

fn query_performance_surface_to(socket: &Path) -> Result<Vec<PerformanceSurfaceHint>, String> {
    let params = serde_json::json!({});
    let request = jsonrpc_request_line("compute.performance_surface.query", &params, 1)?;

    let response = transport::unix_jsonrpc_line(socket, &request)?;
    let val: serde_json::Value =
        serde_json::from_str(response.trim()).map_err(|e| format!("parse response: {e}"))?;
    if let Some(err) = val.get("error") {
        let msg = err
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("compute.performance_surface.query RPC error");
        return Err(msg.to_string());
    }

    let Some(result) = val.get("result") else {
        return Err("missing result in performance_surface.query response".to_string());
    };

    parse_hints_from_result(result)
}

fn parse_hints_from_result(
    result: &serde_json::Value,
) -> Result<Vec<PerformanceSurfaceHint>, String> {
    if let Ok(hints) = serde_json::from_value::<Vec<PerformanceSurfaceHint>>(result.clone()) {
        return Ok(hints);
    }

    if let Ok(env) = serde_json::from_value::<HintsEnvelope>(result.clone()) {
        return Ok(env.hints);
    }

    Err("could not parse performance_surface.query result (expected hints array or { \"hints\": [...] })".to_string())
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

    #[test]
    fn query_performance_surface_returns_none_when_toadstool_unavailable() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
                ("FAMILY_ID", Some("perf_surface_query_none_8c1f")),
            ],
            || {
                assert!(query_performance_surface().is_none());
            },
        );
    }

    #[test]
    fn performance_surface_sample_serializes() {
        let s = PerformanceSurfaceSample {
            op: "science.diversity".to_string(),
            unit: "samples".to_string(),
            precision: "f64".to_string(),
            throughput_ops_per_sec: 1.5e6,
            tolerance: 0.01,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["op"], "science.diversity");
        assert_eq!(v["unit"], "samples");
        assert_eq!(v["precision"], "f64");
        assert_eq!(v["throughput_ops_per_sec"], 1.5e6);
        assert_eq!(v["tolerance"], 0.01);
    }

    #[test]
    fn performance_surface_hint_deserializes_from_json() {
        let json = r#"{"op":"science.diversity","recommended_precision":"df64","estimated_throughput":12000000.0}"#;
        let h: PerformanceSurfaceHint = serde_json::from_str(json).unwrap();
        assert_eq!(h.op, "science.diversity");
        assert_eq!(h.recommended_precision, "df64");
        assert_eq!(h.estimated_throughput, 12_000_000.0);
    }

    #[test]
    fn parse_hints_accepts_bare_array() {
        let v = serde_json::json!([
            {"op":"a","recommended_precision":"f64","estimated_throughput":1.0}
        ]);
        let hints = parse_hints_from_result(&v).unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].op, "a");
    }

    #[test]
    fn parse_hints_accepts_hints_envelope() {
        let v = serde_json::json!({
            "hints": [
                {"op":"b","recommended_precision":"df64","estimated_throughput":2.0}
            ]
        });
        let hints = parse_hints_from_result(&v).unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].op, "b");
    }
}
