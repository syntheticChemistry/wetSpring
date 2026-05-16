// SPDX-License-Identifier: AGPL-3.0-or-later
//! skunkBat audit event emitter for wetSpring.
//!
//! Sends structured audit events to the skunkBat primal via JSON-RPC
//! over Unix domain socket. When Phase 3 (JH-5 forwarding) ships,
//! these events flow into rhizoCrypt DAG + sweetGrass braid automatically.
//!
//! All calls degrade gracefully — audit failures never block science.
//!
//! # Protocol
//!
//! JSON-RPC 2.0, newline-delimited, over Unix domain socket.
//! Methods: `audit.event`, `audit.forward`.
//!
//! # Socket discovery
//!
//! 1. `SKUNKBAT_SOCKET` env var
//! 2. `$XDG_RUNTIME_DIR/biomeos/skunkbat-{family_id}.sock`
//! 3. `<temp_dir>/skunkbat-{family_id}.sock`

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = super::timeouts::DISCOVERY;

/// Discover the skunkBat Unix socket path.
///
/// Returns `None` if no skunkBat socket is found (standalone/no-audit mode).
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    super::discover::discover_socket(
        &super::discover::socket_env_var(super::primal_names::SKUNKBAT),
        super::primal_names::SKUNKBAT,
    )
}

/// Audit event severity levels per skunkBat wire protocol.
#[derive(Debug, Clone, Copy)]
pub enum Severity {
    /// Informational trace (pipeline start/stop, config load).
    Info,
    /// Non-critical anomaly (degraded primal, fallback path taken).
    Warning,
    /// Actionable failure (IPC rejection, missing capability).
    Error,
    /// Security-relevant event (auth failure, unauthorized access attempt).
    Security,
}

impl Severity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Security => "security",
        }
    }
}

/// Structured audit event payload.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Event category (e.g. `"ipc"`, `"science"`, `"provenance"`, `"composition"`).
    pub domain: &'static str,
    /// Short action name (e.g. `"pipeline_start"`, `"capability_missing"`).
    pub action: &'static str,
    /// Event severity.
    pub severity: Severity,
    /// Freeform detail string.
    pub detail: String,
    /// Optional structured context (merged into the event payload).
    pub context: Option<Value>,
}

/// Emit an audit event to skunkBat via `audit.event`.
///
/// Returns `Ok(response)` on success. Callers should log and continue —
/// audit failures must not block science operations.
///
/// # Errors
///
/// Returns `Err` if skunkBat is unreachable or rejects the event.
pub fn emit_event(skunkbat_socket: &Path, event: &AuditEvent) -> crate::error::Result<Value> {
    let mut params = json!({
        "primal": crate::PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "domain": event.domain,
        "action": event.action,
        "severity": event.severity.as_str(),
        "detail": event.detail,
        "timestamp_ns": now_nanos(),
    });

    if let Some(ctx) = &event.context {
        if let (Some(base), Some(extra)) = (params.as_object_mut(), ctx.as_object()) {
            for (k, v) in extra {
                base.insert(k.clone(), v.clone());
            }
        }
    }

    let request = json!({
        "jsonrpc": "2.0",
        "method": "audit.event",
        "params": params,
        "id": 1,
    });

    rpc_call(skunkbat_socket, &request)
}

/// Forward an audit event to another primal's audit trail via `audit.forward`.
///
/// Used for cross-primal audit chains (JH-5 forwarding). The `target` is the
/// destination primal name (e.g. `"rhizocrypt"` for DAG anchoring).
///
/// # Errors
///
/// Returns `Err` if skunkBat is unreachable or rejects the forward request.
pub fn forward_event(
    skunkbat_socket: &Path,
    target: &str,
    event: &AuditEvent,
) -> crate::error::Result<Value> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": "audit.forward",
        "params": {
            "source": crate::PRIMAL_NAME,
            "target": target,
            "domain": event.domain,
            "action": event.action,
            "severity": event.severity.as_str(),
            "detail": event.detail,
            "timestamp_ns": now_nanos(),
        },
        "id": 2,
    });

    rpc_call(skunkbat_socket, &request)
}

/// Best-effort audit emit — discovers skunkBat and sends the event.
///
/// Logs a trace on failure; never returns an error to the caller.
/// This is the primary entry point for science code that wants to
/// record audit events without caring about skunkBat availability.
pub fn try_emit(event: &AuditEvent) {
    let Some(socket) = discover_socket() else {
        tracing::trace!(
            domain = event.domain,
            action = event.action,
            "skunkBat not available — audit event suppressed"
        );
        return;
    };
    if let Err(e) = emit_event(&socket, event) {
        tracing::trace!(
            error = %e,
            domain = event.domain,
            action = event.action,
            "skunkBat audit emit failed — continuing"
        );
    }
}

/// Convenience: emit an informational audit event (best-effort).
pub fn audit_info(domain: &'static str, action: &'static str, detail: impl Into<String>) {
    try_emit(&AuditEvent {
        domain,
        action,
        severity: Severity::Info,
        detail: detail.into(),
        context: None,
    });
}

/// Convenience: emit a warning audit event (best-effort).
pub fn audit_warn(domain: &'static str, action: &'static str, detail: impl Into<String>) {
    try_emit(&AuditEvent {
        domain,
        action,
        severity: Severity::Warning,
        detail: detail.into(),
        context: None,
    });
}

fn now_nanos() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

fn rpc_call(socket: &Path, request: &Value) -> crate::error::Result<Value> {
    use crate::error::IpcError;

    let payload =
        serde_json::to_string(request).map_err(|e| IpcError::Codec(format!("serialize: {e}")))?;

    let stream = UnixStream::connect(socket)
        .map_err(|e| IpcError::Connect(format!("skunkBat {}: {e}", socket.display())))?;
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

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| IpcError::Transport(format!("read: {e}")))?;

    if line.is_empty() {
        return Err(IpcError::EmptyResponse.into());
    }

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
            message: format!("skunkBat: {msg}"),
        }
        .into());
    }

    parsed
        .get("result")
        .cloned()
        .ok_or_else(|| IpcError::EmptyResponse.into())
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
    fn discover_socket_explicit() {
        let sock = crate::ipc::test_socket_path("skunkbat_discover_socket_explicit");
        crate::ipc::cleanup_test_socket(&sock);
        std::fs::write(&sock, "").unwrap();

        temp_env::with_var("SKUNKBAT_SOCKET", Some(sock.to_str().unwrap()), || {
            let found = discover_socket();
            assert_eq!(found, Some(sock.clone()));
        });
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn emit_event_nonexistent_socket_errors() {
        let bad_path = crate::ipc::test_socket_path("skunkbat_emit_bad");
        crate::ipc::cleanup_test_socket(&bad_path);
        let event = AuditEvent {
            domain: "test",
            action: "test_event",
            severity: Severity::Info,
            detail: "unit test".into(),
            context: None,
        };
        let err = emit_event(&bad_path, &event).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") || msg.contains("socket"),
            "unexpected error: {msg}"
        );
        crate::ipc::cleanup_test_socket(&bad_path);
    }

    #[test]
    fn forward_event_nonexistent_socket_errors() {
        let bad_path = crate::ipc::test_socket_path("skunkbat_forward_bad");
        crate::ipc::cleanup_test_socket(&bad_path);
        let event = AuditEvent {
            domain: "test",
            action: "test_forward",
            severity: Severity::Warning,
            detail: "forward test".into(),
            context: None,
        };
        let err = forward_event(&bad_path, "rhizocrypt", &event).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("connect") || msg.contains("socket"),
            "unexpected error: {msg}"
        );
        crate::ipc::cleanup_test_socket(&bad_path);
    }

    #[test]
    fn try_emit_does_not_panic_without_socket() {
        let event = AuditEvent {
            domain: "test",
            action: "try_emit_test",
            severity: Severity::Info,
            detail: "should not panic".into(),
            context: None,
        };
        try_emit(&event);
    }

    #[test]
    fn audit_info_does_not_panic() {
        audit_info("test", "info_test", "convenience test");
    }

    #[test]
    fn audit_warn_does_not_panic() {
        audit_warn("test", "warn_test", "convenience test");
    }

    #[test]
    fn severity_as_str() {
        assert_eq!(Severity::Info.as_str(), "info");
        assert_eq!(Severity::Warning.as_str(), "warning");
        assert_eq!(Severity::Error.as_str(), "error");
        assert_eq!(Severity::Security.as_str(), "security");
    }

    #[test]
    fn audit_event_with_context() {
        let bad_path = crate::ipc::test_socket_path("skunkbat_ctx_bad");
        crate::ipc::cleanup_test_socket(&bad_path);
        let event = AuditEvent {
            domain: "science",
            action: "pipeline_start",
            severity: Severity::Info,
            detail: "starting diversity pipeline".into(),
            context: Some(json!({
                "pipeline": "diversity",
                "samples": 42,
            })),
        };
        let err = emit_event(&bad_path, &event).unwrap_err();
        assert!(err.to_string().contains("connect") || err.to_string().contains("socket"));
        crate::ipc::cleanup_test_socket(&bad_path);
    }

    #[test]
    fn now_nanos_is_positive() {
        assert!(now_nanos() > 0);
    }
}
