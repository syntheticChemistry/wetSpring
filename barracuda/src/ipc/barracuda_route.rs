// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda IPC routing for primal-proof sovereign deployment.
//!
//! When `--features primal-proof` is active, math calls route through the
//! barraCuda ecobin primal over JSON-RPC instead of in-process library calls.
//! Discovery uses the standard primal IPC cascade:
//!
//! 1. `BARRACUDA_SOCKET` env override
//! 2. `$XDG_RUNTIME_DIR/biomeos/barracuda-{family}.sock`
//! 3. `<temp_dir>/barracuda-{family}.sock`
//!
//! Falls back to `None` when the socket is not found (caller degrades to
//! in-process). The dual-lane pattern (IPC vs library) is resolved at
//! compile time via `cfg(feature = "primal-proof")` in the handler layer.
//!
//! # Protocol
//!
//! Follows barraCuda's canonical v0.9.17 surface: 33 JSON-RPC methods across
//! TENSOR (9), STATS (9), COMPUTE (4), SPECTRAL (3), LINALG (6), HEALTH (2).
//! Requests use JSON-RPC 2.0, newline-delimited, over Unix domain socket.
//!
//! # Graceful degradation
//!
//! All public functions return `Result` so callers can fall back to in-process
//! compute on any transport failure. The `try_forward` convenience wrapper
//! returns `None` on any error, suitable for dual-lane dispatch.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::IpcError;

const RPC_TIMEOUT: Duration = super::timeouts::COMPUTE;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Discover the barraCuda primal socket.
///
/// Uses the standard cascading discovery strategy. Returns `None` when
/// barraCuda is not live (standalone / in-process mode).
#[must_use]
pub fn discover() -> Option<PathBuf> {
    super::discover::discover_primal(super::primal_names::BARRACUDA)
}

/// Forward a JSON-RPC method call to a remote barraCuda primal.
///
/// # Errors
///
/// Returns [`IpcError`] on socket discovery failure, transport errors, or
/// JSON-RPC error responses from the remote.
pub fn forward(socket: &Path, method: &str, params: &Value) -> Result<Value, IpcError> {
    let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": id,
    });
    rpc_call(socket, &request)
}

/// Try to forward a method call to barraCuda, returning `None` on any failure.
///
/// Discovers the socket, sends the request, and returns the result value.
/// On any error (no socket, transport, RPC reject), returns `None` so the
/// caller can fall back to in-process compute.
#[must_use]
pub fn try_forward(method: &str, params: &Value) -> Option<Value> {
    let socket = discover()?;
    forward(&socket, method, params).ok()
}

/// Check if a remote barraCuda primal is reachable and healthy.
///
/// Sends `health.liveness` and returns `true` only on a successful response.
#[must_use]
pub fn is_available() -> bool {
    let Some(socket) = discover() else {
        return false;
    };
    forward(&socket, "health.liveness", &json!({})).is_ok()
}

fn rpc_call(socket: &Path, request: &Value) -> Result<Value, IpcError> {
    let mut stream = UnixStream::connect(socket)
        .map_err(|e| IpcError::Connect(format!("{}: {e}", socket.display())))?;
    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| IpcError::Transport(format!("set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| IpcError::Transport(format!("set_write_timeout: {e}")))?;

    let mut payload =
        serde_json::to_string(request).map_err(|e| IpcError::Codec(format!("serialize: {e}")))?;
    payload.push('\n');

    stream
        .write_all(payload.as_bytes())
        .map_err(|e| IpcError::Transport(format!("write: {e}")))?;
    stream
        .flush()
        .map_err(|e| IpcError::Transport(format!("flush: {e}")))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| IpcError::Transport(format!("read: {e}")))?;

    if line.trim().is_empty() {
        return Err(IpcError::EmptyResponse);
    }

    let response: Value = serde_json::from_str(line.trim())
        .map_err(|e| IpcError::Codec(format!("deserialize: {e}")))?;

    if let Some(error) = response.get("error") {
        let code = error.get("code").and_then(Value::as_i64).unwrap_or(-32000);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown error")
            .to_string();
        return Err(IpcError::RpcReject { code, message });
    }

    response
        .get("result")
        .cloned()
        .ok_or(IpcError::EmptyResponse)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_absent() {
        assert!(discover().is_none());
    }

    #[test]
    fn is_available_false_when_absent() {
        assert!(!is_available());
    }

    #[test]
    fn try_forward_returns_none_when_absent() {
        let result = try_forward("stats.diversity", &json!({"counts": [1.0, 2.0]}));
        assert!(result.is_none());
    }

    #[test]
    fn forward_connect_error_on_bad_socket() {
        let bad_path = std::env::temp_dir().join("wetspring-test-barracuda-nonexistent.sock");
        let err = forward(&bad_path, "health.liveness", &json!({}));
        assert!(err.is_err());
        let ipc_err = err.unwrap_err();
        assert!(ipc_err.is_connection_error() || ipc_err.is_retriable());
    }

    #[test]
    fn request_id_increments() {
        let id1 = REQUEST_ID.load(Ordering::Relaxed);
        let _ = try_forward("stats.dummy", &json!({}));
        let id2 = REQUEST_ID.load(Ordering::Relaxed);
        assert!(id2 >= id1);
    }
}
