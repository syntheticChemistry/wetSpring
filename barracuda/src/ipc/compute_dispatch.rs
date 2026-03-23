// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed `compute.dispatch.*` IPC client for toadStool compute orchestration.
//!
//! Provides capability-based dispatch to the toadStool compute orchestrator
//! via JSON-RPC 2.0 over Unix domain sockets. Discovery is runtime-only —
//! no hardcoded socket paths.
//!
//! RPC: `compute.dispatch.submit`, `compute.dispatch.result`, `compute.dispatch.capabilities`
//! (toadStool S156+). Discovery: `TOADSTOOL_SOCKET`, then
//! `$XDG_RUNTIME_DIR/biomeos/toadstool-default.sock`, then temp dir; standalone when absent.

pub use super::dispatch_strategy::*;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const RPC_TIMEOUT: Duration = Duration::from_secs(30);

/// Discover the toadStool compute socket.
///
/// Returns `None` if no toadStool socket is found (standalone mode).
#[must_use]
pub fn discover() -> Option<PathBuf> {
    super::discover::discover_socket(
        &super::discover::socket_env_var(super::primal_names::TOADSTOOL),
        super::primal_names::TOADSTOOL,
    )
}

/// Submit a compute workload to toadStool.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if toadStool is not discovered,
/// or transport/RPC errors if the call fails.
pub fn submit(workload_type: &str, params_json: &str) -> Result<DispatchHandle, DispatchError> {
    let socket = discover().ok_or(DispatchError::NoComputePrimal)?;
    submit_to(&socket, workload_type, params_json)
}

/// Submit a compute workload to a specific toadStool socket.
///
/// # Errors
///
/// Returns transport or RPC errors if the call fails.
pub fn submit_to(
    socket: &Path,
    workload_type: &str,
    params_json: &str,
) -> Result<DispatchHandle, DispatchError> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"compute.dispatch.submit","params":{{"workload_type":"{workload_type}","params":{params_json}}},"id":1}}"#,
    );
    let response = rpc_call(socket, &request)?;
    parse_submit_response(&response)
}

/// Poll for a job result from toadStool.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if toadStool is not discovered,
/// or transport/RPC errors if the call fails.
pub fn result(job_id: &str) -> Result<DispatchResult, DispatchError> {
    let socket = discover().ok_or(DispatchError::NoComputePrimal)?;
    result_from(&socket, job_id)
}

/// Poll for a job result from a specific toadStool socket.
///
/// # Errors
///
/// Returns transport or RPC errors if the call fails.
pub fn result_from(socket: &Path, job_id: &str) -> Result<DispatchResult, DispatchError> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"compute.dispatch.result","params":{{"job_id":"{job_id}"}},"id":2}}"#,
    );
    let response = rpc_call(socket, &request)?;
    Ok(parse_result_response(&response, job_id))
}

/// Query available compute backends from toadStool.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if toadStool is not discovered,
/// or transport/RPC errors if the call fails.
pub fn capabilities() -> Result<Vec<ComputeBackend>, DispatchError> {
    let socket = discover().ok_or(DispatchError::NoComputePrimal)?;
    capabilities_from(&socket)
}

/// Query available compute backends from a specific toadStool socket.
///
/// # Errors
///
/// Returns transport or RPC errors if the call fails.
pub fn capabilities_from(socket: &Path) -> Result<Vec<ComputeBackend>, DispatchError> {
    let request =
        r#"{"jsonrpc":"2.0","method":"compute.dispatch.capabilities","params":{},"id":3}"#;
    let response = rpc_call(socket, request)?;
    parse_capabilities_response(&response)
}

fn rpc_call(socket: &Path, request: &str) -> Result<String, DispatchError> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| DispatchError::Transport(format!("connect {}: {e}", socket.display())))?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| DispatchError::Transport(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| DispatchError::Transport(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| DispatchError::Transport(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| DispatchError::Transport(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| DispatchError::Transport(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| DispatchError::Transport(format!("read: {e}")))?;

    if line.is_empty() {
        return Err(DispatchError::Transport(
            "empty response from toadStool".to_string(),
        ));
    }

    if let Some((code, message)) = super::protocol::extract_rpc_error(&line) {
        return Err(DispatchError::Rpc { code, message });
    }

    Ok(line)
}

/// Submit a workload with outcome-based error separation.
///
/// Unlike [`submit`], this separates protocol errors (retriable) from
/// application errors (deterministic rejection by toadStool).
///
/// # Errors
///
/// Never returns `Err`; all failure modes are encoded in the
/// [`DispatchOutcome`] variants.
#[must_use]
pub fn submit_outcome(workload_type: &str, params_json: &str) -> DispatchOutcome<DispatchHandle> {
    let Some(socket) = discover() else {
        return DispatchOutcome::Protocol(DispatchError::NoComputePrimal);
    };
    submit_outcome_to(&socket, workload_type, params_json)
}

/// Submit a workload to a specific socket with outcome-based error separation.
#[must_use]
pub fn submit_outcome_to(
    socket: &Path,
    workload_type: &str,
    params_json: &str,
) -> DispatchOutcome<DispatchHandle> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"compute.dispatch.submit","params":{{"workload_type":"{workload_type}","params":{params_json}}},"id":1}}"#,
    );
    match rpc_call_outcome(socket, &request) {
        DispatchOutcome::Success(resp) => match parse_submit_response(&resp) {
            Ok(handle) => DispatchOutcome::Success(handle),
            Err(e) => DispatchOutcome::Protocol(e),
        },
        DispatchOutcome::Protocol(e) => DispatchOutcome::Protocol(e),
        DispatchOutcome::Application { code, message } => {
            DispatchOutcome::Application { code, message }
        }
    }
}

/// Low-level RPC call returning [`DispatchOutcome`].
///
/// Separates transport errors ([`Protocol`](DispatchOutcome::Protocol)) from
/// JSON-RPC error responses ([`Application`](DispatchOutcome::Application)).
fn rpc_call_outcome(socket: &Path, request: &str) -> DispatchOutcome<String> {
    let stream = match UnixStream::connect(socket) {
        Ok(s) => s,
        Err(e) => {
            return DispatchOutcome::Protocol(DispatchError::Transport(format!(
                "connect {}: {e}",
                socket.display()
            )));
        }
    };

    if let Err(e) = stream.set_read_timeout(Some(RPC_TIMEOUT)) {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!(
            "set read timeout: {e}"
        )));
    }
    if let Err(e) = stream.set_write_timeout(Some(RPC_TIMEOUT)) {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!(
            "set write timeout: {e}"
        )));
    }

    let mut writer = std::io::BufWriter::new(&stream);
    if let Err(e) = writer.write_all(request.as_bytes()) {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!("write: {e}")));
    }
    if let Err(e) = writer.write_all(b"\n") {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!("write newline: {e}")));
    }
    if let Err(e) = writer.flush() {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!("flush: {e}")));
    }

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    if let Err(e) = reader.read_line(&mut line) {
        return DispatchOutcome::Protocol(DispatchError::Transport(format!("read: {e}")));
    }

    if line.is_empty() {
        return DispatchOutcome::Protocol(DispatchError::Transport(
            "empty response from toadStool".to_string(),
        ));
    }

    if let Some((code, message)) = super::protocol::extract_rpc_error(&line) {
        return DispatchOutcome::Application { code, message };
    }

    DispatchOutcome::Success(line)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_without_socket() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                assert!(discover().is_none());
            },
        );
    }

    #[test]
    fn discover_explicit_env() {
        let sock = crate::ipc::test_socket_path("compute_dispatch_discover_explicit_env");
        crate::ipc::cleanup_test_socket(&sock);
        std::fs::write(&sock, "").unwrap();

        temp_env::with_var("TOADSTOOL_SOCKET", Some(sock.to_str().unwrap()), || {
            let found = discover();
            assert_eq!(found, Some(sock.clone()));
        });
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn submit_no_primal_errors() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let err = submit("gemm", "{}").unwrap_err();
                assert!(
                    matches!(err, DispatchError::NoComputePrimal),
                    "expected NoComputePrimal, got {err}"
                );
            },
        );
    }

    #[test]
    fn result_no_primal_errors() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let err = result("job-123").unwrap_err();
                assert!(matches!(err, DispatchError::NoComputePrimal));
            },
        );
    }

    #[test]
    fn capabilities_no_primal_errors() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let err = capabilities().unwrap_err();
                assert!(matches!(err, DispatchError::NoComputePrimal));
            },
        );
    }

    #[test]
    fn parse_submit_response_ok() {
        let sock = crate::ipc::test_socket_path("compute_dispatch_parse_submit_response_ok");
        crate::ipc::cleanup_test_socket(&sock);
        let response = format!(
            r#"{{"jsonrpc":"2.0","result":{{"job_id":"abc-123","compute_socket":"{}"}},"id":1}}"#,
            sock.display()
        );
        let handle = parse_submit_response(&response).unwrap();
        assert_eq!(handle.job_id, "abc-123");
        assert_eq!(handle.compute_socket, Some(sock.clone()));
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn parse_submit_response_no_compute_socket() {
        let response = r#"{"jsonrpc":"2.0","result":{"job_id":"def-456"},"id":1}"#;
        let handle = parse_submit_response(response).unwrap();
        assert_eq!(handle.job_id, "def-456");
        assert!(handle.compute_socket.is_none());
    }

    #[test]
    fn parse_submit_response_missing_job_id() {
        let response = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
        let err = parse_submit_response(response).unwrap_err();
        assert!(matches!(err, DispatchError::MalformedResponse(_)));
    }

    #[test]
    fn parse_result_response_completed() {
        let response =
            r#"{"jsonrpc":"2.0","result":{"status":"completed","payload":"[1.0,2.0]"},"id":2}"#;
        let dr = parse_result_response(response, "job-x");
        assert_eq!(dr.status, "completed");
        assert_eq!(dr.payload.as_deref(), Some("[1.0,2.0]"));
        assert!(dr.error.is_none());
    }

    #[test]
    fn parse_result_response_failed() {
        let response =
            r#"{"jsonrpc":"2.0","result":{"status":"failed","error":"out of memory"},"id":2}"#;
        let dr = parse_result_response(response, "job-y");
        assert_eq!(dr.status, "failed");
        assert_eq!(dr.error.as_deref(), Some("out of memory"));
    }

    #[test]
    fn extract_json_string_basic() {
        let json = r#"{"foo":"bar","baz":"qux"}"#;
        assert_eq!(extract_json_string(json, "foo").as_deref(), Some("bar"));
        assert_eq!(extract_json_string(json, "baz").as_deref(), Some("qux"));
        assert!(extract_json_string(json, "missing").is_none());
    }

    #[test]
    fn submit_outcome_no_primal() {
        temp_env::with_vars(
            [
                ("TOADSTOOL_SOCKET", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let outcome = submit_outcome("gemm", "{}");
                assert!(outcome.is_retriable());
                assert!(!outcome.is_success());
            },
        );
    }

    #[test]
    fn dispatch_outcome_into_result_success() {
        let outcome: DispatchOutcome<i32> = DispatchOutcome::Success(42);
        assert!(outcome.is_success());
        assert_eq!(outcome.into_result().unwrap(), 42);
    }

    #[test]
    fn dispatch_outcome_into_result_protocol() {
        let outcome: DispatchOutcome<i32> =
            DispatchOutcome::Protocol(DispatchError::NoComputePrimal);
        assert!(outcome.is_retriable());
        let err = outcome.into_result().unwrap_err();
        assert!(matches!(err, DispatchError::NoComputePrimal));
    }

    #[test]
    fn dispatch_outcome_into_result_application() {
        let outcome: DispatchOutcome<i32> = DispatchOutcome::Application {
            code: -32602,
            message: "invalid params".into(),
        };
        assert!(!outcome.is_retriable());
        assert!(!outcome.is_success());
        let err = outcome.into_result().unwrap_err();
        assert!(matches!(err, DispatchError::Rpc { code: -32602, .. }));
    }

    #[test]
    fn dispatch_error_display() {
        let e1 = DispatchError::NoComputePrimal;
        assert!(e1.to_string().contains("no toadStool"));

        let e2 = DispatchError::Transport("connect refused".into());
        assert!(e2.to_string().contains("connect refused"));

        let e3 = DispatchError::Rpc {
            code: -32601,
            message: "method not found".into(),
        };
        assert!(e3.to_string().contains("-32601"));

        let e4 = DispatchError::MalformedResponse("no job_id".into());
        assert!(e4.to_string().contains("no job_id"));
    }
}
