// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute dispatch outcomes, errors, and typed toadStool response shapes.
//!
//! Defines how callers classify **protocol** vs **application** failures and the
//! structured payloads returned from `compute.dispatch.*` RPC methods. Parsing
//! helpers turn newline-delimited JSON-RPC lines into these types; see
//! [`super::compute_dispatch`] for socket I/O and orchestration.

use std::path::PathBuf;

/// Errors from compute dispatch operations.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// No toadStool compute socket found (standalone mode).
    #[error("no toadStool compute primal discovered")]
    NoComputePrimal,
    /// Transport-level failure (socket connect, read, write).
    #[error("compute dispatch transport: {0}")]
    Transport(String),
    /// toadStool returned a JSON-RPC error.
    #[error("compute dispatch RPC [{code}]: {message}")]
    Rpc {
        /// JSON-RPC error code.
        code: i64,
        /// Error message from toadStool.
        message: String,
    },
    /// Response missing expected fields.
    #[error("compute dispatch malformed response: {0}")]
    MalformedResponse(String),
}

/// Handle returned by `compute.dispatch.submit`.
#[derive(Debug, Clone)]
pub struct DispatchHandle {
    /// Unique job identifier assigned by toadStool.
    pub job_id: String,
    /// Socket path for polling results (may differ from dispatch socket).
    pub compute_socket: Option<PathBuf>,
}

/// Result payload from `compute.dispatch.result`.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Job identifier.
    pub job_id: String,
    /// Job status: "completed", "running", "failed".
    pub status: String,
    /// Result payload (JSON string) when status is "completed".
    pub payload: Option<String>,
    /// Error message when status is "failed".
    pub error: Option<String>,
}

/// Available compute backend from `compute.dispatch.capabilities`.
#[derive(Debug, Clone)]
pub struct ComputeBackend {
    /// Backend identifier (e.g., "gpu-vulkan", "cpu-avx2", "npu-akida").
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Supported workload types.
    pub workload_types: Vec<String>,
}

/// Outcome of a compute dispatch operation that separates protocol-level
/// IPC errors from application-level logic errors.
///
/// Following the groundSpring/airSpring/sweetGrass pattern, callers can:
/// - **Retry** on [`Protocol`](Self::Protocol) (transient transport failures)
/// - **Report** on [`Application`](Self::Application) (workload rejected, invalid params)
/// - **Use** on [`Success`](Self::Success) (job handle, result, capabilities)
#[derive(Debug)]
pub enum DispatchOutcome<T> {
    /// The RPC call succeeded and returned a valid payload.
    Success(T),
    /// Protocol-level failure (transport, codec, no primal found).
    /// These are potentially retriable.
    Protocol(DispatchError),
    /// Application-level failure (toadStool accepted the call but the
    /// workload itself failed — e.g. invalid params, unsupported type).
    Application {
        /// JSON-RPC error code from toadStool.
        code: i64,
        /// Human-readable error description.
        message: String,
    },
}

impl<T> DispatchOutcome<T> {
    /// Whether the outcome represents a successful result.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Whether the error is potentially retriable (protocol-level).
    #[must_use]
    pub const fn is_retriable(&self) -> bool {
        matches!(self, Self::Protocol(_))
    }

    /// Convert to a standard `Result`, collapsing both error kinds.
    ///
    /// # Errors
    ///
    /// Returns `Err` for both [`Protocol`](Self::Protocol) and
    /// [`Application`](Self::Application) outcomes.
    pub fn into_result(self) -> Result<T, DispatchError> {
        match self {
            Self::Success(val) => Ok(val),
            Self::Protocol(err) => Err(err),
            Self::Application { code, message } => Err(DispatchError::Rpc { code, message }),
        }
    }
}

pub(crate) fn parse_submit_response(response: &str) -> Result<DispatchHandle, DispatchError> {
    let job_id = extract_json_string(response, "job_id").ok_or_else(|| {
        DispatchError::MalformedResponse("missing job_id in submit response".to_string())
    })?;
    let compute_socket = extract_json_string(response, "compute_socket").map(PathBuf::from);
    Ok(DispatchHandle {
        job_id,
        compute_socket,
    })
}

pub(crate) fn parse_result_response(response: &str, job_id: &str) -> DispatchResult {
    let status = extract_json_string(response, "status").unwrap_or_else(|| "unknown".to_string());
    let payload = extract_json_string(response, "payload");
    let error = extract_json_string(response, "error");
    DispatchResult {
        job_id: job_id.to_string(),
        status,
        payload,
        error,
    }
}

pub(crate) fn parse_capabilities_response(
    response: &str,
) -> Result<Vec<ComputeBackend>, DispatchError> {
    let mut backends = Vec::new();

    if !response.contains("\"result\"") {
        return Err(DispatchError::MalformedResponse(
            "missing result field".to_string(),
        ));
    }

    let backends_start = response.find("\"backends\"");
    if backends_start.is_none() {
        return Ok(backends);
    }

    let mut depth = 0_i32;
    let mut in_backends = false;
    let mut current_start = 0;

    let bytes = response.as_bytes();
    let Some(arr_start) = response[backends_start.unwrap_or(0)..].find('[') else {
        return Ok(backends);
    };
    let arr_abs = backends_start.unwrap_or(0) + arr_start;

    for (i, &b) in bytes.iter().enumerate().skip(arr_abs) {
        match b {
            b'[' if !in_backends => {
                in_backends = true;
                depth = 1;
            }
            b'{' if in_backends => {
                if depth == 1 {
                    current_start = i;
                }
                depth += 1;
            }
            b'}' if in_backends => {
                depth -= 1;
                if depth == 1 {
                    let obj = &response[current_start..=i];
                    let id = extract_json_string(obj, "id").unwrap_or_default();
                    let description = extract_json_string(obj, "description").unwrap_or_default();
                    backends.push(ComputeBackend {
                        id,
                        description,
                        workload_types: Vec::new(),
                    });
                }
            }
            b']' if in_backends && depth == 1 => break,
            _ => {}
        }
    }

    Ok(backends)
}

/// Extract a JSON string value for a given key (lightweight, no serde).
pub(crate) fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = json.find(&needle)?;
    let value_start = start + needle.len();
    let rest = &json[value_start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}
