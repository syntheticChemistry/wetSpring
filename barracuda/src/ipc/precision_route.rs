// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed `barracuda.precision.route` IPC client.
//!
//! Tier 2 precision advisory: queries barraCuda for recommended precision
//! tier given a domain and hardware hint. Used before workload dispatch to
//! select DF64/F64/F32 paths.
//!
//! RPC: `barracuda.precision.route` (barraCuda S250+).
//! Discovery: `BARRACUDA_SOCKET`, then `barracuda-{family_id}.sock`.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const RPC_TIMEOUT: Duration = super::timeouts::COMPUTE;

/// Precision routing advice from `barracuda.precision.route`.
#[derive(Debug)]
pub struct PrecisionAdvice {
    /// Recommended precision tier (e.g. `"DF64"`, `"F64"`, `"F32"`).
    pub recommended_tier: String,
    /// Whether fused multiply-add is safe for this domain+hardware.
    pub fma_safe: bool,
    /// Whether a shader compiler is required.
    pub requires_compiler: bool,
    /// Hardware hint echoed or refined by barraCuda.
    pub hardware_hint: String,
}

/// Errors from `barracuda.precision.route` RPC calls.
#[derive(Debug)]
pub enum PrecisionError {
    /// barraCuda socket not discovered.
    NoBarraCuda,
    /// Transport-level failure.
    Transport(String),
    /// Protocol-level failure.
    Protocol(String),
}

impl std::fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBarraCuda => f.write_str("barraCuda not discovered"),
            Self::Transport(msg) => write!(f, "transport: {msg}"),
            Self::Protocol(msg) => write!(f, "protocol: {msg}"),
        }
    }
}

/// Discover the barraCuda precision routing socket.
#[must_use]
pub fn discover() -> Option<PathBuf> {
    super::discover::discover_socket(
        &super::discover::socket_env_var(super::primal_names::BARRACUDA),
        super::primal_names::BARRACUDA,
    )
}

/// Query barraCuda for precision routing advice.
///
/// # Errors
///
/// Returns [`PrecisionError::NoBarraCuda`] if barraCuda is not discovered,
/// or transport/protocol errors.
pub fn route(domain: &str, hardware_hint: &str) -> Result<PrecisionAdvice, PrecisionError> {
    let socket = discover().ok_or(PrecisionError::NoBarraCuda)?;
    route_at(&socket, domain, hardware_hint)
}

/// Query precision routing from a specific barraCuda socket.
///
/// # Errors
///
/// Returns transport or protocol errors.
pub fn route_at(
    socket: &Path,
    domain: &str,
    hardware_hint: &str,
) -> Result<PrecisionAdvice, PrecisionError> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"barracuda.precision.route","params":{{"domain":"{domain}","hardware_hint":"{hardware_hint}"}},"id":1}}"#,
    );
    let response = rpc_call(socket, &request)?;
    parse_route_response(&response)
}

fn rpc_call(socket: &Path, request: &str) -> Result<String, PrecisionError> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| PrecisionError::Transport(format!("connect {}: {e}", socket.display())))?;

    stream
        .set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| PrecisionError::Transport(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| PrecisionError::Transport(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| PrecisionError::Transport(format!("write: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| PrecisionError::Transport(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| PrecisionError::Transport(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| PrecisionError::Transport(format!("read: {e}")))?;

    if line.is_empty() {
        return Err(PrecisionError::Transport(
            "empty response from barraCuda".to_string(),
        ));
    }
    Ok(line)
}

fn parse_route_response(response: &str) -> Result<PrecisionAdvice, PrecisionError> {
    let v: serde_json::Value = serde_json::from_str(response)
        .map_err(|e| PrecisionError::Protocol(format!("JSON parse: {e}")))?;

    if let Some(err) = v.get("error") {
        return Err(PrecisionError::Protocol(format!(
            "RPC error: {}",
            err.get("message").and_then(|m| m.as_str()).unwrap_or("unknown")
        )));
    }

    let result = v
        .get("result")
        .ok_or_else(|| PrecisionError::Protocol("missing result".to_string()))?;

    Ok(PrecisionAdvice {
        recommended_tier: result
            .get("recommended_tier")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("F64")
            .to_string(),
        fma_safe: result
            .get("fma_safe")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        requires_compiler: result
            .get("requires_compiler")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        hardware_hint: result
            .get("hardware_hint")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_route_response_ok() {
        let json = r#"{"jsonrpc":"2.0","result":{"recommended_tier":"DF64","fma_safe":true,"requires_compiler":false,"hardware_hint":"compute"},"id":1}"#;
        let result = parse_route_response(json).expect("parse ok");
        assert_eq!(result.recommended_tier, "DF64");
        assert!(result.fma_safe);
        assert!(!result.requires_compiler);
        assert_eq!(result.hardware_hint, "compute");
    }

    #[test]
    fn parse_route_response_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"code":-32602,"message":"unknown domain"},"id":1}"#;
        let result = parse_route_response(json);
        assert!(result.is_err());
    }
}
