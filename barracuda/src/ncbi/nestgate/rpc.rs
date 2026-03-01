// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC transport over Unix socket.

use crate::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Send a JSON-RPC request over a Unix socket and read the response.
pub fn rpc_call(socket: &Path, request: &str) -> crate::error::Result<String> {
    let stream = UnixStream::connect_addr(
        &std::os::unix::net::SocketAddr::from_pathname(socket)
            .map_err(|e| Error::Ncbi(format!("invalid socket path: {e}")))?,
    )
    .map_err(|e| Error::Ncbi(format!("NestGate connect {}: {e}", socket.display())))?;

    stream
        .set_read_timeout(Some(READ_TIMEOUT))
        .map_err(|e| Error::Ncbi(format!("set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(CONNECT_TIMEOUT))
        .map_err(|e| Error::Ncbi(format!("set write timeout: {e}")))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| Error::Ncbi(format!("write to NestGate: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| Error::Ncbi(format!("write newline: {e}")))?;
    writer
        .flush()
        .map_err(|e| Error::Ncbi(format!("flush: {e}")))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| Error::Ncbi(format!("read from NestGate: {e}")))?;

    if line.is_empty() {
        return Err(Error::Ncbi("NestGate returned empty response".to_string()));
    }

    Ok(line)
}

/// Minimal JSON string escaping for values embedded in RPC requests.
#[must_use]
pub fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Extract the error message from a JSON-RPC error response.
#[must_use]
pub fn extract_error(response: &str) -> String {
    if let Some(start) = response.find("\"message\"") {
        if let Some(colon) = response[start..].find(':') {
            let after_colon = &response[start + colon + 1..];
            let trimmed = after_colon.trim_start();
            if let Some(inner) = trimmed.strip_prefix('"') {
                if let Some(end) = inner.find('"') {
                    return inner[..end].to_string();
                }
            }
        }
    }
    format!(
        "NestGate RPC error: {}",
        &response[..response.len().min(200)]
    )
}

/// Extract the `result.value` or `result` string from a JSON-RPC response.
///
/// # Errors
///
/// Returns `Err` if the response does not contain a valid `result` field.
pub fn extract_result_value(response: &str) -> crate::error::Result<String> {
    if let Some(start) = response.find("\"result\"") {
        if let Some(colon) = response[start..].find(':') {
            let after_colon = &response[start + colon + 1..];
            let trimmed = after_colon.trim_start();
            if let Some(inner) = trimmed.strip_prefix('"') {
                if let Some(end) = inner.find('"') {
                    let raw = &inner[..end];
                    return Ok(raw.replace("\\n", "\n").replace("\\\"", "\""));
                }
            }
            if let Some(end) = trimmed.find('}') {
                return Ok(trimmed[..end].to_string());
            }
        }
    }
    Err(Error::Ncbi(
        "could not extract result from NestGate response".to_string(),
    ))
}

/// Health check: verify `NestGate` is alive.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the health check fails.
pub fn health(socket: &Path) -> crate::error::Result<()> {
    let request = r#"{"jsonrpc":"2.0","method":"health","params":{},"id":0}"#;
    let response = rpc_call(socket, request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        Ok(())
    }
}

/// Extract a FASTA string from a biomeOS capability.call response.
///
/// Handles both `{"result":{"fasta":">seq..."}}` and bare `{"result":">seq..."}`.
pub fn extract_fasta_from_response(response: &str) -> crate::error::Result<String> {
    if let Some(start) = response.find("\"fasta\"") {
        if let Some(colon) = response[start..].find(':') {
            let after = &response[start + colon + 1..];
            let trimmed = after.trim_start();
            if let Some(inner) = trimmed.strip_prefix('"') {
                if let Some(end) = inner.find('"') {
                    let raw = &inner[..end];
                    return Ok(raw.replace("\\n", "\n").replace("\\\"", "\""));
                }
            }
        }
    }
    extract_result_value(response)
}
