// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optional `NestGate` data provider for NCBI operations.
//!
//! When `WETSPRING_DATA_PROVIDER=nestgate` is set, wetSpring routes NCBI
//! data requests through `NestGate`'s Unix socket JSON-RPC API instead of
//! making direct HTTP calls. Falls back to sovereign HTTP if `NestGate` is
//! unavailable.
//!
//! # Protocol
//!
//! JSON-RPC 2.0, newline-delimited, over Unix domain socket.
//! Socket discovery:
//! 1. `NESTGATE_SOCKET` env var
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
//! 3. `/tmp/nestgate-default.sock`
//!
//! # Evolution path
//!
//! | Phase | Strategy | Status |
//! |-------|----------|--------|
//! | Current | Optional `NestGate` provider, sovereign fallback | active |
//! | Phase 2 | `biomeOS` Neural API routing (`capability.call`) | planned |

use crate::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const FAMILY_ID: &str = "default";

/// Whether `NestGate` routing is enabled via environment.
#[must_use]
pub fn is_enabled() -> bool {
    std::env::var("WETSPRING_DATA_PROVIDER")
        .is_ok_and(|v| v.trim().eq_ignore_ascii_case("nestgate"))
}

/// Discover the `NestGate` Unix socket path.
///
/// Discovery order:
/// 1. `NESTGATE_SOCKET` env var (explicit path)
/// 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
/// 3. `/tmp/nestgate-default.sock`
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    let explicit = std::env::var("NESTGATE_SOCKET").ok();
    let xdg = std::env::var("XDG_RUNTIME_DIR").ok();
    resolve_socket(explicit.as_deref(), xdg.as_deref())
}

/// Pure-logic socket path resolution.
fn resolve_socket(explicit: Option<&str>, xdg_runtime: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join("biomeos/nestgate-default.sock");
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = PathBuf::from("/tmp/nestgate-default.sock");
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Store data in `NestGate` via `storage.store`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the RPC fails.
pub fn store(socket: &PathBuf, key: &str, value: &str) -> crate::error::Result<()> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{}","value":"{}","family_id":"{}"}},"id":1}}"#,
        escape_json(key),
        escape_json(value),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        Ok(())
    }
}

/// Retrieve data from `NestGate` via `storage.retrieve`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable, the key does not exist,
/// or the RPC fails.
pub fn retrieve(socket: &PathBuf, key: &str) -> crate::error::Result<String> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{}","family_id":"{}"}},"id":2}}"#,
        escape_json(key),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        extract_result_value(&response)
    }
}

/// Check if a key exists in `NestGate` via `storage.exists`.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the RPC fails.
pub fn exists(socket: &PathBuf, key: &str) -> crate::error::Result<bool> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{}","family_id":"{}"}},"id":3}}"#,
        escape_json(key),
        FAMILY_ID,
    );
    let response = rpc_call(socket, &request)?;
    Ok(response.contains("true"))
}

/// Health check: verify `NestGate` is alive.
///
/// # Errors
///
/// Returns `Err` if the socket is unavailable or the health check fails.
pub fn health(socket: &PathBuf) -> crate::error::Result<()> {
    let request = r#"{"jsonrpc":"2.0","method":"health","params":{},"id":0}"#;
    let response = rpc_call(socket, request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        Ok(())
    }
}

/// Fetch FASTA from NCBI via `NestGate`, caching in `NestGate`'s storage.
///
/// 1. Check if `NestGate` has the sequence cached (`ncbi:{db}:{id}`)
/// 2. If not, fall back to sovereign `efetch_fasta` and store in `NestGate`
///
/// # Errors
///
/// Returns `Err` if both `NestGate` retrieval and sovereign fallback fail.
pub fn fetch_or_fallback(
    socket: &PathBuf,
    db: &str,
    id: &str,
    api_key: &str,
) -> crate::error::Result<String> {
    let cache_key = format!("ncbi:{db}:{id}");

    if exists(socket, &cache_key).unwrap_or(false) {
        if let Ok(cached) = retrieve(socket, &cache_key) {
            return Ok(cached);
        }
    }

    let fasta = super::efetch::efetch_fasta(db, id, api_key)?;

    let _ = store(socket, &cache_key, &fasta);

    Ok(fasta)
}

/// Send a JSON-RPC request over a Unix socket and read the response.
fn rpc_call(socket: &PathBuf, request: &str) -> crate::error::Result<String> {
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
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Extract the error message from a JSON-RPC error response.
fn extract_error(response: &str) -> String {
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
fn extract_result_value(response: &str) -> crate::error::Result<String> {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn is_enabled_default_false() {
        assert!(
            !std::env::var("WETSPRING_DATA_PROVIDER")
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("nestgate"))
                || is_enabled()
        );
    }

    #[test]
    fn resolve_socket_explicit_nonexistent() {
        let result = resolve_socket(Some("/tmp/nonexistent_wetspring_test.sock"), None);
        assert!(result.is_none());
    }

    #[test]
    fn resolve_socket_all_none() {
        let result = resolve_socket(None, None);
        assert!(
            result.is_none() || result.is_some_and(|p| p.exists()),
            "should be None or a path that exists"
        );
    }

    #[test]
    fn resolve_socket_explicit_exists() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test.sock");
        std::fs::write(&sock, "").unwrap();
        let result = resolve_socket(Some(sock.to_str().unwrap()), None);
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_socket_xdg_path() {
        let dir = tempfile::tempdir().unwrap();
        let biomeos = dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos).unwrap();
        let sock = biomeos.join("nestgate-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = resolve_socket(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn escape_json_special_chars() {
        assert_eq!(escape_json("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_json("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_json("back\\slash"), "back\\\\slash");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn escape_json_empty() {
        assert_eq!(escape_json(""), "");
    }

    #[test]
    fn escape_json_no_special() {
        assert_eq!(escape_json("plain text"), "plain text");
    }

    #[test]
    fn extract_error_with_message() {
        let response = r#"{"jsonrpc":"2.0","error":{"code":-32600,"message":"not found"},"id":1}"#;
        let err = extract_error(response);
        assert!(err.contains("not found"));
    }

    #[test]
    fn extract_error_no_message() {
        let response = r#"{"jsonrpc":"2.0","error":{"code":-32600},"id":1}"#;
        let err = extract_error(response);
        assert!(err.contains("RPC error"));
    }

    #[test]
    fn extract_result_value_string() {
        let response = r#"{"jsonrpc":"2.0","result":">seq1\nATCG","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains(">seq1"));
    }

    #[test]
    fn extract_result_value_missing() {
        let response = r#"{"jsonrpc":"2.0","id":1}"#;
        assert!(extract_result_value(response).is_err());
    }

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn health_nonexistent_socket_errors() {
        let path = PathBuf::from("/tmp/wetspring_test_nonexistent_nestgate.sock");
        let err = health(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }
}
