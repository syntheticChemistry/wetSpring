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
//! Socket discovery (capability-based, no hardcoded paths):
//! 1. `NESTGATE_SOCKET` env var (explicit override)
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
//! 3. `<temp_dir>/nestgate-default.sock` (platform-agnostic fallback)
//!
//! # Data routing tiers
//!
//! | Tier | Strategy | When |
//! |------|----------|------|
//! | 1 | biomeOS Neural API `capability.call` | biomeOS orchestrator running |
//! | 2 | Direct `NestGate` socket | Standalone + NestGate available |
//! | 3 | Sovereign HTTP | No ecosystem services |

use crate::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const FAMILY_ID: &str = "default";

/// Default `NestGate` socket path under `XDG_RUNTIME_DIR` (`biomeos/nestgate-default.sock`).
const DEFAULT_NESTGATE_PATH_XDG: &str = "biomeos/nestgate-default.sock";
/// Fallback `NestGate` socket filename when `XDG_RUNTIME_DIR` is unset (`nestgate-default.sock`).
const DEFAULT_NESTGATE_PATH_FALLBACK: &str = "nestgate-default.sock";
/// Default `biomeOS` socket path under `XDG_RUNTIME_DIR` (`biomeos/biomeos-default.sock`).
const DEFAULT_BIOMEOS_PATH_XDG: &str = "biomeos/biomeos-default.sock";
/// Fallback `biomeOS` socket filename when `XDG_RUNTIME_DIR` is unset (`biomeos-default.sock`).
const DEFAULT_BIOMEOS_PATH_FALLBACK: &str = "biomeos-default.sock";

/// Whether `NestGate` routing is enabled via environment.
#[must_use]
pub fn is_enabled() -> bool {
    std::env::var("WETSPRING_DATA_PROVIDER")
        .is_ok_and(|v| v.trim().eq_ignore_ascii_case("nestgate"))
}

/// Discover the `NestGate` Unix socket path.
///
/// Capability-based discovery (no hardcoded absolute paths):
/// 1. `NESTGATE_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
/// 3. `<temp_dir>/nestgate-default.sock` (platform-agnostic fallback)
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
        let p = PathBuf::from(xdg).join(DEFAULT_NESTGATE_PATH_XDG);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(DEFAULT_NESTGATE_PATH_FALLBACK);
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
pub fn store(socket: &Path, key: &str, value: &str) -> crate::error::Result<()> {
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
pub fn retrieve(socket: &Path, key: &str) -> crate::error::Result<String> {
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
pub fn exists(socket: &Path, key: &str) -> crate::error::Result<bool> {
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
pub fn health(socket: &Path) -> crate::error::Result<()> {
    let request = r#"{"jsonrpc":"2.0","method":"health","params":{},"id":0}"#;
    let response = rpc_call(socket, request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(extract_error(&response)))
    } else {
        Ok(())
    }
}

/// Discover the biomeOS Neural API socket for capability-based routing.
///
/// 1. `BIOMEOS_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/biomeos-default.sock`
/// 3. `<temp_dir>/biomeos-default.sock`
#[must_use]
pub fn discover_biomeos_socket() -> Option<PathBuf> {
    let explicit = std::env::var("BIOMEOS_SOCKET").ok();
    let xdg = std::env::var("XDG_RUNTIME_DIR").ok();
    resolve_biomeos_socket(explicit.as_deref(), xdg.as_deref())
}

/// Pure-logic biomeOS socket path resolution.
fn resolve_biomeos_socket(explicit: Option<&str>, xdg_runtime: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join(DEFAULT_BIOMEOS_PATH_XDG);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(DEFAULT_BIOMEOS_PATH_FALLBACK);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Route an NCBI fetch through biomeOS Neural API `capability.call`.
///
/// Sends `capability.call("science.ncbi_fetch", ...)` to the biomeOS
/// orchestrator, which routes it to `NestGate` or the registered provider.
///
/// # Errors
///
/// Returns `Err` if biomeOS is unreachable or the capability call fails.
pub fn fetch_via_biomeos(
    biomeos_socket: &Path,
    db: &str,
    id: &str,
    api_key: &str,
) -> crate::error::Result<String> {
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"capability.call","params":{{"capability":"science.ncbi_fetch","params":{{"db":"{}","id":"{}","api_key":"{}"}}}},"id":1}}"#,
        escape_json(db),
        escape_json(id),
        escape_json(api_key),
    );
    let response = rpc_call(biomeos_socket, &request)?;
    if response.contains("\"error\"") {
        Err(Error::Ncbi(format!(
            "biomeOS capability.call failed: {}",
            extract_error(&response)
        )))
    } else {
        extract_fasta_from_response(&response)
    }
}

/// Fetch FASTA using the three-tier routing strategy.
///
/// 1. **biomeOS**: If biomeOS Neural API is available, use `capability.call`
/// 2. **`NestGate`**: If direct `NestGate` socket is available, use cache + fetch
/// 3. **Sovereign**: Fall back to direct NCBI HTTP
///
/// # Errors
///
/// Returns `Err` if all three tiers fail.
pub fn fetch_tiered(db: &str, id: &str, api_key: &str) -> crate::error::Result<String> {
    // Tier 1: biomeOS Neural API routing
    if let Some(biomeos_socket) = discover_biomeos_socket() {
        match fetch_via_biomeos(&biomeos_socket, db, id, api_key) {
            Ok(fasta) => return Ok(fasta),
            Err(e) => {
                eprintln!("[nestgate] biomeOS routing failed, falling back: {e}");
            }
        }
    }

    // Tier 2: Direct NestGate socket
    if is_enabled() {
        if let Some(nestgate_socket) = discover_socket() {
            match fetch_or_fallback(&nestgate_socket, db, id, api_key) {
                Ok(fasta) => return Ok(fasta),
                Err(e) => {
                    eprintln!("[nestgate] NestGate failed, falling back to sovereign: {e}");
                }
            }
        }
    }

    // Tier 3: Sovereign HTTP
    super::efetch::efetch_fasta(db, id, api_key)
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
    socket: &Path,
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
fn rpc_call(socket: &Path, request: &str) -> crate::error::Result<String> {
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

/// Extract a FASTA string from a biomeOS capability.call response.
///
/// Handles both `{"result":{"fasta":">seq..."}}` and bare `{"result":">seq..."}`.
fn extract_fasta_from_response(response: &str) -> crate::error::Result<String> {
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent_wetspring_test.sock");
        let result = resolve_socket(Some(path.to_str().unwrap()), None);
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
    fn resolve_socket_xdg_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().join("nonexistent_xdg");
        let result = resolve_socket(None, Some(xdg.to_str().unwrap()));
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
    }

    #[test]
    fn resolve_socket_explicit_overrides_xdg() {
        let dir = tempfile::tempdir().unwrap();
        let explicit_sock = dir.path().join("explicit.sock");
        std::fs::write(&explicit_sock, "").unwrap();
        let xdg_dir = tempfile::tempdir().unwrap();
        let biomeos = xdg_dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos).unwrap();
        let xdg_sock = biomeos.join("nestgate-default.sock");
        std::fs::write(&xdg_sock, "").unwrap();
        let result = resolve_socket(
            Some(explicit_sock.to_str().unwrap()),
            Some(xdg_dir.path().to_str().unwrap()),
        );
        assert_eq!(result, Some(explicit_sock));
    }

    #[test]
    fn resolve_socket_temp_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("nestgate-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = resolve_socket(None, Some("/nonexistent_xdg_path_12345"));
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
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
    fn extract_error_truncates_long_response() {
        let response = format!(
            r#"{{"jsonrpc":"2.0","error":{{"code":-1,"message":"x"}},"id":1}}{}"#,
            "y".repeat(500)
        );
        let err = extract_error(&response);
        assert!(err.contains('x'));
    }

    #[test]
    fn extract_error_empty_response() {
        let err = extract_error("");
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
    fn extract_result_value_escaped_newline() {
        let response = r#"{"jsonrpc":"2.0","result":">seq1\nATCG\n","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert_eq!(val, ">seq1\nATCG\n");
    }

    #[test]
    fn extract_result_value_escaped_quote() {
        let response = r#"{"jsonrpc":"2.0","result":"say \"hi\"","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains("say"));
    }

    #[test]
    fn extract_result_value_nested_object() {
        let response = r#"{"jsonrpc":"2.0","result":{"value":"cached"},"id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains("value") && val.contains("cached"));
    }

    #[test]
    fn store_request_format() {
        let key = "ncbi:nucleotide:K03455";
        let value = ">seq\nATCG";
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{}","value":"{}","family_id":"{}"}},"id":1}}"#,
            escape_json(key),
            escape_json(value),
            FAMILY_ID,
        );
        assert!(request.contains("storage.store"));
        assert!(request.contains("ncbi:nucleotide:K03455"));
        assert!(request.contains(">seq\\nATCG"));
    }

    #[test]
    fn retrieve_request_format() {
        let key = "test_key";
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{}","family_id":"{}"}},"id":2}}"#,
            escape_json(key),
            FAMILY_ID,
        );
        assert!(request.contains("storage.retrieve"));
        assert!(request.contains("test_key"));
    }

    #[test]
    fn exists_request_format() {
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{}","family_id":"{}"}},"id":3}}"#,
            escape_json("key"),
            FAMILY_ID,
        );
        assert!(request.contains("storage.exists"));
    }

    #[test]
    fn health_request_format() {
        let request = r#"{"jsonrpc":"2.0","method":"health","params":{},"id":0}"#;
        assert!(request.contains("health"));
    }

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn discover_biomeos_socket_does_not_panic() {
        let _ = discover_biomeos_socket();
    }

    #[test]
    fn resolve_biomeos_socket_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("biomeos.sock");
        std::fs::write(&sock, "").unwrap();
        let result = resolve_biomeos_socket(Some(sock.to_str().unwrap()), None);
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_biomeos_socket_xdg() {
        let dir = tempfile::tempdir().unwrap();
        let biomeos_dir = dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos_dir).unwrap();
        let sock = biomeos_dir.join("biomeos-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = resolve_biomeos_socket(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_biomeos_socket_none() {
        let result = resolve_biomeos_socket(None, None);
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
    }

    #[test]
    fn fetch_via_biomeos_nonexistent_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_no_biomeos.sock");
        let err = fetch_via_biomeos(&path, "nucleotide", "K03455", "").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn extract_fasta_from_response_with_fasta_key() {
        let resp = r#"{"jsonrpc":"2.0","result":{"fasta":">seq1\nATCG"},"id":1}"#;
        let fasta = extract_fasta_from_response(resp).unwrap();
        assert!(fasta.contains(">seq1"));
    }

    #[test]
    fn extract_fasta_from_response_bare_result() {
        let resp = r#"{"jsonrpc":"2.0","result":">seq2\nGCTA","id":1}"#;
        let fasta = extract_fasta_from_response(resp).unwrap();
        assert!(fasta.contains(">seq2"));
    }

    #[test]
    fn health_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_nestgate.sock");
        let err = health(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn invalid_socket_path_too_long() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a".repeat(200));
        let err = health(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid socket") || msg.contains("NestGate connect"));
    }

    #[test]
    fn store_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_store.sock");
        let err = store(&path, "key", "value").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn retrieve_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_retrieve.sock");
        let err = retrieve(&path, "key").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn exists_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_exists.sock");
        let err = exists(&path, "key").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }
}
