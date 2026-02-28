// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nest atomic storage client — JSON-RPC over Unix socket to NestGate.
//!
//! Provides a typed client for NestGate's storage API using the biomeOS
//! JSON-RPC 2.0 protocol. Supports both structured JSON storage and
//! binary blob storage (base64-encoded).
//!
//! # Socket Discovery
//!
//! NestGate socket is discovered in order:
//! 1. `NESTGATE_SOCKET` environment variable
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
//! 3. `/tmp/nestgate-default.sock`
//!
//! # NUCLEUS Role
//!
//! Nest = Tower + NestGate + Squirrel. This module implements the NestGate
//! storage interface that validation binaries and pipelines use to persist
//! and retrieve datasets without hardcoded paths.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const NESTGATE_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_FAMILY: &str = "default";

/// NestGate storage client.
///
/// Wraps a Unix socket path and provides typed methods for all
/// NestGate storage operations.
#[derive(Debug, Clone)]
pub struct NestClient {
    socket: PathBuf,
    family_id: String,
}

/// Result of a storage operation.
#[derive(Debug)]
pub struct StoreResult {
    /// Whether the operation succeeded.
    pub ok: bool,
    /// Raw JSON-RPC response (for inspection).
    pub raw: String,
}

/// Result of a list operation.
#[derive(Debug)]
pub struct ListResult {
    /// Keys matching the prefix.
    pub keys: Vec<String>,
    /// Raw JSON-RPC response.
    pub raw: String,
}

/// Result of a retrieve operation.
#[derive(Debug)]
pub struct RetrieveResult {
    /// The stored value as a JSON string.
    pub value: Option<String>,
    /// Raw JSON-RPC response.
    pub raw: String,
}

impl NestClient {
    /// Create a client for a discovered NestGate socket.
    ///
    /// Returns `None` if no NestGate socket is found.
    #[must_use]
    pub fn discover() -> Option<Self> {
        discover_nestgate_socket().map(|socket| Self {
            socket,
            family_id: DEFAULT_FAMILY.to_string(),
        })
    }

    /// Create a client for a specific socket path.
    #[must_use]
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            family_id: DEFAULT_FAMILY.to_string(),
        }
    }

    /// Set the family ID for all subsequent operations.
    #[must_use]
    pub fn with_family(mut self, family_id: impl Into<String>) -> Self {
        self.family_id = family_id.into();
        self
    }

    /// The socket path this client is connected to.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket
    }

    /// Check if a key exists in NestGate storage.
    pub fn exists(&self, key: &str) -> Result<bool, String> {
        let escaped = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{escaped}","family_id":"{family}"}},"id":1}}"#,
        );
        let resp = rpc(&self.socket, &req)?;
        Ok(resp.contains("true"))
    }

    /// Store a JSON value by key.
    pub fn store(&self, key: &str, value: &str) -> Result<StoreResult, String> {
        let escaped_key = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{escaped_key}","value":{value},"family_id":"{family}"}},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// Store a binary blob (base64-encoded) by key.
    pub fn store_blob(&self, key: &str, data: &[u8]) -> Result<StoreResult, String> {
        let escaped_key = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let encoded = base64_encode(data);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store_blob","params":{{"key":"{escaped_key}","blob":"{encoded}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// Retrieve a JSON value by key.
    pub fn retrieve(&self, key: &str) -> Result<RetrieveResult, String> {
        let escaped_key = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        let value = extract_result_value(&raw);
        Ok(RetrieveResult { value, raw })
    }

    /// Retrieve a binary blob by key.
    pub fn retrieve_blob(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let escaped_key = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve_blob","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        if raw.contains("\"error\"") {
            return Ok(None);
        }
        let b64 = extract_result_string(&raw, "blob")
            .or_else(|| extract_result_string(&raw, "data"));
        Ok(b64.map(|encoded| base64_decode(&encoded)))
    }

    /// Delete a key from storage.
    pub fn delete(&self, key: &str) -> Result<StoreResult, String> {
        let escaped_key = escape_json_str(key);
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.delete","params":{{"key":"{escaped_key}","family_id":"{family}"}},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        Ok(StoreResult {
            ok: !raw.contains("\"error\""),
            raw,
        })
    }

    /// List keys with an optional prefix filter.
    pub fn list(&self, prefix: Option<&str>) -> Result<ListResult, String> {
        let family = escape_json_str(&self.family_id);
        let params = prefix.map_or_else(
            || format!(r#"{{"family_id":"{family}"}}"#),
            |p| {
                let escaped = escape_json_str(p);
                format!(r#"{{"family_id":"{family}","prefix":"{escaped}"}}"#)
            },
        );
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.list","params":{params},"id":1}}"#,
        );
        let raw = rpc(&self.socket, &req)?;
        let keys = extract_string_array(&raw, "keys")
            .or_else(|| extract_result_array(&raw))
            .unwrap_or_default();
        Ok(ListResult { keys, raw })
    }

    /// Get storage statistics.
    pub fn stats(&self) -> Result<String, String> {
        let family = escape_json_str(&self.family_id);
        let req = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.stats","params":{{"family_id":"{family}"}},"id":1}}"#,
        );
        rpc(&self.socket, &req)
    }

    /// Ingest a local file into NestGate blob storage.
    ///
    /// Reads the file and stores it as a base64-encoded blob under the
    /// given key. For large files, this loads the entire file into memory.
    pub fn ingest_file(&self, key: &str, path: &Path) -> Result<StoreResult, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        self.store_blob(key, &data)
    }

    /// Store dataset metadata as structured JSON.
    ///
    /// Stores provenance (source, accession, file count, total bytes)
    /// under the key `meta:<dataset>`.
    pub fn store_dataset_metadata(
        &self,
        dataset: &str,
        source: &str,
        file_count: usize,
        total_bytes: u64,
    ) -> Result<StoreResult, String> {
        let meta_key = format!("meta:{dataset}");
        let value = format!(
            r#"{{"dataset":"{dataset}","source":"{source}","file_count":{file_count},"total_bytes":{total_bytes},"stored_at":"{}"}}"#,
            chrono_lite_now(),
        );
        self.store(&meta_key, &value)
    }
}

// ── Socket discovery ────────────────────────────────────────────────

/// Discover the NestGate Unix socket.
///
/// Public so `data.rs` can share the same discovery logic.
#[must_use]
pub fn discover_nestgate_socket() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("NESTGATE_SOCKET") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join("biomeos/nestgate-default.sock");
        if p.exists() {
            return Some(p);
        }
    }
    let fallback = std::env::temp_dir().join("nestgate-default.sock");
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

// ── JSON-RPC transport ──────────────────────────────────────────────

fn rpc(socket: &Path, request: &str) -> Result<String, String> {
    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| format!("invalid NestGate socket: {e}"))?;
    let stream =
        UnixStream::connect_addr(&addr).map_err(|e| format!("NestGate connect: {e}"))?;
    stream
        .set_read_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;
    stream
        .set_write_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("NestGate returned empty response".to_string());
    }
    Ok(line)
}

// ── Minimal JSON helpers ────────────────────────────────────────────

fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn extract_result_value(json: &str) -> Option<String> {
    let start = json.find("\"result\"")?;
    let after = &json[start + 8..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if rest.starts_with('{') || rest.starts_with('[') || rest.starts_with('"') {
        let end = find_value_end(rest)?;
        Some(rest[..end].to_string())
    } else {
        let end = rest.find([',', '}'])?;
        Some(rest[..end].trim().to_string())
    }
}

fn extract_result_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after = &json[start + pattern.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let inner = rest.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn extract_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after = &json[start + pattern.len()..];
    let arr_start = after.find('[')?;
    let arr_end = after[arr_start..].find(']')?;
    let content = &after[arr_start + 1..arr_start + arr_end];
    Some(parse_string_list(content))
}

fn extract_result_array(json: &str) -> Option<Vec<String>> {
    let start = json.find("\"result\"")?;
    let after = &json[start + 8..];
    let arr_start = after.find('[')?;
    let arr_end = after[arr_start..].find(']')?;
    let content = &after[arr_start + 1..arr_start + arr_end];
    Some(parse_string_list(content))
}

fn parse_string_list(content: &str) -> Vec<String> {
    content
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim().trim_matches('"');
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect()
}

fn find_value_end(s: &str) -> Option<usize> {
    let first = s.as_bytes().first()?;
    match first {
        b'"' => {
            let end = s[1..].find('"')? + 2;
            Some(end)
        }
        b'{' | b'[' => {
            let (open, close) = if *first == b'{' { (b'{', b'}') } else { (b'[', b']') };
            let mut depth = 0;
            for (i, &ch) in s.as_bytes().iter().enumerate() {
                if ch == open {
                    depth += 1;
                } else if ch == close {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
            }
            None
        }
        _ => s.find([',', '}']).or(Some(s.len())),
    }
}

// ── Base64 encode/decode (no external dependency) ───────────────────

const B64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        result.push(B64_CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

#[allow(clippy::cast_possible_truncation)]
fn base64_decode(encoded: &str) -> Vec<u8> {
    let clean: Vec<u8> = encoded.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut result = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let vals: Vec<u8> = chunk
            .iter()
            .map(|&b| b64_val(b))
            .collect();
        let n = (u32::from(vals[0]) << 18)
            | (u32::from(vals[1]) << 12)
            | (u32::from(vals[2]) << 6)
            | u32::from(vals[3]);
        result.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            result.push((n >> 8) as u8);
        }
        if chunk[3] != b'=' {
            result.push(n as u8);
        }
    }
    result
}

const fn b64_val(ch: u8) -> u8 {
    match ch {
        b'A'..=b'Z' => ch - b'A',
        b'a'..=b'z' => ch - b'a' + 26,
        b'0'..=b'9' => ch - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => 0,
    }
}

/// Minimal ISO 8601 timestamp without chrono dependency.
fn chrono_lite_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{s:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let ydays = if is_leap(year) { 366 } else { 365 };
        if days < ydays {
            break;
        }
        days -= ydays;
        year += 1;
    }
    let leap = is_leap(year);
    let mdays: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 0;
    for (i, &md) in mdays.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    (year, month, days + 1)
}

const fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        let data = b"Hello, NestGate!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_decode(""), Vec::<u8>::new());
    }

    #[test]
    fn base64_padding_1() {
        let encoded = base64_encode(b"A");
        assert!(encoded.ends_with("=="));
        assert_eq!(base64_decode(&encoded), b"A");
    }

    #[test]
    fn base64_padding_2() {
        let encoded = base64_encode(b"AB");
        assert!(encoded.ends_with('='));
        assert_eq!(base64_decode(&encoded), b"AB");
    }

    #[test]
    fn base64_no_padding() {
        let encoded = base64_encode(b"ABC");
        assert!(!encoded.contains('='));
        assert_eq!(base64_decode(&encoded), b"ABC");
    }

    #[test]
    fn base64_binary_data() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn escape_json_handles_special_chars() {
        assert_eq!(escape_json_str(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape_json_str("simple"), "simple");
    }

    #[test]
    fn chrono_lite_produces_iso8601() {
        let ts = chrono_lite_now();
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        assert!(ts.len() >= 19);
    }

    #[test]
    fn extract_result_value_string() {
        let json = r#"{"jsonrpc":"2.0","result":"hello","id":1}"#;
        assert_eq!(extract_result_value(json), Some("\"hello\"".to_string()));
    }

    #[test]
    fn extract_result_value_object() {
        let json = r#"{"jsonrpc":"2.0","result":{"key":"val"},"id":1}"#;
        let val = extract_result_value(json).unwrap();
        assert!(val.contains("key"));
    }

    #[test]
    fn extract_result_value_bool() {
        let json = r#"{"jsonrpc":"2.0","result":true,"id":1}"#;
        assert_eq!(extract_result_value(json), Some("true".to_string()));
    }

    #[test]
    fn nest_client_with_family() {
        let client = NestClient::new(PathBuf::from("/tmp/test.sock"))
            .with_family("wetspring");
        assert_eq!(client.family_id, "wetspring");
    }

    #[test]
    fn socket_discovery_does_not_panic() {
        let _ = discover_nestgate_socket();
    }

    #[test]
    fn days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2026-02-28 = day 20,512 from epoch
        let (y, m, d) = days_to_ymd(20_512);
        assert_eq!(y, 2026);
        assert_eq!(m, 2);
        assert_eq!(d, 28);
    }
}
