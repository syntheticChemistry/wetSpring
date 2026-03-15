// SPDX-License-Identifier: AGPL-3.0-or-later

//! Songbird mesh discovery — Unix socket RPC and JSON parsing.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};

const SONGBIRD_TIMEOUT: Duration = Duration::from_secs(3);

/// Discover the Songbird Unix socket path.
///
/// 1. `SONGBIRD_SOCKET` env var
/// 2. `$XDG_RUNTIME_DIR/biomeos/songbird-default.sock`
/// 3. `<temp_dir>/songbird-default.sock`
#[must_use]
pub fn discover_songbird_socket() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SONGBIRD_SOCKET") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join("biomeos/songbird-default.sock");
        if p.exists() {
            return Some(p);
        }
    }
    let fallback = std::env::temp_dir().join("songbird-default.sock");
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

/// Query Songbird for registered primals that expose compute capabilities.
///
/// Sends `discovery.discover("compute")` and parses the response into
/// `Substrate` entries with `SubstrateOrigin::Mesh`.
pub fn query_songbird_substrates(socket: &Path) -> Result<Vec<Substrate>, crate::error::SongbirdError> {
    let request = r#"{"jsonrpc":"2.0","method":"discovery.discover","params":{"capability":"compute"},"id":1}"#;
    let response = songbird_rpc(socket, request)?;
    parse_songbird_substrates(&response)
}

/// Send a JSON-RPC request to Songbird and read the response.
fn songbird_rpc(socket: &Path, request: &str) -> Result<String, crate::error::SongbirdError> {
    use crate::error::SongbirdError;

    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| SongbirdError::InvalidSocket(e.to_string()))?;
    let stream =
        UnixStream::connect_addr(&addr).map_err(|e| SongbirdError::Connect(e.to_string()))?;
    stream
        .set_read_timeout(Some(SONGBIRD_TIMEOUT))
        .map_err(|e| SongbirdError::Timeout(e.to_string()))?;
    stream
        .set_write_timeout(Some(SONGBIRD_TIMEOUT))
        .map_err(|e| SongbirdError::Timeout(e.to_string()))?;

    let mut writer = io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| SongbirdError::Write(e.to_string()))?;
    writer
        .write_all(b"\n")
        .map_err(|e| SongbirdError::Write(e.to_string()))?;
    writer
        .flush()
        .map_err(|e| SongbirdError::Flush(e.to_string()))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| SongbirdError::Read(e.to_string()))?;

    if line.is_empty() {
        return Err(SongbirdError::EmptyResponse);
    }
    Ok(line)
}

/// Parse Songbird discovery response into remote substrates.
///
/// Expected format: `{"result":[{"name":"strandgate","capabilities":["compute","gpu","f64"],...}]}`
/// Gracefully handles missing fields and unknown capability strings.
pub fn parse_songbird_substrates(response: &str) -> Result<Vec<Substrate>, crate::error::SongbirdError> {
    let mut substrates = Vec::new();

    if response.contains("\"error\"") {
        return Err(crate::error::SongbirdError::ErrorResponse);
    }

    let Some(result_start) = response.find("\"result\"") else {
        return Ok(substrates);
    };
    let after = &response[result_start..];
    let Some(arr_start) = after.find('[') else {
        return Ok(substrates);
    };
    let Some(arr_end) = after.rfind(']') else {
        return Ok(substrates);
    };
    if arr_start >= arr_end {
        return Ok(substrates);
    }

    let arr_content = &after[arr_start + 1..arr_end];

    for entry in split_json_objects(arr_content) {
        if let Some(sub) = parse_single_service(entry) {
            substrates.push(sub);
        }
    }

    Ok(substrates)
}

/// Split a JSON array body into individual object strings.
pub fn split_json_objects(content: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, ch) in content.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        objects.push(&content[s..=i]);
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

/// Parse a single Songbird service entry into a Substrate.
pub fn parse_single_service(json: &str) -> Option<Substrate> {
    let gate_name = extract_json_string(json, "name")?;

    let caps_str = extract_json_array_strings(json, "capabilities");
    let mut capabilities = vec![Capability::F32Compute];
    let mut kind = SubstrateKind::Cpu;
    let mut has_f64 = false;

    for cap in &caps_str {
        match cap.as_str() {
            "f64" | "f64_compute" => {
                capabilities.push(Capability::F64Compute);
                has_f64 = true;
            }
            "gpu" | "shader" | "shader_dispatch" => {
                kind = SubstrateKind::Gpu;
                capabilities.push(Capability::ShaderDispatch);
            }
            "npu" | "quant" | "quantized_inference" => {
                kind = SubstrateKind::Npu;
                capabilities.push(Capability::QuantizedInference { bits: 8 });
            }
            "reduce" | "scalar_reduce" => {
                capabilities.push(Capability::ScalarReduce);
            }
            "simd" => {
                capabilities.push(Capability::SimdVector);
            }
            _ => {}
        }
    }

    Some(Substrate {
        kind,
        identity: Identity::named(&gate_name),
        properties: Properties {
            has_f64,
            ..Properties::default()
        },
        capabilities,
        origin: SubstrateOrigin::Mesh { gate_name },
    })
}

/// Extract a JSON string value by key (minimal parser — no serde dependency).
pub fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    let inner = trimmed.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

/// Extract a JSON array of strings by key (minimal parser — no serde dependency).
pub fn extract_json_array_strings(json: &str, key: &str) -> Vec<String> {
    let pattern = format!("\"{key}\"");
    let Some(start) = json.find(&pattern) else {
        return Vec::new();
    };
    let after_key = &json[start + pattern.len()..];
    let Some(arr_start) = after_key.find('[') else {
        return Vec::new();
    };
    let Some(arr_end) = after_key[arr_start..].find(']') else {
        return Vec::new();
    };
    let arr_content = &after_key[arr_start + 1..arr_start + arr_end];

    arr_content
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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_string_basic() {
        let json = r#"{"name": "strandgate", "version": "1.0"}"#;
        assert_eq!(extract_json_string(json, "name").unwrap(), "strandgate");
        assert_eq!(extract_json_string(json, "version").unwrap(), "1.0");
    }

    #[test]
    fn extract_json_string_missing_key() {
        let json = r#"{"name": "strandgate"}"#;
        assert!(extract_json_string(json, "missing").is_none());
    }

    #[test]
    fn extract_json_array_strings_basic() {
        let json = r#"{"capabilities": ["compute", "gpu", "f64"]}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert_eq!(caps, vec!["compute", "gpu", "f64"]);
    }

    #[test]
    fn extract_json_array_strings_empty() {
        let json = r#"{"capabilities": []}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert!(caps.is_empty());
    }

    #[test]
    fn extract_json_array_strings_missing_key() {
        let json = r#"{"name": "test"}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert!(caps.is_empty());
    }

    #[test]
    fn split_json_objects_basic() {
        let content = r#"{"a": 1}, {"b": 2}"#;
        let objects = split_json_objects(content);
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn split_json_objects_nested() {
        let content = r#"{"a": {"inner": 1}}, {"b": 2}"#;
        let objects = split_json_objects(content);
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn split_json_objects_empty() {
        assert!(split_json_objects("").is_empty());
    }

    #[test]
    fn parse_songbird_substrates_empty_result() {
        let response = r#"{"jsonrpc":"2.0","result":[],"id":1}"#;
        let subs = parse_songbird_substrates(response).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn parse_songbird_substrates_error_response() {
        let response = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"fail"},"id":1}"#;
        assert!(parse_songbird_substrates(response).is_err());
    }

    #[test]
    fn parse_songbird_substrates_single_gpu_node() {
        let response = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["compute","gpu","f64"]}],"id":1}"#;
        let subs = parse_songbird_substrates(response).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].kind, SubstrateKind::Gpu);
        assert!(subs[0].properties.has_f64);
    }

    #[test]
    fn parse_single_service_cpu() {
        let json = r#"{"name": "local-cpu", "capabilities": ["compute", "f64"]}"#;
        let sub = parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Cpu);
        assert!(sub.properties.has_f64);
    }

    #[test]
    fn parse_single_service_npu() {
        let json = r#"{"name": "akida-node", "capabilities": ["compute", "npu"]}"#;
        let sub = parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Npu);
    }

    #[test]
    fn parse_single_service_missing_name() {
        let json = r#"{"capabilities": ["compute"]}"#;
        assert!(parse_single_service(json).is_none());
    }

    #[test]
    fn parse_songbird_substrates_no_result_key() {
        let response = r#"{"jsonrpc":"2.0","id":1}"#;
        let subs = parse_songbird_substrates(response).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn discover_songbird_socket_returns_none_without_env() {
        assert!(discover_songbird_socket().is_none());
    }
}
