// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware inventory — assemble probed substrates.
//!
//! Runs GPU probing (via wgpu/barracuda), CPU probing (procfs), and
//! NPU probing (local device nodes). Returns every substrate discovered
//! on this machine. If it's not in the inventory, it doesn't exist to us.
//!
//! When Songbird is running (Tower atomic), `discover_with_tower()` extends
//! the local inventory with substrates reported by remote NUCLEUS gates.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::probe;
use crate::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};

const SONGBIRD_TIMEOUT: Duration = Duration::from_secs(3);

/// Discover all compute substrates on this machine.
///
/// Always returns at least one entry (CPU). GPUs come from wgpu adapter
/// enumeration. NPUs come from device node probing.
#[must_use]
pub fn discover() -> Vec<Substrate> {
    let mut substrates = Vec::new();
    substrates.push(probe::probe_cpu());
    substrates.extend(probe::probe_gpus());
    substrates.extend(probe::probe_npus());
    substrates
}

/// Discover local substrates, then extend with remote substrates from Songbird.
///
/// Falls back to local-only discovery if Songbird is not running. Remote
/// substrates are tagged with `SubstrateOrigin::Mesh` so dispatch can
/// account for network latency.
#[must_use]
pub fn discover_with_tower() -> Vec<Substrate> {
    let mut substrates = discover();

    if let Some(socket) = discover_songbird_socket() {
        match query_songbird_substrates(&socket) {
            Ok(remote) => {
                let n = remote.len();
                substrates.extend(remote);
                if n > 0 {
                    eprintln!("[forge] Songbird: {n} remote substrate(s) discovered");
                }
            }
            Err(e) => {
                eprintln!("[forge] Songbird query failed (local-only): {e}");
            }
        }
    }

    substrates
}

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
fn query_songbird_substrates(socket: &Path) -> Result<Vec<Substrate>, String> {
    let request = r#"{"jsonrpc":"2.0","method":"discovery.discover","params":{"capability":"compute"},"id":1}"#;
    let response = songbird_rpc(socket, request)?;
    parse_songbird_substrates(&response)
}

/// Send a JSON-RPC request to Songbird and read the response.
fn songbird_rpc(socket: &Path, request: &str) -> Result<String, String> {
    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| format!("invalid Songbird socket: {e}"))?;
    let stream = UnixStream::connect_addr(&addr).map_err(|e| format!("Songbird connect: {e}"))?;
    stream
        .set_read_timeout(Some(SONGBIRD_TIMEOUT))
        .map_err(|e| format!("set timeout: {e}"))?;
    stream
        .set_write_timeout(Some(SONGBIRD_TIMEOUT))
        .map_err(|e| format!("set timeout: {e}"))?;

    let mut writer = io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("Songbird returned empty response".to_string());
    }
    Ok(line)
}

/// Parse Songbird discovery response into remote substrates.
///
/// Expected format: `{"result":[{"name":"strandgate","capabilities":["compute","gpu","f64"],...}]}`
/// Gracefully handles missing fields and unknown capability strings.
fn parse_songbird_substrates(response: &str) -> Result<Vec<Substrate>, String> {
    let mut substrates = Vec::new();

    if response.contains("\"error\"") {
        return Err("Songbird returned error".to_string());
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
fn split_json_objects(content: &str) -> Vec<&str> {
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
fn parse_single_service(json: &str) -> Option<Substrate> {
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
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    let inner = trimmed.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

/// Extract a JSON array of strings by key.
fn extract_json_array_strings(json: &str, key: &str) -> Vec<String> {
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

/// Write a human-readable inventory to the given writer.
///
/// # Errors
///
/// Returns an I/O error if writing fails.
pub fn write_inventory(substrates: &[Substrate], w: &mut impl Write) -> io::Result<()> {
    writeln!(w, "┌──────────────────────────────────────────────┐")?;
    writeln!(w, "│  wetSpring Forge — Hardware Inventory         │")?;
    writeln!(w, "├──────────────────────────────────────────────┤")?;

    for (i, s) in substrates.iter().enumerate() {
        let origin_tag = match &s.origin {
            SubstrateOrigin::Local => "",
            SubstrateOrigin::Mesh { gate_name } => gate_name.as_str(),
        };
        if origin_tag.is_empty() {
            writeln!(w, "│ {i}: {s}")?;
        } else {
            writeln!(w, "│ {i}: {s} [mesh: {origin_tag}]")?;
        }

        if let Some(ref backend) = s.identity.backend {
            writeln!(w, "│    backend: {backend}")?;
        }
        if let Some(idx) = s.identity.adapter_index {
            writeln!(w, "│    adapter: {idx}")?;
        }
        if let Some(ref node) = s.identity.device_node {
            writeln!(w, "│    device:  {node}")?;
        }
        if let Some(cores) = s.properties.core_count {
            let threads = s.properties.thread_count.unwrap_or(cores);
            writeln!(w, "│    cores:   {cores} ({threads} threads)")?;
        }
        if s.properties.has_f64 {
            writeln!(w, "│    SHADER_F64: YES")?;
        }

        writeln!(w, "│    caps:    {}", s.capability_summary())?;
        writeln!(w, "│")?;
    }

    let local_count = substrates
        .iter()
        .filter(|s| s.origin == SubstrateOrigin::Local)
        .count();
    let mesh_count = substrates.len() - local_count;
    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    writeln!(
        w,
        "│  Total: {local_count} local, {mesh_count} mesh, {gpu_count} GPU(s), {npu_count} NPU(s)"
    )?;
    writeln!(w, "└──────────────────────────────────────────────┘")?;
    Ok(())
}

/// Print a human-readable inventory to stdout.
///
/// Convenience wrapper around [`write_inventory`] for CLI usage.
pub fn print_inventory(substrates: &[Substrate]) {
    let mut stdout = io::stdout().lock();
    let _ = write_inventory(substrates, &mut stdout);
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_always_has_cpu() {
        let subs = discover();
        assert!(subs.iter().any(|s| s.kind == SubstrateKind::Cpu));
    }

    #[test]
    fn discover_finds_gpus_via_wgpu() {
        let subs = discover();
        let gpus: Vec<_> = subs
            .iter()
            .filter(|s| s.kind == SubstrateKind::Gpu)
            .collect();
        for gpu in &gpus {
            assert!(gpu.has(&crate::substrate::Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
        }
    }

    #[test]
    fn inventory_has_one_cpu() {
        let subs = discover();
        let cpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Cpu).count();
        assert_eq!(cpu_count, 1);
    }

    #[test]
    fn all_local_substrates_have_local_origin() {
        let subs = discover();
        for sub in &subs {
            assert_eq!(sub.origin, SubstrateOrigin::Local);
        }
    }

    #[test]
    fn discover_with_tower_includes_local() {
        let subs = discover_with_tower();
        assert!(subs.iter().any(|s| s.kind == SubstrateKind::Cpu));
        let local_count = subs
            .iter()
            .filter(|s| s.origin == SubstrateOrigin::Local)
            .count();
        assert!(local_count >= 1, "should have at least CPU");
    }

    #[test]
    fn songbird_socket_discovery_does_not_panic() {
        let _ = discover_songbird_socket();
    }

    #[test]
    fn parse_songbird_empty_result() {
        let resp = r#"{"jsonrpc":"2.0","result":[],"id":1}"#;
        let subs = parse_songbird_substrates(resp).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn parse_songbird_single_gpu_gate() {
        let resp = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["compute","gpu","f64"]}],"id":1}"#;
        let subs = parse_songbird_substrates(resp).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].kind, SubstrateKind::Gpu);
        assert!(subs[0].has(&Capability::F64Compute));
        assert!(subs[0].has(&Capability::ShaderDispatch));
        assert_eq!(
            subs[0].origin,
            SubstrateOrigin::Mesh {
                gate_name: "strandgate".to_string()
            }
        );
    }

    #[test]
    fn parse_songbird_multiple_gates() {
        let resp = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["gpu","f64"]},{"name":"biomegate","capabilities":["npu","quant"]}],"id":1}"#;
        let subs = parse_songbird_substrates(resp).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].kind, SubstrateKind::Gpu);
        assert_eq!(subs[1].kind, SubstrateKind::Npu);
    }

    #[test]
    fn parse_songbird_error_response() {
        let resp = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"no service"},"id":1}"#;
        let result = parse_songbird_substrates(resp);
        assert!(result.is_err());
    }

    #[test]
    fn extract_json_string_basic() {
        let json = r#"{"name":"eastgate","type":"tower"}"#;
        assert_eq!(
            extract_json_string(json, "name"),
            Some("eastgate".to_string())
        );
        assert_eq!(extract_json_string(json, "type"), Some("tower".to_string()));
        assert!(extract_json_string(json, "missing").is_none());
    }

    #[test]
    fn extract_json_array_strings_basic() {
        let json = r#"{"capabilities":["gpu","f64","reduce"]}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert_eq!(caps, vec!["gpu", "f64", "reduce"]);
    }

    #[test]
    fn extract_json_array_strings_missing() {
        let json = r#"{"name":"test"}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert!(caps.is_empty());
    }

    #[test]
    fn split_json_objects_basic() {
        let content = r#"{"a":1},{"b":2}"#;
        let objects = split_json_objects(content);
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn split_json_objects_nested() {
        let content = r#"{"a":{"b":1}},{"c":2}"#;
        let objects = split_json_objects(content);
        assert_eq!(objects.len(), 2);
        assert!(objects[0].contains("\"b\""));
    }

    #[test]
    fn split_json_objects_empty() {
        let objects = split_json_objects("");
        assert!(objects.is_empty());
    }

    #[test]
    fn parse_single_service_gpu() {
        let json = r#"{"name":"strandgate","capabilities":["gpu","f64","reduce"]}"#;
        let sub = parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Gpu);
        assert!(sub.properties.has_f64);
        assert!(sub.has(&Capability::ShaderDispatch));
        assert!(sub.has(&Capability::F64Compute));
        assert!(sub.has(&Capability::ScalarReduce));
    }

    #[test]
    fn parse_single_service_npu() {
        let json = r#"{"name":"edgegate","capabilities":["npu","quant"]}"#;
        let sub = parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Npu);
        assert!(!sub.properties.has_f64);
    }

    #[test]
    fn parse_single_service_cpu_simd() {
        let json = r#"{"name":"farmgate","capabilities":["simd","f64"]}"#;
        let sub = parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Cpu);
        assert!(sub.properties.has_f64);
        assert!(sub.has(&Capability::SimdVector));
    }

    #[test]
    fn parse_single_service_missing_name() {
        let json = r#"{"capabilities":["gpu"]}"#;
        assert!(parse_single_service(json).is_none());
    }

    #[test]
    fn parse_songbird_no_result() {
        let resp = r#"{"jsonrpc":"2.0","id":1}"#;
        let subs = parse_songbird_substrates(resp).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn write_inventory_cpu_only() {
        let subs = vec![Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("eastgate-cpu"),
            properties: Properties {
                has_f64: true,
                core_count: Some(16),
                thread_count: Some(24),
                ..Properties::default()
            },
            capabilities: vec![Capability::F32Compute, Capability::F64Compute],
            origin: SubstrateOrigin::Local,
        }];
        let mut buf = Vec::new();
        write_inventory(&subs, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("wetSpring Forge"));
        assert!(output.contains("1 local"));
        assert!(output.contains("0 mesh"));
        assert!(output.contains("cores"));
    }

    #[test]
    fn write_inventory_mixed() {
        let subs = vec![
            Substrate {
                kind: SubstrateKind::Cpu,
                identity: Identity::named("cpu"),
                properties: Properties::default(),
                capabilities: vec![Capability::F32Compute],
                origin: SubstrateOrigin::Local,
            },
            Substrate {
                kind: SubstrateKind::Gpu,
                identity: Identity {
                    name: "gpu".to_string(),
                    driver: None,
                    backend: Some("vulkan".to_string()),
                    adapter_index: Some(0),
                    device_node: None,
                    pci_id: None,
                },
                properties: Properties {
                    has_f64: true,
                    ..Properties::default()
                },
                capabilities: vec![Capability::ShaderDispatch],
                origin: SubstrateOrigin::Local,
            },
            Substrate {
                kind: SubstrateKind::Gpu,
                identity: Identity::named("remote"),
                properties: Properties::default(),
                capabilities: vec![Capability::ShaderDispatch],
                origin: SubstrateOrigin::Mesh {
                    gate_name: "strandgate".to_string(),
                },
            },
        ];
        let mut buf = Vec::new();
        write_inventory(&subs, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("2 local"));
        assert!(output.contains("1 mesh"));
        assert!(output.contains("2 GPU"));
        assert!(output.contains("SHADER_F64: YES"));
        assert!(output.contains("mesh: strandgate"));
        assert!(output.contains("backend: vulkan"));
        assert!(output.contains("adapter: 0"));
    }

    #[test]
    fn write_inventory_with_npu() {
        let subs = vec![
            Substrate {
                kind: SubstrateKind::Cpu,
                identity: Identity::named("cpu"),
                properties: Properties::default(),
                capabilities: vec![],
                origin: SubstrateOrigin::Local,
            },
            Substrate {
                kind: SubstrateKind::Npu,
                identity: Identity {
                    name: "akida".to_string(),
                    driver: None,
                    backend: None,
                    adapter_index: None,
                    device_node: Some("/dev/akida0".to_string()),
                    pci_id: None,
                },
                properties: Properties::default(),
                capabilities: vec![Capability::QuantizedInference { bits: 8 }],
                origin: SubstrateOrigin::Local,
            },
        ];
        let mut buf = Vec::new();
        write_inventory(&subs, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("1 NPU"));
        assert!(output.contains("device:"));
    }

    #[test]
    fn extract_json_array_strings_with_spaces() {
        let json = r#"{"capabilities": [ "gpu" , "f64" ]}"#;
        let caps = extract_json_array_strings(json, "capabilities");
        assert_eq!(caps, vec!["gpu", "f64"]);
    }
}
