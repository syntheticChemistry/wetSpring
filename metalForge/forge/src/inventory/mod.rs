// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware inventory — assemble probed substrates.
//!
//! Runs GPU probing (via wgpu/barracuda), CPU probing (procfs), and
//! NPU probing (local device nodes). Returns every substrate discovered
//! on this machine. If it's not in the inventory, it doesn't exist to us.
//!
//! When Songbird is running (Tower atomic), `discover_with_tower()` extends
//! the local inventory with substrates reported by remote NUCLEUS gates.

mod output;
mod songbird;

use crate::probe;
use crate::substrate::Substrate;

pub use output::{print_inventory, write_inventory};
pub use songbird::discover_songbird_socket;

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
        match songbird::query_songbird_substrates(&socket) {
            Ok(remote) => {
                let n = remote.len();
                substrates.extend(remote);
                if n > 0 {
                    tracing::info!(count = n, "Songbird remote substrates discovered");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Songbird query failed, local-only");
            }
        }
    }

    substrates
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use crate::substrate::{
        Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
    };

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
            assert!(gpu.has(&Capability::ShaderDispatch));
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
        let subs = songbird::parse_songbird_substrates(resp).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn parse_songbird_single_gpu_gate() {
        let resp = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["compute","gpu","f64"]}],"id":1}"#;
        let subs = songbird::parse_songbird_substrates(resp).unwrap();
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
        let subs = songbird::parse_songbird_substrates(resp).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].kind, SubstrateKind::Gpu);
        assert_eq!(subs[1].kind, SubstrateKind::Npu);
    }

    #[test]
    fn parse_songbird_error_response() {
        let resp = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"no service"},"id":1}"#;
        let result = songbird::parse_songbird_substrates(resp);
        assert!(result.is_err());
    }

    #[test]
    fn extract_json_string_basic() {
        let json = r#"{"name":"eastgate","type":"tower"}"#;
        assert_eq!(
            songbird::extract_json_string(json, "name"),
            Some("eastgate".to_string())
        );
        assert_eq!(
            songbird::extract_json_string(json, "type"),
            Some("tower".to_string())
        );
        assert!(songbird::extract_json_string(json, "missing").is_none());
    }

    #[test]
    fn extract_json_array_strings_basic() {
        let json = r#"{"capabilities":["gpu","f64","reduce"]}"#;
        let caps = songbird::extract_json_array_strings(json, "capabilities");
        assert_eq!(caps, vec!["gpu", "f64", "reduce"]);
    }

    #[test]
    fn extract_json_array_strings_missing() {
        let json = r#"{"name":"test"}"#;
        let caps = songbird::extract_json_array_strings(json, "capabilities");
        assert!(caps.is_empty());
    }

    #[test]
    fn split_json_objects_basic() {
        let content = r#"{"a":1},{"b":2}"#;
        let objects = songbird::split_json_objects(content);
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn split_json_objects_nested() {
        let content = r#"{"a":{"b":1}},{"c":2}"#;
        let objects = songbird::split_json_objects(content);
        assert_eq!(objects.len(), 2);
        assert!(objects[0].contains("\"b\""));
    }

    #[test]
    fn split_json_objects_empty() {
        let objects = songbird::split_json_objects("");
        assert!(objects.is_empty());
    }

    #[test]
    fn parse_single_service_gpu() {
        let json = r#"{"name":"strandgate","capabilities":["gpu","f64","reduce"]}"#;
        let sub = songbird::parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Gpu);
        assert!(sub.properties.has_f64);
        assert!(sub.has(&Capability::ShaderDispatch));
        assert!(sub.has(&Capability::F64Compute));
        assert!(sub.has(&Capability::ScalarReduce));
    }

    #[test]
    fn parse_single_service_npu() {
        let json = r#"{"name":"edgegate","capabilities":["npu","quant"]}"#;
        let sub = songbird::parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Npu);
        assert!(!sub.properties.has_f64);
    }

    #[test]
    fn parse_single_service_cpu_simd() {
        let json = r#"{"name":"farmgate","capabilities":["simd","f64"]}"#;
        let sub = songbird::parse_single_service(json).unwrap();
        assert_eq!(sub.kind, SubstrateKind::Cpu);
        assert!(sub.properties.has_f64);
        assert!(sub.has(&Capability::SimdVector));
    }

    #[test]
    fn parse_single_service_missing_name() {
        let json = r#"{"capabilities":["gpu"]}"#;
        assert!(songbird::parse_single_service(json).is_none());
    }

    #[test]
    fn parse_songbird_no_result() {
        let resp = r#"{"jsonrpc":"2.0","id":1}"#;
        let subs = songbird::parse_songbird_substrates(resp).unwrap();
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
        let caps = songbird::extract_json_array_strings(json, "capabilities");
        assert_eq!(caps, vec!["gpu", "f64"]);
    }
}
