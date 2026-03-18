// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware probing — GPU via wgpu/barracuda, NPU and CPU locally.
//!
//! GPU discovery leans on wgpu (which toadstool/barracuda uses). We get
//! adapter name, device type, driver, backend, and feature flags (`SHADER_F64`)
//! directly from the Vulkan/wgpu layer — no sysfs reimplementation needed.
//!
//! NPU discovery is local (probing `/dev/akida*`). This is evolution that
//! `ToadStool` can absorb once NPU substrate support matures upstream.
//!
//! CPU discovery reads `/proc/cpuinfo` and `/proc/meminfo`.

use crate::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};
use std::fs;

/// Probe all GPU adapters via wgpu.
///
/// Uses the same wgpu instance/backend configuration that barracuda uses.
/// Each adapter becomes a substrate with capabilities derived from its
/// feature flags (`SHADER_F64` → [`Capability::F64Compute`], etc.).
#[must_use]
pub fn probe_gpus() -> Vec<Substrate> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let mut gpus = Vec::new();

    for (idx, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        let features = adapter.features();

        if info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }

        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let mut capabilities = vec![Capability::F32Compute, Capability::ShaderDispatch];
        if has_f64 {
            capabilities.push(Capability::F64Compute);
            capabilities.push(Capability::ScalarReduce);
        }
        if has_timestamps {
            capabilities.push(Capability::TimestampQuery);
        }

        gpus.push(Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: info.name.clone(),
                driver: Some(format!("{} ({})", info.driver, info.driver_info)),
                backend: Some(format!("{:?}", info.backend)),
                adapter_index: Some(idx),
                device_node: None,
                pci_id: None,
            },
            properties: Properties {
                has_f64,
                has_timestamps,
                ..Properties::default()
            },
            capabilities,
            origin: SubstrateOrigin::Local,
        });
    }

    gpus
}

/// Probe CPU capabilities via platform-native sources.
///
/// On Linux, reads `/proc/cpuinfo` and `/proc/meminfo`. On other platforms,
/// returns a baseline CPU substrate with default capabilities.
#[must_use]
pub fn probe_cpu() -> Substrate {
    let probe = probe_cpu_platform();
    let name = probe.model.unwrap_or_else(|| String::from("Unknown CPU"));

    let mut capabilities = vec![Capability::F64Compute, Capability::F32Compute];
    if probe.has_avx2 {
        capabilities.push(Capability::SimdVector);
    }

    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named(name),
        properties: Properties {
            memory_bytes: probe.memory_bytes,
            core_count: probe.cores,
            thread_count: probe.threads,
            cache_kb: probe.cache_kb,
            ..Properties::default()
        },
        capabilities,
        origin: SubstrateOrigin::Local,
    }
}

/// Raw CPU probe results prior to substrate construction.
struct CpuProbeResult {
    model: Option<String>,
    cores: Option<u32>,
    threads: Option<u32>,
    cache_kb: Option<u32>,
    has_avx2: bool,
    memory_bytes: Option<u64>,
}

/// Platform-specific CPU probing (Linux: reads `/proc/cpuinfo` and `/proc/meminfo`).
#[cfg(target_os = "linux")]
fn probe_cpu_platform() -> CpuProbeResult {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let (model, cores, threads, cache_kb, has_avx2) = parse_cpuinfo(&cpuinfo);
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let memory_bytes = parse_meminfo(&meminfo);
    CpuProbeResult {
        model,
        cores,
        threads,
        cache_kb,
        has_avx2,
        memory_bytes,
    }
}

/// Fallback CPU probing for non-Linux platforms.
#[cfg(not(target_os = "linux"))]
fn probe_cpu_platform() -> CpuProbeResult {
    CpuProbeResult {
        model: None,
        cores: None,
        threads: None,
        cache_kb: None,
        has_avx2: false,
        memory_bytes: None,
    }
}

/// NPU environment variable name, sourced from the niche registry.
const NPU_ENV_VAR: &str = wetspring_barracuda::niche::NPU_DEVICE_ENV_VAR;

/// Probe for NPU devices via capability-based discovery.
///
/// Discovery order:
/// 1. `WETSPRING_NPU_DEVICE` env var (explicit override — any NPU vendor)
/// 2. Scan `/dev/akida*` device nodes (`BrainChip` AKD series)
///
/// Returns all discovered NPU substrates with their capabilities.
#[must_use]
pub fn probe_npus() -> Vec<Substrate> {
    let mut npus = Vec::new();

    let candidates = discover_npu_device_nodes();
    for device_path in candidates {
        let akida_path = std::path::Path::new(&device_path);
        if akida_path.exists() {
            npus.push(Substrate {
                kind: SubstrateKind::Npu,
                identity: Identity {
                    name: npu_name_from_device(&device_path),
                    device_node: Some(device_path),
                    ..Identity::named("NPU")
                },
                properties: Properties::default(),
                capabilities: vec![
                    Capability::F32Compute,
                    Capability::QuantizedInference { bits: 8 },
                    Capability::QuantizedInference { bits: 4 },
                    Capability::BatchInference { max_batch: 8 },
                    Capability::WeightMutation,
                ],
                origin: SubstrateOrigin::Local,
            });
        }
    }

    npus
}

/// Discover NPU device nodes via env var or filesystem scan.
fn discover_npu_device_nodes() -> Vec<String> {
    if let Ok(explicit) = std::env::var(NPU_ENV_VAR) {
        return vec![explicit];
    }

    let mut nodes = Vec::new();
    if let Ok(entries) = fs::read_dir("/dev") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("akida") {
                    nodes.push(entry.path().to_string_lossy().into_owned());
                }
            }
        }
    }

    if nodes.is_empty() {
        nodes.push(wetspring_barracuda::niche::discover_npu_device());
    }
    nodes
}

/// Derive NPU name from device node path.
fn npu_name_from_device(path: &str) -> String {
    if path.contains("akida") {
        String::from("BrainChip AKD1000")
    } else {
        format!("NPU ({path})")
    }
}

/// Parse `/proc/cpuinfo` content into (model, cores, threads, `cache_kb`, `has_avx2`).
fn parse_cpuinfo(content: &str) -> (Option<String>, Option<u32>, Option<u32>, Option<u32>, bool) {
    let mut model = None;
    let mut cores = None;
    let mut siblings = None;
    let mut cache_kb = None;
    let mut has_avx2 = false;

    for line in content.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "model name" if model.is_none() => model = Some(val.to_string()),
                "cpu cores" if cores.is_none() => cores = val.parse().ok(),
                "siblings" if siblings.is_none() => siblings = val.parse().ok(),
                "cache size" if cache_kb.is_none() => {
                    cache_kb = val.trim_end_matches(" KB").parse().ok();
                }
                "flags" if !has_avx2 => {
                    has_avx2 = val.split_whitespace().any(|f| f == "avx2");
                }
                _ => {}
            }
        }
    }

    (model, cores, siblings, cache_kb, has_avx2)
}

/// Parse `/proc/meminfo` content into total memory bytes.
fn parse_meminfo(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_discovered() {
        let cpu = probe_cpu();
        assert_eq!(cpu.kind, SubstrateKind::Cpu);
        assert!(cpu.has(&Capability::F64Compute));
        assert!(!cpu.identity.name.is_empty());
    }

    #[test]
    fn gpu_probe_uses_wgpu() {
        let gpus = probe_gpus();
        for gpu in &gpus {
            assert_eq!(gpu.kind, SubstrateKind::Gpu);
            assert!(gpu.has(&Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
            assert!(gpu.identity.driver.is_some());
        }
    }

    #[test]
    fn parse_cpuinfo_extracts_model() {
        let content = "model name\t: Intel(R) Core(TM) i9-12900K\ncpu cores\t: 8\nsiblings\t: 24\ncache size\t: 30720 KB\nflags\t\t: fpu vme de sse sse2 avx avx2\n";
        let (model, cores, threads, cache, avx2) = parse_cpuinfo(content);
        assert_eq!(model.unwrap(), "Intel(R) Core(TM) i9-12900K");
        assert_eq!(cores.unwrap(), 8);
        assert_eq!(threads.unwrap(), 24);
        assert_eq!(cache.unwrap(), 30720);
        assert!(avx2);
    }

    #[test]
    fn parse_cpuinfo_empty() {
        let (model, cores, threads, cache, avx2) = parse_cpuinfo("");
        assert!(model.is_none());
        assert!(cores.is_none());
        assert!(threads.is_none());
        assert!(cache.is_none());
        assert!(!avx2);
    }

    #[test]
    fn parse_meminfo_extracts_total() {
        let content = "MemTotal:       32749772 kB\nMemFree:        15000000 kB\n";
        let bytes = parse_meminfo(content).unwrap();
        assert_eq!(bytes, 32_749_772 * 1024);
    }

    #[test]
    fn parse_meminfo_empty() {
        assert!(parse_meminfo("").is_none());
    }

    #[test]
    fn parse_cpuinfo_no_avx2() {
        let content = "flags\t\t: fpu vme de sse sse2 avx\n";
        let (_, _, _, _, avx2) = parse_cpuinfo(content);
        assert!(!avx2);
    }
}
