// SPDX-License-Identifier: AGPL-3.0-or-later
//! petalTongue-compatible visualization for metalForge hardware inventory,
//! NUCLEUS atomics, and mixed hardware dispatch.
//!
//! Reuses wetSpring barracuda's `DataChannel` / `EcologyScenario` types.
//! No petalTongue crate dependency — JSON schema + IPC only.

use wetspring_barracuda::visualization::{
    DataChannel, EcologyScenario, ScenarioEdge, ScenarioNode, ScientificRange,
};

use crate::dispatch;
use crate::substrate::{Substrate, SubstrateKind};
use crate::workloads;

fn node(id: &str, name: &str, node_type: &str, caps: &[&str]) -> ScenarioNode {
    ScenarioNode {
        id: id.into(),
        name: name.into(),
        node_type: node_type.into(),
        family: "metalforge".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: caps.iter().map(|s| (*s).into()).collect(),
        data_channels: vec![],
        scientific_ranges: vec![],
    }
}

fn edge(from: &str, to: &str, label: &str) -> ScenarioEdge {
    ScenarioEdge {
        from: from.into(),
        to: to.into(),
        edge_type: "data_flow".into(),
        label: label.into(),
    }
}

fn scaffold(name: &str, description: &str) -> EcologyScenario {
    EcologyScenario {
        name: name.into(),
        description: description.into(),
        version: "1.0.0".into(),
        mode: "live-ecosystem".into(),
        domain: "default".into(),
        nodes: vec![],
        edges: vec![],
    }
}

/// Build a hardware inventory scenario from discovered substrates.
///
/// Each substrate becomes a node with gauge channels for memory, cores,
/// and capability counts.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "memory bytes and device counts are safely representable in f64"
)]
pub fn inventory_scenario(substrates: &[Substrate]) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "metalForge Hardware Inventory",
        "Runtime-discovered compute substrates — GPU, NPU, CPU",
    );

    let mut gpu_count = 0_u32;
    let mut npu_count = 0_u32;
    let mut cpu_count = 0_u32;

    for sub in substrates {
        let (kind_label, id_prefix) = match sub.kind {
            SubstrateKind::Gpu => {
                gpu_count += 1;
                ("GPU", format!("gpu_{gpu_count}"))
            }
            SubstrateKind::Npu => {
                npu_count += 1;
                ("NPU", format!("npu_{npu_count}"))
            }
            SubstrateKind::Cpu => {
                cpu_count += 1;
                ("CPU", format!("cpu_{cpu_count}"))
            }
        };

        let mut sub_node = node(
            &id_prefix,
            &format!("{kind_label}: {}", sub.identity.name),
            "storage",
            &["hardware.substrate"],
        );

        if let Some(mem) = sub.properties.memory_bytes {
            let mem_gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
            sub_node.data_channels.push(DataChannel::Gauge {
                id: format!("{id_prefix}_memory"),
                label: "Memory".into(),
                value: mem_gb,
                min: 0.0,
                max: mem_gb * 1.5,
                unit: "GiB".into(),
                normal_range: [0.0, mem_gb],
                warning_range: [0.0, 0.0],
            });
        }

        if let Some(cores) = sub.properties.core_count {
            sub_node.data_channels.push(DataChannel::Gauge {
                id: format!("{id_prefix}_cores"),
                label: "Cores".into(),
                value: f64::from(cores),
                min: 0.0,
                max: f64::from(cores) * 1.5,
                unit: "cores".into(),
                normal_range: [0.0, f64::from(cores)],
                warning_range: [0.0, 0.0],
            });
        }

        let cap_names: Vec<String> = sub.capabilities.iter().map(|c| format!("{c:?}")).collect();
        sub_node.data_channels.push(DataChannel::Bar {
            id: format!("{id_prefix}_caps"),
            label: "Capabilities".into(),
            categories: cap_names.clone(),
            values: vec![1.0; cap_names.len()],
            unit: "present".into(),
        });

        if sub.properties.has_f64 {
            sub_node.scientific_ranges.push(ScientificRange {
                label: "f64 compute available".into(),
                min: 1.0,
                max: 1.0,
                status: "normal".into(),
            });
        }

        s.nodes.push(sub_node);
    }

    let summary_node = {
        let mut n = node("summary", "Hardware Summary", "data", &["hardware.summary"]);
        n.data_channels.push(DataChannel::Bar {
            id: "substrate_counts".into(),
            label: "Substrate Counts".into(),
            categories: vec!["GPU".into(), "NPU".into(), "CPU".into()],
            values: vec![
                f64::from(gpu_count),
                f64::from(npu_count),
                f64::from(cpu_count),
            ],
            unit: "devices".into(),
        });
        n
    };
    s.nodes.push(summary_node);

    let mut edges = Vec::new();
    for sub_node in &s.nodes {
        if sub_node.id != "summary" {
            edges.push(edge(&sub_node.id, "summary", "probed"));
        }
    }

    (s, edges)
}

/// Build a workload dispatch scenario showing which substrates handle which domains.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "workload count < 100 — safely representable in f64"
)]
pub fn dispatch_scenario(substrates: &[Substrate]) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "metalForge Workload Dispatch",
        "Capability-based routing of science workloads to compute substrates",
    );

    let all_bio = workloads::all_workloads();

    let mut dispatch_node = node(
        "dispatch",
        "Dispatch Router",
        "compute",
        &["dispatch.route"],
    );

    let workload_names: Vec<String> = all_bio.iter().map(|bw| bw.workload.name.clone()).collect();
    let routable: Vec<f64> = all_bio
        .iter()
        .map(|bw| {
            if dispatch::route(&bw.workload, substrates).is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    dispatch_node.data_channels.push(DataChannel::Bar {
        id: "routable_workloads".into(),
        label: "Routable Workloads".into(),
        categories: workload_names,
        values: routable,
        unit: "routable".into(),
    });

    let routable_count = all_bio
        .iter()
        .filter(|bw| dispatch::route(&bw.workload, substrates).is_some())
        .count();

    dispatch_node.data_channels.push(DataChannel::Gauge {
        id: "route_coverage".into(),
        label: "Route Coverage".into(),
        value: routable_count as f64,
        min: 0.0,
        max: all_bio.len() as f64,
        unit: "workloads".into(),
        normal_range: [all_bio.len() as f64 * 0.8, all_bio.len() as f64],
        warning_range: [all_bio.len() as f64 * 0.5, all_bio.len() as f64 * 0.8],
    });

    s.nodes.push(dispatch_node);
    (s, vec![])
}

/// Build a NUCLEUS topology scenario showing Tower→Node→Nest flow.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "substrate count < 100 — safely representable in f64"
)]
pub fn nucleus_scenario(substrates: &[Substrate]) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "metalForge NUCLEUS Topology",
        "Tower discovery → Node compute → Nest storage atomic flow",
    );

    let tower = node("tower", "Tower (Discovery)", "pipeline", &["nucleus.tower"]);
    s.nodes.push(tower);

    let mut node_atomic = node(
        "node_atomic",
        "Node (Compute)",
        "compute",
        &["nucleus.node"],
    );
    node_atomic.data_channels.push(DataChannel::Gauge {
        id: "substrates_discovered".into(),
        label: "Substrates Discovered".into(),
        value: substrates.len() as f64,
        min: 0.0,
        max: 10.0,
        unit: "substrates".into(),
        normal_range: [1.0, 10.0],
        warning_range: [0.0, 1.0],
    });
    s.nodes.push(node_atomic);

    let nest = node("nest", "Nest (Storage)", "storage", &["nucleus.nest"]);
    s.nodes.push(nest);

    let edges = vec![
        edge("tower", "node_atomic", "discover → dispatch"),
        edge("node_atomic", "nest", "compute → store"),
        edge("nest", "tower", "store → federate"),
    ];

    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::{Capability, Identity, Properties, SubstrateOrigin};

    fn test_substrates() -> Vec<Substrate> {
        vec![
            Substrate {
                kind: SubstrateKind::Gpu,
                identity: Identity {
                    name: "RTX 4070".into(),
                    driver: Some("NVIDIA".into()),
                    backend: Some("Vulkan".into()),
                    adapter_index: Some(0),
                    device_node: None,
                    pci_id: Some("10de:2786".into()),
                },
                properties: Properties {
                    memory_bytes: Some(12 * 1024 * 1024 * 1024),
                    core_count: None,
                    thread_count: None,
                    cache_kb: None,
                    has_f64: true,
                    has_timestamps: true,
                },
                capabilities: vec![Capability::F64Compute, Capability::F32Compute],
                origin: SubstrateOrigin::Local,
            },
            Substrate {
                kind: SubstrateKind::Cpu,
                identity: Identity {
                    name: "i9-12900K".into(),
                    driver: None,
                    backend: None,
                    adapter_index: None,
                    device_node: None,
                    pci_id: None,
                },
                properties: Properties {
                    memory_bytes: Some(64 * 1024 * 1024 * 1024),
                    core_count: Some(16),
                    thread_count: Some(24),
                    cache_kb: Some(30720),
                    has_f64: true,
                    has_timestamps: false,
                },
                capabilities: vec![
                    Capability::F64Compute,
                    Capability::CpuCompute,
                    Capability::SimdVector,
                ],
                origin: SubstrateOrigin::Local,
            },
        ]
    }

    #[test]
    fn inventory_builds() {
        let subs = test_substrates();
        let (scenario, edges) = inventory_scenario(&subs);
        assert_eq!(scenario.nodes.len(), 3); // GPU + CPU + summary
        assert!(!edges.is_empty());
    }

    #[test]
    fn dispatch_builds() {
        let subs = test_substrates();
        let (scenario, _) = dispatch_scenario(&subs);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn nucleus_builds() {
        let subs = test_substrates();
        let (scenario, edges) = nucleus_scenario(&subs);
        assert_eq!(scenario.nodes.len(), 3);
        assert_eq!(edges.len(), 3);
    }
}
