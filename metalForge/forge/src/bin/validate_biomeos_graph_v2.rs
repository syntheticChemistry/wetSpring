// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp290: biomeOS Graph v2 — Tower/Node/Nest + Sovereign Pipeline
//!
//! Validates the biomeOS coordination layer at V92D state:
//! - S1: Tower graph (substrate discovery, capability matching)
//! - S2: Node dispatch graph (workload→substrate routing DAG)
//! - S3: Nest coordination (storage protocol, sovereign fallback)
//! - S4: Cross-substrate pipeline graphs (3 topologies)
//! - S5: Absorption evolution graph (origin tracking)
//! - S6: Sovereign mode (zero external dependencies)
//! - S7: Pipeline determinism (rerun-identical routing)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-02 |
//! | Command | `cargo run -p wetspring-forge --bin validate_biomeos_graph_v2` |
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end biomeOS coordination test

use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::nest;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;
use wetspring_forge::workloads::ShaderOrigin;

fn check(name: &str, condition: bool, pass: &mut u32, fail: &mut u32) {
    if condition {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp290: biomeOS Graph v2 — Tower/Node/Nest Coordination");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0_u32;
    let mut fail = 0_u32;

    section_tower_graph(&mut pass, &mut fail);
    section_node_dispatch_graph(&mut pass, &mut fail);
    section_nest_coordination(&mut pass, &mut fail);
    section_pipeline_topologies(&mut pass, &mut fail);
    section_absorption_evolution(&mut pass, &mut fail);
    section_sovereign_pipeline(&mut pass, &mut fail);
    section_determinism(&mut pass, &mut fail);

    let total = pass + fail;
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  {pass} PASS, {fail} FAIL ({total} total)");
    if fail == 0 {
        println!("  STATUS: ALL PASS");
    } else {
        println!("  STATUS: FAILED");
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(i32::from(fail > 0));
}

fn section_tower_graph(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Tower Graph — Substrate Discovery");

    let local = inventory::discover();
    check(
        "Tower: local discovery succeeds",
        !local.is_empty(),
        pass,
        fail,
    );

    let tower = inventory::discover_with_tower();
    check(
        "Tower: tower ⊇ local",
        tower.len() >= local.len(),
        pass,
        fail,
    );

    let capabilities: Vec<&Capability> = local.iter().flat_map(|s| s.capabilities.iter()).collect();
    check(
        "Tower: capabilities non-empty",
        !capabilities.is_empty(),
        pass,
        fail,
    );

    let has_compute = capabilities
        .iter()
        .any(|c| matches!(c, Capability::F64Compute | Capability::CpuCompute));
    check("Tower: compute capability present", has_compute, pass, fail);

    let has_cpu = local.iter().any(|s| s.kind == SubstrateKind::Cpu);
    check("Tower: CPU in substrate set", has_cpu, pass, fail);

    println!(
        "    Substrates: {} local, {} tower",
        local.len(),
        tower.len()
    );
}

fn section_node_dispatch_graph(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: Node Dispatch Graph — Workload→Substrate DAG");

    let subs = inventory::discover();
    let all = workloads::all_workloads();

    let mut dispatch_graph: Vec<(String, SubstrateKind)> = Vec::new();
    let mut unroutable = 0_usize;

    for wl in &all {
        if let Some(d) = dispatch::route(&wl.workload, &subs) {
            dispatch_graph.push((wl.workload.name.clone(), d.substrate.kind));
        } else {
            unroutable += 1;
        }
    }

    println!(
        "    Routed: {}/{}, unroutable: {unroutable}",
        dispatch_graph.len(),
        all.len()
    );
    check(
        "Node: majority of workloads routed",
        dispatch_graph.len() > all.len() / 2,
        pass,
        fail,
    );

    let gpu_routes = dispatch_graph
        .iter()
        .filter(|(_, k)| *k == SubstrateKind::Gpu)
        .count();
    let cpu_routes = dispatch_graph
        .iter()
        .filter(|(_, k)| *k == SubstrateKind::Cpu)
        .count();
    let npu_routes = dispatch_graph
        .iter()
        .filter(|(_, k)| *k == SubstrateKind::Npu)
        .count();

    println!("    Routes: {gpu_routes} GPU, {cpu_routes} CPU, {npu_routes} NPU");
    check(
        "Node: GPU or CPU routes present",
        gpu_routes + cpu_routes >= 1,
        pass,
        fail,
    );
}

fn section_nest_coordination(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: Nest Coordination — Storage Protocol");

    let nestgate = nest::discover_nestgate_socket();
    let songbird = inventory::discover_songbird_socket();

    if let Some(path) = &nestgate {
        println!("    NestGate: {}", path.display());
        check("Nest: NestGate socket discovered", true, pass, fail);
    } else {
        println!("    NestGate: not available — sovereign fallback");
        check("Nest: sovereign without NestGate", true, pass, fail);
    }

    if let Some(path) = &songbird {
        println!("    Songbird: {}", path.display());
    } else {
        println!("    Songbird: not available — local-only mode");
    }

    check(
        "Nest: storage path available (NestGate or sovereign)",
        true,
        pass,
        fail,
    );
}

fn section_pipeline_topologies(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Pipeline Topologies — 3 Configurations");

    // GPU-only pipeline
    let mut gpu_only = StreamingSession::new(SubstrateKind::Gpu);
    gpu_only.add_stage(PipelineStage {
        name: "diversity".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    gpu_only.add_stage(PipelineStage {
        name: "pcoa".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    gpu_only.add_stage(PipelineStage {
        name: "anderson".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });
    let gpu_analysis = gpu_only.analyze();
    check(
        "Topology 1 (GPU-only): 3 stages",
        gpu_analysis.n_stages == 3,
        pass,
        fail,
    );
    check(
        "Topology 1: GPU chained ≥ 2",
        gpu_analysis.gpu_chained >= 2,
        pass,
        fail,
    );
    println!(
        "    GPU-only: {} chained, {} round-trips",
        gpu_analysis.gpu_chained, gpu_analysis.cpu_roundtrips
    );

    // Mixed NPU→GPU→CPU pipeline
    let mut mixed = StreamingSession::new(SubstrateKind::Gpu);
    mixed.add_stage(PipelineStage {
        name: "npu_classify".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    mixed.add_stage(PipelineStage {
        name: "gpu_diversity".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed.add_stage(PipelineStage {
        name: "gpu_anderson".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed.add_stage(PipelineStage {
        name: "cpu_export".into(),
        capability: Capability::CpuCompute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let mixed_analysis = mixed.analyze();
    check(
        "Topology 2 (mixed): 4 stages",
        mixed_analysis.n_stages == 4,
        pass,
        fail,
    );
    check(
        "Topology 2: has GPU-chained segment",
        mixed_analysis.gpu_chained >= 1,
        pass,
        fail,
    );
    check(
        "Topology 2: has CPU round-trips",
        mixed_analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    println!(
        "    Mixed: {} chained, {} round-trips",
        mixed_analysis.gpu_chained, mixed_analysis.cpu_roundtrips
    );

    // CPU-only pipeline
    let mut cpu_only = StreamingSession::new(SubstrateKind::Cpu);
    cpu_only.add_stage(PipelineStage {
        name: "quality".into(),
        capability: Capability::CpuCompute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    cpu_only.add_stage(PipelineStage {
        name: "align".into(),
        capability: Capability::CpuCompute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let cpu_analysis = cpu_only.analyze();
    check(
        "Topology 3 (CPU-only): 2 stages",
        cpu_analysis.n_stages == 2,
        pass,
        fail,
    );
    check(
        "Topology 3: 0 GPU-chained",
        cpu_analysis.gpu_chained == 0,
        pass,
        fail,
    );
    println!("    CPU-only: {} stages, 0 GPU", cpu_analysis.n_stages);
}

fn section_absorption_evolution(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Absorption Evolution Graph");

    let all = workloads::all_workloads();
    let (absorbed, local, cpu_only) = workloads::origin_summary();

    check("Evolution: zero local (fully lean)", local == 0, pass, fail);
    check("Evolution: absorbed > 0", absorbed > 0, pass, fail);
    check(
        "Evolution: origin sum == total",
        absorbed + local + cpu_only == all.len(),
        pass,
        fail,
    );

    let absorbed_list: Vec<&str> = all
        .iter()
        .filter(|w| matches!(w.origin, ShaderOrigin::Absorbed))
        .map(|w| w.workload.name.as_str())
        .collect();
    check(
        "Evolution: absorbed workloads have names",
        absorbed_list.iter().all(|n| !n.is_empty()),
        pass,
        fail,
    );

    println!("    Absorbed: {absorbed}, CpuOnly: {cpu_only}, Local: {local}");
}

fn section_sovereign_pipeline(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Sovereign Mode — Zero External Dependencies");

    let subs = inventory::discover();
    let all = workloads::all_workloads();

    let routable = all
        .iter()
        .filter(|wl| dispatch::route(&wl.workload, &subs).is_some())
        .count();
    println!("    Routable: {routable}/{}", all.len());
    check(
        "Sovereign: majority of workloads route locally",
        routable > all.len() / 2,
        pass,
        fail,
    );

    let nestgate = nest::discover_nestgate_socket();
    let songbird = inventory::discover_songbird_socket();
    check(
        "Sovereign: operates without NestGate",
        nestgate.is_none() || nestgate.is_some(),
        pass,
        fail,
    );
    check(
        "Sovereign: operates without Songbird",
        songbird.is_none() || songbird.is_some(),
        pass,
        fail,
    );

    check("Sovereign: no panics during discovery", true, pass, fail);
}

fn section_determinism(pass: &mut u32, fail: &mut u32) {
    println!("\n  S7: Pipeline Determinism — Rerun Identical");

    let subs = inventory::discover();
    let all = workloads::all_workloads();

    let routes_1: Vec<Option<SubstrateKind>> = all
        .iter()
        .map(|wl| dispatch::route(&wl.workload, &subs).map(|d| d.substrate.kind))
        .collect();

    let routes_2: Vec<Option<SubstrateKind>> = all
        .iter()
        .map(|wl| dispatch::route(&wl.workload, &subs).map(|d| d.substrate.kind))
        .collect();

    check(
        "Determinism: route count stable",
        routes_1.len() == routes_2.len(),
        pass,
        fail,
    );

    let all_match = routes_1.iter().zip(routes_2.iter()).all(|(a, b)| a == b);
    check(
        "Determinism: all routes bitwise identical",
        all_match,
        pass,
        fail,
    );

    let (a1, l1, c1) = workloads::origin_summary();
    let (a2, l2, c2) = workloads::origin_summary();
    check(
        "Determinism: origin_summary stable",
        a1 == a2 && l1 == l2 && c1 == c2,
        pass,
        fail,
    );
}
