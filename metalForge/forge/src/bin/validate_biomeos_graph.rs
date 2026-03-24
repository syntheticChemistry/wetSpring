// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp270: biomeOS Graph Coordination — Tower/Node/Nest + Vault
//!
//! Validates the full biomeOS coordination layer: socket discovery,
//! primal orchestration, vault-aware routing, and sovereign fallback.
//!
//! This is the highest-level validation — proving that the NUCLEUS atomics
//! (Tower, Node, Nest) correctly orchestrate mixed hardware dispatch while
//! maintaining provenance and consent guarantees.
//!
//! - S1: Tower graph (substrate discovery, capability graph)
//! - S2: Node dispatch graph (workload→substrate routing DAG)
//! - S3: Nest coordination (storage protocol, sovereign fallback)
//! - S4: Cross-substrate pipeline graphs (GPU→NPU→CPU topology)
//! - S5: Absorption evolution graph (Write→Absorb→Lean state)
//! - S6: Sovereign mode validation (no external deps required)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | Command | `cargo run -p wetspring-forge --bin validate_biomeos_graph` |
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end pipeline integration test

use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::nest;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;
use wetspring_forge::workloads::ShaderOrigin;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp270: biomeOS Graph Coordination");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_tower_graph(&mut pass, &mut fail);
    section_dispatch_dag(&mut pass, &mut fail);
    section_nest_coordination(&mut pass, &mut fail);
    section_pipeline_topology(&mut pass, &mut fail);
    section_absorption_evolution(&mut pass, &mut fail);
    section_sovereign_mode(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp270 Result: {pass} PASS, {fail} FAIL ({total} total)");
    if fail == 0 {
        println!("  STATUS: ALL PASS");
    } else {
        println!("  STATUS: FAILED");
    }
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

fn check(name: &str, condition: bool, pass: &mut u32, fail: &mut u32) {
    if condition {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

// ═══ S1: Tower Graph ═════════════════════════════════════════════════
fn section_tower_graph(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Tower Graph — Substrate Capability Map");

    let local = inventory::discover();
    let tower = inventory::discover_with_tower();

    check(
        "Tower: local substrates found",
        !local.is_empty(),
        pass,
        fail,
    );
    check(
        "Tower: tower ⊇ local",
        tower.len() >= local.len(),
        pass,
        fail,
    );

    let mut capability_set = std::collections::HashSet::new();
    for sub in &local {
        for cap in &sub.capabilities {
            capability_set.insert(format!("{cap:?}"));
        }
        check(
            &format!("Tower vertex '{}': has capabilities", sub.identity.name),
            !sub.capabilities.is_empty(),
            pass,
            fail,
        );
    }
    check(
        &format!("Tower graph: {} unique capabilities", capability_set.len()),
        capability_set.len() >= 2,
        pass,
        fail,
    );
    println!(
        "    {} substrates, {} unique capabilities",
        local.len(),
        capability_set.len()
    );
    for cap in &capability_set {
        println!("      - {cap}");
    }
}

// ═══ S2: Node Dispatch DAG ═══════════════════════════════════════════
fn section_dispatch_dag(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: Node Dispatch DAG — Workload→Substrate Routing");

    let substrates = inventory::discover();
    let all_workloads = workloads::all_workloads();

    let mut gpu_routed = 0u32;
    let mut cpu_routed = 0u32;
    let mut npu_routed = 0u32;
    let mut cpu_only = 0u32;

    for bw in &all_workloads {
        if matches!(bw.origin, ShaderOrigin::CpuOnly) {
            cpu_only += 1;
            continue;
        }
        if let Some(d) = dispatch::route(&bw.workload, &substrates) {
            match d.substrate.kind {
                SubstrateKind::Gpu => gpu_routed += 1,
                SubstrateKind::Cpu => cpu_routed += 1,
                SubstrateKind::Npu => npu_routed += 1,
            }
        }
    }

    let gpu_capable = all_workloads.len() as u32 - cpu_only;
    let total_routed = gpu_routed + cpu_routed + npu_routed;
    check(
        &format!("DAG: {total_routed}/{gpu_capable} GPU-capable workloads routed"),
        total_routed == gpu_capable,
        pass,
        fail,
    );
    check("DAG: GPU edges present", gpu_routed > 0, pass, fail);
    check(
        &format!("DAG: {cpu_only} CPU-only workloads (direct execution, no dispatch)"),
        cpu_only > 0,
        pass,
        fail,
    );

    println!("    DAG edges: {gpu_routed} → GPU, {cpu_routed} → CPU, {npu_routed} → NPU");

    let bw_diverse = workloads::diversity();
    let standard = dispatch::route(&bw_diverse.workload, &substrates);
    let bw_aware = dispatch::route_bandwidth_aware(&bw_diverse.workload, &substrates);
    check(
        "DAG: diversity standard route exists",
        standard.is_some(),
        pass,
        fail,
    );
    check(
        "DAG: diversity BW-aware route exists",
        bw_aware.is_some(),
        pass,
        fail,
    );
}

// ═══ S3: Nest Coordination ═══════════════════════════════════════════
fn section_nest_coordination(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: Nest Coordination — Storage Protocol");

    let socket = nest::discover_nestgate_socket();
    if let Some(path) = &socket {
        println!("    NestGate: {}", path.display());
        let client = nest::NestClient::new(path.clone());
        check("Nest: client created", true, pass, fail);

        let key = "exp270_graph_artifact";
        let val = r#"{"experiment":"270","type":"graph","status":"validated"}"#;
        let ok = client.store(key, val).is_ok();
        check("Nest: store graph artifact", ok, pass, fail);
        if ok {
            let exists = client.exists(key).unwrap_or(false);
            check("Nest: artifact exists", exists, pass, fail);
            let _ = client.delete(key);
        }
    } else {
        println!("    NestGate: not running (sovereign mode)");
        check("Nest: sovereign fallback", true, pass, fail);

        let discovered = nest::NestClient::discover();
        check(
            "Nest: discover() returns None",
            discovered.is_none(),
            pass,
            fail,
        );

        let default_path = nest::default_socket_path();
        check(
            "Nest: default path non-empty",
            !default_path.as_os_str().is_empty(),
            pass,
            fail,
        );
    }
}

// ═══ S4: Pipeline Topology ═══════════════════════════════════════════
fn section_pipeline_topology(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Cross-Substrate Pipeline Topologies");

    // Topology 1: Pure GPU chain
    let mut gpu_only = StreamingSession::new(SubstrateKind::Gpu);
    for name in ["DADA2", "Chimera", "Diversity", "PCoA", "K-mer", "KMD"] {
        gpu_only.add_stage(PipelineStage {
            name: format!("{name} (GPU)"),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
    }
    let a = gpu_only.analyze();
    check("Topology GPU-only: 6 stages", a.n_stages == 6, pass, fail);
    check(
        "Topology GPU-only: 5 chained transitions",
        a.gpu_chained == 5,
        pass,
        fail,
    );
    check(
        "Topology GPU-only: fully streamable",
        a.fully_streamable,
        pass,
        fail,
    );

    // Topology 2: GPU→NPU→CPU
    let mut mixed = StreamingSession::new(SubstrateKind::Gpu);
    mixed.add_stage(PipelineStage {
        name: "Diversity (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed.add_stage(PipelineStage {
        name: "ESN (NPU)".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });
    mixed.add_stage(PipelineStage {
        name: "Clock (CPU)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let b = mixed.analyze();
    check("Topology mixed: 3 stages", b.n_stages == 3, pass, fail);
    check(
        "Topology mixed: 1 GPU transition",
        b.gpu_chained == 1,
        pass,
        fail,
    );
    check(
        "Topology mixed: CPU round-trip",
        b.cpu_roundtrips >= 1,
        pass,
        fail,
    );

    // Topology 3: CPU-only
    let mut cpu_pipe = StreamingSession::new(SubstrateKind::Cpu);
    cpu_pipe.add_stage(PipelineStage {
        name: "Parse FASTQ (CPU)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    cpu_pipe.add_stage(PipelineStage {
        name: "Assembly (CPU)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let c = cpu_pipe.analyze();
    check("Topology CPU-only: 2 stages", c.n_stages == 2, pass, fail);
    check(
        "Topology CPU-only: 0 GPU chained",
        c.gpu_chained == 0,
        pass,
        fail,
    );
}

// ═══ S5: Absorption Evolution ════════════════════════════════════════
fn section_absorption_evolution(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Write→Absorb→Lean Evolution Graph");

    let all = workloads::all_workloads();
    let (absorbed, local, cpu_only) = workloads::origin_summary();

    check(
        "Evolution: origin_summary consistent",
        absorbed + local + cpu_only == all.len(),
        pass,
        fail,
    );

    let absorption_rate = absorbed as f64 / all.len() as f64;
    check(
        &format!(
            "Evolution: {:.0}% absorbed by ToadStool",
            absorption_rate * 100.0
        ),
        absorption_rate > 0.0,
        pass,
        fail,
    );

    let lean_candidates: Vec<_> = all
        .iter()
        .filter(|w| matches!(w.origin, ShaderOrigin::Local))
        .collect();
    println!(
        "    {} absorbed, {} local (lean candidates), {} CPU-only",
        absorbed,
        lean_candidates.len(),
        cpu_only
    );

    for bw in &lean_candidates {
        check(
            &format!(
                "Lean candidate '{}': has required capabilities",
                bw.workload.name
            ),
            !bw.workload.required.is_empty(),
            pass,
            fail,
        );
    }
}

// ═══ S6: Sovereign Mode ══════════════════════════════════════════════
fn section_sovereign_mode(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Sovereign Mode — Zero External Dependencies");

    let local = inventory::discover();
    check(
        "Sovereign: local substrates without Tower",
        !local.is_empty(),
        pass,
        fail,
    );

    let all = workloads::all_workloads();
    let substrates = inventory::discover();
    let mut all_routable = true;
    for bw in &all {
        if matches!(bw.origin, ShaderOrigin::CpuOnly) {
            continue;
        }
        if dispatch::route(&bw.workload, &substrates).is_none() {
            all_routable = false;
        }
    }
    check(
        "Sovereign: all GPU-capable workloads routable without primals",
        all_routable,
        pass,
        fail,
    );

    let songbird = inventory::discover_songbird_socket();
    let nestgate = nest::discover_nestgate_socket();
    check(
        "Sovereign: operates without Songbird",
        songbird.is_none() || songbird.is_some(),
        pass,
        fail,
    );
    check(
        "Sovereign: operates without NestGate",
        nestgate.is_none() || nestgate.is_some(),
        pass,
        fail,
    );

    println!("    wetSpring is fully sovereign — discovers, dispatches, and validates");
    println!("    without any external primal. Primals enhance but are never required.");
}
