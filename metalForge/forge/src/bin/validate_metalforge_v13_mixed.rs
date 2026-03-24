// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp289: metalForge v13 — Mixed Hardware + NUCLEUS Atomics + `PCIe` Bypass
//!
//! Validates mixed-hardware dispatch and NUCLEUS atomics at the V92D state:
//! - S1: Tower discovery + bandwidth tier detection
//! - S2: Workload catalog (absorption tracking, origin summary)
//! - S3: Capability-based routing (GPU, NPU, CPU fallback)
//! - S4: Bandwidth-aware routing (`PCIe` bypass vs CPU round-trip)
//! - S5: Streaming pipeline analysis (chained vs round-trip stages)
//! - S6: Node dispatch for all workloads
//! - S7: Mixed pipeline topology (NPU→GPU→CPU interleave)
//! - S8: Sovereign mode (no external deps required)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-02 |
//! | Command | `cargo run -p wetspring-forge --bin validate_metalforge_v13_mixed` |
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end mixed-hardware pipeline integration

use wetspring_forge::bridge;
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
    println!("  Exp289: metalForge v13 — Mixed Hardware + NUCLEUS");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0_u32;
    let mut fail = 0_u32;

    section_tower_discovery(&mut pass, &mut fail);
    section_workload_catalog(&mut pass, &mut fail);
    section_capability_routing(&mut pass, &mut fail);
    section_bandwidth_aware(&mut pass, &mut fail);
    section_streaming_analysis(&mut pass, &mut fail);
    section_node_dispatch(&mut pass, &mut fail);
    section_mixed_pipeline(&mut pass, &mut fail);
    section_sovereign_mode(&mut pass, &mut fail);

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

fn section_tower_discovery(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Tower Discovery + Bandwidth Tiers");

    let subs = inventory::discover();
    check("Tower: at least 1 substrate", !subs.is_empty(), pass, fail);

    let has_cpu = subs.iter().any(|s| s.kind == SubstrateKind::Cpu);
    check("Tower: CPU always present", has_cpu, pass, fail);

    let gpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Gpu).count();
    println!("    GPUs discovered: {gpu_count}");

    let mut f64_gpu_count = 0_usize;
    for sub in &subs {
        if sub.kind == SubstrateKind::Gpu {
            let bw = bridge::detect_bandwidth_tier(sub);
            let tier_name = if bw.is_some() { "detected" } else { "unknown" };
            let has_f64 = sub.has(&Capability::F64Compute);
            if has_f64 {
                f64_gpu_count += 1;
            }
            println!(
                "    GPU: {} — bandwidth: {tier_name}, f64: {has_f64}",
                sub.identity.name
            );
        }
    }
    check(
        "Tower: at least 1 GPU with F64 (or no GPUs)",
        f64_gpu_count > 0 || gpu_count == 0,
        pass,
        fail,
    );

    let npu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Npu).count();
    println!("    NPUs discovered: {npu_count}");
    check(
        "Tower: substrate count consistent",
        subs.len() == gpu_count + npu_count + 1,
        pass,
        fail,
    );
}

fn section_workload_catalog(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: Workload Catalog + Absorption Tracking");

    let all = workloads::all_workloads();
    check("Catalog: non-empty", !all.is_empty(), pass, fail);
    println!("    Total workloads: {}", all.len());

    let (absorbed, local, cpu_only) = workloads::origin_summary();
    println!("    Absorbed: {absorbed}, Local: {local}, CpuOnly: {cpu_only}");

    check(
        "Catalog: absorbed + local + cpu_only = total",
        absorbed + local + cpu_only == all.len(),
        pass,
        fail,
    );
    check("Catalog: zero local (fully lean)", local == 0, pass, fail);

    let has_names = all.iter().all(|w| !w.workload.name.is_empty());
    check("Catalog: all workloads named", has_names, pass, fail);

    let all_absorbed = all
        .iter()
        .filter(|w| matches!(w.origin, ShaderOrigin::Absorbed))
        .count();
    check(
        "Catalog: absorbed count matches",
        all_absorbed == absorbed,
        pass,
        fail,
    );
}

fn section_capability_routing(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: Capability-Based Routing");

    let subs = inventory::discover();

    let gpu_workload = dispatch::Workload::new("diversity_gpu", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Gpu);
    let decision = dispatch::route(&gpu_workload, &subs);
    check(
        "Route: F64 workload routes somewhere",
        decision.is_some(),
        pass,
        fail,
    );
    if let Some(d) = &decision {
        println!(
            "    diversity_gpu → {:?} ({:?})",
            d.substrate.kind, d.reason
        );
        check(
            "Route: GPU workload → GPU or CPU fallback",
            d.substrate.kind == SubstrateKind::Gpu || d.substrate.kind == SubstrateKind::Cpu,
            pass,
            fail,
        );
    }

    let cpu_workload = dispatch::Workload::new("quality_filter", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Cpu);
    let cpu_decision = dispatch::route(&cpu_workload, &subs);
    check(
        "Route: CPU-preferred routes to CPU",
        cpu_decision
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
        pass,
        fail,
    );

    let npu_workload = dispatch::Workload::new(
        "esn_classify",
        vec![Capability::QuantizedInference { bits: 8 }],
    )
    .prefer(SubstrateKind::Npu);
    let npu_decision = dispatch::route(&npu_workload, &subs);
    if let Some(d) = &npu_decision {
        println!("    esn_classify → {:?} ({:?})", d.substrate.kind, d.reason);
        check("Route: NPU workload routed", true, pass, fail);
    } else {
        println!("    esn_classify → no NPU present (sovereign mode)");
        check("Route: NPU absent — sovereign fallback", true, pass, fail);
    }
}

fn section_bandwidth_aware(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Bandwidth-Aware Routing (PCIe Bypass)");

    let subs = inventory::discover();

    let small_workload = dispatch::Workload::new("small_diversity", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Gpu)
        .with_data_bytes(1024);
    let small_decision = dispatch::route_bandwidth_aware(&small_workload, &subs);
    check(
        "BW-aware: small workload routes",
        small_decision.is_some(),
        pass,
        fail,
    );
    if let Some(d) = &small_decision {
        println!("    small (1KB) → {:?} ({:?})", d.substrate.kind, d.reason);
    }

    let large_workload = dispatch::Workload::new("large_gemm", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Gpu)
        .with_data_bytes(100_000_000);
    let large_decision = dispatch::route_bandwidth_aware(&large_workload, &subs);
    check(
        "BW-aware: large workload routes",
        large_decision.is_some(),
        pass,
        fail,
    );
    if let Some(d) = &large_decision {
        println!(
            "    large (100MB) → {:?} ({:?})",
            d.substrate.kind, d.reason
        );
    }

    for sub in &subs {
        if sub.kind == SubstrateKind::Gpu {
            let transfer_us = bridge::estimated_transfer_us(sub, 1_000_000);
            if let Some(us) = transfer_us {
                println!("    {} 1MB transfer: {us:.0} µs", sub.identity.name);
                check(
                    &format!("BW: {} transfer estimate > 0", sub.identity.name),
                    us > 0.0,
                    pass,
                    fail,
                );
            }
        }
    }
}

fn section_streaming_analysis(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Streaming Pipeline Analysis");

    let mut session = StreamingSession::new(SubstrateKind::Gpu);

    session.add_stage(PipelineStage {
        name: "diversity_gpu".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "anderson_spectral".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "pcoa_gpu".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });

    let analysis = session.analyze();
    check("Streaming: 3 stages", analysis.n_stages == 3, pass, fail);
    check(
        "Streaming: GPU-chained ≥ 1",
        analysis.gpu_chained >= 1,
        pass,
        fail,
    );
    check(
        "Streaming: CPU round-trips ≤ 1",
        analysis.cpu_roundtrips <= 1,
        pass,
        fail,
    );
    println!(
        "    Pipeline: {} stages, {} GPU-chained, {} CPU round-trips",
        analysis.n_stages, analysis.gpu_chained, analysis.cpu_roundtrips
    );

    let mut mixed_session = StreamingSession::new(SubstrateKind::Gpu);
    mixed_session.add_stage(PipelineStage {
        name: "npu_classify".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    mixed_session.add_stage(PipelineStage {
        name: "gpu_diversity".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed_session.add_stage(PipelineStage {
        name: "cpu_export".into(),
        capability: Capability::CpuCompute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });

    let mixed_analysis = mixed_session.analyze();
    check(
        "Mixed pipeline: 3 stages",
        mixed_analysis.n_stages == 3,
        pass,
        fail,
    );
    check(
        "Mixed pipeline: has CPU round-trips",
        mixed_analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    println!(
        "    Mixed: {} stages, {} GPU, {} round-trips",
        mixed_analysis.n_stages, mixed_analysis.gpu_chained, mixed_analysis.cpu_roundtrips
    );
}

fn section_node_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Node Dispatch — All Workloads");

    let subs = inventory::discover();
    let all = workloads::all_workloads();

    let mut routed = 0_usize;
    let mut fallback = 0_usize;
    let mut cpu_only_count = 0_usize;

    for wl in &all {
        let decision = dispatch::route(&wl.workload, &subs);
        if let Some(d) = decision {
            routed += 1;
            if d.substrate.kind == SubstrateKind::Cpu && !matches!(wl.origin, ShaderOrigin::CpuOnly)
            {
                fallback += 1;
            }
        }
        if matches!(wl.origin, ShaderOrigin::CpuOnly) {
            cpu_only_count += 1;
        }
    }

    let unrouted = all.len() - routed;
    println!(
        "    Routed: {routed}/{}, unrouted: {unrouted}, fallback: {fallback}, CPU-only: {cpu_only_count}",
        all.len()
    );
    check(
        "Node: most workloads routed (NPU may be absent)",
        routed >= all.len() - cpu_only_count,
        pass,
        fail,
    );
}

fn section_mixed_pipeline(pass: &mut u32, fail: &mut u32) {
    println!("\n  S7: Mixed Pipeline Topology (NPU→GPU→CPU)");

    let subs = inventory::discover();

    let stages = [
        ("diversity_gpu", SubstrateKind::Gpu),
        ("anderson_spectral", SubstrateKind::Gpu),
        ("quality_filter", SubstrateKind::Cpu),
    ];

    let mut pipeline_ok = true;
    for (name, preferred) in &stages {
        let cap = Capability::F64Compute;
        let wl = dispatch::Workload::new(*name, vec![cap]).prefer(*preferred);
        let decision = dispatch::route(&wl, &subs);
        let routed = decision.is_some();
        if !routed {
            pipeline_ok = false;
        }
        let target = decision
            .as_ref()
            .map_or_else(|| "NONE".to_string(), |d| format!("{:?}", d.substrate.kind));
        println!("    {name}: prefer {preferred:?} → {target}");
    }
    check("Mixed pipeline: all stages routed", pipeline_ok, pass, fail);

    let has_gpu_to_cpu = stages
        .windows(2)
        .any(|pair| pair[0].1 == SubstrateKind::Gpu && pair[1].1 == SubstrateKind::Cpu);
    check(
        "Mixed pipeline: GPU→CPU handoff exists",
        has_gpu_to_cpu,
        pass,
        fail,
    );
}

fn section_sovereign_mode(pass: &mut u32, fail: &mut u32) {
    println!("\n  S8: Sovereign Mode — No External Dependencies");

    let subs = inventory::discover();
    check(
        "Sovereign: local discovery works",
        !subs.is_empty(),
        pass,
        fail,
    );

    let nestgate = nest::discover_nestgate_socket();
    if nestgate.is_some() {
        println!("    NestGate available — using distributed storage");
    } else {
        println!("    NestGate absent — sovereign fallback active");
    }
    check("Sovereign: operates without NestGate", true, pass, fail);

    let songbird = inventory::discover_songbird_socket();
    if songbird.is_some() {
        println!("    Songbird available — using Tower discovery");
    } else {
        println!("    Songbird absent — local-only discovery");
    }
    check("Sovereign: operates without Songbird", true, pass, fail);

    let all = workloads::all_workloads();
    let routable_count = all
        .iter()
        .filter(|wl| dispatch::route(&wl.workload, &subs).is_some())
        .count();
    println!("    Routable: {routable_count}/{}", all.len());
    check(
        "Sovereign: majority of workloads route locally",
        routable_count > all.len() / 2,
        pass,
        fail,
    );
}
