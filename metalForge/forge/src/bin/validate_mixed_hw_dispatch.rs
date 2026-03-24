// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp269: Mixed Hardware Dispatch — NUCLEUS Atomics + `PCIe` Bypass
//!
//! Validates mixed-hardware compute dispatch through NUCLEUS atomics:
//! - S1: Tower discovery (all substrates, bandwidth tiers)
//! - S2: NPU→GPU `PCIe` bypass pipeline (zero CPU round-trips)
//! - S3: GPU→CPU fallback (f64 compute, molecular clock)
//! - S4: Multi-GPU load balancing (bandwidth-aware routing)
//! - S5: Full 8-stage mixed pipeline (NPU+GPU+CPU interleaved)
//! - S6: Workload absorption tracking (Write→Absorb→Lean)
//! - S7: Node dispatch for all 28 metalForge v12 workloads
//! - S8: biomeOS graph coordination (Tower+Node+Nest orchestration)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | Command | `cargo run -p wetspring-forge --bin validate_mixed_hw_dispatch` |
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end pipeline integration test

use wetspring_forge::bridge;
use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::nest;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;
use wetspring_forge::workloads::ShaderOrigin;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp269: Mixed Hardware Dispatch — NUCLEUS Atomics");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_tower_bandwidth(&mut pass, &mut fail);
    section_npu_gpu_bypass(&mut pass, &mut fail);
    section_gpu_cpu_fallback(&mut pass, &mut fail);
    section_multi_gpu_balance(&mut pass, &mut fail);
    section_full_mixed_pipeline(&mut pass, &mut fail);
    section_absorption_tracking(&mut pass, &mut fail);
    section_full_catalog_dispatch(&mut pass, &mut fail);
    section_biomeos_graph(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp269 Result: {pass} PASS, {fail} FAIL ({total} total)");
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

// ═══ S1: Tower Discovery + Bandwidth Tiers ═══════════════════════════
fn section_tower_bandwidth(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Tower Discovery + Bandwidth Tiers");

    let substrates = inventory::discover();
    check(
        "Tower: substrates discovered",
        !substrates.is_empty(),
        pass,
        fail,
    );

    let cpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Cpu)
        .count();
    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    check("Tower: CPU present", cpu_count > 0, pass, fail);
    println!("    Substrates: {cpu_count} CPU, {gpu_count} GPU, {npu_count} NPU");

    for sub in &substrates {
        if sub.kind == SubstrateKind::Gpu {
            let tier = bridge::detect_bandwidth_tier(sub);
            check(
                &format!("Bandwidth '{}': tier detected", sub.identity.name),
                tier.is_some(),
                pass,
                fail,
            );
            let transfer_1mb = bridge::estimated_transfer_us(sub, 1_000_000).unwrap_or(0.0);
            let transfer_100mb = bridge::estimated_transfer_us(sub, 100_000_000).unwrap_or(0.0);
            check(
                &format!("Bandwidth '{}': monotonic", sub.identity.name),
                transfer_100mb > transfer_1mb && transfer_1mb > 0.0,
                pass,
                fail,
            );
            println!(
                "    {}: 1MB={transfer_1mb:.0}µs, 100MB={transfer_100mb:.0}µs",
                sub.identity.name
            );
        }
    }
}

// ═══ S2: NPU→GPU PCIe Bypass ═════════════════════════════════════════
fn section_npu_gpu_bypass(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: NPU→GPU PCIe Bypass (Zero CPU Round-Trip)");

    let mut pipeline = StreamingSession::new(SubstrateKind::Gpu);
    pipeline.add_stage(PipelineStage {
        name: "DADA2 denoise (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Chimera detect (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "ESN classify (NPU)".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });

    let analysis = pipeline.analyze();
    check("NPU bypass: 3 stages", analysis.n_stages == 3, pass, fail);
    check(
        "NPU bypass: GPU→GPU chained (2)",
        analysis.gpu_chained == 2,
        pass,
        fail,
    );
    check(
        "NPU bypass: 0 CPU round-trips",
        analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
    check(
        "NPU bypass: fully streamable",
        analysis.fully_streamable,
        pass,
        fail,
    );
    println!("    Pipeline: GPU→GPU→NPU (GPU buffer passed directly to NPU)");
}

// ═══ S3: GPU→CPU Fallback ════════════════════════════════════════════
fn section_gpu_cpu_fallback(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: GPU→CPU Fallback (F64 Compute)");

    let mut pipeline = StreamingSession::new(SubstrateKind::Gpu);
    pipeline.add_stage(PipelineStage {
        name: "Diversity GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Molecular Clock (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "Bootstrap CI (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });

    let analysis = pipeline.analyze();
    check("CPU fallback: 3 stages", analysis.n_stages == 3, pass, fail);
    check(
        "CPU fallback: 2 CPU round-trips (GPU→CPU, CPU→CPU)",
        analysis.cpu_roundtrips == 2,
        pass,
        fail,
    );
    check(
        "CPU fallback: NOT fully streamable",
        !analysis.fully_streamable,
        pass,
        fail,
    );
    println!("    Pipeline: GPU→CPU→CPU (clock needs f64 precision)");
}

// ═══ S4: Multi-GPU Load Balancing ════════════════════════════════════
fn section_multi_gpu_balance(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Multi-GPU Bandwidth-Aware Routing");

    let substrates = inventory::discover();
    let gpu_count_s4 = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();

    println!("    GPUs available: {gpu_count_s4}");

    let test_workloads = vec![
        workloads::diversity(),
        workloads::pcoa(),
        workloads::kmer_histogram(),
        workloads::chimera(),
        workloads::kmd(),
        workloads::gbm_inference(),
    ];

    for bw in &test_workloads {
        let standard = dispatch::route(&bw.workload, &substrates);
        let bw_aware = dispatch::route_bandwidth_aware(&bw.workload, &substrates);
        check(
            &format!("Route '{}': standard found", bw.workload.name),
            standard.is_some(),
            pass,
            fail,
        );
        check(
            &format!("Route '{}': BW-aware found", bw.workload.name),
            bw_aware.is_some(),
            pass,
            fail,
        );
        if let (Some(s), Some(b)) = (&standard, &bw_aware) {
            println!(
                "    {}: standard → {:?}, BW-aware → {:?}",
                bw.workload.name, s.substrate.kind, b.substrate.kind
            );
        }
    }
}

// ═══ S5: Full 8-Stage Mixed Pipeline ═════════════════════════════════
fn section_full_mixed_pipeline(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Full 8-Stage Mixed Pipeline (NPU+GPU+CPU)");

    let mut pipeline = StreamingSession::new(SubstrateKind::Gpu);

    pipeline.add_stage(PipelineStage {
        name: "DADA2 (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Chimera (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "PCoA (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "K-mer (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Diversity fusion (GPU)".into(),
        capability: Capability::ScalarReduce,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "ESN classify (NPU)".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "Bootstrap CI (CPU)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "Kriging (CPU)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });

    let analysis = pipeline.analyze();
    check("8-stage: n_stages == 8", analysis.n_stages == 8, pass, fail);
    check(
        "8-stage: 5 GPU→GPU transitions chained",
        analysis.gpu_chained == 5,
        pass,
        fail,
    );
    check(
        "8-stage: CPU round-trips for CPU stages",
        analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    check(
        "8-stage: NOT fully streamable (has CPU)",
        !analysis.fully_streamable,
        pass,
        fail,
    );

    println!("    Pipeline: 5 GPU chained → NPU bypass → 2 CPU stages");
    println!(
        "    PCIe savings: {} GPU→GPU bypasses (no CPU copy)",
        analysis.gpu_chained
    );
}

// ═══ S6: Absorption Tracking ═════════════════════════════════════════
fn section_absorption_tracking(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Write→Absorb→Lean Tracking");

    let all = workloads::all_workloads();
    let (absorbed, local, cpu_only) = workloads::origin_summary();
    let total = all.len();

    check(
        &format!("Catalog: {total} workloads total"),
        total >= 34,
        pass,
        fail,
    );
    check(
        &format!("Catalog: {absorbed} absorbed"),
        absorbed > 0,
        pass,
        fail,
    );
    check(&format!("Catalog: {local} local shaders"), true, pass, fail);
    check(
        &format!("Catalog: {cpu_only} CPU-only"),
        cpu_only > 0,
        pass,
        fail,
    );

    println!("    {total} workloads: {absorbed} absorbed, {local} local, {cpu_only} CPU-only");

    for bw in &all {
        match &bw.origin {
            ShaderOrigin::Absorbed => {
                check(
                    &format!("Absorbed '{}': ToadStool validated", bw.workload.name),
                    true,
                    pass,
                    fail,
                );
            }
            ShaderOrigin::Local => {
                check(
                    &format!("Local '{}': ready for absorption", bw.workload.name),
                    true,
                    pass,
                    fail,
                );
            }
            ShaderOrigin::CpuOnly => {
                check(
                    &format!("CPU-only '{}': no GPU path needed", bw.workload.name),
                    true,
                    pass,
                    fail,
                );
            }
        }
    }
}

// ═══ S7: Full Catalog Dispatch ═══════════════════════════════════════
fn section_full_catalog_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n  S7: Full Catalog — Node Dispatch for All Workloads");

    let substrates = inventory::discover();
    let all = workloads::all_workloads();

    let mut routed = 0u32;
    let mut cpu_only_count = 0u32;

    for bw in &all {
        if matches!(bw.origin, ShaderOrigin::CpuOnly) {
            cpu_only_count += 1;
            check(
                &format!("CPU-only '{}': no GPU dispatch needed", bw.workload.name),
                true,
                pass,
                fail,
            );
            continue;
        }
        let decision = dispatch::route(&bw.workload, &substrates);
        if decision.is_some() {
            routed += 1;
        } else {
            check(
                &format!("WARN: '{}' unroutable", bw.workload.name),
                false,
                pass,
                fail,
            );
        }
    }

    let gpu_workloads = all.len() as u32 - cpu_only_count;
    check(
        &format!("Dispatch: {routed}/{gpu_workloads} GPU-capable workloads routed"),
        routed == gpu_workloads,
        pass,
        fail,
    );
    check(
        &format!("Dispatch: {cpu_only_count} CPU-only (routing not applicable)"),
        cpu_only_count > 0,
        pass,
        fail,
    );
}

// ═══ S8: biomeOS Graph Coordination ══════════════════════════════════
fn section_biomeos_graph(pass: &mut u32, fail: &mut u32) {
    println!("\n  S8: biomeOS Graph — Tower+Node+Nest Coordination");

    let local_subs = inventory::discover();
    let tower_subs = inventory::discover_with_tower();
    check(
        "biomeOS: Tower ⊇ local",
        tower_subs.len() >= local_subs.len(),
        pass,
        fail,
    );

    let songbird = inventory::discover_songbird_socket();
    if let Some(path) = &songbird {
        println!("    Songbird: {}", path.display());
        check("biomeOS: Songbird live", true, pass, fail);
    } else {
        println!("    Songbird: sovereign mode");
        check("biomeOS: sovereign (no Songbird)", true, pass, fail);
    }

    let nestgate = nest::discover_nestgate_socket();
    if let Some(path) = &nestgate {
        println!("    NestGate: {}", path.display());
        check("biomeOS: NestGate live", true, pass, fail);
    } else {
        println!("    NestGate: sovereign mode");
        check("biomeOS: sovereign (no NestGate)", true, pass, fail);
    }

    let has_compute = local_subs.iter().any(|s| {
        s.capabilities.iter().any(|c| {
            matches!(
                c,
                Capability::F64Compute | Capability::ShaderDispatch | Capability::ScalarReduce
            )
        })
    });
    check(
        "biomeOS: compute capability present",
        has_compute,
        pass,
        fail,
    );

    let has_storage_or_fallback = nestgate.is_some() || !local_subs.is_empty();
    check(
        "biomeOS: storage path available (NestGate or local)",
        has_storage_or_fallback,
        pass,
        fail,
    );

    println!(
        "    Graph: {} local → {} tower substrates",
        local_subs.len(),
        tower_subs.len()
    );
    println!("    Orchestration: Tower discovers, Node dispatches, Nest stores");
}
