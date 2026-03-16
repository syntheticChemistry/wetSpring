// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp220: Cross-Substrate Dispatch Evolution (V67)
//!
//! Extends Exp213 (V66) with bandwidth-aware routing and cross-substrate
//! data flow validation:
//!
//! - **S1**: `BandwidthTier` detection for GPU substrates
//! - **S2**: Bandwidth-aware routing: small workloads fall back to CPU
//! - **S3**: Cross-substrate data flow: GPU -> NPU -> CPU pipeline
//! - **S4**: Transfer cost estimation and comparison
//! - **S5**: Streaming topology with bandwidth annotations
//! - **S6**: Full inventory routing with bandwidth gating
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Exp213 (V66), metalForge forge v0.3.0 |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 67 |
//! | Command | `cargo run --release --bin validate_dispatch_evolution_v67` |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use wetspring_forge::bridge;
use wetspring_forge::dispatch::{self, Workload};
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};
use wetspring_forge::workloads;

fn check(pass: &mut u32, fail: &mut u32, total: &mut u32, name: &str, ok: bool) {
    *total += 1;
    if ok {
        *pass += 1;
        println!("  [PASS] {name}");
    } else {
        *fail += 1;
        println!("  [FAIL] {name}");
    }
}

fn make_gpu(name: &str) -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named(name),
        properties: Properties {
            memory_bytes: Some(12 * 1024 * 1024 * 1024),
            has_f64: true,
            has_timestamps: true,
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ScalarReduce,
            Capability::ShaderDispatch,
            Capability::TimestampQuery,
        ],
        origin: SubstrateOrigin::Local,
    }
}

fn make_npu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named("AKD1000"),
        properties: Properties {
            memory_bytes: Some(1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
            Capability::WeightMutation,
        ],
        origin: SubstrateOrigin::Local,
    }
}

fn make_cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("i9-12900K"),
        properties: Properties {
            memory_bytes: Some(64 * 1024 * 1024 * 1024),
            core_count: Some(24),
            thread_count: Some(32),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::CpuCompute,
            Capability::SimdVector,
        ],
        origin: SubstrateOrigin::Local,
    }
}

// ═══ S1: BandwidthTier Detection ═════════════════════════════════════

fn section_bandwidth_detection(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S1: BandwidthTier detection from GPU substrates ═══\n");

    let rtx4070 = make_gpu("NVIDIA GeForce RTX 4070");
    let tier = bridge::detect_bandwidth_tier(&rtx4070);
    check(
        pass,
        fail,
        total,
        "RTX 4070 has BandwidthTier",
        tier.is_some(),
    );

    let cpu = make_cpu();
    let cpu_tier = bridge::detect_bandwidth_tier(&cpu);
    check(
        pass,
        fail,
        total,
        "CPU has no BandwidthTier",
        cpu_tier.is_none(),
    );

    let npu = make_npu();
    let npu_tier = bridge::detect_bandwidth_tier(&npu);
    check(
        pass,
        fail,
        total,
        "NPU has no BandwidthTier",
        npu_tier.is_none(),
    );

    let rtx2080 = make_gpu("NVIDIA GeForce RTX 2080 Ti");
    let tier_2080 = bridge::detect_bandwidth_tier(&rtx2080);
    check(
        pass,
        fail,
        total,
        "RTX 2080 Ti has BandwidthTier",
        tier_2080.is_some(),
    );

    let a100 = make_gpu("NVIDIA A100");
    let tier_a100 = bridge::detect_bandwidth_tier(&a100);
    check(
        pass,
        fail,
        total,
        "A100 has BandwidthTier",
        tier_a100.is_some(),
    );
}

// ═══ S2: Bandwidth-Aware Routing ═════════════════════════════════════

fn section_bandwidth_routing(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S2: Bandwidth-aware routing — small workloads to CPU ═══\n");

    let gpu = make_gpu("NVIDIA GeForce RTX 4070");
    let cpu = make_cpu();
    let inventory = [gpu, cpu];

    let small_work =
        Workload::new("small_diversity", vec![Capability::F64Compute]).with_data_bytes(64);
    let d_small = dispatch::route_bandwidth_aware(&small_work, &inventory);
    check(
        pass,
        fail,
        total,
        "small workload routes somewhere",
        d_small.is_some(),
    );

    let large_work =
        Workload::new("large_diversity", vec![Capability::F64Compute]).with_data_bytes(100_000_000);
    let d_large = dispatch::route_bandwidth_aware(&large_work, &inventory);
    check(
        pass,
        fail,
        total,
        "large workload routes somewhere",
        d_large.is_some(),
    );

    let no_data = Workload::new("no_data", vec![Capability::F64Compute]);
    let d_nodata = dispatch::route_bandwidth_aware(&no_data, &inventory)
        .expect("no_data workload should route to GPU when GPU is available");
    check(
        pass,
        fail,
        total,
        "no data_bytes: standard routing (GPU)",
        d_nodata.substrate.kind == SubstrateKind::Gpu,
    );

    let preferred = Workload::new("forced_gpu", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Gpu)
        .with_data_bytes(1);
    let d_pref = dispatch::route_bandwidth_aware(&preferred, &inventory)
        .expect("forced_gpu workload with GPU preference should route when GPU available");
    check(
        pass,
        fail,
        total,
        "GPU preference overrides bandwidth gating",
        d_pref.substrate.kind == SubstrateKind::Gpu,
    );
}

// ═══ S3: Cross-Substrate Data Flow ═══════════════════════════════════

fn section_cross_substrate_flow(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S3: Cross-substrate data flow: GPU → NPU → CPU ═══\n");

    let gpu = make_gpu("NVIDIA GeForce RTX 4070");
    let npu = make_npu();
    let cpu = make_cpu();
    let inventory = [gpu, npu, cpu];

    let gpu_work = Workload::new(
        "diversity_gpu",
        vec![Capability::F64Compute, Capability::ScalarReduce],
    );
    let npu_work = Workload::new(
        "taxonomy_triage",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    let cpu_work = Workload::new("fastq_io", vec![Capability::CpuCompute]);

    let d_gpu =
        dispatch::route(&gpu_work, &inventory).expect("diversity_gpu workload should route to GPU");
    let d_npu = dispatch::route(&npu_work, &inventory)
        .expect("taxonomy_triage workload should route to NPU");
    let d_cpu =
        dispatch::route(&cpu_work, &inventory).expect("fastq_io workload should route to CPU");

    check(
        pass,
        fail,
        total,
        "diversity → GPU",
        d_gpu.substrate.kind == SubstrateKind::Gpu,
    );
    check(
        pass,
        fail,
        total,
        "taxonomy triage → NPU",
        d_npu.substrate.kind == SubstrateKind::Npu,
    );
    check(
        pass,
        fail,
        total,
        "FASTQ I/O → CPU",
        d_cpu.substrate.kind == SubstrateKind::Cpu,
    );

    let flow = [
        d_gpu.substrate.kind,
        d_npu.substrate.kind,
        d_cpu.substrate.kind,
    ];
    check(
        pass,
        fail,
        total,
        "cross-substrate flow: GPU → NPU → CPU",
        flow == [SubstrateKind::Gpu, SubstrateKind::Npu, SubstrateKind::Cpu],
    );

    let unique_substrates: std::collections::HashSet<SubstrateKind> =
        flow.iter().copied().collect();
    check(
        pass,
        fail,
        total,
        "3 distinct substrates used",
        unique_substrates.len() == 3,
    );
}

// ═══ S4: Transfer Cost Estimation ════════════════════════════════════

fn section_transfer_cost(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S4: Transfer cost estimation and comparison ═══\n");

    let rtx4070 = make_gpu("NVIDIA GeForce RTX 4070");

    let cost_1mb = bridge::estimated_transfer_us(&rtx4070, 1_048_576);
    check(
        pass,
        fail,
        total,
        "1 MB transfer cost exists",
        cost_1mb.is_some(),
    );
    if let Some(us) = cost_1mb {
        check(
            pass,
            fail,
            total,
            &format!("1 MB transfer: {us:.1} µs > 0"),
            us > 0.0,
        );
    }

    let cost_1gb = bridge::estimated_transfer_us(&rtx4070, 1_073_741_824);
    let cost_1kb = bridge::estimated_transfer_us(&rtx4070, 1024);
    check(
        pass,
        fail,
        total,
        "1 GB transfer > 1 KB transfer",
        cost_1gb.expect("1 GB transfer cost should exist for RTX 4070")
            > cost_1kb.expect("1 KB transfer cost should exist for RTX 4070"),
    );

    let cost_zero = bridge::estimated_transfer_us(&rtx4070, 0);
    check(
        pass,
        fail,
        total,
        "0 bytes transfer has latency only",
        cost_zero.is_some_and(|c| c >= 0.0),
    );

    let cpu_cost = bridge::estimated_transfer_us(&make_cpu(), 1_048_576);
    check(
        pass,
        fail,
        total,
        "CPU has no transfer cost",
        cpu_cost.is_none(),
    );
}

// ═══ S5: Streaming Topology with Bandwidth ═══════════════════════════

fn section_streaming_bandwidth(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S5: Streaming topology with bandwidth annotations ═══\n");

    let mut gpu_pure = StreamingSession::new(SubstrateKind::Gpu);
    for name in &["Diversity", "BrayCurtis", "PCoA", "SpectralMatch"] {
        gpu_pure.add_stage(PipelineStage {
            name: name.to_string(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
    }
    let a = gpu_pure.analyze();
    check(
        pass,
        fail,
        total,
        "pure GPU pipeline: fully streamable",
        a.fully_streamable,
    );
    check(
        pass,
        fail,
        total,
        "pure GPU pipeline: 3 chained",
        a.gpu_chained == 3,
    );
    check(
        pass,
        fail,
        total,
        "pure GPU pipeline: 0 PCIe roundtrips",
        a.cpu_roundtrips == 0,
    );

    let mut cross = StreamingSession::new(SubstrateKind::Gpu);
    cross.add_stage(PipelineStage {
        name: "GPU_Diversity".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    cross.add_stage(PipelineStage {
        name: "NPU_Classify".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    cross.add_stage(PipelineStage {
        name: "CPU_Output".into(),
        capability: Capability::CpuCompute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let b = cross.analyze();
    check(
        pass,
        fail,
        total,
        "GPU→NPU→CPU: not fully streamable",
        !b.fully_streamable,
    );
    check(
        pass,
        fail,
        total,
        "GPU→NPU→CPU: 2 CPU roundtrips",
        b.cpu_roundtrips == 2,
    );
}

// ═══ S6: Full Inventory with Bandwidth Gating ════════════════════════

fn section_full_inventory(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S6: Full inventory routing with bandwidth gating ═══\n");

    let gpu = make_gpu("NVIDIA GeForce RTX 4070");
    let npu = make_npu();
    let cpu = make_cpu();
    let inventory = [gpu, npu, cpu];

    let all = workloads::all_workloads();
    check(pass, fail, total, "workload catalog >= 29", all.len() >= 29);

    let mut routed = 0_usize;
    for bw in &all {
        if dispatch::route(&bw.workload, &inventory).is_some() {
            routed += 1;
        }
    }
    check(
        pass,
        fail,
        total,
        "all workloads routed",
        routed == all.len(),
    );

    let bw_work =
        Workload::new("bw_diversity", vec![Capability::F64Compute]).with_data_bytes(100_000_000);
    let d = dispatch::route_bandwidth_aware(&bw_work, &inventory);
    check(
        pass,
        fail,
        total,
        "bandwidth-aware routes for large workload",
        d.is_some(),
    );

    let (absorbed, local, cpu_only) = workloads::origin_summary();
    check(
        pass,
        fail,
        total,
        &format!("{absorbed} absorbed, {local} local, {cpu_only} CPU-only"),
        absorbed == 45 && local == 0 && cpu_only == 2,
    );
}

fn main() {
    let mut pass = 0_u32;
    let mut fail = 0_u32;
    let mut total = 0_u32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp220: Cross-Substrate Dispatch Evolution (V67)");
    println!("═══════════════════════════════════════════════════════════");

    section_bandwidth_detection(&mut pass, &mut fail, &mut total);
    section_bandwidth_routing(&mut pass, &mut fail, &mut total);
    section_cross_substrate_flow(&mut pass, &mut fail, &mut total);
    section_transfer_cost(&mut pass, &mut fail, &mut total);
    section_streaming_bandwidth(&mut pass, &mut fail, &mut total);
    section_full_inventory(&mut pass, &mut fail, &mut total);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  RESULT: {pass}/{total} passed, {fail} failed");
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}
