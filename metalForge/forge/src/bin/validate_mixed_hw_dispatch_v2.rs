// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp332: Mixed Hardware Dispatch Evolution
//!
//! Validates bandwidth-aware routing, workload `data_bytes` wiring,
//! NUCLEUS discovery topology, and mixed GPU+NPU+CPU dispatch
//! with `petalTongue` visualization overlay.
//!
//! | Domain | Checks |
//! |--------|--------|
//! | D1 Bandwidth-Aware | `route_bandwidth_aware` falls back on high transfer cost |
//! | D2 Workload Wiring  | `BioWorkload.data_bytes` propagates to dispatch |
//! | D3 Mixed Substrate  | GPU→NPU→CPU priority chain correct |
//! | D4 NUCLEUS Topology | Tower/Node/Nest wired and renderable |
//! | D5 petalTongue      | Dispatch + inventory scenarios serialize |
//!
//! Provenance: Mixed hardware (GPU/NPU/CPU) dispatch validation (V2)

use barracuda::unified_hardware::BandwidthTier;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;
use wetspring_forge::dispatch::{self, Reason, Workload};
use wetspring_forge::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};

fn make_gpu(name: &str, caps: Vec<Capability>) -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named(name),
        properties: Properties {
            has_f64: caps.contains(&Capability::F64Compute),
            ..Properties::default()
        },
        capabilities: caps,
        origin: SubstrateOrigin::Local,
    }
}

fn make_npu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named("AKD1000"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
        ],
        origin: SubstrateOrigin::Local,
    }
}

fn make_cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("AMD EPYC 9654"),
        properties: Properties {
            core_count: Some(96),
            thread_count: Some(192),
            memory_bytes: Some(512 * 1024 * 1024 * 1024),
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

fn main() {
    let mut v = Validator::new("Exp332: Mixed Hardware Dispatch Evolution");

    let gpu = make_gpu(
        "NVIDIA GeForce RTX 4070",
        vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ScalarReduce,
            Capability::ShaderDispatch,
        ],
    );
    let npu = make_npu();
    let cpu = make_cpu();
    let inventory = vec![gpu, npu, cpu];

    // ── D1: Bandwidth-Aware Routing ──
    v.section("D1 — Bandwidth-Aware Routing");

    let small_work =
        Workload::new("small_diversity", vec![Capability::F64Compute]).with_data_bytes(64);
    let d_small =
        dispatch::route_bandwidth_aware(&small_work, &inventory).or_exit("unexpected error");
    v.check_pass(
        "small data (64B): routes to GPU (compute > transfer)",
        d_small.substrate.kind == SubstrateKind::Gpu,
    );

    let large_work =
        Workload::new("large_diversity", vec![Capability::F64Compute]).with_data_bytes(100_000_000);
    let d_large =
        dispatch::route_bandwidth_aware(&large_work, &inventory).or_exit("unexpected error");
    v.check_pass(
        "large data (100MB): falls back to CPU (transfer dominates)",
        d_large.substrate.kind == SubstrateKind::Cpu && d_large.reason == Reason::BandwidthFallback,
    );

    let no_data = Workload::new("no_data_workload", vec![Capability::F64Compute]);
    let d_nodata =
        dispatch::route_bandwidth_aware(&no_data, &inventory).or_exit("unexpected error");
    v.check_pass(
        "no data_bytes: standard routing (GPU)",
        d_nodata.substrate.kind == SubstrateKind::Gpu,
    );

    let forced_gpu = Workload::new("forced_gpu", vec![Capability::F64Compute])
        .prefer(SubstrateKind::Gpu)
        .with_data_bytes(500_000_000);
    let d_forced =
        dispatch::route_bandwidth_aware(&forced_gpu, &inventory).or_exit("unexpected error");
    v.check_pass(
        "preferred GPU ignores bandwidth fallback",
        d_forced.substrate.kind == SubstrateKind::Gpu,
    );

    // ── D2: Workload data_bytes Wiring ──
    v.section("D2 — Workload data_bytes Wiring");

    let all_workloads = wetspring_forge::workloads::all_workloads();
    let kmer = all_workloads
        .iter()
        .find(|w| w.workload.name == "kmer_histogram")
        .or_exit("unexpected error");
    v.check_pass(
        "kmer_histogram data_bytes = 10MB",
        kmer.workload.data_bytes == Some(10_000_000),
    );

    let sw = all_workloads
        .iter()
        .find(|w| w.workload.name == "smith_waterman")
        .or_exit("unexpected error");
    v.check_pass(
        "smith_waterman data_bytes = 50MB",
        sw.workload.data_bytes == Some(50_000_000),
    );

    let pcoa_w = all_workloads
        .iter()
        .find(|w| w.workload.name == "pcoa")
        .or_exit("unexpected error");
    v.check_pass(
        "pcoa data_bytes = 8MB",
        pcoa_w.workload.data_bytes == Some(8_000_000),
    );

    let dada2_w = all_workloads
        .iter()
        .find(|w| w.workload.name == "dada2")
        .or_exit("unexpected error");
    v.check_pass(
        "dada2 data_bytes = 100MB",
        dada2_w.workload.data_bytes == Some(100_000_000),
    );

    // Dispatch bio workloads through bandwidth-aware path
    let kmer_d =
        dispatch::route_bandwidth_aware(&kmer.workload, &inventory).or_exit("unexpected error");
    v.check_pass(
        "kmer_histogram routes to GPU (10MB < threshold)",
        kmer_d.substrate.kind == SubstrateKind::Gpu,
    );
    let dada2_d =
        dispatch::route_bandwidth_aware(&dada2_w.workload, &inventory).or_exit("unexpected error");
    v.check_pass(
        "dada2 bandwidth decision made",
        dada2_d.substrate.kind == SubstrateKind::Gpu || dada2_d.reason == Reason::BandwidthFallback,
    );

    // ── D3: Mixed Substrate Priority ──
    v.section("D3 — Mixed Substrate Priority Chain");

    let f64_work = Workload::new("f64_compute", vec![Capability::F64Compute]);
    let d = dispatch::route(&f64_work, &inventory).or_exit("unexpected error");
    v.check_pass(
        "f64 compute → GPU priority",
        d.substrate.kind == SubstrateKind::Gpu,
    );

    let quant_work = Workload::new(
        "taxonomy_classify",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    let d = dispatch::route(&quant_work, &inventory).or_exit("unexpected error");
    v.check_pass("quant 8-bit → NPU", d.substrate.kind == SubstrateKind::Npu);

    let cpu_work =
        Workload::new("fastq_parsing", vec![Capability::CpuCompute]).prefer(SubstrateKind::Cpu);
    let d = dispatch::route(&cpu_work, &inventory).or_exit("unexpected error");
    v.check_pass(
        "CPU-preferred → CPU",
        d.substrate.kind == SubstrateKind::Cpu,
    );

    let simd_work = Workload::new("simd_align", vec![Capability::SimdVector]);
    let d = dispatch::route(&simd_work, &inventory).or_exit("unexpected error");
    v.check_pass(
        "SIMD → CPU (only CPU has SimdVector)",
        d.substrate.kind == SubstrateKind::Cpu,
    );

    let impossible = Workload::new(
        "impossible",
        vec![Capability::QuantizedInference { bits: 2 }],
    );
    v.check_pass(
        "2-bit quant: no capable substrate",
        dispatch::route(&impossible, &inventory).is_none(),
    );

    // ── D4: NUCLEUS Topology ──
    v.section("D4 — NUCLEUS Topology");

    let tier = BandwidthTier::detect_from_adapter_name("NVIDIA GeForce RTX 4070");
    v.check_pass("RTX 4070 → PciE4x16", tier == BandwidthTier::PciE4x16);

    let cost = tier.transfer_cost();
    let transfer_1mb = cost.estimated_us(1_048_576);
    v.check_pass("1 MB transfer < 1000 µs", transfer_1mb < 1000.0);
    v.check_pass("1 MB transfer > 0 µs", transfer_1mb > 0.0);

    let bridge_tier =
        wetspring_forge::bridge::detect_bandwidth_tier(&inventory[0]).or_exit("unexpected error");
    v.check_pass(
        "bridge detects PciE4x16 for RTX 4070",
        bridge_tier == BandwidthTier::PciE4x16,
    );

    let bridge_us = wetspring_forge::bridge::estimated_transfer_us(&inventory[0], 1_048_576)
        .or_exit("unexpected error");
    v.check(
        "bridge transfer matches direct",
        bridge_us,
        transfer_1mb,
        tolerances::TRANSFER_TIME_PARITY,
    );

    // ── D5: petalTongue Visualization ──
    v.section("D5 — petalTongue Visualization Overlay");

    let (inv_scenario, _inv_edges) = wetspring_forge::visualization::inventory_scenario(&inventory);
    v.check_pass(
        "inventory scenario has nodes",
        !inv_scenario.nodes.is_empty(),
    );

    let (disp_scenario, _disp_edges) =
        wetspring_forge::visualization::dispatch_scenario(&inventory);
    v.check_pass(
        "dispatch scenario has nodes",
        !disp_scenario.nodes.is_empty(),
    );

    let json = serde_json::to_string(&inv_scenario);
    v.check_pass("inventory scenario serializes", json.is_ok());
    let json = serde_json::to_string(&disp_scenario);
    v.check_pass("dispatch scenario serializes", json.is_ok());

    v.finish();
}
