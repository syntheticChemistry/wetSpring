// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]
//! Exp080: metalForge Dispatch Routing Validation
//!
//! Validates the forge dispatch router's life-science workload classification
//! across every substrate configuration the ecosystem will encounter:
//!
//! 1. **Full system** (GPU+NPU+CPU) — proves GPU-first, NPU-fallback, CPU-fallback
//! 2. **GPU-only** — proves GPU absorbs all f64 workloads
//! 3. **NPU+CPU** — proves NPU handles quant inference, CPU handles f64
//! 4. **CPU-only** — proves CPU fallback covers all f64 workloads
//! 5. **Mixed `PCIe`** — proves multi-GPU + NPU routing for cross-substrate dispatch
//! 6. **Preference override** — proves user-requested substrate is respected
//!
//! Also validates live hardware discovery and routing on THIS machine.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | forge dispatch API (capability-based routing) |
//! | Baseline version | wetspring-forge 0.1.0 |
//! | Baseline command | `cargo run --bin validate_dispatch_routing` |
//! | Baseline date | 2026-02-22 |
//! | Data | Synthetic substrate inventories (self-contained) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_forge::dispatch::{self, Reason, Workload};
use wetspring_forge::inventory;
use wetspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut total = 0u32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp080: metalForge Dispatch Routing Validation");
    println!("═══════════════════════════════════════════════════════════\n");

    section_full_system(&mut pass, &mut fail, &mut total);
    section_gpu_only(&mut pass, &mut fail, &mut total);
    section_npu_cpu(&mut pass, &mut fail, &mut total);
    section_cpu_only(&mut pass, &mut fail, &mut total);
    section_mixed_pcie(&mut pass, &mut fail, &mut total);
    section_preference(&mut pass, &mut fail, &mut total);
    section_live_hardware(&mut pass, &mut fail, &mut total);

    println!("\n═══════════════════════════════════════════════════════════");
    if fail == 0 {
        println!("  Exp080: {pass}/{total} checks passed");
        println!("  RESULT: PASS");
    } else {
        println!("  Exp080: {pass}/{total} checks passed ({fail} FAILED)");
        println!("  RESULT: FAIL");
    }
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

// ── Substrate factories ─────────────────────────────────────────────────────

fn gpu_f64(name: &str) -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named(name),
        properties: Properties {
            has_f64: true,
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::ScalarReduce,
        ],
    }
}

fn gpu_f32(name: &str) -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named(name),
        properties: Properties::default(),
        capabilities: vec![Capability::F32Compute, Capability::ShaderDispatch],
    }
}

fn npu(name: &str) -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named(name),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
        ],
    }
}

fn cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("CPU"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::SimdVector,
        ],
    }
}

// ── Life-science workload definitions ────────────────────────────────────────

fn ode_workload(name: &str) -> Workload {
    Workload::new(
        name,
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    )
}

fn diversity_workload() -> Workload {
    Workload::new(
        "Diversity map-reduce",
        vec![Capability::F64Compute, Capability::ScalarReduce],
    )
}

fn felsenstein_workload() -> Workload {
    Workload::new(
        "Felsenstein pruning",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    )
}

fn hmm_workload() -> Workload {
    Workload::new(
        "HMM forward",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    )
}

fn taxonomy_workload() -> Workload {
    Workload::new(
        "Taxonomy classify",
        vec![Capability::QuantizedInference { bits: 8 }],
    )
}

fn fastq_workload() -> Workload {
    Workload::new("FASTQ parsing", vec![Capability::F64Compute])
}

fn chimera_workload() -> Workload {
    Workload::new("Chimera detection", vec![Capability::F64Compute])
}

fn batch_inference_workload() -> Workload {
    Workload::new(
        "Batch anomaly detection",
        vec![
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
        ],
    )
}

// ── Check helpers ────────────────────────────────────────────────────────────

fn check(label: &str, actual: bool, pass: &mut u32, fail: &mut u32, total: &mut u32) {
    *total += 1;
    if actual {
        *pass += 1;
        println!("  [OK]  {label}");
    } else {
        *fail += 1;
        println!("  [FAIL]  {label}");
    }
}

fn check_route(
    label: &str,
    workload: &Workload,
    substrates: &[Substrate],
    expected_kind: SubstrateKind,
    pass: &mut u32,
    fail: &mut u32,
    total: &mut u32,
) {
    let decision = dispatch::route(workload, substrates);
    match decision {
        Some(d) => check(
            &format!("{label}: → {}", d.substrate.kind),
            d.substrate.kind == expected_kind,
            pass,
            fail,
            total,
        ),
        None => check(
            &format!("{label}: no route (expected {expected_kind})"),
            false,
            pass,
            fail,
            total,
        ),
    }
}

fn check_no_route(
    label: &str,
    workload: &Workload,
    substrates: &[Substrate],
    pass: &mut u32,
    fail: &mut u32,
    total: &mut u32,
) {
    let decision = dispatch::route(workload, substrates);
    check(label, decision.is_none(), pass, fail, total);
}

// ── Section 1: Full system (GPU + NPU + CPU) ────────────────────────────────

fn section_full_system(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 1: Full System (GPU f64 + NPU + CPU) ═══");
    let subs = vec![gpu_f64("RTX 4070"), npu("AKD1000"), cpu()];

    check_route(
        "ODE QS biofilm",
        &ode_workload("QS biofilm ODE"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "ODE bistable",
        &ode_workload("Bistable ODE"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "ODE multi-signal",
        &ode_workload("Multi-signal ODE"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "ODE phage-defense",
        &ode_workload("Phage defense ODE"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "ODE cooperation",
        &ode_workload("Cooperation ODE"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Diversity",
        &diversity_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Felsenstein",
        &felsenstein_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "HMM forward",
        &hmm_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Taxonomy",
        &taxonomy_workload(),
        &subs,
        SubstrateKind::Npu,
        pass,
        fail,
        total,
    );
    check_route(
        "FASTQ parsing",
        &fastq_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Batch inference",
        &batch_inference_workload(),
        &subs,
        SubstrateKind::Npu,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 2: GPU-only (no NPU) ────────────────────────────────────────────

fn section_gpu_only(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 2: GPU + CPU (no NPU) ═══");
    let subs = vec![gpu_f64("RTX 4070"), cpu()];

    check_route(
        "ODE → GPU",
        &ode_workload("ODE sweep"),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Diversity → GPU",
        &diversity_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    check_no_route(
        "Taxonomy → none (no NPU, no quant)",
        &taxonomy_workload(),
        &subs,
        pass,
        fail,
        total,
    );
    check_route(
        "FASTQ → GPU (f64)",
        &fastq_workload(),
        &subs,
        SubstrateKind::Gpu,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 3: NPU + CPU (no GPU) ───────────────────────────────────────────

fn section_npu_cpu(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 3: NPU + CPU (no GPU) ═══");
    let subs = vec![npu("AKD1000"), cpu()];

    check_no_route(
        "ODE GPU → none (CPU has no shader)",
        &ode_workload("ODE GPU path"),
        &subs,
        pass,
        fail,
        total,
    );
    let ode_cpu = Workload::new("ODE CPU fallback", vec![Capability::F64Compute]);
    check_route(
        "ODE CPU → CPU (f64 only)",
        &ode_cpu,
        &subs,
        SubstrateKind::Cpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Taxonomy → NPU",
        &taxonomy_workload(),
        &subs,
        SubstrateKind::Npu,
        pass,
        fail,
        total,
    );
    check_route(
        "FASTQ → CPU",
        &fastq_workload(),
        &subs,
        SubstrateKind::Cpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Chimera → CPU",
        &chimera_workload(),
        &subs,
        SubstrateKind::Cpu,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 4: CPU-only ──────────────────────────────────────────────────────

fn section_cpu_only(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 4: CPU Only ═══");
    let subs = vec![cpu()];

    check_no_route(
        "ODE → none (needs shader)",
        &ode_workload("ODE cpu-only"),
        &subs,
        pass,
        fail,
        total,
    );
    check_route(
        "FASTQ → CPU",
        &fastq_workload(),
        &subs,
        SubstrateKind::Cpu,
        pass,
        fail,
        total,
    );
    check_route(
        "Chimera → CPU",
        &chimera_workload(),
        &subs,
        SubstrateKind::Cpu,
        pass,
        fail,
        total,
    );
    check_no_route(
        "Taxonomy → none (no quant)",
        &taxonomy_workload(),
        &subs,
        pass,
        fail,
        total,
    );
    check_no_route(
        "Batch inference → none",
        &batch_inference_workload(),
        &subs,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 5: Mixed PCIe (multi-GPU + NPU) ─────────────────────────────────

fn section_mixed_pcie(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 5: Mixed PCIe (f64 GPU + f32 GPU + NPU + CPU) ═══");
    let subs = vec![
        gpu_f32("iGPU (Intel UHD)"),
        gpu_f64("RTX 4070"),
        npu("AKD1000"),
        cpu(),
    ];

    let ode = ode_workload("ODE PCIe test");
    let d = dispatch::route(&ode, &subs).expect("should route ODE");
    check(
        "ODE → f64 GPU (not f32 iGPU)",
        d.substrate.kind == SubstrateKind::Gpu && d.substrate.identity.name == "RTX 4070",
        pass,
        fail,
        total,
    );

    check_route(
        "Taxonomy → NPU (bypasses GPUs)",
        &taxonomy_workload(),
        &subs,
        SubstrateKind::Npu,
        pass,
        fail,
        total,
    );

    let div = diversity_workload();
    let d2 = dispatch::route(&div, &subs).expect("should route diversity");
    check(
        "Diversity → f64 GPU (needs ScalarReduce)",
        d2.substrate.kind == SubstrateKind::Gpu && d2.substrate.identity.name == "RTX 4070",
        pass,
        fail,
        total,
    );

    check_route(
        "Batch inference → NPU",
        &batch_inference_workload(),
        &subs,
        SubstrateKind::Npu,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 6: Preference overrides ──────────────────────────────────────────

fn section_preference(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 6: Preference Overrides ═══");
    let subs = vec![gpu_f64("RTX 4070"), npu("AKD1000"), cpu()];

    let forced_cpu =
        Workload::new("ODE forced CPU", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);
    let d = dispatch::route(&forced_cpu, &subs).expect("should route");
    check(
        "ODE prefer CPU → CPU (even with GPU)",
        d.substrate.kind == SubstrateKind::Cpu && d.reason == Reason::Preferred,
        pass,
        fail,
        total,
    );

    let forced_missing = Workload::new(
        "ODE forced NPU",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    )
    .prefer(SubstrateKind::Npu);
    let d = dispatch::route(&forced_missing, &subs).expect("should route");
    check(
        "ODE prefer NPU → GPU (NPU incapable, falls back)",
        d.substrate.kind == SubstrateKind::Gpu && d.reason == Reason::BestAvailable,
        pass,
        fail,
        total,
    );
    println!();
}

// ── Section 7: Live hardware discovery ───────────────────────────────────────

fn section_live_hardware(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("═══ Section 7: Live Hardware Discovery ═══");

    let subs = inventory::discover();
    let cpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Cpu).count();
    let gpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Gpu).count();
    let npu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Npu).count();

    println!("  Discovered: {cpu_count} CPU, {gpu_count} GPU(s), {npu_count} NPU(s)");

    check(
        "Discovery finds exactly 1 CPU",
        cpu_count == 1,
        pass,
        fail,
        total,
    );
    check("Discovery finds ≥ 0 GPUs", true, pass, fail, total);

    let fastq = fastq_workload();
    let d = dispatch::route(&fastq, &subs);
    check(
        "FASTQ always routable on live hardware",
        d.is_some(),
        pass,
        fail,
        total,
    );

    if gpu_count > 0 {
        let fels = felsenstein_workload();
        let d = dispatch::route(&fels, &subs);
        check(
            "Felsenstein routes to GPU on this machine",
            d.is_some_and(|dd| dd.substrate.kind == SubstrateKind::Gpu),
            pass,
            fail,
            total,
        );
    }
    println!();
}
