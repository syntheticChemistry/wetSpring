// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems
//!
//! Comprehensive cross-spring evolution benchmark and validation on ToadStool S87.
//! Tracks shader provenance through the ecosystem: when and where each primitive
//! was written, who absorbed it, and which springs now consume it.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cross_spring_evolution_s87` |
//!
//! Validation class: Cross-spring
//!
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

mod spring_core;
mod spring_extended;

use barracuda::shaders::Precision;
use crate::gpu::GpuF64;
use crate::validation::timing::BenchRowEvolved;
use crate::validation::{self, OrExit, Validator, bench_print};

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    bench_print(label, f)
}

/// Run the `validate_cross_spring_evolution_s87` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let mut timings: Vec<BenchRowEvolved> = Vec::new();

    println!("ToadStool pin: S87 (2dc26792) — 264 ComputeDispatch ops, 144 consumed by wetSpring");
    println!(
        "S87 highlights: FHE shader fixes, gpu_helpers refactor, device-lost recovery, unsafe audit"
    );
    println!(
        "Cross-spring: hotSpring + wetSpring + neuralSpring + airSpring + groundSpring + wateringHole\n"
    );

    // §0: GPU Init + hotSpring Precision Architecture
    v.section("§0 GPU Init + hotSpring Precision Architecture (S87)");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .or_exit("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");

    let strategy = gpu.fp64_strategy();
    let precision = gpu.optimal_precision();
    let is_lost = gpu.is_lost();
    let threshold = gpu.dispatch_threshold();
    let caps = gpu.capabilities();
    let exp_workaround = caps.needs_exp_f64_workaround();
    let log_workaround = caps.needs_log_f64_workaround();

    println!("  GPU: {}", gpu.adapter_name);
    println!("  Fp64Strategy: {strategy:?}");
    println!("    Written: hotSpring v0.4.0 (Feb 2026)");
    println!("    Absorbed: ToadStool S58 (Feb 24)");
    println!("    Evolved: S67 (auto-detect), S80 (NVK workarounds), S87 (device-lost recovery)");
    println!("    Consumed by: ALL springs (precision layer is universal)");
    println!("  Precision: {precision:?}");
    println!("  is_lost: {is_lost}, dispatch_threshold: {threshold}");
    println!("  NVK exp workaround: {exp_workaround}, log: {log_workaround}");

    v.check_pass("GPU initialized", true);
    v.check_pass("device not lost (S87 recovery)", !is_lost);
    v.check_pass(
        "Fp64Strategy detected",
        matches!(
            strategy,
            barracuda::device::Fp64Strategy::Native | barracuda::device::Fp64Strategy::Hybrid
        ),
    );
    v.check_pass(
        "Precision F64 or Df64",
        matches!(precision, Precision::F64 | Precision::Df64),
    );

    let device = gpu.to_wgpu_device();

    // §1-§5: Core cross-spring sections
    spring_core::validate(v, &gpu, &device, &mut timings);

    // §6-§12: Extended sections + benchmarks
    spring_extended::validate(v, &mut timings);

    // Summary
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems        ║");
    println!("║                                                                        ║");
    println!("║  ToadStool S87 (2dc26792) — 264 ComputeDispatch ops, 144 by wetSpring ║");
    println!("║  S87: FHE fixes + gpu_helpers refactor + device-lost + unsafe audit    ║");
    println!("║                                                                        ║");
    println!("║  Cross-Spring Shader Evolution (when → where → who benefits):          ║");
    println!("║   hotSpring  → DF64 (S58), spectral (S26), grid (S40), NVK (S80)      ║");
    println!("║   wetSpring  → Bio diversity (S63), ODE (S58), alignment (S31)         ║");
    println!("║   neuralSpring → GEMM (S64), graph (S54), pairwise (S27)              ║");
    println!("║   airSpring  → Hydrology (S70/S81), Richards PDE (S83)                ║");
    println!("║   groundSpring → Bootstrap (S70), evolution (S70), topology (S81)     ║");
    println!("║   wateringHole → Boltzmann (S76), Brent/L-BFGS (S83)                  ║");
    println!("║                                                                        ║");
    println!("║  Key compositions:                                                     ║");
    println!("║   wetSpring NMF = wetSpring bio × neuralSpring GEMM × hotSpring DF64  ║");
    println!("║   wetSpring PCoA = wetSpring BC × neuralSpring Eigh × hotSpring prec  ║");
    println!("║   All springs benefit from hotSpring precision layer (universal)        ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_cross_spring_evolution_s87");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "cross_spring_evolution_s87",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Both,
        provenance_crate: "validate_cross_spring_evolution_s87",
        provenance_date: "2026-05-20",
        description: "# Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems",
    },
    run: |v, _ctx| run_as_scenario(v),
};
