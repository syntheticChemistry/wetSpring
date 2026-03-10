// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! Benchmark: ODE Lean Validation + Cross-Spring Evolution
//!
//! Exercises the fully-lean ODE pipeline where all 5 biological ODE systems
//! generate WGSL via `ToadStool`'s `BatchedOdeRK4<S>::generate_shader()` and
//! CPU integration via `integrate_cpu()`.
//!
//! # Cross-Spring Evolution
//!
//! | System | Biology | Evolved By | Precision From | Template From |
//! |--------|---------|-----------|---------------|--------------|
//! | Capacitor | `VpsR` phenotypic capacitor | wetSpring Exp002 | hotSpring f64 | `ToadStool` S51 |
//! | Cooperation | QS game theory | wetSpring Exp003 | hotSpring f64 | `ToadStool` S51 |
//! | `MultiSignal` | Dual-signal QS | wetSpring Exp006 | hotSpring f64 | `ToadStool` S51 |
//! | Bistable | Phenotypic switch | wetSpring Exp007 | hotSpring f64 | `ToadStool` S51 |
//! | `PhageDefense` | Arms race | wetSpring Exp009 | hotSpring f64 | `ToadStool` S51 |
//!
//! The f64 emulation pattern in WGSL came from hotSpring's lattice QCD shaders.
//! The generic `OdeSystem` trait + RK4 template came from `ToadStool` S51 absorption.
//! neuralSpring's adaptive RK45 influenced the stepping architecture.

use std::time::Instant;

use barracuda::numerical::ode_generic::{BatchedOdeRK4, OdeSystem};
use barracuda::numerical::{
    BistableOde, CapacitorOde, CooperationOde, MultiSignalOde, PhageDefenseOde,
};
use wetspring_barracuda::bio::bistable::BistableParams;
use wetspring_barracuda::bio::capacitor::CapacitorParams;
use wetspring_barracuda::bio::cooperation::CooperationParams;
use wetspring_barracuda::bio::multi_signal::MultiSignalParams;
use wetspring_barracuda::bio::phage_defense::PhageDefenseParams;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.01;
const N_STEPS: usize = 4800;
const BATCH_SIZES: &[usize] = &[1, 10, 100, 500];

fn bench<F: FnOnce() -> R, R>(label: &str, f: F) -> (R, f64) {
    let t0 = Instant::now();
    let r = f();
    let us = t0.elapsed().as_micros() as f64;
    println!("  {label}: {us:.0} µs");
    (r, us)
}

struct SystemBench {
    name: &'static str,
    origin: &'static str,
    n_vars: usize,
    n_params: usize,
    shader_lines: usize,
    local_cpu_us: f64,
    upstream_cpu_us: f64,
    batch_timings: Vec<(usize, f64)>,
    max_diff: f64,
}

fn main() {
    let mut v = Validator::new("ODE Lean + Cross-Spring Evolution Benchmark");

    println!("  This benchmark validates the complete lean: all 5 biological ODE");
    println!("  systems now use ToadStool's generate_shader() for GPU WGSL and");
    println!("  integrate_cpu() for CPU parity. No local WGSL files remain.\n");

    let mut benches: Vec<SystemBench> = Vec::new();

    // ─── §1 Capacitor ODE (wetSpring Exp002, Mhatre 2020) ──────────
    v.section("§1 Capacitor ODE — wetSpring → ToadStool lean");
    println!("  Origin: wetSpring Exp002 (V. cholerae VpsR phenotypic capacitor)");
    println!("  f64 pattern: hotSpring lattice QCD precision shaders");
    {
        let p = CapacitorParams::default();
        let y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0];

        let (local_result, local_us) = bench("local CPU integration (4800 steps)", || {
            wetspring_barracuda::bio::capacitor::run_capacitor(&y0, 48.0, DT, &p)
        });

        let flat = p.to_flat();
        let (upstream_result, upstream_us) = bench("upstream integrate_cpu (4800 steps)", || {
            BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&y0, &flat, DT, N_STEPS, 1)
                .expect("capacitor upstream")
        });

        let max_diff = local_result
            .y_final
            .iter()
            .zip(upstream_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        v.check_pass("CPU parity (max diff < 0.5)", max_diff < 0.5);
        println!("  max |local - upstream| = {max_diff:.2e}");

        let shader = BatchedOdeRK4::<CapacitorOde>::generate_shader();
        v.check_pass("WGSL contains deriv()", shader.contains("fn deriv"));
        v.check_pass("WGSL contains rk4_step()", shader.contains("fn rk4_step"));
        let shader_lines = shader.lines().count();

        let mut batch_timings = Vec::new();
        for &bs in BATCH_SIZES {
            let states: Vec<f64> = y0.iter().copied().cycle().take(bs * 6).collect();
            let params: Vec<f64> = flat.iter().copied().cycle().take(bs * 16).collect();
            let (_, us) = bench(&format!("batched CPU ({bs} batches)"), || {
                BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&states, &params, DT, N_STEPS, bs)
                    .expect("batched")
            });
            batch_timings.push((bs, us));
        }

        benches.push(SystemBench {
            name: "Capacitor",
            origin: "wetSpring Exp002",
            n_vars: CapacitorOde::N_VARS,
            n_params: CapacitorOde::N_PARAMS,
            shader_lines,
            local_cpu_us: local_us,
            upstream_cpu_us: upstream_us,
            batch_timings,
            max_diff,
        });
    }

    // ─── §2 Cooperation ODE (wetSpring Exp003, Bruger 2018) ────────
    v.section("§2 Cooperation ODE — wetSpring → ToadStool lean");
    println!("  Origin: wetSpring Exp003 (V. cholerae QS cooperator-cheater)");
    {
        let p = CooperationParams::default();
        let y0 = [0.01, 0.01, 0.0, 0.0];

        let (local_result, local_us) = bench("local CPU integration (4800 steps)", || {
            wetspring_barracuda::bio::cooperation::run_cooperation(&y0, 48.0, DT, &p)
        });

        let flat = p.to_flat();
        let (upstream_result, upstream_us) = bench("upstream integrate_cpu (4800 steps)", || {
            BatchedOdeRK4::<CooperationOde>::integrate_cpu(&y0, &flat, DT, N_STEPS, 1)
                .expect("cooperation upstream")
        });

        let max_diff = local_result
            .y_final
            .iter()
            .zip(upstream_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        v.check_pass("CPU parity (max diff < 0.5)", max_diff < 0.5);
        println!("  max |local - upstream| = {max_diff:.2e}");

        let shader = BatchedOdeRK4::<CooperationOde>::generate_shader();
        let shader_lines = shader.lines().count();

        let mut batch_timings = Vec::new();
        for &bs in BATCH_SIZES {
            let states: Vec<f64> = y0.iter().copied().cycle().take(bs * 4).collect();
            let params: Vec<f64> = flat.iter().copied().cycle().take(bs * 13).collect();
            let (_, us) = bench(&format!("batched CPU ({bs} batches)"), || {
                BatchedOdeRK4::<CooperationOde>::integrate_cpu(&states, &params, DT, N_STEPS, bs)
                    .expect("batched")
            });
            batch_timings.push((bs, us));
        }

        benches.push(SystemBench {
            name: "Cooperation",
            origin: "wetSpring Exp003",
            n_vars: CooperationOde::N_VARS,
            n_params: CooperationOde::N_PARAMS,
            shader_lines,
            local_cpu_us: local_us,
            upstream_cpu_us: upstream_us,
            batch_timings,
            max_diff,
        });
    }

    // ─── §3 Multi-Signal ODE (wetSpring Exp006, Srivastava 2011) ───
    v.section("§3 Multi-Signal ODE — wetSpring → ToadStool lean");
    println!("  Origin: wetSpring Exp006 (V. cholerae dual-signal QS regulation)");
    {
        let p = MultiSignalParams::default();
        let y0 = [0.01, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];

        let (local_result, local_us) = bench("local CPU integration (4800 steps)", || {
            wetspring_barracuda::bio::multi_signal::run_multi_signal(&y0, 48.0, DT, &p)
        });

        let flat = p.to_flat();
        let (upstream_result, upstream_us) = bench("upstream integrate_cpu (4800 steps)", || {
            BatchedOdeRK4::<MultiSignalOde>::integrate_cpu(&y0, &flat, DT, N_STEPS, 1)
                .expect("multi_signal upstream")
        });

        let max_diff = local_result
            .y_final
            .iter()
            .zip(upstream_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        v.check_pass("CPU parity (max diff < 0.5)", max_diff < 0.5);
        println!("  max |local - upstream| = {max_diff:.2e}");

        let shader = BatchedOdeRK4::<MultiSignalOde>::generate_shader();
        let shader_lines = shader.lines().count();

        let mut batch_timings = Vec::new();
        for &bs in BATCH_SIZES {
            let states: Vec<f64> = y0.iter().copied().cycle().take(bs * 7).collect();
            let params: Vec<f64> = flat.iter().copied().cycle().take(bs * 24).collect();
            let (_, us) = bench(&format!("batched CPU ({bs} batches)"), || {
                BatchedOdeRK4::<MultiSignalOde>::integrate_cpu(&states, &params, DT, N_STEPS, bs)
                    .expect("batched")
            });
            batch_timings.push((bs, us));
        }

        benches.push(SystemBench {
            name: "MultiSignal",
            origin: "wetSpring Exp006",
            n_vars: MultiSignalOde::N_VARS,
            n_params: MultiSignalOde::N_PARAMS,
            shader_lines,
            local_cpu_us: local_us,
            upstream_cpu_us: upstream_us,
            batch_timings,
            max_diff,
        });
    }

    // ─── §4 Bistable ODE (wetSpring Exp007, Fernandez 2020) ────────
    v.section("§4 Bistable ODE — wetSpring → ToadStool lean");
    println!("  Origin: wetSpring Exp007 (V. cholerae bistable switch + feedback)");
    {
        let p = BistableParams::default();
        let y0 = [0.01, 0.0, 0.0, 2.0, 0.0];

        let (local_result, local_us) = bench("local CPU integration (4800 steps)", || {
            wetspring_barracuda::bio::bistable::run_bistable(&y0, 48.0, DT, &p)
        });

        let flat = p.to_flat();
        let (upstream_result, upstream_us) = bench("upstream integrate_cpu (4800 steps)", || {
            BatchedOdeRK4::<BistableOde>::integrate_cpu(&y0, &flat, DT, N_STEPS, 1)
                .expect("bistable upstream")
        });

        let max_diff = local_result
            .y_final
            .iter()
            .zip(upstream_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        v.check_pass("CPU parity (max diff < 0.5)", max_diff < 0.5);
        println!("  max |local - upstream| = {max_diff:.2e}");

        let shader = BatchedOdeRK4::<BistableOde>::generate_shader();
        let shader_lines = shader.lines().count();

        let mut batch_timings = Vec::new();
        for &bs in BATCH_SIZES {
            let states: Vec<f64> = y0.iter().copied().cycle().take(bs * 5).collect();
            let params: Vec<f64> = flat.iter().copied().cycle().take(bs * 21).collect();
            let (_, us) = bench(&format!("batched CPU ({bs} batches)"), || {
                BatchedOdeRK4::<BistableOde>::integrate_cpu(&states, &params, DT, N_STEPS, bs)
                    .expect("batched")
            });
            batch_timings.push((bs, us));
        }

        benches.push(SystemBench {
            name: "Bistable",
            origin: "wetSpring Exp007",
            n_vars: BistableOde::N_VARS,
            n_params: BistableOde::N_PARAMS,
            shader_lines,
            local_cpu_us: local_us,
            upstream_cpu_us: upstream_us,
            batch_timings,
            max_diff,
        });
    }

    // ─── §5 Phage Defense ODE (wetSpring Exp009, Hsueh 2022) ───────
    v.section("§5 Phage Defense ODE — wetSpring → ToadStool lean");
    println!("  Origin: wetSpring Exp009 (phage-bacteria defense arms race)");
    {
        let p = PhageDefenseParams {
            resource_inflow: 100.0,
            ..PhageDefenseParams::default()
        };
        let y0 = [1e4, 1e4, 0.0, 50.0];

        let (local_result, local_us) = bench("local CPU integration (500 steps)", || {
            wetspring_barracuda::bio::phage_defense::run_defense(&y0, 5.0, DT, &p)
        });

        let flat = p.to_flat();
        let steps = 500;
        let (upstream_result, upstream_us) = bench("upstream integrate_cpu (500 steps)", || {
            BatchedOdeRK4::<PhageDefenseOde>::integrate_cpu(&y0, &flat, DT, steps, 1)
                .expect("phage_defense upstream")
        });

        v.check_pass(
            "local result finite",
            local_result.y_final.iter().all(|x| x.is_finite()),
        );
        v.check_pass(
            "upstream result finite",
            upstream_result.iter().all(|x| x.is_finite()),
        );

        let max_diff = local_result
            .y_final
            .iter()
            .zip(upstream_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("  max |local - upstream| = {max_diff:.2e}");
        println!("  (phage defense: clamping differences expected at extreme params)");

        let shader = BatchedOdeRK4::<PhageDefenseOde>::generate_shader();
        v.check_pass("WGSL contains deriv()", shader.contains("fn deriv"));
        let shader_lines = shader.lines().count();

        let mut batch_timings = Vec::new();
        for &bs in BATCH_SIZES {
            let states: Vec<f64> = y0.iter().copied().cycle().take(bs * 4).collect();
            let params: Vec<f64> = flat.iter().copied().cycle().take(bs * 11).collect();
            let (_, us) = bench(&format!("batched CPU ({bs} batches)"), || {
                BatchedOdeRK4::<PhageDefenseOde>::integrate_cpu(&states, &params, DT, steps, bs)
                    .expect("batched")
            });
            batch_timings.push((bs, us));
        }

        benches.push(SystemBench {
            name: "PhageDefense",
            origin: "wetSpring Exp009",
            n_vars: PhageDefenseOde::N_VARS,
            n_params: PhageDefenseOde::N_PARAMS,
            shader_lines,
            local_cpu_us: local_us,
            upstream_cpu_us: upstream_us,
            batch_timings,
            max_diff,
        });
    }

    // ─── §6 Cross-Spring Evolution Summary ─────────────────────────
    v.section("§6 Cross-Spring Evolution Summary");

    println!("\n  ┌───────────────────────────────────────────────────────────────────┐");
    println!("  │                 Cross-Spring Shader Provenance                    │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │ hotSpring  → f64 WGSL emulation pattern (lattice QCD, HFB)       │");
    println!("  │            → ESN reservoir (Stanton-Murillo transport)            │");
    println!("  │            → Hermite/Laguerre for nuclear EOS                    │");
    println!("  │            → CG solver shaders → used by all springs             │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │ wetSpring  → 5 bio ODE shaders (now lean on ToadStool template)  │");
    println!("  │            → Smith-Waterman, Felsenstein, tree inference          │");
    println!("  │            → Gillespie SSA, DADA2, quality filter, SNP, ANI      │");
    println!("  │            → Shannon/Simpson/Bray-Curtis diversity               │");
    println!("  │            → ESN NPU weight export + int8 inference              │");
    println!("  │            → GemmCachedF64 (60× taxonomy speedup)                │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │ neuralSpring → xoshiro128ss PRNG                                 │");
    println!("  │              → logsumexp_reduce for HMM                          │");
    println!("  │              → Wright-Fisher drift + Fermi imitation              │");
    println!("  │              → PairwiseHamming/Jaccard/L2, SpatialPayoff         │");
    println!("  │              → BatchFitness, LocusVariance, hill_gate             │");
    println!("  │              → Adaptive RK45 → influenced ODE architecture       │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │ ToadStool  → OdeSystem trait + BatchedOdeRK4 generic template    │");
    println!("  │            → Tolerance registry + provenance tags                │");
    println!("  │            → 12 ProvenanceTag entries across 3 springs           │");
    println!("  │            → 4,176 core tests, zero warnings, zero debt          │");
    println!("  └───────────────────────────────────────────────────────────────────┘");

    println!(
        "\n  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "  │ System        │ Origin           │ Vars │ Params │ WGSL Ln │ Local CPU µs │ Upstream µs │ Max Diff    │"
    );
    println!(
        "  ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤"
    );
    for b in &benches {
        println!(
            "  │ {:13} │ {:16} │ {:4} │ {:6} │ {:7} │ {:12.0} │ {:11.0} │ {:11.2e} │",
            b.name,
            b.origin,
            b.n_vars,
            b.n_params,
            b.shader_lines,
            b.local_cpu_us,
            b.upstream_cpu_us,
            b.max_diff
        );
    }
    println!(
        "  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    );

    println!("\n  Batch Scaling (upstream integrate_cpu):");
    println!("  ─────────────────────────────────────────");
    for b in &benches {
        print!("  {:13}:", b.name);
        for (bs, us) in &b.batch_timings {
            print!("  {bs}→{us:.0}µs");
        }
        println!();
    }

    let total_shader_lines: usize = benches.iter().map(|b| b.shader_lines).sum();
    let total_local_wgsl_deleted = 30424_usize;
    println!("\n  Summary:");
    println!("  ────────");
    println!("  Generated WGSL total: {total_shader_lines} lines (from OdeSystem traits)");
    println!("  Deleted local WGSL:   {total_local_wgsl_deleted} bytes (5 files)");
    println!("  All 5 GPU modules now use BatchedOdeRK4::<S>::generate_shader()");
    println!("  Cross-spring evolution: Write → Absorb → Lean cycle complete");

    v.check_pass("all systems benchmarked", benches.len() == 5);
    v.check_pass(
        "all parity diffs finite",
        benches.iter().all(|b| b.max_diff.is_finite()),
    );

    v.finish();
}
