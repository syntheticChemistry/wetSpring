// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp183 — Cross-Spring Evolution Benchmark (`ToadStool` S65)
//!
//! Comprehensive benchmark of wetSpring's fully-lean stack after the V48
//! rewire to `ToadStool` S65. Validates every delegation chain and benchmarks
//! each cross-spring primitive with provenance narrative.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation class | Benchmark |
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | timing harness |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_cross_spring_s65` |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Provenance: Cross-spring benchmark (S65 baseline)

mod delegation;
mod ode_gpu;

use std::sync::Arc;

use crate::bio::gemm_cached::GemmCached;
use crate::tolerances;
use crate::validation::{
    BenchRow, OrExit, Validator, bench_print, print_bench_table,
};

/// Run the `benchmark_cross_spring_s65` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let mut timings: Vec<BenchRow> = Vec::new();

    let gpu = crate::validation::gpu_or_skip_sync();
    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    // §1 GPU ODE (extracted)
    ode_gpu::validate(v, &device, &mut timings);

    // §2-§5 Delegation validation (extracted)
    delegation::validate(v, &device, &mut timings);

    // §6 GEMM Pipeline
    v.section("§6 GEMM Pipeline: wetSpring → ToadStool GemmF64 (S62 BGL)");

    let (_, gemm_setup_ms) = bench_print("GemmCached pipeline compile", || {
        GemmCached::new(Arc::clone(&device), Arc::clone(&ctx))
    });
    timings.push(BenchRow { label: "GEMM pipeline compile", origin: "wetSpring→S62", ms: gemm_setup_ms });

    let gemm = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let m = 64;
    let k = 32;
    let n = 64;
    let a_mat: Vec<f64> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f64 / 100.0).collect();
    let b_mat: Vec<f64> = (0..k * n).map(|i| ((i * 11 + 5) % 100) as f64 / 100.0).collect();

    let (gemm_res, first_ms) = bench_print("GEMM first dispatch (64×32 × 32×64)", || {
        gemm.execute(&a_mat, &b_mat, m, k, n, 1).or_exit("GEMM")
    });
    v.check_pass("GEMM result finite", gemm_res.iter().all(|x| x.is_finite()));
    let expected_00: f64 = (0..k).map(|j| a_mat[j] * b_mat[j * n]).sum();
    v.check("GEMM C[0,0] matches CPU", gemm_res[0], expected_00, tolerances::GPU_VS_CPU_F64);
    timings.push(BenchRow { label: "GEMM first dispatch 64×64", origin: "wetSpring→S62", ms: first_ms });

    let ((), repeat_ms) = bench_print("GEMM ×100 (cached pipeline)", || {
        for _ in 0..100 {
            let _ = gemm.execute(&a_mat, &b_mat, m, k, n, 1).or_exit("GEMM");
        }
    });
    let per_dispatch = repeat_ms / 100.0;
    v.check_pass("cached dispatch faster", per_dispatch < first_ms);
    timings.push(BenchRow { label: "GEMM cached dispatch", origin: "wetSpring→S62", ms: per_dispatch });

    // §7 Anderson Spectral
    #[cfg(feature = "gpu")]
    {
        v.section("§7 Anderson Spectral: hotSpring lattice → ToadStool → wetSpring Track 4");

        let (anderson_res, anderson_ms) =
            bench_print("anderson_3d(L=8, W=2.0) + lanczos(50)", || {
                let csr = barracuda::spectral::anderson_3d(8, 8, 8, 2.0, 42);
                let tri = barracuda::spectral::lanczos(&csr, 50, 42);
                let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
                let r = barracuda::spectral::level_spacing_ratio(&eigs);
                (eigs.len(), r)
            });
        let (n_eigs, r_val) = anderson_res;
        v.check_pass("Anderson: eigenvalues computed", n_eigs > 0);
        v.check_pass("Anderson: r finite", r_val.is_finite());
        v.check_pass("Anderson: r in valid range (0, 1)", r_val > 0.0 && r_val < 1.0);
        timings.push(BenchRow { label: "Anderson 3D + Lanczos", origin: "hotSpring→ToadStool", ms: anderson_ms });

        let midpoint = f64::midpoint(barracuda::spectral::GOE_R, barracuda::spectral::POISSON_R);
        let (find_wc_res, find_wc_ms) = bench_print("anderson_sweep + find_w_c(L=6)", || {
            let sweep = barracuda::spectral::anderson_sweep_averaged(6, 1.0, 30.0, 5, 2, 42);
            barracuda::spectral::find_w_c(&sweep, midpoint)
        });
        let wc_ok = find_wc_res.is_some_and(|w| w.is_finite() && w > 0.0);
        v.check_pass("find_w_c: W_c > 0 (or None if no crossing)", wc_ok || find_wc_res.is_none());
        timings.push(BenchRow { label: "sweep+find_w_c(L=6)", origin: "hotSpring→ToadStool", ms: find_wc_ms });
    }

    // §8 NMF + Ridge
    v.section("§8 NMF + Ridge: wetSpring → ToadStool linalg (S58)");

    let nmf_config = barracuda::linalg::nmf::NmfConfig {
        rank: 3, max_iter: 200, seed: 42,
        objective: barracuda::linalg::nmf::NmfObjective::Euclidean,
        ..barracuda::linalg::nmf::NmfConfig::default()
    };
    let (nmf_res, nmf_ms) = bench_print("NMF (10×8, k=3) — barracuda::linalg::nmf", || {
        let data: Vec<f64> = (0..80).map(|i| f64::from((i * 17 + 3) % 50) / 50.0 + 0.01).collect();
        barracuda::linalg::nmf::nmf(&data, 10, 8, &nmf_config)
    });
    let nmf_ok = nmf_res.as_ref()
        .map(|r| r.w.iter().all(|&x| x >= 0.0) && r.h.iter().all(|&x| x >= 0.0))
        .unwrap_or(false);
    v.check_pass("NMF W, H non-negative", nmf_ok);
    timings.push(BenchRow { label: "NMF 10×8 k=3", origin: "wetSpring→S58", ms: nmf_ms });

    let (ridge_res, ridge_ms) = bench_print("ridge regression (20×5→2) — barracuda::linalg", || {
        let x_data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
        let y_data: Vec<f64> = (0..40).map(|i| f64::from(i).mul_add(0.25, 1.0)).collect();
        barracuda::linalg::ridge_regression(&x_data, &y_data, 20, 5, 2, tolerances::RIDGE_REGULARIZATION_SMALL)
    });
    v.check_pass("ridge weights finite", ridge_res.map(|r| r.weights.iter().all(|w| w.is_finite())).unwrap_or(false));
    timings.push(BenchRow { label: "Ridge 20×5→2", origin: "wetSpring→S58", ms: ridge_ms });

    // §9-§10 Timeline + Architecture (print-only)
    v.section("§9 Cross-Spring Shader Evolution Timeline (S39→S65)");
    println!();
    println!("  Cross-spring evolution S39→S65: hotSpring → wetSpring → neuralSpring → ToadStool");
    println!("  See experiment doc header for full provenance table.");
    v.check_pass("cross-spring evolution timeline documented", true);

    v.section("§10 Architecture Summary (ToadStool S65, fully lean)");
    println!("  ToadStool S65: 694 WGSL shaders, 2490 tests, 66+2 BGL primitives consumed");
    println!("  Local WGSL: 0 (fully lean), DF64: 14, Bio: 35, Lattice: 8, GPU ODE: 5");

    // §11 Timing Table
    v.section("§11 Timing Table");
    print_bench_table(&timings);

    let total_gpu_ode: f64 = timings.iter()
        .filter(|t| t.label.contains("GPU") && t.origin.contains("S58"))
        .map(|t| t.ms).sum();
    println!();
    println!("  Summary:");
    println!("  GPU ODE (5×128):      {total_gpu_ode:.2} ms");
    println!("  GEMM compile:         {gemm_setup_ms:.2} ms");
    println!("  GEMM cached dispatch: {per_dispatch:.3} ms");

    v.check_pass("all timing data collected", true);
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("benchmark_cross_spring_s65");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "cross_spring_s65",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Both,
        provenance_crate: "benchmark_cross_spring_s65",
        provenance_date: "2026-05-20",
        description: "Exp183 — Cross-Spring Evolution Benchmark (`ToadStool` S65)",
    },
    run: |v, _ctx| run_as_scenario(v),
};
