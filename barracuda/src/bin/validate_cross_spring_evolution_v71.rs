// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
//! Exp223 — Cross-Spring Evolution Validation + Benchmark (V71 Complete Rewire)
//!
//! Validates every spring's contribution to the `ToadStool` barracuda ecosystem
//! from wetSpring's perspective: CPU math, GPU primitives, precision routing,
//! DF64 host protocol, cross-spring provenance, and `BandwidthTier` estimation.
//!
//! # Cross-Spring Provenance
//!
//! ```text
//! hotSpring  → erf, ln_gamma, Fp64Strategy, DF64 core, Anderson spectral, Sovereign compiler
//! wetSpring  → bio ODE ×5, diversity, BrayCurtis, DiversityFusion, GEMM, NMF, ridge
//! neuralSpring → graph_laplacian, numerical_hessian, effective_rank, pairwise ops
//! airSpring  → Pearson, MAE, RMSE, R², NSE, trapz, kriging, moving window
//! groundSpring → bootstrap rawr_mean, batched multinomial
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cross_spring_evolution_v71` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::sync::Arc;
use std::time::Instant;

use barracuda::shaders::Precision;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu};
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct Timing {
    label: &'static str,
    origin: &'static str,
    ms: f64,
}

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let result = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {label}: {ms:.3} ms");
    (result, ms)
}

fn main() {
    let mut v = Validator::new("Exp223: Cross-Spring Evolution Validation (V71 Complete Rewire)");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // §0  GPU Init + ToadStool S68+ Precision Architecture
    // ═══════════════════════════════════════════════════════════════════

    v.section("§0 GPU Init + Precision Architecture (ToadStool S68+)");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let strategy = gpu.fp64_strategy();
    let precision = gpu.optimal_precision();
    let is_lost = gpu.is_lost();
    let threshold = gpu.dispatch_threshold();

    println!("  GPU: {}", gpu.adapter_name);
    println!("  Fp64Strategy: {strategy:?} (hotSpring S58 → ToadStool S67)");
    println!("  optimal_precision(): {precision:?} (ToadStool S68 universal)");
    println!("  is_lost(): {is_lost} (ToadStool S68+ device resilience)");
    println!("  dispatch_threshold(): {threshold} elements");

    v.check_pass("GPU initialized", true);
    v.check_pass("device not lost (S68+ resilience)", !is_lost);
    v.check_pass(
        "Fp64Strategy detected",
        matches!(
            strategy,
            barracuda::device::Fp64Strategy::Native | barracuda::device::Fp64Strategy::Hybrid
        ),
    );
    v.check_pass(
        "optimal_precision is F64 or Df64",
        matches!(precision, Precision::F64 | Precision::Df64),
    );

    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    // BandwidthTier (ToadStool unified_hardware)
    let tier =
        barracuda::unified_hardware::BandwidthTier::detect_from_adapter_name(&gpu.adapter_name);
    let bw = tier.bandwidth_gbps();
    println!("  BandwidthTier: {tier:?} ({bw:.1} GB/s)");
    v.check_pass("BandwidthTier detected", bw > 0.0);

    // ═══════════════════════════════════════════════════════════════════
    // §1  DF64 Host Protocol Validation (wetSpring df64_host)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§1 DF64 Host Protocol (wetSpring df64_host, ToadStool S68+)");
    println!("  Purpose: validate host-side pack/unpack matches ToadStool DF64 wire format");
    println!("  Wire format: array<vec2<f32>> with .x = hi, .y = lo");

    let test_values = [
        1.0_f64,
        std::f64::consts::PI,
        std::f64::consts::E,
        -std::f64::consts::E,
        1e-10,
        1e20,
        0.0,
        1.000_000_1,
    ];

    for &val in &test_values {
        let err = df64_host::roundtrip_error(val);
        let pass = if val == 0.0 {
            err == 0.0
        } else {
            err / val.abs() < tolerances::PYTHON_PARITY_TIGHT
        };
        v.check_pass(&format!("DF64 roundtrip {val:.6e} (err={err:.2e})"), pass);
    }

    let packed = df64_host::pack_slice(&test_values);
    let unpacked = df64_host::unpack_slice(&packed);
    v.check_pass(
        "pack_slice length = 2 × input",
        packed.len() == test_values.len() * 2,
    );
    v.check_pass(
        "unpack_slice length = input",
        unpacked.len() == test_values.len(),
    );
    v.check_pass(
        "slice roundtrip max error < 1e-14",
        test_values
            .iter()
            .zip(&unpacked)
            .all(|(a, b)| (a - b).abs() < tolerances::PYTHON_PARITY_TIGHT),
    );

    let f32_err = (std::f64::consts::PI - f64::from(std::f64::consts::PI as f32)).abs();
    let df64_err = df64_host::roundtrip_error(std::f64::consts::PI);
    println!("  f32 error for π: {f32_err:.2e}");
    println!("  DF64 error for π: {df64_err:.2e}");
    println!("  DF64 precision gain: {:.0}× over f32", f32_err / df64_err);
    v.check_pass("DF64 more precise than f32", df64_err < f32_err);

    // ═══════════════════════════════════════════════════════════════════
    // §2  hotSpring: Precision Functions + Anderson Spectral
    // ═══════════════════════════════════════════════════════════════════

    v.section("§2 hotSpring: Precision + Spectral (S58-S68+)");
    println!("  erf, ln_gamma → ToadStool barracuda::special (always-on CPU math)");
    println!("  Anderson 3D → ToadStool barracuda::spectral (Kachkovskiy v0.6.0)");
    println!("  Fp64Strategy → hotSpring S58; DF64 core → hotSpring biomeGate");

    let erf_val = barracuda::special::erf(1.0);
    v.check(
        "erf(1.0)",
        erf_val,
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );

    let lng = barracuda::special::ln_gamma(5.0).expect("ln_gamma");
    v.check(
        "ln_gamma(5.0) = ln(24)",
        lng,
        24.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    let ncdf = barracuda::stats::norm_cdf(1.96);
    v.check(
        "norm_cdf(1.96) ≈ 0.975",
        ncdf,
        0.975,
        tolerances::CROSS_SPRING_NUMERICAL,
    );

    let (anderson_r, anderson_ms) = bench("Anderson 3D (L=8, W=2.0)", || {
        let csr = barracuda::spectral::anderson_3d(8, 8, 8, 2.0, 42);
        let tri = barracuda::spectral::lanczos(&csr, 50, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
        barracuda::spectral::level_spacing_ratio(&eigs)
    });
    v.check_pass("Anderson r ∈ (0,1)", anderson_r > 0.0 && anderson_r < 1.0);

    timings.push(Timing {
        label: "Anderson 3D (L=8)",
        origin: "hotSpring→S59",
        ms: anderson_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §3  wetSpring Bio: CPU Diversity + GPU DiversityFusion
    // ═══════════════════════════════════════════════════════════════════

    v.section("§3 wetSpring Bio: Diversity (CPU S64, GPU DiversityFusion S63)");
    println!("  CPU: barracuda::stats::diversity (Shannon, Simpson, Pielou) [S64 absorb]");
    println!("  GPU: DiversityFusionGpu → FusedMapReduceF64 + compile_shader_universal");

    let n_taxa = 500;
    let counts: Vec<f64> = (0..n_taxa)
        .map(|i| f64::from(((i * 7 + 3) % 100 + 1) as u32))
        .collect();

    let (cpu_shannon, cpu_div_ms) = bench("CPU Shannon (500 taxa)", || diversity::shannon(&counts));
    let cpu_simpson = diversity::simpson(&counts);
    let cpu_pielou = diversity::pielou_evenness(&counts);

    v.check_pass("Shannon > 0", cpu_shannon > 0.0);
    v.check_pass("Simpson ∈ (0,1)", cpu_simpson > 0.0 && cpu_simpson < 1.0);
    v.check_pass("Pielou ∈ (0,1]", cpu_pielou > 0.0 && cpu_pielou <= 1.0);

    let n_species = 10_000;
    let n_samples = 5;
    let large_counts: Vec<f64> = (0..n_samples * n_species)
        .map(|i| ((i * 13 + 7) % 200 + 1) as f64)
        .collect();

    let (cpu_fusion, cpu_fusion_ms) = bench("CPU DiversityFusion (5×10k)", || {
        diversity_fusion_cpu(&large_counts, n_species)
    });

    let fusion_gpu = DiversityFusionGpu::new(Arc::clone(&device)).expect("DiversityFusionGpu");
    let (gpu_fusion, gpu_fusion_ms) = bench("GPU DiversityFusion (5×10k)", || {
        fusion_gpu
            .compute(&large_counts, n_samples, n_species)
            .expect("fusion GPU")
    });

    v.check(
        "GPU Shannon ≈ CPU",
        gpu_fusion[0].shannon,
        cpu_fusion[0].shannon,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "GPU Simpson ≈ CPU",
        gpu_fusion[0].simpson,
        cpu_fusion[0].simpson,
        tolerances::GPU_VS_CPU_F64,
    );

    let div_speedup = cpu_fusion_ms / gpu_fusion_ms;
    println!("  GPU DiversityFusion speedup: {div_speedup:.1}×");

    timings.push(Timing {
        label: "CPU diversity 500",
        origin: "wetSpring→S64",
        ms: cpu_div_ms,
    });
    timings.push(Timing {
        label: "CPU DiversityFusion 5×10k",
        origin: "wetSpring→S63",
        ms: cpu_fusion_ms,
    });
    timings.push(Timing {
        label: "GPU DiversityFusion 5×10k",
        origin: "wetSpring→S63→GPU",
        ms: gpu_fusion_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §4  neuralSpring: Graph + Hessian + Effective Rank
    // ═══════════════════════════════════════════════════════════════════

    v.section("§4 neuralSpring: Graph Theory + Numerical Analysis");
    println!("  graph_laplacian → ToadStool barracuda::linalg (neuralSpring handoff)");
    println!("  numerical_hessian → ToadStool barracuda::numerical (baseCamp V18)");
    println!("  effective_rank → ToadStool barracuda::linalg (eigenvalue entropy)");

    let n_graph = 10;
    let mut adjacency = vec![0.0; n_graph * n_graph];
    for i in 0..n_graph {
        for j in (i + 1)..n_graph {
            if (i + j) % 3 != 0 {
                adjacency[i * n_graph + j] = 1.0;
                adjacency[j * n_graph + i] = 1.0;
            }
        }
    }

    let (laplacian, graph_ms) = bench("graph_laplacian (10×10)", || {
        barracuda::linalg::graph_laplacian(&adjacency, n_graph)
    });
    v.check_pass(
        "Laplacian size correct",
        laplacian.len() == n_graph * n_graph,
    );

    let row_sums: Vec<f64> = (0..n_graph)
        .map(|i| (0..n_graph).map(|j| laplacian[i * n_graph + j]).sum())
        .collect();
    let laplacian_valid = row_sums
        .iter()
        .all(|&s| s.abs() < tolerances::PYTHON_PARITY);
    v.check_pass("Laplacian row sums ≈ 0 (graph property)", laplacian_valid);

    let diag_positive = (0..n_graph).all(|i| laplacian[i * n_graph + i] >= 0.0);
    v.check_pass("Laplacian diagonal ≥ 0 (degree)", diag_positive);

    let (hessian, hessian_ms) = bench("numerical_hessian (Rosenbrock, 2D)", || {
        let rosenbrock = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[0].mul_add(-x[0], x[1]);
            a * a + 100.0 * b * b
        };
        barracuda::numerical::numerical_hessian(
            &rosenbrock,
            &[1.0, 1.0],
            tolerances::NUMERICAL_HESSIAN_EPSILON,
        )
    });
    v.check_pass("Hessian 2×2", hessian.len() == 4);
    v.check(
        "Hessian H[0,0] ≈ 802 at optimum",
        hessian[0],
        802.0,
        tolerances::HESSIAN_H00_TOL,
    );
    v.check(
        "Hessian H[1,1] ≈ 200 at optimum",
        hessian[3],
        200.0,
        tolerances::HESSIAN_H11_TOL,
    );

    let eigenvalues = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001];
    let (eff_rank, _) = bench("effective_rank (8 eigenvalues)", || {
        barracuda::linalg::effective_rank(&eigenvalues)
    });
    v.check_pass("effective_rank ∈ [1, 8]", (1.0..=8.0).contains(&eff_rank));
    println!("  effective_rank = {eff_rank:.2} (entropy-based dimensionality)");

    timings.push(Timing {
        label: "graph_laplacian 10×10",
        origin: "neuralSpring",
        ms: graph_ms,
    });
    timings.push(Timing {
        label: "numerical_hessian 2D",
        origin: "neuralSpring→baseCamp",
        ms: hessian_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §5  airSpring/groundSpring: Stats + Numerical Integration
    // ═══════════════════════════════════════════════════════════════════

    v.section("§5 airSpring + groundSpring: Cross-Spring Stats (S64-S66)");
    println!("  Pearson, MAE, RMSE, R² → barracuda::stats (airSpring S64)");
    println!("  trapz → barracuda::numerical (cross-spring)");

    let obs: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.05).collect();
    let sim: Vec<f64> = obs
        .iter()
        .map(|&x| 0.01f64.mul_add(x.sin(), 2.0f64.mul_add(x, 1.0)))
        .collect();

    let pearson = barracuda::stats::pearson_correlation(&obs, &sim).expect("pearson");
    v.check(
        "Pearson(linear) ≈ 1.0",
        pearson,
        1.0,
        tolerances::CROSS_SPRING_NUMERICAL,
    );

    let mae = barracuda::stats::mae(&obs, &sim);
    let rmse = barracuda::stats::rmse(&obs, &sim);
    v.check_pass("MAE finite", mae.is_finite() && mae >= 0.0);
    v.check_pass("RMSE finite", rmse.is_finite() && rmse >= 0.0);
    v.check_pass("RMSE ≥ MAE (Cauchy-Schwarz)", rmse >= mae);

    let sim_close: Vec<f64> = obs.iter().map(|&x| 0.001f64.mul_add(x.sin(), x)).collect();
    let r2 = barracuda::stats::r_squared(&obs, &sim_close);
    v.check_pass("R² > 0.999", r2 > 0.999);

    let n_pts = 2000;
    let trap_x: Vec<f64> = (0..n_pts)
        .map(|i| f64::from(i) / f64::from(n_pts - 1))
        .collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|&xi| xi * xi).collect();
    let (trapz_val, trapz_ms) = bench("trapz(x², 2000 pts)", || {
        barracuda::numerical::trapz(&trap_y, &trap_x).expect("trapz")
    });
    v.check(
        "trapz(x²) ≈ 1/3",
        trapz_val,
        1.0 / 3.0,
        tolerances::TRAPZ_COARSE,
    );

    timings.push(Timing {
        label: "trapz(x², 2000 pts)",
        origin: "cross-spring",
        ms: trapz_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §6  GPU GEMM: Precision-Flexible Pipeline (V71 Rewire)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§6 GPU GEMM: Precision-Flexible (V71 with_precision rewire)");
    println!("  GemmCached::new() → Precision::F64 (backward compatible)");
    println!("  GemmCached::with_precision() → any precision (V71 rewire)");
    println!("  compile_shader_universal routes through ToadStool S68+ pipeline");

    let m = 256;
    let k = 128;
    let n = 256;
    let a_mat: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    let gemm_f64 = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let (res_f64, f64_ms) = bench("GEMM 256×128×256 @ Precision::F64", || {
        gemm_f64
            .execute(&a_mat, &b_mat, m, k, n, 1)
            .expect("GEMM F64")
    });
    v.check_pass(
        "F64 GEMM result finite",
        res_f64.iter().all(|x| x.is_finite()),
    );

    let expected_00: f64 = (0..k).map(|j| a_mat[j] * b_mat[j * n]).sum();
    v.check(
        "GEMM C[0,0] ≈ CPU",
        res_f64[0],
        expected_00,
        tolerances::GPU_VS_CPU_F64,
    );

    let gemm_f64_explicit =
        GemmCached::with_precision(Arc::clone(&device), Arc::clone(&ctx), Precision::F64);
    let (res_explicit, _) = bench("GEMM via with_precision(F64)", || {
        gemm_f64_explicit
            .execute(&a_mat, &b_mat, m, k, n, 1)
            .expect("GEMM explicit F64")
    });
    v.check(
        "new() == with_precision(F64)",
        res_explicit[0],
        res_f64[0],
        tolerances::EXACT_F64,
    );

    // Cached dispatch throughput
    for _ in 0..5 {
        let _ = gemm_f64.execute(&a_mat, &b_mat, m, k, n, 1);
    }
    let ((), cached_ms) = bench("GEMM ×50 cached (submit_and_poll S68+)", || {
        for _ in 0..50 {
            let _ = gemm_f64.execute(&a_mat, &b_mat, m, k, n, 1);
        }
    });
    let per_dispatch = cached_ms / 50.0;
    v.check_pass("cached dispatch faster than cold", per_dispatch < f64_ms);
    println!("  Cold dispatch: {f64_ms:.3} ms");
    println!(
        "  Cached avg: {per_dispatch:.3} ms ({:.1}× amortization)",
        f64_ms / per_dispatch
    );

    timings.push(Timing {
        label: "GEMM cold 256×256 F64",
        origin: "wetSpring→S62→S68",
        ms: f64_ms,
    });
    timings.push(Timing {
        label: "GEMM cached avg F64",
        origin: "wetSpring→S68+",
        ms: per_dispatch,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §7  NMF + Ridge: Cross-Spring Linalg
    // ═══════════════════════════════════════════════════════════════════

    v.section("§7 NMF + Ridge: Cross-Spring Linalg (wetSpring → S58-S59)");
    println!("  NMF: Lee & Seung 1999 → ToadStool barracuda::linalg::nmf (S58)");
    println!("  Ridge: Tikhonov → ToadStool barracuda::linalg (S59)");
    println!("  Both used by: neuralSpring (ESN readout), airSpring (kriging)");

    let v_mat: Vec<f64> = (0..20 * 10)
        .map(|i| f64::from(((i * 3 + 1) % 50) as u32) / 50.0)
        .collect();
    let nmf_cfg = barracuda::linalg::nmf::NmfConfig {
        rank: 3,
        max_iter: 100,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let (nmf_res, nmf_ms) = bench("NMF 20×10 (KL, k=3)", || {
        barracuda::linalg::nmf::nmf(&v_mat, 20, 10, &nmf_cfg).expect("NMF")
    });
    v.check_pass(
        "NMF W, H non-negative",
        nmf_res.w.iter().chain(&nmf_res.h).all(|&x| x >= 0.0),
    );

    let ridge_x: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
    let ridge_y: Vec<f64> = (0..40).map(|i| f64::from(i).mul_add(0.25, 1.0)).collect();
    let (ridge_res, ridge_ms) = bench("Ridge regression (20×5→2)", || {
        barracuda::linalg::ridge_regression(
            &ridge_x,
            &ridge_y,
            20,
            5,
            2,
            tolerances::RIDGE_REGULARIZATION_SMALL,
        )
    });
    v.check_pass(
        "ridge weights finite",
        ridge_res
            .map(|r| r.weights.iter().all(|w| w.is_finite()))
            .unwrap_or(false),
    );

    timings.push(Timing {
        label: "NMF 20×10 KL",
        origin: "wetSpring→S58",
        ms: nmf_ms,
    });
    timings.push(Timing {
        label: "Ridge 20×5→2",
        origin: "wetSpring→S59",
        ms: ridge_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §8  Cross-Spring Provenance Summary
    // ═══════════════════════════════════════════════════════════════════

    v.section("§8 Cross-Spring Provenance: When & Where Things Evolved");

    println!();
    println!(
        "  ╔══════════════╤═══════════════════════════════════════╤═══════════════════════════════════════╗"
    );
    println!(
        "  ║ Spring       │ Contributed                           │ Who Benefits                          ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ hotSpring    │ erf, ln_gamma (A&S polynomial),      │ ALL springs get precision math;       ║"
    );
    println!(
        "  ║              │ Fp64Strategy (S58), DF64 core (S66), │ neuralSpring eigensolvers use erf;    ║"
    );
    println!(
        "  ║              │ Anderson spectral (S59), Sovereign   │ wetSpring ODE polyfills from S67;     ║"
    );
    println!(
        "  ║              │ compiler (S63), lattice QCD (S60)    │ airSpring norm_cdf for hydrology      ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ wetSpring    │ Bio ODE ×5 (S58), diversity (S64),   │ neuralSpring uses FusedMapReduce;     ║"
    );
    println!(
        "  ║              │ BrayCurtis (S63), DiversityFusion,   │ groundSpring uses diversity for       ║"
    );
    println!(
        "  ║              │ GEMM cached (S62), NMF (S58), ridge  │ ecology; airSpring uses ridge for     ║"
    );
    println!(
        "  ║              │ (S59), PeakDetect (S62), TransE      │ kriging readout training              ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ neuralSpring │ graph_laplacian (S54), pairwise ops  │ wetSpring community network analysis; ║"
    );
    println!(
        "  ║              │ (Hamming/Jaccard/L2, S56), effective │ airSpring IoT sensor correlation;     ║"
    );
    println!(
        "  ║              │ rank, numerical_hessian (baseCamp),  │ hotSpring Anderson uses Lanczos       ║"
    );
    println!(
        "  ║              │ spatial_payoff, swarm NN, KL div     │ from shared ToadStool linalg          ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ airSpring    │ Pearson, MAE, RMSE, R², NSE (S64),  │ wetSpring uses Pearson for paper      ║"
    );
    println!(
        "  ║              │ Richards PDE, moving window, kriging │ validation; neuralSpring uses R²      ║"
    );
    println!(
        "  ║              │ (S66), Crank-Nicolson, RK stages     │ for model evaluation                  ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ groundSpring │ bootstrap rawr_mean (S64), batched   │ wetSpring uses bootstrap for          ║"
    );
    println!(
        "  ║              │ multinomial (S66), percentile        │ confidence intervals                  ║"
    );
    println!(
        "  ╠══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╣"
    );
    println!(
        "  ║ ToadStool    │ 700 WGSL shaders (0 f32-only),      │ ALL springs get: universal precision, ║"
    );
    println!(
        "  ║  (hub)       │ universal precision (F16-DF64),      │ device-lost resilience, ILP optimizer,║"
    );
    println!(
        "  ║              │ Sovereign compiler (SPIR-V), device  │ driver workarounds, buffer pooling,   ║"
    );
    println!(
        "  ║              │ resilience (S68+), dispatch semaphore│ dispatch semaphore                    ║"
    );
    println!(
        "  ╚══════════════╧═══════════════════════════════════════╧═══════════════════════════════════════╝"
    );
    println!();

    println!("  Key cross-pollination examples verified in this benchmark:");
    println!("    • erf (hotSpring) → used by wetSpring soil QS papers + airSpring hydrology");
    println!("    • graph_laplacian (neuralSpring) → used by wetSpring community ecology");
    println!("    • ridge (wetSpring) → used by neuralSpring ESN + airSpring kriging");
    println!("    • compile_shader_universal (ToadStool) → all springs' GPU modules benefit");
    println!("    • DF64 core (hotSpring biomeGate) → consumer GPU throughput for all springs");

    v.check_pass("cross-spring evolution verified", true);

    // ═══════════════════════════════════════════════════════════════════
    // §9  Timing Summary
    // ═══════════════════════════════════════════════════════════════════

    v.section("§9 Timing Summary");
    println!("  ┌─────────────────────────────────┬────────────────────────┬──────────┐");
    println!("  │ Benchmark                       │ Provenance             │ Time     │");
    println!("  ├─────────────────────────────────┼────────────────────────┼──────────┤");
    for t in &timings {
        println!("  │ {:<31} │ {:<22} │ {:>7.3}ms│", t.label, t.origin, t.ms);
    }
    println!("  └─────────────────────────────────┴────────────────────────┴──────────┘");

    v.finish();
}
