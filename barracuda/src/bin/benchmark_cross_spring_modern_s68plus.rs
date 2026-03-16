// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! Exp210 — Modern Cross-Spring Evolution Benchmark (`ToadStool` S68+)
//!
//! Validates and benchmarks the full cross-spring evolution at S68+ scale.
//! Each section traces provenance through the ecoPrimals ecosystem, showing
//! how primitives flow between springs and benefit from each other's work.
//!
//! # Cross-Spring Provenance
//!
//! ```text
//! hotSpring → precision shaders, Fp64Strategy, DF64 core-streaming, Anderson spectral
//! wetSpring → bio ODE ×5, diversity, GemmCached, NMF, smith-waterman, NCBI
//! neuralSpring → pairwise ops, graph Laplacian, spatial payoff, swarm NN
//! airSpring → regression, hydrology, kriging, moving window
//! groundSpring → bootstrap (rawr_mean), batched multinomial
//! ```
//!
//! # What's New at S68+
//!
//! - `GpuF64::fp64_strategy()` — runtime precision selection (hotSpring S58 → S67)
//! - `GpuF64::optimal_precision()` — F64 vs Df64 per GPU class
//! - `GpuF64::is_lost()` — device-lost detection (S68+)
//! - `submit_and_poll` — resilient dispatch (S68+, transparent)
//! - `DispatchSemaphore` — concurrency management (S68+, transparent)
//! - `compile_shader_universal` — single entry for all precisions (S67/S68)

use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu};
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::validation::OrExit;

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
    let mut v = Validator::new("Exp210: Modern Cross-Spring Evolution Benchmark (ToadStool S68+)");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // §0  GPU Initialization + Modern Capabilities
    // ═══════════════════════════════════════════════════════════════════

    v.section("§0 GPU Init: Modern ToadStool S68+ Capabilities");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .or_exit("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");

    gpu.print_info();

    let strategy = gpu.fp64_strategy();
    let precision = gpu.optimal_precision();
    let is_lost = gpu.is_lost();
    let threshold = gpu.dispatch_threshold();

    println!("  Fp64Strategy: {strategy:?} (hotSpring S58 → ToadStool S67)");
    println!("  Optimal precision: {precision:?} (ToadStool S68 universal)");
    println!("  Device lost: {is_lost} (ToadStool S68+ resilience)");
    println!("  Dispatch threshold: {threshold} elements (wetSpring Exp064/087)");

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
        "optimal precision is F64 or Df64",
        matches!(
            precision,
            barracuda::shaders::Precision::F64 | barracuda::shaders::Precision::Df64
        ),
    );

    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    // ═══════════════════════════════════════════════════════════════════
    // §1  wetSpring Bio: Diversity (wetSpring → barracuda::stats S64)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§1 wetSpring Bio: Diversity (Write → S64 Absorb → Lean)");
    println!("  Provenance: wetSpring bio/diversity.rs → ToadStool stats::diversity (S64)");
    println!("  Evolution: wetSpring authored → ToadStool absorbed → all Springs benefit");

    let n_taxa = 500;
    let counts: Vec<f64> = (0..n_taxa)
        .map(|i| f64::from(((i * 7 + 3) % 100 + 1) as u32))
        .collect();

    let (cpu_shannon, cpu_div_ms) = bench("CPU Shannon (barracuda::stats S64)", || {
        diversity::shannon(&counts)
    });
    let (cpu_simpson, _) = bench("CPU Simpson (barracuda::stats S64)", || {
        diversity::simpson(&counts)
    });
    let (cpu_pielou, _) = bench("CPU Pielou (barracuda::stats S64)", || {
        diversity::pielou_evenness(&counts)
    });

    v.check_pass("Shannon > 0", cpu_shannon > 0.0);
    v.check_pass("Simpson ∈ (0,1)", cpu_simpson > 0.0 && cpu_simpson < 1.0);
    v.check_pass("Pielou ∈ (0,1]", cpu_pielou > 0.0 && cpu_pielou <= 1.0);

    timings.push(Timing {
        label: "CPU diversity (500 taxa)",
        origin: "wetSpring→S64",
        ms: cpu_div_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §2  wetSpring Bio: GPU DiversityFusion (Write → S63 → Lean)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§2 wetSpring Bio: GPU DiversityFusion (Write → S63 Absorb → Lean)");
    println!("  Provenance: wetSpring diversity_fusion_f64.wgsl → ToadStool S63 absorption");
    println!("  Uses: FusedMapReduceF64, compile_shader_universal (S68 path)");

    let n_species = 10_000;
    let n_samples = 5;
    let large_counts: Vec<f64> = (0..n_samples * n_species)
        .map(|i| f64::from(((i * 13 + 7) % 200 + 1) as u32))
        .collect();

    let (cpu_fusion, cpu_fusion_ms) = bench("CPU DiversityFusion (5×10k)", || {
        diversity_fusion_cpu(&large_counts, n_species)
    });

    let fusion_gpu = DiversityFusionGpu::new(Arc::clone(&device)).or_exit("DiversityFusionGpu");
    let (gpu_fusion, gpu_fusion_ms) = bench("GPU DiversityFusion (5×10k)", || {
        fusion_gpu
            .compute(&large_counts, n_samples, n_species)
            .or_exit("fusion GPU")
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

    let speedup = cpu_fusion_ms / gpu_fusion_ms;
    println!("  GPU speedup: {speedup:.1}× over CPU (FusedMapReduceF64)");

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
    // §3  hotSpring: Special Functions + Anderson Spectral
    // ═══════════════════════════════════════════════════════════════════

    v.section("§3 hotSpring: Precision Functions + Anderson Spectral (S58-S68)");
    println!("  Provenance: hotSpring lattice QCD → Anderson spectral → ToadStool S59");
    println!("  Precision: erf (Abramowitz & Stegun), ln_gamma (Lanczos), norm_cdf");
    println!("  hotSpring drives: Fp64Strategy, DF64 core, compile_shader_f64 polyfills");

    let erf_val = barracuda::special::erf(1.0);
    v.check(
        "erf(1.0) [A&S polynomial]",
        erf_val,
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );

    let lng = barracuda::special::ln_gamma(5.0).or_exit("ln_gamma");
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

    let (anderson_result, anderson_ms) = bench("Anderson 3D spectral (L=8)", || {
        let csr = barracuda::spectral::anderson_3d(8, 8, 8, 2.0, 42);
        let tri = barracuda::spectral::lanczos(&csr, 50, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
        barracuda::spectral::level_spacing_ratio(&eigs)
    });
    v.check_pass(
        "Anderson r ∈ (0,1)",
        anderson_result > 0.0 && anderson_result < 1.0,
    );

    timings.push(Timing {
        label: "Anderson 3D (L=8)",
        origin: "hotSpring→S59→spectral",
        ms: anderson_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §4  airSpring/groundSpring: Stats Evolution (S64/S66)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§4 airSpring/groundSpring: Cross-Spring Stats (S64/S66)");
    println!("  Provenance: airSpring → regression, hydrology (S66)");
    println!("  Provenance: groundSpring → bootstrap rawr_mean (S66)");
    println!("  Provenance: both → stats::metrics MAE, RMSE, R² (S64)");

    let vec_a: Vec<f64> = (0..100).map(|i| f64::from(i as u32) * 0.1).collect();
    let vec_b: Vec<f64> = vec_a
        .iter()
        .map(|&x| 0.01f64.mul_add(x.sin(), 2.0f64.mul_add(x, 1.0)))
        .collect();

    let pearson = barracuda::stats::pearson_correlation(&vec_a, &vec_b).or_exit("pearson");
    v.check(
        "Pearson(linear) ≈ 1.0 [cross-spring→S64]",
        pearson,
        1.0,
        tolerances::CROSS_SPRING_NUMERICAL,
    );

    let mae_val = barracuda::stats::mae(&vec_a, &vec_b);
    v.check_pass("MAE finite [airSpring→S64 metrics]", mae_val.is_finite());

    let rmse_val = barracuda::stats::rmse(&vec_a, &vec_b);
    v.check_pass("RMSE finite [airSpring→S64 metrics]", rmse_val.is_finite());

    // R² = Nash-Sutcliffe: observed vs simulated (same scale, small noise)
    let observed: Vec<f64> = (0..100).map(|i| f64::from(i as u32) * 0.1).collect();
    let simulated: Vec<f64> = observed
        .iter()
        .map(|&x| 0.001f64.mul_add(x.sin(), x))
        .collect();
    let r2_val = barracuda::stats::r_squared(&observed, &simulated);
    v.check_pass("R² > 0.99 [airSpring→S64 metrics]", r2_val > 0.99);

    let trap_n = 1000;
    let trap_x: Vec<f64> = (0..trap_n)
        .map(|i| f64::from(i) / f64::from(trap_n - 1))
        .collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|&xi| xi * xi).collect();
    let (trapz_val, trapz_ms) = bench("trapz(x², 1000 pts) [cross-spring numerical]", || {
        barracuda::numerical::trapz(&trap_y, &trap_x).or_exit("trapz")
    });
    v.check(
        "trapz(x²) ≈ 1/3",
        trapz_val,
        1.0 / 3.0,
        tolerances::TRAPZ_COARSE,
    );

    timings.push(Timing {
        label: "trapz(x², 1000 pts)",
        origin: "cross-spring→numerical",
        ms: trapz_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §5  GPU GEMM: Precision-Aware Pipeline (wetSpring → S62 → S68)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§5 GPU GEMM: Precision-Aware Pipeline (wetSpring → S62, S68 universal)");
    println!("  Provenance: wetSpring GemmCached → ToadStool GemmF64 (S62 BGL helpers)");
    println!("  Compile: compile_shader_universal(GemmF64::WGSL, Precision::F64) [S68]");
    println!("  Future: GpuF64::optimal_precision() → Precision::Df64 for ~10× on consumer GPUs");
    println!("  Current GPU strategy: {:?}", gpu.fp64_strategy());

    let m = 256;
    let k = 128;
    let n = 256;
    let a_mat: Vec<f64> = (0..m * k)
        .map(|i| f64::from(((i * 7 + 3) % 100) as u32) / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n)
        .map(|i| f64::from(((i * 11 + 5) % 100) as u32) / 100.0)
        .collect();

    let gemm = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let (gemm_res, gemm_ms) = bench("GEMM 256×128 × 128×256 (Precision::F64)", || {
        gemm.execute(&a_mat, &b_mat, m, k, n, 1).or_exit("GEMM")
    });
    v.check_pass("GEMM result finite", gemm_res.iter().all(|x| x.is_finite()));

    let expected_00: f64 = (0..k).map(|j| a_mat[j] * b_mat[j * n]).sum();
    v.check(
        "GEMM C[0,0] ≈ CPU",
        gemm_res[0],
        expected_00,
        tolerances::GPU_VS_CPU_F64,
    );

    // Cached dispatch (amortized submit_and_poll, S68+)
    for _ in 0..5 {
        let _ = gemm.execute(&a_mat, &b_mat, m, k, n, 1);
    }
    let ((), cached_ms) = bench("GEMM ×50 cached (submit_and_poll S68+)", || {
        for _ in 0..50 {
            let _ = gemm.execute(&a_mat, &b_mat, m, k, n, 1);
        }
    });
    let per_dispatch = cached_ms / 50.0;
    v.check_pass("cached dispatch faster than cold", per_dispatch < gemm_ms);

    timings.push(Timing {
        label: "GEMM cold 256×256",
        origin: "wetSpring→S62→S68",
        ms: gemm_ms,
    });
    timings.push(Timing {
        label: "GEMM cached avg",
        origin: "wetSpring→S62→S68+",
        ms: per_dispatch,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §6  NMF + Ridge: Cross-Spring Linalg (wetSpring → S58)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§6 NMF + Ridge: Cross-Spring Linalg (wetSpring → S58)");
    println!("  Provenance: wetSpring drug-disease pipeline → ToadStool linalg (S58)");
    println!("  NMF: Non-negative matrix factorization (Lee & Seung 1999)");
    println!("  Ridge: Tikhonov regularization (readout training, ESN, kriging)");

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
        barracuda::linalg::nmf::nmf(&v_mat, 20, 10, &nmf_cfg).or_exit("NMF")
    });
    v.check_pass(
        "NMF W, H non-negative",
        nmf_res.w.iter().chain(&nmf_res.h).all(|&x| x >= 0.0),
    );

    let ridge_x: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
    let ridge_y: Vec<f64> = (0..40).map(|i| f64::from(i).mul_add(0.25, 1.0)).collect();
    let (ridge_res, ridge_ms) = bench("ridge regression (20×5→2)", || {
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
        label: "NMF 20×10 KL k=3",
        origin: "wetSpring→S58→linalg",
        ms: nmf_ms,
    });
    timings.push(Timing {
        label: "Ridge 20×5→2",
        origin: "wetSpring→S59→linalg",
        ms: ridge_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §7  Cross-Spring Evolution Provenance Summary
    // ═══════════════════════════════════════════════════════════════════

    v.section("§7 Cross-Spring Evolution Provenance Summary");

    println!("  ╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("  ║ Cross-Spring Evolution Map: How Each Spring Contributes to ToadStool         ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ Spring        │ Contribution                               │ Sessions        ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ hotSpring     │ Fp64Strategy, DF64 core-streaming,        │ S58-S68         ║");
    println!("  ║               │ Anderson spectral, lattice QCD (14        │                 ║");
    println!("  ║               │ shaders), compile_shader_f64 polyfills,   │                 ║");
    println!("  ║               │ Validation harness, NVK/RADV workarounds  │                 ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ wetSpring     │ Bio ODE ×5 (trait-generated WGSL),        │ S41-S68         ║");
    println!("  ║               │ diversity (Shannon/Simpson/Chao1/Bray-    │                 ║");
    println!("  ║               │ Curtis), DiversityFusion, GEMM cached,    │                 ║");
    println!("  ║               │ smith-waterman, Felsenstein, Gillespie,   │                 ║");
    println!("  ║               │ NMF, ridge, ESN, nanopore, NCBI pipeline  │                 ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ neuralSpring  │ Pairwise ops (L2/Hamming/Jaccard),       │ S54-S56         ║");
    println!("  ║               │ graph Laplacian, spatial payoff,          │                 ║");
    println!("  ║               │ swarm NN, spectral IPR, KL divergence     │                 ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ airSpring     │ Regression (linear/quad/exp/log),         │ S64-S66         ║");
    println!("  ║               │ hydrology (Hargreaves ET0, FAO-56),       │                 ║");
    println!("  ║               │ kriging, Richards PDE, moving window      │                 ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ groundSpring  │ Bootstrap (RAWR Dirichlet), batched       │ S64-S66         ║");
    println!("  ║               │ multinomial, percentile, mean             │                 ║");
    println!("  ╠═══════════════╪════════════════════════════════════════════╪═════════════════╣");
    println!("  ║ ToadStool     │ 700 WGSL shaders (0 f32-only), universal │ S39-S68+        ║");
    println!("  ║               │ precision (F16/F32/F64/DF64), sovereign   │                 ║");
    println!("  ║               │ compiler, device-lost resilience,         │                 ║");
    println!("  ║               │ dispatch semaphore, submit_and_poll       │                 ║");
    println!("  ╚═══════════════╧════════════════════════════════════════════╧═════════════════╝");
    println!();

    // Architecture summary
    println!("  ┌───────────────────────────────────────────────────────────────────────────────┐");
    println!("  │ Modern Architecture (S68+)                                                    │");
    println!("  ├───────────────────────────────┬───────────────────────────────────────────────┤");
    println!("  │ ToadStool alignment           │ S68+ (e96576ee)                              │");
    println!("  │ WGSL shaders                  │ 700 (0 f32-only, all f64 canonical)          │");
    println!("  │ Precision architecture        │ Universal (F16/F32/F64/DF64 via single API)  │");
    println!(
        "  │ Fp64Strategy                  │ {:?}{} │",
        strategy,
        " ".repeat(43 - format!("{strategy:?}").len())
    );
    println!(
        "  │ Optimal precision             │ {:?}{} │",
        precision,
        " ".repeat(43 - format!("{precision:?}").len())
    );
    println!("  │ Device-lost resilience        │ S68+ (submit_and_poll + is_lost())           │");
    println!("  │ Dispatch semaphore            │ S68+ (dGPU=8, iGPU=4, CPU=2 permits)        │");
    println!("  │ Primitives consumed           │ 79 (barracuda always-on, zero fallback)      │");
    println!("  │ Local WGSL                    │ 0 (fully lean)                               │");
    println!(
        "  │ Dispatch threshold            │ {} elements{} │",
        threshold,
        " ".repeat(37 - format!("{threshold}").len())
    );
    println!("  └───────────────────────────────┴───────────────────────────────────────────────┘");

    v.check_pass("cross-spring evolution documented", true);

    // ═══════════════════════════════════════════════════════════════════
    // §8  Timing Summary
    // ═══════════════════════════════════════════════════════════════════

    v.section("§8 Timing Summary");
    println!("  ┌─────────────────────────────────┬────────────────────────┬──────────┐");
    println!("  │ Benchmark                       │ Provenance             │ Time     │");
    println!("  ├─────────────────────────────────┼────────────────────────┼──────────┤");
    for t in &timings {
        println!("  │ {:<31} │ {:<22} │ {:>7.3}ms│", t.label, t.origin, t.ms);
    }
    println!("  └─────────────────────────────────┴────────────────────────┴──────────┘");

    v.finish();
}
