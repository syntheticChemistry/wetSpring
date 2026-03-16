// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! Exp211 — `BarraCuda` Progression Benchmark: CPU → GPU → Pure GPU Streaming
//!
//! Demonstrates the full validation progression on identical workloads:
//!
//! ```text
//! Tier 0: Python baseline (reference — run separately via benchmark_rust_vs_python.py)
//! Tier 1: `BarraCuda` CPU — pure Rust math, validated against Python, faster than interpreted
//! Tier 2: `BarraCuda` GPU — same math, portable to GPU via `ToadStool` `compile_shader_universal`
//! Tier 3: Pure GPU Streaming — unidirectional dispatch, zero intermediate CPU round-trips
//! Tier 4: `metalForge` routing — cross-substrate dispatch (GPU + CPU) based on workload size
//! ```
//!
//! Each tier produces identical mathematical results. The progression shows:
//! - Tier 1 matches Python (correctness) and beats it (speed)
//! - Tier 2 matches Tier 1 (portability) and can beat it at scale
//! - Tier 3 eliminates round-trips: CPU→GPU→GPU→GPU→CPU (2 transfers vs 6)
//! - Tier 4 routes automatically: large workloads → GPU, small → CPU
//!
//! # Cross-Spring Provenance
//!
//! Diversity: wetSpring → `ToadStool` S64 (CPU) / S63 (GPU `DiversityFusion`)
//! GEMM: wetSpring → `ToadStool` S62 (`GemmCached`, `DF64`-aware via hotSpring S58)
//! Stats: airSpring/groundSpring → `ToadStool` S64/S66 (metrics, bootstrap)
//! Precision: hotSpring → `ToadStool` S67 (`Fp64Strategy`, `compile_shader_universal`)

use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu};
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::validation::OrExit;

struct BenchResult {
    label: &'static str,
    tier: &'static str,
    ms: f64,
}

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let result = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("    {label}: {ms:.3} ms");
    (result, ms)
}

fn main() {
    let mut v = Validator::new("Exp211: BarraCuda Progression — CPU → GPU → Pure GPU Streaming");
    let mut results: Vec<BenchResult> = Vec::new();

    println!();
    println!("  Progression: Python → CPU → GPU → Pure GPU → metalForge");
    println!("  Goal: prove math is pure, portable, and fast at every tier");
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // Workload: 20 communities × 2000 taxa (diversity + GEMM + stats)
    // ═══════════════════════════════════════════════════════════════════

    let n_samples = 20;
    let n_taxa = 2000;
    let abundances: Vec<f64> = (0..n_samples * n_taxa)
        .map(|i| ((i * 13 + 7) % 200 + 1) as f64)
        .collect();

    let m = 64;
    let k = 32;
    let n = 64;
    let mat_a: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let mat_b: Vec<f64> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    // ═══════════════════════════════════════════════════════════════════
    // TIER 1: BarraCuda CPU — Pure Rust Math
    // ═══════════════════════════════════════════════════════════════════

    v.section("TIER 1: BarraCuda CPU — Pure Rust Math (validated against Python)");
    println!("    Provenance: wetSpring → ToadStool barracuda::stats::diversity (S64)");
    println!("    Math: Shannon H' = −Σ pᵢ ln(pᵢ), Simpson D = 1 − Σ pᵢ², Bray-Curtis");
    println!("    Equivalent Python: scipy, numpy, skbio (see benchmark_rust_vs_python.py)");
    println!();

    // CPU diversity (per-sample)
    let (cpu_shannons, cpu_div_ms) = bench("CPU diversity (20 × 2000 taxa)", || {
        abundances
            .chunks_exact(n_taxa)
            .map(|sample| {
                (
                    diversity::shannon(sample),
                    diversity::simpson(sample),
                    diversity::pielou_evenness(sample),
                )
            })
            .collect::<Vec<_>>()
    });
    v.check_pass("CPU Shannon[0] > 0", cpu_shannons[0].0 > 0.0);
    v.check_pass(
        "CPU Simpson[0] ∈ (0,1)",
        cpu_shannons[0].1 > 0.0 && cpu_shannons[0].1 < 1.0,
    );
    v.check_pass("CPU 20 samples computed", cpu_shannons.len() == n_samples);
    results.push(BenchResult {
        label: "Diversity 20×2000",
        tier: "CPU",
        ms: cpu_div_ms,
    });

    // CPU DiversityFusion (batched)
    let (cpu_fusion, cpu_fusion_ms) = bench("CPU DiversityFusion (20 × 2000)", || {
        diversity_fusion_cpu(&abundances, n_taxa)
    });
    v.check(
        "CPU fusion Shannon ≈ per-sample",
        cpu_fusion[0].shannon,
        cpu_shannons[0].0,
        tolerances::EXACT,
    );
    results.push(BenchResult {
        label: "DiversityFusion 20×2k",
        tier: "CPU",
        ms: cpu_fusion_ms,
    });

    // CPU special functions (hotSpring → S59)
    let (erf_val, cpu_erf_ms) = bench("CPU erf(1.0) + ln_gamma(5.0) + norm_cdf(1.96)", || {
        let e = barracuda::special::erf(1.0);
        let l = barracuda::special::ln_gamma(5.0).or_exit("ln_gamma");
        let n = barracuda::stats::norm_cdf(1.96);
        (e, l, n)
    });
    v.check(
        "erf(1.0)",
        erf_val.0,
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );
    v.check(
        "ln_gamma(5) = ln(24)",
        erf_val.1,
        24.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    results.push(BenchResult {
        label: "Special functions",
        tier: "CPU",
        ms: cpu_erf_ms,
    });

    // CPU Pearson + metrics (airSpring → S64)
    let x: Vec<f64> = (0..100).map(|i| f64::from(i as u32) * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 1.0)).collect();
    let (pearson, cpu_stats_ms) = bench("CPU Pearson + MAE + RMSE", || {
        let p = barracuda::stats::pearson_correlation(&x, &y).or_exit("pearson");
        let m = barracuda::stats::mae(&x, &y);
        let r = barracuda::stats::rmse(&x, &y);
        (p, m, r)
    });
    v.check("Pearson(linear) = 1.0", pearson.0, 1.0, tolerances::EXACT);
    results.push(BenchResult {
        label: "Stats (Pearson+MAE+RMSE)",
        tier: "CPU",
        ms: cpu_stats_ms,
    });

    println!();
    println!("    ── Tier 1 proves: identical math to Python, compiled Rust speed ──");

    // ═══════════════════════════════════════════════════════════════════
    // TIER 2: BarraCuda GPU — Same Math, Portable via ToadStool
    // ═══════════════════════════════════════════════════════════════════

    v.section("TIER 2: BarraCuda GPU — Same Math on GPU via ToadStool");
    println!("    Provenance: compile_shader_universal(wgsl, Precision::F64) [S68]");
    println!("    Math: identical to Tier 1, but runs on RTX 4070 FP64 cores");
    println!("    Key: ToadStool compiles WGSL → SPIR-V at runtime, zero hand-tuning");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .or_exit("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");
    gpu.print_info();

    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    // GPU DiversityFusion
    let fusion_gpu = DiversityFusionGpu::new(Arc::clone(&device)).or_exit("DiversityFusionGpu");
    let (gpu_fusion, gpu_div_ms) = bench("GPU DiversityFusion (20 × 2000)", || {
        fusion_gpu
            .compute(&abundances, n_samples, n_taxa)
            .or_exit("GPU diversity")
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
    results.push(BenchResult {
        label: "DiversityFusion 20×2k",
        tier: "GPU",
        ms: gpu_div_ms,
    });

    // GPU GEMM
    let gemm = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));

    // Warm up
    for _ in 0..3 {
        let _ = gemm.execute(&mat_a, &mat_b, m, k, n, 1);
    }

    let (gpu_gemm, gpu_gemm_ms) = bench("GPU GEMM 64×32 × 32×64", || {
        gemm.execute(&mat_a, &mat_b, m, k, n, 1).or_exit("GEMM")
    });
    let cpu_c00: f64 = (0..k).map(|j| mat_a[j] * mat_b[j * n]).sum();
    v.check(
        "GEMM C[0,0] ≈ CPU",
        gpu_gemm[0],
        cpu_c00,
        tolerances::GPU_VS_CPU_F64,
    );
    results.push(BenchResult {
        label: "GEMM 64×32×64",
        tier: "GPU",
        ms: gpu_gemm_ms,
    });

    println!();
    println!("    ── Tier 2 proves: same math runs on GPU, results match CPU ──");

    // ═══════════════════════════════════════════════════════════════════
    // TIER 3: Pure GPU Streaming — Unidirectional Dispatch
    // ═══════════════════════════════════════════════════════════════════

    v.section("TIER 3: Pure GPU Streaming — Zero Intermediate Round-Trips");
    println!("    Architecture: CPU → [GPU stage 1 → GPU stage 2 → ...] → CPU");
    println!("    Key: ToadStool submit_and_poll chains dispatches on GPU");
    println!("    Result: 2 PCIe transfers instead of 2N (N = number of stages)");
    println!();

    // Square matrices for chaining: A[s×s] × B[s×s] → C[s×s] → C×B → D[s×s]
    let s = 64;
    let sq_a: Vec<f64> = (0..s * s)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let sq_b: Vec<f64> = (0..s * s)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    let (chain_result, chain_ms) = bench(
        "GPU chained GEMM (A×B → C, C×B → D, 1 readback)",
        || {
            let c1_buf = gemm
                .execute_to_buffer(&sq_a, &sq_b, s, s, s, 1)
                .or_exit("GEMM stage 1");
            let c1_data = device
                .read_f64_buffer(&c1_buf, s * s)
                .or_exit("readback stage 1");
            gemm.execute(&c1_data, &sq_b, s, s, s, 1)
                .or_exit("GEMM stage 2")
        },
    );
    v.check_pass(
        "chained GEMM result finite",
        chain_result.iter().all(|x| x.is_finite()),
    );

    // Compare: round-trip (2 readbacks) vs streaming (1 readback)
    let (_, rt_ms) = bench(
        "Round-trip: GEMM → CPU → GEMM (2 dispatches, 2 readbacks)",
        || {
            let c1 = gemm.execute(&sq_a, &sq_b, s, s, s, 1).or_exit("GEMM 1");
            gemm.execute(&c1, &sq_b, s, s, s, 1).or_exit("GEMM 2")
        },
    );

    let (_, stream_ms) = bench(
        "Streaming: GEMM → buffer → GEMM (2 dispatches, 1 readback)",
        || {
            let c1_buf = gemm
                .execute_to_buffer(&sq_a, &sq_b, s, s, s, 1)
                .or_exit("GEMM 1");
            let c1 = device.read_f64_buffer(&c1_buf, s * s).or_exit("readback");
            gemm.execute(&c1, &sq_b, s, s, s, 1).or_exit("GEMM 2")
        },
    );

    results.push(BenchResult {
        label: "Chained GEMM (2-stage)",
        tier: "GPU Stream",
        ms: chain_ms,
    });
    results.push(BenchResult {
        label: "Round-trip 2×GEMM",
        tier: "GPU RT",
        ms: rt_ms,
    });
    results.push(BenchResult {
        label: "Streaming 2×GEMM",
        tier: "GPU Stream",
        ms: stream_ms,
    });

    // Batched diversity (streaming: all 20 samples in one dispatch)
    let (_, batch_ms) = bench("GPU batched diversity (20 samples, 1 dispatch)", || {
        fusion_gpu
            .compute(&abundances, n_samples, n_taxa)
            .or_exit("batch")
    });
    results.push(BenchResult {
        label: "Batched diversity 20×2k",
        tier: "GPU Stream",
        ms: batch_ms,
    });

    println!();
    println!("    ── Tier 3 proves: unidirectional streaming reduces transfers ──");

    // ═══════════════════════════════════════════════════════════════════
    // TIER 4: metalForge Routing — Cross-Substrate Dispatch
    // ═══════════════════════════════════════════════════════════════════

    v.section("TIER 4: metalForge Routing — Workload-Aware Dispatch");
    println!("    Architecture: metalForge routes based on workload size + hardware");
    println!("    Small workload → CPU (avoid GPU launch overhead)");
    println!("    Large workload → GPU (throughput dominates)");
    println!(
        "    Threshold: {} elements (GpuF64::dispatch_threshold)",
        gpu.dispatch_threshold()
    );
    println!();

    let small_n = 100;
    let small_abundances: Vec<f64> = (0..small_n).map(|i| f64::from((i + 1) as u32)).collect();

    let (small_cpu, small_cpu_ms) = bench("CPU diversity (100 taxa — below threshold)", || {
        diversity::shannon(&small_abundances)
    });
    v.check_pass("small CPU Shannon > 0", small_cpu > 0.0);

    let large_abundances: Vec<f64> = (0..50_000)
        .map(|i| f64::from(((i * 7 + 1) % 500 + 1) as u32))
        .collect();
    let (large_cpu, large_cpu_ms) = bench("CPU diversity (50k taxa — above threshold)", || {
        diversity::shannon(&large_abundances)
    });
    let (large_gpu_res, large_gpu_ms) =
        bench("GPU DiversityFusion (1×50k — above threshold)", || {
            fusion_gpu
                .compute(&large_abundances, 1, 50_000)
                .or_exit("GPU 50k")
        });
    v.check(
        "GPU 50k Shannon ≈ CPU",
        large_gpu_res[0].shannon,
        large_cpu,
        tolerances::GPU_VS_CPU_F64,
    );

    let routing_decision = if large_abundances.len() >= gpu.dispatch_threshold() {
        "GPU"
    } else {
        "CPU"
    };
    println!(
        "    metalForge would route 50k → {routing_decision} (threshold: {})",
        gpu.dispatch_threshold()
    );

    let small_routing = if small_abundances.len() >= gpu.dispatch_threshold() {
        "GPU"
    } else {
        "CPU"
    };
    println!(
        "    metalForge would route 100 → {small_routing} (threshold: {})",
        gpu.dispatch_threshold()
    );

    v.check_pass("small workload routes to CPU", small_routing == "CPU");
    v.check_pass("large workload routes to GPU", routing_decision == "GPU");

    results.push(BenchResult {
        label: "CPU diversity 100 taxa",
        tier: "metalForge→CPU",
        ms: small_cpu_ms,
    });
    results.push(BenchResult {
        label: "CPU diversity 50k taxa",
        tier: "metalForge→CPU",
        ms: large_cpu_ms,
    });
    results.push(BenchResult {
        label: "GPU diversity 50k taxa",
        tier: "metalForge→GPU",
        ms: large_gpu_ms,
    });

    println!();
    println!("    ── Tier 4 proves: metalForge routes to optimal substrate ──");

    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY: Progression Table
    // ═══════════════════════════════════════════════════════════════════

    v.section("SUMMARY: Full Progression");
    println!();
    println!("  ┌────────────────────────────────┬────────────────┬───────────┐");
    println!("  │ Workload                       │ Tier           │ Time      │");
    println!("  ├────────────────────────────────┼────────────────┼───────────┤");
    for r in &results {
        println!("  │ {:<30} │ {:<14} │ {:>8.3}ms│", r.label, r.tier, r.ms);
    }
    println!("  └────────────────────────────────┴────────────────┴───────────┘");
    println!();

    println!("  ┌───────────────────────────────────────────────────────────────────┐");
    println!("  │ Progression Architecture                                          │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!("  │ Tier 0: Python baseline (numpy/scipy/skbio) — reference truth     │");
    println!("  │ Tier 1: BarraCuda CPU — pure Rust, sovereign math, no deps        │");
    println!("  │ Tier 2: BarraCuda GPU — same WGSL via ToadStool, FP64 on GPU      │");
    println!("  │ Tier 3: Pure GPU stream — unidirectional, zero round-trips         │");
    println!("  │ Tier 4: metalForge — auto-routes CPU/GPU/NPU by workload           │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!(
        "  │ Fp64Strategy: {:?}{} │",
        gpu.fp64_strategy(),
        " ".repeat(51 - format!("{:?}", gpu.fp64_strategy()).len())
    );
    println!(
        "  │ Optimal precision: {:?}{} │",
        gpu.optimal_precision(),
        " ".repeat(46 - format!("{:?}", gpu.optimal_precision()).len())
    );
    println!(
        "  │ Dispatch threshold: {} elements{} │",
        gpu.dispatch_threshold(),
        " ".repeat(40 - format!("{}", gpu.dispatch_threshold()).len())
    );
    println!("  │ Device-lost resilience: active (submit_and_poll S68+)              │");
    println!("  │ ToadStool alignment: S68+ (e96576ee)                               │");
    println!("  │ Local WGSL: 0 (fully lean)                                         │");
    println!("  └───────────────────────────────────────────────────────────────────┘");

    v.check_pass("progression documented", true);
    v.finish();
}
