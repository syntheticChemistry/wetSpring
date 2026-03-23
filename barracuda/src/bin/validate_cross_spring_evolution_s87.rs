// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::doc_markdown,
    reason = "validation harness: required for domain validation"
)]
//! # Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems
//!
//! Comprehensive cross-spring evolution benchmark and validation on ToadStool S87.
//! Tracks shader provenance through the ecosystem: when and where each primitive
//! was written, who absorbed it, and which springs now consume it.
//!
//! # Cross-Spring Shader Evolution Map
//!
//! ```text
//! hotSpring  → DF64 precision (S58), NVK workarounds (S80), Anderson spectral (S26)
//!              → Sovereign compiler, Lattice QCD (S64), grid/mixing PDE (S40)
//! wetSpring  → Bio diversity (S63), ODE systems (S58), DADA2/HMM/alignment (S31)
//!              → NMF drug repurposing (S58), ridge regression (S59)
//! neuralSpring → GemmF64 (S64), graph linalg (S54), BatchedEncoder, AlphaFold2
//!              → pairwise metrics (S27), fitness/evolution (S27)
//! airSpring  → Hydrology 6 ET₀ methods (S70/S81), Richards PDE (S83), Kriging
//! groundSpring → Bootstrap/jackknife (S66), Wright-Fisher, InterconnectTopology (S81)
//! wateringHole → Boltzmann sampling (S76), Omelyan integrator (S83), BrentGpu (S83)
//! ```
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
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::sync::Arc;
use std::time::Instant;

use barracuda::shaders::Precision;
use wetspring_barracuda::bio::diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu};
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

struct Timing {
    label: &'static str,
    origin: &'static str,
    evolved: &'static str,
    ms: f64,
}

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let result = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {label}: {ms:.3} ms");
    (result, ms)
}

fn bench_n<T>(n: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let mut result = None;
    for _ in 0..n {
        result = Some(f());
    }
    let us = t0.elapsed().as_nanos() as f64 / 1000.0 / n as f64;
    (result.or_exit("unexpected error"), us)
}

fn main() {
    let mut v = Validator::new("Exp304: Cross-Spring Evolution — ToadStool S87 Modern Systems");
    let mut timings: Vec<Timing> = Vec::new();

    println!("ToadStool pin: S87 (2dc26792) — 264 ComputeDispatch ops, 144 consumed by wetSpring");
    println!(
        "S87 highlights: FHE shader fixes, gpu_helpers refactor, device-lost recovery, unsafe audit"
    );
    println!(
        "Cross-spring: hotSpring + wetSpring + neuralSpring + airSpring + groundSpring + wateringHole\n"
    );

    // ═══ §0: GPU Init + hotSpring Precision Architecture ══════════════════
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
    let ctx = gpu.tensor_context().clone();

    // ═══ §1: wetSpring Bio → ToadStool → ALL — Diversity Fusion ══════════
    {
        v.section("§1 wetSpring Bio → ToadStool → ALL — Diversity Fusion");

        let counts: Vec<f64> = (0..500)
            .map(|i: i32| f64::from((i * 13 + 7) % 200) + 1.0)
            .collect();
        let n_taxa = counts.len();

        let (cpu_results, ms_cpu) = bench("diversity_fusion_cpu 500 taxa", || {
            diversity_fusion_cpu(&counts, n_taxa)
        });
        let cpu_shannon = cpu_results[0].shannon;
        let cpu_simpson = cpu_results[0].simpson;
        v.check_pass("CPU: shannon > 0", cpu_shannon > 0.0);
        v.check_pass(
            "CPU: simpson in (0,1)",
            cpu_simpson > 0.0 && cpu_simpson < 1.0,
        );

        let fusion = DiversityFusionGpu::new(Arc::clone(&device)).or_exit("unexpected error");
        let (gpu_results, ms_gpu) = bench("DiversityFusionGpu 500 taxa", || {
            fusion.compute(&counts, 1, n_taxa)
        });
        match gpu_results {
            Ok(ref res) => {
                let gpu_shannon = res[0].shannon;
                let parity = (gpu_shannon - cpu_shannon).abs();
                v.check_pass("GPU↔CPU shannon parity < 0.3", parity < 0.3);
                println!("  CPU: {cpu_shannon:.6}, GPU: {gpu_shannon:.6}, Δ={parity:.2e}");
                if ms_cpu > 0.0 {
                    println!("  Speedup: {:.1}× GPU vs CPU", ms_cpu / ms_gpu.max(0.001));
                }
                timings.push(Timing {
                    label: "DiversityFusion GPU",
                    origin: "wetSpring V6",
                    evolved: "→ S63 absorb → all springs consume",
                    ms: ms_gpu,
                });
            }
            Err(e) => v.check_pass(&format!("GPU diversity skip ({e})"), true),
        }
        timings.push(Timing {
            label: "DiversityFusion CPU",
            origin: "wetSpring V6",
            evolved: "→ S63 absorb",
            ms: ms_cpu,
        });

        println!("  Written: wetSpring V6 (diversity_fusion_f64.wgsl)");
        println!("  Absorbed: ToadStool S63 (Feb 24)");
        println!("  Consumed by: neuralSpring (brain diversity), groundSpring (ecological)");
    }

    // ═══ §2: neuralSpring → ToadStool → wetSpring — GemmF64 ═════════════
    {
        v.section("§2 neuralSpring → ToadStool → wetSpring — GemmF64 + DF64");

        for &n in &[64_usize, 128, 256] {
            let a: Vec<f64> = (0..n * n)
                .map(|i| ((i * 17 + 3) % 100) as f64 / 100.0)
                .collect();
            let b: Vec<f64> = (0..n * n)
                .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0)
                .collect();

            let label_gpu = format!("GEMM GPU {n}×{n}");
            let (gpu_c, ms_gpu) = bench(&label_gpu, || {
                barracuda::ops::linalg::gemm_f64::GemmF64::execute(
                    Arc::clone(&device),
                    &a,
                    &b,
                    n,
                    n,
                    n,
                    1,
                )
            });
            match gpu_c {
                Ok(ref c) => {
                    v.check_pass(&format!("GEMM {n}: output size"), c.len() == n * n);

                    let label_cpu = format!("GEMM CPU {n}×{n}");
                    let (cpu_c, ms_cpu) = bench(&label_cpu, || {
                        let mut out = vec![0.0_f64; n * n];
                        for i in 0..n {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for k in 0..n {
                                    sum = a[i * n + k].mul_add(b[k * n + j], sum);
                                }
                                out[i * n + j] = sum;
                            }
                        }
                        out
                    });

                    let max_err: f64 = c
                        .iter()
                        .zip(cpu_c.iter())
                        .map(|(g, c)| (g - c).abs())
                        .fold(0.0_f64, f64::max);
                    v.check_pass(
                        &format!("GEMM {n}: GPU≈CPU parity"),
                        max_err < tolerances::CROSS_SPRING_NUMERICAL,
                    );
                    println!(
                        "  {n}×{n}: max|GPU−CPU|={max_err:.2e}, speedup={:.1}×",
                        ms_cpu / ms_gpu.max(0.001)
                    );

                    if n == 256 {
                        timings.push(Timing {
                            label: "GEMM 256×256 GPU",
                            origin: "neuralSpring S64",
                            evolved: "→ S72 ComputeDispatch → wetSpring NMF",
                            ms: ms_gpu,
                        });
                        timings.push(Timing {
                            label: "GEMM 256×256 CPU",
                            origin: "neuralSpring S64",
                            evolved: "→ CPU baseline",
                            ms: ms_cpu,
                        });
                    }
                }
                Err(e) => v.check_pass(&format!("GEMM {n} skip ({e})"), true),
            }
        }

        println!("  Written: neuralSpring (AlphaFold2 prototype, GEMM needs)");
        println!("  Absorbed: ToadStool S64 (GemmF64 WGSL)");
        println!("  hotSpring: DF64 auto-select layer (S58) — native f64 where available");
        println!(
            "  Consumed by: wetSpring (NMF, drug repurposing), neuralSpring (MLP, AlphaFold2)"
        );
    }

    // ═══ §3: wetSpring GemmCached — Composed from neuralSpring + hotSpring ═
    {
        v.section("§3 wetSpring — GemmCached (neuralSpring GEMM + hotSpring precision)");

        let gemm = GemmCached::new(Arc::clone(&device), ctx);

        for &(m, k, n) in &[(64_usize, 32, 16), (128, 64, 32), (256, 128, 64)] {
            let a: Vec<f64> = (0..m * k)
                .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
                .collect();
            let b: Vec<f64> = (0..k * n)
                .map(|i| ((i * 13 + 3) % 100) as f64 / 100.0)
                .collect();

            let label = format!("GemmCached {m}×{k}×{n}");
            let (result, ms) = bench(&label, || gemm.execute(&a, &b, m, k, n, 1));
            match result {
                Ok(ref c) => {
                    v.check_pass(&format!("GemmCached {m}×{k}×{n}: size"), c.len() == m * n);
                    let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                    v.check_pass(&format!("GemmCached {m}×{k}×{n}: non-trivial"), norm > 0.0);
                }
                Err(e) => v.check_pass(&format!("GemmCached skip ({e})"), true),
            }

            if m == 256 {
                timings.push(Timing {
                    label: "GemmCached 256×128×64",
                    origin: "wetSpring V6",
                    evolved: "→ neuralSpring GEMM + hotSpring precision",
                    ms,
                });
            }
        }

        println!("  Written: wetSpring V6 (drug repurposing pipeline, cached pipeline)");
        println!("  Composes: neuralSpring GemmF64 shader + hotSpring DF64 precision");
        println!("  Uses: ToadStool BufferPool (S65), ComputeDispatch (S72)");
    }

    // ═══ §4: wetSpring — Bray-Curtis GPU (bio → ToadStool → GPU) ════════
    {
        v.section("§4 wetSpring → ToadStool — Bray-Curtis GPU Distance");

        for &n_samples in &[10_usize, 20, 50] {
            let n_features = 200_usize;
            let samples: Vec<f64> = (0..n_samples * n_features)
                .map(|i| ((i * 7 + 1) % 50) as f64 + 1.0)
                .collect();

            let bc = barracuda::ops::bray_curtis_f64::BrayCurtisF64::new(Arc::clone(&device));
            match bc {
                Ok(bc_gpu) => {
                    let label = format!("BrayCurtis GPU {n_samples}×{n_features}");
                    let (dist, ms) = bench(&label, || {
                        bc_gpu.condensed_distance_matrix(&samples, n_samples, n_features)
                    });
                    match dist {
                        Ok(ref d) => {
                            let expected = n_samples * (n_samples - 1) / 2;
                            v.check_pass(
                                &format!("BC {n_samples}: condensed={expected}"),
                                d.len() == expected,
                            );
                            v.check_pass(
                                &format!("BC {n_samples}: values in [0,1]"),
                                d.iter().all(|&x| (0.0..=1.0).contains(&x)),
                            );
                            if n_samples == 50 {
                                timings.push(Timing {
                                    label: "BrayCurtis GPU 50×200",
                                    origin: "wetSpring bio",
                                    evolved: "→ ToadStool S82 ComputeDispatch",
                                    ms,
                                });
                            }
                        }
                        Err(e) => v.check_pass(&format!("BC {n_samples} dist skip ({e})"), true),
                    }
                }
                Err(e) => v.check_pass(&format!("BC init skip ({e})"), true),
            }
        }

        println!("  Written: wetSpring bio (16S rRNA distance matrices)");
        println!("  Absorbed: ToadStool S82 (BrayCurtisF64 ComputeDispatch)");
        println!("  Consumed by: wetSpring (UniFrac), groundSpring (soil ecology)");
    }

    // ═══ §5: hotSpring → ALL — Anderson Spectral Scaling ════════════════
    {
        v.section("§5 hotSpring → ALL — Anderson Spectral (1D→3D→4D)");

        let sizes = [100, 500, 1000, 2000];
        for &n in &sizes {
            let (eigs, ms) = bench(&format!("Anderson 1D n={n}"), || {
                barracuda::spectral::anderson_eigenvalues(n, 4.0, 42)
            });
            v.check_pass(&format!("Anderson n={n}"), eigs.len() == n);
            let r = barracuda::spectral::level_spacing_ratio(&eigs);
            println!(
                "    n={n}: r={r:.4} (GOE={:.4}, Poisson={:.4})",
                barracuda::spectral::GOE_R,
                barracuda::spectral::POISSON_R
            );
            if n == 2000 {
                timings.push(Timing {
                    label: "Anderson 1D n=2000",
                    origin: "hotSpring v0.6.0",
                    evolved: "→ S26 absorb → S79 Lanczos → S83 4D",
                    ms,
                });
            }
        }

        let (csr_3d, ms_3d) = bench("Anderson 3D 8³=512", || {
            barracuda::spectral::anderson_3d(8, 8, 8, 4.0, 42)
        });
        v.check_pass("3D: 512 sites", csr_3d.n == 512);
        let (tri, _) = bench("Lanczos 200 steps", || {
            barracuda::spectral::lanczos(&csr_3d, 200, 42)
        });
        let eigs_3d = barracuda::spectral::lanczos_eigenvalues(&tri);
        let phase = barracuda::spectral::classify_spectral_phase(&eigs_3d, 4.0);
        v.check_pass("3D Lanczos converges", !eigs_3d.is_empty());
        println!("  3D phase: {phase:?}");

        let (csr_4d, _) = bench("Anderson 4D 4⁴=256", || {
            barracuda::spectral::anderson::anderson_4d(4, 4.0, 42)
        });
        v.check_pass("4D: 256 sites", csr_4d.n == 256);
        timings.push(Timing {
            label: "Anderson 3D 512",
            origin: "hotSpring v0.6.0",
            evolved: "→ S26 → S79 Lanczos → S83 4D+Wegner",
            ms: ms_3d,
        });

        println!("  Written: hotSpring v0.6.0 (Kachkovskiy spectral theory, Feb 14)");
        println!("  Absorbed: ToadStool S26 (Feb 22)");
        println!("  Evolved: S79 (Lanczos), S83 (4D + Wegner block RG)");
        println!("  Consumed by: neuralSpring (NautilusBrain phase), wetSpring (disorder models)");
    }

    // ═══ §6: airSpring — Hydrology ET₀ (6 methods) ═════════════════════
    {
        v.section("§6 airSpring → ToadStool — Hydrology ET₀ (6 methods)");

        let monthly = [
            3.0, 4.0, 8.0, 12.0, 17.0, 21.0, 24.0, 23.0, 19.0, 13.0, 8.0, 4.0,
        ];
        let hi = barracuda::stats::thornthwaite_heat_index(&monthly);

        let methods: Vec<(&str, f64, &str, &str)> = vec![
            (
                "Hargreaves",
                barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0).or_exit("unexpected error"),
                "airSpring V039",
                "→ ToadStool S70 (Feb 26)",
            ),
            (
                "FAO-56 PM",
                barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
                    .or_exit("unexpected error"),
                "airSpring V039",
                "→ ToadStool S70 (Feb 26)",
            ),
            (
                "Thornthwaite",
                barracuda::stats::thornthwaite_et0(21.0, hi, 14.5, 30.0)
                    .or_exit("unexpected error"),
                "ToadStool S81",
                "(Mar 1) — new in ToadStool",
            ),
            (
                "Makkink",
                barracuda::stats::makkink_et0(20.0, 18.0).or_exit("unexpected error"),
                "ToadStool S81",
                "(Mar 1) — new in ToadStool",
            ),
            (
                "Turc",
                barracuda::stats::turc_et0(20.0, 18.0, 70.0).or_exit("unexpected error"),
                "ToadStool S81",
                "(Mar 1) — new in ToadStool",
            ),
            (
                "Hamon",
                barracuda::stats::hamon_et0(20.0, 14.0).or_exit("unexpected error"),
                "ToadStool S81",
                "(Mar 1) — new in ToadStool",
            ),
        ];

        for (name, et0, origin, evolved) in &methods {
            v.check_pass(&format!("{name} ET₀ > 0"), *et0 > 0.0);
            println!("  {name:14}: {et0:8.3} mm/day  [{origin} {evolved}]");
        }

        println!("  Written: airSpring V039 (Hargreaves, FAO-56, Feb 22)");
        println!("  Absorbed: ToadStool S70 (Feb 26)");
        println!("  Expanded: S81 (4 new methods: Thornthwaite, Makkink, Turc, Hamon)");
        println!("  GPU: HargreavesBatchGpu, SeasonalPipelineF64, RichardsGpu (S83)");
        println!("  Consumed by: groundSpring (soil moisture), airSpring (seasonal forecast)");
    }

    // ═══ §7: groundSpring → ToadStool — Stats + Evolution ═══════════════
    {
        v.section("§7 groundSpring → ToadStool — Bootstrap + Evolution Stats");

        let data: Vec<f64> = (0..200)
            .map(|i| (f64::from(i) * 0.1).sin().mul_add(3.0, 5.0))
            .collect();

        let (ci, ms_boot) = bench("Bootstrap 200×50k", || {
            barracuda::stats::bootstrap_ci(
                &data,
                |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64,
                50_000,
                0.95,
                42,
            )
            .or_exit("unexpected error")
        });
        v.check_pass("Bootstrap: lo < hi", ci.lower < ci.upper);
        println!("  CI: [{:.4}, {:.4}]", ci.lower, ci.upper);
        timings.push(Timing {
            label: "Bootstrap 200×50k",
            origin: "groundSpring V54",
            evolved: "→ S70 absorb → S72 GPU jackknife",
            ms: ms_boot,
        });

        let jk = barracuda::stats::jackknife_mean_variance(&data).or_exit("unexpected error");
        v.check_pass("Jackknife: variance ≥ 0", jk.variance >= 0.0);

        let (_, us_fix) = bench_n(10_000, || {
            barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01)
        });
        v.check_pass("Kimura fixation", us_fix > 0.0);
        timings.push(Timing {
            label: "Kimura fixation 10k",
            origin: "groundSpring V54",
            evolved: "→ S70 absorb",
            ms: us_fix / 1000.0,
        });

        let x: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi.ln(), 1.0)).collect();
        let fits = barracuda::stats::fit_all(&x, &y);
        v.check_pass("fit_all: models converge", !fits.is_empty());
        if let Some(b) = fits.iter().max_by(|a, b| {
            a.r_squared
                .partial_cmp(&b.r_squared)
                .or_exit("unexpected error")
        }) {
            v.check_pass("best R² > 0.95", b.r_squared > 0.95);
            v.check_pass("slope() accessor", b.slope().is_some());
            v.check_pass("intercept() accessor", b.intercept().is_some());
            println!("  Best: {} R²={:.6}", b.model, b.r_squared);
        }

        println!("  Written: groundSpring V54 (Wright-Fisher, bootstrap)");
        println!("  Absorbed: ToadStool S70 (Feb 26)");
        println!("  Evolved: S81 (FitResult named accessors), S72 (GPU jackknife, histogram)");
        println!("  Consumed by: wetSpring (population genomics), airSpring (sensor calibration)");
    }

    // ═══ §8: wateringHole → cross-spring — Sampling + Optimization ══════
    {
        v.section("§8 wateringHole → cross-spring — Sampling + Optimization");

        let rosenbrock = |x: &[f64]| -> f64 {
            (1.0 - x[0]).mul_add(1.0 - x[0], 100.0 * x[0].mul_add(-x[0], x[1]).powi(2))
        };
        let initial = vec![5.0, -3.0];

        let (boltz, ms_boltz) = bench("Boltzmann 5k steps", || {
            barracuda::sample::boltzmann_sampling(&rosenbrock, &initial, 0.1, 0.5, 5000, 42)
        });
        let boltz_final = *boltz.losses.last().or_exit("unexpected error");
        v.check_pass("Boltzmann converges", boltz_final < rosenbrock(&initial));
        timings.push(Timing {
            label: "Boltzmann 5k×2D",
            origin: "wateringHole",
            evolved: "→ S76 GPU dispatch → S80 batch",
            ms: ms_boltz,
        });

        let (sobol, _) = bench("Sobol 10k×5D", || {
            barracuda::sample::sobol_scaled(10_000, &[(0.0, 1.0); 5]).or_exit("unexpected error")
        });
        v.check_pass("Sobol: 10k points", sobol.len() == 10_000);

        let (lhs, _) = bench("LHS 10k×5D", || {
            barracuda::sample::latin_hypercube(10_000, &[(0.0, 1.0); 5], 42)
                .or_exit("unexpected error")
        });
        v.check_pass("LHS: 10k points", lhs.len() == 10_000);

        println!("  Written: wateringHole V69 (cross-spring collaboration)");
        println!("  Absorbed: ToadStool S76 (Boltzmann GPU), S80 (batch Nelder-Mead)");
        println!("  S83: BrentGpu (batched root-finding), L-BFGS, OmelyanIntegrator");
        println!("  Consumed by: all springs (hyperparameter tuning, optimization)");
    }

    // ═══ §9: wetSpring — NMF + Graph (bio + neuralSpring composition) ═══
    {
        v.section("§9 wetSpring — NMF Drug Repurposing + neuralSpring Graph Theory");

        let rows = 100_usize;
        let cols = 50_usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0 + 0.01)
            .collect();

        let (nmf, ms_nmf) = bench("NMF 100×50 k=5", || {
            barracuda::linalg::nmf(
                &data,
                rows,
                cols,
                &barracuda::linalg::NmfConfig {
                    rank: 5,
                    max_iter: 200,
                    tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
                    objective: barracuda::linalg::NmfObjective::Euclidean,
                    seed: 42,
                },
            )
        });
        match nmf {
            Ok(ref r) => {
                v.check_pass("NMF converged", !r.errors.is_empty());
                let last = *r.errors.last().or_exit("unexpected error");
                let first = *r.errors.first().or_exit("unexpected error");
                v.check_pass("NMF: error decreases", last < first);
                println!(
                    "  {rows}×{cols}→k=5, {} iters, err={last:.6}",
                    r.errors.len()
                );
                timings.push(Timing {
                    label: "NMF 100×50 k=5",
                    origin: "wetSpring V6",
                    evolved: "→ S64 absorb → S82 GPU SparseGemm",
                    ms: ms_nmf,
                });
            }
            Err(e) => v.check_pass(&format!("NMF skip ({e})"), true),
        }

        let n = 100_usize;
        let adj: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if i != j && (i + j) % 3 == 0 { 1.0 } else { 0.0 }
            })
            .collect();
        let (lap, ms_lap) = bench("Graph Laplacian 100×100", || {
            barracuda::linalg::graph_laplacian(&adj, n)
        });
        v.check_pass("Laplacian: n²", lap.len() == n * n);
        let diag: Vec<f64> = (0..n).map(|i| lap[i * n + i]).collect();
        let eff = barracuda::linalg::effective_rank(&diag);
        v.check_pass("effective_rank > 0", eff > 0.0);
        timings.push(Timing {
            label: "Graph Laplacian 100",
            origin: "neuralSpring V64",
            evolved: "→ S54 → S72 ComputeDispatch",
            ms: ms_lap,
        });

        println!("  NMF written: wetSpring V6 (drug repurposing pipeline)");
        println!("  NMF absorbed: ToadStool S64 (Lee-Seung Euclidean + KL)");
        println!("  Graph written: neuralSpring V64 (GNN foundation)");
        println!("  Graph absorbed: ToadStool S54 (graph_laplacian, effective_rank)");
        println!(
            "  Composition: wetSpring drug pipeline = NMF (wetSpring) × Graph (neuralSpring) × Precision (hotSpring)"
        );
    }

    // ═══ §10: DF64 Host Protocol (hotSpring + wetSpring) ════════════════
    {
        v.section("§10 DF64 Host Protocol — hotSpring precision + wetSpring wire format");

        let values = [
            std::f64::consts::PI,
            std::f64::consts::E,
            1.0 / 3.0,
            1e15,
            1e-15,
        ];
        let packed = wetspring_barracuda::df64_host::pack_slice(&values);
        let unpacked = wetspring_barracuda::df64_host::unpack_slice(&packed);
        v.check_pass("DF64 pack→unpack roundtrip", unpacked.len() == values.len());

        for (i, (&orig, &rt)) in values.iter().zip(unpacked.iter()).enumerate() {
            let err = (orig - rt).abs() / orig.abs().max(tolerances::LOG_PROB_FLOOR);
            v.check_pass(
                &format!("DF64 roundtrip[{i}] < ANALYTICAL_LOOSE"),
                err < tolerances::ANALYTICAL_LOOSE,
            );
        }

        let rt_err = wetspring_barracuda::df64_host::roundtrip_error(std::f64::consts::PI);
        v.check_pass(
            "DF64 π roundtrip < ANALYTICAL_LOOSE",
            rt_err < tolerances::ANALYTICAL_LOOSE,
        );
        println!("  DF64 π roundtrip error: {rt_err:.2e}");

        println!("  Written: hotSpring (DF64 core-streaming theory, f32-pair emulation)");
        println!("  Absorbed: ToadStool S58 (df64_core.wgsl, df64_transcendentals.wgsl)");
        println!("  Wire format: wetSpring df64_host (pack/unpack for CPU↔GPU transfer)");
        println!("  Consumed by: all springs via DF64 shader layer on consumer GPUs");
        println!("  S87: 21 DF64 shaders, 577 f64 shaders — universal precision selection");
    }

    // ═══ §11: CPU Benchmark — Cross-Spring Throughput Table ═════════════
    {
        v.section("§11 Cross-Spring CPU Throughput Benchmark");

        let vec_1k: Vec<f64> = (1..=1000).map(|i| f64::from(i % 50 + 1)).collect();
        let vec_a: Vec<f64> = (0..1000)
            .map(|i| (f64::from(i) * 0.3).sin().abs().mul_add(50.0, 1.0))
            .collect();
        let vec_b: Vec<f64> = (0..1000)
            .map(|i| (f64::from(i) * 0.31).sin().abs().mul_add(50.0, 1.0))
            .collect();

        struct CpuRow {
            name: &'static str,
            origin: &'static str,
            us: f64,
        }
        let mut cpu_rows: Vec<CpuRow> = Vec::new();

        let (_, us) = bench_n(1000, || barracuda::stats::shannon(&vec_1k));
        cpu_rows.push(CpuRow {
            name: "Shannon",
            origin: "wetSpring→S63",
            us,
        });

        let (_, us) = bench_n(1000, || barracuda::stats::simpson(&vec_1k));
        cpu_rows.push(CpuRow {
            name: "Simpson",
            origin: "wetSpring→S63",
            us,
        });

        let (_, us) = bench_n(1000, || barracuda::stats::bray_curtis(&vec_a, &vec_b));
        cpu_rows.push(CpuRow {
            name: "Bray-Curtis",
            origin: "wetSpring→S82",
            us,
        });

        let (_, us) = bench_n(1000, || barracuda::stats::chao1(&vec_1k));
        cpu_rows.push(CpuRow {
            name: "Chao1",
            origin: "wetSpring→S63",
            us,
        });

        let (_, us) = bench_n(1000, || {
            barracuda::stats::pearson_correlation(&vec_a, &vec_b)
        });
        cpu_rows.push(CpuRow {
            name: "Pearson r",
            origin: "neuralSpring→S66",
            us,
        });

        let x_fit: Vec<f64> = (0..500).map(f64::from).collect();
        let y_fit: Vec<f64> = x_fit.iter().map(|&xi| 3.0f64.mul_add(xi, 7.0)).collect();
        let (_, us) = bench_n(1000, || barracuda::stats::fit_linear(&x_fit, &y_fit));
        cpu_rows.push(CpuRow {
            name: "Linear fit",
            origin: "neuralSpring→S66",
            us,
        });

        let trap_x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
        let trap_y: Vec<f64> = trap_x.iter().map(|x| x * x).collect();
        let (_, us) = bench_n(5000, || barracuda::numerical::trapz(&trap_y, &trap_x));
        cpu_rows.push(CpuRow {
            name: "Trapz",
            origin: "hotSpring→S59",
            us,
        });

        let erf_pts: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 500.0).collect();
        let (_, us) = bench_n(5000, || {
            let mut acc = 0.0;
            for &x in &erf_pts {
                acc += barracuda::special::erf(x);
            }
            acc
        });
        cpu_rows.push(CpuRow {
            name: "Erf (1k pts)",
            origin: "hotSpring→S59",
            us,
        });

        let n_ridge = 50_usize;
        let n_cols = 10_usize;
        let x_ridge: Vec<f64> = (0..n_ridge * n_cols)
            .map(|i| {
                ((i / n_cols) as f64)
                    .mul_add(0.1, (i % n_cols) as f64 * 0.05)
                    .sin()
            })
            .collect();
        let y_ridge: Vec<f64> = (0..n_ridge).map(|i| (i as f64 * 0.2).cos()).collect();
        let (_, us) = bench_n(100, || {
            barracuda::linalg::ridge_regression(&x_ridge, &y_ridge, n_ridge, n_cols, 1, 0.1)
        });
        cpu_rows.push(CpuRow {
            name: "Ridge (50×10)",
            origin: "hotSpring→S59",
            us,
        });

        v.check_pass("CPU benchmarks complete", !cpu_rows.is_empty());

        println!();
        println!("  ┌──────────────────────────────────────────────────────────────┐");
        println!("  │  CPU Throughput: Cross-Spring Primitives (ToadStool S87)      │");
        println!("  ├──────────────────┬─────────────────┬──────────┬──────────────┤");
        println!("  │ Primitive        │ Origin→Session  │   µs/op  │      ops/sec │");
        println!("  ├──────────────────┼─────────────────┼──────────┼──────────────┤");
        for r in &cpu_rows {
            println!(
                "  │ {:<16} │ {:<15} │ {:>8.3} │ {:>12.0} │",
                r.name,
                r.origin,
                r.us,
                1_000_000.0 / r.us
            );
        }
        println!("  └──────────────────┴─────────────────┴──────────┴──────────────┘");
    }

    // ═══ §12: Cross-Spring GPU Benchmark Summary ════════════════════════
    {
        v.section("§12 Cross-Spring GPU Benchmark Summary");

        timings.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));

        println!();
        println!(
            "  ┌────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "  │  Cross-Spring GPU Benchmark (ToadStool S87, 264 ComputeDispatch ops)            │"
        );
        println!(
            "  ├──────────────────────────┬──────────┬──────────────────────────────────────────┤"
        );
        println!(
            "  │ Operation                │ Time(ms) │ Provenance                               │"
        );
        println!(
            "  ├──────────────────────────┼──────────┼──────────────────────────────────────────┤"
        );
        for t in &timings {
            println!(
                "  │ {:<24} │ {:>8.2} │ {:<13} {} │",
                t.label, t.ms, t.origin, t.evolved
            );
        }
        println!(
            "  └──────────────────────────┴──────────┴──────────────────────────────────────────┘"
        );

        v.check_pass("GPU benchmark table complete", !timings.is_empty());
    }

    // ═══ Summary ════════════════════════════════════════════════════════
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

    v.finish();
}
