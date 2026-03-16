// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::unwrap_used,
    reason = "validation harness: fail-fast on setup errors"
)]
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
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! # Exp249: Cross-Spring Evolution Benchmark — `ToadStool` S70+++ Provenance
//!
//! A comprehensive provenance-annotated benchmark showing how primitives evolved
//! across the ecoPrimals ecosystem:
//!
//! ```text
//! ┌────────────┐  precision, Fp64, Anderson  ┌─────────────┐
//! │ hotSpring   │ ─────────────────────────► │  `ToadStool`   │
//! │ (GPU forge) │                            │  (`BarraCuda`) │
//! └────────────┘                            └──────┬───────┘
//!                                                   │
//! ┌────────────┐  bio ODE, diversity, HMM    ┌──────┘
//! │ wetSpring   │ ◄─────────────────────────►│
//! │ (microbiome)│  consumes 90+ primitives    │
//! └────────────┘                              │
//!                                             │
//! ┌────────────┐  WrightFisher, HillGate,    │
//! │ neuralSpring│ StencilCoop, SwarmNN ──────►│
//! │ (neuro-evo) │  bootstrap, regression      │
//! └────────────┘                              │
//!                                             │
//! ┌────────────┐  jackknife, evolution,       │
//! │ groundSpring│ detection_power ───────────►│
//! │ (popgen)    │                             │
//! └────────────┘                              │
//!                                             │
//! ┌────────────┐  hydrology, FAO56, soil     │
//! │ airSpring   │ ──────────────────────────►│
//! │ (atmosphere)│                             │
//! └────────────┘
//! ```
//!
//! Each section benchmarks a domain, traces its cross-spring lineage,
//! and validates the primitive still produces correct output at S70+++.
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | `ToadStool` | S70+++ (`1dd7e338`) |
//! | Command | `cargo run --bin benchmark_cross_spring_evolution_s70` |

use std::time::Instant;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct ProvenanceTiming {
    domain: &'static str,
    primitive: &'static str,
    origin_spring: &'static str,
    absorbed_at: &'static str,
    cpu_us: f64,
    check_count: u32,
}

fn bench_us<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed().as_micros() as f64)
}

fn main() {
    let mut v = Validator::new("Exp249: Cross-Spring Evolution Benchmark (S70+++ Provenance)");
    let mut timings: Vec<ProvenanceTiming> = Vec::new();
    let t_total = Instant::now();

    // ═══════════════════════════════════════════════════════════════════════
    // §1  groundSpring → ToadStool S70: Population Genetics
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§1 groundSpring → S70: Kimura Fixation + Quasispecies");

    let (p_neutral, us_kimura) = bench_us(|| {
        let mut acc = 0.0;
        for _ in 0..10_000 {
            acc += barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01);
        }
        acc / 10_000.0
    });
    v.check(
        "Kimura neutral drift: P_fix = p0",
        p_neutral,
        0.01,
        tolerances::LIMIT_CONVERGENCE,
    );

    let (mu_c, us_thresh) = bench_us(|| {
        let mut acc = 0.0;
        for _ in 0..10_000 {
            acc += barracuda::stats::error_threshold(10.0, 100).unwrap_or(0.0);
        }
        acc / 10_000.0
    });
    let expected_mu_c = 1.0 - 10.0_f64.powf(-1.0 / 100.0);
    v.check(
        "Eigen error threshold",
        mu_c,
        expected_mu_c,
        tolerances::ANALYTICAL_F64,
    );
    timings.push(ProvenanceTiming {
        domain: "popgen",
        primitive: "kimura_fixation_prob",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_kimura / 10.0,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "popgen",
        primitive: "error_threshold",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_thresh / 10.0,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §2  groundSpring → ToadStool S70: Rare Biosphere Detection
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§2 groundSpring → S70: Detection Power/Threshold");

    let rare_probs = [0.001, 0.005, 0.01, 0.05];
    let ((), us_detect) = bench_us(|| {
        for _ in 0..10_000 {
            for &p in &rare_probs {
                let d = barracuda::stats::detection_threshold(p, 0.95);
                let pw = barracuda::stats::detection_power(p, d);
                std::hint::black_box((d, pw));
            }
        }
    });

    for &p in &rare_probs {
        let d = barracuda::stats::detection_threshold(p, 0.95);
        let pw = barracuda::stats::detection_power(p, d);
        v.check_pass(&format!("p={p}: power ≥ 0.95"), pw >= 0.95);
    }
    timings.push(ProvenanceTiming {
        domain: "rare biosphere",
        primitive: "detection_threshold+power",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_detect / 10.0,
        check_count: 4,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §3  groundSpring → ToadStool S70: Jackknife Resampling
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§3 groundSpring → S70: Jackknife");

    let jk_data: Vec<f64> = (0..200)
        .map(|i| (f64::from(i) * 0.37).sin().abs() * 10.0)
        .collect();
    let (jk_result, us_jk) = bench_us(|| {
        let mut last = None;
        for _ in 0..1_000 {
            last = barracuda::stats::jackknife_mean_variance(&jk_data);
        }
        last.unwrap()
    });
    v.check_pass("Jackknife: estimate > 0", jk_result.estimate > 0.0);
    v.check_pass("Jackknife: SE ≥ 0", jk_result.std_error >= 0.0);

    let (jk_shannon, us_jk_gen) = bench_us(|| {
        barracuda::stats::jackknife(&jk_data, |d| {
            let total: f64 = d.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            -d.iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| {
                    let p = x / total;
                    p * p.ln()
                })
                .sum::<f64>()
        })
        .unwrap()
    });
    v.check_pass(
        "Generalized jackknife Shannon > 0",
        jk_shannon.estimate > 0.0,
    );
    timings.push(ProvenanceTiming {
        domain: "resampling",
        primitive: "jackknife_mean_variance",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_jk,
        check_count: 2,
    });
    timings.push(ProvenanceTiming {
        domain: "resampling",
        primitive: "jackknife (generalized)",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_jk_gen,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §4  neuralSpring → ToadStool S54: Bootstrap CI
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§4 neuralSpring → S54: Bootstrap Confidence Intervals");

    let bs_data: Vec<f64> = (0..100).map(|i| 2.0 + (f64::from(i) * 0.1).sin()).collect();
    let (ci, us_bs) = bench_us(|| {
        barracuda::stats::bootstrap_ci(
            &bs_data,
            |d| d.iter().sum::<f64>() / d.len() as f64,
            10_000,
            0.95,
            42,
        )
        .unwrap()
    });
    v.check_pass("Bootstrap: CI lower < upper", ci.lower < ci.upper);
    v.check_pass("Bootstrap: SE > 0", ci.std_error > 0.0);
    v.check_pass("Bootstrap: n_bootstrap = 10000", ci.n_bootstrap == 10_000);

    let (rawr, us_rawr) =
        bench_us(|| barracuda::stats::rawr_mean(&bs_data, 5_000, 0.95, 77).unwrap());
    v.check_pass("RAWR: CI lower < upper", rawr.lower < rawr.upper);
    timings.push(ProvenanceTiming {
        domain: "confidence",
        primitive: "bootstrap_ci (10k)",
        origin_spring: "neuralSpring",
        absorbed_at: "S54",
        cpu_us: us_bs,
        check_count: 3,
    });
    timings.push(ProvenanceTiming {
        domain: "confidence",
        primitive: "rawr_mean (5k)",
        origin_spring: "neuralSpring",
        absorbed_at: "S54",
        cpu_us: us_rawr,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §5  neuralSpring → ToadStool S66: Regression / Growth Curves
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§5 neuralSpring → S66: Regression Suite");

    let x: Vec<f64> = (1..=100).map(f64::from).collect();
    let y_exp: Vec<f64> = x.iter().map(|&xi| 1.5 * (0.03 * xi).exp()).collect();
    let y_quad: Vec<f64> = x
        .iter()
        .map(|&xi| (0.02 * xi).mul_add(xi, -xi) + 10.0)
        .collect();
    let y_log: Vec<f64> = x.iter().map(|&xi| 8.0f64.mul_add(xi.ln(), 3.0)).collect();

    let (fe, us_exp) = bench_us(|| barracuda::stats::fit_exponential(&x, &y_exp));
    let (fq, us_quad) = bench_us(|| barracuda::stats::fit_quadratic(&x, &y_quad));
    let (fl, us_log) = bench_us(|| barracuda::stats::fit_logarithmic(&x, &y_log));

    v.check_pass(
        "Exponential fit: R² > 0.99",
        fe.as_ref().is_some_and(|f| f.r_squared > 0.99),
    );
    v.check_pass(
        "Quadratic fit: R² > 0.99",
        fq.as_ref().is_some_and(|f| f.r_squared > 0.99),
    );
    v.check_pass(
        "Logarithmic fit: R² > 0.99",
        fl.as_ref().is_some_and(|f| f.r_squared > 0.99),
    );

    let (all_fits, us_all) = bench_us(|| barracuda::stats::fit_all(&x, &y_log));
    let best = all_fits
        .iter()
        .max_by(|a, b| a.r_squared.partial_cmp(&b.r_squared).unwrap());
    v.check_pass(
        "fit_all selects logarithmic",
        best.is_some_and(|b| b.model == "logarithmic"),
    );
    timings.push(ProvenanceTiming {
        domain: "regression",
        primitive: "fit_exponential",
        origin_spring: "neuralSpring",
        absorbed_at: "S66",
        cpu_us: us_exp,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "regression",
        primitive: "fit_quadratic",
        origin_spring: "neuralSpring",
        absorbed_at: "S66",
        cpu_us: us_quad,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "regression",
        primitive: "fit_logarithmic",
        origin_spring: "neuralSpring",
        absorbed_at: "S66",
        cpu_us: us_log,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "regression",
        primitive: "fit_all (model select)",
        origin_spring: "neuralSpring",
        absorbed_at: "S66",
        cpu_us: us_all,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §6  wetSpring → ToadStool S64: Diversity + Chao1
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§6 wetSpring → S64/S70: Diversity Suite");

    let abundances = vec![
        10.0, 25.0, 3.0, 1.0, 42.0, 7.0, 15.0, 2.0, 8.0, 30.0, 1.0, 1.0, 50.0,
    ];
    let (shannon, us_sh) = bench_us(|| {
        let mut s = 0.0;
        for _ in 0..10_000 {
            s = barracuda::stats::shannon(&abundances);
        }
        s
    });
    v.check_pass("Shannon H' > 0", shannon > 0.0);

    let (simpson, us_si) = bench_us(|| {
        let mut s = 0.0;
        for _ in 0..10_000 {
            s = barracuda::stats::simpson(&abundances);
        }
        s
    });
    v.check_pass("Simpson ∈ (0, 1)", simpson > 0.0 && simpson < 1.0);

    let counts_u64: Vec<u64> = abundances.iter().map(|&a| a as u64).collect();
    let (chao1c, us_chao) = bench_us(|| {
        let mut chao_acc = 0.0;
        for _ in 0..10_000 {
            chao_acc = barracuda::stats::chao1_classic(&counts_u64);
        }
        chao_acc
    });
    let s_obs = counts_u64.iter().filter(|&&c| c > 0).count() as f64;
    v.check_pass("chao1_classic ≥ S_obs", chao1c >= s_obs);
    timings.push(ProvenanceTiming {
        domain: "diversity",
        primitive: "shannon",
        origin_spring: "wetSpring",
        absorbed_at: "S44",
        cpu_us: us_sh / 10.0,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "diversity",
        primitive: "simpson",
        origin_spring: "wetSpring",
        absorbed_at: "S44",
        cpu_us: us_si / 10.0,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "diversity",
        primitive: "chao1_classic (u64)",
        origin_spring: "groundSpring",
        absorbed_at: "S70",
        cpu_us: us_chao / 10.0,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §7  hotSpring → ToadStool S58-S67: Precision Infrastructure
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§7 hotSpring → S58-S67: Special Functions (precision)");

    let (erf_val, us_erf) = bench_us(|| {
        let mut e = 0.0;
        for _ in 0..100_000 {
            e = barracuda::special::erf(1.0);
        }
        e
    });
    v.check(
        "erf(1) ≈ 0.8427",
        erf_val,
        0.842_700_792_949_715,
        tolerances::GPU_VS_CPU_F64,
    );

    let (gamma_val, us_gamma) = bench_us(|| {
        let mut g = 0.0;
        for _ in 0..100_000 {
            g = barracuda::special::ln_gamma(5.0).unwrap();
        }
        g
    });
    v.check(
        "ln_Γ(5) = ln(24)",
        gamma_val,
        24.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    let (ncdf, us_norm) = bench_us(|| {
        let mut n = 0.0;
        for _ in 0..100_000 {
            n = barracuda::stats::norm_cdf(0.0);
        }
        n
    });
    v.check("Φ(0) = 0.5", ncdf, 0.5, tolerances::EXACT_F64);
    timings.push(ProvenanceTiming {
        domain: "special",
        primitive: "erf",
        origin_spring: "hotSpring",
        absorbed_at: "S58",
        cpu_us: us_erf / 100.0,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "special",
        primitive: "ln_gamma",
        origin_spring: "hotSpring",
        absorbed_at: "S58",
        cpu_us: us_gamma / 100.0,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "special",
        primitive: "norm_cdf",
        origin_spring: "hotSpring",
        absorbed_at: "S58",
        cpu_us: us_norm / 100.0,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §8  wetSpring → ToadStool S64: Linear Algebra (CPU paths)
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§8 wetSpring/neuralSpring → S64: Linear Algebra (CPU)");

    let n = 50;
    let mut adj = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            if (i + j) % 3 == 0 {
                adj[i * n + j] = 1.0;
                adj[j * n + i] = 1.0;
            }
        }
    }

    let (lap, us_lap) = bench_us(|| barracuda::linalg::graph_laplacian(&adj, n));
    v.check_pass("Laplacian: non-empty", !lap.is_empty());
    {
        let diag_sum: f64 = (0..n).map(|i| lap[i * n + i]).sum();
        let off_diag_sum: f64 = lap.iter().sum::<f64>() - diag_sum;
        v.check(
            "Laplacian row-sum ≈ 0",
            diag_sum + off_diag_sum,
            0.0,
            tolerances::LAPLACIAN_ROW_SUM,
        );
    }
    timings.push(ProvenanceTiming {
        domain: "linalg",
        primitive: "graph_laplacian (50×50)",
        origin_spring: "neuralSpring",
        absorbed_at: "S51",
        cpu_us: us_lap,
        check_count: 2,
    });

    let (ridge, us_ridge) = bench_us(|| {
        let x_mat: Vec<f64> = (0..300).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let y_vec: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.3).cos()).collect();
        barracuda::linalg::ridge_regression(&x_mat, &y_vec, 100, 3, 1, 0.1)
    });
    v.check_pass("Ridge regression: Ok", ridge.is_ok());
    timings.push(ProvenanceTiming {
        domain: "linalg",
        primitive: "ridge_regression (100×3)",
        origin_spring: "neuralSpring",
        absorbed_at: "S51",
        cpu_us: us_ridge,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §9  wetSpring → ToadStool S44-S64: Correlation + Metrics
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§9 wetSpring → S44: Correlation + Metrics");

    let a: Vec<f64> = (0..1000).map(|i| (f64::from(i) * 0.01).sin()).collect();
    let b: Vec<f64> = (0..1000)
        .map(|i| 0.01f64.mul_add((f64::from(i) * 0.1).cos(), (f64::from(i) * 0.01).sin()))
        .collect();

    let (pear, us_pear) = bench_us(|| {
        let mut p = 0.0;
        for _ in 0..1_000 {
            p = barracuda::stats::pearson_correlation(&a, &b).unwrap();
        }
        p
    });
    v.check_pass("Pearson r > 0.99 (nearly identical)", pear > 0.99);

    let (bc, us_bc) = bench_us(|| {
        let mut d = 0.0;
        for _ in 0..1_000 {
            d = barracuda::stats::bray_curtis(
                &a.iter().map(|x| x.abs()).collect::<Vec<_>>(),
                &b.iter().map(|x| x.abs()).collect::<Vec<_>>(),
            );
        }
        d
    });
    v.check_pass("Bray-Curtis ∈ [0, 1]", (0.0..=1.0).contains(&bc));
    timings.push(ProvenanceTiming {
        domain: "correlation",
        primitive: "pearson_correlation (1k)",
        origin_spring: "wetSpring",
        absorbed_at: "S44",
        cpu_us: us_pear,
        check_count: 1,
    });
    timings.push(ProvenanceTiming {
        domain: "diversity",
        primitive: "bray_curtis (1k)",
        origin_spring: "wetSpring",
        absorbed_at: "S44",
        cpu_us: us_bc,
        check_count: 1,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // §10  Cross-Spring Composition: End-to-End Rare Biosphere Pipeline
    // ═══════════════════════════════════════════════════════════════════════
    v.section("§10 Cross-Spring Composition: Rare Biosphere Depth Design");

    let t_pipeline = Instant::now();

    let target_abundances = [0.001, 0.005, 0.01, 0.05];
    let target_power = 0.95;

    for &p in &target_abundances {
        let depth = barracuda::stats::detection_threshold(p, target_power);
        let actual = barracuda::stats::detection_power(p, depth);
        v.check_pass(
            &format!("p={p}: depth design achieves target power"),
            actual >= target_power,
        );
    }

    let depths_for_bs: Vec<f64> = target_abundances
        .iter()
        .map(|&p| barracuda::stats::detection_threshold(p, 0.95) as f64)
        .collect();
    let depth_ci = barracuda::stats::bootstrap_ci(
        &depths_for_bs,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5_000,
        0.95,
        99,
    )
    .unwrap();
    v.check_pass("Depth CI lower > 0", depth_ci.lower > 0.0);

    let depth_jk =
        barracuda::stats::jackknife(&depths_for_bs, |d| d.iter().sum::<f64>() / d.len() as f64)
            .unwrap();
    v.check_pass("Depth JK SE > 0", depth_jk.std_error > 0.0);

    let pipeline_us = t_pipeline.elapsed().as_micros() as f64;
    timings.push(ProvenanceTiming {
        domain: "composition",
        primitive: "rare biosphere pipeline (detect+bs+jk)",
        origin_spring: "ground+neural",
        absorbed_at: "S54/S70",
        cpu_us: pipeline_us,
        check_count: 6,
    });
    println!("  Rare biosphere pipeline: {pipeline_us:.0} µs");
    println!(
        "    depth_ci:  {:.0} [{:.0}, {:.0}]",
        depth_ci.estimate, depth_ci.lower, depth_ci.upper
    );
    println!(
        "    depth_jk:  {:.4} ± {:.4}",
        depth_jk.estimate, depth_jk.std_error
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Provenance Report
    // ═══════════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let total_checks: u32 = timings.iter().map(|t| t.check_count).sum();

    println!();
    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║              Cross-Spring Evolution Provenance Map — S70+++                      ║"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ {:15} │ {:30} │ {:14} │ {:5} │ {:>10} ║",
        "Domain", "Primitive", "Origin Spring", "At", "µs/call"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mut by_spring: std::collections::BTreeMap<&str, Vec<&ProvenanceTiming>> =
        std::collections::BTreeMap::new();
    for t in &timings {
        by_spring.entry(t.origin_spring).or_default().push(t);
    }

    for (spring, entries) in &by_spring {
        for t in entries {
            println!(
                "║ {:15} │ {:30} │ {:14} │ {:5} │ {:>10.1} ║",
                t.domain, t.primitive, spring, t.absorbed_at, t.cpu_us
            );
        }
        println!(
            "╠═══════════════════════════════════════════════════════════════════════════════════╣"
        );
    }

    println!(
        "║ TOTAL: {} primitives, {} checks, {:.1} ms elapsed {:>24} ║",
        timings.len(),
        total_checks,
        total_ms,
        ""
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    );

    println!();
    println!("  Cross-Spring Flow Summary:");
    println!("  ─────────────────────────────────────────────────────────────────");
    let spring_counts: std::collections::BTreeMap<&str, usize> =
        timings
            .iter()
            .fold(std::collections::BTreeMap::new(), |mut m, t| {
                *m.entry(t.origin_spring).or_default() += 1;
                m
            });
    for (spring, count) in &spring_counts {
        println!("    {spring:16} → ToadStool: {count} primitives benchmarked");
    }
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  All springs contribute to the shared ecosystem via ToadStool.");
    println!("  wetSpring consumes 90+ primitives from this pool.");
    println!("  ═════════════════════════════════════════════════════════════════");

    v.finish();
}
