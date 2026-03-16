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
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
//! # Exp271: Cross-Spring S79 Evolution Validation + Benchmark
//!
//! Validates all cross-spring evolved `ToadStool` primitives consumed by wetSpring,
//! with provenance annotations tracking which spring contributed each primitive
//! and benchmarking each domain.
//!
//! # Cross-Spring Provenance Map
//!
//! | Domain | Spring Origin | `ToadStool` Module | Session |
//! |--------|--------------|-------------------|---------|
//! | Diversity (Shannon, Simpson, Chao1) | wetSpring | `stats::diversity` | S64 |
//! | Bray-Curtis beta diversity | wetSpring | `stats::diversity` | S64 |
//! | Rarefaction curves | wetSpring | `stats::diversity` | S64 |
//! | Spectral analysis (Anderson, Lanczos) | hotSpring | `spectral` | v0.6.0 |
//! | Spectral phase classification | neuralSpring | `spectral::stats` | V69→S79 |
//! | Population genetics (Kimura, error threshold) | groundSpring | `stats::evolution` | S70 |
//! | Jackknife statistics | groundSpring | `stats::jackknife` | S70 |
//! | Bootstrap CI | multiple springs | `stats::bootstrap` | S64+ |
//! | Hydrology (`Hargreaves` ET₀) | airSpring/groundSpring | `stats::hydrology` | S70 |
//! | Regression (linear, quadratic, exponential) | airSpring | `stats::regression` | S66 |
//! | Ridge regression | wetSpring→`ToadStool` | `linalg::ridge_regression` | S59 |
//! | NMF matrix factorization | wetSpring→`ToadStool` | `linalg::nmf` | S64 |
//! | Special functions (erf, `ln_gamma`) | multiple springs | `special` | S64 |
//! | Boltzmann sampling | wateringHole | `sample::boltzmann_sampling` | V69 |
//! | Moving window stats | airSpring/wetSpring | `stats::moving_window_f64` | S66 |
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | Provenance type | Analytical (mathematical invariants + cross-spring parity) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_cross_spring_s79` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct DomainResult {
    name: &'static str,
    spring: &'static str,
    session: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp271: Cross-Spring S79 Evolution");
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══ S1: Alpha Diversity (wetSpring → ToadStool S64) ════════════════════
    {
        let t = Instant::now();
        v.section("S1: Alpha Diversity [wetSpring → S64]");

        let counts = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
        let h = barracuda::stats::shannon(&counts);
        let s = barracuda::stats::simpson(&counts);
        let obs = barracuda::stats::observed_features(&counts);
        let c = barracuda::stats::chao1(&counts);
        let j = barracuda::stats::pielou_evenness(&counts);
        let ad = barracuda::stats::alpha_diversity(&counts);

        v.check_pass("shannon > 0", h > 0.0);
        v.check_pass("simpson in [0,1]", (0.0..=1.0).contains(&s));
        v.check_pass("observed = 8", (obs - 8.0).abs() < f64::EPSILON);
        v.check_pass("chao1 ≥ observed", c >= obs);
        v.check_pass("pielou in [0,1]", (0.0..=1.0).contains(&j));
        v.check_pass(
            "alpha_diversity consistent",
            (ad.shannon - h).abs() < tolerances::ANALYTICAL_F64,
        );

        let uniform = vec![25.0; 4];
        let h_uniform = barracuda::stats::shannon(&uniform);
        v.check_pass(
            "shannon(uniform) = ln(4)",
            (h_uniform - 4.0_f64.ln()).abs() < tolerances::PYTHON_PARITY,
        );

        let freq = vec![0.25; 4];
        let h_freq = barracuda::stats::shannon_from_frequencies(&freq);
        v.check_pass(
            "shannon_from_freq parity",
            (h_freq - h_uniform).abs() < tolerances::PYTHON_PARITY,
        );

        let counts_u64 = [50_u64, 30, 20, 10, 5, 3, 2, 1];
        let c_classic = barracuda::stats::chao1_classic(&counts_u64);
        v.check_pass("chao1_classic ≥ S_obs", c_classic >= 8.0);

        domains.push(DomainResult {
            name: "Alpha Diversity",
            spring: "wetSpring",
            session: "S64",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 9,
        });
    }

    // ═══ S2: Beta Diversity + Rarefaction (wetSpring → ToadStool S64) ══════
    {
        let t = Instant::now();
        v.section("S2: Beta Diversity + Rarefaction [wetSpring → S64]");

        let a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
        let b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
        let bc = barracuda::stats::bray_curtis(&a, &b);
        v.check_pass("bray_curtis in [0,1]", (0.0..=1.0).contains(&bc));
        v.check_pass(
            "bray_curtis(x,x) = 0",
            barracuda::stats::bray_curtis(&a, &a).abs() < tolerances::EXACT_F64,
        );
        v.check_pass(
            "bray_curtis symmetric",
            (bc - barracuda::stats::bray_curtis(&b, &a)).abs() < tolerances::EXACT_F64,
        );

        let samples = vec![a, b, vec![5.0, 5.0, 5.0, 5.0, 5.0]];
        let condensed = barracuda::stats::bray_curtis_condensed(&samples);
        v.check_pass("condensed len = 3", condensed.len() == 3);

        let matrix = barracuda::stats::bray_curtis_matrix(&samples);
        v.check_pass(
            "matrix diagonal = 0",
            matrix[0].abs() < tolerances::EXACT_F64 && matrix[8].abs() < tolerances::EXACT_F64,
        );

        let counts = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
        let total: f64 = counts.iter().sum();
        let depths: Vec<f64> = (1..=total as u64).map(|d| d as f64).collect();
        let curve = barracuda::stats::rarefaction_curve(&counts, &depths);
        v.check_pass(
            "rarefaction monotonic",
            curve
                .windows(2)
                .all(|w| w[1] >= w[0] - tolerances::RAREFACTION_MONOTONIC),
        );
        v.check_pass(
            "rarefaction(full) = S_obs",
            (curve.last().unwrap() - 8.0).abs() < tolerances::PHYLO_LIKELIHOOD,
        );

        domains.push(DomainResult {
            name: "Beta + Rarefaction",
            spring: "wetSpring",
            session: "S64",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 7,
        });
    }

    // ═══ S3: Spectral Analysis (hotSpring + neuralSpring → ToadStool) ══════
    {
        let t = Instant::now();
        v.section("S3: Spectral Analysis [hotSpring v0.6.0 + neuralSpring S79]");

        let mat = barracuda::spectral::anderson_3d(6, 6, 6, 4.0, 42);
        let n = 6 * 6 * 6;
        let tri = barracuda::spectral::lanczos(&mat, n, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);

        v.check_pass("eigenvalues non-empty", !eigs.is_empty());
        v.check_pass("eigenvalues count ≤ n", eigs.len() <= n);

        let r = barracuda::spectral::level_spacing_ratio(&eigs);
        v.check_pass("LSR in [0, 1]", (0.0..=1.0).contains(&r));

        let bw = barracuda::spectral::spectral_bandwidth(&eigs);
        v.check_pass("bandwidth > 0", bw > 0.0);

        let cond = barracuda::spectral::spectral_condition_number(&eigs);
        v.check_pass("condition_number > 0", cond > 0.0);

        let analysis = barracuda::spectral::SpectralAnalysis::from_eigenvalues(eigs.clone(), 1.0);
        v.check_pass(
            "SpectralAnalysis bandwidth",
            (analysis.bandwidth - bw).abs() < tolerances::ANALYTICAL_F64,
        );
        v.check_pass(
            "SpectralAnalysis cond",
            (analysis.condition_number - cond).abs() < tolerances::ANALYTICAL_F64,
        );

        let bands = barracuda::spectral::detect_bands(&eigs, 5.0);
        v.check_pass("detect_bands non-empty", !bands.is_empty());

        let phase = format!("{:?}", analysis.phase);
        v.check_pass(
            "phase valid",
            ["Bulk", "EdgeOfChaos", "Chaotic"].contains(&phase.as_str()),
        );

        println!("  Anderson 3D (L=6, W=4): LSR={r:.4}, bw={bw:.2}, phase={phase}");
        println!("  hotSpring v0.6.0 (Kachkovskiy) → spectral, neuralSpring V69 → phase");

        domains.push(DomainResult {
            name: "Spectral Analysis",
            spring: "hotSpring+neuralSpring",
            session: "v0.6.0+S79",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 9,
        });
    }

    // ═══ S4: Population Genetics (groundSpring → ToadStool S70) ════════════
    {
        let t = Instant::now();
        v.section("S4: Population Genetics [groundSpring → S70]");

        let fix_neutral = barracuda::stats::kimura_fixation_prob(100, 0.0, 0.5);
        v.check_pass(
            "neutral fixation ≈ 0.5",
            (fix_neutral - 0.5).abs() < tolerances::CF2_SPACING_TOL,
        );

        let fix_pos = barracuda::stats::kimura_fixation_prob(100, 0.01, 0.01);
        let fix_neg = barracuda::stats::kimura_fixation_prob(100, -0.01, 0.01);
        v.check_pass("positive > negative", fix_pos > fix_neg);

        let threshold = barracuda::stats::error_threshold(10.0, 100);
        v.check_pass("error_threshold Some", threshold.is_some());
        if let Some(mu_c) = threshold {
            v.check_pass("mu_c in (0, 1)", mu_c > 0.0 && mu_c < 1.0);
            println!("  Error threshold (σ=10, L=100): μ_c = {mu_c:.6}");
        }

        let power = barracuda::stats::detection_power(0.01, 1000);
        v.check_pass("detection_power in [0,1]", (0.0..=1.0).contains(&power));

        let depth = barracuda::stats::detection_threshold(0.01, 0.95);
        v.check_pass("detection_threshold > 0", depth > 0);
        println!("  groundSpring drift.rs/quasispecies.rs → ToadStool S70");

        domains.push(DomainResult {
            name: "Population Genetics",
            spring: "groundSpring",
            session: "S70",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 6,
        });
    }

    // ═══ S5: Jackknife Statistics (groundSpring → ToadStool S70) ═══════════
    {
        let t = Instant::now();
        v.section("S5: Jackknife [groundSpring → S70]");

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let jk = barracuda::stats::jackknife_mean_variance(&data);
        v.check_pass("jackknife Some", jk.is_some());
        if let Some(ref result) = jk {
            v.check_pass(
                "mean ≈ 5.5",
                (result.estimate - 5.5).abs() < tolerances::PYTHON_PARITY,
            );
            v.check_pass("variance > 0", result.variance > 0.0);
        }

        let jk_fn = barracuda::stats::jackknife(&data, |s| s.iter().sum::<f64>() / s.len() as f64);
        v.check_pass("jackknife(custom) Some", jk_fn.is_some());
        println!("  groundSpring jackknife.rs → ToadStool S70");

        domains.push(DomainResult {
            name: "Jackknife",
            spring: "groundSpring",
            session: "S70",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 4,
        });
    }

    // ═══ S6: Regression (airSpring → ToadStool S66) ═══════════════════════
    {
        let t = Instant::now();
        v.section("S6: Regression [airSpring → S66]");

        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0_f64.mul_add(xi, 3.0)).collect();

        let linear = barracuda::stats::fit_linear(&x, &y);
        v.check_pass("fit_linear Some", linear.is_some());
        if let Some(ref fit) = linear {
            v.check_pass(
                "r² ≈ 1.0",
                (fit.r_squared - 1.0).abs() < tolerances::GPU_VS_CPU_F64,
            );
            v.check_pass(
                "slope ≈ 2.0",
                (fit.params[0] - 2.0).abs() < tolerances::GPU_VS_CPU_F64,
            );
        }

        let y_quad: Vec<f64> = x.iter().map(|&xi| xi.mul_add(xi, xi + 1.0)).collect();
        let quad = barracuda::stats::fit_quadratic(&x, &y_quad);
        v.check_pass("fit_quadratic Some", quad.is_some());

        let all = barracuda::stats::fit_all(&x, &y);
        v.check_pass("fit_all non-empty", !all.is_empty());
        println!("  airSpring precision agriculture → ToadStool S66");

        domains.push(DomainResult {
            name: "Regression",
            spring: "airSpring",
            session: "S66",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 5,
        });
    }

    // ═══ S7: Hydrology (airSpring/groundSpring → ToadStool S70) ═══════════
    {
        let t = Instant::now();
        v.section("S7: Hydrology [airSpring/groundSpring → S70]");

        let et0 = barracuda::stats::hargreaves_et0(15.0, 30.0, 15.0);
        v.check_pass("hargreaves_et0 Some", et0.is_some());
        v.check_pass("et0 > 0", et0.unwrap_or(0.0) > 0.0);

        let batch =
            barracuda::stats::hargreaves_et0_batch(&[15.0, 14.0], &[30.0, 28.0], &[15.0, 12.0]);
        v.check_pass("batch Some", batch.is_some());
        v.check_pass(
            "batch len = 2",
            batch.as_ref().is_some_and(|b| b.len() == 2),
        );

        let kc = barracuda::stats::crop_coefficient(0.3, 1.15, 15, 30);
        v.check_pass(
            "crop_coefficient in [0.3, 1.15]",
            (0.3..=1.15).contains(&kc),
        );

        let theta = barracuda::stats::soil_water_balance(0.25, 10.0, 0.0, 5.0, 0.35);
        v.check_pass("soil_water_balance > 0", theta > 0.0);

        println!("  ET₀ = {:.2} mm/day", et0.unwrap_or(0.0));
        println!("  airSpring V035 + groundSpring V10 → ToadStool S70");

        domains.push(DomainResult {
            name: "Hydrology (FAO-56)",
            spring: "airSpring+groundSpring",
            session: "S70",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 6,
        });
    }

    // ═══ S8: Special Functions (multi-spring → S64) ════════════════════════
    {
        let t = Instant::now();
        v.section("S8: Special Functions [multi-spring → S64]");

        let e = barracuda::special::erf(1.0);
        v.check_pass(
            "erf(1) ≈ 0.8427",
            (e - 0.842_700_792_949_714_9).abs() < tolerances::ERF_PARITY,
        );
        v.check_pass(
            "erf(-x) = -erf(x)",
            (e + barracuda::special::erf(-1.0)).abs() < tolerances::PYTHON_PARITY,
        );

        let lg = barracuda::special::ln_gamma(5.0);
        v.check_pass("ln_gamma(5) Ok", lg.is_ok());
        if let Ok(val) = lg {
            v.check_pass(
                "ln_gamma(5) ≈ ln(24)",
                (val - 24.0_f64.ln()).abs() < tolerances::GPU_VS_CPU_F64,
            );
        }

        let gp = barracuda::special::regularized_gamma_p(1.0, 1.0);
        v.check_pass("gamma_p Ok", gp.is_ok());

        let cdf = barracuda::stats::norm_cdf(0.0);
        v.check_pass(
            "norm_cdf(0) = 0.5",
            (cdf - 0.5).abs() < tolerances::PYTHON_PARITY,
        );
        println!("  A&S 7.1.26 → ToadStool S64 (multi-spring absorption)");

        domains.push(DomainResult {
            name: "Special Functions",
            spring: "multi-spring",
            session: "S64",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 6,
        });
    }

    // ═══ S9: Bootstrap CI (multi-spring → S64+) ═══════════════════════════
    {
        let t = Instant::now();
        v.section("S9: Bootstrap CI [multi-spring → S64+]");

        let data: Vec<f64> = (1..=100).map(f64::from).collect();
        let ci = barracuda::stats::bootstrap_mean(&data, 1000, 0.95, 42);
        v.check_pass("bootstrap_mean Ok", ci.is_ok());
        if let Ok(ref result) = ci {
            v.check_pass("CI lower < upper", result.lower < result.upper);
            v.check_pass(
                "CI contains 50.5",
                result.lower < 50.5 && result.upper > 50.5,
            );
            println!(
                "  Bootstrap CI (95%): [{:.2}, {:.2}]",
                result.lower, result.upper
            );
        }

        let rawr = barracuda::stats::rawr_mean(&data, 500, 0.95, 42);
        v.check_pass("rawr_mean Ok", rawr.is_ok());
        println!("  Efron & Tibshirani (1993) → ToadStool S64+");

        domains.push(DomainResult {
            name: "Bootstrap CI",
            spring: "multi-spring",
            session: "S64+",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 4,
        });
    }

    // ═══ S10: Linear Algebra (wetSpring → ToadStool S59+) ═════════════════
    {
        let t = Instant::now();
        v.section("S10: Linear Algebra [wetSpring → S59+]");

        let adj = vec![
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let lap = barracuda::linalg::graph_laplacian(&adj, 4);
        let row_sum: f64 = lap[0..4].iter().sum();
        v.check_pass(
            "laplacian row sum = 0",
            row_sum.abs() < tolerances::ANALYTICAL_F64,
        );

        let ridge = barracuda::linalg::ridge_regression(
            &(0..20).map(f64::from).collect::<Vec<_>>(),
            &(0..20)
                .map(|i| 3.0_f64.mul_add(f64::from(i), 1.0))
                .collect::<Vec<_>>(),
            20,
            1,
            1,
            0.001,
        );
        v.check_pass("ridge_regression Ok", ridge.is_ok());

        use barracuda::linalg::nmf::{NmfConfig, NmfObjective};
        let v_mat: Vec<f64> = (0..12)
            .map(|i| 1.0_f64.mul_add(f64::from(i), 1.0).abs())
            .collect();
        let config = NmfConfig {
            rank: 2,
            max_iter: 100,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: NmfObjective::Euclidean,
            seed: 42,
        };
        let nmf = barracuda::linalg::nmf::nmf(&v_mat, 3, 4, &config);
        v.check_pass("nmf Ok", nmf.is_ok());
        if let Ok(ref result) = nmf {
            v.check_pass("NMF W non-neg", result.w.iter().all(|&x| x >= 0.0));
            v.check_pass("NMF H non-neg", result.h.iter().all(|&x| x >= 0.0));
        }
        println!("  wetSpring → ToadStool S59 (ridge), S64 (NMF, Laplacian)");

        domains.push(DomainResult {
            name: "Linear Algebra",
            spring: "wetSpring",
            session: "S59+",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 5,
        });
    }

    // ═══ S11: Moving Window Stats (airSpring/wetSpring → S66) ══════════════
    {
        let t = Instant::now();
        v.section("S11: Moving Window Stats [airSpring/wetSpring → S66]");

        let data: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let mw = barracuda::stats::moving_window_stats_f64(&data, 10);
        v.check_pass("window Some", mw.is_some());
        if let Some(ref result) = mw {
            v.check_pass("mean len = 91", result.mean.len() == 91);
            v.check_pass("mean[0] finite", result.mean[0].is_finite());
            v.check_pass("variance ≥ 0", result.variance.iter().all(|&v| v >= 0.0));
        }
        println!("  airSpring + wetSpring monitoring → ToadStool S66");

        domains.push(DomainResult {
            name: "Moving Window",
            spring: "airSpring+wetSpring",
            session: "S66",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 4,
        });
    }

    // ═══ S12: Boltzmann Sampling (wateringHole → ToadStool V69) ═══════════
    {
        let t = Instant::now();
        v.section("S12: Boltzmann Sampling [wateringHole → V69]");

        let loss_fn = |params: &[f64]| -> f64 { params.iter().map(|x| x * x).sum() };
        let initial = vec![5.0, -3.0, 2.0];
        let initial_loss = loss_fn(&initial);
        let result = barracuda::sample::boltzmann_sampling(&loss_fn, &initial, 1.0, 0.5, 500, 42);
        v.check_pass("losses non-empty", !result.losses.is_empty());
        v.check_pass(
            "accept in [0,1]",
            (0.0..=1.0).contains(&result.acceptance_rate),
        );
        let final_loss = result.losses.last().copied().unwrap_or(f64::MAX);
        v.check_pass("final ≤ initial", final_loss <= initial_loss);
        println!(
            "  {} steps, accept={:.0}%",
            result.losses.len(),
            result.acceptance_rate * 100.0
        );
        println!("  wateringHole V69 → ToadStool (Metropolis-Hastings)");

        domains.push(DomainResult {
            name: "Boltzmann Sampling",
            spring: "wateringHole",
            session: "V69",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 3,
        });
    }

    // ═══ S13: Correlation & Stats (multi-spring) ═══════════════════════════
    {
        let t = Instant::now();
        v.section("S13: Correlation [multi-spring]");

        let x: Vec<f64> = (0..50).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0_f64.mul_add(xi, 1.0)).collect();

        let pearson = barracuda::stats::pearson_correlation(&x, &y);
        v.check_pass("pearson Ok", pearson.is_ok());
        if let Ok(r) = pearson {
            v.check_pass("pearson ≈ 1.0", (r - 1.0).abs() < tolerances::PYTHON_PARITY);
        }

        let cov = barracuda::stats::covariance(&x, &y);
        v.check_pass("covariance Ok", cov.is_ok());
        v.check_pass("covariance > 0", cov.unwrap_or(0.0) > 0.0);

        let mean_x = barracuda::stats::mean(&x);
        v.check_pass(
            "mean ≈ 24.5",
            (mean_x - 24.5).abs() < tolerances::PYTHON_PARITY,
        );
        println!("  multi-spring → ToadStool stats::correlation");

        domains.push(DomainResult {
            name: "Correlation",
            spring: "multi-spring",
            session: "S64",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 5,
        });
    }

    // ═══ Summary ═══════════════════════════════════════════════════════════
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Provenance Benchmark — ToadStool S79               ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:22} │ {:18} │ {:>8} │ {:>7} │ {:>3} ║",
        "Domain", "Spring", "Session", "Time", "✓"
    );
    println!("╠════════════════════════════════════════════════════════════════════╣");

    let mut total_checks = 0_u32;
    let mut total_ms = 0.0_f64;
    for d in &domains {
        println!(
            "║ {:22} │ {:18} │ {:>8} │ {:6.2}ms │ {:>3} ║",
            d.name, d.spring, d.session, d.ms, d.checks
        );
        total_checks += d.checks;
        total_ms += d.ms;
    }

    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:22} │ {:18} │ {:>8} │ {:6.2}ms │ {:>3} ║",
        "TOTAL", "6 springs", "", total_ms, total_checks
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("  Cross-Spring Evolution Tree:");
    println!("  ┌─ hotSpring v0.6.0 ── spectral (Anderson, Lanczos, level statistics)");
    println!("  ├─ neuralSpring V69 ── spectral phase (Bulk/EdgeOfChaos/Chaotic)");
    println!("  ├─ wetSpring S64 ───── diversity, Bray-Curtis, rarefaction, NMF, ridge");
    println!("  ├─ groundSpring S70 ── Kimura fixation, error threshold, jackknife");
    println!("  ├─ airSpring S66 ───── regression, moving window, hydrology");
    println!("  ├─ wateringHole V69 ── Boltzmann sampling (Metropolis-Hastings MCMC)");
    println!("  └─ multi-spring ────── special functions, bootstrap, correlation");
    println!();
    println!("  All primitives: ToadStool BarraCUDA S79 → consumed by wetSpring.");
    println!("  844 WGSL shaders, all f64-canonical. Zero local shaders.");

    v.finish();
}
