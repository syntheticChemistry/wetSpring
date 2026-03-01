// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp248: BarraCuda CPU v18 — Extended Stats Rewire
//!
//! Wires newly available ToadStool S70+++ CPU stats primitives:
//! - `stats::bootstrap::{bootstrap_ci, rawr_mean}` — confidence intervals
//! - `stats::regression::{fit_exponential, fit_quadratic, fit_logarithmic}` — growth curves
//! - Cross-validates with existing wetSpring bio modules (diversity, pangenome Heaps law)
//!
//! # Provenance
//! - `bootstrap_ci`, `rawr_mean` — ToadStool core stats (S54, neuralSpring absorption)
//! - `fit_exponential`, `fit_quadratic`, `fit_logarithmic` — ToadStool S66 (neuralSpring)
//! - `jackknife` — groundSpring S70 (already consumed V82)
//! - `chao1_classic` — groundSpring S70 (already consumed V82)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | Command | `cargo run --bin validate_barracuda_cpu_v18` |

use std::time::Instant;
use wetspring_barracuda::validation::Validator;

struct Timing {
    name: &'static str,
    cpu_us: f64,
}

fn main() {
    let mut v = Validator::new("Exp248: BarraCuda CPU v18 — Extended Stats Rewire");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══ S1: Bootstrap Confidence Intervals ═════════════════════════════════
    // Provenance: ToadStool core stats (S54, neuralSpring → ToadStool)
    v.section("S1: bootstrap_ci — Confidence Intervals (neuralSpring → ToadStool)");

    let diversity_data = [
        2.15, 2.31, 1.98, 2.42, 2.05, 2.28, 2.11, 2.37, 2.19, 2.25,
        2.33, 2.08, 2.22, 2.40, 2.03, 2.29, 2.14, 2.36, 2.20, 2.27,
    ];

    let t0 = Instant::now();
    let ci = barracuda::stats::bootstrap_ci(
        &diversity_data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        10_000,
        0.95,
        42,
    )
    .expect("bootstrap_ci");
    let bs_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing { name: "bootstrap_ci(10k)", cpu_us: bs_us });

    v.check_pass("CI lower < estimate", ci.lower < ci.estimate);
    v.check_pass("CI upper > estimate", ci.upper > ci.estimate);
    v.check_pass("Confidence = 0.95", (ci.confidence - 0.95).abs() < 1e-12);
    v.check_pass("n_bootstrap = 10000", ci.n_bootstrap == 10_000);
    v.check_pass("SE > 0", ci.std_error > 0.0);

    let true_mean: f64 = diversity_data.iter().sum::<f64>() / diversity_data.len() as f64;
    v.check("Estimate ≈ sample mean", ci.estimate, true_mean, 0.05);
    v.check_pass("CI contains true mean", ci.lower <= true_mean && true_mean <= ci.upper);
    println!("  Bootstrap: {:.4} [{:.4}, {:.4}] SE={:.4}", ci.estimate, ci.lower, ci.upper, ci.std_error);

    // ═══ S2: RAWR Bootstrap Mean ════════════════════════════════════════════
    v.section("S2: rawr_mean — RAWR Bootstrap (neuralSpring → ToadStool)");

    let abundances = [10.0, 25.0, 3.0, 1.0, 42.0, 7.0, 15.0, 2.0, 8.0, 30.0];

    let t1 = Instant::now();
    let rawr = barracuda::stats::rawr_mean(&abundances, 5_000, 0.95, 123)
        .expect("rawr_mean");
    let rawr_us = t1.elapsed().as_micros() as f64;
    timings.push(Timing { name: "rawr_mean(5k)", cpu_us: rawr_us });

    let sample_mean: f64 = abundances.iter().sum::<f64>() / abundances.len() as f64;
    v.check("RAWR estimate ≈ sample mean", rawr.estimate, sample_mean, 2.0);
    v.check_pass("RAWR CI lower < upper", rawr.lower < rawr.upper);
    v.check_pass("RAWR SE > 0", rawr.std_error > 0.0);
    println!("  RAWR: {:.4} [{:.4}, {:.4}] SE={:.4}", rawr.estimate, rawr.lower, rawr.upper, rawr.std_error);

    // ═══ S3: Bootstrap + Jackknife Cross-Validation ═════════════════════════
    v.section("S3: Bootstrap vs Jackknife — SE Comparison (cross-spring validation)");

    let jk = barracuda::stats::jackknife_mean_variance(&diversity_data).unwrap();
    let bs_se = ci.std_error;
    let jk_se = jk.std_error;

    v.check_pass("Both SE > 0", bs_se > 0.0 && jk_se > 0.0);
    v.check_pass(
        "Bootstrap SE ≈ Jackknife SE (within 3×)",
        bs_se / jk_se < 3.0 && jk_se / bs_se < 3.0,
    );
    println!("  Bootstrap SE = {bs_se:.6}, Jackknife SE = {jk_se:.6}, ratio = {:.2}", bs_se / jk_se);

    // ═══ S4: Regression — Exponential Fit ═══════════════════════════════════
    // Provenance: ToadStool S66 (neuralSpring → ToadStool)
    v.section("S4: fit_exponential — Growth Curve (neuralSpring S66 → ToadStool)");

    let x_exp: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y_exp: Vec<f64> = x_exp.iter().map(|&x| 2.0 * (0.15 * x).exp() + 0.5).collect();

    let t2 = Instant::now();
    let fit_exp = barracuda::stats::fit_exponential(&x_exp, &y_exp);
    let exp_us = t2.elapsed().as_micros() as f64;
    timings.push(Timing { name: "fit_exponential", cpu_us: exp_us });

    v.check_pass("fit_exponential: Some", fit_exp.is_some());
    if let Some(ref fe) = fit_exp {
        v.check_pass("Model name = 'exponential'", fe.model == "exponential");
        v.check_pass("R² > 0.95", fe.r_squared > 0.95);
        println!("  Exponential: params={:?}, R²={:.6}, RMSE={:.6}", fe.params, fe.r_squared, fe.rmse);

        let pred = fe.predict_one(10.0);
        v.check_pass("predict_one(10) is Some", pred.is_some());
        if let Some(p) = pred {
            let expected = 2.0 * (0.15_f64 * 10.0).exp() + 0.5;
            v.check("predict_one(10) ≈ truth", p, expected, expected * 0.15);
        }
    }

    // ═══ S5: Regression — Quadratic Fit ═════════════════════════════════════
    v.section("S5: fit_quadratic — Parabolic Growth (neuralSpring S66 → ToadStool)");

    let x_quad: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
    let y_quad: Vec<f64> = x_quad.iter().map(|&x| 0.5 * x * x - 2.0 * x + 3.0).collect();

    let t3 = Instant::now();
    let fit_quad = barracuda::stats::fit_quadratic(&x_quad, &y_quad);
    let quad_us = t3.elapsed().as_micros() as f64;
    timings.push(Timing { name: "fit_quadratic", cpu_us: quad_us });

    v.check_pass("fit_quadratic: Some", fit_quad.is_some());
    if let Some(ref fq) = fit_quad {
        v.check_pass("Model name = 'quadratic'", fq.model == "quadratic");
        v.check_pass("R² > 0.99 (exact data)", fq.r_squared > 0.99);
        println!("  Quadratic: params={:?}, R²={:.6}", fq.params, fq.r_squared);
    }

    // ═══ S6: Regression — Logarithmic Fit ═══════════════════════════════════
    v.section("S6: fit_logarithmic — Log Growth (neuralSpring S66 → ToadStool)");

    let x_log: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let y_log: Vec<f64> = x_log.iter().map(|&x| 5.0 * x.ln() + 2.0).collect();

    let t4 = Instant::now();
    let fit_log = barracuda::stats::fit_logarithmic(&x_log, &y_log);
    let log_us = t4.elapsed().as_micros() as f64;
    timings.push(Timing { name: "fit_logarithmic", cpu_us: log_us });

    v.check_pass("fit_logarithmic: Some", fit_log.is_some());
    if let Some(ref fl) = fit_log {
        v.check_pass("Model name = 'logarithmic'", fl.model == "logarithmic");
        v.check_pass("R² > 0.99 (exact data)", fl.r_squared > 0.99);
        println!("  Logarithmic: params={:?}, R²={:.6}", fl.params, fl.r_squared);
    }

    // ═══ S7: Regression — fit_all Model Selection ═══════════════════════════
    v.section("S7: fit_all — Best Model Selection (ToadStool)");

    let t5 = Instant::now();
    let all_fits = barracuda::stats::fit_all(&x_log, &y_log);
    let all_us = t5.elapsed().as_micros() as f64;
    timings.push(Timing { name: "fit_all", cpu_us: all_us });

    v.check_pass("fit_all: non-empty", !all_fits.is_empty());
    let best = all_fits.iter().max_by(|a, b| a.r_squared.partial_cmp(&b.r_squared).unwrap());
    if let Some(fb) = best {
        v.check_pass(
            "Best model = logarithmic (for log data)",
            fb.model == "logarithmic",
        );
        println!("  Best model: {} (R²={:.6})", fb.model, fb.r_squared);
        for f in &all_fits {
            println!("    {} → R²={:.6}", f.model, f.r_squared);
        }
    }

    // ═══ S8: Bootstrap + Regression Cross-Validation ════════════════════════
    // Use bootstrap CI on regression residuals to prove confidence
    v.section("S8: Bootstrap CI on Regression Residuals (cross-spring composition)");

    if let Some(ref fe) = fit_exp {
        let residuals: Vec<f64> = x_exp
            .iter()
            .zip(y_exp.iter())
            .filter_map(|(&x, &y)| fe.predict_one(x).map(|p| y - p))
            .collect();

        let resid_ci = barracuda::stats::bootstrap_ci(
            &residuals,
            |d| d.iter().sum::<f64>() / d.len() as f64,
            5_000,
            0.95,
            99,
        )
        .expect("residual bootstrap");

        v.check_pass(
            "Residual CI contains 0 (unbiased fit)",
            resid_ci.lower <= 0.0 && 0.0 <= resid_ci.upper,
        );
        println!(
            "  Residual mean = {:.6} [{:.6}, {:.6}]",
            resid_ci.estimate, resid_ci.lower, resid_ci.upper
        );
    }

    // ═══ S9: Pangenome Heaps Law — fit_linear vs fit_exponential ════════════
    v.section("S9: Heaps Law Composition (fit_linear already used, fit_exp new)");

    let genomes: Vec<f64> = (1..=50).map(|i| i as f64).collect();
    let new_genes: Vec<f64> = genomes.iter().map(|&g| 500.0 * g.powf(0.6)).collect();

    let heaps_linear = barracuda::stats::fit_linear(
        &genomes.iter().map(|g| g.ln()).collect::<Vec<_>>(),
        &new_genes.iter().map(|n| n.ln()).collect::<Vec<_>>(),
    );
    v.check_pass("Heaps linear fit: Some", heaps_linear.is_some());
    if let Some(ref hl) = heaps_linear {
        v.check("Heaps exponent ≈ 0.6", hl.params[0], 0.6, 0.05);
        println!("  Heaps linear (log-log): exponent={:.4}, R²={:.4}", hl.params[0], hl.r_squared);
    }

    // ═══ S10: Detection Power + Bootstrap — Sampling Design ═════════════════
    v.section("S10: Sampling Design — Detection Power + Bootstrap (cross-spring)");

    let rare_abundances = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2];
    for &p in &rare_abundances {
        let depth = barracuda::stats::detection_threshold(p, 0.95);
        let actual = barracuda::stats::detection_power(p, depth);
        v.check_pass(&format!("p={p}: depth achieves ≥ 0.95 power"), actual >= 0.95);
    }

    let obs_data: Vec<f64> = rare_abundances.iter()
        .map(|&p| barracuda::stats::detection_power(p, 5000) * 100.0)
        .collect();
    let obs_ci = barracuda::stats::bootstrap_ci(
        &obs_data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5_000, 0.95, 77,
    ).expect("power bootstrap");
    v.check_pass("Power CI: lower > 0", obs_ci.lower > 0.0);
    v.check_pass("Power CI: meaningful range", obs_ci.upper >= obs_ci.lower);
    println!(
        "  Mean detection power at D=5000: {:.2}% [{:.2}%, {:.2}%]",
        obs_ci.estimate, obs_ci.lower, obs_ci.upper
    );

    // ═══ Timing Summary ═════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  CPU Timing Summary");
    println!("  ─────────────────────────────────────────────────────────────");
    for t in &timings {
        println!("  {:<24} {:>10.0} µs", t.name, t.cpu_us);
    }
    println!("═══════════════════════════════════════════════════════════════");

    // ═══ Provenance Map ═════════════════════════════════════════════════════
    println!();
    println!("  Cross-Spring Provenance (consumed in this experiment):");
    println!("  ───────────────────────────────────────────────────────────");
    println!("  neuralSpring → ToadStool: bootstrap_ci, rawr_mean (S54)");
    println!("  neuralSpring → ToadStool: fit_exponential, fit_quadratic, fit_logarithmic (S66)");
    println!("  groundSpring → ToadStool: jackknife_mean_variance (S70)");
    println!("  groundSpring → ToadStool: detection_power, detection_threshold (S70)");
    println!("  wetSpring    → ToadStool: fit_linear (S64, Heaps law composition)");
    println!("  ═══════════════════════════════════════════════════════════");

    v.finish();
}
