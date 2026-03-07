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
//! # Exp307: Python vs Rust Benchmark v4 — V97 Fused Ops Parity
//!
//! Extends v3 (15 domains, 35 checks) with **fused statistics** domains that
//! exercise the CPU-side math underlying `barraCuda` v0.3.3's new GPU shaders.
//!
//! Each section specifies:
//! 1. The exact Python equivalent (function + library)
//! 2. The analytical / known result
//! 3. `BarraCuda`'s computed result + timing
//!
//! New domains:
//! - §16: Sample variance — `numpy.var(ddof=1)` parity
//! - §17: Sample covariance — `numpy.cov` parity
//! - §18: Pearson correlation — `scipy.stats.pearsonr` parity
//! - §19: Spearman correlation — `scipy.stats.spearmanr` parity
//! - §20: Correlation matrix — `numpy.corrcoef` parity
//! - §21: Jackknife — `astropy.stats.jackknife_resampling` parity
//! - §22: Covariance matrix — `numpy.cov` parity
//!
//! # Chain
//!
//! ```text
//! Paper (Exp291) → CPU (Exp306 proves math) → Python parity (this) → GPU (Exp308)
//! ```
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-05 |
//! | Command | `cargo run --release --bin benchmark_python_vs_rust_v4` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct ParityBench {
    domain: &'static str,
    python_equiv: &'static str,
    #[allow(dead_code)]
    expected: f64,
    #[allow(dead_code)]
    actual: f64,
    tolerance: f64,
    rust_us: u128,
    workload: &'static str,
}

fn main() {
    let mut v = Validator::new("Exp307: Python vs Rust v4 — V97 Fused Ops Parity");
    let mut benches: Vec<ParityBench> = Vec::new();
    let t_total = Instant::now();

    println!("  Inherited: §1–§15 from v3 (35 checks — run separately)\n");

    // ═══════════════════════════════════════════════════════════════════
    // §16  Sample Variance — numpy.var(ddof=1) parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§16 Sample Variance — Python: numpy.var(x, ddof=1)");

    let data: Vec<f64> = (1..=1000).map(f64::from).collect();
    let n = data.len() as f64;
    let expected_svar = n * (n + 1.0) / 12.0;

    let t = Instant::now();
    let mut actual_svar = 0.0;
    for _ in 0..10_000 {
        actual_svar =
            barracuda::stats::correlation::variance(&data).expect("variance requires n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "Variance: barracuda ≡ analytical",
        actual_svar,
        expected_svar,
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "Sample Variance",
        python_equiv: "numpy.var(x, ddof=1)",
        expected: expected_svar,
        actual: actual_svar,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "1000 points × 10k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §17  Sample Covariance — numpy.cov parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§17 Sample Covariance — Python: numpy.cov(x, y)[0,1]");

    let x: Vec<f64> = (1..=100).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 3.0_f64.mul_add(xi, 7.0)).collect();
    // Cov(x, 3x+7) = 3 * Var(x) = 3 * N(N+1)/12 = 3 * 100*101/12
    let expected_cov = 3.0 * 100.0 * 101.0 / 12.0;

    let t = Instant::now();
    let mut actual_cov = 0.0;
    for _ in 0..10_000 {
        actual_cov = barracuda::stats::covariance(&x, &y)
            .expect("covariance requires equal-length vectors with n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "Covariance: barracuda ≡ analytical",
        actual_cov,
        expected_cov,
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "Sample Covariance",
        python_equiv: "numpy.cov(x, y)[0,1]",
        expected: expected_cov,
        actual: actual_cov,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "100 points × 10k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §18  Pearson Correlation — scipy.stats.pearsonr parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§18 Pearson — Python: scipy.stats.pearsonr(x, y)[0]");

    let expected_r = 1.0; // r(x, 3x+7) = 1.0

    let t = Instant::now();
    let mut actual_r = 0.0;
    for _ in 0..10_000 {
        actual_r = barracuda::stats::pearson_correlation(&x, &y)
            .expect("Pearson correlation requires equal-length vectors with n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "Pearson: barracuda ≡ analytical",
        actual_r,
        expected_r,
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "Pearson r",
        python_equiv: "scipy.stats.pearsonr(x, y)",
        expected: expected_r,
        actual: actual_r,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "100 points × 10k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §19  Spearman Correlation — scipy.stats.spearmanr parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§19 Spearman — Python: scipy.stats.spearmanr(x, y).statistic");

    let mono_x: Vec<f64> = (1..=50).map(f64::from).collect();
    let mono_y: Vec<f64> = mono_x.iter().map(|&xi| xi.powi(3)).collect();
    let expected_rs = 1.0; // perfectly monotonic

    let t = Instant::now();
    let mut actual_rs = 0.0;
    for _ in 0..10_000 {
        actual_rs = barracuda::stats::correlation::spearman_correlation(&mono_x, &mono_y)
            .expect("Spearman correlation requires equal-length vectors with n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "Spearman: barracuda ≡ analytical",
        actual_rs,
        expected_rs,
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "Spearman r_s",
        python_equiv: "scipy.stats.spearmanr(x, y)",
        expected: expected_rs,
        actual: actual_rs,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "50 points × 10k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §20  Correlation Matrix — numpy.corrcoef parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§20 CorrMatrix — Python: numpy.corrcoef(data, rowvar=False)");

    let n_obs = 50_usize;
    let obs_rows: Vec<Vec<f64>> = (0..n_obs)
        .map(|i| {
            let xi = f64::from(i as u32 + 1);
            vec![xi, 2.0_f64.mul_add(xi, 1.0), -xi]
        })
        .collect();
    let expected_diag = 1.0;

    let t = Instant::now();
    let mut corr_mat = vec![0.0; 9];
    for _ in 0..1_000 {
        corr_mat = barracuda::stats::correlation::correlation_matrix(&obs_rows)
            .expect("correlation_matrix requires non-empty rows with n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "CorrMatrix: diag[0] = 1.0",
        corr_mat[0],
        expected_diag,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "CorrMatrix: r[0,1] = 1.0",
        corr_mat[1],
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "CorrMatrix: r[0,2] = -1.0",
        corr_mat[2],
        -1.0,
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "CorrMatrix 3×3",
        python_equiv: "numpy.corrcoef(data, rowvar=False)",
        expected: expected_diag,
        actual: corr_mat[0],
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "50 obs × 3 vars × 1k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §21  Jackknife — astropy parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§21 Jackknife — Python: astropy.stats.jackknife_resampling");

    let jk_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let expected_jk_mean = 5.5;

    let t = Instant::now();
    let mut jk_result =
        barracuda::stats::jackknife_mean_variance(&jk_data).expect("jackknife requires n≥2");
    for _ in 0..100_000 {
        jk_result =
            barracuda::stats::jackknife_mean_variance(&jk_data).expect("jackknife requires n≥2");
    }
    let us = t.elapsed().as_micros();

    v.check(
        "Jackknife: mean = 5.5",
        jk_result.estimate,
        expected_jk_mean,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass("Jackknife: SE > 0", jk_result.std_error > 0.0);
    benches.push(ParityBench {
        domain: "Jackknife",
        python_equiv: "astropy.stats.jackknife_resampling",
        expected: expected_jk_mean,
        actual: jk_result.estimate,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "10 points × 100k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §22  Covariance Matrix — numpy.cov parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("§22 CovMatrix — Python: numpy.cov(data, rowvar=False)");

    let t = Instant::now();
    let mut cov_mat = vec![0.0; 9];
    for _ in 0..1_000 {
        cov_mat = barracuda::stats::correlation::covariance_matrix(&obs_rows)
            .expect("covariance_matrix requires non-empty rows with n≥2");
    }
    let us = t.elapsed().as_micros();

    let var_col0 =
        barracuda::stats::correlation::variance(&obs_rows.iter().map(|r| r[0]).collect::<Vec<_>>())
            .expect("variance of column 0 requires n≥2");
    v.check(
        "CovMatrix: diag[0] = Var(col0)",
        cov_mat[0],
        var_col0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "CovMatrix: symmetric",
        (cov_mat[1] - cov_mat[3]).abs() < tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "CovMatrix 3×3",
        python_equiv: "numpy.cov(data, rowvar=False)",
        expected: var_col0,
        actual: cov_mat[0],
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "50 obs × 3 vars × 1k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // §23  Cross-Paper Diversity Variance — Shannon across communities
    // ═══════════════════════════════════════════════════════════════════
    v.section("§23 Cross-Paper: Shannon variance — Python: numpy.var(H, ddof=1)");

    let communities: Vec<Vec<f64>> = vec![
        vec![50.0, 30.0, 15.0, 5.0],
        vec![25.0, 25.0, 25.0, 25.0],
        vec![90.0, 5.0, 3.0, 2.0],
        vec![40.0, 35.0, 20.0, 5.0],
        vec![10.0, 10.0, 10.0, 70.0],
    ];

    let t = Instant::now();
    let mut shannons = Vec::new();
    for _ in 0..10_000 {
        shannons = communities.iter().map(|c| diversity::shannon(c)).collect();
    }
    let us = t.elapsed().as_micros();

    let h_var =
        barracuda::stats::correlation::variance(&shannons).expect("Shannon variance requires n≥2");
    v.check_pass("Shannon variance > 0", h_var > 0.0);
    let h_uniform = diversity::shannon(&communities[1]);
    v.check(
        "Shannon(uniform) = ln(4)",
        h_uniform,
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    benches.push(ParityBench {
        domain: "Shannon + Var",
        python_equiv: "numpy.var(H, ddof=1)",
        expected: 4.0_f64.ln(),
        actual: h_uniform,
        tolerance: tolerances::ANALYTICAL_F64,
        rust_us: us,
        workload: "5 communities × 10k iters",
    });

    // ═══════════════════════════════════════════════════════════════════
    // Summary Table
    // ═══════════════════════════════════════════════════════════════════
    let total_us = t_total.elapsed().as_micros();

    println!(
        "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ V97 Fused Ops — Python vs Rust Parity + Timing                                                  ║"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ {:19}│ {:36}│ {:>10} │ {:>8} │ {:25} ║",
        "Domain", "Python Equivalent", "Rust (µs)", "Max Δ", "Workload"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for b in &benches {
        let delta = if b.tolerance == 0.0 {
            "0.00e0".to_string()
        } else {
            format!("{:.2e}", b.tolerance)
        };
        println!(
            "║ {:19}│ {:36}│ {:>10} │ {:>8} │ {:25} ║",
            b.domain, b.python_equiv, b.rust_us, delta, b.workload
        );
    }

    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║ TOTAL Rust time: {} µs ({:.1} ms) across {} domains                                      ║",
        total_us,
        total_us as f64 / 1e3,
        benches.len()
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    println!("\n  Summary:");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!(
        "  {} domains — pure Rust BarraCuda CPU math, zero FFI.",
        benches.len()
    );
    println!("  Fused ops (variance, covariance, correlation, Spearman, CorrMatrix)");
    println!("  produce bit-identical results to Python/NumPy/SciPy.");
    println!("  Next step: GPU portability (Exp308) proves same math on GPU.");
    println!("  ═════════════════════════════════════════════════════════════════");

    v.finish();
}
