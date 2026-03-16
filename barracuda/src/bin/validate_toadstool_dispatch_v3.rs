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
//! # Exp267: `ToadStool` Dispatch v3 — Pure Rust Math Validation
//!
//! Validates pure Rust math across all `barracuda` primitives consumed
//! by wetSpring. Proves that every `ToadStool` abstraction layer preserves
//! mathematical correctness from analytical formulae through CPU.
//!
//! Sections:
//! - S1: Stats regression (bootstrap, jackknife, correlation, regression)
//! - S2: Linalg (graph Laplacian, effective rank, ridge regression, NMF)
//! - S3: Special functions (`erf`, `ln_gamma`)
//! - S4: Numerical (Hessian, trapezoidal integration)
//! - S5: Diversity round-trip (wetSpring bio → `barracuda::stats` identity)
//! - S6: Spectral (Anderson, Lanczos, level spacing ratio)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_toadstool_dispatch_v3` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp267: ToadStool Dispatch v3 — Pure Rust Math Validation");
    let t_total = Instant::now();

    // ═══ S1: Stats Regression ═════════════════════════════════════════════
    v.section("S1: barracuda::stats Regression");

    let data_5 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_5 = barracuda::stats::mean(&data_5);
    v.check("Mean([1..5]) = 3.0", mean_5, 3.0, tolerances::EXACT_F64);

    let ci = barracuda::stats::bootstrap_ci(
        &data_5,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        10_000,
        0.95,
        42,
    )
    .unwrap();
    v.check_pass("Bootstrap CI: lower < estimate", ci.lower <= ci.estimate);
    v.check_pass("Bootstrap CI: estimate < upper", ci.estimate <= ci.upper);
    v.check_pass("Bootstrap SE > 0", ci.std_error > 0.0);

    let jk = barracuda::stats::jackknife_mean_variance(&data_5).unwrap();
    v.check(
        "Jackknife mean = 3.0",
        jk.estimate,
        3.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass("Jackknife SE > 0", jk.std_error > 0.0);

    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let pearson = barracuda::stats::pearson_correlation(&x, &y).unwrap();
    v.check(
        "Pearson(x, 2x) = 1.0",
        pearson,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let spearman = barracuda::stats::spearman_correlation(&x, &y).unwrap();
    v.check(
        "Spearman(x, 2x) = 1.0",
        spearman,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let fit = barracuda::stats::fit_linear(&x, &y).unwrap();
    v.check(
        "Linear fit: slope = 2.0",
        fit.params[0],
        2.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "Linear fit: intercept = 0.0",
        fit.params[1],
        0.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "Linear fit: r² ≈ 1.0",
        fit.r_squared,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    let exp_x = [0.0_f64, 1.0, 2.0, 3.0];
    let exp_y: Vec<f64> = exp_x.iter().map(|&xi| (2.0 * xi).exp()).collect();
    let exp_fit = barracuda::stats::fit_exponential(&exp_x, &exp_y).unwrap();
    v.check_pass("Exp fit: R² > 0.95", exp_fit.r_squared > 0.95);

    // ═══ S2: Linalg ══════════════════════════════════════════════════════
    v.section("S2: barracuda::linalg Regression");

    let n_g = 4;
    let adj: Vec<f64> = vec![
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let laplacian = barracuda::linalg::graph_laplacian(&adj, n_g);
    v.check(
        "Laplacian[0][0] = degree 2",
        laplacian[0],
        2.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Laplacian[0][1] = -adj = -1",
        laplacian[1],
        -1.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Laplacian[2][2] = degree 3",
        laplacian[2 * n_g + 2],
        3.0,
        tolerances::EXACT_F64,
    );

    let row_sums: Vec<f64> = (0..n_g)
        .map(|i| (0..n_g).map(|j| laplacian[i * n_g + j]).sum())
        .collect();
    let max_row_sum = row_sums.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    v.check(
        "Laplacian row sums = 0",
        max_row_sum,
        0.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );

    let eigenvalues = [10.0, 5.0, 2.0, 0.5, 0.01, 0.001];
    let eff_rank = barracuda::linalg::effective_rank(&eigenvalues);
    v.check_pass("Effective rank > 0", eff_rank > 0.0);
    v.check_pass("Effective rank ≤ n", eff_rank <= eigenvalues.len() as f64);

    let n_s = 3;
    let n_f = 2;
    let ridge_x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ridge_y: Vec<f64> = vec![5.0, 11.0, 17.0];
    let ridge = barracuda::linalg::ridge_regression(
        &ridge_x,
        &ridge_y,
        n_s,
        n_f,
        1,
        tolerances::RIDGE_REGULARIZATION_DEFAULT,
    );
    v.check_pass("Ridge regression: Ok", ridge.is_ok());
    if let Ok(r) = &ridge {
        v.check_pass(
            "Ridge weights finite",
            r.weights.iter().all(|w| w.is_finite()),
        );
    }

    let v_mat: Vec<f64> = (0..60)
        .map(|i| f64::from(((i * 3 + 1) % 50) as u32) / 50.0 + 0.01)
        .collect();
    let nmf_cfg = barracuda::linalg::nmf::NmfConfig {
        rank: 2,
        max_iter: 100,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_result = barracuda::linalg::nmf::nmf(&v_mat, 6, 10, &nmf_cfg);
    v.check_pass("NMF: converged", nmf_result.is_ok());
    if let Ok(nmf) = &nmf_result {
        v.check_pass("NMF W ≥ 0", nmf.w.iter().all(|&x| x >= 0.0));
        v.check_pass("NMF H ≥ 0", nmf.h.iter().all(|&x| x >= 0.0));
    }

    // ═══ S3: Special Functions ════════════════════════════════════════════
    v.section("S3: barracuda::special Functions");

    let erf_0 = barracuda::special::erf(0.0);
    v.check("erf(0) = 0", erf_0, 0.0, tolerances::EXACT_F64);

    let erf_inf = barracuda::special::erf(10.0);
    v.check("erf(∞) → 1", erf_inf, 1.0, tolerances::ANALYTICAL_LOOSE);

    let erfc_0 = barracuda::special::erfc(0.0);
    v.check("erfc(0) = 1", erfc_0, 1.0, tolerances::EXACT_F64);

    let erf_complement = barracuda::special::erf(1.5) + barracuda::special::erfc(1.5);
    v.check(
        "erf(x) + erfc(x) = 1",
        erf_complement,
        1.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );

    let ln_gamma_1 = barracuda::special::ln_gamma(1.0).unwrap();
    v.check(
        "ln_gamma(1) = 0",
        ln_gamma_1,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    let ln_gamma_5 = barracuda::special::ln_gamma(5.0).unwrap();
    let expected_ln_24 = (24.0_f64).ln();
    v.check(
        "ln_gamma(5) = ln(4!) = ln(24)",
        ln_gamma_5,
        expected_ln_24,
        tolerances::ANALYTICAL_LOOSE,
    );

    // ═══ S4: Numerical ═══════════════════════════════════════════════════
    v.section("S4: barracuda::numerical Integration + Differentiation");

    let trap_x: Vec<f64> = (0..=100).map(|i| f64::from(i) * 0.01).collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|&x| x * x).collect();
    let trap_area = barracuda::numerical::trapz(&trap_y, &trap_x).unwrap();
    v.check(
        "∫x² dx [0,1] ≈ 1/3",
        trap_area,
        1.0 / 3.0,
        tolerances::TRAPZ_101,
    );

    let hessian = barracuda::numerical::numerical_hessian(
        &|x: &[f64]| x[0].mul_add(x[0], x[1] * x[1]),
        &[1.0, 1.0],
        tolerances::NUMERICAL_HESSIAN_EPSILON,
    );
    v.check(
        "Hessian[0][0] of x²+y² = 2",
        hessian[0],
        2.0,
        tolerances::HESSIAN_TEST_TOL,
    );
    v.check(
        "Hessian[1][1] of x²+y² = 2",
        hessian[3],
        2.0,
        tolerances::HESSIAN_TEST_TOL,
    );
    v.check(
        "Hessian[0][1] cross ≈ 0",
        hessian[1],
        0.0,
        tolerances::HESSIAN_TEST_TOL,
    );

    // ═══ S5: Diversity Round-Trip ════════════════════════════════════════
    v.section("S5: wetSpring bio::diversity → barracuda::stats Identity");

    let counts = [10.0, 20.0, 30.0, 5.0, 15.0, 8.0, 12.0, 25.0];

    let ws_shannon = diversity::shannon(&counts);
    let bc_shannon = barracuda::stats::shannon(&counts);
    v.check(
        "Shannon: bio == barracuda",
        ws_shannon,
        bc_shannon,
        tolerances::EXACT,
    );

    let ws_simpson = diversity::simpson(&counts);
    let bc_simpson = barracuda::stats::simpson(&counts);
    v.check(
        "Simpson: bio == barracuda",
        ws_simpson,
        bc_simpson,
        tolerances::EXACT,
    );

    let ws_chao1 = diversity::chao1(&counts);
    let bc_chao1 = barracuda::stats::chao1(&counts);
    v.check(
        "Chao1: bio == barracuda",
        ws_chao1,
        bc_chao1,
        tolerances::EXACT,
    );

    let a = [10.0, 20.0, 30.0, 5.0];
    let b = [15.0, 10.0, 25.0, 12.0];
    let ws_bc = diversity::bray_curtis(&a, &b);
    let bc_bc = barracuda::stats::bray_curtis(&a, &b);
    v.check(
        "Bray-Curtis: bio == barracuda",
        ws_bc,
        bc_bc,
        tolerances::EXACT,
    );

    // ═══ S6: Spectral (Anderson Coupling) ════════════════════════════════
    v.section("S6: barracuda::spectral Anderson Model");

    let csr = barracuda::spectral::anderson_3d(4, 4, 4, 4.0, 42);
    let tri = barracuda::spectral::lanczos(&csr, 30, 42);
    let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
    v.check_pass("Lanczos: eigenvalues computed", !eigs.is_empty());
    v.check_pass(
        "Lanczos: all finite",
        eigs.iter().all(|e: &f64| e.is_finite()),
    );

    let lsr = barracuda::spectral::level_spacing_ratio(&eigs);
    v.check_pass("Level spacing ratio > 0", lsr > 0.0);
    v.check_pass("Level spacing ratio < 1", lsr < 1.0);
    println!(
        "  Anderson(L=4, W=4): {:.4} LSR, {} eigenvalues",
        lsr,
        eigs.len()
    );

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("ToadStool Dispatch v3 Summary");
    println!("  Pure Rust math across 6 barracuda domains:");
    println!("    stats:      bootstrap, jackknife, correlation, regression");
    println!("    linalg:     Laplacian, rank, ridge, NMF");
    println!("    special:    erf, erfc, ln_gamma");
    println!("    numerical:  trapezoid, Hessian");
    println!("    diversity:  wetSpring→barracuda identity proof");
    println!("    spectral:   Anderson 3D, Lanczos, LSR");
    println!("  Total: {total_ms:.1} ms");
    println!();

    v.finish();
}
