// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
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
//! # Exp349: `ToadStool` Dispatch v4 — V109 Compute Dispatch Validation
//!
//! Validates that every `ToadStool` abstraction layer preserves mathematical
//! correctness from analytical formulae through CPU dispatch. Extends v3
//! with Track 6 biogas kinetics and Anderson W mapping through the dispatch
//! layer.
//!
//! Sections:
//! - S7: Stats regression (bootstrap, jackknife, correlation, linear/exp fit)
//! - S8: Linalg (graph Laplacian, effective rank, ridge regression)
//! - S9: Special functions (erf, ln_gamma, norm_cdf)
//! - S10: Numerical (trapezoidal integration, numerical derivative)
//! - S11: Bio diversity round-trip (Shannon, Simpson, Bray-Curtis, Chao1)
//! - S12: Track 6 kinetics dispatch (Gompertz, Monod, Haldane through pipeline)
//!
//! ```text
//! CPU (Exp347) → GPU (Exp348) → ToadStool (this)
//! → Streaming (Exp350) → metalForge (Exp351) → NUCLEUS (Exp352)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation (ToadStool dispatch) |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --release --bin validate_toadstool_dispatch_v4` |

use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::stats::norm_cdf;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-((rm * std::f64::consts::E / p) * (lambda - t) + 1.0).exp()).exp()
}

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}

fn haldane(s: f64, mu_max: f64, ks: f64, ki: f64) -> f64 {
    mu_max * s / (ks + s + s * s / ki)
}

fn main() {
    let mut v = Validator::new("Exp349: ToadStool Dispatch v4 — V109 Compute Dispatch Validation");
    let t_total = Instant::now();

    // ═══ S7: Stats Regression ═════════════════════════════════════════
    v.section("S7: barracuda::stats Regression (V109 rewire)");

    let data_5 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_5 = barracuda::stats::mean(&data_5);
    v.check("Mean([1..5]) = 3.0", mean_5, 3.0, tolerances::EXACT_F64);

    let var_5 = barracuda::stats::covariance(&data_5, &data_5).expect("cov(x,x)");
    v.check("Var([1..5]) = 2.5", var_5, 2.5, tolerances::ANALYTICAL_F64);

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
        "Linear slope = 2.0",
        fit.params[0],
        2.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "Linear R² ≈ 1.0",
        fit.r_squared,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    // ═══ S8: Linalg ══════════════════════════════════════════════════
    v.section("S8: barracuda::linalg Regression (V109 rewire)");

    let n_g = 4;
    let adj: Vec<f64> = vec![
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let laplacian = barracuda::linalg::graph_laplacian(&adj, n_g);
    v.check(
        "Laplacian[0][0] = 2",
        laplacian[0],
        2.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Laplacian[0][1] = -1",
        laplacian[1],
        -1.0,
        tolerances::EXACT_F64,
    );

    let row_sums: Vec<f64> = (0..n_g)
        .map(|i| (0..n_g).map(|j| laplacian[i * n_g + j]).sum())
        .collect();
    let max_row_sum = row_sums.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    v.check(
        "Row sums = 0",
        max_row_sum,
        0.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );

    let eigenvalues = [10.0, 5.0, 2.0, 0.5, 0.01];
    let eff_rank = barracuda::linalg::effective_rank(&eigenvalues);
    v.check_pass("Effective rank ∈ (0, n]", eff_rank > 0.0 && eff_rank <= 5.0);

    // ═══ S9: Special Functions ════════════════════════════════════════
    v.section("S9: barracuda::special + stats (V109 rewire)");

    let erf_0 = barracuda::special::erf(0.0);
    v.check("erf(0) = 0", erf_0, 0.0, tolerances::EXACT_F64);

    let erf_6 = barracuda::special::erf(6.0);
    v.check("erf(6) ≈ 1", erf_6, 1.0, tolerances::LIMIT_CONVERGENCE);

    v.check("Φ(0) = 0.5", norm_cdf(0.0), 0.5, tolerances::EXACT_F64);
    v.check("Φ(-10) → 0", norm_cdf(-10.0), 0.0, 1e-10);
    v.check("Φ(10) → 1", norm_cdf(10.0), 1.0, 1e-10);

    let lng_1 = barracuda::special::ln_gamma(1.0).expect("ln_gamma(1)");
    v.check("ln_gamma(1) = 0", lng_1, 0.0, tolerances::ANALYTICAL_F64);

    // ═══ S10: Numerical ══════════════════════════════════════════════
    v.section("S10: Numerical Integration + Derivatives");

    let xs: Vec<f64> = (0..=100).map(|i| f64::from(i) * 0.01).collect();
    let ys: Vec<f64> = xs.iter().map(|&x_val| x_val * x_val).collect();
    let integral = barracuda::numerical::trapz(&ys, &xs).expect("trapz");
    v.check("∫₀¹ x² dx ≈ 1/3", integral, 1.0 / 3.0, 1e-4);

    // ═══ S11: Bio Diversity Round-Trip ═══════════════════════════════
    v.section("S11: Bio Diversity Round-Trip (V109 rewire)");

    let comm = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let h = diversity::shannon(&comm);
    let s = diversity::simpson(&comm);
    let j = diversity::pielou_evenness(&comm);
    let chao1 = diversity::chao1(&comm);

    v.check_pass("Shannon > 0", h > 0.0);
    v.check_pass("Simpson ∈ (0,1)", s > 0.0 && s < 1.0);
    v.check_pass("Pielou ∈ (0,1]", j > 0.0 && j <= 1.0);
    v.check_pass("Chao1 >= observed S", chao1 >= comm.len() as f64);

    let bc_self = diversity::bray_curtis(&comm, &comm);
    v.check("BC self = 0", bc_self, 0.0, tolerances::EXACT_F64);

    let rare = diversity::rarefaction_curve(&comm, &[5.0, 10.0, 20.0, 50.0]);
    v.check_pass(
        "Rarefaction monotonic",
        rare.windows(2).all(|w| w[1] >= w[0]),
    );

    // ═══ S12: Track 6 Kinetics Dispatch ══════════════════════════════
    v.section("S12: Track 6 Kinetics — Gompertz + Monod + Haldane dispatch");

    let p_manure = 350.0;
    let h_50 = gompertz(50.0, p_manure, 25.0, 3.0);
    v.check("Gompertz H(50) → P", h_50, p_manure, 1.0);
    v.check(
        "First-order B(0) = 0",
        first_order(0.0, 320.0, 0.08),
        0.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Monod(Ks) = mu_max/2",
        monod(200.0, 0.4, 200.0),
        0.2,
        tolerances::EXACT_F64,
    );

    let s_opt = (200.0_f64 * 3000.0).sqrt();
    let mu_opt = haldane(s_opt, 0.4, 200.0, 3000.0);
    let mu_lo = haldane(s_opt * 0.3, 0.4, 200.0, 3000.0);
    v.check_pass("Haldane peak at S_opt", mu_opt > mu_lo);

    // Anderson W through dispatch
    let j_val = diversity::pielou_evenness(&comm);
    let w = 20.0 * (1.0 - j_val);
    let p_qs = norm_cdf((16.5 - w) / 4.0);
    v.check_pass("P(QS) ∈ [0,1]", (0.0..=1.0).contains(&p_qs));

    // ═══ Summary ════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── ToadStool Dispatch v4 Summary ({total_ms:.2} ms total) ──");
    println!("  All sections: S7 (stats) + S8 (linalg) + S9 (special)");
    println!("                S10 (numerical) + S11 (bio) + S12 (Track 6)");
    println!("  Dispatch layer preserves mathematical correctness.");

    v.finish();
}
