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
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! Cross-spring evolution validation — modern `ToadStool` primitives.
//!
//! Validates primitives evolved from ALL five Springs, proving the
//! cross-spring evolution model: each Spring contributes domain expertise
//! to `BarraCuda`, and all Springs benefit.
//!
//! | Spring | Contribution |
//! |--------|-------------|
//! | hotSpring | Precision math, `ReduceScalarPipeline`, ESN, DF64 |
//! | wetSpring | Bio diversity, HMM, Felsenstein, DADA2, `log_f64` fix |
//! | neuralSpring | ML ops, pairwise, eigensolver, tensor session |
//! | groundSpring | Evolution (Kimura), jackknife, bootstrap, grid |
//! | airSpring | Hydrology, Brent optimizer, seasonal pipeline |
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cross_spring_evolution_modern` |
//!
//! Validation class: Cross-spring
//!
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Cross-Spring Evolution — Modern ToadStool Primitives");

    // ═══ hotSpring: Precision math ═══════════════════════════════════════════
    v.section("═══ hotSpring: Precision math ═══");

    // 1. mean — basic mean with known values (exact by construction)
    let data_mean = [1.0, 2.0, 3.0, 4.0, 5.0];
    let m = barracuda::stats::mean(&data_mean);
    v.check("mean([1..5]) = 3.0", m, 3.0, tolerances::EXACT);

    // 2. variance — sample variance (n-1): [1,2,3,4,5] → 10/4 = 2.5
    let var = barracuda::stats::correlation::variance(&data_mean).or_exit("variance");
    let expected_var = 2.5; // sum((x-3)²)/(n-1) = (4+1+0+1+4)/4 = 2.5
    v.check(
        "variance([1..5]) = 2.5",
        var,
        expected_var,
        tolerances::ANALYTICAL_F64,
    );

    // 3. ridge_regression — known system y = 2*x1 + 3*x2
    // X: 3 samples, 2 features. y = [8, 13, 18] for x1=[1,2,3], x2=[2,3,4]
    let ridge_x: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
    let ridge_y: Vec<f64> = vec![8.0, 13.0, 18.0]; // 2*1+3*2=8, 2*2+3*3=13, 2*3+3*4=18
    let ridge = barracuda::linalg::ridge_regression(
        &ridge_x,
        &ridge_y,
        3,
        2,
        1,
        tolerances::RIDGE_REGULARIZATION_SMALL,
    )
    .or_exit("ridge_regression");
    v.check(
        "ridge: slope x1 ≈ 2",
        ridge.weights[0],
        2.0,
        tolerances::RIDGE_TEST_TOL,
    );
    v.check(
        "ridge: slope x2 ≈ 3",
        ridge.weights[1],
        3.0,
        tolerances::RIDGE_TEST_TOL,
    );

    // ═══ wetSpring: Bio diversity ═════════════════════════════════════════════
    v.section("═══ wetSpring: Bio diversity ═══");

    // 4. shannon — uniform community of 4 species: H = ln(4) ≈ 1.386
    let uniform = [25.0, 25.0, 25.0, 25.0];
    let h = barracuda::stats::shannon(&uniform);
    let expected_h = 4.0_f64.ln();
    v.check(
        "shannon(uniform,4) = ln(4)",
        h,
        expected_h,
        tolerances::ANALYTICAL_F64,
    );

    // 5. simpson — uniform 10 species: D = 1 - 10*(0.1²) = 0.9
    let uniform10 = [100.0; 10];
    let sim = barracuda::stats::simpson(&uniform10);
    v.check(
        "simpson(uniform,10) = 0.9",
        sim,
        0.9,
        tolerances::ANALYTICAL_F64,
    );

    // 6. bray_curtis — identical vectors → 0; disjoint → 1
    let a = [10.0, 20.0, 30.0];
    let b = [10.0, 20.0, 30.0];
    let bc_same = barracuda::stats::bray_curtis(&a, &b);
    v.check(
        "bray_curtis(identical) = 0",
        bc_same,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    let c = [10.0, 0.0, 0.0];
    let d = [0.0, 0.0, 10.0];
    let bc_disjoint = barracuda::stats::bray_curtis(&c, &d);
    v.check(
        "bray_curtis(disjoint) = 1",
        bc_disjoint,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // 7. chao1 — bias-corrected: S_obs + f1*(f1-1)/(2*(f2+1))
    let counts = vec![10.0, 5.0, 3.0, 2.0, 1.0, 1.0, 20.0, 7.0, 1.0];
    let chao = barracuda::stats::chao1(&counts);
    let s_obs = 9.0_f64;
    let f1 = 3.0_f64;
    let f2 = 1.0_f64;
    let expected_chao = s_obs + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0));
    v.check(
        "chao1 matches bias-corrected",
        chao,
        expected_chao,
        tolerances::ANALYTICAL_F64,
    );

    // 8. erf — error function at known points (Abramowitz & Stegun)
    let erf_0 = barracuda::special::erf(0.0);
    v.check("erf(0) = 0", erf_0, 0.0, tolerances::ERF_PARITY);
    let erf_1 = barracuda::special::erf(1.0);
    v.check(
        "erf(1) ≈ 0.8427",
        erf_1,
        0.842_700_792_949_714_9,
        tolerances::ERF_PARITY,
    );

    // ═══ neuralSpring: ML/stats ══════════════════════════════════════════════
    v.section("═══ neuralSpring: ML/stats ═══");

    // 9. pearson_correlation — perfect linear y = 2x + 1 → r = 1
    let x_lin: Vec<f64> = (0..20).map(f64::from).collect();
    let y_lin: Vec<f64> = x_lin.iter().map(|&x| 2.0f64.mul_add(x, 1.0)).collect();
    let pearson = barracuda::stats::pearson_correlation(&x_lin, &y_lin).or_exit("unexpected error");
    v.check(
        "pearson(perfect linear) = 1",
        pearson,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // 10. spearman_correlation — perfect ranks → ρ = 1
    let x_rank: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_rank: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // perfect negative
    let spearman =
        barracuda::stats::spearman_correlation(&x_rank, &y_rank).or_exit("unexpected error");
    v.check(
        "spearman(perfect neg ranks) = -1",
        spearman,
        -1.0,
        tolerances::ANALYTICAL_F64,
    );

    // 11. fit_linear — known line y = 3x + 2
    let x_fit: Vec<f64> = (0..15).map(f64::from).collect();
    let y_fit: Vec<f64> = x_fit.iter().map(|&x| 3.0f64.mul_add(x, 2.0)).collect();
    let fit = barracuda::stats::fit_linear(&x_fit, &y_fit).or_exit("unexpected error");
    v.check(
        "fit_linear: slope ≈ 3",
        fit.params[0],
        3.0,
        tolerances::RIDGE_TEST_TOL,
    );
    v.check(
        "fit_linear: intercept ≈ 2",
        fit.params[1],
        2.0,
        tolerances::RIDGE_TEST_TOL,
    );

    // ═══ groundSpring: Evolution ═════════════════════════════════════════════
    v.section("═══ groundSpring: Evolution ═══");

    // 12. kimura_fixation_prob — neutral: P_fix = p0 (drift)
    let p_neutral = barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01);
    v.check(
        "kimura(neutral): P_fix = p0",
        p_neutral,
        0.01,
        tolerances::LIMIT_CONVERGENCE,
    );

    // 13. detection_power — P(detect) = 1 - (1-p)^D, exact by construction
    let power = barracuda::stats::detection_power(0.001, 1000);
    let expected_power = 1.0 - 0.999_f64.powi(1000);
    v.check(
        "detection_power matches analytic",
        power,
        expected_power,
        tolerances::ANALYTICAL_F64,
    );

    // 14. jackknife_mean_variance — [1,2,3,4,5] → mean=3, var>0
    let jk_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let jk = barracuda::stats::jackknife_mean_variance(&jk_data).or_exit("unexpected error");
    v.check(
        "jackknife mean = 3",
        jk.estimate,
        3.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass("jackknife variance ≥ 0", jk.variance >= 0.0);

    // ═══ airSpring: Hydrology → general stats ═══════════════════════════════
    v.section("═══ airSpring: Hydrology → general stats ═══");

    // 15. trapz — ∫₀¹ x² dx = 1/3 (exact by trapezoid on fine grid)
    let trap_x: Vec<f64> = (0..1001).map(|i| f64::from(i) / 1000.0).collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|x| x * x).collect();
    let trapz_val = barracuda::numerical::trapz(&trap_y, &trap_x).or_exit("trapz");
    v.check(
        "trapz(x²) ≈ 1/3",
        trapz_val,
        1.0 / 3.0,
        tolerances::TRAPZ_COARSE,
    );

    // 16. fit_exponential — y = 2*exp(0.15*x) + 0.5
    let x_exp: Vec<f64> = (0..20).map(f64::from).collect();
    let y_exp: Vec<f64> = x_exp
        .iter()
        .map(|&x| 2.0f64.mul_add((0.15 * x).exp(), 0.5))
        .collect();
    let fit_exp = barracuda::stats::fit_exponential(&x_exp, &y_exp).or_exit("unexpected error");
    v.check_pass("fit_exponential: Some", true);
    v.check_pass("fit_exponential R² > 0.95", fit_exp.r_squared > 0.95);
    let pred = fit_exp.predict_one(10.0).or_exit("unexpected error");
    let expected_pred = 2.0f64.mul_add((0.15_f64 * 10.0).exp(), 0.5);
    v.check(
        "fit_exponential predict(10) ≈ truth",
        pred,
        expected_pred,
        expected_pred * 0.05,
    );

    // ═══ Summary ════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Cross-Spring Evolution: 16 primitives validated");
    println!("  hotSpring:   mean, variance, ridge_regression");
    println!("  wetSpring:   shannon, simpson, bray_curtis, chao1, erf");
    println!("  neuralSpring: pearson, spearman, fit_linear");
    println!("  groundSpring: kimura_fixation_prob, detection_power, jackknife");
    println!("  airSpring:   trapz, fit_exponential");
    println!("═══════════════════════════════════════════════════════════════");

    v.finish();
}
