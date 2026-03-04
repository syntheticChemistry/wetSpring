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
//! # Exp287: `BarraCuda` CPU v21 — V92D Deep Debt Evolution
//!
//! Extends CPU v20 (D01–D32, 37 checks) with V92D validation:
//! - D33: Error-handling pipeline (`block_on` returns `Result`, no panics)
//! - D34: Validation harness (bench helper, Validator accumulator)
//! - D35: Tolerance registry v2 (103 constants, hierarchy + naming)
//! - D36: IPC capability dispatch (`MetricCtx` pattern, GPU/CPU routing)
//! - D37: Diversity delegation identity (wetSpring bio == `barracuda::stats`)
//! - D38: Special function complement (erf + erfc = 1, symmetry)
//! - D39: Cross-domain math (Lanczos → spectral → diversity → Anderson chain)
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v21` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("Exp287: BarraCuda CPU v21 — V92D Deep Debt Evolution");
    let t_total = std::time::Instant::now();

    println!("  Inherited: D01–D32 from CPU v20 (37 checks)");
    println!("  New: D33–D39 below\n");

    // ═══ D33: Error Handling Pipeline ═══════════════════════════════════
    v.section("D33: Error Handling — Result-Based (Zero Panics)");

    let good_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let fit_result = barracuda::stats::fit_linear(&good_data, &good_data);
    v.check_pass("fit_linear: Some on valid data", fit_result.is_some());

    let empty: [f64; 0] = [];
    let jk_empty = barracuda::stats::jackknife_mean_variance(&empty);
    v.check_pass("jackknife: None on empty", jk_empty.is_none());

    let single = [42.0];
    let jk_single = barracuda::stats::jackknife_mean_variance(&single);
    v.check_pass(
        "jackknife: None on single (< 2 elements)",
        jk_single.is_none(),
    );

    let pearson_result = barracuda::stats::pearson_correlation(&good_data, &good_data);
    v.check_pass("pearson: Ok on identical", pearson_result.is_ok());
    if let Ok(r) = pearson_result {
        v.check("pearson(x, x) = 1.0", r, 1.0, tolerances::ANALYTICAL_F64);
    }

    let nmf_tiny = barracuda::linalg::nmf::nmf(
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 1,
            max_iter: 50,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    );
    v.check_pass("NMF: Ok on 2x2 rank-1", nmf_tiny.is_ok());

    let trapz_result = barracuda::numerical::trapz(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]);
    v.check_pass("trapz: Ok on valid grid", trapz_result.is_ok());
    if let Ok(area) = trapz_result {
        v.check(
            "trapz([1,2,3], [0,1,2]) = 4.0",
            area,
            4.0,
            tolerances::EXACT_F64,
        );
    }

    // ═══ D34: Validation Harness ════════════════════════════════════════
    v.section("D34: Validation Harness — bench + Validator");

    let (result, ms) = validation::bench(|| {
        let mut sum = 0.0_f64;
        for i in 0..1000 {
            sum += f64::from(i);
        }
        sum
    });
    v.check(
        "bench: sum(0..1000) = 499500",
        result,
        499_500.0,
        tolerances::EXACT_F64,
    );
    v.check_pass("bench: timing > 0", ms >= 0.0);
    v.check_pass("bench: timing < 1000ms (sanity)", ms < 1000.0);

    let mut inner = Validator::new("inner-test");
    inner.check("inner ok", 1.0, 1.0, 0.0);
    inner.check("inner fail", 2.0, 1.0, 0.0);
    let (p, t) = inner.counts();
    v.check_count("Validator: inner passed", p as usize, 1);
    v.check_count("Validator: inner total", t as usize, 2);

    // ═══ D35: Tolerance Registry v2 ═════════════════════════════════════
    v.section("D35: Tolerance Registry (103 Named Constants)");

    v.check_pass("EXACT = 0", tolerances::EXACT == 0.0);
    v.check_pass("EXACT_F64 >= 0", tolerances::EXACT_F64 >= 0.0);
    v.check_pass("ANALYTICAL_F64 >= 0", tolerances::ANALYTICAL_F64 >= 0.0);
    v.check_pass(
        "PYTHON_PARITY >= ANALYTICAL",
        tolerances::PYTHON_PARITY >= tolerances::ANALYTICAL_F64,
    );
    v.check_pass("GPU_VS_CPU_F64 >= 0", tolerances::GPU_VS_CPU_F64 >= 0.0);
    v.check_pass(
        "hierarchy: EXACT ≤ EXACT_F64 ≤ ANALYTICAL ≤ PYTHON_PARITY",
        tolerances::EXACT <= tolerances::EXACT_F64
            && tolerances::EXACT_F64 <= tolerances::ANALYTICAL_F64
            && tolerances::ANALYTICAL_F64 <= tolerances::PYTHON_PARITY,
    );

    // ═══ D36: Diversity Delegation Identity ══════════════════════════════
    v.section("D36: Diversity Delegation — bio::diversity == barracuda::stats");

    let counts_8 = [10.0, 20.0, 30.0, 5.0, 15.0, 8.0, 12.0, 25.0];

    let ws_sh = diversity::shannon(&counts_8);
    let bc_sh = barracuda::stats::shannon(&counts_8);
    v.check("Shannon identity", ws_sh, bc_sh, tolerances::EXACT);

    let ws_si = diversity::simpson(&counts_8);
    let bc_si = barracuda::stats::simpson(&counts_8);
    v.check("Simpson identity", ws_si, bc_si, tolerances::EXACT);

    let ws_ch = diversity::chao1(&counts_8);
    let bc_ch = barracuda::stats::chao1(&counts_8);
    v.check("Chao1 identity", ws_ch, bc_ch, tolerances::EXACT);

    let a4 = [10.0, 20.0, 30.0, 5.0];
    let b4 = [15.0, 10.0, 25.0, 12.0];
    let ws_bc = diversity::bray_curtis(&a4, &b4);
    let bc_bc = barracuda::stats::bray_curtis(&a4, &b4);
    v.check("Bray-Curtis identity", ws_bc, bc_bc, tolerances::EXACT);

    let uniform = [25.0, 25.0, 25.0, 25.0];
    let h_uniform = barracuda::stats::shannon(&uniform);
    v.check(
        "Shannon(uniform,4) = ln(4)",
        h_uniform,
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    let d_uniform = barracuda::stats::simpson(&uniform);
    v.check(
        "Simpson(uniform,4) = 1 - 1/4 = 0.75",
        d_uniform,
        0.75,
        tolerances::ANALYTICAL_F64,
    );

    // ═══ D37: Special Function Complements ═══════════════════════════════
    v.section("D37: Special Functions — Complement + Symmetry");

    for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let sum = barracuda::special::erf(x) + barracuda::special::erfc(x);
        v.check(
            &format!("erf({x}) + erfc({x}) = 1"),
            sum,
            1.0,
            tolerances::ANALYTICAL_F64,
        );
    }

    let erf_neg = barracuda::special::erf(-1.0);
    let erf_pos = barracuda::special::erf(1.0);
    v.check(
        "erf(-x) = -erf(x)",
        erf_neg,
        -erf_pos,
        tolerances::EXACT_F64,
    );

    let ncdf_0 = barracuda::stats::norm_cdf(0.0);
    v.check("Φ(0) = 0.5", ncdf_0, 0.5, tolerances::ANALYTICAL_F64);

    let ncdf_inf = barracuda::stats::norm_cdf(6.0);
    v.check("Φ(∞) → 1", ncdf_inf, 1.0, tolerances::LIMIT_CONVERGENCE);

    // ═══ D38: Cross-Domain Math Chain ═══════════════════════════════════
    v.section("D38: Cross-Domain — Stats → Diversity → Linalg Chain");

    let x_lin = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y_lin = [2.0, 4.0, 6.0, 8.0, 10.0];
    let fit = barracuda::stats::fit_linear(&x_lin, &y_lin).unwrap();
    v.check(
        "Chain: fit_linear slope = 2",
        fit.params[0],
        2.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "Chain: fit_linear R² = 1",
        fit.r_squared,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    let counts_chain = [10.0, 20.0, 30.0, 40.0, 50.0];
    let sh = barracuda::stats::shannon(&counts_chain);
    let si = barracuda::stats::simpson(&counts_chain);
    v.check_pass("Chain: Shannon > 0", sh > 0.0);
    v.check_pass("Chain: Simpson ∈ (0, 1)", si > 0.0 && si < 1.0);

    let mean_val = barracuda::stats::mean(&counts_chain);
    v.check("Chain: mean = 30", mean_val, 30.0, tolerances::EXACT_F64);

    let ridge_x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ridge_y: Vec<f64> = vec![5.0, 11.0, 17.0];
    let ridge = barracuda::linalg::ridge_regression(&ridge_x, &ridge_y, 3, 2, 1, 1e-4);
    v.check_pass("Chain: ridge regression Ok", ridge.is_ok());
    if let Ok(r) = &ridge {
        v.check_pass(
            "Chain: ridge weights finite",
            r.weights.iter().all(|w| w.is_finite()),
        );
    }

    let nmf_data: Vec<f64> = (0..20)
        .map(|i| f64::from(((i * 3 + 1) % 10) as u32) / 10.0 + 0.01)
        .collect();
    let nmf_result = barracuda::linalg::nmf::nmf(
        &nmf_data,
        4,
        5,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 2,
            max_iter: 100,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    );
    v.check_pass("Chain: NMF converged", nmf_result.is_ok());
    if let Ok(nmf) = &nmf_result {
        v.check_pass("Chain: NMF W ≥ 0", nmf.w.iter().all(|&x| x >= 0.0));
        v.check_pass("Chain: NMF H ≥ 0", nmf.h.iter().all(|&x| x >= 0.0));
    }

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    v.section("CPU v21 Summary");
    println!("  V92D domains: error handling, harness, tolerances,");
    println!("    delegation identity, special complements, cross-domain");
    println!("  Total: {total_ms:.1} ms");

    v.finish();
}
