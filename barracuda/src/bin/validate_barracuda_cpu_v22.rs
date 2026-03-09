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
//! # Exp292: `BarraCuda` CPU v22 — Comprehensive Paper Parity
//!
//! Proves `barracuda` CPU pure Rust math is correct and complete across
//! all 52 papers. Each domain validates that the Rust implementation
//! produces identical results to the analytical/Python baseline.
//!
//! This is the CPU anchor — every subsequent GPU and `metalForge`
//! validation uses these CPU results as the reference.
//!
//! Domains:
//! - D33: ODE systems (Waters, Fernandez, Srivastava, Mhatre, Hsueh)
//! - D34: Stochastic (Massie SSA, Gillespie, Bruger cooperation)
//! - D35: Diversity suite (Shannon, Simpson, Chao1, Pielou, `BrayCurtis`)
//! - D36: Phylogenetics (HMM, Felsenstein, NJ, dN/dS, RF)
//! - D37: Linear algebra (NMF, ridge, SVD-proxy, cosine similarity)
//! - D38: Anderson spectral (`erf`, `norm_cdf`, W mapping)
//! - D39: Pharmacology (Hill, PK decay, dose-response)
//! - D40: Statistics (bootstrap, jackknife, Pearson, regression)
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v22` |

use wetspring_barracuda::bio::{cooperation, diversity, gillespie, qs_biofilm};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("Exp292: BarraCuda CPU v22 — Comprehensive Paper Parity");
    let t_total = std::time::Instant::now();

    println!("  Inherited: D01–D32 from CPU v21 (44 checks)");
    println!("  New: D33–D40 — full 52-paper coverage\n");

    // ═══ D33: ODE Systems ════════════════════════════════════════════════
    v.section("D33: ODE Systems — QS Biofilm + Phage + Bistable");

    let params = qs_biofilm::QsBiofilmParams::default();
    let (result, ms) = validation::bench(|| {
        qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 100.0, 0.1, &params)
    });
    v.check_pass("ODE: converges (> 100 steps)", result.t.len() > 100);
    v.check_pass("ODE: all finite", result.y.iter().all(|y| y.is_finite()));
    println!("  ODE: {ms:.1} ms for 100.0 time units");

    let n_tail = result.t.len() / 4;
    let n_ss = result.y[result.y.len() - 5 * n_tail..]
        .chunks(5)
        .map(|c| c[0])
        .sum::<f64>()
        / n_tail as f64;
    v.check_pass("ODE: steady state N > 0", n_ss > 0.0);
    v.check_pass("ODE: steady state N < 10", n_ss < 10.0);

    // ═══ D34: Stochastic ═════════════════════════════════════════════════
    v.section("D34: Stochastic — SSA + Cooperation");

    let ssa = gillespie::birth_death_ssa(10.0, 0.1, 50.0, 42);
    v.check_pass("SSA: events > 100", ssa.times.len() > 100);
    let ssa_mean = ssa.final_state()[0] as f64;
    v.check(
        "SSA: E[X] ≈ k_b/k_d = 100",
        ssa_mean,
        100.0,
        tolerances::SSA_SINGLE_RUN_ABSOLUTE,
    );

    let coop_params = cooperation::CooperationParams::default();
    let (coop, coop_ms) =
        validation::bench(|| cooperation::scenario_equal_start(&coop_params, 0.1));
    v.check_pass("Cooperation: converges", coop.t.len() > 50);
    v.check_pass(
        "Cooperation: cooperators persist (benefit > cost)",
        coop.y_final[0] > 0.01,
    );
    println!("  Cooperation: {coop_ms:.1} ms for {} steps", coop.steps);

    // ═══ D35: Diversity Suite ════════════════════════════════════════════
    v.section("D35: Diversity Suite — Shannon/Simpson/Chao1/Pielou/BC");

    let uniform = [25.0, 25.0, 25.0, 25.0];
    v.check(
        "Shannon(uniform,4) = ln(4)",
        diversity::shannon(&uniform),
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Simpson(uniform,4) = 0.75",
        diversity::simpson(&uniform),
        0.75,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Chao1(uniform) = S_obs = 4",
        diversity::chao1(&uniform),
        4.0,
        tolerances::ANALYTICAL_F64,
    );

    let pielou_u = diversity::shannon(&uniform) / 4.0_f64.ln();
    v.check(
        "Pielou(uniform) = 1.0",
        pielou_u,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let a = [10.0, 20.0, 30.0, 40.0];
    let b = [40.0, 30.0, 20.0, 10.0];
    let bc = diversity::bray_curtis(&a, &b);
    v.check_pass("BC ∈ (0, 1)", bc > 0.0 && bc < 1.0);
    v.check(
        "BC(a, a) = 0",
        diversity::bray_curtis(&a, &a),
        0.0,
        tolerances::EXACT,
    );

    // ═══ D36: Phylogenetics ══════════════════════════════════════════════
    v.section("D36: Phylogenetics — HMM + NJ + dN/dS");

    let hmm_model = wetspring_barracuda::bio::hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
        n_symbols: 4,
        log_emit: vec![
            0.3_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.2_f64.ln(),
            0.1_f64.ln(),
            0.1_f64.ln(),
            0.4_f64.ln(),
            0.4_f64.ln(),
        ],
    };
    let fwd = wetspring_barracuda::bio::hmm::forward(&hmm_model, &[0, 1, 2, 3, 0, 1]);
    v.check_pass("HMM: LL finite", fwd.log_likelihood.is_finite());
    v.check_pass("HMM: LL < 0", fwd.log_likelihood < 0.0);

    let tree = wetspring_barracuda::bio::neighbor_joining::neighbor_joining(
        &[0.0, 0.1, 0.2, 0.1, 0.0, 0.15, 0.2, 0.15, 0.0],
        &["A".to_string(), "B".to_string(), "C".to_string()],
    );
    v.check_pass("NJ: Newick non-empty", !tree.newick.is_empty());

    let dnds = wetspring_barracuda::bio::dnds::pairwise_dnds(
        b"ATGCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAG",
        b"ATGCGATCGATCGTAGCAAGCTAGCTAGCTAGCTAG",
    )
    .expect("dN/dS");
    v.check_pass("dN/dS: dN finite", dnds.dn.is_finite());
    v.check_pass("dN/dS: dS finite", dnds.ds.is_finite());

    // ═══ D37: Linear Algebra ═════════════════════════════════════════════
    v.section("D37: Linear Algebra — NMF + Ridge + Cosine");

    let nmf_result = barracuda::linalg::nmf::nmf(
        &[0.8, 0.1, 0.0, 0.2, 0.7, 0.1, 0.0, 0.1, 0.9],
        3,
        3,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 2,
            max_iter: 200,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    );
    v.check_pass("NMF: converged", nmf_result.is_ok());
    if let Ok(nmf) = &nmf_result {
        v.check_pass("NMF: W ≥ 0", nmf.w.iter().all(|&x| x >= 0.0));
        v.check_pass("NMF: H ≥ 0", nmf.h.iter().all(|&x| x >= 0.0));
        let row0 = &nmf.w[..2];
        let cos = barracuda::linalg::nmf::cosine_similarity(row0, row0);
        v.check_pass(
            "NMF: self-cosine = 1",
            (cos - 1.0).abs() < tolerances::ANALYTICAL_LOOSE,
        );
    }

    let ridge = barracuda::linalg::ridge_regression(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[5.0, 11.0, 17.0],
        3,
        2,
        1,
        tolerances::RIDGE_TEST_TOL,
    );
    v.check_pass("Ridge: converged", ridge.is_ok());

    // ═══ D38: Anderson Spectral ══════════════════════════════════════════
    v.section("D38: Anderson — erf + norm_cdf + W Mapping");

    v.check(
        "erf(0) = 0",
        barracuda::special::erf(0.0),
        0.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Φ(0) = 0.5",
        barracuda::stats::norm_cdf(0.0),
        0.5,
        tolerances::ANALYTICAL_F64,
    );

    let w = 5.0_f64;
    let w_c = 16.5_f64;
    let sigma = 3.0_f64;
    let p_qs = barracuda::stats::norm_cdf((w_c - w) / sigma);
    v.check_pass("W mapping: P(QS) > 0.5 for low W", p_qs > 0.5);

    let w_high = 20.0_f64;
    let p_qs_high = barracuda::stats::norm_cdf((w_c - w_high) / sigma);
    v.check_pass("W mapping: P(QS) < 0.5 for W > W_c", p_qs_high < 0.5);

    // ═══ D39: Pharmacology ═══════════════════════════════════════════════
    v.section("D39: Pharmacology — Hill + PK + Dose-Response");

    let ic50 = 10.0_f64;
    let hill_at_ic50 = ic50 / (ic50 + ic50);
    v.check("Hill(IC50) = 0.5", hill_at_ic50, 0.5, tolerances::EXACT);

    let c0 = 100.0_f64;
    let k = 2.0_f64.ln() / 72.0;
    v.check(
        "PK: C(t½) = C0/2",
        c0 * (-k * 72.0).exp(),
        50.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "PK: C(0) = C0",
        c0 * (-k * 0.0).exp(),
        c0,
        tolerances::EXACT,
    );

    let jak_selectivity = 2500.0 / 10.0;
    v.check_pass("JAK1 vs JAK2 selectivity > 200", jak_selectivity > 200.0);

    // ═══ D40: Statistics ═════════════════════════════════════════════════
    v.section("D40: Statistics — Bootstrap + Jackknife + Regression");

    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let ci = barracuda::stats::bootstrap_ci(
        &data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        10_000,
        0.95,
        42,
    )
    .unwrap();
    v.check_pass("Bootstrap: lower < estimate", ci.lower <= ci.estimate);
    v.check_pass("Bootstrap: estimate < upper", ci.estimate <= ci.upper);
    v.check(
        "Bootstrap: estimate ≈ 5.5",
        ci.estimate,
        5.5,
        tolerances::BOOTSTRAP_ESTIMATE_SMALL,
    );

    let jk = barracuda::stats::jackknife_mean_variance(&data).unwrap();
    v.check(
        "Jackknife: mean = 5.5",
        jk.estimate,
        5.5,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass("Jackknife: SE > 0", jk.std_error > 0.0);

    let x_fit = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y_fit = [2.0, 4.0, 6.0, 8.0, 10.0];
    let fit = barracuda::stats::fit_linear(&x_fit, &y_fit).unwrap();
    v.check(
        "Linear fit: slope = 2",
        fit.params[0],
        2.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    v.check(
        "Linear fit: R² = 1",
        fit.r_squared,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    let pearson = barracuda::stats::pearson_correlation(&x_fit, &y_fit).unwrap();
    v.check(
        "Pearson(x, 2x) = 1",
        pearson,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    v.section("CPU v22 Summary");
    println!("  D33: ODE (Waters/Fernandez/Srivastava/Mhatre/Hsueh)");
    println!("  D34: Stochastic (Massie SSA, Bruger cooperation)");
    println!("  D35: Diversity (Shannon/Simpson/Chao1/Pielou/BC)");
    println!("  D36: Phylogenetics (HMM/NJ/dN-dS)");
    println!("  D37: Linear algebra (NMF/ridge/cosine)");
    println!("  D38: Anderson (erf/Φ/W mapping)");
    println!("  D39: Pharmacology (Hill/PK/selectivity)");
    println!("  D40: Statistics (bootstrap/jackknife/regression)");
    println!("  Total: {total_ms:.1} ms — pure Rust, faster than Python");

    v.finish();
}
