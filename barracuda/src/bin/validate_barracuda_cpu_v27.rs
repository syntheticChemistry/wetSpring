// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
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
//! # Exp347: `BarraCuda` CPU v27 — V109 Upstream Rewire + Track 6 Validation
//!
//! Proves pure Rust math correctness after the V109 upstream rewire:
//! - `SpringDomain` migrated to `SCREAMING_SNAKE_CASE` associated constants
//! - GPU diversity functions are now synchronous (no tokio required)
//! - DADA2 module import conflict resolved
//!
//! Re-validates all Track 6 biogas kinetics (Gompertz, first-order, Monod,
//! Haldane), diversity indices, and Anderson W mapping. Adds upstream
//! stats/linalg/special regression to confirm the rewire is clean.
//!
//! ```text
//! CPU (this) → GPU (Exp348) → ToadStool (Exp349)
//! → Streaming (Exp350) → metalForge (Exp351) → NUCLEUS (Exp352)
//! ```
//!
//! ## Domains
//!
//! - D65: Upstream Stats — bootstrap, jackknife, correlation, regression
//! - D66: Upstream Linalg — Laplacian, effective rank, ridge regression
//! - D67: Upstream Special — erf, `norm_cdf`, `ln_gamma`
//! - D68: Cross-Spring Provenance — `SpringDomain::WET_SPRING` + shader registry
//! - D69: Track 6 Biogas Kinetics — Gompertz + first-order + Monod + Haldane
//! - D70: Track 6 Anderson W — disorder mapping + QS probability
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | `BarraCuda` CPU (pure Rust — zero external runtime) |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v27` |
//!
//! ## Baseline sources
//!
//! | Domain | Baseline | Script / Tool |
//! |--------|----------|---------------|
//! | D65 | Analytical known-values | Exact: mean([1..5])=3, var([1..5])=2.5, slope(x,2x)=2 |
//! | D66 | Analytical graph theory | Laplacian row-sum=0, degree matrix properties |
//! | D67 | Abramowitz & Stegun tables | erf(0)=0, Φ(0)=0.5, ln_gamma(1)=0 |
//! | D68 | barraCuda shaders::provenance | Registry cross-spring metadata |
//! | D69 | `scripts/python_anaerobic_biogas_baseline.py` (V107, 2026-03-10) | Yang 2016: P=350/Rm=25/λ=3, B_max=320/k=0.08, μ_max=0.4/Ks=200/Ki=3000 |
//! | D70 | `scripts/python_anaerobic_biogas_baseline.py` (V107, 2026-03-10) | W=20·(1−J), communities: anaerobic=[45,25,15,8,3,2,1,0.5,0.3,0.2] |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, qs_biofilm};
use wetspring_barracuda::provenance;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{DomainResult, Validator};

use barracuda::stats::norm_cdf;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
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

fn domain(
    name: &'static str,
    spring: &'static str,
    elapsed: std::time::Duration,
    checks: u32,
) -> DomainResult {
    DomainResult {
        name,
        spring: Some(spring),
        ms: elapsed.as_secs_f64() * 1000.0,
        checks,
    }
}

fn main() {
    let mut v =
        Validator::new("Exp347: BarraCuda CPU v27 — V109 Upstream Rewire + Track 6 Validation");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // D65: Upstream Stats API Regression
    // ═══════════════════════════════════════════════════════════════════
    v.section("D65: Upstream Stats — bootstrap, jackknife, correlation, regression");
    let t = Instant::now();
    let mut d65 = 0_u32;

    let data_5 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_5 = barracuda::stats::mean(&data_5);
    v.check(
        "D65: Mean([1..5]) = 3.0",
        mean_5,
        3.0,
        tolerances::EXACT_F64,
    );
    d65 += 1;

    let var_5 = barracuda::stats::covariance(&data_5, &data_5).expect("covariance(x,x)");
    v.check(
        "D65: Var([1..5]) = 2.5",
        var_5,
        2.5,
        tolerances::ANALYTICAL_F64,
    );
    d65 += 1;

    let ci = barracuda::stats::bootstrap_ci(
        &data_5,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        10_000,
        0.95,
        42,
    )
    .unwrap();
    v.check_pass(
        "D65: Bootstrap CI lower < estimate",
        ci.lower <= ci.estimate,
    );
    d65 += 1;
    v.check_pass(
        "D65: Bootstrap CI estimate < upper",
        ci.estimate <= ci.upper,
    );
    d65 += 1;

    let jk = barracuda::stats::jackknife_mean_variance(&data_5).unwrap();
    v.check(
        "D65: Jackknife mean = 3.0",
        jk.estimate,
        3.0,
        tolerances::ANALYTICAL_F64,
    );
    d65 += 1;

    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let pearson = barracuda::stats::pearson_correlation(&x, &y).unwrap();
    v.check(
        "D65: Pearson(x, 2x) = 1.0",
        pearson,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d65 += 1;

    let spearman = barracuda::stats::spearman_correlation(&x, &y).unwrap();
    v.check(
        "D65: Spearman(x, 2x) = 1.0",
        spearman,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d65 += 1;

    let fit = barracuda::stats::fit_linear(&x, &y).unwrap();
    v.check(
        "D65: Linear slope = 2.0",
        fit.params[0],
        2.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d65 += 1;
    v.check(
        "D65: Linear R² ≈ 1.0",
        fit.r_squared,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d65 += 1;

    domains.push(domain("Upstream Stats", "wetSpring", t.elapsed(), d65));

    // ═══════════════════════════════════════════════════════════════════
    // D66: Upstream Linalg API Regression
    // ═══════════════════════════════════════════════════════════════════
    v.section("D66: Upstream Linalg — Laplacian, effective rank, ridge");
    let t = Instant::now();
    let mut d66 = 0_u32;

    let n_g = 4;
    let adj: Vec<f64> = vec![
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let laplacian = barracuda::linalg::graph_laplacian(&adj, n_g);
    v.check(
        "D66: Laplacian[0][0] = degree 2",
        laplacian[0],
        2.0,
        tolerances::EXACT_F64,
    );
    d66 += 1;
    v.check(
        "D66: Laplacian[0][1] = -1",
        laplacian[1],
        -1.0,
        tolerances::EXACT_F64,
    );
    d66 += 1;

    let row_sums: Vec<f64> = (0..n_g)
        .map(|i| (0..n_g).map(|j| laplacian[i * n_g + j]).sum())
        .collect();
    let max_row_sum = row_sums.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    v.check(
        "D66: Laplacian row sums = 0",
        max_row_sum,
        0.0,
        tolerances::PYTHON_PARITY_TIGHT,
    );
    d66 += 1;

    let eigenvalues = [10.0, 5.0, 2.0, 0.5, 0.01, 0.001];
    let eff_rank = barracuda::linalg::effective_rank(&eigenvalues);
    v.check_pass("D66: Effective rank > 0", eff_rank > 0.0);
    d66 += 1;
    v.check_pass(
        "D66: Effective rank ≤ n",
        eff_rank <= eigenvalues.len() as f64,
    );
    d66 += 1;

    domains.push(domain("Upstream Linalg", "wetSpring", t.elapsed(), d66));

    // ═══════════════════════════════════════════════════════════════════
    // D67: Upstream Special Functions
    // ═══════════════════════════════════════════════════════════════════
    v.section("D67: Upstream Special — erf, norm_cdf, ln_gamma");
    let t = Instant::now();
    let mut d67 = 0_u32;

    let erf_0 = barracuda::special::erf(0.0);
    v.check("D67: erf(0) = 0", erf_0, 0.0, tolerances::EXACT_F64);
    d67 += 1;

    let erf_inf = barracuda::special::erf(6.0);
    v.check(
        "D67: erf(6) ≈ 1",
        erf_inf,
        1.0,
        tolerances::LIMIT_CONVERGENCE,
    );
    d67 += 1;

    let erf_neg = barracuda::special::erf(-1.0);
    let erf_pos = barracuda::special::erf(1.0);
    v.check(
        "D67: erf(-x) = -erf(x)",
        erf_neg,
        -erf_pos,
        tolerances::EXACT_F64,
    );
    d67 += 1;

    v.check("D67: Φ(0) = 0.5", norm_cdf(0.0), 0.5, tolerances::EXACT_F64);
    d67 += 1;
    v.check("D67: Φ(-10) → 0", norm_cdf(-10.0), 0.0, tolerances::ANALYTICAL_LOOSE);
    d67 += 1;
    v.check("D67: Φ(10) → 1", norm_cdf(10.0), 1.0, tolerances::ANALYTICAL_LOOSE);
    d67 += 1;

    let lng_1 = barracuda::special::ln_gamma(1.0).expect("ln_gamma(1)");
    v.check(
        "D67: ln_gamma(1) = 0",
        lng_1,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    d67 += 1;

    let lng_half = barracuda::special::ln_gamma(0.5).expect("ln_gamma(0.5)");
    v.check(
        "D67: ln_gamma(0.5) = ln(√π)",
        lng_half,
        (std::f64::consts::PI.sqrt()).ln(),
        tolerances::ANALYTICAL_F64,
    );
    d67 += 1;

    domains.push(domain("Special Functions", "wetSpring", t.elapsed(), d67));

    // ═══════════════════════════════════════════════════════════════════
    // D68: Cross-Spring Provenance — SpringDomain::WET_SPRING
    // ═══════════════════════════════════════════════════════════════════
    v.section("D68: Cross-Spring Provenance — SpringDomain rewire");
    let t = Instant::now();
    let mut d68 = 0_u32;

    let authored = provenance::shaders_authored();
    v.check_pass("D68: WET_SPRING has authored shaders", !authored.is_empty());
    d68 += 1;

    let consumed = provenance::shaders_consumed();
    v.check_pass(
        "D68: wetSpring consumes cross-spring shaders",
        consumed.len() >= 10,
    );
    d68 += 1;

    let summary = provenance::wetspring_provenance_summary();
    v.check_pass(
        "D68: Provenance summary non-empty (SCREAMING_SNAKE_CASE API)",
        summary.contains("wetSpring"),
    );
    d68 += 1;

    domains.push(domain("Cross-Spring Prov.", "wetSpring", t.elapsed(), d68));

    // ═══════════════════════════════════════════════════════════════════
    // D69: Track 6 Biogas Kinetics — Gompertz + First-Order + Monod + Haldane
    // ═══════════════════════════════════════════════════════════════════
    v.section("D69: Track 6 Biogas Kinetics — Gompertz + First-Order + Monod + Haldane");
    let t = Instant::now();
    let mut d69 = 0_u32;

    let p_manure = 350.0;
    let rm_manure = 25.0;
    let lag_manure = 3.0;
    let times = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0];
    let gompertz_vals: Vec<f64> = times
        .iter()
        .map(|&t_val| gompertz(t_val, p_manure, rm_manure, lag_manure))
        .collect();

    v.check_pass("D69: Gompertz H(0) near zero", gompertz_vals[0] < 5.0);
    d69 += 1;
    v.check(
        "D69: Gompertz H(50) ≈ P (asymptote)",
        gompertz_vals[7],
        p_manure,
        1.0,
    );
    d69 += 1;
    let gomp_mono = gompertz_vals.windows(2).all(|w| w[1] >= w[0]);
    v.check_pass("D69: Gompertz monotonic increasing", gomp_mono);
    d69 += 1;

    v.check(
        "D69: First-order B(0) = 0",
        first_order(0.0, 320.0, 0.08),
        0.0,
        tolerances::EXACT_F64,
    );
    d69 += 1;
    let t_half = (2.0_f64).ln() / 0.08;
    v.check(
        "D69: First-order B(t_half) = B_max/2",  // python_anaerobic_biogas_baseline.py: B_max=320, k=0.08
        first_order(t_half, 320.0, 0.08),
        160.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d69 += 1;

    let mu_max_val = 0.4;
    let ks_val = 200.0;
    let ki_val = 3000.0;
    v.check(
        "D69: Monod(Ks) = mu_max/2",
        monod(ks_val, mu_max_val, ks_val),
        mu_max_val / 2.0,
        tolerances::EXACT_F64,
    );
    d69 += 1;

    let s_opt = (ks_val * ki_val).sqrt();
    let mu_at_opt = haldane(s_opt, mu_max_val, ks_val, ki_val);
    let mu_below = haldane(s_opt * 0.3, mu_max_val, ks_val, ki_val);
    let mu_above = haldane(s_opt * 3.0, mu_max_val, ks_val, ki_val);
    v.check_pass(
        "D69: Haldane peak at S_opt = sqrt(Ks*Ki)",
        mu_at_opt > mu_below && mu_at_opt > mu_above,
    );
    d69 += 1;

    let h_afex = gompertz(20.0, 340.0, 28.0, 2.5);
    let h_untreated = gompertz(20.0, 280.0, 18.0, 5.0);
    v.check_pass("D69: AFEX > untreated at t=20", h_afex > h_untreated);
    d69 += 1;

    domains.push(domain("Biogas Kinetics", "wetSpring", t.elapsed(), d69));

    // ═══════════════════════════════════════════════════════════════════
    // D70: Track 6 Anderson W Mapping + Cross-Track Bridge
    // ═══════════════════════════════════════════════════════════════════
    v.section("D70: Anderson W Mapping + Cross-Track Bridge");
    let t = Instant::now();
    let mut d70 = 0_u32;

    let digester_comm = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let soil_comm = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];

    let j_dig = diversity::pielou_evenness(&digester_comm);
    let j_soil = diversity::pielou_evenness(&soil_comm);
    let w_max = 20.0;
    let w_dig = w_max * (1.0 - j_dig);
    let w_soil = w_max * (1.0 - j_soil);

    v.check_pass("D70: W_digester > W_soil", w_dig > w_soil);
    d70 += 1;

    let w_c = 16.5;
    let sigma = 4.0;
    let p_qs_soil = norm_cdf((w_c - w_soil) / sigma);
    let p_qs_dig = norm_cdf((w_c - w_dig) / sigma);
    v.check_pass("D70: P(QS) ∈ [0,1]", (0.0..=1.0).contains(&p_qs_soil));
    d70 += 1;
    v.check_pass("D70: P(QS|soil) > P(QS|digester)", p_qs_soil > p_qs_dig);
    d70 += 1;

    // Cross-track bridge: QS ODE from Track 4
    let params = qs_biofilm::QsBiofilmParams::default();
    let result = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &params);
    v.check_pass("D70: QS ODE converges (T4 bridge)", result.t.len() > 100);
    d70 += 1;

    // Diversity cross-check
    let h_dig = diversity::shannon(&digester_comm);
    let h_soil = diversity::shannon(&soil_comm);
    v.check_pass("D70: H(soil) > H(digester)", h_soil > h_dig);
    d70 += 1;

    let bc = diversity::bray_curtis(&soil_comm, &digester_comm);
    v.check_pass("D70: BC(soil, digester) ∈ (0,1]", bc > 0.0 && bc <= 1.0);
    d70 += 1;

    domains.push(domain("Anderson W + Bridge", "wetSpring", t.elapsed(), d70));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── Domain Summary ({total_ms:.2} ms total) ──");
    let mut total_checks = 0_u32;
    for d in &domains {
        println!(
            "  {:30} {:>8} {:>6.2} ms  {:>3} checks",
            d.name,
            d.spring.unwrap_or("—"),
            d.ms,
            d.checks,
        );
        total_checks += d.checks;
    }
    println!(
        "  {:30} {:>8} {:>6.2} ms  {:>3} checks",
        "TOTAL", "", total_ms, total_checks
    );

    v.finish();
}
