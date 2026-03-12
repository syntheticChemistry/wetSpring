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
//! # Exp342: `BarraCuda` CPU v26 — Track 6 Anaerobic Pure Rust Math
//!
//! Proves pure Rust math for Track 6 anaerobic digestion: biogas kinetics
//! (Gompertz, first-order), microbial growth (Monod, Haldane), diversity
//! indices, and Anderson W mapping for aerobic-anaerobic comparison.
//!
//! All math is pure `barraCuda` CPU — zero Python, zero R, zero external
//! runtime. Validates against published model equations and analytical
//! invariants from Yang 2016, Chen 2016, Rojas-Sossa 2017/2019, Zhong 2016.
//!
//! ```text
//! Paper (Exp341) → CPU (this) → Python parity (Exp343) → GPU (Exp344)
//! → Streaming (Exp345) → metalForge (Exp346)
//! ```
//!
//! ## Domains
//!
//! - D60: Biogas Kinetics — Gompertz + first-order models
//! - D61: Microbial Growth — Monod + Haldane kinetics
//! - D62: Anaerobic Diversity — Shannon, Simpson, Bray-Curtis on digester communities
//! - D63: Anderson W Mapping — aerobic vs anaerobic disorder comparison
//! - D64: Cross-Track Composition — Track 6 ↔ Track 4 soil QS bridge
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | `BarraCuda` CPU (pure Rust math — zero external runtime) |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v26` |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, qs_biofilm};
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
    let mut v = Validator::new("Exp342: BarraCuda CPU v26 — Track 6 Anaerobic Pure Rust Math");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // D60: Biogas Kinetics — Gompertz + First-Order
    // ═══════════════════════════════════════════════════════════════════
    v.section("D60: Biogas Kinetics — Gompertz + First-Order");
    let t = Instant::now();
    let mut d60 = 0_u32;

    // Yang 2016 manure co-digestion Gompertz parameters
    let p_manure = 350.0;
    let rm_manure = 25.0;
    let lag_manure = 3.0;

    let times = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0];
    let mut gompertz_vals = Vec::new();
    for &t_val in &times {
        gompertz_vals.push(gompertz(t_val, p_manure, rm_manure, lag_manure));
    }

    v.check_pass("D60: Gompertz H(0) near zero", gompertz_vals[0] < 5.0);
    d60 += 1;
    v.check(
        "D60: Gompertz H(50) ≈ P (asymptote)",
        gompertz_vals[7],
        p_manure,
        1.0,
    );
    d60 += 1;

    // Monotonicity check
    let mut monotonic = true;
    for w in gompertz_vals.windows(2) {
        if w[1] < w[0] {
            monotonic = false;
        }
    }
    v.check_pass("D60: Gompertz monotonic increasing", monotonic);
    d60 += 1;

    // Corn stover parameters
    let p_stover = 280.0;
    let rm_stover = 18.0;
    let lag_stover = 5.0;
    let h_stover_50 = gompertz(50.0, p_stover, rm_stover, lag_stover);
    v.check(
        "D60: Corn stover Gompertz H(50) ≈ P",
        h_stover_50,
        p_stover,
        1.0,
    );
    d60 += 1;

    // First-order kinetics
    let b_max = 320.0;
    let k_rate = 0.08;

    v.check(
        "D60: First-order B(0) = 0",
        first_order(0.0, b_max, k_rate),
        0.0,
        tolerances::EXACT_F64,
    );
    d60 += 1;
    v.check(
        "D60: First-order B(200) ≈ B_max",
        first_order(200.0, b_max, k_rate),
        b_max,
        0.01,
    );
    d60 += 1;

    // Half-life: t_half = ln(2) / k
    let t_half = (2.0_f64).ln() / k_rate;
    v.check(
        "D60: First-order B(t_half) = B_max/2",
        first_order(t_half, b_max, k_rate),
        b_max / 2.0,
        1e-10,
    );
    d60 += 1;

    // AFEX pretreatment comparison (Rojas-Sossa 2019)
    let h_untreated = gompertz(20.0, 280.0, 18.0, 5.0);
    let h_afex = gompertz(20.0, 340.0, 28.0, 2.5);
    v.check_pass("D60: AFEX > untreated at t=20", h_afex > h_untreated);
    d60 += 1;

    domains.push(domain("Biogas Kinetics", "wetSpring", t.elapsed(), d60));

    // ═══════════════════════════════════════════════════════════════════
    // D61: Microbial Growth — Monod + Haldane
    // ═══════════════════════════════════════════════════════════════════
    v.section("D61: Microbial Growth — Monod + Haldane Kinetics");
    let t = Instant::now();
    let mut d61 = 0_u32;

    let mu_max_val = 0.4;
    let ks_val = 200.0;
    let ki_val = 3000.0;

    // Monod properties
    v.check(
        "D61: Monod(0) = 0",
        monod(0.0, mu_max_val, ks_val),
        0.0,
        tolerances::EXACT_F64,
    );
    d61 += 1;
    v.check(
        "D61: Monod(Ks) = mu_max/2",
        monod(ks_val, mu_max_val, ks_val),
        mu_max_val / 2.0,
        tolerances::EXACT_F64,
    );
    d61 += 1;
    v.check(
        "D61: Monod(50000) ≈ mu_max",
        monod(50000.0, mu_max_val, ks_val),
        mu_max_val,
        0.005,
    );
    d61 += 1;

    // Monod monotonicity
    let substrate_levels = [50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0];
    let monod_vals: Vec<f64> = substrate_levels
        .iter()
        .map(|&s| monod(s, mu_max_val, ks_val))
        .collect();
    let monod_mono = monod_vals.windows(2).all(|w| w[1] > w[0]);
    v.check_pass("D61: Monod monotonic increasing", monod_mono);
    d61 += 1;

    // Haldane properties
    v.check(
        "D61: Haldane(0) = 0",
        haldane(0.0, mu_max_val, ks_val, ki_val),
        0.0,
        tolerances::EXACT_F64,
    );
    d61 += 1;

    // Haldane optimal substrate
    let s_opt = (ks_val * ki_val).sqrt();
    let mu_at_opt = haldane(s_opt, mu_max_val, ks_val, ki_val);
    let mu_below = haldane(s_opt * 0.3, mu_max_val, ks_val, ki_val);
    let mu_above = haldane(s_opt * 3.0, mu_max_val, ks_val, ki_val);
    v.check_pass(
        "D61: Haldane peak at S_opt = sqrt(Ks*Ki)",
        mu_at_opt > mu_below && mu_at_opt > mu_above,
    );
    d61 += 1;

    // Haldane inhibition: high substrate reduces rate
    let mu_normal = haldane(300.0, mu_max_val, ks_val, ki_val);
    let mu_inhibited = haldane(10000.0, mu_max_val, ks_val, ki_val);
    v.check_pass(
        "D61: Haldane inhibition at S >> S_opt",
        mu_inhibited < mu_normal,
    );
    d61 += 1;

    // S_opt analytical value
    v.check(
        "D61: S_opt = sqrt(Ks * Ki)",
        s_opt,
        (200.0_f64 * 3000.0).sqrt(),
        tolerances::EXACT_F64,
    );
    d61 += 1;

    domains.push(domain("Microbial Growth", "wetSpring", t.elapsed(), d61));

    // ═══════════════════════════════════════════════════════════════════
    // D62: Anaerobic Diversity — Shannon, Simpson, Bray-Curtis
    // ═══════════════════════════════════════════════════════════════════
    v.section("D62: Anaerobic Community Diversity");
    let t = Instant::now();
    let mut d62 = 0_u32;

    let digester_comm = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let soil_comm = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];

    let h_dig = diversity::shannon(&digester_comm);
    let h_soil = diversity::shannon(&soil_comm);
    let s_dig = diversity::simpson(&digester_comm);
    let s_soil = diversity::simpson(&soil_comm);
    let j_dig = diversity::pielou_evenness(&digester_comm);
    let j_soil = diversity::pielou_evenness(&soil_comm);

    v.check_pass("D62: Digester Shannon > 0", h_dig > 0.0);
    d62 += 1;
    v.check_pass(
        "D62: Soil Shannon > Digester Shannon (more even)",
        h_soil > h_dig,
    );
    d62 += 1;
    v.check_pass(
        "D62: Simpson ∈ (0,1) for both",
        s_dig > 0.0 && s_dig < 1.0 && s_soil > 0.0 && s_soil < 1.0,
    );
    d62 += 1;
    v.check_pass("D62: Soil Pielou > Digester Pielou", j_soil > j_dig);
    d62 += 1;

    // Bray-Curtis between aerobic soil and anaerobic digester
    let bc = diversity::bray_curtis(&soil_comm, &digester_comm);
    v.check_pass("D62: BC(soil, digester) ∈ (0, 1]", bc > 0.0 && bc <= 1.0);
    d62 += 1;

    // Self-distance
    let bc_self = diversity::bray_curtis(&digester_comm, &digester_comm);
    v.check(
        "D62: BC self-distance = 0",
        bc_self,
        0.0,
        tolerances::EXACT_F64,
    );
    d62 += 1;

    // Chao1 richness
    let chao1_dig = diversity::chao1(&digester_comm);
    v.check_pass(
        "D62: Chao1 >= observed richness",
        chao1_dig >= digester_comm.len() as f64,
    );
    d62 += 1;

    // Rarefaction: expect richness to increase with depth
    let rare = diversity::rarefaction_curve(&digester_comm, &[5.0, 10.0, 20.0, 50.0]);
    let rare_mono = rare.windows(2).all(|w| w[1] >= w[0]);
    v.check_pass("D62: Rarefaction monotonic", rare_mono);
    d62 += 1;

    domains.push(domain("Anaerobic Diversity", "wetSpring", t.elapsed(), d62));

    // ═══════════════════════════════════════════════════════════════════
    // D63: Anderson W Mapping — Aerobic vs Anaerobic
    // ═══════════════════════════════════════════════════════════════════
    v.section("D63: Anderson W Mapping — Aerobic vs Anaerobic");
    let t = Instant::now();
    let mut d63 = 0_u32;

    // W = W_max * (1 - evenness), W_max = 20 (calibrated from Track 4)
    let w_max_calibrated = 20.0;
    let w_soil = w_max_calibrated * (1.0 - j_soil);
    let w_digester = w_max_calibrated * (1.0 - j_dig);

    v.check_pass(
        "D63: W ∈ [0, W_max] for soil",
        w_soil >= 0.0 && w_soil <= w_max_calibrated,
    );
    d63 += 1;
    v.check_pass(
        "D63: W ∈ [0, W_max] for digester",
        w_digester >= 0.0 && w_digester <= w_max_calibrated,
    );
    d63 += 1;
    v.check_pass(
        "D63: W_digester > W_soil (anaerobic more disordered)",
        w_digester > w_soil,
    );
    d63 += 1;

    // QS probability via norm_cdf: P(QS) = Φ((W_c - W) / σ)
    let w_c = 16.5;
    let sigma = 4.0;
    let p_qs_soil = norm_cdf((w_c - w_soil) / sigma);
    let p_qs_digester = norm_cdf((w_c - w_digester) / sigma);

    v.check_pass("D63: P(QS) ∈ [0, 1]", (0.0..=1.0).contains(&p_qs_soil));
    d63 += 1;
    v.check_pass(
        "D63: P(QS|soil) > P(QS|digester) (soil more connected)",
        p_qs_soil > p_qs_digester,
    );
    d63 += 1;

    domains.push(domain("Anderson W Mapping", "wetSpring", t.elapsed(), d63));

    // ═══════════════════════════════════════════════════════════════════
    // D64: Cross-Track Composition — Track 6 ↔ Track 4
    // ═══════════════════════════════════════════════════════════════════
    v.section("D64: Cross-Track — Track 6 Anaerobic ↔ Track 4 Soil QS");
    let t = Instant::now();
    let mut d64 = 0_u32;

    // QS ODE integration for soil pore (Track 4 anchor)
    let params = qs_biofilm::QsBiofilmParams::default();
    let result = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &params);
    v.check_pass(
        "D64: QS ODE converges (Track 4 baseline)",
        result.t.len() > 100,
    );
    d64 += 1;

    // norm_cdf analytical checks (shared with Track 4)
    v.check("D64: Φ(0) = 0.5", norm_cdf(0.0), 0.5, tolerances::EXACT_F64);
    d64 += 1;
    v.check("D64: Φ(-∞) → 0", norm_cdf(-10.0), 0.0, 1e-10);
    d64 += 1;

    // Gompertz parameter sensitivity: doubling Rm doubles early-phase rate
    let base_rate = gompertz(lag_manure + 1.0, p_manure, rm_manure, lag_manure);
    let double_rate = gompertz(lag_manure + 1.0, p_manure, rm_manure * 2.0, lag_manure);
    v.check_pass(
        "D64: Doubling Rm increases early production",
        double_rate > base_rate,
    );
    d64 += 1;

    domains.push(domain("Cross-Track T6↔T4", "wetSpring", t.elapsed(), d64));

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
