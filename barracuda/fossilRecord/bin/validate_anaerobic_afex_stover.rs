// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! Exp339: Rojas-Sossa 2019 — AFEX corn stover.
//! Lightweight validator for single-paper math.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Type | Analytical |
//! | Date | 2026-03-23 |
//! | Command | `cargo run --bin validate_anaerobic_afex_stover` |
//!
//! Provenance: AFEX corn stover anaerobic digestion kinetics (Track 6)

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;

use wetspring_barracuda::validation::Validator;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn main() {
    let mut v = Validator::new("Exp339: Rojas-Sossa 2019 — AFEX Corn Stover");

    v.section("Untreated vs AFEX-treated Gompertz");
    let p_untreated = 280.0;
    let rm_untreated = 18.0;
    let lag_untreated = 5.0;
    let p_afex = 340.0;
    let rm_afex = 28.0;
    let lag_afex = 2.5;

    v.check_pass("AFEX higher P (methane potential)", p_afex > p_untreated);
    v.check_pass("AFEX higher Rm (max rate)", rm_afex > rm_untreated);
    v.check_pass("AFEX lower λ (shorter lag)", lag_afex < lag_untreated);

    let h_untreated_20 = gompertz(20.0, p_untreated, rm_untreated, lag_untreated);
    let h_afex_20 = gompertz(20.0, p_afex, rm_afex, lag_afex);
    v.check_pass("AFEX more yield at t=20", h_afex_20 > h_untreated_20);

    let h_untreated_50 = gompertz(50.0, p_untreated, rm_untreated, lag_untreated);
    v.check(
        "Untreated H(50) ≈ P",
        h_untreated_50,
        p_untreated,
        tolerances::BIOGAS_KINETICS_ASYMPTOTIC,
    );

    v.section("First-order comparison");
    let b_max = 320.0;
    let k_untreated = 0.06;
    let k_afex = 0.10;
    let b_untreated_30 = first_order(30.0, b_max, k_untreated);
    let b_afex_30 = first_order(30.0, b_max, k_afex);
    v.check_pass(
        "AFEX first-order faster at t=30",
        b_afex_30 > b_untreated_30,
    );

    v.section("Anderson W: AFEX-treated community more ordered");
    let untreated_comm = [45.0, 25.0, 15.0, 8.0, 5.0, 2.0];
    let afex_comm = [35.0, 28.0, 22.0, 10.0, 3.0, 2.0];
    let j_untreated = diversity::pielou_evenness(&untreated_comm);
    let j_afex = diversity::pielou_evenness(&afex_comm);
    let w_max = 20.0;
    let w_untreated = w_max * (1.0 - j_untreated);
    let w_afex = w_max * (1.0 - j_afex);
    v.check_pass(
        "AFEX-treated more ordered (lower W than untreated)",
        w_afex < w_untreated,
    );

    let h_untreated = diversity::shannon(&untreated_comm);
    let h_afex = diversity::shannon(&afex_comm);
    v.check_pass("Untreated Shannon > 0", h_untreated > 0.0);
    v.check_pass("AFEX Shannon > 0", h_afex > 0.0);

    let bc = diversity::bray_curtis(&untreated_comm, &afex_comm);
    v.check_pass("BC(untreated, AFEX) ∈ (0, 1]", bc > 0.0 && bc <= 1.0);

    let chao1_afex = diversity::chao1(&afex_comm);
    v.check_pass(
        "Chao1(AFEX) >= observed richness",
        chao1_afex >= afex_comm.len() as f64,
    );

    v.finish();
}
