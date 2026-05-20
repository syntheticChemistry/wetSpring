// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! Exp337: Chen 2016 — Culture conditions response.
//! Lightweight validator for single-paper math.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Type | Analytical |
//! | Date | 2026-03-23 |
//! | Command | `cargo run --bin validate_anaerobic_culture_response` |
//!
//! Provenance: Anaerobic culture response curves (Track 6)

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn first_order(t: f64, b_max: f64, k: f64) -> f64 {
    b_max * (1.0 - (-k * t).exp())
}

fn main() {
    let mut v = Validator::new("Exp337: Chen 2016 — Anaerobic Culture Conditions Response");

    v.section("First-order kinetics (B_max=320, k=0.08)");
    let b_max = 320.0;
    let k = 0.08;

    v.check(
        "First-order B(0) = 0",
        first_order(0.0, b_max, k),
        0.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "First-order B(∞) → B_max",
        first_order(200.0, b_max, k),
        b_max,
        0.01,
    );
    v.check_pass(
        "First-order monotonic (B(10) < B(20) < B(30))",
        first_order(10.0, b_max, k) < first_order(20.0, b_max, k)
            && first_order(20.0, b_max, k) < first_order(30.0, b_max, k),
    );

    let t_half = (2.0_f64).ln() / k;
    v.check(
        "First-order B(t_half) = B_max/2",
        first_order(t_half, b_max, k),
        b_max / 2.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    v.section("Thermophilic vs mesophilic at t=30");
    let k_thermo = 0.12;
    let k_meso = 0.06;
    let b_thermo_30 = first_order(30.0, b_max, k_thermo);
    let b_meso_30 = first_order(30.0, b_max, k_meso);
    v.check_pass(
        "Thermophilic faster than mesophilic at t=30",
        b_thermo_30 > b_meso_30,
    );

    v.section("Community diversity — mesophilic vs thermophilic");
    let meso_comm = [30.0, 25.0, 20.0, 15.0, 10.0];
    let thermo_comm = [50.0, 20.0, 15.0, 10.0, 5.0];
    let h_meso = diversity::shannon(&meso_comm);
    let h_thermo = diversity::shannon(&thermo_comm);
    v.check_pass("Mesophilic Shannon > 0", h_meso > 0.0);
    v.check_pass("Thermophilic Shannon > 0", h_thermo > 0.0);
    let s_meso = diversity::simpson(&meso_comm);
    let s_thermo = diversity::simpson(&thermo_comm);
    v.check_pass(
        "Simpson ∈ (0, 1) for both communities",
        s_meso > 0.0 && s_meso < 1.0 && s_thermo > 0.0 && s_thermo < 1.0,
    );

    let j_meso = diversity::pielou_evenness(&meso_comm);
    let j_thermo = diversity::pielou_evenness(&thermo_comm);
    v.check_pass("Mesophilic more even than thermophilic", j_meso > j_thermo);

    let bc_mt = diversity::bray_curtis(&meso_comm, &thermo_comm);
    v.check_pass("BC(meso, thermo) > 0", bc_mt > 0.0);

    v.section("Anderson W comparison");
    let w_max = 20.0;
    let w_meso = w_max * (1.0 - j_meso);
    let w_thermo = w_max * (1.0 - j_thermo);
    v.check_pass("W_meso ∈ [0, W_max]", w_meso >= 0.0 && w_meso <= w_max);
    v.check_pass(
        "W_thermo ∈ [0, W_max]",
        w_thermo >= 0.0 && w_thermo <= w_max,
    );
    v.check_pass(
        "W_thermophilic > W_mesophilic (thermo more disordered)",
        w_thermo > w_meso,
    );

    let chao1_meso = diversity::chao1(&meso_comm);
    v.check_pass(
        "Chao1(meso) >= observed richness",
        chao1_meso >= meso_comm.len() as f64,
    );

    v.finish();
}
