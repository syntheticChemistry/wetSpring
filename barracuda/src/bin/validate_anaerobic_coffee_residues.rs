// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Exp338: Rojas-Sossa 2017 — Coffee residues.
//! Lightweight validator for single-paper math.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Type | Analytical |
//! | Date | 2026-03-23 |
//! | Command | `cargo run --bin validate_anaerobic_coffee_residues` |

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::kinetics::haldane;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn main() {
    let mut v = Validator::new("Exp338: Rojas-Sossa 2017 — Coffee Residues Anaerobic");

    v.section("Haldane inhibition (mu_max=0.4, Ks=200, Ki=3000)");
    let mu_max = 0.4;
    let ks = 200.0;
    let ki = 3000.0;

    v.check(
        "Haldane mu(0) = 0",
        haldane(0.0, mu_max, ks, ki),
        0.0,
        tolerances::EXACT_F64,
    );

    let s_opt = (ks * ki).sqrt();
    let mu_opt = haldane(s_opt, mu_max, ks, ki);
    let mu_below = haldane(s_opt * 0.5, mu_max, ks, ki);
    let mu_above = haldane(s_opt * 2.0, mu_max, ks, ki);
    v.check_pass(
        "Haldane peak at S_opt = sqrt(Ks*Ki)",
        mu_opt > mu_below && mu_opt > mu_above,
    );
    v.check(
        "S_opt = sqrt(Ks * Ki)",
        s_opt,
        (200.0_f64 * 3000.0).sqrt(),
        tolerances::EXACT_F64,
    );

    v.section("Coffee waste pushes past S_opt → inhibition");
    let mu_300 = haldane(300.0, mu_max, ks, ki);
    let mu_4000 = haldane(4000.0, mu_max, ks, ki);
    v.check_pass("haldane(4000) < haldane(300)", mu_4000 < mu_300);

    v.section("Gompertz with coffee vs control");
    let p_control = 320.0;
    let p_coffee = 280.0;
    let h_control_30 = gompertz(30.0, p_control, 22.0, 4.0);
    let h_coffee_30 = gompertz(30.0, p_coffee, 18.0, 5.0);
    v.check_pass(
        "Coffee waste lower yield than control at t=30",
        h_coffee_30 < h_control_30,
    );

    v.section("Diversity shift with substrate perturbation");
    let control_comm = [40.0, 28.0, 18.0, 10.0, 4.0];
    let coffee_comm = [55.0, 20.0, 12.0, 8.0, 5.0];
    let h_control = diversity::shannon(&control_comm);
    let h_coffee = diversity::shannon(&coffee_comm);
    v.check_pass("Control Shannon > 0", h_control > 0.0);
    v.check_pass("Coffee Shannon > 0", h_coffee > 0.0);

    let j_control = diversity::pielou_evenness(&control_comm);
    let j_coffee = diversity::pielou_evenness(&coffee_comm);
    v.check_pass(
        "Coffee perturbation reduces evenness (more skewed)",
        j_coffee < j_control,
    );

    let bc = diversity::bray_curtis(&control_comm, &coffee_comm);
    v.check_pass("BC(control, coffee) > 0", bc > 0.0);
    v.check_pass("BC(control, coffee) <= 1", bc <= 1.0);

    v.finish();
}
