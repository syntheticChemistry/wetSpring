// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Exp340: Zhong 2016 — Fungal fermentation on digestate.
//! Lightweight validator for single-paper math.

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::stats::norm_cdf;

fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}

fn main() {
    let mut v = Validator::new("Exp340: Zhong 2016 — Fungal Fermentation on Digestate");

    v.section("Monod kinetics (mu_max=0.35, Ks=150)");
    let mu_max = 0.35;
    let ks = 150.0;

    v.check(
        "Monod mu(0) = 0",
        monod(0.0, mu_max, ks),
        0.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Monod mu(Ks) = mu_max/2",
        monod(ks, mu_max, ks),
        mu_max / 2.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "Monod mu(∞) → mu_max",
        monod(50000.0, mu_max, ks),
        mu_max,
        0.01,
    );
    let monod_vals = [50.0, 100.0, 200.0, 500.0];
    let mono = monod_vals
        .windows(2)
        .all(|w| monod(w[1], mu_max, ks) > monod(w[0], mu_max, ks));
    v.check_pass("Monod monotonic increasing", mono);

    v.section("Aerobic-anaerobic W transition");
    let aerobic_comm = [35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];
    let anaerobic_comm = [45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let j_aerobic = diversity::pielou_evenness(&aerobic_comm);
    let j_anaerobic = diversity::pielou_evenness(&anaerobic_comm);
    let w_max = 20.0;
    let w_aerobic = w_max * (1.0 - j_aerobic);
    let w_anaerobic = w_max * (1.0 - j_anaerobic);
    v.check_pass(
        "Aerobic community more ordered (lower W)",
        w_aerobic < w_anaerobic,
    );

    let h_aerobic = diversity::shannon(&aerobic_comm);
    v.check_pass("Aerobic Shannon > 0", h_aerobic > 0.0);

    v.section("P(QS) comparison via norm_cdf");
    let w_c = 16.5;
    let sigma = 4.0;
    let p_qs_aerobic = norm_cdf((w_c - w_aerobic) / sigma);
    let p_qs_anaerobic = norm_cdf((w_c - w_anaerobic) / sigma);
    v.check_pass(
        "P(QS|aerobic) > P(QS|anaerobic) (aerobic more connected)",
        p_qs_aerobic > p_qs_anaerobic,
    );
    v.check_pass(
        "P(QS) ∈ [0, 1] for both",
        (0.0..=1.0).contains(&p_qs_aerobic) && (0.0..=1.0).contains(&p_qs_anaerobic),
    );

    v.check(
        "norm_cdf(0) = 0.5",
        norm_cdf(0.0),
        0.5,
        tolerances::EXACT_F64,
    );

    let bc = diversity::bray_curtis(&aerobic_comm, &anaerobic_comm);
    v.check_pass("BC(aerobic, anaerobic) > 0", bc > 0.0);

    v.finish();
}
