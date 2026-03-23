// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! Exp336: Yang 2016 — Anaerobic co-digestion phylogenetics.
//! Lightweight validator for single-paper math.

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn gompertz(t: f64, p: f64, rm: f64, lambda: f64) -> f64 {
    p * (-(rm * std::f64::consts::E / p)
        .mul_add(lambda - t, 1.0)
        .exp())
    .exp()
}

fn main() {
    let mut v = Validator::new("Exp336: Yang 2016 — Anaerobic Co-Digestion Phylogenetics");

    v.section("Modified Gompertz (P=350, Rm=25, λ=3)");
    let p = 350.0;
    let rm = 25.0;
    let lambda = 3.0;

    let h0 = gompertz(0.0, p, rm, lambda);
    v.check_pass("Gompertz H(0) ≈ 0", h0 < 5.0);

    let h_inf = gompertz(200.0, p, rm, lambda);
    v.check(
        "Gompertz H(∞) → P",
        h_inf,
        p,
        tolerances::BIOGAS_KINETICS_ASYMPTOTIC,
    );

    let h5 = gompertz(5.0, p, rm, lambda);
    let h10 = gompertz(10.0, p, rm, lambda);
    let h20 = gompertz(20.0, p, rm, lambda);
    v.check_pass(
        "Gompertz monotonic (H(5) < H(10) < H(20))",
        h5 < h10 && h10 < h20,
    );

    v.section("Shannon diversity — anaerobic community");
    let digester = [45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let h_dig = diversity::shannon(&digester);
    v.check_pass("Digester Shannon > 0", h_dig > 0.0);
    let s_dig = diversity::simpson(&digester);
    v.check_pass("Digester Simpson ∈ (0, 1)", s_dig > 0.0 && s_dig < 1.0);

    v.section("Bray-Curtis — digester vs soil");
    let soil = [35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];
    let bc = diversity::bray_curtis(&digester, &soil);
    v.check_pass("BC(digester, soil) ∈ (0, 1]", bc > 0.0 && bc <= 1.0);

    let bc_self = diversity::bray_curtis(&digester, &digester);
    v.check("BC self-distance = 0", bc_self, 0.0, tolerances::EXACT_F64);

    v.section("Anderson W mapping: W = 20*(1-evenness)");
    let j_dig = diversity::pielou_evenness(&digester);
    let j_soil = diversity::pielou_evenness(&soil);
    let w_max = 20.0;
    let w_dig = w_max * (1.0 - j_dig);
    let w_soil = w_max * (1.0 - j_soil);
    v.check_pass("W_digester ∈ [0, W_max]", w_dig >= 0.0 && w_dig <= w_max);
    v.check_pass("W_soil ∈ [0, W_max]", w_soil >= 0.0 && w_soil <= w_max);
    v.check_pass(
        "W_digester > W_soil (anaerobic more disordered)",
        w_dig > w_soil,
    );

    v.section("Chao1 and rarefaction");
    let chao1_dig = diversity::chao1(&digester);
    v.check_pass(
        "Chao1 >= observed richness",
        chao1_dig >= digester.len() as f64,
    );
    let rare = diversity::rarefaction_curve(&digester, &[5.0, 10.0, 20.0, 50.0]);
    let rare_mono = rare.windows(2).all(|w| w[1] >= w[0]);
    v.check_pass("Rarefaction monotonic", rare_mono);

    v.finish();
}
