// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    reason = "validation harness: results printed to stdout, f64 timing arithmetic"
)]
//! # Exp377: Hormesis Biphasic Dose-Response Model
//!
//! Validates the biphasic dose-response model (`bio::hormesis`) against known
//! hormesis curves from Calabrese & Mattson (2017). Establishes that the
//! Hill-based stimulation × inhibition model correctly produces the canonical
//! J-shaped hormetic curve and that the Anderson disorder mapping connects
//! the hormetic zone to the near-critical regime (W ≈ W_c).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (pure Rust math) |
//! | Date | 2026-05-17 |
//! | Command | `cargo run --release --bin validate_hormesis_biphasic` |
//! | Chain | Anderson QS (Exp107–156) → Gonzales IC50 (Exp280) → **This** → Exp378 → Exp379 |
//!
//! Provenance: Hormesis biphasic dose-response validation (Exp377)

use std::time::Instant;
use wetspring_barracuda::bio::hormesis::{self, DoseRegime, HormesisParams};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp377: Hormesis Biphasic Dose-Response Model");
    let t0 = Instant::now();

    let params = HormesisParams::new(0.3, 1.0, 2.0, 100.0, 3.0)
        .expect("default hormesis params are valid");

    let doses: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();

    // §1  R(dose=0) = 1.0  (baseline)
    v.section("D01: Baseline response");
    let r0 = hormesis::response(0.0, &params);
    v.check("R(dose=0) = 1.0", r0, 1.0, tolerances::ANALYTICAL_F64);

    // §2  R(dose→∞) → 0.0  (full inhibition at high dose)
    v.section("D02: Asymptotic inhibition");
    let r_high = hormesis::response(1e6, &params);
    v.check("R(dose=1e6) < 0.01", r_high, 0.0, tolerances::ASYMPTOTIC_LIMIT);

    // §3  Peak response > 1.0  (hormesis definition)
    v.section("D03: Hormetic peak");
    let peak = hormesis::find_peak(&doses, &params);
    let (peak_dose, peak_response) = peak.expect("hormetic peak must exist");
    v.check_pass("peak response > 1.0", peak_response > 1.0);
    v.check_pass(
        "peak dose between K_stim and K_inh",
        peak_dose > params.k_stim && peak_dose < params.k_inh,
    );

    // §4  Hormetic zone width > 0
    v.section("D04: Hormetic zone");
    let zone = hormesis::hormetic_zone(&doses, &params);
    let (zone_low, zone_high, zone_peak) = zone.expect("hormetic zone must exist");
    v.check_pass("zone width > 0", zone_high > zone_low);
    v.check_pass("zone peak > 1.0", zone_peak > 1.0);

    // §5  Dose regime classification
    v.section("D05: Regime classification");
    let pt_zero = hormesis::evaluate(0.0, &params);
    v.check_pass("dose=0 → Subthreshold", pt_zero.regime == DoseRegime::Subthreshold);
    let pt_toxic = hormesis::evaluate(500.0, &params);
    v.check_pass("dose=500 → Toxic", pt_toxic.regime == DoseRegime::Toxic);

    // §6  Sweep consistency (all points evaluated)
    v.section("D06: Sweep consistency");
    let sweep = hormesis::sweep(&doses, &params);
    v.check_count("sweep produces 1000 points", sweep.len(), 1000);
    v.check_pass(
        "sweep first point = baseline",
        (sweep[0].response - 1.0).abs() < tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "sweep contains hormetic regime",
        sweep.iter().any(|p| p.regime == DoseRegime::Hormetic),
    );
    v.check_pass(
        "sweep contains toxic regime",
        sweep.iter().any(|p| p.regime == DoseRegime::Toxic),
    );

    // §7  Anderson disorder mapping
    v.section("D07: Anderson disorder mapping");
    let w_baseline = 5.0;
    let sensitivity = 0.5;
    let gamma = 1.0;
    let w_c = 16.5;

    let w_at_peak = hormesis::dose_to_disorder(peak_dose, w_baseline, sensitivity, gamma);
    v.check_pass("W at peak > W baseline", w_at_peak > w_baseline);
    v.check_pass(
        "W at peak in near-critical regime (W_baseline < W_peak < 2×W_c)",
        w_at_peak > w_baseline && w_at_peak < 2.0 * w_c,
    );
    v.check_pass(
        "predicted zone exists",
        hormesis::predict_hormetic_zone_from_wc(w_baseline, w_c, sensitivity, gamma, 0.1).is_some(),
    );

    // §8  Disorder sweep with spectral analysis
    v.section("D08: Disorder sweep");
    let disorder_doses: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
    let hp = HormesisParams::new(0.3, 1.0, 2.0, 100.0, 3.0).expect("valid params");
    let disorder_sweep =
        hormesis::sweep_with_disorder(&disorder_doses, &hp, w_baseline, sensitivity, gamma);
    v.check_count("disorder sweep = 50 points", disorder_sweep.len(), 50);
    v.check("first disorder = W_baseline", disorder_sweep[0].1, w_baseline, tolerances::ANALYTICAL_F64);

    println!("\nTotal wall time: {:.2?}", t0.elapsed());
    v.finish();
}
