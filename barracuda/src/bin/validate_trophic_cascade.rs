// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    clippy::cast_precision_loss,
    reason = "validation harness: results printed to stdout, f64 timing arithmetic"
)]
//! # Exp378: Trophic Cascade via Anderson Lattice
//!
//! Models a multi-species trophic network as an Anderson lattice where
//! pesticide dose differentially perturbs species' effective disorder.
//! Tests the prediction that predators (sensitive) localize (collapse)
//! before prey (resistant), producing a trophic cascade.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (pure Rust math) |
//! | Date | 2026-05-17 |
//! | Command | `cargo run --release --bin validate_trophic_cascade` |
//! | Chain | Hormesis Model (Exp377) → **This** → Joint Colonization (Exp379) |
//!
//! Provenance: Trophic cascade Anderson lattice validation (Exp378)

use std::time::Instant;
use wetspring_barracuda::bio::anderson_spectral;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::hormesis::{self, HormesisParams};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp378: Trophic Cascade via Anderson Lattice");
    let t0 = Instant::now();

    // §1  Predator-prey sensitivity parameters
    v.section("D01: Differential sensitivity setup");

    let predator_params =
        HormesisParams::new(0.2, 0.5, 2.0, 30.0, 3.0).expect("predator params valid");
    let prey_params =
        HormesisParams::new(0.3, 2.0, 2.0, 200.0, 3.0).expect("prey params valid");

    v.check_pass(
        "predator K_inh < prey K_inh (more sensitive)",
        predator_params.k_inh < prey_params.k_inh,
    );

    // §2  Dose sweep — predator collapses before prey
    v.section("D02: Differential collapse");

    let doses: Vec<f64> = (0..200).map(|i| i as f64 * 1.0).collect();
    let predator_sweep = hormesis::sweep(&doses, &predator_params);
    let prey_sweep = hormesis::sweep(&doses, &prey_params);

    let predator_collapse_idx = predator_sweep.iter().position(|p| p.response < 0.5);
    let prey_collapse_idx = prey_sweep.iter().position(|p| p.response < 0.5);

    v.check_pass(
        "predator collapses (response < 0.5)",
        predator_collapse_idx.is_some(),
    );

    if let (Some(pred_idx), Some(prey_idx)) = (predator_collapse_idx, prey_collapse_idx) {
        v.check_pass("predator collapses before prey", pred_idx < prey_idx);
    } else if predator_collapse_idx.is_some() {
        v.check_pass("predator collapses but prey survives in range", true);
    }

    // §3  Anderson spectral sweep — disorder increases with dose
    v.section("D03: Anderson spectral sweep");

    let lattice_l = 8;
    let w_values: Vec<f64> = (1..=8).map(|i| f64::from(i) * 4.0).collect();
    let sweep = anderson_spectral::sweep(lattice_l, &w_values, 500, 42);
    v.check_count("spectral sweep = 8 W values", sweep.len(), 8);

    v.check_pass(
        "all <r> values in physical range [0.2, 0.8]",
        sweep.iter().all(|p| p.r > 0.2 && p.r < 0.8),
    );

    // §4  Spectral bandwidth increases with disorder
    v.section("D04: Spectral bandwidth trend");
    v.check_pass(
        "bandwidth at highest W > bandwidth at lowest W",
        sweep[sweep.len() - 1].bandwidth > sweep[0].bandwidth,
    );

    // §5  Trophic diversity under pesticide gradient
    v.section("D05: Trophic diversity under pesticide");

    let species_responses_low_dose: Vec<f64> = vec![
        hormesis::response(5.0, &predator_params),
        hormesis::response(5.0, &prey_params),
        hormesis::response(5.0, &prey_params) * 0.8,
        hormesis::response(5.0, &predator_params) * 0.6,
    ];
    let species_responses_high_dose: Vec<f64> = vec![
        hormesis::response(100.0, &predator_params),
        hormesis::response(100.0, &prey_params),
        hormesis::response(100.0, &prey_params) * 0.8,
        hormesis::response(100.0, &predator_params) * 0.6,
    ];

    let counts_low: Vec<f64> = species_responses_low_dose
        .iter()
        .map(|r| (r * 1000.0).max(1.0))
        .collect();
    let counts_high: Vec<f64> = species_responses_high_dose
        .iter()
        .map(|r| (r * 1000.0).max(1.0))
        .collect();

    let shannon_low = diversity::shannon(&counts_low);
    let shannon_high = diversity::shannon(&counts_high);
    v.check_pass(
        "Shannon diversity higher at low dose than high dose",
        shannon_low > shannon_high,
    );

    let pielou_low = diversity::pielou_evenness(&counts_low);
    let pielou_high = diversity::pielou_evenness(&counts_high);
    v.check_pass("Pielou evenness higher at low dose", pielou_low > pielou_high);

    // §6  Hormetic dose-disorder mapping for trophic analysis
    v.section("D06: Dose → disorder mapping for trophic species");

    let w_baseline = 5.0;
    let sensitivity = 0.5;
    let gamma = 1.0;

    v.check_pass(
        "disorder increases with dose",
        hormesis::dose_to_disorder(100.0, w_baseline, sensitivity, gamma)
            > hormesis::dose_to_disorder(5.0, w_baseline, sensitivity, gamma),
    );
    v.check(
        "W at zero dose = baseline",
        hormesis::dose_to_disorder(0.0, w_baseline, sensitivity, gamma),
        w_baseline,
        tolerances::ANALYTICAL_F64,
    );

    println!("\nTotal wall time: {:.2?}", t0.elapsed());
    v.finish();
}
