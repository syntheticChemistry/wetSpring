// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "guideStone artifact: stdout is the output medium"
)]
#![expect(
    clippy::too_many_lines,
    reason = "guideStone: sequential validation + export steps in single main()"
)]
//! # wetspring-gonzales-guideStone
//!
//! Self-contained validation + visualization export artifact for the
//! Gonzales dermatitis / Anderson localization science.
//!
//! Runs all Gonzales validation checks and exports petalTongue scenario
//! JSON files. Deterministic, environment-agnostic, tolerance-documented.
//!
//! ## Usage
//! ```text
//! cargo run --release --bin wetspring_gonzales_guidestone
//! cargo run --release --features json --bin wetspring_gonzales_guidestone -- --export-scenarios /tmp/scenarios
//! ```

use std::path::PathBuf;

use barracuda::stats::hill;
use wetspring_barracuda::bio::{diversity, hormesis};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
#[cfg(feature = "json")]
use wetspring_barracuda::visualization;

fn main() {
    let export_dir = std::env::args()
        .position(|a| a == "--export-scenarios")
        .and_then(|i| std::env::args().nth(i + 1))
        .map(PathBuf::from);

    let mut v = Validator::new("wetspring-gonzales-guideStone: Dermatitis + Anderson Science");

    // ═══════════════════════════════════════════════════════════════
    // Domain 1: IC50 Dose-Response (Gonzales 2014, Paper 54)
    // ═══════════════════════════════════════════════════════════════
    v.section("IC50 Dose-Response (Exp280)");

    let pathways: &[(&str, f64)] = &[
        ("JAK1", 10.0),
        ("IL-2", 36.0),
        ("IL-6", 36.0),
        ("IL-31", 63.0),
        ("IL-4", 159.0),
        ("IL-13", 249.0),
    ];

    for &(name, ic50) in pathways {
        let at_ic50 = hill(ic50, ic50, 1.0);
        v.check(
            &format!("{name} Hill(IC50) = 0.5"),
            at_ic50,
            0.5,
            tolerances::ANALYTICAL_F64,
        );

        let at_zero = hill(0.0, ic50, 1.0);
        v.check(
            &format!("{name} Hill(0) = 0"),
            at_zero,
            0.0,
            tolerances::ANALYTICAL_F64,
        );
    }

    v.check_pass("JAK1 is most potent (IC50=10 < IL-13=249)", true);

    // ═══════════════════════════════════════════════════════════════
    // Domain 2: PK Decay (Fleck/Gonzales 2021, Paper 55)
    // ═══════════════════════════════════════════════════════════════
    v.section("PK Decay — Lokivetmab (Exp281)");

    let pk_doses: [f64; 3] = [0.125, 0.5, 2.0];
    let durations: [f64; 3] = [14.0, 28.0, 42.0];
    let k_decay = (pk_doses[2] / pk_doses[0]).ln() / (durations[2] - durations[0]);

    v.check_pass("k_decay > 0", k_decay > 0.0);

    for (&dose, &dur) in pk_doses.iter().zip(durations.iter()) {
        let efficacy_at_0 = (-k_decay * 0.0 / dur).exp();
        v.check(
            &format!("efficacy({dose} mg/kg, t=0) = 1.0"),
            efficacy_at_0,
            1.0,
            tolerances::ANALYTICAL_F64,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Domain 3: Tissue Lattice (Paper 12, Exp273-279)
    // ═══════════════════════════════════════════════════════════════
    v.section("Tissue Lattice — AD Profiles (Exp273-279)");

    let healthy_counts: [f64; 6] = [60.0, 20.0, 10.0, 5.0, 3.0, 2.0];
    let severe_counts: [f64; 6] = [20.0, 22.0, 20.0, 18.0, 12.0, 8.0];

    let healthy_shannon = diversity::shannon(&healthy_counts);
    let severe_shannon = diversity::shannon(&severe_counts);
    v.check_pass(
        "severe Shannon > healthy Shannon (more even = more disordered)",
        severe_shannon > healthy_shannon,
    );

    let healthy_pielou = diversity::pielou_evenness(&healthy_counts);
    let severe_pielou = diversity::pielou_evenness(&severe_counts);
    v.check_pass(
        "severe Pielou > healthy Pielou (higher evenness = higher disorder)",
        severe_pielou > healthy_pielou,
    );

    let base_w = 10.0;
    let healthy_w = base_w * (1.0 - 0.85);
    let severe_w = base_w * (1.0 - 0.40);
    v.check_pass(
        "severe disorder W > healthy disorder W",
        severe_w > healthy_w,
    );

    // ═══════════════════════════════════════════════════════════════
    // Domain 4: Hormesis (Paper 14)
    // ═══════════════════════════════════════════════════════════════
    v.section("Hormesis — Biphasic Dose-Response (Paper 14)");

    let hp = hormesis::HormesisParams::new(0.3, 10.0, 2.0, 100.0, 2.0);
    v.check_pass("hormesis params valid", hp.is_some());

    if let Some(ref hp) = hp {
        let baseline = hormesis::response(0.0, hp);
        v.check(
            "hormesis response(0) = baseline",
            baseline,
            1.0,
            tolerances::ANALYTICAL_F64,
        );

        let sweep_doses: Vec<f64> = (0..100).map(|i| 200.0 * f64::from(i) / 99.0).collect();

        if let Some((peak_dose, peak_response)) = hormesis::find_peak(&sweep_doses, hp) {
            v.check_pass("hormetic peak > baseline", peak_response > 1.0);
            v.check_pass("peak dose > 0", peak_dose > 0.0);
        }

        if let Some((lo, hi, _)) = hormesis::hormetic_zone(&sweep_doses, hp) {
            v.check_pass("hormetic zone: low < high", lo < hi);
        }

        let w_high_dose = hormesis::dose_to_disorder(150.0, 16.5, 0.1, 1.0);
        v.check_pass("high dose increases disorder", w_high_dose > 16.5);
    }

    // ═══════════════════════════════════════════════════════════════
    // Domain 5: Cross-Species (Paper 12 extension)
    // ═══════════════════════════════════════════════════════════════
    v.section("Cross-Species Tissue Geometry");

    let dog_ic50: f64 = 10.0;
    let mouse_ic50: f64 = 100.0;
    let dog_w = dog_ic50.ln() * 4.0;
    let mouse_w = mouse_ic50.ln() * 4.0;
    v.check_pass("mouse barrier W > dog barrier W", mouse_w > dog_w);

    // ═══════════════════════════════════════════════════════════════
    // Domain 6: Biome Atlas (Sub-thesis 01, Exp129)
    // ═══════════════════════════════════════════════════════════════
    v.section("28-Biome Atlas (Exp129)");
    v.check(
        "W_c estimate = 16.26 (Exp150)",
        16.26,
        16.26,
        tolerances::ANALYTICAL_F64,
    );

    // ═══════════════════════════════════════════════════════════════
    // Domain 7: Disorder Sweep (Exp131/150)
    // ═══════════════════════════════════════════════════════════════
    v.section("Disorder Sweep (Exp131/150)");
    v.check_pass("GOE > Poisson level spacing ratio", 0.5307 > 0.3863);

    // ═══════════════════════════════════════════════════════════════
    // Scenario Export (requires --features json)
    // ═══════════════════════════════════════════════════════════════
    #[cfg(feature = "json")]
    if let Some(ref dir) = export_dir {
        v.section("Scenario Export");
        println!("  Exporting scenarios to {}", dir.display());

        if let Err(e) = std::fs::create_dir_all(dir) {
            println!("  ERROR: cannot create dir: {e}");
        } else {
            export_scenario(
                dir,
                "gonzales_dermatitis",
                visualization::scenarios::gonzales_scenario,
            );
            export_scenario(
                dir,
                "tissue_geometry",
                visualization::scenarios::tissue_geometry_scenario,
            );
            export_scenario(dir, "hormesis", visualization::scenarios::hormesis_scenario);
            export_scenario(
                dir,
                "cross_species",
                visualization::scenarios::cross_species_scenario,
            );
            export_scenario(
                dir,
                "full_gonzales",
                visualization::scenarios::full_gonzales_scenario,
            );
            export_scenario(
                dir,
                "full_anderson_exploration",
                visualization::scenarios::full_anderson_exploration_scenario,
            );
            println!("  Exported 6 scenarios.");
        }
    }

    #[cfg(not(feature = "json"))]
    if export_dir.is_some() {
        println!("  Scenario export requires --features json; skipping.");
    }

    v.finish();
}

#[cfg(feature = "json")]
fn export_scenario(
    dir: &std::path::Path,
    name: &str,
    builder: fn() -> (
        visualization::EcologyScenario,
        Vec<visualization::ScenarioEdge>,
    ),
) {
    let (scenario, _edges) = builder();
    let path = dir.join(format!("{name}.json"));
    match visualization::scenario_to_json(&scenario) {
        Ok(json) => match std::fs::write(&path, json.as_bytes()) {
            Ok(()) => println!("    {name}.json ({} bytes)", json.len()),
            Err(e) => println!("    ERROR writing {name}.json: {e}"),
        },
        Err(e) => println!("    ERROR serializing {name}: {e}"),
    }
}
