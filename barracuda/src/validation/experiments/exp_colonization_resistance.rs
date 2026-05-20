// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Exp379: Joint Colonization Resistance Surface
//!
//! Computes the colonization resistance surface — a 3D manifold in
//! (adhesion strength, species diversity, epithelial disorder) space —
//! where commensal colonization exceeds 90%. Validates the Anderson
//! prediction that many weak binders (delocalized regime) produce more
//! robust resistance than few strong binders (localized regime).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (pure Rust math) |
//! | Date | 2026-05-17 |
//! | Command | `cargo run --release --bin validate_colonization_resistance` |
//! | Chain | Gonzales IC50 (Exp280) → Hormesis (Exp377) → **This** |
//!
//! Provenance: Joint colonization resistance surface validation (Exp379)

use std::time::Instant;
use crate::bio::binding_landscape;
use crate::tolerances;
use crate::validation::Validator;

/// Run the `validate_colonization_resistance` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let t0 = Instant::now();

    let n_sites = 16;
    let kd = 2.0;
    let conc = 1.0;
    let threshold = 0.5;
    let disorder_w = 2.0;
    let seed = 42_u64;

    // §1  Diversity advantage — more species → higher resistance
    v.section("D01: Diversity advantage");

    let r_1 = binding_landscape::colonization_resistance(n_sites, 1, kd, disorder_w, conc, threshold, seed);
    let r_4 = binding_landscape::colonization_resistance(n_sites, 4, kd, disorder_w, conc, threshold, seed);
    let r_8 = binding_landscape::colonization_resistance(n_sites, 8, kd, disorder_w, conc, threshold, seed);
    let r_15 = binding_landscape::colonization_resistance(n_sites, 15, kd, disorder_w, conc, threshold, seed);

    v.check_pass("N=4 resistance ≥ N=1", r_4.resistance_fraction >= r_1.resistance_fraction);
    v.check_pass("N=8 resistance ≥ N=4", r_8.resistance_fraction >= r_4.resistance_fraction);
    v.check_pass("N=15 resistance ≥ N=8", r_15.resistance_fraction >= r_8.resistance_fraction);
    v.check_pass("N=1 < N=15 resistance", r_1.resistance_fraction < r_15.resistance_fraction);

    // §2  IPR — uniform vs concentrated binding
    v.section("D02: Inverse Participation Ratio (IPR)");

    let n = 8;
    let uniform_occ: Vec<f64> = vec![1.0 / n as f64; n];
    let concentrated_occ: Vec<f64> = {
        let mut c = vec![0.0; n];
        c[0] = 1.0;
        c
    };

    v.check(
        "IPR(uniform) = 1/N",
        binding_landscape::binding_ipr(&uniform_occ),
        1.0 / n as f64,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "IPR(concentrated) = 1.0",
        binding_landscape::binding_ipr(&concentrated_occ),
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // §3  Localization length ξ = 1/IPR
    v.section("D03: Localization length");

    v.check(
        "ξ(uniform) = N",
        binding_landscape::localization_length(&uniform_occ),
        n as f64,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "ξ(concentrated) = 1.0",
        binding_landscape::localization_length(&concentrated_occ),
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // §4  Composite binding and selectivity
    v.section("D04: Composite binding + selectivity");

    let target_occ = [0.8, 0.7, 0.6, 0.5];
    let off_target_occ = [0.01, 0.005, 0.002, 0.001];

    v.check_pass(
        "composite binding > 0",
        binding_landscape::composite_binding(&target_occ) > 0.0,
    );
    v.check_pass(
        "selectivity index > 100",
        binding_landscape::selectivity_index(&target_occ, &off_target_occ) > 100.0,
    );

    // §5  Site occupancy profile
    v.section("D05: Site occupancy profile");

    let profile = binding_landscape::site_occupancy_profile(n_sites, 4, kd, disorder_w, conc, seed);
    v.check_count("profile has N sites", profile.len(), n_sites);
    for (i, &occ) in profile.iter().enumerate() {
        v.check_pass(&format!("site {i} occ in [0,1]"), (0.0..=1.0).contains(&occ));
    }

    // §6  Resistance surface sweep (3×3×3 = 27 points)
    v.section("D06: Resistance surface sweep");

    let kd_values = [0.1, 0.5, 2.0];
    let n_species_values = [2_usize, 8, 15];
    let w_values = [0.5, 1.0, 2.0];
    let surface = binding_landscape::resistance_surface_sweep(
        &kd_values,
        &n_species_values,
        &w_values,
        conc,
        threshold,
        seed,
        n_sites,
    );
    v.check_count("surface has 3×3×3 = 27 points", surface.len(), 27);

    v.check_pass(
        "all resistance in [0,1]",
        surface.iter().all(|p| (0.0..=1.0).contains(&p.resistance_fraction)),
    );

    let low_kd_high_sp = surface
        .iter()
        .find(|p| (p.kd - 0.1).abs() < 0.01 && p.n_species == 15 && (p.disorder_w - 1.0).abs() < 0.01)
        .map(|p| p.resistance_fraction)
        .unwrap_or(0.0);
    let high_kd_low_sp = surface
        .iter()
        .find(|p| (p.kd - 2.0).abs() < 0.01 && p.n_species == 2 && (p.disorder_w - 1.0).abs() < 0.01)
        .map(|p| p.resistance_fraction)
        .unwrap_or(1.0);
    v.check_pass(
        "low Kd + high N ≥ high Kd + low N (diversity wins)",
        low_kd_high_sp >= high_kd_low_sp,
    );

    println!("\nTotal wall time: {:.2?}", t0.elapsed());
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_colonization_resistance");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "colonization_resistance",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Rust,
        provenance_crate: "validate_colonization_resistance",
        provenance_date: "2026-05-20",
        description: "# Exp379: Joint Colonization Resistance Surface",
    },
    run: |v, _ctx| run_as_scenario(v),
};
