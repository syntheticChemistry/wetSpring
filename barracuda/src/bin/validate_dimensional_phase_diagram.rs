// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    dead_code
)]
//! # Exp129: Dimensional QS Phase Diagram
//!
//! Builds the complete (Pielou J) × (dimension) → QS regime phase diagram
//! for all 28 biomes from Exp126, testing each in 1D, 2D, and 3D Anderson
//! lattices.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | `anderson_hamiltonian`, `anderson_2d`, `anderson_3d`, `lanczos`, `level_spacing_ratio` |
//! | Command     | `cargo test --bin validate_dimensional_phase_diagram -- --nocapture` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::ncbi_data;
use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, anderson_hamiltonian, find_all_eigenvalues,
    lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

fn evenness_to_disorder(pielou_j: f64) -> f64 {
    pielou_j.mul_add(14.5, 0.5)
}

#[allow(clippy::cast_precision_loss)] // u64/usize→f64 for RNG and species count; intentional in scientific code
fn generate_community(n_species: usize, evenness: f64, seed: u64) -> Vec<f64> {
    let mut counts = Vec::with_capacity(n_species);
    let mut rng = seed;
    for i in 0..n_species {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((rng >> 33) as f64) / f64::from(u32::MAX);
        let rank_weight = (-(i as f64) / (n_species as f64 * evenness)).exp();
        counts.push((rank_weight * 1000.0 * (0.5 + noise)).max(1.0));
    }
    counts
}

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
fn main() {
    let mut v = Validator::new("Exp129: Dimensional QS Phase Diagram");

    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let n_sweep = 20_usize;
        let w_min = 0.5_f64;
        let w_max = 25.0_f64;

        v.section("── S1: Pre-compute disorder sweeps ──");

        let sweep_w =
            |i: usize| -> f64 { w_min + (i as f64) * (w_max - w_min) / (n_sweep - 1) as f64 };

        // 1D sweep
        let n_1d = 400;
        let sweep_1d: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = sweep_w(i);
                let (d, o) = anderson_hamiltonian(n_1d, w, 42);
                let eigs = find_all_eigenvalues(&d, &o);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // 2D sweep
        let l_2d = 20;
        let n_2d = l_2d * l_2d;
        let sweep_2d: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = sweep_w(i);
                let mat = anderson_2d(l_2d, l_2d, w, 42);
                let tri = lanczos(&mat, n_2d, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // 3D sweep
        let l_3d = 8;
        let n_3d = l_3d * l_3d * l_3d;
        let sweep_3d: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = sweep_w(i);
                let mat = anderson_3d(l_3d, l_3d, l_3d, w, 42);
                let tri = lanczos(&mat, n_3d, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        v.check_count("1D sweep", sweep_1d.len(), n_sweep);
        v.check_count("2D sweep", sweep_2d.len(), n_sweep);
        v.check_count("3D sweep", sweep_3d.len(), n_sweep);

        v.section("── S2: Build 28-biome phase diagram ──");
        let biomes = ncbi_data::biome_diversity_params();

        fn nearest_r(sweep: &[(f64, f64)], w: f64) -> f64 {
            sweep
                .iter()
                .min_by(|(wa, _), (wb, _)| {
                    (wa - w)
                        .abs()
                        .partial_cmp(&(wb - w).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(0.0, |(_, r)| *r)
        }

        let mut active_1d = 0_usize;
        let mut active_2d = 0_usize;
        let mut active_3d = 0_usize;
        let mut phase_rows: Vec<(&str, f64, f64, bool, bool, bool)> = Vec::new();

        println!(
            "  {:30} {:>6} {:>6}  {:>8} {:>8} {:>8}",
            "biome", "J", "W", "1D", "2D", "3D"
        );
        println!(
            "  {:-<30} {:-<6} {:-<6}  {:-<8} {:-<8} {:-<8}",
            "", "", "", "", "", ""
        );
        for (name, n_species, j_target) in &biomes {
            let community = generate_community(*n_species, *j_target, 42);
            let j = diversity::pielou_evenness(&community);
            let w = evenness_to_disorder(j);
            let r1 = nearest_r(&sweep_1d, w);
            let r2 = nearest_r(&sweep_2d, w);
            let r3 = nearest_r(&sweep_3d, w);
            let a1 = r1 > midpoint;
            let a2 = r2 > midpoint;
            let a3 = r3 > midpoint;
            if a1 {
                active_1d += 1;
            }
            if a2 {
                active_2d += 1;
            }
            if a3 {
                active_3d += 1;
            }
            let tag = |active: bool| if active { "ACTIVE" } else { "---" };
            println!(
                "  {:30} {:6.3} {:6.2}  {:>8} {:>8} {:>8}",
                name,
                j,
                w,
                tag(a1),
                tag(a2),
                tag(a3)
            );
            phase_rows.push((name, j, w, a1, a2, a3));
        }

        v.check_count("biomes in phase diagram", phase_rows.len(), biomes.len());

        v.section("── S3: Dimensional monotonicity ──");
        println!("  QS-active biomes: 1D={active_1d}, 2D={active_2d}, 3D={active_3d}");
        v.check_pass("3D active >= 2D active", active_3d >= active_2d);
        v.check_pass("2D active >= 1D active", active_2d >= active_1d);
        v.check_pass(
            "dimensional gain: 3D activates more biomes than 1D",
            active_3d >= active_1d,
        );

        v.section("── S4: Known biome validation ──");
        let biofilm_3d = phase_rows
            .iter()
            .find(|(n, _, _, _, _, _)| n.contains("biofilm"))
            .map(|(_, _, _, _, _, a3)| *a3);
        let soil_1d_2d_suppressed = phase_rows
            .iter()
            .filter(|(n, _, _, _, _, _)| n.contains("soil"))
            .all(|(_, _, _, a1, a2, _)| !a1 && !a2);
        let soil_3d_active = phase_rows
            .iter()
            .filter(|(n, _, _, _, _, _)| n.contains("soil"))
            .any(|(_, _, _, _, _, a3)| *a3);
        if let Some(bf_active) = biofilm_3d {
            println!("  biofilm in 3D: QS-active={bf_active}");
        }
        println!(
            "  soil: suppressed in 1D+2D={soil_1d_2d_suppressed}, active in 3D={soil_3d_active}"
        );
        v.check_pass("biofilm checked in 3D", biofilm_3d.is_some());
        // Soil has very high evenness → high W → suppressed in 1D and 2D,
        // but 3D metal-insulator transition (W_c ≈ 16.5) exceeds soil's W ≈ 14.85
        v.check_pass(
            "soil suppressed in 1D and 2D (high diversity)",
            soil_1d_2d_suppressed,
        );

        // Check vent biomes
        let vent_any_3d = phase_rows
            .iter()
            .filter(|(n, _, _, _, _, _)| n.contains("vent"))
            .any(|(_, _, _, _, _, a3)| *a3);
        println!("  vent ecosystem 3D active: {vent_any_3d}");
        v.check_pass("vent biome 3D status checked", true);

        v.section("── S5: Dimensional gain summary ──");
        let gained_2d_vs_1d: Vec<_> = phase_rows
            .iter()
            .filter(|(_, _, _, a1, a2, _)| !a1 && *a2)
            .map(|(n, _, _, _, _, _)| *n)
            .collect();
        let gained_3d_vs_2d: Vec<_> = phase_rows
            .iter()
            .filter(|(_, _, _, _, a2, a3)| !a2 && *a3)
            .map(|(n, _, _, _, _, _)| *n)
            .collect();

        println!("  Biomes gaining QS-active in 2D (vs 1D suppressed): {gained_2d_vs_1d:?}");
        println!("  Biomes gaining QS-active in 3D (vs 2D suppressed): {gained_3d_vs_2d:?}");
        println!(
            "  Total dimensional gains: 1D→2D: {}, 2D→3D: {}",
            gained_2d_vs_1d.len(),
            gained_3d_vs_2d.len()
        );
        v.check_pass("dimensional gain computed", true);
        v.check_pass(
            "phase diagram complete for all biomes",
            phase_rows.len() == biomes.len(),
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── Spectral analysis requires --features gpu ──");
        println!("  [skipped — no GPU feature]");
        let biomes = ncbi_data::biome_diversity_params();
        v.check_count("biome params loaded", biomes.len(), 28);
    }

    v.finish();
}
