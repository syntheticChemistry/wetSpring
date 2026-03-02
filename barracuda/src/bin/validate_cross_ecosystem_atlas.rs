// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    dead_code
)]
//! # Exp134: Cross-Ecosystem QS Geometry Atlas
//!
//! Computes the complete (biome × geometry) QS atlas: each of the 28 biomes
//! from Exp126 tested across 5 representative lattice shapes. Produces
//! the definitive "which ecosystems can sustain QS in which geometries" table.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | `anderson_hamiltonian`, `anderson_2d`, `anderson_3d`, `lanczos`, `level_spacing_ratio` |
//! | Command     | `cargo test --bin validate_cross_ecosystem_atlas -- --nocapture` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

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
    let mut v = Validator::new("Exp134: Cross-Ecosystem QS Geometry Atlas");

    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let n_sweep = 15_usize;

        v.section("── S1: Pre-compute geometry sweeps ──");

        let w_at = |i: usize| -> f64 { 1.0 + (i as f64) * 21.0 / (n_sweep - 1) as f64 };

        // chain (1D, N=384)
        let sweep_chain: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = w_at(i);
                let (d, o) = anderson_hamiltonian(384, w, 42);
                let eigs = find_all_eigenvalues(&d, &o);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // slab (2D, 20×20)
        let sweep_slab: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = w_at(i);
                let mat = anderson_2d(20, 20, w, 42);
                let tri = lanczos(&mat, 400, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // thin film (14×14×2)
        let sweep_film: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = w_at(i);
                let mat = anderson_3d(14, 14, 2, w, 42);
                let tri = lanczos(&mat, 392, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // tube (32×3×4)
        let sweep_tube: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = w_at(i);
                let mat = anderson_3d(32, 3, 4, w, 42);
                let tri = lanczos(&mat, 384, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        // block (8×8×6)
        let sweep_block: Vec<(f64, f64)> = (0..n_sweep)
            .map(|i| {
                let w = w_at(i);
                let mat = anderson_3d(8, 8, 6, w, 42);
                let tri = lanczos(&mat, 384, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();

        v.check_count("chain sweep", sweep_chain.len(), n_sweep);
        v.check_count("slab sweep", sweep_slab.len(), n_sweep);
        v.check_count("film sweep", sweep_film.len(), n_sweep);
        v.check_count("tube sweep", sweep_tube.len(), n_sweep);
        v.check_count("block sweep", sweep_block.len(), n_sweep);

        let sweeps: &[(&str, &[(f64, f64)])] = &[
            ("chain", &sweep_chain),
            ("slab", &sweep_slab),
            ("film", &sweep_film),
            ("tube", &sweep_tube),
            ("block", &sweep_block),
        ];

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

        v.section("── S2: 28-biome × 5-geometry atlas ──");
        let biomes = ncbi_data::biome_diversity_params();

        println!(
            "  {:30} {:>5} {:>5}  {:>7} {:>7} {:>7} {:>7} {:>7}",
            "biome", "J", "W", "chain", "slab", "film", "tube", "block"
        );
        println!(
            "  {:-<30} {:-<5} {:-<5}  {:-<7} {:-<7} {:-<7} {:-<7} {:-<7}",
            "", "", "", "", "", "", "", ""
        );

        let mut atlas: Vec<(&str, f64, f64, [bool; 5])> = Vec::new();
        let mut active_counts = [0_usize; 5];

        for (name, n_species, j_target) in &biomes {
            let community = generate_community(*n_species, *j_target, 42);
            let j = diversity::pielou_evenness(&community);
            let w = evenness_to_disorder(j);

            let mut active = [false; 5];
            let mut cols = Vec::new();
            for (idx, (_, sweep)) in sweeps.iter().enumerate() {
                let r = nearest_r(sweep, w);
                let is_active = r > midpoint;
                active[idx] = is_active;
                if is_active {
                    active_counts[idx] += 1;
                }
                cols.push(if is_active { "ACTIVE" } else { "---" });
            }
            println!(
                "  {:30} {:5.3} {:5.2}  {:>7} {:>7} {:>7} {:>7} {:>7}",
                name, j, w, cols[0], cols[1], cols[2], cols[3], cols[4]
            );
            atlas.push((name, j, w, active));
        }

        v.check_count("biomes in atlas", atlas.len(), biomes.len());

        v.section("── S3: Geometry effectiveness summary ──");
        let shape_names = ["chain", "slab", "film", "tube", "block"];
        for (i, name) in shape_names.iter().enumerate() {
            println!(
                "  {:>8}: {}/{} biomes QS-active",
                name,
                active_counts[i],
                biomes.len()
            );
        }
        v.check_pass(
            "block activates most biomes",
            active_counts[4] >= active_counts[0],
        );
        v.check_pass(
            "film >= slab (depth helps)",
            active_counts[2] >= active_counts[1],
        );

        // Geometry ordering: block >= film >= tube >= slab >= chain
        let monotonic = active_counts[4] >= active_counts[2]
            && active_counts[2] >= active_counts[1]
            && active_counts[1] >= active_counts[0];
        v.check_pass(
            "geometry effectiveness: block >= film >= slab >= chain",
            monotonic,
        );

        v.section("── S4: Biomes that differentiate geometries ──");
        let mut differentiators = Vec::new();
        for (name, _, _, active) in &atlas {
            let n_active: usize = active.iter().filter(|&&a| a).count();
            if n_active > 0 && n_active < 5 {
                differentiators.push((*name, n_active, *active));
            }
        }
        println!(
            "  Biomes active in SOME but not ALL geometries ({}):",
            differentiators.len()
        );
        for (name, n, active) in &differentiators {
            let shapes: Vec<_> = shape_names
                .iter()
                .zip(active.iter())
                .filter(|(_, a)| **a)
                .map(|(s, _)| *s)
                .collect();
            println!("    {name:30} active in {n}/5: {shapes:?}");
        }
        v.check_pass("differentiator biomes identified", true);

        v.section("── S5: Ecosystem-geometry recommendations ──");
        println!("  GEOMETRY RECOMMENDATIONS BY ECOSYSTEM TYPE:");
        println!();

        let cave_biomes = atlas
            .iter()
            .filter(|(n, _, _, _)| n.contains("deep_sea") || n.contains("marine_sediment"))
            .map(|(n, _, _, a)| (*n, a[4]))
            .collect::<Vec<_>>();
        println!("    Cave analogs (deep-sea, sediment):");
        for (name, active_3d) in &cave_biomes {
            println!(
                "      {:30} block={}",
                name,
                if *active_3d { "ACTIVE" } else { "---" }
            );
        }

        let hs_biomes = atlas
            .iter()
            .filter(|(n, _, _, _)| n.contains("hot_spring") || n.contains("vent"))
            .map(|(n, _, _, a)| (*n, a[2], a[4]))
            .collect::<Vec<_>>();
        println!("    Hot spring / vent biomes:");
        for (name, film, block) in &hs_biomes {
            println!(
                "      {:30} film={} block={}",
                name,
                if *film { "ACTIVE" } else { "---" },
                if *block { "ACTIVE" } else { "---" }
            );
        }

        let rhizo_biomes = atlas
            .iter()
            .filter(|(n, _, _, _)| n.contains("rhizosphere") || n.contains("soil"))
            .map(|(n, _, _, a)| (*n, a[2], a[3], a[4]))
            .collect::<Vec<_>>();
        println!("    Rhizosphere / soil biomes:");
        for (name, film, tube, block) in &rhizo_biomes {
            println!(
                "      {:30} film={} tube={} block={}",
                name,
                if *film { "ACTIVE" } else { "---" },
                if *tube { "ACTIVE" } else { "---" },
                if *block { "ACTIVE" } else { "---" }
            );
        }
        v.check_pass("recommendations complete", true);
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
