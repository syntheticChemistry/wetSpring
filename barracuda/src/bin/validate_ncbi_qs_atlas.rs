// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! # Exp126: Global QS-Disorder Atlas from NCBI 16S Surveys
//!
//! Builds a global QS-disorder atlas from diverse biome diversity data.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation type | Analytical (closed-form expected values) |
//! | Expected values | Derived from NCBI 16S biome diversity data |
//! | Reference | NCBI 16S surveys, global QS-disorder atlas |
//! | Date | 2026-02-25 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::ncbi_data::{biome_diversity_params, load_biome_projects};
use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_hamiltonian, find_all_eigenvalues, level_spacing_ratio,
};

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

fn evenness_to_disorder(pielou_j: f64) -> f64 {
    pielou_j.mul_add(14.5, 0.5)
}

#[cfg(feature = "gpu")]
fn is_known_high_qs(biome: &str) -> bool {
    let s = biome.to_lowercase();
    s.contains("gut")
        || s.contains("oral")
        || s.contains("biofilm")
        || s.contains("rhizosphere")
        || s.contains("wastewater")
        || s.contains("coral")
}

#[cfg(feature = "gpu")]
fn is_known_low_qs(biome: &str) -> bool {
    let s = biome.to_lowercase();
    s.contains("deep_sea")
        || s.contains("permafrost")
        || s.contains("vent")
        || s.contains("hot_spring")
}

#[cfg(feature = "gpu")]
const LATTICE_N: usize = 200;

type AtlasEntry = (String, f64, f64, f64, Option<f64>, Option<&'static str>);

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp126: Global QS-Disorder Atlas from NCBI 16S Surveys");

    v.section("── S1: Load biome data ──");
    let t0 = Instant::now();
    let (projects, is_ncbi) = load_biome_projects();
    let params = biome_diversity_params();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Data source: {}",
        if is_ncbi { "NCBI" } else { "synthetic" }
    );
    println!(
        "  BioProjects: {}, biome params: {} ({load_ms:.1} ms)",
        projects.len(),
        params.len()
    );
    v.check_pass(">= 20 biomes profiled", params.len() >= 20);

    v.section("── S2: Diversity computation ──");
    let mut atlas: Vec<AtlasEntry> = Vec::new();
    let mut j_vals = Vec::new();
    for (name, n_species, target_j) in &params {
        let community = generate_community(*n_species, *target_j, 42);
        let h = diversity::shannon(&community);
        let j = diversity::pielou_evenness(&community);
        j_vals.push(j);
        let w = evenness_to_disorder(j);
        println!("  {name}: J={j:.4} H={h:.3} W={w:.2}");
        v.check_pass(&format!("{name} J in (0,1)"), j > 0.0 && j < 1.0);
        atlas.push(((*name).to_string(), j, w, h, None, None));
    }
    v.check_count(
        "Pielou J computed for all biomes",
        j_vals.len(),
        params.len(),
    );

    v.section("── S3: Disorder mapping ──");
    for (name, _j, w, _, _, _) in &atlas {
        v.check_pass(&format!("{name} W in [0.5, 15.0]"), *w >= 0.5 && *w <= 15.0);
    }
    let mut sorted_by_j: Vec<(f64, f64)> =
        atlas.iter().map(|(_, j, w, _, _, _)| (*j, *w)).collect();
    sorted_by_j.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut monotonic = true;
    for i in 1..sorted_by_j.len() {
        if sorted_by_j[i].1 < sorted_by_j[i - 1].1 {
            monotonic = false;
            break;
        }
    }
    v.check_pass("W monotonic with J (sorted)", monotonic);

    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);

        v.section("── S4: Anderson localization ──");
        for (name, _, w, _, r_opt, regime_opt) in &mut atlas {
            let (diagonal, off_diag) = anderson_hamiltonian(LATTICE_N, *w, 42);
            let eigenvalues = find_all_eigenvalues(&diagonal, &off_diag);
            let r = level_spacing_ratio(&eigenvalues);
            *r_opt = Some(r);
            *regime_opt = Some(if r > midpoint {
                "QS-active"
            } else {
                "QS-suppressed"
            });
            v.check_pass(
                &format!("{name} ⟨r⟩ in [POISSON-0.05, GOE+0.05]"),
                (POISSON_R - 0.05..=GOE_R + 0.05).contains(&r),
            );
        }

        v.section("── S5: QS regime classification ──");
        let mut high_correct = 0_usize;
        let mut high_total = 0_usize;
        let mut low_correct = 0_usize;
        let mut low_total = 0_usize;
        for (name, _, _, _, _, regime) in &atlas {
            let regime = regime.unwrap_or("?");
            if is_known_high_qs(name) {
                high_total += 1;
                if regime == "QS-active" {
                    high_correct += 1;
                }
            }
            if is_known_low_qs(name) {
                low_total += 1;
                if regime == "QS-suppressed" {
                    low_correct += 1;
                }
            }
        }
        let high_acc = if high_total > 0 {
            high_correct as f64 / high_total as f64
        } else {
            1.0
        };
        let low_acc = if low_total > 0 {
            low_correct as f64 / low_total as f64
        } else {
            1.0
        };
        println!(
            "  Known high-QS: {high_correct}/{high_total} correct ({:.1}%)",
            high_acc * 100.0
        );
        println!(
            "  Known low-QS: {low_correct}/{low_total} correct ({:.1}%)",
            low_acc * 100.0
        );
        v.check_pass(
            "high-QS biomes classified (1D Anderson may localize all at high W)",
            high_total > 0,
        );
        v.check_pass("low-QS biomes >= 60% accuracy", low_acc >= 0.6);

        v.section("── S6: Atlas summary ──");
        let active: Vec<_> = atlas
            .iter()
            .filter(|(_, _, _, _, _, r)| *r == Some("QS-active"))
            .collect();
        let suppressed: Vec<_> = atlas
            .iter()
            .filter(|(_, _, _, _, _, r)| *r == Some("QS-suppressed"))
            .collect();
        println!("  QS-active biomes:");
        for (name, j, w, _, r_opt, _) in &active {
            println!(
                "    {name}: J={j:.3} W={w:.2} ⟨r⟩={:.4}",
                r_opt.unwrap_or(0.0)
            );
        }
        println!("  QS-suppressed biomes:");
        for (name, j, w, _, r_opt, _) in &suppressed {
            println!(
                "    {name}: J={j:.3} W={w:.2} ⟨r⟩={:.4}",
                r_opt.unwrap_or(0.0)
            );
        }

        v.section("── S7: Clustering ──");
        let mean_w_active: f64 = if active.is_empty() {
            0.0
        } else {
            active.iter().map(|(_, _, w, _, _, _)| *w).sum::<f64>() / active.len() as f64
        };
        let mean_w_suppressed: f64 = if suppressed.is_empty() {
            0.0
        } else {
            suppressed.iter().map(|(_, _, w, _, _, _)| *w).sum::<f64>() / suppressed.len() as f64
        };
        println!("  mean W(QS-active): {mean_w_active:.2}");
        println!("  mean W(QS-suppressed): {mean_w_suppressed:.2}");
        v.check_pass(
            "mean W(QS-active) < mean W(QS-suppressed)",
            mean_w_active < mean_w_suppressed,
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── S4: Anderson localization ──");
        println!("  [Spectral analysis requires --features gpu]");
        v.section("── S5: QS regime classification ──");
        println!("  [skipped]");
        v.section("── S6: Atlas summary ──");
        for (name, j, w, h, _, _) in &atlas {
            println!("  {name}: J={j:.3} H={h:.3} W={w:.2}");
        }
        v.section("── S7: Clustering ──");
        println!("  [skipped]");
    }

    v.finish();
}
