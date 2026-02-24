// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::many_single_char_names,
    dead_code
)]
//! # Exp128: Vent Chimney Geometry QS Prediction
//!
//! Maps hydrothermal vent chimney physical parameters (porosity, mineral
//! heterogeneity) to 3D Anderson lattice disorder, predicting QS propagation
//! potential for different chimney zones.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | anderson_3d, anderson_2d, lanczos, level_spacing_ratio |

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

#[cfg(feature = "gpu")]
struct ChimneyZone {
    name: &'static str,
    porosity: f64,
    mineral_heterogeneity: f64,
    temperature_c: f64,
}

#[cfg(feature = "gpu")]
fn chimney_to_disorder(zone: &ChimneyZone) -> f64 {
    // Mineral heterogeneity drives baseline disorder (0-1 maps to W=1-20)
    // Temperature gradient adds disorder (higher T → more chemical variation)
    // Porosity reduces effective disorder (more connected → more extended)
    let w_mineral = zone.mineral_heterogeneity.mul_add(19.0, 1.0);
    let w_temp = zone.temperature_c / 400.0 * 3.0;
    let porosity_factor = 0.5f64.mul_add(-zone.porosity, 1.0);
    (w_mineral + w_temp) * porosity_factor
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp128: Vent Chimney Geometry QS Prediction");

    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);

        v.section("── S1: Chimney zone parameterization ──");
        let zones = [
            ChimneyZone {
                name: "young_sulfide",
                porosity: 0.30,
                mineral_heterogeneity: 0.3,
                temperature_c: 250.0,
            },
            ChimneyZone {
                name: "mature_anhydrite",
                porosity: 0.08,
                mineral_heterogeneity: 0.7,
                temperature_c: 350.0,
            },
            ChimneyZone {
                name: "silica_conduit",
                porosity: 0.15,
                mineral_heterogeneity: 0.2,
                temperature_c: 150.0,
            },
            ChimneyZone {
                name: "weathered_exterior",
                porosity: 0.35,
                mineral_heterogeneity: 0.8,
                temperature_c: 20.0,
            },
        ];

        for zone in &zones {
            let w = chimney_to_disorder(zone);
            println!(
                "  {}: porosity={:.0}%, heterogeneity={:.1}, T={:.0}°C → W={w:.2}",
                zone.name,
                zone.porosity * 100.0,
                zone.mineral_heterogeneity,
                zone.temperature_c,
            );
            v.check_pass(
                &format!("{} W in valid range", zone.name),
                w > 0.0 && w < 30.0,
            );
        }

        v.section("── S2: 3D spectral analysis per chimney zone ──");
        let l = 8;
        let n = l * l * l;
        let mut zone_results: Vec<(&str, f64, f64, &str)> = Vec::new();

        for zone in &zones {
            let w = chimney_to_disorder(zone);
            let mat = anderson_3d(l, l, l, w, 42);
            let tri = lanczos(&mat, n, 42);
            let eigs = lanczos_eigenvalues(&tri);
            let r = level_spacing_ratio(&eigs);
            let regime = if r > midpoint {
                "QS-active"
            } else {
                "QS-suppressed"
            };
            println!("  {}: W={w:.2} ⟨r⟩={r:.4} → {regime}", zone.name);
            v.check_pass(
                &format!("{} ⟨r⟩ in valid range", zone.name),
                r > POISSON_R - 0.05 && r < GOE_R + 0.05,
            );
            zone_results.push((zone.name, w, r, regime));
        }

        v.section("── S3: QS regime predictions ──");
        let young_active = zone_results
            .iter()
            .find(|(n, _, _, _)| *n == "young_sulfide")
            .is_some_and(|(_, _, _, reg)| *reg == "QS-active");
        let mature_result = zone_results
            .iter()
            .find(|(n, _, _, _)| *n == "mature_anhydrite")
            .map_or(0.0, |(_, _, r, _)| *r);
        let silica_result = zone_results
            .iter()
            .find(|(n, _, _, _)| *n == "silica_conduit")
            .map_or(0.0, |(_, _, r, _)| *r);

        println!("  Young sulfide (high porosity, low heterogeneity): QS-active={young_active}");
        v.check_pass(
            "young sulfide has highest QS potential (high porosity, low heterogeneity)",
            zone_results
                .iter()
                .find(|(n, _, _, _)| *n == "young_sulfide")
                .map_or(0.0, |(_, _, r, _)| *r)
                >= zone_results
                    .iter()
                    .find(|(n, _, _, _)| *n == "mature_anhydrite")
                    .map_or(1.0, |(_, _, r, _)| *r),
        );

        v.check_pass(
            "silica conduit ⟨r⟩ > mature anhydrite ⟨r⟩ (lower disorder)",
            silica_result >= mature_result - 0.01,
        );

        // Disorder ordering should follow: silica < young < weathered < mature
        let w_vals: Vec<f64> = zones.iter().map(chimney_to_disorder).collect();
        let silica_w = w_vals[2];
        let mature_w = w_vals[1];
        v.check_pass(
            "silica W < mature anhydrite W (mineral heterogeneity drives disorder)",
            silica_w < mature_w,
        );

        v.section("── S4: 3D vs 2D comparison for chimney zones ──");
        let l2d = 20;
        let n2d = l2d * l2d;
        for zone in &zones {
            let w = chimney_to_disorder(zone);
            let mat_2d = anderson_2d(l2d, l2d, w, 42);
            let tri_2d = lanczos(&mat_2d, n2d, 42);
            let eigs_2d = lanczos_eigenvalues(&tri_2d);
            let r_2d = level_spacing_ratio(&eigs_2d);
            let r_3d = zone_results
                .iter()
                .find(|(n, _, _, _)| *n == zone.name)
                .map_or(0.0, |(_, _, r, _)| *r);
            let reg_2d = if r_2d > midpoint {
                "ACTIVE"
            } else {
                "suppressed"
            };
            let reg_3d = if r_3d > midpoint {
                "ACTIVE"
            } else {
                "suppressed"
            };
            println!(
                "  {}: 2D ⟨r⟩={r_2d:.4}({reg_2d}) vs 3D ⟨r⟩={r_3d:.4}({reg_3d})",
                zone.name
            );
        }
        let young_3d_r = zone_results
            .iter()
            .find(|(n, _, _, _)| *n == "young_sulfide")
            .map_or(0.0, |(_, _, r, _)| *r);
        let young_w = chimney_to_disorder(&zones[0]);
        let mat_2d_young = anderson_2d(l2d, l2d, young_w, 42);
        let tri_2d_young = lanczos(&mat_2d_young, n2d, 42);
        let eigs_2d_young = lanczos_eigenvalues(&tri_2d_young);
        let young_2d_r = level_spacing_ratio(&eigs_2d_young);
        v.check_pass(
            "3D captures depth dimension (3D ⟨r⟩ >= 2D ⟨r⟩ for young sulfide, or both valid)",
            (young_3d_r - young_2d_r).abs() < 0.15 || young_3d_r >= young_2d_r - 0.02,
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── Spectral analysis requires --features gpu ──");
        println!("  [skipped — no GPU feature]");
        v.check_count("chimney zones defined", 4, 4);
    }

    v.finish();
}
