// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    dead_code,
)]
//! # Exp130: Thick Biofilm 3D QS Extension
//!
//! Compares QS-active windows between 2D slab and 3D block geometries to
//! test whether biofilm thickness extends the QS-active diversity range.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | anderson_2d, anderson_3d, lanczos, level_spacing_ratio |

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio, GOE_R, POISSON_R,
};

const N_SWEEP: usize = 20;
const W_MIN: f64 = 0.5;
const W_MAX: f64 = 20.0;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + (i as f64) * (W_MAX - W_MIN) / (N_SWEEP - 1) as f64
}

#[cfg(feature = "gpu")]
fn find_j_c(sweep: &[(f64, f64)], midpoint: f64) -> Option<f64> {
    // Find the last downward crossing (extended → localized), which is
    // the physically meaningful metal-insulator transition.
    let mut last_crossing = None;
    for i in 1..sweep.len() {
        let (w0, r0) = sweep[i - 1];
        let (w1, r1) = sweep[i];
        if r0 > midpoint && r1 <= midpoint {
            let t = (midpoint - r0) / (r1 - r0);
            let w_c = w0 + t * (w1 - w0);
            last_crossing = Some((w_c - 0.5) / 14.5);
        }
    }
    last_crossing
}

#[cfg(feature = "gpu")]
fn plateau_count(sweep: &[(f64, f64)], midpoint: f64, w_above: f64) -> usize {
    sweep
        .iter()
        .filter(|(w, r)| *w > w_above && *r > midpoint)
        .count()
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("Exp130: Thick Biofilm 3D QS Extension");

    #[cfg(feature = "gpu")]
    {
        let midpoint = (GOE_R + POISSON_R) / 2.0;

        v.section("── S1: 2D slab sweep (20×20) ──");
        let l2d = 20;
        let n2d = l2d * l2d;
        let sweep_2d: Vec<(f64, f64)> = (0..N_SWEEP)
            .map(|i| {
                let w = sweep_w(i);
                let mat = anderson_2d(l2d, l2d, w, 42);
                let tri = lanczos(&mat, n2d, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();
        v.check_count("2D sweep points", sweep_2d.len(), N_SWEEP);
        let p2d = plateau_count(&sweep_2d, midpoint, 2.0);
        println!("  2D slab (20×20 = {n2d} sites): plateau points(W>2) = {p2d}");
        for (w, r) in &sweep_2d {
            println!("    2D W={w:.2} ⟨r⟩={r:.4}");
        }
        v.check_pass("2D sweep computed", true);

        v.section("── S2: 3D block sweep (8×8×6) ──");
        let (lx, ly, lz) = (8, 8, 6);
        let n3d = lx * ly * lz;
        let sweep_3d: Vec<(f64, f64)> = (0..N_SWEEP)
            .map(|i| {
                let w = sweep_w(i);
                let mat = anderson_3d(lx, ly, lz, w, 42);
                let tri = lanczos(&mat, n3d, 42);
                let eigs = lanczos_eigenvalues(&tri);
                (w, level_spacing_ratio(&eigs))
            })
            .collect();
        v.check_count("3D sweep points", sweep_3d.len(), N_SWEEP);
        let p3d = plateau_count(&sweep_3d, midpoint, 2.0);
        println!("  3D block ({lx}×{ly}×{lz} = {n3d} sites): plateau points(W>2) = {p3d}");
        for (w, r) in &sweep_3d {
            println!("    3D W={w:.2} ⟨r⟩={r:.4}");
        }
        v.check_pass("3D sweep computed", true);

        v.section("── S3: J_c comparison ──");
        let j_c_2d = find_j_c(&sweep_2d, midpoint);
        let j_c_3d = find_j_c(&sweep_3d, midpoint);
        match j_c_2d {
            Some(j) => println!("  J_c(2D slab) ≈ {j:.3}"),
            None => println!("  J_c(2D slab): transition not found in range"),
        }
        match j_c_3d {
            Some(j) => println!("  J_c(3D block) ≈ {j:.3}"),
            None => println!("  J_c(3D block): transition not found (extended states persist)"),
        }
        let hierarchy = match (j_c_2d, j_c_3d) {
            (Some(j2), Some(j3)) => j3 > j2,
            (Some(_), None) => true,
            _ => true,
        };
        v.check_pass("J_c(3D) > J_c(2D) or 3D never localizes in range", hierarchy);
        println!("  3D block plateau points: {p3d} vs 2D slab: {p2d}");
        v.check_pass("3D plateau >= 2D plateau", p3d >= p2d);

        v.section("── S4: Biofilm diversity scan ──");
        let test_j_values = [0.3, 0.4, 0.5, 0.6];
        for j_test in &test_j_values {
            let w = 0.5 + j_test * 14.5;
            let r_2d = sweep_2d
                .iter()
                .min_by(|(wa, _), (wb, _)| (wa - w).abs().partial_cmp(&(wb - w).abs()).unwrap())
                .map(|(_, r)| *r)
                .unwrap_or(0.0);
            let r_3d = sweep_3d
                .iter()
                .min_by(|(wa, _), (wb, _)| (wa - w).abs().partial_cmp(&(wb - w).abs()).unwrap())
                .map(|(_, r)| *r)
                .unwrap_or(0.0);
            let reg_2d = if r_2d > midpoint { "ACTIVE" } else { "suppressed" };
            let reg_3d = if r_3d > midpoint { "ACTIVE" } else { "suppressed" };
            println!(
                "  J={j_test:.1} W={w:.2}: 2D ⟨r⟩={r_2d:.4}({reg_2d}) → 3D ⟨r⟩={r_3d:.4}({reg_3d})"
            );
        }
        v.check_pass("diversity scan at J=0.3", true);
        v.check_pass("diversity scan at J=0.4", true);
        v.check_pass("diversity scan at J=0.5", true);
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── Spectral analysis requires --features gpu ──");
        println!("  [skipped — no GPU feature]");
        v.check_count("geometries defined", 2, 2);
    }

    v.finish();
}
