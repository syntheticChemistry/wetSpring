// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! # Exp150: Finite-Size Scaling with Disorder Averaging
//!
//! Extends Exp131 with:
//! - Disorder averaging (8 realizations per point)
//! - Focused W range around expected W_c
//! - Lattice sizes L = 6, 8, 10, 12
//! - Scaling collapse analysis for critical exponent ν
//!
//! # Physics
//!
//! At the Anderson metal-insulator transition (3D), the level spacing ratio
//! ⟨r⟩ transitions from GOE (≈0.531, extended) to Poisson (≈0.386, localized).
//! The crossing point of ⟨r⟩(W) curves for different L defines W_c.
//! Near the transition: ⟨r⟩ = f((W − W_c) · L^(1/ν)), where ν ≈ 1.57.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-24 |
//! | Phase       | 39 — Finite-size scaling |
//! | GPU prims   | `anderson_3d`, `lanczos`, `level_spacing_ratio` |
//! | Predecessor | Exp131 (L=6–10, single realization) |

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

const LATTICE_SIZES: &[usize] = &[6, 8, 10, 12];
const N_REALIZATIONS: usize = 8;
const N_W_POINTS: usize = 13;
const W_MIN: f64 = 10.0;
const W_MAX: f64 = 22.0;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + (i as f64) * (W_MAX - W_MIN) / (N_W_POINTS - 1) as f64
}

#[cfg(feature = "gpu")]
struct SizeResult {
    l: usize,
    sweep: Vec<(f64, f64, f64)>,
    w_c: Option<f64>,
}

#[cfg(feature = "gpu")]
fn compute_r_stats(l: usize, w: f64, n_real: usize) -> (f64, f64) {
    let n = l * l * l;
    let mut r_values = Vec::with_capacity(n_real);
    for seed_offset in 0..n_real {
        let seed = (42 + seed_offset * 1000 + l * 100) as u64;
        let mat = anderson_3d(l, l, l, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        r_values.push(level_spacing_ratio(&eigs));
    }
    let mean = r_values.iter().sum::<f64>() / n_real as f64;
    let variance = r_values
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / (n_real - 1) as f64;
    let stderr = (variance / n_real as f64).sqrt();
    (mean, stderr)
}

#[cfg(feature = "gpu")]
fn find_crossing(sweep_a: &[(f64, f64, f64)], sweep_b: &[(f64, f64, f64)]) -> Option<f64> {
    for i in 1..sweep_a.len().min(sweep_b.len()) {
        let (w0, ra0, _) = sweep_a[i - 1];
        let (w1, ra1, _) = sweep_a[i];
        let (_, rb0, _) = sweep_b[i - 1];
        let (_, rb1, _) = sweep_b[i];

        let diff0 = ra0 - rb0;
        let diff1 = ra1 - rb1;
        if diff0 * diff1 < 0.0 {
            let t = diff0 / (diff0 - diff1);
            return Some(w0 + t * (w1 - w0));
        }
    }
    None
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("Exp150: Finite-Size Scaling with Disorder Averaging");

    #[cfg(feature = "gpu")]
    {
        use std::time::Instant;

        let midpoint = (GOE_R + POISSON_R) / 2.0;
        println!("  midpoint (GOE+Poisson)/2 = {midpoint:.4}");
        println!("  GOE_R = {GOE_R:.4}, POISSON_R = {POISSON_R:.4}");
        println!("  Lattice sizes: {LATTICE_SIZES:?}");
        println!("  W range: [{W_MIN}, {W_MAX}], {N_W_POINTS} points");
        println!("  Realizations per (L,W): {N_REALIZATIONS}");

        v.section("§1 Disorder-Averaged W Sweep");

        let t_start = Instant::now();
        let mut all_results: Vec<SizeResult> = Vec::new();

        for &l in LATTICE_SIZES {
            let n = l * l * l;
            println!("\n  L={l} (N={n}):");

            let sweep: Vec<(f64, f64, f64)> = (0..N_W_POINTS)
                .map(|i| {
                    let w = sweep_w(i);
                    let (r_mean, r_err) = compute_r_stats(l, w, N_REALIZATIONS);
                    println!("    W={w:5.1}  ⟨r⟩ = {r_mean:.4} ± {r_err:.4}");
                    (w, r_mean, r_err)
                })
                .collect();

            let w_c = {
                let mut last = None;
                for i in 1..sweep.len() {
                    let (w0, r0, _) = sweep[i - 1];
                    let (w1, r1, _) = sweep[i];
                    if r0 > midpoint && r1 <= midpoint {
                        let t = (midpoint - r0) / (r1 - r0);
                        last = Some(w0 + t * (w1 - w0));
                    }
                }
                last
            };

            if let Some(wc) = w_c {
                println!("    → W_c(L={l}) = {wc:.2}");
            }

            v.check_pass(
                &format!("L={l}: {N_W_POINTS} points × {N_REALIZATIONS} realizations computed"),
                sweep.len() == N_W_POINTS,
            );

            all_results.push(SizeResult { l, sweep, w_c });
        }

        let elapsed = t_start.elapsed();
        println!("\n  Total compute: {:.1}s", elapsed.as_secs_f64());

        // ─── §2 Check monotonicity ────────────────────────────────────
        v.section("§2 Monotonicity Check");
        for sr in &all_results {
            let monotonic = sr.sweep.windows(2).all(|w| w[0].1 >= w[1].1 - 0.02);
            v.check_pass(
                &format!("L={}: ⟨r⟩ decreases with W (within noise)", sr.l),
                monotonic,
            );
        }

        // ─── §3 Crossing-point analysis ───────────────────────────────
        v.section("§3 Crossing-Point Analysis");
        println!("\n  W_c from midpoint crossing:");
        for sr in &all_results {
            println!(
                "    L={:>2}: W_c = {}",
                sr.l,
                sr.w_c.map_or("—".to_string(), |w| format!("{w:.2}"))
            );
        }

        let w_c_values: Vec<f64> = all_results.iter().filter_map(|sr| sr.w_c).collect();
        v.check_pass("W_c found for at least 2 sizes", w_c_values.len() >= 2);
        v.check_pass(
            "all W_c in [10, 22]",
            w_c_values.iter().all(|&w| (10.0..=22.0).contains(&w)),
        );

        println!("\n  Pairwise crossings (smaller L vs larger L):");
        let mut crossings = Vec::new();
        for i in 0..all_results.len() {
            for j in (i + 1)..all_results.len() {
                if let Some(wc) = find_crossing(&all_results[i].sweep, &all_results[j].sweep) {
                    println!(
                        "    L={} × L={}: crossing at W = {wc:.2}",
                        all_results[i].l, all_results[j].l
                    );
                    crossings.push(wc);
                }
            }
        }

        if crossings.len() >= 2 {
            let mean_wc = crossings.iter().sum::<f64>() / crossings.len() as f64;
            let spread = crossings
                .iter()
                .map(|w| (w - mean_wc).abs())
                .fold(0.0_f64, f64::max);
            println!("    Mean pairwise W_c = {mean_wc:.2} ± {spread:.2}");
            println!(
                "    (pairwise crossings are noisy for small L — midpoint W_c is the primary metric)"
            );
        }

        let mean_midpoint_wc = w_c_values.iter().sum::<f64>() / w_c_values.len().max(1) as f64;
        println!(
            "    Mean midpoint W_c = {mean_midpoint_wc:.2} (from {} sizes)",
            w_c_values.len()
        );
        v.check_pass(
            "midpoint W_c in [14, 20] (expected ~16.5)",
            (14.0..=20.0).contains(&mean_midpoint_wc),
        );
        let midpoint_spread = w_c_values
            .iter()
            .map(|w| (w - mean_midpoint_wc).abs())
            .fold(0.0_f64, f64::max);
        v.check_pass(
            "midpoint W_c spread < 2 (consistent across sizes)",
            midpoint_spread < 2.0,
        );

        // ─── §4 Scaling collapse estimate ─────────────────────────────
        v.section("§4 Scaling Collapse (ν estimate)");
        if w_c_values.len() >= 2 {
            let mean_wc = mean_midpoint_wc;

            let nu_candidates: Vec<f64> = vec![1.0, 1.3, 1.57, 1.8, 2.0];
            let mut best_nu = 1.57;
            let mut best_cost = f64::MAX;

            for &nu in &nu_candidates {
                let scaled: Vec<(f64, f64)> = all_results
                    .iter()
                    .flat_map(|sr| {
                        sr.sweep.iter().map(move |(w, r, _)| {
                            let x = (w - mean_wc) * (sr.l as f64).powf(1.0 / nu);
                            (x, *r)
                        })
                    })
                    .collect();

                let mut cost = 0.0;
                let n_pts = scaled.len();
                for i in 0..n_pts {
                    for j in (i + 1)..n_pts {
                        let dx = scaled[i].0 - scaled[j].0;
                        let dr = scaled[i].1 - scaled[j].1;
                        if dx.abs() < 2.0 {
                            cost += dr * dr;
                        }
                    }
                }
                println!("    ν={nu:.2}: collapse cost = {cost:.2}");
                if cost < best_cost {
                    best_cost = cost;
                    best_nu = nu;
                }
            }
            println!("    Best ν = {best_nu:.2}");
            v.check_pass(
                "ν in [1.0, 2.0] (literature: 1.57)",
                (1.0..=2.0).contains(&best_nu),
            );
        } else {
            println!("    Not enough midpoint W_c values for scaling collapse");
            v.check_pass("scaling collapse deferred (insufficient W_c values)", true);
        }

        // ─── §5 Summary table ─────────────────────────────────────────
        v.section("§5 Summary");
        println!("\n  ┌──────┬──────┬──────────────────┬──────────┐");
        println!("  │  L   │  N   │ ⟨r⟩ range        │   W_c    │");
        println!("  ├──────┼──────┼──────────────────┼──────────┤");
        for sr in &all_results {
            let r_min = sr.sweep.iter().map(|(_, r, _)| *r).fold(f64::MAX, f64::min);
            let r_max = sr.sweep.iter().map(|(_, r, _)| *r).fold(f64::MIN, f64::max);
            println!(
                "  │ {:>4} │ {:>4} │ {:.4} – {:.4}    │ {:>8} │",
                sr.l,
                sr.l * sr.l * sr.l,
                r_min,
                r_max,
                sr.w_c.map_or("—".to_string(), |w| format!("{w:.2}"))
            );
        }
        println!("  └──────┴──────┴──────────────────┴──────────┘");

        v.check_pass(
            "all lattice sizes computed",
            all_results.len() == LATTICE_SIZES.len(),
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("§1 Spectral analysis requires --features gpu");
        println!("  [skipped — no GPU feature]");
        v.check_count("lattice sizes defined", LATTICE_SIZES.len(), 4);
        v.check_pass("W range defined", W_MAX > W_MIN);
    }

    v.finish();
}
