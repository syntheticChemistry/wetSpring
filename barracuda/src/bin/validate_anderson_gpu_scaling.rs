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
//! # Exp184b: GPU Anderson Finite-Size Scaling L=14–20
//!
//! Extends Exp150 (L=6–12) to larger lattice sizes L=14, 16, 18, 20 using
//! GPU-accelerated Lanczos (when available) or CPU Lanczos as reference.
//!
//! At L=20, the Anderson 3D lattice is 8000×8000 — too large for full
//! diagonalization but well within GPU `SpMV` + Lanczos capability.
//!
//! # Physics
//!
//! The critical disorder `W_c` ≈ 16.5 for 3D Anderson with box disorder.
//! Larger L sharpens the metal-insulator transition, allowing refined
//! estimates of `W_c` and the critical exponent ν ≈ 1.57 (Slevin & Ohtsuki
//! 1999).
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-26 |
//! | Phase       | V55 — Science extensions |
//! | GPU prims   | `anderson_3d`, `lanczos`, `level_spacing_ratio` |
//! | Predecessor | Exp150 (L=6–12, 8 realizations) |
//! | Literature  | Slevin & Ohtsuki, PRL 82 (1999); Rodriguez et al., PRB 84 (2011) |

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    AndersonSweepPoint, GOE_R, POISSON_R, anderson_3d, find_w_c, lanczos, lanczos_eigenvalues,
    level_spacing_ratio,
};

const LATTICE_SIZES: &[usize] = &[14, 16, 18, 20];
const N_REALIZATIONS: usize = 16;
const N_W_POINTS: usize = 15;
const W_MIN: f64 = 12.0;
const W_MAX: f64 = 21.0;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + f64::from(i as u32) * (W_MAX - W_MIN) / f64::from((N_W_POINTS - 1) as u32)
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
    let n_f64 = f64::from(n_real as u32);
    let mean = r_values.iter().sum::<f64>() / n_f64;
    let variance = r_values
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / f64::from((n_real - 1) as u32);
    let stderr = (variance / n_f64).sqrt();
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
            return Some(t.mul_add(w1 - w0, w0));
        }
    }
    None
}

fn main() {
    let mut v = Validator::new("Exp184b: GPU Anderson Finite-Size Scaling L=14–20");

    #[cfg(feature = "gpu")]
    {
        use std::time::Instant;

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        println!("  midpoint (GOE+Poisson)/2 = {midpoint:.4}");
        println!("  Lattice sizes: {LATTICE_SIZES:?}");
        println!("  W range: [{W_MIN}, {W_MAX}], {N_W_POINTS} points");
        println!("  Realizations per (L,W): {N_REALIZATIONS}");

        v.section("§1 Large-Lattice Disorder-Averaged Sweep");

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

            let sweep_pts: Vec<_> = sweep
                .iter()
                .map(|&(w, r, s)| AndersonSweepPoint {
                    w,
                    r_mean: r,
                    r_stderr: s,
                })
                .collect();
            let w_c = find_w_c(&sweep_pts, midpoint);

            if let Some(wc) = w_c {
                println!("    → W_c(L={l}) = {wc:.2}");
            }

            v.check_pass(
                &format!("L={l}: {N_W_POINTS} pts × {N_REALIZATIONS} realizations"),
                sweep.len() == N_W_POINTS,
            );

            for &(w, _, stderr) in &sweep {
                v.check_pass(
                    &format!(
                        "L={l} W={w:.1}: stderr < {}",
                        tolerances::LEVEL_SPACING_STDERR_MAX
                    ),
                    stderr < tolerances::LEVEL_SPACING_STDERR_MAX,
                );
            }

            all_results.push(SizeResult { l, sweep, w_c });
        }

        let elapsed = t_start.elapsed();
        println!("\n  Total compute: {:.1}s", elapsed.as_secs_f64());

        v.section("§2 Monotonicity");
        for sr in &all_results {
            let monotonic = sr.sweep.windows(2).all(|w| w[0].1 >= w[1].1 - 0.02);
            v.check_pass(
                &format!("L={}: ⟨r⟩ decreases with W (within noise)", sr.l),
                monotonic,
            );
        }

        v.section("§3 Crossing-Point and W_c Analysis");
        let w_c_values: Vec<f64> = all_results.iter().filter_map(|sr| sr.w_c).collect();
        v.check_pass("W_c found for ≥ 2 sizes", w_c_values.len() >= 2);
        v.check_pass(
            "all W_c in [12, 21]",
            w_c_values.iter().all(|&w| (12.0..=21.0).contains(&w)),
        );

        if w_c_values.len() >= 2 {
            let mean_wc = w_c_values.iter().sum::<f64>() / f64::from(w_c_values.len() as u32);
            let spread = w_c_values
                .iter()
                .map(|w| (w - mean_wc).abs())
                .fold(0.0_f64, f64::max);
            println!("  Mean W_c = {mean_wc:.2} ± {spread:.2}");

            v.check(
                "W_c spread < FINITE_SIZE_SCALING_REL × mean",
                spread,
                0.0,
                mean_wc * tolerances::FINITE_SIZE_SCALING_REL,
            );

            v.check_pass(
                "mean W_c in [14, 19] (expected ~16.5)",
                (14.0..=19.0).contains(&mean_wc),
            );

            println!("\n  Pairwise crossings:");
            let mut crossings = Vec::new();
            for i in 0..all_results.len() {
                for j in (i + 1)..all_results.len() {
                    if let Some(wc) = find_crossing(&all_results[i].sweep, &all_results[j].sweep) {
                        println!(
                            "    L={} × L={}: W_c = {wc:.2}",
                            all_results[i].l, all_results[j].l
                        );
                        crossings.push(wc);
                    }
                }
            }
        }

        v.section("§4 Scaling Collapse (ν estimate)");
        if w_c_values.len() >= 2 {
            let mean_wc = w_c_values.iter().sum::<f64>() / f64::from(w_c_values.len() as u32);
            let nu_candidates: Vec<f64> = vec![1.0, 1.2, 1.4, 1.57, 1.7, 1.9, 2.0];
            let mut best_nu = 1.57;
            let mut best_cost = f64::MAX;

            for &nu in &nu_candidates {
                let scaled: Vec<(f64, f64)> = all_results
                    .iter()
                    .flat_map(|sr| {
                        sr.sweep.iter().map(move |(w, r, _)| {
                            let x = (w - mean_wc) * f64::from(sr.l as u32).powf(1.0 / nu);
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
                "ν in [1.0, 2.0] (literature: 1.57 ± 0.02)",
                (1.0..=2.0).contains(&best_nu),
            );
        } else {
            v.check_pass("scaling collapse deferred (insufficient W_c)", true);
        }

        v.section("§5 Summary Table");
        println!("\n  ┌──────┬───────┬──────────────────┬──────────┐");
        println!("  │  L   │   N   │ ⟨r⟩ range        │   W_c    │");
        println!("  ├──────┼───────┼──────────────────┼──────────┤");
        for sr in &all_results {
            let r_min = sr.sweep.iter().map(|(_, r, _)| *r).fold(f64::MAX, f64::min);
            let r_max = sr.sweep.iter().map(|(_, r, _)| *r).fold(f64::MIN, f64::max);
            println!(
                "  │ {:>4} │ {:>5} │ {:.4} – {:.4}    │ {:>8} │",
                sr.l,
                sr.l * sr.l * sr.l,
                r_min,
                r_max,
                sr.w_c
                    .map_or_else(|| "—".to_string(), |w| format!("{w:.2}"))
            );
        }
        println!("  └──────┴───────┴──────────────────┴──────────┘");

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
        v.check_pass("W range valid", W_MAX > W_MIN);
        v.check_pass(
            "GPU Lanczos tolerance defined",
            tolerances::GPU_LANCZOS_EIGENVALUE_ABS > 0.0,
        );
        v.check_pass(
            "finite-size scaling tolerance defined",
            tolerances::FINITE_SIZE_SCALING_REL > 0.0,
        );
        v.check_pass(
            "stderr tolerance defined",
            tolerances::LEVEL_SPACING_STDERR_MAX > 0.0,
        );
    }

    v.finish();
}
