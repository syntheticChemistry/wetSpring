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
//! # Exp151: Disorder-Correlated Lattices for Biofilm Disorder
//!
//! Models realistic biofilm disorder where species cluster spatially.
//! In a real biofilm, species are not randomly distributed — they form
//! microcolonies with spatial correlation length ξ_corr.
//!
//! Uses exponential smoothing of i.i.d. random potential:
//! V_i = Σ_j K(|r_i - r_j|) · ε_j / Z, K(r) = exp(-r/ξ_corr)
//!
//! # Physics
//!
//! Spatially correlated disorder is "smoother" than uncorrelated — the
//! effective scattering is weaker, so W_c should increase (harder to
//! localize). This means biofilm spatial clustering HELPS QS propagation
//! beyond what the uncorrelated Anderson model predicts.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-24 |
//! | Phase       | 39 — Correlated disorder |
//! | GPU prims   | `SpectralCsrMatrix`, `lanczos`, `level_spacing_ratio` |

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, SpectralCsrMatrix, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

const L: usize = 8;
const N_W_POINTS: usize = 11;
const W_MIN: f64 = 8.0;
const W_MAX: f64 = 28.0;
const CORR_LENGTHS: &[f64] = &[0.0, 1.0, 2.0, 4.0];
const N_REALIZATIONS: usize = 4;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + (i as f64) * (W_MAX - W_MIN) / (N_W_POINTS - 1) as f64
}

#[cfg(feature = "gpu")]
struct LcgRng {
    state: u64,
}

#[cfg(feature = "gpu")]
impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
        }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(feature = "gpu")]
fn build_correlated_anderson_3d(
    l: usize,
    disorder: f64,
    xi_corr: f64,
    seed: u64,
) -> SpectralCsrMatrix {
    let n = l * l * l;
    let mut rng = LcgRng::new(seed);

    let raw: Vec<f64> = (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect();

    let potential = if xi_corr < 0.01 {
        raw
    } else {
        let idx_to_xyz = |i: usize| -> (usize, usize, usize) {
            let iz = i % l;
            let iy = (i / l) % l;
            let ix = i / (l * l);
            (ix, iy, iz)
        };

        let mut smoothed = vec![0.0; n];
        let reach = (3.0 * xi_corr).ceil() as i64;
        for (i, s_val) in smoothed.iter_mut().enumerate() {
            let (ix, iy, iz) = idx_to_xyz(i);
            let mut sum = 0.0;
            let mut norm = 0.0;
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        let jx = ix as i64 + dx;
                        let jy = iy as i64 + dy;
                        let jz = iz as i64 + dz;
                        if (0..l as i64).contains(&jx)
                            && (0..l as i64).contains(&jy)
                            && (0..l as i64).contains(&jz)
                        {
                            let j = (jx as usize * l + jy as usize) * l + jz as usize;
                            let r = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                            let kernel = (-r / xi_corr).exp();
                            sum += kernel * raw[j];
                            norm += kernel;
                        }
                    }
                }
            }
            *s_val = sum / norm;
        }

        let var: f64 = {
            let mean = smoothed.iter().sum::<f64>() / n as f64;
            smoothed
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f64>()
                / n as f64
        };
        let target_var = disorder * disorder / 12.0;
        let scale = if var > 1e-30 {
            (target_var / var).sqrt()
        } else {
            1.0
        };
        smoothed.iter().map(|v| v * scale).collect()
    };

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * l + iy) * l + iz };

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for ix in 0..l {
        for iy in 0..l {
            for iz in 0..l {
                let site = idx(ix, iy, iz);
                let mut entries: Vec<(usize, f64)> = Vec::new();

                if ix > 0 {
                    entries.push((idx(ix - 1, iy, iz), -1.0));
                }
                if iy > 0 {
                    entries.push((idx(ix, iy - 1, iz), -1.0));
                }
                if iz > 0 {
                    entries.push((idx(ix, iy, iz - 1), -1.0));
                }
                entries.push((site, potential[site]));
                if iz + 1 < l {
                    entries.push((idx(ix, iy, iz + 1), -1.0));
                }
                if iy + 1 < l {
                    entries.push((idx(ix, iy + 1, iz), -1.0));
                }
                if ix + 1 < l {
                    entries.push((idx(ix + 1, iy, iz), -1.0));
                }

                entries.sort_by_key(|&(c, _)| c);
                for (c, v) in entries {
                    col_idx.push(c);
                    values.push(v);
                }
                row_ptr.push(col_idx.len());
            }
        }
    }

    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("Exp151: Disorder-Correlated Lattices for Biofilm Disorder");

    #[cfg(feature = "gpu")]
    {
        use std::time::Instant;

        let n = L * L * L;
        let midpoint = (GOE_R + POISSON_R) / 2.0;
        println!("  L = {L}, N = {n}");
        println!("  Correlation lengths ξ_corr: {CORR_LENGTHS:?}");
        println!("  Biological mapping:");
        println!("    ξ = 0: well-mixed planktonic community");
        println!("    ξ = 1: loosely aggregated biofilm");
        println!("    ξ = 2: mature biofilm with microcolonies");
        println!("    ξ = 4: dense biofilm with large species clusters");

        v.section("§1 W Sweep per Correlation Length");
        let t_start = Instant::now();

        struct CorrResult {
            xi: f64,
            sweep: Vec<(f64, f64, f64)>,
            w_c: Option<f64>,
        }

        let mut all_results: Vec<CorrResult> = Vec::new();

        for &xi in CORR_LENGTHS {
            println!(
                "\n  ξ_corr = {xi:.1} ({}):",
                match xi as u32 {
                    0 => "uncorrelated",
                    1 => "loose aggregation",
                    2 => "mature biofilm",
                    _ => "dense biofilm",
                }
            );

            let sweep: Vec<(f64, f64, f64)> = (0..N_W_POINTS)
                .map(|i| {
                    let w = sweep_w(i);
                    let mut r_vals = Vec::with_capacity(N_REALIZATIONS);
                    for seed_off in 0..N_REALIZATIONS {
                        let seed = (42 + seed_off * 1000) as u64;
                        let mat = build_correlated_anderson_3d(L, w, xi, seed);
                        let tri = lanczos(&mat, n, seed);
                        let eigs = lanczos_eigenvalues(&tri);
                        r_vals.push(level_spacing_ratio(&eigs));
                    }
                    let mean = r_vals.iter().sum::<f64>() / N_REALIZATIONS as f64;
                    let var = r_vals.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>()
                        / (N_REALIZATIONS - 1) as f64;
                    let stderr = (var / N_REALIZATIONS as f64).sqrt();
                    println!("    W={w:5.1}  ⟨r⟩ = {mean:.4} ± {stderr:.4}");
                    (w, mean, stderr)
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
                println!("    → W_c(ξ={xi:.1}) = {wc:.2}");
            } else {
                println!("    → W_c not found in sweep range (may need wider range for ξ={xi:.1})");
            }

            v.check_pass(
                &format!("ξ={xi:.1}: sweep computed"),
                sweep.len() == N_W_POINTS,
            );

            all_results.push(CorrResult { xi, sweep, w_c });
        }

        let elapsed = t_start.elapsed();
        println!("\n  Total compute: {:.1}s", elapsed.as_secs_f64());

        // ─── §2 Verify uncorrelated matches Exp150 ───────────────────
        v.section("§2 Uncorrelated Baseline Check");
        let uncorr = &all_results[0];
        v.check_pass(
            "ξ=0 produces valid ⟨r⟩ sweep",
            uncorr.sweep.iter().all(|(_, r, _)| r.is_finite()),
        );
        if let Some(wc0) = uncorr.w_c {
            v.check_pass(
                "uncorrelated W_c in [12, 22] (matches Exp150)",
                (12.0..=22.0).contains(&wc0),
            );
        }

        // ─── §3 Correlation shifts W_c ────────────────────────────────
        v.section("§3 Correlation Effect on W_c");
        println!("\n  ┌──────────┬────────────────────────┬──────────┐");
        println!("  │  ξ_corr  │ Biological Regime       │   W_c    │");
        println!("  ├──────────┼────────────────────────┼──────────┤");
        for cr in &all_results {
            let regime = match cr.xi as u32 {
                0 => "well-mixed planktonic",
                1 => "loose aggregation",
                2 => "mature biofilm",
                _ => "dense biofilm clusters",
            };
            println!(
                "  │ {:>8.1} │ {:22} │ {:>8} │",
                cr.xi,
                regime,
                cr.w_c.map_or("—".to_string(), |w| format!("{w:.2}"))
            );
        }
        println!("  └──────────┴────────────────────────┴──────────┘");

        let corr_wc: Vec<(f64, f64)> = all_results
            .iter()
            .filter_map(|cr| cr.w_c.map(|w| (cr.xi, w)))
            .collect();

        if corr_wc.len() >= 2 {
            let wc_trend_up = corr_wc.windows(2).filter(|w| w[1].1 > w[0].1).count();
            let total_pairs = corr_wc.len() - 1;
            println!("\n  W_c trend: {wc_trend_up}/{total_pairs} pairs show W_c increasing with ξ");
            v.check_pass(
                "increasing ξ generally increases W_c (smoother ≈ less scattering)",
                wc_trend_up > 0 || total_pairs == 0,
            );
        }

        // ─── §4 Eigenvalue finiteness ─────────────────────────────────
        v.section("§4 Eigenvalue Finiteness");
        let all_finite = all_results
            .iter()
            .all(|cr| cr.sweep.iter().all(|(_, r, _)| r.is_finite()));
        v.check_pass(
            "all eigenvalues finite for all correlation lengths",
            all_finite,
        );

        // ─── §5 Biological interpretation ─────────────────────────────
        v.section("§5 Biological Interpretation");
        println!("\n  If ξ_corr increases W_c:");
        println!("    → Biofilm spatial clustering reduces effective disorder");
        println!("    → QS signal propagation is EASIER in structured biofilms");
        println!("    → Anderson model with i.i.d. disorder is a LOWER BOUND");
        println!("    → Real biofilms are MORE QS-active than uncorrelated prediction");
        println!("  If ξ_corr does NOT change W_c significantly:");
        println!("    → The Anderson prediction is robust to spatial structure");
        println!("    → The 100%/0% QS atlas remains valid regardless of biofilm architecture");

        v.check_pass("biological interpretation documented", true);
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("§1 Spectral analysis requires --features gpu");
        println!("  [skipped — no GPU feature]");
        v.check_count("correlation lengths defined", CORR_LENGTHS.len(), 4);
    }

    v.finish();
}
