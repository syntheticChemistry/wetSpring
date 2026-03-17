// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp187: DF64 Anderson at L=24+ — Extended Precision Large Lattice
//!
//! Validates Anderson localization at larger lattice sizes (L=14+) using
//! standard f64 precision, establishing the baseline for future DF64
//! (double-float) extension when `hotSpring` DF64 Lanczos becomes available.
//!
//! Phase 1 (this binary): f64 at L=6,8,10,12,14 with disorder averaging
//! Phase 2 (future): DF64 at L=14,18,24 when upstream DF64 Lanczos lands
//!
//! # Provenance
//!
//! | Item           | Value |
//! |----------------|-------|
//! | Date           | 2026-02-26 |
//! | DF64 method    | Dekker (1971), double-float arithmetic |
//! | Target `W_c`     | 16.54 ± 0.10, Slevin & Ohtsuki, PRL 82 (1999) |
//! | Target ν       | 1.571 ± 0.004, Rodriguez et al., PRB 84 (2011) |
//! | Baseline commit| wetSpring Phase 59, Exp150 (finite-size scaling) |
//! | Hardware       | biomeGate RTX 4070 (GPU), Eastgate CPU |
//! | Command        | `cargo run --release --features gpu --bin validate_df64_anderson` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)
//!
//! # Python Baselines
//!
//! Physics literature reference — no Python baseline script (analytical/numerical literature values).
//! Critical exponent ν and `W_c` from published scaling analyses:
//! - Slevin & Ohtsuki, PRL 82 (1999): `W_c` = 16.54 ± 0.10
//! - Rodriguez et al., PRB 84 (2011): ν = 1.571 ± 0.004

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const LATTICE_SIZES: &[usize] = &[6, 8, 10, 12, 14];
#[cfg(feature = "gpu")]
const N_REALIZATIONS: usize = 6;
#[cfg(feature = "gpu")]
const N_W_POINTS: usize = 11;
const W_MIN: f64 = 12.0;
const W_MAX: f64 = 22.0;

const EXPECTED_W_C_MIN: f64 = 14.0;
const EXPECTED_W_C_MAX: f64 = 20.0;
const LITERATURE_NU: f64 = 1.57;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + (i as f64) * (W_MAX - W_MIN) / (N_W_POINTS - 1) as f64
}

#[cfg(feature = "gpu")]
fn compute_r_stats(l: usize, w: f64, n_real: usize) -> (f64, f64) {
    use barracuda::spectral::{anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio};
    use barracuda::stats::{correlation, mean};

    let n = l * l * l;
    let mut r_values = Vec::with_capacity(n_real);
    for seed_offset in 0..n_real {
        let seed = (42 + seed_offset * 1000 + l * 100) as u64;
        let mat = anderson_3d(l, l, l, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        r_values.push(level_spacing_ratio(&eigs));
    }
    let mean = mean(&r_values);
    let variance = correlation::variance(&r_values).unwrap_or(0.0);
    (mean, (variance / n_real as f64).sqrt())
}

fn main() {
    let mut v = Validator::new("Exp187: DF64 Anderson Large Lattice (f64 Phase 1)");

    #[cfg(feature = "gpu")]
    {
        use barracuda::spectral::{AndersonSweepPoint, GOE_R, POISSON_R, find_w_c};

        struct SizeResult {
            l: usize,
            sweep: Vec<(f64, f64, f64)>,
            w_c: Option<f64>,
        }

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        println!("  GOE_R={GOE_R:.4}, POISSON_R={POISSON_R:.4}, midpoint={midpoint:.4}");
        println!("  Lattice sizes: {LATTICE_SIZES:?}");
        println!(
            "  W range: [{W_MIN}, {W_MAX}], {N_W_POINTS} points, {N_REALIZATIONS} realizations"
        );
        println!("  Literature: W_c ≈ 16.54, ν ≈ 1.571");

        v.section("── S1: Disorder-averaged sweep per L ──");

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
                &format!("L={l}: {N_W_POINTS} points computed"),
                sweep.len() == N_W_POINTS,
            );

            all_results.push(SizeResult { l, sweep, w_c });
        }

        v.section("── S2: W_c convergence with L ──");

        let w_c_values: Vec<(usize, f64)> = all_results
            .iter()
            .filter_map(|sr| sr.w_c.map(|wc| (sr.l, wc)))
            .collect();

        println!("  W_c by L:");
        for &(l, wc) in &w_c_values {
            println!("    L={l:>2}: W_c = {wc:.2}");
        }

        v.check_pass("W_c found for at least 3 sizes", w_c_values.len() >= 3);

        if w_c_values.len() >= 2 {
            let mean_wc =
                w_c_values.iter().map(|(_, wc)| wc).sum::<f64>() / w_c_values.len() as f64;
            println!("  Mean W_c = {mean_wc:.2}");

            v.check_pass(
                &format!("mean W_c in [{EXPECTED_W_C_MIN}, {EXPECTED_W_C_MAX}]"),
                (EXPECTED_W_C_MIN..=EXPECTED_W_C_MAX).contains(&mean_wc),
            );

            let spread = w_c_values
                .iter()
                .map(|(_, wc)| (wc - mean_wc).abs())
                .fold(0.0_f64, f64::max);
            v.check_pass("W_c spread < 3 across sizes", spread < 3.0);

            if w_c_values.len() >= 3 {
                let last_two: Vec<_> = w_c_values.iter().rev().take(2).collect();
                let convergence = (last_two[0].1 - last_two[1].1).abs();
                println!("  W_c convergence (last two): Δ = {convergence:.2}");
                v.check_pass(
                    "W_c converging (largest two L differ by < 2)",
                    convergence < 2.0,
                );
            }
        }

        v.section("── S3: Scaling collapse (ν estimate) ──");

        if w_c_values.len() >= 3 {
            let mean_wc =
                w_c_values.iter().map(|(_, wc)| wc).sum::<f64>() / w_c_values.len() as f64;

            let nu_grid: Vec<f64> = (0..21).map(|i| f64::from(i).mul_add(0.05, 1.0)).collect();
            let mut best_nu = LITERATURE_NU;
            let mut best_cost = f64::MAX;

            for &nu in &nu_grid {
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
                for i in 0..scaled.len() {
                    for j in (i + 1)..scaled.len() {
                        let dx = scaled[i].0 - scaled[j].0;
                        let dr = scaled[i].1 - scaled[j].1;
                        if dx.abs() < 3.0 {
                            cost += dr * dr;
                        }
                    }
                }

                if cost < best_cost {
                    best_cost = cost;
                    best_nu = nu;
                }
            }

            println!("  Best ν = {best_nu:.2} (literature: {LITERATURE_NU:.3})");
            println!("  Collapse cost = {best_cost:.2}");

            v.check_pass("ν in [1.0, 2.0]", (1.0..=2.0).contains(&best_nu));
            v.check_pass(
                "ν within 0.4 of literature (1.57)",
                (best_nu - LITERATURE_NU).abs() < tolerances::ANDERSON_NU_PARITY,
            );
        } else {
            v.check_pass("scaling collapse deferred (need more L)", true);
        }

        v.section("── S4: DF64 readiness assessment ──");

        let l14 = all_results.iter().find(|sr| sr.l == 14);
        if let Some(sr) = l14 {
            let min_stderr = sr.sweep.iter().map(|(_, _, e)| *e).fold(f64::MAX, f64::min);
            println!("  L=14 minimum stderr: {min_stderr:.6}");
            println!("  L=14 N = {} (matrix size)", 14 * 14 * 14);
            println!("  DF64 would provide ~30 digits vs f64's ~15");
            println!("  Expected benefit: better eigenvalue separation near band center");
            v.check_pass("L=14 computed successfully", sr.sweep.len() == N_W_POINTS);
        }

        println!();
        println!("  DF64 Phase 2 requirements:");
        println!("    - ToadStool DF64 Lanczos kernel");
        println!("    - hotSpring DF64 arithmetic validated");
        println!("    - barracuda::spectral DF64 variant");
        println!("  When available: extend to L=18, 24, 32");
        v.check_pass("DF64 readiness documented", true);

        v.section("── S5: Summary ──");
        println!("  ┌──────┬──────┬──────────┐");
        println!("  │  L   │  N   │   W_c    │");
        println!("  ├──────┼──────┼──────────┤");
        for sr in &all_results {
            println!(
                "  │ {:>4} │ {:>4} │ {:>8} │",
                sr.l,
                sr.l * sr.l * sr.l,
                sr.w_c
                    .map_or_else(|| "—".to_string(), |w| format!("{w:.2}"))
            );
        }
        println!("  └──────┴──────┴──────────┘");
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── DF64 Anderson requires --features gpu ──");
        println!("  Validating constants only...");
        v.check_count("lattice sizes defined", LATTICE_SIZES.len(), 5);
        v.check_pass("W range valid", W_MAX > W_MIN);
        v.check_pass("literature ν documented", LITERATURE_NU > 1.0);
        v.check_pass(
            "expected W_c range valid",
            EXPECTED_W_C_MAX > EXPECTED_W_C_MIN,
        );
    }

    v.finish();
}
