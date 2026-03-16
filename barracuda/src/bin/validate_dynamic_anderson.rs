// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp186: Dynamic Anderson W(t) — Community Evolution Under Perturbation
//!
//! Models how microbial community diversity evolves under perturbation
//! using time-varying Anderson disorder W(t). Three scenarios:
//!   - S1: Tillage → no-till transition (exponential W decay)
//!   - S2: Antibiotic perturbation (spike + recovery)
//!   - S3: Seasonal cycle (sinusoidal W)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Source | Analytical (closed-form W(t) models) |
//! | Date | 2026-02-26 |
//! | Theory | Anderson (1958), time-dependent generalization; seasonal oscillation W(t) = W₀ + A·sin(2πt/365) |
//! | Soil data | Islam et al. (2014), Brandt farm 15-year study |
//! | Antibiotic | Dethlefsen & Relman, PNAS 108 (2011) |
//! | Seasonal | Lauber et al., Applied Env. Microbiol. 75 (2009) |
//! | Baseline commit | wetSpring Phase 59 |
//! | Hardware | biomeGate RTX 4070 (GPU), Eastgate CPU |
//! | Command | `cargo run --release --features gpu --bin validate_dynamic_anderson` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const N_REALIZATIONS: usize = 4;

fn w_tillage(t: f64) -> f64 {
    let tau = 3.0;
    let w_inf = 12.0;
    (20.0_f64 - w_inf).mul_add((-t / tau).exp(), w_inf)
}

fn w_antibiotic(t: f64) -> f64 {
    let w_healthy = 14.0;
    let w_ab = 25.0;
    let tau_recover = 14.0;
    if t < 0.0 {
        w_healthy
    } else if t <= 7.0 {
        let frac = t / 7.0;
        w_healthy + frac * (w_ab - w_healthy)
    } else {
        let dt = t - 7.0;
        (w_ab - w_healthy).mul_add((-dt / tau_recover).exp(), w_healthy)
    }
}

fn w_seasonal(t: f64) -> f64 {
    let w_0 = 16.0;
    let amplitude = 4.0;
    w_0 + amplitude * (2.0 * std::f64::consts::PI * t / 365.0).sin()
}

#[cfg(feature = "gpu")]
fn compute_r_at_w(l: usize, w: f64, n_real: usize) -> (f64, f64) {
    use barracuda::spectral::{anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio};
    use barracuda::stats::{correlation, mean};

    let n = l * l * l;
    let mut r_values = Vec::with_capacity(n_real);
    for seed_offset in 0..n_real {
        let seed = (42 + seed_offset * 1000) as u64;
        let mat = anderson_3d(l, l, l, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        r_values.push(level_spacing_ratio(&eigs));
    }
    let mean = mean(&r_values);
    let variance = correlation::variance(&r_values).unwrap_or(0.0);
    let stderr = (variance / n_real as f64).sqrt();
    (mean, stderr)
}

fn main() {
    let mut v = Validator::new("Exp186: Dynamic Anderson W(t) — Community Evolution");

    #[cfg(feature = "gpu")]
    {
        use barracuda::spectral::{GOE_R, POISSON_R};
use wetspring_barracuda::validation::OrExit;

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let l = 8;

        println!("  GOE_R={GOE_R:.4}, POISSON_R={POISSON_R:.4}, midpoint={midpoint:.4}");
        println!("  Lattice L={l}, realizations={N_REALIZATIONS}");

        // ─── S1: Tillage → No-Till Transition ────────────────────────
        v.section("── S1: Tillage → No-Till Transition ──");

        let t_points: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let mut trajectory: Vec<(f64, f64, f64, f64)> = Vec::new();

        for &t in &t_points {
            let w = w_tillage(t);
            let (r_mean, r_err) = compute_r_at_w(l, w, N_REALIZATIONS);
            trajectory.push((t, w, r_mean, r_err));
        }

        println!("  {:>6} {:>8} {:>8} {:>8}", "t(yr)", "W(t)", "⟨r⟩", "±σ");
        for &(t, w, r, err) in &trajectory {
            let marker = if r > midpoint { "●" } else { "○" };
            println!("  {t:>6.1} {w:>8.2} {r:>8.4} {err:>8.4}  {marker}");
        }

        let r_start = trajectory.first().or_exit("unexpected error").2;
        let r_end = trajectory.last().or_exit("unexpected error").2;

        v.check_pass("r(t=0) < midpoint (localized start)", r_start < midpoint);
        v.check_pass("r(t=∞) > midpoint (extended end)", r_end > midpoint);

        let crossing_idx = trajectory
            .windows(2)
            .position(|w| w[0].2 < midpoint && w[1].2 >= midpoint);
        if let Some(idx) = crossing_idx {
            let t_cross = trajectory[idx].0;
            println!("  W_c crossing at t ≈ {t_cross:.1} years");
            v.check_pass(
                "transition time within plausible range",
                (0.5..=8.0).contains(&t_cross),
            );
        } else {
            v.check_pass(
                "W_c crossing detected (or already extended)",
                r_end > midpoint,
            );
        }

        v.check_pass("r increases over time (recovery)", r_end > r_start);

        // ─── S2: Antibiotic Perturbation ─────────────────────────────
        v.section("── S2: Antibiotic Perturbation ──");

        let ab_t: Vec<f64> = (0..30).map(|i| f64::from(i) * 2.0).collect();
        let mut ab_traj: Vec<(f64, f64, f64, f64)> = Vec::new();

        for &t in &ab_t {
            let w = w_antibiotic(t);
            let (r_mean, r_err) = compute_r_at_w(l, w, N_REALIZATIONS);
            ab_traj.push((t, w, r_mean, r_err));
        }

        println!("  {:>6} {:>8} {:>8} {:>8}", "t(day)", "W(t)", "⟨r⟩", "±σ");
        for &(t, w, r, err) in &ab_traj {
            let marker = if r > midpoint { "●" } else { "○" };
            println!("  {t:>6.0} {w:>8.2} {r:>8.4} {err:>8.4}  {marker}");
        }

        let r_pre = ab_traj.first().or_exit("unexpected error").2;
        let r_during = ab_traj
            .iter()
            .filter(|(t, _, _, _)| (3.0..=10.0).contains(t))
            .map(|(_, _, r, _)| *r)
            .fold(f64::MAX, f64::min);
        let r_post = ab_traj.last().or_exit("unexpected error").2;

        v.check_pass("r drops during antibiotic treatment", r_during < r_pre);
        v.check_pass("r recovers after treatment", r_post > r_during);

        let recovery_point = ab_traj
            .iter()
            .skip_while(|(t, _, _, _)| *t <= 7.0)
            .find(|(_, _, r, _)| *r > midpoint);
        if let Some((t_rec, _, _, _)) = recovery_point {
            println!("  Recovery to extended at t ≈ {t_rec:.0} days");
            v.check_pass(
                "recovery within 60 days of antibiotic end",
                *t_rec - 7.0 <= 60.0,
            );
        } else {
            v.check_pass("recovery trajectory trending upward", r_post > r_during);
        }

        // ─── S3: Seasonal Cycle ──────────────────────────────────────
        v.section("── S3: Seasonal Cycle ──");

        let seasonal_t: Vec<f64> = (0..24).map(|i| f64::from(i) * 30.0).collect();
        let mut seasonal_traj: Vec<(f64, f64, f64, f64)> = Vec::new();

        for &t in &seasonal_t {
            let w = w_seasonal(t);
            let (r_mean, r_err) = compute_r_at_w(l, w, N_REALIZATIONS);
            seasonal_traj.push((t, w, r_mean, r_err));
        }

        println!("  {:>6} {:>8} {:>8} {:>8}", "t(day)", "W(t)", "⟨r⟩", "±σ");
        for &(t, w, r, err) in &seasonal_traj {
            let marker = if r > midpoint { "●" } else { "○" };
            println!("  {t:>6.0} {w:>8.2} {r:>8.4} {err:>8.4}  {marker}");
        }

        let r_min = seasonal_traj
            .iter()
            .map(|(_, _, r, _)| *r)
            .fold(f64::MAX, f64::min);
        let r_max = seasonal_traj
            .iter()
            .map(|(_, _, r, _)| *r)
            .fold(f64::MIN, f64::max);

        println!("  r range: [{r_min:.4}, {r_max:.4}]");

        v.check_pass(
            "r oscillates (range > DYNAMIC_WT_EXACT)",
            r_max - r_min > tolerances::DYNAMIC_WT_EXACT,
        );
        v.check_pass(
            "r range brackets midpoint (W_0 ≈ W_c)",
            r_min < midpoint && r_max > midpoint,
        );

        let first_half: f64 = seasonal_traj
            .iter()
            .take(12)
            .map(|(_, _, r, _)| *r)
            .sum::<f64>()
            / 12.0;
        let second_half: f64 = seasonal_traj
            .iter()
            .skip(12)
            .map(|(_, _, r, _)| *r)
            .sum::<f64>()
            / 12.0;
        let periodicity = (first_half - second_half).abs() < tolerances::DYNAMIC_WT_PERIODICITY;
        v.check_pass(
            "approximate yearly periodicity (half means within 0.05)",
            periodicity,
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── Dynamic Anderson requires --features gpu ──");
        println!("  Validating W(t) functions only...");

        v.check_pass(
            "tillage W(0) = 20",
            (w_tillage(0.0) - 20.0).abs() < tolerances::DYNAMIC_WT_EXACT,
        );
        v.check_pass(
            "tillage W(∞) ≈ 12",
            (w_tillage(100.0) - 12.0).abs() < tolerances::DYNAMIC_WT_ASYMPTOTIC,
        );
        v.check_pass(
            "antibiotic W(-1) = 14",
            (w_antibiotic(-1.0) - 14.0).abs() < tolerances::DYNAMIC_WT_EXACT,
        );
        v.check_pass("antibiotic W(3.5) > 14", w_antibiotic(3.5) > 14.0);
        v.check_pass(
            "antibiotic W(100) ≈ 14",
            (w_antibiotic(100.0) - 14.0).abs() < tolerances::DYNAMIC_WT_ASYMPTOTIC,
        );
        v.check_pass(
            "seasonal W(0) = 16",
            (w_seasonal(0.0) - 16.0).abs() < tolerances::DYNAMIC_WT_EXACT,
        );
        v.check_pass(
            "seasonal W(91.25) ≈ 20",
            (w_seasonal(91.25) - 20.0).abs() < tolerances::SEASONAL_OSCILLATION,
        );
    }

    v.finish();
}
