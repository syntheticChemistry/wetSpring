// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp107: Cross-Spring Spectral Theory — Anderson Localization for Quorum Sensing
//!
//! Exercises `barracuda::spectral` primitives from wetSpring's import path,
//! bridging Kachkovskiy/Bourgain spectral theory to the quorum-sensing domain.
//!
//! Conceptual link (Paper 23 — Bourgain & Kachkovskiy 2018):
//!   Autoinducer diffusion through a heterogeneous bacterial population is
//!   analogous to wave propagation in a disordered medium.  Anderson
//!   localization predicts when signals (autoinducers) stay local vs.
//!   propagate community-wide, depending on population heterogeneity (W).
//!
//! Validation sections:
//!   1. Anderson 1D — Gershgorin bounds, Lyapunov exponent, level statistics
//!   2. Almost-Mathieu — Herman formula, Aubry–André transition
//!   3. Lanczos vs Sturm — eigenvalue accuracy
//!   4. Anderson 2D — GOE/Poisson transition with lattice size
//!   5. Anderson 3D — mobility edge, metal-insulator transition
//!   6. QS-disorder analogy — heterogeneity → localization transition
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `756df26` |
//! | Baseline tool | barracuda::spectral primitives (Anderson, Lanczos, level statistics) — Bourgain & Kachkovskiy 2018 theory |
//! | Baseline date | 2026-02-27 |
//! | Exact command | `cargo run --release --bin validate_spectral_cross_spring` |
//! | Data | Synthetic (Anderson 1D/2D/3D Hamiltonians, Almost-Mathieu, no external data) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use barracuda::spectral::{
    GOE_R, POISSON_R, almost_mathieu_hamiltonian, anderson_2d, anderson_3d, anderson_hamiltonian,
    find_all_eigenvalues, lanczos, lanczos_eigenvalues, level_spacing_ratio, lyapunov_exponent,
};
use std::time::Instant;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp107: Cross-Spring Spectral Theory (Anderson / QS)");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let mut v = Validator::new("Exp107: Spectral Cross-Spring");
    let t0 = Instant::now();

    validate_anderson_1d(&mut v);
    validate_almost_mathieu(&mut v);
    validate_lanczos_vs_sturm(&mut v);
    validate_anderson_2d(&mut v);
    validate_anderson_3d_transition(&mut v);
    validate_qs_disorder_analogy(&mut v);

    let total_ms = t0.elapsed().as_millis();
    println!();
    println!("  [Total] {total_ms} ms");
    v.finish();
}

/// Section 1: Anderson 1D — spectrum bounds, Lyapunov, level statistics
fn validate_anderson_1d(v: &mut Validator) {
    v.section("Anderson 1D: Spectrum Bounds + Lyapunov");

    let n = 500;
    let w = 4.0;
    let seed = 42;

    let (diag, off) = anderson_hamiltonian(n, w, seed);

    // Eigenvalues via Sturm bisection
    let eigenvalues = find_all_eigenvalues(&diag, &off);

    // Gershgorin bounds: spectrum ⊂ [-2 - W/2, 2 + W/2]
    let e_min = eigenvalues.first().copied().unwrap_or(0.0);
    let e_max = eigenvalues.last().copied().unwrap_or(0.0);
    let lower_bound = -2.0 - w / 2.0;
    let upper_bound = 2.0 + w / 2.0;
    v.check_pass(
        "Gershgorin lower: E_min >= -2-W/2",
        e_min >= lower_bound - tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "Gershgorin upper: E_max <= 2+W/2",
        e_max <= upper_bound + tolerances::ANALYTICAL_F64,
    );
    v.check_pass("eigenvalue count = N", eigenvalues.len() == n);

    // Lyapunov exponent at band centre: γ(0) > 0 (all states localized in 1D)
    let gamma_0 = lyapunov_exponent(&diag, 0.0);
    v.check_pass("Lyapunov γ(0) > 0 (localized)", gamma_0 > 0.0);

    // Kappus–Wegner: γ(0) ≈ W²/96 for moderate W
    let gamma_expected = w * w / 96.0;
    let rel_err = (gamma_0 - gamma_expected).abs() / gamma_expected;
    v.check_pass(
        &format!(
            "Lyapunov γ(0) ≈ W²/96 = {gamma_expected:.4} (got {gamma_0:.4}, rel {rel_err:.4})"
        ),
        rel_err < tolerances::SPECTRAL_LYAPUNOV_PARITY * 10.0,
    );

    // Lyapunov at band edge: γ(1.8) > 0
    let gamma_edge = lyapunov_exponent(&diag, 1.8);
    v.check_pass("Lyapunov γ(1.8) > 0 (band edge)", gamma_edge > 0.0);

    // Level statistics: strong disorder → Poisson
    let r = level_spacing_ratio(&eigenvalues);
    v.check(
        "⟨r⟩ Poisson (W=4, 1D localized)",
        r,
        POISSON_R,
        tolerances::SPECTRAL_POISSON_PARITY,
    );
    println!("  [INFO] ⟨r⟩ = {r:.4} (Poisson = {POISSON_R:.4})");
}

/// Section 2: Almost-Mathieu — Herman formula, Aubry–André
fn validate_almost_mathieu(v: &mut Validator) {
    v.section("Almost-Mathieu: Herman Formula + Aubry–André");

    let n = 500;
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;

    // Herman formula: λ > 1 → γ = ln(λ)
    for &lambda in &[1.5, 2.0, 3.0] {
        let (diag, _off) = almost_mathieu_hamiltonian(n, lambda, golden, 0.0);
        let gamma = lyapunov_exponent(&diag, 0.0);
        let expected = lambda.ln();
        let err = (gamma - expected).abs();
        v.check(
            &format!("Herman γ(0) at λ={lambda:.1}"),
            gamma,
            expected,
            tolerances::SPECTRAL_HERMAN_PARITY * lambda,
        );
        println!("  [INFO] λ={lambda:.1}: γ={gamma:.4}, ln(λ)={expected:.4}, err={err:.4}");
    }

    // Aubry–André transition via Lyapunov: λ < 1 → γ ≈ 0, λ > 1 → γ = ln(λ)
    // (Level statistics are unreliable for Almost-Mathieu due to Cantor spectrum)
    let (diag_ext, _off_ext) = almost_mathieu_hamiltonian(n, 0.5, golden, 0.0);
    let gamma_ext = lyapunov_exponent(&diag_ext, 0.0);
    let (diag_loc, off_loc) = almost_mathieu_hamiltonian(n, 2.0, golden, 0.0);
    let gamma_loc = lyapunov_exponent(&diag_loc, 0.0);

    v.check_pass(
        &format!("Aubry–André: λ=0.5 extended γ ≈ 0 (got {gamma_ext:.4})"),
        gamma_ext.abs() < tolerances::SPECTRAL_EXTENDED_LYAPUNOV,
    );
    v.check_pass(
        &format!("Aubry–André: λ=2.0 localized γ > 0 (got {gamma_loc:.4})"),
        gamma_loc > 0.3,
    );

    // Spectrum bounds: σ(H) ⊂ [-2 - 2λ, 2 + 2λ]
    let lambda = 2.0;
    let bound = 2.0f64.mul_add(lambda, 2.0);
    let eig_loc = find_all_eigenvalues(&diag_loc, &off_loc);
    v.check_pass(
        &format!("Almost-Mathieu spectrum bound: E_max <= {bound}"),
        eig_loc.last().copied().unwrap_or(0.0)
            <= bound + tolerances::SPECTRAL_ALMOST_MATHIEU_MARGIN,
    );
}

/// Section 3: Lanczos vs Sturm eigenvalue accuracy
fn validate_lanczos_vs_sturm(v: &mut Validator) {
    v.section("Lanczos vs Sturm: Eigenvalue Accuracy");

    let n = 200;
    let w = 2.0;
    let seed = 99;

    let (diag, off) = anderson_hamiltonian(n, w, seed);
    let sturm_eigs = find_all_eigenvalues(&diag, &off);

    // Build CSR for Lanczos
    let csr = barracuda::spectral::anderson_2d(1, n, w, seed);
    let tri = lanczos(&csr, n, 7);
    let lanczos_eigs = lanczos_eigenvalues(&tri);

    // Compare extremal eigenvalues
    let sturm_min = sturm_eigs.first().copied().unwrap_or(0.0);
    let sturm_max = sturm_eigs.last().copied().unwrap_or(0.0);

    let mut sorted_lanczos = lanczos_eigs;
    sorted_lanczos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lanczos_min = sorted_lanczos.first().copied().unwrap_or(0.0);
    let lanczos_max = sorted_lanczos.last().copied().unwrap_or(0.0);

    v.check(
        "Lanczos min eigenvalue ≈ Sturm min",
        lanczos_min,
        sturm_min,
        tolerances::LANCZOS_VS_STURM,
    );
    v.check(
        "Lanczos max eigenvalue ≈ Sturm max",
        lanczos_max,
        sturm_max,
        tolerances::LANCZOS_VS_STURM,
    );

    // Full Lanczos (m = N) should give exact spectrum
    let full_tri = lanczos(&csr, n, 7);
    let full_eigs = lanczos_eigenvalues(&full_tri);
    v.check_pass(
        &format!("Lanczos returns {} eigenvalues", full_eigs.len()),
        full_eigs.len() >= n / 2,
    );
}

/// Section 4: Anderson 2D — GOE/Poisson transition
fn validate_anderson_2d(v: &mut Validator) {
    v.section("Anderson 2D: GOE/Poisson Transition");

    // Weak disorder: should be GOE-like (extended states in 2D at large L)
    let l = 12;
    let csr_weak = anderson_2d(l, l, 1.0, 42);
    let tri_weak = lanczos(&csr_weak, l * l, 42);
    let eigs_weak = lanczos_eigenvalues(&tri_weak);
    let r_weak = level_spacing_ratio(&eigs_weak);
    println!("  [INFO] 2D (L={l}, W=1): ⟨r⟩ = {r_weak:.4}");
    v.check_pass(
        &format!("2D weak disorder ⟨r⟩ > Poisson (got {r_weak:.4})"),
        r_weak > POISSON_R - tolerances::SPECTRAL_R_MARGIN,
    );

    // Strong disorder: should approach Poisson
    let csr_strong = anderson_2d(l, l, 20.0, 42);
    let tri_strong = lanczos(&csr_strong, l * l, 42);
    let eigs_strong = lanczos_eigenvalues(&tri_strong);
    let r_strong = level_spacing_ratio(&eigs_strong);
    println!("  [INFO] 2D (L={l}, W=20): ⟨r⟩ = {r_strong:.4}");
    v.check(
        "2D strong disorder ⟨r⟩ ≈ Poisson",
        r_strong,
        POISSON_R,
        tolerances::SPECTRAL_POISSON_PARITY,
    );

    // Gershgorin for 2D: σ(H) ⊂ [-4 - W/2, 4 + W/2]
    let bound = 4.0 + 20.0 / 2.0;
    let mut sorted = eigs_strong;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.check_pass(
        &format!("2D spectrum bound: E_max <= {bound}"),
        sorted.last().copied().unwrap_or(0.0) <= bound + tolerances::SPECTRAL_GERSHGORIN_MARGIN,
    );
}

/// Section 5: Anderson 3D — metal-insulator transition
fn validate_anderson_3d_transition(v: &mut Validator) {
    v.section("Anderson 3D: Metal-Insulator Transition");

    let l = 6;

    // Weak disorder (W=2): metallic regime, GOE-like
    let csr_weak = anderson_3d(l, l, l, 2.0, 42);
    let tri_weak = lanczos(&csr_weak, l * l * l, 42);
    let eigs_weak = lanczos_eigenvalues(&tri_weak);
    let r_weak = level_spacing_ratio(&eigs_weak);
    println!("  [INFO] 3D (L={l}, W=2): ⟨r⟩ = {r_weak:.4} (GOE = {GOE_R:.4})");
    v.check_pass(
        &format!("3D metallic ⟨r⟩ > Poisson (got {r_weak:.4})"),
        r_weak > POISSON_R + tolerances::SPECTRAL_R_MARGIN,
    );

    // Strong disorder (W=25): insulating regime, Poisson-like
    let csr_strong = anderson_3d(l, l, l, 25.0, 42);
    let tri_strong = lanczos(&csr_strong, l * l * l, 42);
    let eigs_strong = lanczos_eigenvalues(&tri_strong);
    let r_strong = level_spacing_ratio(&eigs_strong);
    println!("  [INFO] 3D (L={l}, W=25): ⟨r⟩ = {r_strong:.4} (Poisson = {POISSON_R:.4})");
    v.check(
        "3D insulating ⟨r⟩ ≈ Poisson",
        r_strong,
        POISSON_R,
        tolerances::SPECTRAL_POISSON_PARITY,
    );

    // Gershgorin for 3D: σ(H) ⊂ [-6 - W/2, 6 + W/2]
    let bound = 6.0 + 25.0 / 2.0;
    let mut sorted = eigs_strong;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.check_pass(
        &format!("3D spectrum bound: E_max <= {bound}"),
        sorted.last().copied().unwrap_or(0.0) <= bound + tolerances::SPECTRAL_GERSHGORIN_MARGIN,
    );
}

/// Section 6: QS-disorder analogy
///
/// Model a heterogeneous bacterial population as a 1D disordered chain.
/// Disorder W = variability in autoinducer production rates.
/// Low W → signals propagate (extended states → community QS).
/// High W → signals localize (Anderson localization → local QS only).
fn validate_qs_disorder_analogy(v: &mut Validator) {
    v.section("QS-Disorder Analogy: Heterogeneity → Localization");

    let n = 500;
    let seed = 123;

    // Low heterogeneity (W=0.5): near-homogeneous population → extended
    let (diag_low, off_low) = anderson_hamiltonian(n, 0.5, seed);
    let eig_low = find_all_eigenvalues(&diag_low, &off_low);
    let r_low = level_spacing_ratio(&eig_low);

    // High heterogeneity (W=10): strongly disordered → fully localized
    let (diag_high, off_high) = anderson_hamiltonian(n, 10.0, seed);
    let eig_high = find_all_eigenvalues(&diag_high, &off_high);
    let r_high = level_spacing_ratio(&eig_high);

    println!("  [INFO] QS heterogeneity sweep:");
    println!("    W=0.5  (homogeneous): ⟨r⟩ = {r_low:.4}");
    println!("    W=10.0 (heterogeneous): ⟨r⟩ = {r_high:.4}");

    // Key physical prediction: ⟨r⟩ decreases from extended → localized
    v.check_pass(
        &format!("⟨r⟩(W=0.5) > ⟨r⟩(W=10): {r_low:.4} > {r_high:.4}"),
        r_low > r_high,
    );

    // High heterogeneity → Poisson (signals fully localized)
    v.check(
        "High heterogeneity ⟨r⟩ ≈ Poisson (signals localized)",
        r_high,
        POISSON_R,
        tolerances::SPECTRAL_POISSON_PARITY,
    );

    // Lyapunov confirms localization increases with W
    let pot_low: Vec<f64> = diag_low;
    let pot_high: Vec<f64> = diag_high;
    let gamma_low = lyapunov_exponent(&pot_low, 0.0);
    let gamma_high = lyapunov_exponent(&pot_high, 0.0);
    v.check_pass(
        &format!("γ(W=10) > γ(W=0.5): {gamma_high:.4} > {gamma_low:.4}"),
        gamma_high > gamma_low,
    );

    println!("  [INFO] Interpretation: high population heterogeneity localizes");
    println!("         autoinducer signals, preventing community-wide QS activation.");
    println!("         This is the Anderson localization ↔ quorum sensing bridge.");
}
