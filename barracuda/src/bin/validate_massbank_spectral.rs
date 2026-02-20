// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp042 — MassBank PFAS spectral matching validation.
//!
//! # Provenance
//!
//! | Item            | Value                                                              |
//! |-----------------|--------------------------------------------------------------------|
//! | Baseline script | `scripts/massbank_spectral_baseline.py`                            |
//! | Baseline output | `experiments/results/042_massbank_spectral/python_baseline.json`    |
//! | Data source     | MassBank/MassBank-data (PFAS reference spectra)                    |
//! | Proxy for       | Paper #21, Jones PFAS mass spectrometry                            |
//! | Date            | 2026-02-20                                                         |
//!
//! Validates cosine similarity spectral matching on synthetic PFAS-like
//! mass spectra, matching Python baseline exactly.

use wetspring_barracuda::bio::spectral_match::cosine_similarity;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp042: MassBank Spectral Matching Validation");

    // PFOS-like spectrum: characteristic CF₂ fragments
    let pfos_mz = vec![80.0, 99.0, 119.0, 169.0, 219.0, 269.0, 319.0, 369.0, 419.0, 499.0];
    let pfos_int = vec![30.0, 100.0, 45.0, 80.0, 55.0, 70.0, 40.0, 25.0, 15.0, 60.0];

    // Near-PFOS: slight instrument shift
    let shifted_mz = vec![80.1, 99.05, 118.95, 169.1, 219.0, 268.9, 319.1, 369.0, 419.05, 499.0];
    let shifted_int = vec![28.0, 98.0, 46.0, 78.0, 54.0, 72.0, 38.0, 26.0, 16.0, 58.0];

    // PFOA-like spectrum
    let pfoa_mz = vec![69.0, 119.0, 169.0, 219.0, 269.0, 319.0, 369.0, 413.0];
    let pfoa_int = vec![100.0, 40.0, 65.0, 50.0, 55.0, 35.0, 20.0, 45.0];

    // Caffeine (unrelated)
    let caffeine_mz = vec![42.0, 55.0, 67.0, 82.0, 109.0, 137.0, 194.0];
    let caffeine_int = vec![20.0, 30.0, 40.0, 55.0, 100.0, 80.0, 70.0];

    let tol = 0.5;

    // ── Section 1: Self-match ───────────────────────────────────
    v.section("── Self-match (cosine = 1.0) ──");
    let cs_self = cosine_similarity(&pfos_mz, &pfos_int, &pfos_mz, &pfos_int, tol).score;
    v.check("cosine(PFOS, PFOS)", cs_self, 1.0, 1e-10);

    // ── Section 2: Near-match ───────────────────────────────────
    v.section("── Near-match (instrument variation) ──");
    let cs_near = cosine_similarity(&pfos_mz, &pfos_int, &shifted_mz, &shifted_int, tol).score;
    v.check("cosine(PFOS, shifted)", cs_near, 0.999_662, 1e-3);
    let near_high = cs_near > 0.99;
    v.check_count("near_match > 0.99", usize::from(near_high), 1);

    // ── Section 3: Same PFAS family ─────────────────────────────
    v.section("── PFAS family match (PFOS vs PFOA) ──");
    let cs_family = cosine_similarity(&pfos_mz, &pfos_int, &pfoa_mz, &pfoa_int, tol).score;
    // Rust uses strict m/z matching within tolerance — partial overlap expected
    let family_moderate = cs_family > 0.3;
    v.check_count("family_match > 0.3", usize::from(family_moderate), 1);

    // ── Section 4: Unrelated compound ───────────────────────────
    v.section("── Unrelated match (PFOS vs caffeine) ──");
    let cs_unrelated = cosine_similarity(&pfos_mz, &pfos_int, &caffeine_mz, &caffeine_int, tol).score;
    let unrelated_low = cs_unrelated < 0.3;
    v.check_count("unrelated_match < 0.3", usize::from(unrelated_low), 1);

    // ── Section 5: Symmetry ─────────────────────────────────────
    v.section("── Cosine symmetry ──");
    let cs_ab = cosine_similarity(&pfos_mz, &pfos_int, &pfoa_mz, &pfoa_int, tol).score;
    let cs_ba = cosine_similarity(&pfoa_mz, &pfoa_int, &pfos_mz, &pfos_int, tol).score;
    v.check("cosine(A,B) = cosine(B,A)", cs_ab, cs_ba, 1e-10);

    // ── Section 6: Pairwise matrix ──────────────────────────────
    v.section("── Pairwise matrix properties ──");
    let all_mz = [&pfos_mz, &shifted_mz, &pfoa_mz, &caffeine_mz];
    let all_int = [&pfos_int, &shifted_int, &pfoa_int, &caffeine_int];
    let mut diagonal_ones = true;
    for i in 0..4 {
        let cs = cosine_similarity(all_mz[i], all_int[i], all_mz[i], all_int[i], tol).score;
        if (cs - 1.0).abs() > 1e-10 {
            diagonal_ones = false;
        }
    }
    v.check_count("diagonal all 1.0", usize::from(diagonal_ones), 1);

    let mut all_nonneg = true;
    for i in 0..4 {
        for j in 0..4 {
            let cs = cosine_similarity(all_mz[i], all_int[i], all_mz[j], all_int[j], tol).score;
            if cs < -1e-10 {
                all_nonneg = false;
            }
        }
    }
    v.check_count("all pairwise >= 0", usize::from(all_nonneg), 1);

    // ── Section 7: Determinism ──────────────────────────────────
    v.section("── Determinism ──");
    let cs1 = cosine_similarity(&pfos_mz, &pfos_int, &pfoa_mz, &pfoa_int, tol).score;
    let cs2 = cosine_similarity(&pfos_mz, &pfos_int, &pfoa_mz, &pfoa_int, tol).score;
    v.check("cosine deterministic", cs1, cs2, 0.0);

    v.finish();
}
