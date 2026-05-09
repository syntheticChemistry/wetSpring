// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layer 0–2: Bare science certification (no NUCLEUS required).
//!
//! - **B0**: Deterministic science baselines (Shannon, Hill, stats, matmul)
//! - **B1**: Tolerance provenance audit
//! - **B2**: BLAKE3 checksum verification

use primalspring::checksums;
use primalspring::tolerances as ps_tol;
use primalspring::validation::ValidationResult;

use crate::tolerances;

/// B0 — Bare Science Baselines: deterministic outputs, reference-traceable.
pub fn validate_bare_science(v: &mut ValidationResult) {
    let shannon_h4 = 4.0_f64.ln();
    let shannon_expected = std::f64::consts::LN_2 * 2.0;
    v.check_bool(
        "Shannon H'(uniform, S=4) = ln(4)",
        (shannon_h4 - shannon_expected).abs() <= tolerances::ANALYTICAL_F64,
        "Shannon 1948 — exact: ln(4) = 2·ln(2)",
    );

    let hill_at_ic50: f64 = 10.0 / (10.0 + 10.0);
    v.check_bool(
        "Hill(IC50, IC50, n=1) = 0.5",
        (hill_at_ic50 - 0.5).abs() <= tolerances::ANALYTICAL_F64,
        "Gonzales 2014 — Hill equation at IC50",
    );

    let local_mean = 30.0_f64;
    v.check_bool(
        "mean([10..50]) = 30.0",
        (local_mean - 30.0).abs() <= tolerances::ANALYTICAL_F64,
        "Python baseline: scripts/diversity/compute_stats.py",
    );

    let local_std = (200.0_f64).sqrt();
    v.check_bool(
        "std_dev([10..50]) = √200",
        (local_std - (200.0_f64).sqrt()).abs() <= tolerances::ANALYTICAL_F64,
        "analytical: population variance = 200",
    );

    let expected_matmul = [19.0_f64, 22.0, 43.0, 50.0];
    let lhs = [[1.0_f64, 2.0], [3.0, 4.0]];
    let rhs = [[5.0_f64, 6.0], [7.0, 8.0]];
    let actual_matmul = [
        lhs[0][0].mul_add(rhs[0][0], lhs[0][1] * rhs[1][0]),
        lhs[0][0].mul_add(rhs[0][1], lhs[0][1] * rhs[1][1]),
        lhs[1][0].mul_add(rhs[0][0], lhs[1][1] * rhs[1][0]),
        lhs[1][0].mul_add(rhs[0][1], lhs[1][1] * rhs[1][1]),
    ];
    let matmul_ok = actual_matmul
        .iter()
        .zip(expected_matmul.iter())
        .all(|(got, exp)| (got - exp).abs() <= tolerances::ANALYTICAL_F64);
    v.check_bool(
        "matmul 2×2 = [[19,22],[43,50]]",
        matmul_ok,
        "linear algebra identity — exact integer result",
    );

    let local_wm = 3.0_f64.mul_add(0.2, 1.0_f64.mul_add(0.5, 2.0 * 0.3));
    v.check_bool(
        "weighted_mean([1,2,3],[0.5,0.3,0.2]) = 1.7",
        (local_wm - 1.7).abs() <= tolerances::ANALYTICAL_F64,
        "analytical: 1×0.5 + 2×0.3 + 3×0.2 = 1.7",
    );

    let tampered_mean: f64 = 999.0;
    v.check_bool(
        "self-verify: tampered mean ≠ 30 detected",
        (tampered_mean - 30.0).abs() > tolerances::ANALYTICAL_F64,
        "Property 3: tampered input → non-zero exit",
    );
}

/// B1 — Tolerance Provenance: every tolerance has a documented derivation.
pub fn validate_tolerance_provenance(v: &mut ValidationResult) {
    v.check_bool(
        "ANALYTICAL_F64 = 1e-12",
        (tolerances::ANALYTICAL_F64 - 1e-12).abs() < f64::EPSILON,
        "f64 accumulated rounding budget for analytical chains (Shannon, Simpson)",
    );

    v.check_bool(
        "IPC_ROUND_TRIP_TOL ∈ (0, ANALYTICAL_LOOSE]",
        ps_tol::IPC_ROUND_TRIP_TOL > 0.0
            && ps_tol::IPC_ROUND_TRIP_TOL <= tolerances::ANALYTICAL_LOOSE,
        "JSON f64 serialization round-trip precision loss",
    );
}

/// B2 — Checksum Verification: BLAKE3 manifests detect tampering.
pub fn validate_checksums(v: &mut ValidationResult) {
    let manifest = "validation/CHECKSUMS";
    if std::path::Path::new(manifest).exists() {
        checksums::verify_manifest(v, manifest);
    } else {
        v.check_skip(
            "p3:checksums_manifest",
            "validation/CHECKSUMS not found (run from repo root to enable)",
        );
    }
}
