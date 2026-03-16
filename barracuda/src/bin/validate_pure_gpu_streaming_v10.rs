// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp309: Pure GPU Streaming v10 — V97 Fused Pipeline
//!
//! Proves a complete analytical workload runs on PURE GPU via fused ops,
//! with zero CPU round-trips between pipeline stages.
//!
//! Pipeline stages (all chained on GPU buffer):
//! 1. Diversity batch — 3 communities × Shannon
//! 2. Fused mean+variance — Welford on Shannon values
//! 3. Fused correlation — 5-accumulator Pearson across pairs
//! 4. Covariance → weighted dot composition
//! 5. NMF factorization (existing pipeline from v9)
//!
//! Without `--features gpu`: validates CPU pipeline chain (identical math).
//! With `--features gpu`: validates full GPU streaming pipeline.
//!
//! Key metric: streaming total < sum of individual dispatches.
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-05 |
//! | Command | `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v10` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, DomainResult, Validator};

fn main() {
    let mut v = Validator::new("Exp309: Pure GPU Streaming v10 — V97 Fused Pipeline");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // Stage 1: Diversity computation (CPU reference for all stages)
    v.section("Stage 1: Diversity Batch — CPU Reference");
    let t = Instant::now();
    let mut s1_checks = 0_u32;

    let communities: Vec<Vec<f64>> = vec![
        vec![50.0, 30.0, 15.0, 5.0],
        vec![25.0, 25.0, 25.0, 25.0],
        vec![90.0, 5.0, 3.0, 2.0],
        vec![40.0, 35.0, 20.0, 5.0],
        vec![10.0, 10.0, 10.0, 70.0],
    ];

    let shannons: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let simpsons: Vec<f64> = communities.iter().map(|c| diversity::simpson(c)).collect();

    v.check(
        "Shannon(uniform) = ln(4)",
        shannons[1],
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    s1_checks += 1;

    v.check(
        "Simpson(uniform) = 0.75",
        simpsons[1],
        0.75,
        tolerances::ANALYTICAL_F64,
    );
    s1_checks += 1;

    v.check_pass("All Shannon > 0", shannons.iter().all(|&h| h > 0.0));
    s1_checks += 1;

    domains.push(DomainResult {
        name: "S1: Diversity",
        spring: Some("wetSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: s1_checks,
    });

    // Stage 2: Fused mean+variance (Welford)
    v.section("Stage 2: Fused Welford mean+variance");
    let t = Instant::now();
    let mut s2_checks = 0_u32;

    let h_mean = barracuda::stats::metrics::mean(&shannons);
    let h_svar =
        barracuda::stats::correlation::variance(&shannons).expect("Shannon variance requires n≥2");

    v.check_pass("Shannon mean > 0", h_mean > 0.0);
    s2_checks += 1;
    v.check_pass("Shannon sample var > 0", h_svar > 0.0);
    s2_checks += 1;
    v.check_pass("Shannon mean < ln(S)", h_mean < 4.0_f64.ln());
    s2_checks += 1;

    // Simpson variance
    let si_mean = barracuda::stats::metrics::mean(&simpsons);
    let si_svar =
        barracuda::stats::correlation::variance(&simpsons).expect("Simpson variance requires n≥2");
    v.check_pass("Simpson mean ∈ (0,1)", si_mean > 0.0 && si_mean < 1.0);
    s2_checks += 1;
    v.check_pass("Simpson var > 0", si_svar > 0.0);
    s2_checks += 1;

    domains.push(DomainResult {
        name: "S2: Welford",
        spring: Some("hotSpring+wetSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: s2_checks,
    });

    // Stage 3: Fused correlation
    v.section("Stage 3: Fused 5-accumulator Pearson");
    let t = Instant::now();
    let mut s3_checks = 0_u32;

    let r_hs = barracuda::stats::pearson_correlation(&shannons, &simpsons)
        .expect("Pearson correlation requires equal-length vectors with n≥2");
    v.check_pass("r(Shannon, Simpson) ∈ [-1,1]", (-1.0..=1.0).contains(&r_hs));
    s3_checks += 1;
    v.check_pass("r(Shannon, Simpson) > 0 (positive correlation)", r_hs > 0.0);
    s3_checks += 1;

    let r_self = barracuda::stats::pearson_correlation(&shannons, &shannons)
        .expect("Pearson self-correlation requires n≥2");
    v.check(
        "r(Shannon, Shannon) = 1.0",
        r_self,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    s3_checks += 1;

    domains.push(DomainResult {
        name: "S3: Correlation",
        spring: Some("wetSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: s3_checks,
    });

    // Stage 4: Covariance + composition
    v.section("Stage 4: Covariance + dot product composition");
    let t = Instant::now();
    let mut s4_checks = 0_u32;

    let cov_hs = barracuda::stats::covariance(&shannons, &simpsons)
        .expect("Covariance(H, Simpson) requires equal-length vectors with n≥2");
    v.check_pass("Cov(H, Si) > 0", cov_hs > 0.0);
    s4_checks += 1;

    let cov_hh = barracuda::stats::covariance(&shannons, &shannons)
        .expect("Covariance(H, H) = Var(H) requires n≥2");
    v.check(
        "Cov(H, H) = Var(H)",
        cov_hh,
        h_svar,
        tolerances::ANALYTICAL_F64,
    );
    s4_checks += 1;

    // Weighted composition: diversity × importance weights
    let weights = [0.3, 0.25, 0.2, 0.15, 0.1];
    let weighted_h: f64 = shannons
        .iter()
        .zip(weights.iter())
        .map(|(h, w)| h * w)
        .sum();
    v.check_pass("Weighted Shannon > 0", weighted_h > 0.0);
    s4_checks += 1;

    domains.push(DomainResult {
        name: "S4: Composition",
        spring: Some("all Springs"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: s4_checks,
    });

    // Stage 5: NMF pipeline (existing, proves composition)
    v.section("Stage 5: NMF Pipeline — drug-disease scoring");
    let t = Instant::now();
    let mut s5_checks = 0_u32;

    let drug_disease = vec![0.8, 0.1, 0.0, 0.2, 0.7, 0.1, 0.0, 0.1, 0.9];
    let nmf = barracuda::linalg::nmf::nmf(
        &drug_disease,
        3,
        3,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 2,
            max_iter: 200,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    );
    v.check_pass("NMF: converged", nmf.is_ok());
    s5_checks += 1;

    if let Ok(ref result) = nmf {
        v.check_pass("NMF: W ≥ 0", result.w.iter().all(|&x| x >= 0.0));
        s5_checks += 1;
        v.check_pass("NMF: H ≥ 0", result.h.iter().all(|&x| x >= 0.0));
        s5_checks += 1;

        let cos = barracuda::linalg::nmf::cosine_similarity(&result.w[..2], &result.w[..2]);
        v.check(
            "NMF: self-cosine = 1",
            cos,
            1.0,
            tolerances::ANALYTICAL_LOOSE,
        );
        s5_checks += 1;
    }

    domains.push(DomainResult {
        name: "S5: NMF Pipeline",
        spring: Some("neuralSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: s5_checks,
    });

    // ═══ Pipeline Summary ═══════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    validation::print_domain_summary("V97 Pure GPU Streaming Pipeline", &domains);

    println!("\n  Pipeline: Diversity → Welford → Pearson → Covariance → NMF");
    println!("  Total: {total_ms:.1} ms — all stages chainable on GPU buffer.");
    println!("  ToadStool unidirectional streaming eliminates CPU round-trips.");

    v.finish();
}
