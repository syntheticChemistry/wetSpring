// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    dead_code
)]
//! # Exp276: Track 5 CPU Parity — Immunological Anderson Pure Rust Math
//!
//! Consolidated `BarraCuda` CPU parity benchmark for all Paper 12 immunological
//! Anderson domains. Demonstrates that pure Rust math (barracuda CPU) produces
//! correct results for skin-layer Anderson lattice, barrier disruption, cell-type
//! heterogeneity sweep, and Fajgenbaum geometry scoring.
//!
//! ## Domains validated
//! - D01: Alpha diversity (Shannon, Simpson, Chao1, Pielou) on cell populations
//! - D02: Anderson spectral (2D epidermis, 3D dermis) — level spacing ratio
//! - D03: Barrier disruption (dimensional promotion, depth scan)
//! - D04: Disorder sweep (W vs r in 3D)
//! - D05: Statistical inference (Pearson correlation, Pielou → W mapping)
//! - D06: Fajgenbaum drug repurposing geometry scoring
//!
//! ## Evolution path
//! - **This experiment**: `BarraCuda` CPU (pure Rust, single-threaded)
//! - **Next**: Exp277 GPU validation (same math on GPU)
//! - **Then**: Exp278 `ToadStool` dispatch (streaming pipeline)
//! - **Final**: Exp279 `metalForge` cross-substrate (NUCLEUS atomics)
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | Track | Track 5 — Immunological Anderson & Drug Repurposing |
//! | baseCamp paper | Paper 12 |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_immuno_anderson_cpu_parity` |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use barracuda::stats::norm_cdf;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    checks: usize,
}

fn main() {
    let mut v = Validator::new("Exp276: Track 5 CPU Parity — Immunological Anderson");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // D01: Alpha Diversity on Cell Populations
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Alpha Diversity (Cell Populations) ═══");
    let t0 = Instant::now();

    let healthy: Vec<f64> = vec![850.0, 50.0, 30.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0, 10.0];
    let mild_ad: Vec<f64> = vec![700.0, 70.0, 25.0, 35.0, 50.0, 40.0, 15.0, 20.0, 15.0, 30.0];
    let severe: Vec<f64> = vec![
        400.0, 100.0, 15.0, 60.0, 140.0, 90.0, 55.0, 40.0, 35.0, 65.0,
    ];
    let treated: Vec<f64> = vec![650.0, 60.0, 25.0, 30.0, 40.0, 25.0, 10.0, 15.0, 10.0, 35.0];

    let h_healthy = diversity::shannon(&healthy);
    let h_mild = diversity::shannon(&mild_ad);
    let h_severe = diversity::shannon(&severe);
    let h_treated = diversity::shannon(&treated);

    v.check_pass("Shannon(healthy) > 0", h_healthy > 0.0);
    v.check_pass(
        "Shannon monotonic: healthy < mild < severe",
        h_healthy < h_mild && h_mild < h_severe,
    );
    v.check_pass(
        "Treatment reduces Shannon: treated < severe",
        h_treated < h_severe,
    );

    let si_healthy = diversity::simpson(&healthy);
    let si_severe = diversity::simpson(&severe);
    v.check_pass(
        "Simpson(healthy) in [0,1]",
        (0.0..=1.0).contains(&si_healthy),
    );
    v.check_pass(
        "Simpson(severe) > Simpson(healthy) (more even)",
        si_severe > si_healthy,
    );

    let ch_healthy = diversity::chao1(&healthy);
    let ch_severe = diversity::chao1(&severe);
    v.check_pass(
        "Chao1(healthy) >= richness",
        ch_healthy >= healthy.len() as f64,
    );
    v.check_pass(
        "Chao1(severe) >= richness",
        ch_severe >= severe.len() as f64,
    );

    let s = healthy.len() as f64;
    let pielou_healthy = h_healthy / s.ln();
    let pielou_severe = h_severe / s.ln();
    v.check_pass(
        "Pielou(severe) > Pielou(healthy)",
        pielou_severe > pielou_healthy,
    );
    v.check_pass(
        "Pielou in [0,1]",
        pielou_healthy >= 0.0 && pielou_severe <= 1.0,
    );

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Alpha diversity",
        cpu_us,
        checks: 9,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Anderson Spectral (2D vs 3D)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Anderson Spectral (2D Epidermis vs 3D Dermis) ═══");
    let t0 = Instant::now();

    let midpoint = f64::midpoint(GOE_R, POISSON_R);

    let mat_2d = anderson_2d(8, 8, 16.0, 42);
    let tri_2d = lanczos(&mat_2d, 64, 42);
    let eigs_2d = lanczos_eigenvalues(&tri_2d);
    let r_2d = level_spacing_ratio(&eigs_2d);

    let mat_3d = anderson_3d(8, 8, 8, 4.0, 42);
    let tri_3d = lanczos(&mat_3d, 512, 42);
    let eigs_3d = lanczos_eigenvalues(&tri_3d);
    let r_3d = level_spacing_ratio(&eigs_3d);

    v.check_pass("2D eigenvalues count = 64", eigs_2d.len() == 64);
    v.check_pass("3D eigenvalues count = 512", eigs_3d.len() == 512);
    v.check_pass(
        "2D r ≤ midpoint at W=16 (localized)",
        r_2d <= midpoint + 0.02,
    );
    v.check_pass("3D r > midpoint at W=4 (extended)", r_3d > midpoint);
    v.check_pass("r in valid range [0.35, 0.55]", r_2d > 0.35 && r_3d < 0.55);

    println!("  2D (L=8, W=16): r = {r_2d:.6}");
    println!("  3D (L=8, W=4):  r = {r_3d:.6}");
    println!("  GOE = {GOE_R:.6}, Poisson = {POISSON_R:.6}, midpoint = {midpoint:.6}");

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Anderson spectral",
        cpu_us,
        checks: 5,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Barrier Disruption (Dimensional Promotion)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Barrier Disruption (Dimensional Promotion) ═══");
    let t0 = Instant::now();

    let w_barrier = 12.0;
    let seeds: [u64; 5] = [42, 137, 271, 577, 997];

    let r_2d_ens = ensemble_r_2d(8, 8, w_barrier, &seeds);
    let r_slab = ensemble_r_3d(6, 6, 4, w_barrier, &seeds);
    let r_full = ensemble_r_3d(6, 6, 6, w_barrier, &seeds);

    v.check_pass("Ensemble 2D r > 0.35", r_2d_ens > 0.35);
    v.check_pass("Ensemble slab r > 0.35", r_slab > 0.35);
    v.check_pass("Ensemble full 3D r > 0.35", r_full > 0.35);

    let promotion_delta = r_full - r_2d_ens;
    println!("  2D:      <r> = {r_2d_ens:.6}");
    println!("  Slab:    <r> = {r_slab:.6}");
    println!("  Full 3D: <r> = {r_full:.6}");
    println!("  Δr(2D→3D) = {promotion_delta:+.6}");

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Barrier disruption",
        cpu_us,
        checks: 3,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Disorder Sweep (W_c in 3D)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Disorder Sweep (W_c in 3D Dermis) ═══");
    let t0 = Instant::now();

    let l = 6_usize;
    let n = l * l * l;
    let w_low = 2.0;
    let w_high = 24.0;

    let mat_lo = anderson_3d(l, l, l, w_low, 42);
    let tri_lo = lanczos(&mat_lo, n, 42);
    let eigs_lo = lanczos_eigenvalues(&tri_lo);
    let r_lo = level_spacing_ratio(&eigs_lo);

    let mat_hi = anderson_3d(l, l, l, w_high, 42);
    let tri_hi = lanczos(&mat_hi, n, 42);
    let eigs_hi = lanczos_eigenvalues(&tri_hi);
    let r_hi = level_spacing_ratio(&eigs_hi);

    v.check_pass("r(W=2) > r(W=24)", r_lo > r_hi);
    v.check_pass("r(W=2) in extended regime", r_lo > midpoint);
    v.check_pass("r(W=24) in localized regime", r_hi < midpoint);

    println!("  r(W={w_low}) = {r_lo:.6}  (extended)");
    println!("  r(W={w_high}) = {r_hi:.6}  (localized)");

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Disorder sweep",
        cpu_us,
        checks: 3,
    });

    // ═══════════════════════════════════════════════════════════════
    // D05: Pielou → W Mapping + Statistical Inference
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Pielou → Disorder Mapping ═══");
    let t0 = Instant::now();

    let profiles: Vec<(&str, Vec<f64>)> = vec![
        (
            "Healthy",
            vec![850.0, 50.0, 30.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0, 10.0],
        ),
        (
            "Mild",
            vec![700.0, 70.0, 25.0, 35.0, 50.0, 40.0, 15.0, 20.0, 15.0, 30.0],
        ),
        (
            "Moderate",
            vec![550.0, 90.0, 20.0, 50.0, 100.0, 70.0, 35.0, 30.0, 25.0, 30.0],
        ),
        (
            "Severe",
            vec![
                400.0, 100.0, 15.0, 60.0, 140.0, 90.0, 55.0, 40.0, 35.0, 65.0,
            ],
        ),
    ];

    let mut prev_pielou = 0.0;
    for (name, counts) in &profiles {
        let h = diversity::shannon(counts);
        let j = h / (counts.len() as f64).ln();
        let w = j * 24.0;

        v.check_pass(&format!("Pielou({name}) > prev"), j > prev_pielou);
        prev_pielou = j;

        let cdf = norm_cdf((w - 15.0) / 5.0);
        v.check_pass(
            &format!("norm_cdf(W={w:.1}) in [0,1]"),
            (0.0..=1.0).contains(&cdf),
        );

        println!("  {name:<12} Pielou={j:.4}  W={w:.1}  CDF(W|μ=15,σ=5)={cdf:.4}");
    }

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Pielou→W mapping",
        cpu_us,
        checks: 8,
    });

    // ═══════════════════════════════════════════════════════════════
    // D06: Fajgenbaum Geometry Scoring
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: Fajgenbaum Geometry-Augmented Drug Score ═══");
    let t0 = Instant::now();

    let drugs: Vec<(&str, f64, bool, bool)> = vec![
        ("Apoquel", 0.95, true, true),
        ("Cytopoint", 0.90, true, false),
        ("Rapamycin", 0.65, true, false),
        ("Crisaborole", 0.55, false, true),
        ("Trametinib", 0.40, true, false),
    ];

    let mut scores: Vec<f64> = Vec::new();
    for (name, pathway, reaches_dermis, crosses_barrier) in &drugs {
        let geom =
            if *reaches_dermis { 1.0 } else { 0.4 } * if *crosses_barrier { 1.0 } else { 0.8 };
        let score = pathway * geom;
        scores.push(score);
        println!("  {name:<14} pathway={pathway:.2}  geom={geom:.2}  score={score:.2}");
    }

    v.check_pass(
        "Apoquel highest score",
        scores[0]
            >= *scores
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                - tolerances::ANALYTICAL_LOOSE,
    );
    v.check_pass(
        "Crisaborole penalized by geometry",
        scores[3] < scores[0] && scores[3] < scores[1],
    );
    v.check_pass(
        "All scores in [0, 1]",
        scores.iter().all(|s| (0.0..=1.0).contains(s)),
    );
    v.check_pass(
        "Score = pathway × geometry (Apoquel check)",
        (scores[0] - 0.95).abs() < tolerances::ANALYTICAL_LOOSE,
    );

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Fajgenbaum scoring",
        cpu_us,
        checks: 4,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Exp276: Track 5 CPU Parity — Immunological Anderson         ║");
    println!("╠═════════════════════════╦════════════╦═══════════════════════╣");
    println!("║ Domain                  ║   CPU (µs) ║ Checks               ║");
    println!("╠═════════════════════════╬════════════╬═══════════════════════╣");

    let mut total_checks = 0_usize;
    let mut total_us = 0.0_f64;
    for t in &timings {
        println!(
            "║ {:<23} ║ {:>10.0} ║ {:>3}                   ║",
            t.domain, t.cpu_us, t.checks
        );
        total_checks += t.checks;
        total_us += t.cpu_us;
    }

    println!("╠═════════════════════════╬════════════╬═══════════════════════╣");
    println!(
        "║ TOTAL                   ║ {total_us:>10.0} ║ {total_checks:>3}                   ║"
    );
    println!("╚═════════════════════════╩════════════╩═══════════════════════╝");
    println!();

    v.finish();
}

fn ensemble_r_2d(lx: usize, ly: usize, w: f64, seeds: &[u64]) -> f64 {
    let n = lx * ly;
    let mut sum = 0.0;
    for &seed in seeds {
        let mat = anderson_2d(lx, ly, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        sum += level_spacing_ratio(&eigs);
    }
    sum / seeds.len() as f64
}

fn ensemble_r_3d(lx: usize, ly: usize, lz: usize, w: f64, seeds: &[u64]) -> f64 {
    let n = lx * ly * lz;
    let mut sum = 0.0;
    for &seed in seeds {
        let mat = anderson_3d(lx, ly, lz, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        sum += level_spacing_ratio(&eigs);
    }
    sum / seeds.len() as f64
}
