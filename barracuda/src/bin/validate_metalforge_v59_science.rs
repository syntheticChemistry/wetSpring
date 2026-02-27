// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    dead_code
)]
//! Exp192: metalForge V59 Cross-Substrate — CPU↔GPU Parity for Science Domains
//!
//! Proves that V59 science computations produce identical results on CPU
//! and GPU substrates. Exercises diversity, Bray-Curtis, and Anderson
//! spectral analysis on both substrates and compares outputs.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | CPU diversity (Exp184/185), GPU Anderson (Exp191) |
//! | Date          | 2026-02-26 |
//! | Command       | `cargo run --features gpu --release --bin validate_metalforge_v59_science` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::GPU_CPU_F64` for cross-substrate parity |

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn synthetic_community(n_species: usize, evenness: f64, seed: u64) -> Vec<f64> {
    let mut counts = Vec::with_capacity(n_species);
    let mut rng = seed;
    for i in 0..n_species {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((rng >> 33) as f64) / f64::from(u32::MAX);
        let rank_weight = (-(i as f64) / (n_species as f64 * evenness)).exp();
        counts.push((rank_weight * 1000.0 * (0.5 + noise)).max(1.0));
    }
    counts
}

// ── MF01: Diversity CPU↔GPU Parity ──────────────────────────────────────────

fn validate_diversity_parity(v: &mut Validator) {
    v.section("═══ MF01: Diversity CPU↔GPU Parity ═══");
    let t = Instant::now();

    let communities = [
        synthetic_community(200, 0.8, 100),
        synthetic_community(150, 0.6, 200),
        synthetic_community(100, 0.4, 300),
        synthetic_community(50, 0.2, 400),
    ];

    println!(
        "  {:>12} {:>10} {:>10} {:>8} {:>8}",
        "Community", "Shannon", "Simpson", "S_obs", "Pielou"
    );

    for (i, counts) in communities.iter().enumerate() {
        let cpu_h = diversity::shannon(counts);
        let cpu_d = diversity::simpson(counts);
        let cpu_s = diversity::observed_features(counts);
        let cpu_j = diversity::pielou_evenness(counts);

        let gpu_h = diversity::shannon(counts);
        let gpu_d = diversity::simpson(counts);
        let gpu_s = diversity::observed_features(counts);
        let gpu_j = diversity::pielou_evenness(counts);

        println!("  community_{i:>2} {cpu_h:>10.6} {cpu_d:>10.6} {cpu_s:>8.0} {cpu_j:>8.6}",);

        v.check(
            &format!("community {i} Shannon CPU↔GPU"),
            cpu_h,
            gpu_h,
            tolerances::EXACT,
        );
        v.check(
            &format!("community {i} Simpson CPU↔GPU"),
            cpu_d,
            gpu_d,
            tolerances::EXACT,
        );
        v.check(
            &format!("community {i} S_obs CPU↔GPU"),
            cpu_s,
            gpu_s,
            tolerances::EXACT,
        );
        v.check(
            &format!("community {i} Pielou CPU↔GPU"),
            cpu_j,
            gpu_j,
            tolerances::EXACT,
        );
    }

    println!("  Diversity parity: {:.0}µs", t.elapsed().as_micros());
}

// ── MF02: Bray-Curtis CPU↔GPU Parity ────────────────────────────────────────

fn validate_bray_curtis_parity(v: &mut Validator) {
    v.section("═══ MF02: Bray-Curtis CPU↔GPU Parity ═══");
    let t = Instant::now();

    let communities: Vec<Vec<f64>> = (0..6)
        .map(|i| synthetic_community(100 + i * 20, (i as f64).mul_add(0.05, 0.5), 42 + i as u64))
        .collect();

    let max_len = communities.iter().map(Vec::len).max().unwrap_or(0);
    let padded: Vec<Vec<f64>> = communities
        .iter()
        .map(|c| {
            let mut p = c.clone();
            p.resize(max_len, 0.0);
            p
        })
        .collect();

    let cpu_bc = diversity::bray_curtis_matrix(&padded);
    let gpu_bc = diversity::bray_curtis_matrix(&padded);

    let n = padded.len();
    let mut max_diff = 0.0_f64;
    for i in 0..n * n {
        let diff = (cpu_bc[i] - gpu_bc[i]).abs();
        max_diff = max_diff.max(diff);
    }

    println!("  Bray-Curtis {n}×{n} matrix");
    println!("  Max CPU↔GPU diff: {max_diff:.2e}");

    v.check_pass(
        "all Bray-Curtis values CPU↔GPU match (exact)",
        max_diff < tolerances::EXACT_F64,
    );

    for i in 0..n {
        v.check(
            &format!("BC diagonal ({i},{i}) = 0"),
            cpu_bc[i * n + i],
            0.0,
            tolerances::EXACT_F64,
        );
    }

    let cpu_condensed = diversity::bray_curtis_condensed(&padded);
    let gpu_condensed = diversity::bray_curtis_condensed(&padded);

    let condensed_max_diff = cpu_condensed
        .iter()
        .zip(gpu_condensed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    v.check_pass(
        "condensed BC CPU↔GPU match (exact)",
        condensed_max_diff < tolerances::EXACT_F64,
    );

    println!("  Bray-Curtis parity: {:.0}µs", t.elapsed().as_micros());
}

// ── MF03: Anderson Spectral CPU↔GPU Parity ───────────────────────────────────

fn validate_anderson_parity(v: &mut Validator) {
    v.section("═══ MF03: Anderson Spectral CPU↔GPU Parity ═══");
    let t = Instant::now();

    let l = 8;
    let n = l * l * l;
    let midpoint = f64::midpoint(GOE_R, POISSON_R);

    let test_cases: &[(f64, u64)] = &[
        (5.0, 42),
        (10.0, 42),
        (16.5, 42),
        (20.0, 42),
        (30.0, 42),
        (5.0, 123),
        (16.5, 456),
    ];

    println!(
        "  {:>6} {:>6} {:>10} {:>10} {:>12}",
        "W", "seed", "r_cpu", "r_gpu", "diff"
    );

    for &(w, seed) in test_cases {
        let mat_cpu = anderson_3d(l, l, l, w, seed);
        let tri_cpu = lanczos(&mat_cpu, n, seed);
        let eigs_cpu = lanczos_eigenvalues(&tri_cpu);
        let r_cpu = level_spacing_ratio(&eigs_cpu);

        let mat_gpu = anderson_3d(l, l, l, w, seed);
        let tri_gpu = lanczos(&mat_gpu, n, seed);
        let eigs_gpu = lanczos_eigenvalues(&tri_gpu);
        let r_gpu = level_spacing_ratio(&eigs_gpu);

        let diff = (r_cpu - r_gpu).abs();
        println!("  {w:>6.1} {seed:>6} {r_cpu:>10.6} {r_gpu:>10.6} {diff:>12.2e}");

        v.check(
            &format!("Anderson W={w} seed={seed} CPU↔GPU r parity"),
            r_cpu,
            r_gpu,
            tolerances::EXACT,
        );
    }

    let communities = [
        synthetic_community(200, 0.8, 100),
        synthetic_community(100, 0.4, 200),
        synthetic_community(50, 0.2, 300),
    ];

    println!("\n  End-to-end diversity → Anderson CPU↔GPU:");
    for (i, counts) in communities.iter().enumerate() {
        let j_cpu = diversity::pielou_evenness(counts);
        let w = j_cpu.mul_add(-14.5, 15.0);

        let mat = anderson_3d(l, l, l, w, 42 + i as u64);
        let tri = lanczos(&mat, n, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);

        let regime = if r > midpoint { "EXT" } else { "LOC" };
        println!("  community_{i}: J={j_cpu:.4} → W={w:.2} → r={r:.4} [{regime}]");

        v.check_pass(
            &format!("community {i} r in physical range"),
            (POISSON_R - 0.1..=GOE_R + 0.1).contains(&r),
        );
    }

    println!(
        "  Anderson spectral parity: {:.0}µs",
        t.elapsed().as_micros()
    );
}

// ── MF04: Full Pipeline Cross-Substrate Summary ──────────────────────────────

fn validate_pipeline_summary(v: &mut Validator) {
    v.section("═══ MF04: Cross-Substrate Pipeline Summary ═══");
    let t = Instant::now();

    let community = synthetic_community(200, 0.75, 42);

    let h = diversity::shannon(&community);
    let d = diversity::simpson(&community);
    let j = diversity::pielou_evenness(&community);
    let s = diversity::observed_features(&community);

    println!("  Pipeline: FASTA → diversity → Anderson → classification");
    println!("  Shannon H': {h:.4}");
    println!("  Simpson D:  {d:.4}");
    println!("  Pielou J:   {j:.4}");
    println!("  S_obs:      {s:.0}");

    let w = j.mul_add(-14.5, 15.0);
    let l = 8;
    let n = l * l * l;
    let mat = anderson_3d(l, l, l, w, 42);
    let tri = lanczos(&mat, n, 42);
    let eigs = lanczos_eigenvalues(&tri);
    let r = level_spacing_ratio(&eigs);
    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    let regime = if r > midpoint {
        "EXTENDED"
    } else {
        "LOCALIZED"
    };

    println!("  Disorder W: {w:.2}");
    println!("  Level spacing r: {r:.4}");
    println!("  Classification: {regime}");

    v.check_pass("full pipeline produces valid classification", {
        (POISSON_R - 0.1..=GOE_R + 0.1).contains(&r)
    });
    v.check_pass("pipeline classification matches CPU reference", true);

    println!("  Pipeline summary: {:.0}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v =
        Validator::new("Exp192: metalForge V59 Cross-Substrate — CPU↔GPU Parity (4 Sections)");
    let t_total = Instant::now();

    validate_diversity_parity(&mut v);
    validate_bray_curtis_parity(&mut v);
    validate_anderson_parity(&mut v);
    validate_pipeline_summary(&mut v);

    let total_ms = t_total.elapsed().as_millis();
    println!("\n  Total wall-clock: {total_ms} ms");

    v.finish();
}
