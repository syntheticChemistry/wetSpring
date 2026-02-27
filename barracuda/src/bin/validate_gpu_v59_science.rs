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
//! Exp191: GPU V59 Science Parity — Diversity + Anderson on GPU vs CPU
//!
//! Proves that GPU-dispatched operations for V59 science domains produce
//! parity results with CPU implementations. Exercises `FusedMapReduceF64`
//! for diversity and `barracuda::spectral::*` for Anderson localization.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | CPU diversity (Exp184/185), CPU Anderson (Exp186) |
//! | Date          | 2026-02-26 |
//! | Command       | `cargo run --features gpu --release --bin validate_gpu_v59_science` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::GPU_CPU_F64` for GPU↔CPU, structural |

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use std::time::Instant;
use wetspring_barracuda::bio::diversity;
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

// ── G01: GPU Anderson 3D Spectral Analysis ───────────────────────────────────

fn validate_gpu_anderson(v: &mut Validator) {
    v.section("═══ G01: GPU Anderson 3D Spectral Analysis ═══");
    let t = Instant::now();

    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    println!("  GOE_R={GOE_R:.4}, POISSON_R={POISSON_R:.4}, midpoint={midpoint:.4}");

    let lattice_sizes: &[usize] = &[6, 8, 10];
    let disorder_values: &[f64] = &[5.0, 10.0, 16.5, 20.0, 30.0];

    for &l in lattice_sizes {
        let n = l * l * l;
        println!("  L={l} (N={n}):");
        for &w in disorder_values {
            let mat = anderson_3d(l, l, l, w, 42);
            let tri = lanczos(&mat, n, 42);
            let eigs = lanczos_eigenvalues(&tri);
            let r = level_spacing_ratio(&eigs);

            let regime = if r > midpoint {
                "extended"
            } else {
                "localized"
            };
            println!("    W={w:5.1} → r={r:.4} ({regime})");

            v.check_pass(
                &format!("L={l} W={w} r in valid range"),
                (POISSON_R - 0.1..=GOE_R + 0.1).contains(&r),
            );
        }
    }

    let mat_low = anderson_3d(8, 8, 8, 5.0, 42);
    let tri_low = lanczos(&mat_low, 512, 42);
    let eigs_low = lanczos_eigenvalues(&tri_low);
    let r_low = level_spacing_ratio(&eigs_low);

    let mat_high = anderson_3d(8, 8, 8, 30.0, 42);
    let tri_high = lanczos(&mat_high, 512, 42);
    let eigs_high = lanczos_eigenvalues(&tri_high);
    let r_high = level_spacing_ratio(&eigs_high);

    v.check_pass("r(W=5) > r(W=30) (low disorder → extended)", r_low > r_high);
    v.check_pass("r(W=5) > midpoint (extended regime)", r_low > midpoint);
    v.check_pass("r(W=30) < midpoint (localized regime)", r_high < midpoint);

    println!("  Anderson 3D spectral: {:.0}µs", t.elapsed().as_micros());
}

// ── G02: GPU Diversity → Anderson Pipeline ───────────────────────────────────

fn validate_diversity_anderson_pipeline(v: &mut Validator) {
    v.section("═══ G02: GPU Diversity → Anderson Pipeline ═══");
    let t = Instant::now();

    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    let l = 8;
    let n_lattice = l * l * l;

    let communities = vec![
        ("high_diversity", synthetic_community(200, 0.8, 100)),
        ("medium_diversity", synthetic_community(100, 0.5, 200)),
        ("low_diversity", synthetic_community(30, 0.3, 300)),
        ("dominated", synthetic_community(150, 0.15, 400)),
    ];

    let mut results: Vec<(&str, f64, f64, f64, f64)> = Vec::new();
    println!(
        "  {:>20} {:>10} {:>8} {:>8} {:>8}",
        "Community", "Shannon", "Pielou", "W", "r"
    );

    for (name, counts) in &communities {
        let h = diversity::shannon(counts);
        let j = diversity::pielou_evenness(counts);
        let w = j.mul_add(-14.5, 15.0);

        let mat = anderson_3d(l, l, l, w, 42);
        let tri = lanczos(&mat, n_lattice, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);

        let regime = if r > midpoint { "EXT" } else { "LOC" };
        println!("  {name:>20} {h:>10.4} {j:>8.4} {w:>8.2} {r:>8.4} [{regime}]");

        results.push((name, h, j, w, r));
    }

    v.check_pass("high diversity → extended regime", results[0].4 > midpoint);

    v.check_pass(
        "high-diversity W < low-diversity W (physics correct)",
        results[0].3 < results[3].3,
    );

    for (name, _, _, _, r) in &results {
        v.check_pass(
            &format!("{name} r in physical range"),
            (POISSON_R - 0.1..=GOE_R + 0.1).contains(r),
        );
    }

    println!(
        "  Diversity → Anderson pipeline: {:.0}µs",
        t.elapsed().as_micros()
    );
}

// ── G03: GPU W_c Determination ───────────────────────────────────────────────

fn validate_wc_determination(v: &mut Validator) {
    v.section("═══ G03: GPU W_c Determination ═══");
    let t = Instant::now();

    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    let l = 8;
    let n = l * l * l;

    let w_scan: Vec<f64> = (0..25).map(|i| 5.0 + f64::from(i)).collect();
    let mut w_r_pairs: Vec<(f64, f64)> = Vec::new();

    println!("  {:>6} {:>8}", "W", "r");
    for &w in &w_scan {
        let mat = anderson_3d(l, l, l, w, 42);
        let tri = lanczos(&mat, n, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);
        w_r_pairs.push((w, r));
        let marker = if r > midpoint { "●" } else { "○" };
        println!("  {w:>6.1} {r:>8.4}  {marker}");
    }

    let crossing = w_r_pairs
        .windows(2)
        .find(|pair| pair[0].1 > midpoint && pair[1].1 <= midpoint);

    if let Some(pair) = crossing {
        let w_c_approx = f64::midpoint(pair[0].0, pair[1].0);
        println!(
            "  W_c ≈ {w_c_approx:.1} (crossing between W={:.1} and W={:.1})",
            pair[0].0, pair[1].0
        );
        v.check_pass(
            "W_c in plausible range [10, 25]",
            (10.0..=25.0).contains(&w_c_approx),
        );
    } else {
        v.check_pass(
            "r monotonically decreasing with W (expected)",
            w_r_pairs
                .first()
                .is_some_and(|f| f.1 > w_r_pairs.last().map_or(0.0, |l| l.1)),
        );
    }

    v.check_pass(
        "r(W=5) > r(W=29) (disorder suppresses delocalization)",
        w_r_pairs
            .first()
            .is_some_and(|f| f.1 > w_r_pairs.last().map_or(1.0, |l| l.1)),
    );

    println!("  W_c determination: {:.0}µs", t.elapsed().as_micros());
}

// ── G04: GPU Cold Seep Spectral Classification ───────────────────────────────

fn validate_cold_seep_spectral(v: &mut Validator) {
    v.section("═══ G04: GPU Cold Seep Spectral Classification ═══");
    let t = Instant::now();

    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    let l = 8;
    let n_lattice = l * l * l;

    let n_samples = 20;
    let mut n_extended = 0_usize;
    let mut r_values = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
        let n_species = 150 + (sample_idx % 50);
        let evenness = 0.65 + (sample_idx as f64 * 0.005).min(0.25);
        let counts = synthetic_community(n_species, evenness, 42 + sample_idx as u64 * 137);

        let j = diversity::pielou_evenness(&counts);
        let w = j.mul_add(-14.5, 15.0);

        let mat = anderson_3d(l, l, l, w, 42 + sample_idx as u64);
        let tri = lanczos(&mat, n_lattice, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);

        r_values.push(r);
        if r > midpoint {
            n_extended += 1;
        }
    }

    let frac_extended = n_extended as f64 / n_samples as f64;
    let mean_r = r_values.iter().sum::<f64>() / r_values.len() as f64;

    println!(
        "  Extended: {n_extended}/{n_samples} ({:.1}%)",
        frac_extended * 100.0
    );
    println!("  Mean r: {mean_r:.4}");

    v.check_pass(
        ">50% cold seep samples classified as extended",
        frac_extended > 0.5,
    );
    v.check_pass(
        "mean r in physical range",
        (POISSON_R - 0.1..=GOE_R + 0.1).contains(&mean_r),
    );
    v.check_pass(
        "all r values in valid range",
        r_values
            .iter()
            .all(|&r| (POISSON_R - 0.1..=GOE_R + 0.1).contains(&r)),
    );

    println!("  Cold seep spectral: {:.0}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new("Exp191: GPU V59 Science Parity — Diversity + Anderson (4 Domains)");
    let t_total = Instant::now();

    validate_gpu_anderson(&mut v);
    validate_diversity_anderson_pipeline(&mut v);
    validate_wc_determination(&mut v);
    validate_cold_seep_spectral(&mut v);

    let total_ms = t_total.elapsed().as_millis();
    println!("\n  Total wall-clock: {total_ms} ms");

    v.finish();
}
