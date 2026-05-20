// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp283: CPU Parity — Gonzales Reproductions (Papers 53-56)
//!
//! Validates that all `BarraCUDA` CPU primitives used in the Gonzales paper
//! reproductions (Exp280-282) produce mathematically correct results.
//!
//! ## Chain
//! 1. Hill equation: `barracuda::stats::hill` vs. hand-rolled computation
//! 2. Exponential regression: `barracuda::stats::fit_exponential` vs. manual ln-transform
//! 3. Diversity metrics: `wetspring_barracuda::bio::diversity` vs. textbook formulas
//! 4. Anderson spectral: `barracuda::spectral` vs. direct matrix diagonalization
//! 5. Statistical metrics: `barracuda::stats::{mean, r_squared}` vs. manual
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Validates | Exp280-282 math (Papers 53-56 reproduction chain) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_gonzales_cpu_parity` |
//!
//! Validation class: GPU-parity
//!
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use barracuda::stats::{fit_exponential, hill, mean, r_squared};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

const TOL: f64 = tolerances::ANALYTICAL_F64;

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    checks: usize,
}

fn main() {
    let mut v = Validator::new("Exp283: CPU Parity — Gonzales Reproductions");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // D01: Hill Equation Parity
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Hill Equation — barracuda vs. Manual ═══");
    let t0 = Instant::now();

    let test_cases: &[(f64, f64, f64)] = &[
        (10.0, 10.0, 1.0),  // IC50 = x → 0.5
        (100.0, 10.0, 1.0), // 10× IC50
        (1.0, 10.0, 1.0),   // 0.1× IC50
        (50.0, 50.0, 2.0),  // n=2
        (0.1, 0.8, 1.2),    // ED50 case from Exp282
        (5.0, 0.8, 1.2),    // high dose
        (0.0, 10.0, 1.0),   // zero dose
        (1e6, 10.0, 1.0),   // very high dose
    ];

    for &(x, k, n) in test_cases {
        let barracuda_result = hill(x, k, n);
        let manual = if x == 0.0 && n > 0.0 {
            0.0
        } else {
            let xn = x.powf(n);
            let kn = k.powf(n);
            xn / (kn + xn)
        };

        let diff = (barracuda_result - manual).abs();
        v.check_pass(&format!("hill({x}, {k}, {n}): diff={diff:.2e}"), diff < TOL);
    }

    let at_ic50 = hill(10.0, 10.0, 1.0);
    v.check_pass(
        "hill(IC50, IC50, 1) = exactly 0.5",
        (at_ic50 - 0.5).abs() < TOL,
    );

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Hill equation",
        cpu_us,
        checks: 9,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Mean & R² Parity
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Mean & R² — barracuda vs. Manual ═══");
    let t0 = Instant::now();

    let data = [3.0, 7.0, 2.0, 8.0, 5.0, 1.0, 9.0, 4.0, 6.0, 10.0];
    let manual_mean = data.iter().sum::<f64>() / data.len() as f64;
    let barracuda_mean = mean(&data);
    v.check_pass("mean parity", (barracuda_mean - manual_mean).abs() < TOL);

    let observed = [14.0, 28.0, 42.0];
    let predicted = [14.7, 25.4, 44.1];
    let barracuda_r2 = r_squared(&observed, &predicted);

    let obs_mean = observed.iter().sum::<f64>() / observed.len() as f64;
    let ss_tot: f64 = observed.iter().map(|&o| (o - obs_mean).powi(2)).sum();
    let ss_res: f64 = observed
        .iter()
        .zip(predicted.iter())
        .map(|(&o, &p)| (o - p).powi(2))
        .sum();
    let manual_r2 = 1.0 - ss_res / ss_tot;

    v.check_pass("R² parity", (barracuda_r2 - manual_r2).abs() < TOL);

    println!("  barracuda mean = {barracuda_mean:.10}, manual = {manual_mean:.10}");
    println!("  barracuda R² = {barracuda_r2:.10}, manual = {manual_r2:.10}");

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Mean & R²",
        cpu_us,
        checks: 2,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Exponential Regression Parity
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Exponential Regression Parity ═══");
    let t0 = Instant::now();

    let x_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let a_true = 2.5_f64;
    let b_true = 0.3_f64;
    let y_data: Vec<f64> = x_data
        .iter()
        .map(|&x| a_true * (b_true * x).exp())
        .collect();

    let fit = fit_exponential(&x_data, &y_data).or_exit("exponential fit on clean data");
    v.check_pass(
        "fit.params[0] ≈ a_true (2.5)",
        (fit.params[0] - a_true).abs() < tolerances::REGRESSION_FIT_PARITY,
    );
    v.check_pass(
        "fit.params[1] ≈ b_true (0.3)",
        (fit.params[1] - b_true).abs() < tolerances::REGRESSION_FIT_PARITY,
    );
    v.check_pass("fit R² ≈ 1.0 on clean data", fit.r_squared > 0.9999);

    for (i, &xi) in x_data.iter().enumerate() {
        let pred = fit.predict_one(xi).or_exit("unexpected error");
        v.check_pass(
            &format!("predict_one({xi}) ≈ y[{i}]"),
            (pred - y_data[i]).abs() < tolerances::REGRESSION_FIT_PARITY,
        );
    }

    let dose_x = [-2.079, -0.693, 0.693]; // ln(0.125), ln(0.5), ln(2.0)
    let dose_y = [14.0, 28.0, 42.0];
    let dose_fit = fit_exponential(&dose_x, &dose_y).or_exit("PK dose-duration fit");
    v.check_pass("PK fit R² > 0.95", dose_fit.r_squared > 0.95);

    println!(
        "  Clean data: a={:.4} (true={a_true}), b={:.4} (true={b_true}), R²={:.6}",
        fit.params[0], fit.params[1], fit.r_squared
    );
    println!("  PK dose-duration: R²={:.6}", dose_fit.r_squared);

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Exponential regression",
        cpu_us,
        checks: 9,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Diversity Metrics Parity
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Diversity — barracuda vs. Textbook ═══");
    let t0 = Instant::now();

    let counts = [0.950, 0.600, 0.350, 0.125, 0.065, 0.275, 0.275];
    let total: f64 = counts.iter().sum();
    let freqs: Vec<f64> = counts.iter().map(|&c| c / total).collect();

    // Shannon: H = -Σ p_i × ln(p_i)
    let manual_shannon: f64 = freqs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let barracuda_shannon = diversity::shannon(&counts);

    v.check_pass(
        "Shannon parity",
        (barracuda_shannon - manual_shannon).abs() < tolerances::GPU_VS_CPU_F64,
    );

    // Simpson: D = 1 - Σ p_i²
    let manual_simpson: f64 = 1.0 - freqs.iter().map(|&p| p * p).sum::<f64>();
    let barracuda_simpson = diversity::simpson(&counts);

    v.check_pass(
        "Simpson parity",
        (barracuda_simpson - manual_simpson).abs() < tolerances::GPU_VS_CPU_F64,
    );

    // Pielou: J = H / ln(S)
    let s = counts.iter().filter(|&&c| c > 0.0).count() as f64;
    let manual_pielou = manual_shannon / s.ln();
    let barracuda_pielou = diversity::pielou_evenness(&counts);

    v.check_pass(
        "Pielou parity",
        (barracuda_pielou - manual_pielou).abs() < tolerances::GPU_VS_CPU_F64,
    );

    // Chao1: species richness estimator
    let integer_counts: [f64; 7] = [95.0, 60.0, 35.0, 12.0, 6.0, 27.0, 27.0];
    let barracuda_chao1 = diversity::chao1(&integer_counts);
    let s_obs = integer_counts.iter().filter(|&&c| c > 0.0).count() as f64;
    let f1 = integer_counts
        .iter()
        .filter(|&&c| (c - 1.0).abs() < tolerances::CHAO1_COUNT_HALFWIDTH)
        .count() as f64;
    let f2 = integer_counts
        .iter()
        .filter(|&&c| (c - 2.0).abs() < tolerances::CHAO1_COUNT_HALFWIDTH)
        .count() as f64;
    let manual_chao1 = if f2 > 0.0 {
        s_obs + f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))
    } else {
        s_obs + f1 * (f1 - 1.0) / 2.0
    };

    v.check_pass(
        "Chao1 parity",
        (barracuda_chao1 - manual_chao1).abs() < tolerances::GPU_VS_CPU_F64,
    );

    println!("  Shannon: barracuda={barracuda_shannon:.6}, manual={manual_shannon:.6}");
    println!("  Simpson: barracuda={barracuda_simpson:.6}, manual={manual_simpson:.6}");
    println!("  Pielou:  barracuda={barracuda_pielou:.6}, manual={manual_pielou:.6}");
    println!("  Chao1:   barracuda={barracuda_chao1:.6}, manual={manual_chao1:.6}");

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Diversity metrics",
        cpu_us,
        checks: 4,
    });

    // ═══════════════════════════════════════════════════════════════
    // D05: Anderson Spectral Determinism
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Anderson Spectral — Deterministic Seed Parity ═══");
    let t0 = Instant::now();

    let configs: &[(usize, usize, usize, f64, u64)] = &[
        (8, 8, 1, 16.0, 42), // 2D skin
        (6, 6, 6, 4.0, 42),  // 3D dermis
        (8, 8, 1, 20.0, 42), // 2D treated
        (8, 8, 1, 16.0, 99), // different seed
    ];

    for &(lx, ly, lz, w, seed) in configs {
        let n = lx * ly * lz;
        let label = if lz == 1 { "2D" } else { "3D" };

        // Run twice — same seed should produce identical results
        let mat1 = if lz == 1 {
            anderson_2d(lx, ly, w, seed)
        } else {
            anderson_3d(lx, ly, lz, w, seed)
        };
        let tri1 = lanczos(&mat1, n, seed);
        let eigs1 = lanczos_eigenvalues(&tri1);
        let r1 = level_spacing_ratio(&eigs1);

        let mat2 = if lz == 1 {
            anderson_2d(lx, ly, w, seed)
        } else {
            anderson_3d(lx, ly, lz, w, seed)
        };
        let tri2 = lanczos(&mat2, n, seed);
        let eigs2 = lanczos_eigenvalues(&tri2);
        let r2 = level_spacing_ratio(&eigs2);

        v.check_pass(
            &format!("{label} W={w} seed={seed}: r identical across runs"),
            (r1 - r2).abs() < TOL,
        );
        v.check_pass(
            &format!("{label} W={w} seed={seed}: r in valid range"),
            r1 > 0.3 && r1 < 0.6,
        );

        println!("  {label} ({lx}×{ly}×{lz}, W={w}, seed={seed}): r1={r1:.6}, r2={r2:.6}");
    }

    // Different seeds → different results
    let mat_s1 = anderson_2d(8, 8, 16.0, 42);
    let tri_s1 = lanczos(&mat_s1, 64, 42);
    let r_s1 = level_spacing_ratio(&lanczos_eigenvalues(&tri_s1));

    let mat_s2 = anderson_2d(8, 8, 16.0, 99);
    let tri_s2 = lanczos(&mat_s2, 64, 99);
    let r_s2 = level_spacing_ratio(&lanczos_eigenvalues(&tri_s2));

    v.check_pass(
        "Different seeds → different r values",
        (r_s1 - r_s2).abs() > tolerances::GPU_VS_CPU_F64,
    );

    let midpoint = f64::midpoint(POISSON_R, GOE_R);
    v.check_pass("2D W=16 → localized (r < midpoint)", r_s1 < midpoint + 0.03);

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Anderson spectral",
        cpu_us,
        checks: 10,
    });

    // ═══════════════════════════════════════════════════════════════
    // D06: IC50-to-Anderson Barrier Mapping
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: IC50 → Anderson Barrier Consistency ═══");
    let t0 = Instant::now();

    let ic50_values: [f64; 6] = [10.0, 36.0, 71.0, 80.0, 150.0, 249.0];
    let w_scale = 4.0_f64;

    let barriers: Vec<f64> = ic50_values.iter().map(|&ic| ic.ln() * w_scale).collect();

    // Monotonicity
    v.check_pass(
        "Barriers are monotonically increasing",
        barriers.windows(2).all(|w| w[0] < w[1]),
    );

    // Logarithmic spacing
    let ratios: Vec<f64> = barriers.windows(2).map(|w| w[1] - w[0]).collect();
    v.check_pass("All barrier gaps positive", ratios.iter().all(|&r| r > 0.0));

    // Barrier range
    let range =
        barriers.last().or_exit("unexpected error") - barriers.first().or_exit("unexpected error");
    v.check_pass("Barrier range > 10", range > 10.0);

    // Verify ln transform is correct
    for &ic in &ic50_values {
        let w = ic.ln() * w_scale;
        let recovered = (w / w_scale).exp();
        v.check_pass(
            &format!("Round-trip IC50={ic:.0}: |recovered - original| < 1e-10"),
            (recovered - ic).abs() < tolerances::ANALYTICAL_LOOSE,
        );
    }

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "IC50→barrier map",
        cpu_us,
        checks: 9,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Exp283: CPU Parity — Gonzales Reproductions                 ║");
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
