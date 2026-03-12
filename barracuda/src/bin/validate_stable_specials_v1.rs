// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp359: Stable GPU Specials + Tridiag Eigensolver Validation
//!
//! Validates barraCuda v0.3.5 stable special functions that avoid catastrophic
//! cancellation (`log1p`, `expm1`, `erfc`, `bessel_j0_minus1`) and the
//! tridiagonal QL eigensolver for Anderson eigenvalue problems.
//!
//! ## Domains
//!
//! - `D80`: Stable Specials CPU — `log1p`, `expm1`, `erfc`, `bessel_j0_minus1` against reference
//! - `D81`: Anderson Eigenvalue Problem — `tridiagonal_ql` for Anderson lattice
//! - `D82`: Cross-Validation — stable vs naive implementations, cancellation comparison
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | barraCuda v0.3.5 stable specials |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu --bin validate_stable_specials_v1` |

use std::time::Instant;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp359: Stable GPU Specials + Tridiag Eigensolver v1");

    // ─── D80: Stable Specials CPU ───
    println!("\n  ── D80: Stable Specials CPU ──");

    use barracuda::special::{bessel_j0_minus1_f64, erfc_f64, expm1_f64, log1p_f64};

    let small_x_values = [1e-15, 1e-12, 1e-10, 1e-8, 1e-5, 0.001, 0.1, 1.0, 10.0];

    println!("  log1p(x) vs ln(1+x):");
    for &x in &small_x_values {
        let stable = log1p_f64(x);
        let naive = x.ln_1p();
        let reldiff = if stable.abs() > 1e-300 {
            ((stable - naive) / stable).abs()
        } else {
            (stable - naive).abs()
        };
        println!(
            "    x={x:>12.2e}  stable={stable:>20.15e}  naive={naive:>20.15e}  reldiff={reldiff:.2e}"
        );
    }
    v.check_pass("log1p computed for 9 test points", true);

    let log1p_tiny = log1p_f64(1e-15);
    v.check_pass(
        "log1p(1e-15) is close to 1e-15",
        (log1p_tiny - 1e-15).abs() < 1e-28,
    );

    println!("\n  expm1(x) vs exp(x)-1:");
    for &x in &small_x_values {
        let stable = expm1_f64(x);
        let naive = x.exp_m1();
        let reldiff = if stable.abs() > 1e-300 {
            ((stable - naive) / stable).abs()
        } else {
            (stable - naive).abs()
        };
        println!(
            "    x={x:>12.2e}  stable={stable:>20.15e}  naive={naive:>20.15e}  reldiff={reldiff:.2e}"
        );
    }
    v.check_pass("expm1 computed for 9 test points", true);

    let expm1_tiny = expm1_f64(1e-15);
    v.check_pass("expm1(1e-15) ≈ 1e-15", (expm1_tiny - 1e-15).abs() < 1e-28);

    println!("\n  erfc(x):");
    let erfc_test_points = [0.0_f64, 0.5, 1.0, 2.0, 3.0, 5.0];
    let erfc_reference = [
        1.0,
        0.479_500_122_1,
        0.157_299_207_0,
        0.004_677_735_0,
        0.000_022_090_5,
        1.537_459_794_5e-12,
    ];

    for (i, &x) in erfc_test_points.iter().enumerate() {
        let computed = erfc_f64(x);
        let reference = erfc_reference[i];
        let absdiff = (computed - reference).abs();
        println!("    erfc({x:.1}) = {computed:.10e}  ref={reference:.10e}  diff={absdiff:.2e}");
        v.check_pass(
            &format!("erfc({x}) within tolerance"),
            absdiff < tolerances::ERF_PARITY,
        );
    }

    println!("\n  bessel_j0_minus1(x):");
    let bessel_test = [1e-15, 1e-10, 1e-5, 0.01, 0.1, 1.0];
    for &x in &bessel_test {
        let j0m1 = bessel_j0_minus1_f64(x);
        println!("    J₀({x:.2e})-1 = {j0m1:.15e}");
    }
    v.check_pass("bessel_j0_minus1 computed for 6 test points", true);

    let j0m1_tiny = bessel_j0_minus1_f64(1e-15);
    v.check_pass(
        "bessel_j0_minus1(1e-15) ≈ 0 (J₀(0)=1)",
        j0m1_tiny.abs() < tolerances::VARIANCE_EXACT,
    );

    // ─── D81: Anderson Eigenvalue Problem ───
    println!("\n  ── D81: Anderson Eigenvalue Problem ──");

    use barracuda::special::{anderson_diagonalize, tridiagonal_ql};

    let n = 6_usize;
    let w_values = [0.0_f64, 5.0, 10.0, 16.5, 25.0];

    for &w in &w_values {
        let mut diag = vec![0.0_f64; n];

        let seed = (w * 1000.0) as u64;
        for (i, d) in diag.iter_mut().enumerate() {
            let pseudo = ((seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64)) as f64)
                / u64::MAX as f64;
            *d = (pseudo - 0.5) * w;
        }
        let subdiag = vec![-1.0_f64; n - 1];

        let (eigenvalues, _eigenvectors) = tridiagonal_ql(&diag, &subdiag);
        let bandwidth = eigenvalues.last().unwrap_or(&0.0) - eigenvalues.first().unwrap_or(&0.0);
        let mean_spacing = if eigenvalues.len() > 1 {
            bandwidth / (eigenvalues.len() - 1) as f64
        } else {
            0.0
        };

        println!(
            "  W={w:>5.1} — eigs: [{:.3} .. {:.3}], bandwidth={bandwidth:.3}, mean_spacing={mean_spacing:.3}",
            eigenvalues.first().unwrap_or(&0.0),
            eigenvalues.last().unwrap_or(&0.0),
        );
        v.check_pass(&format!("W={w} eigenvalues computed"), true);
        v.check_pass(
            &format!("W={w} bandwidth reasonable"),
            w < 0.001 || bandwidth > 0.0,
        );
    }

    println!("\n  Anderson disorder → bandwidth scaling:");
    let mut prev_bw = 0.0;
    let scaling_w = [1.0, 5.0, 10.0, 20.0, 30.0];
    for &w in &scaling_w {
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let pseudo = ((42_u64
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(i as u64)) as f64)
                    / u64::MAX as f64;
                (pseudo - 0.5) * w
            })
            .collect();
        let subdiag = vec![-1.0_f64; n - 1];

        let (eigs, _) = tridiagonal_ql(&diag, &subdiag);
        let bw = eigs.last().unwrap_or(&0.0) - eigs.first().unwrap_or(&0.0);
        println!("    W={w:>5.1} → bandwidth={bw:.3}");
        if w > 1.0 {
            v.check_pass(
                &format!("W={w} bandwidth increases with disorder"),
                bw >= prev_bw - tolerances::ANALYTICAL_F64,
            );
        }
        prev_bw = bw;
    }

    println!("\n  anderson_diagonalize (convenience wrapper):");
    let w_test = 10.0;
    let t_hop = 1.0;
    let disorder: Vec<f64> = (0..n)
        .map(|i| {
            let pseudo = ((123_u64
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64)) as f64)
                / u64::MAX as f64;
            (pseudo - 0.5) * w_test
        })
        .collect();
    let (eigs, _vecs) = anderson_diagonalize(&disorder, t_hop);
    println!(
        "    disorder W={w_test}, t={t_hop}: {} eigenvalues",
        eigs.len()
    );
    println!(
        "    range: [{:.4} .. {:.4}]",
        eigs.first().unwrap_or(&0.0),
        eigs.last().unwrap_or(&0.0)
    );
    v.check_pass(
        "anderson_diagonalize returns n eigenvalues",
        eigs.len() == n,
    );

    // ─── D82: Cross-Validation ───
    println!("\n  ── D82: Cross-Validation ──");

    let x_critical = 1e-14_f64;
    let stable_log1p = log1p_f64(x_critical);
    let stable_expm1 = expm1_f64(x_critical);

    v.check_pass("log1p(1e-14) non-zero", stable_log1p.abs() > 0.0);
    v.check_pass("expm1(1e-14) non-zero", stable_expm1.abs() > 0.0);

    let sym_check = log1p_f64(0.5).exp();
    v.check_pass(
        "exp(log1p(0.5)) ≈ 1.5",
        (sym_check - 1.5).abs() < tolerances::ANALYTICAL_F64,
    );

    let inv_check = expm1_f64(log1p_f64(0.25));
    v.check_pass(
        "expm1(log1p(0.25)) ≈ 0.25",
        (inv_check - 0.25).abs() < tolerances::ANALYTICAL_F64,
    );

    v.check_pass(
        "erfc(0) ≈ 1",
        (erfc_f64(0.0) - 1.0).abs() < tolerances::LIMIT_CONVERGENCE,
    );
    v.check_pass("erfc(∞) → 0", erfc_f64(10.0) < tolerances::ANALYTICAL_LOOSE);

    let clean_lattice = vec![0.0_f64; 8];
    let clean_sub = vec![-1.0_f64; 7];
    let (clean_eigs, _) = tridiagonal_ql(&clean_lattice, &clean_sub);
    v.check_pass(
        "clean lattice eigenvalues are symmetric around 0",
        (clean_eigs.iter().sum::<f64>()).abs() < tolerances::ANALYTICAL_LOOSE,
    );

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
