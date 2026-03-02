// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::float_cmp
)]
//! # Exp247: `ToadStool` S70+++ Rewire Validation
//!
//! Validates wetSpring consumption of `ToadStool` S70+ new stats primitives:
//! - `stats::evolution` — Kimura fixation, Eigen error threshold, detection power/threshold
//! - `stats::jackknife` — leave-one-out resampling, generalized jackknife
//! - `stats::diversity::chao1_classic` — integer-count Chao1 (Chao 1984)
//!
//! These primitives were absorbed from groundSpring into `ToadStool` at S70.
//! wetSpring now consumes them directly for rare biosphere and population genetics.
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | `ToadStool` Pin | S70+++ (`1dd7e338`) |
//! | Command | `cargo run --bin validate_toadstool_s70_rewire` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp247: ToadStool S70+++ Rewire — New Stats Primitives");

    // ═══ S1: Kimura Fixation Probability ════════════════════════════════════
    v.section("S1: Kimura Fixation Probability (evolution.rs)");

    let p_neutral = barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01);
    v.check("Neutral: P_fix == p0 (drift)", p_neutral, 0.01, 1e-8);

    let p_beneficial = barracuda::stats::kimura_fixation_prob(1000, 0.01, 0.01);
    v.check_pass("Beneficial: P_fix > p0", p_beneficial > 0.01);
    println!("  Beneficial P_fix = {p_beneficial:.6} (p0 = 0.01)");

    let p_deleterious = barracuda::stats::kimura_fixation_prob(1000, -0.01, 0.01);
    v.check_pass("Deleterious: P_fix < p0", p_deleterious < 0.01);
    println!("  Deleterious P_fix = {p_deleterious:.6} (p0 = 0.01)");

    let p_strong = barracuda::stats::kimura_fixation_prob(10_000, 0.1, 0.001);
    v.check_pass("Strong selection: P_fix >> p0", p_strong > 10.0 * 0.001);
    println!("  Strong selection P_fix = {p_strong:.6} (Ne=10000, s=0.1, p0=0.001)");

    let p_fixed = barracuda::stats::kimura_fixation_prob(100, 0.1, 1.0);
    v.check("Already fixed: P_fix ≈ 1.0", p_fixed, 1.0, 1e-6);

    // ═══ S2: Eigen Error Threshold ══════════════════════════════════════════
    v.section("S2: Eigen Error Threshold (quasispecies)");

    let mu_c = barracuda::stats::error_threshold(10.0, 100);
    v.check_pass("σ=10, L=100: Some(μ_c)", mu_c.is_some());
    let mu_c_val = mu_c.unwrap();
    v.check_pass("μ_c ∈ (0, 1)", mu_c_val > 0.0 && mu_c_val < 1.0);
    println!("  μ_c(σ=10, L=100) = {mu_c_val:.6}");

    let expected_mu_c = 1.0 - 10.0_f64.powf(-1.0 / 100.0);
    v.check("μ_c matches analytic", mu_c_val, expected_mu_c, 1e-14);

    let mu_c_large = barracuda::stats::error_threshold(2.0, 10_000).unwrap();
    let mu_c_small = barracuda::stats::error_threshold(2.0, 100).unwrap();
    v.check_pass("Longer genome → lower threshold", mu_c_large < mu_c_small);
    println!("  μ_c(L=10000) = {mu_c_large:.8} < μ_c(L=100) = {mu_c_small:.8}");

    v.check_pass(
        "σ ≤ 1 → None (no selective advantage)",
        barracuda::stats::error_threshold(0.5, 100).is_none(),
    );
    v.check_pass(
        "L=0 → None (degenerate genome)",
        barracuda::stats::error_threshold(10.0, 0).is_none(),
    );

    // ═══ S3: Detection Power ════════════════════════════════════════════════
    v.section("S3: Detection Power (rare biosphere)");

    let power_1k = barracuda::stats::detection_power(0.001, 1000);
    v.check_pass("0.1% at 1000 reads: power > 0.5", power_1k > 0.5);
    println!("  P(detect | p=0.001, D=1000) = {power_1k:.4}");

    let expected_power = 1.0 - 0.999_f64.powi(1000);
    v.check("Power matches analytic", power_1k, expected_power, 1e-12);

    let power_10k = barracuda::stats::detection_power(0.001, 10_000);
    v.check_pass("More reads → higher power", power_10k > power_1k);
    println!("  P(detect | p=0.001, D=10000) = {power_10k:.6}");

    v.check(
        "p=0 → power=0",
        barracuda::stats::detection_power(0.0, 10_000),
        0.0,
        1e-15,
    );
    v.check(
        "p=1 → power=1",
        barracuda::stats::detection_power(1.0, 1),
        1.0,
        1e-12,
    );

    // ═══ S4: Detection Threshold ════════════════════════════════════════════
    v.section("S4: Detection Threshold (minimum depth)");

    let depth_95 = barracuda::stats::detection_threshold(0.001, 0.95);
    v.check_pass("0.1% at 95% power: D* > 2000", depth_95 > 2000);
    println!("  D*(p=0.001, P=0.95) = {depth_95}");

    let expected_depth = (1.0 - 0.95_f64).log(1.0 - 0.001_f64).ceil() as u64;
    v.check_count_u64("D* matches analytic", depth_95, expected_depth);
    println!("  Expected D* = {expected_depth}, got {depth_95}");

    let depth_99 = barracuda::stats::detection_threshold(0.001, 0.99);
    v.check_pass("Higher power → more depth needed", depth_99 > depth_95);
    println!("  D*(p=0.001, P=0.99) = {depth_99}");

    v.check_pass(
        "p=0 → D*=0 (impossible)",
        barracuda::stats::detection_threshold(0.0, 0.95) == 0,
    );

    // ═══ S5: Jackknife Mean Variance ════════════════════════════════════════
    v.section("S5: Jackknife Mean Variance");

    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let jk = barracuda::stats::jackknife_mean_variance(&data).unwrap();
    v.check("Mean = 3.0", jk.estimate, 3.0, 1e-12);
    v.check_pass("Variance ≥ 0", jk.variance >= 0.0);
    v.check_pass("Std error ≥ 0", jk.std_error >= 0.0);
    v.check(
        "Std error = sqrt(variance)",
        jk.std_error,
        jk.variance.sqrt(),
        1e-15,
    );
    println!(
        "  Jackknife mean = {:.4}, var = {:.6}, se = {:.6}",
        jk.estimate, jk.variance, jk.std_error
    );

    let constant_data = [7.0; 20];
    let jk_const = barracuda::stats::jackknife_mean_variance(&constant_data).unwrap();
    v.check("Constant data: mean = 7.0", jk_const.estimate, 7.0, 1e-12);
    v.check("Constant data: variance ≈ 0", jk_const.variance, 0.0, 1e-20);

    v.check_pass(
        "Empty → None",
        barracuda::stats::jackknife_mean_variance(&[]).is_none(),
    );
    v.check_pass(
        "Single → None",
        barracuda::stats::jackknife_mean_variance(&[1.0]).is_none(),
    );

    // ═══ S6: Generalized Jackknife ══════════════════════════════════════════
    v.section("S6: Generalized Jackknife");

    let data_gen = [2.0, 4.0, 6.0, 8.0];
    let jk_gen =
        barracuda::stats::jackknife(&data_gen, |d| d.iter().sum::<f64>() / d.len() as f64).unwrap();
    v.check("Generalized mean ≈ 5.0", jk_gen.estimate, 5.0, 1e-10);
    v.check_pass("Generalized variance ≥ 0", jk_gen.variance >= 0.0);
    println!(
        "  Generalized jackknife mean = {:.4}, se = {:.6}",
        jk_gen.estimate, jk_gen.std_error
    );

    let abundances = [10.0, 20.0, 5.0, 3.0, 1.0, 50.0, 8.0, 15.0];
    let jk_shannon = barracuda::stats::jackknife(&abundances, |d| {
        let total: f64 = d.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        -d.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| {
                let p = x / total;
                p * p.ln()
            })
            .sum::<f64>()
    })
    .unwrap();
    v.check_pass("Jackknife Shannon > 0", jk_shannon.estimate > 0.0);
    v.check_pass("Shannon SE > 0 (non-trivial)", jk_shannon.std_error > 0.0);
    println!(
        "  Jackknife Shannon H' = {:.4} ± {:.4}",
        jk_shannon.estimate, jk_shannon.std_error
    );

    // ═══ S7: chao1_classic (Integer Counts) ═════════════════════════════════
    v.section("S7: chao1_classic (Chao 1984, integer counts)");

    let counts: Vec<u64> = vec![10, 5, 3, 2, 1, 1, 0, 0, 20, 7, 1];
    let chao1_int = barracuda::stats::chao1_classic(&counts);
    let s_obs = counts.iter().filter(|&&c| c > 0).count() as f64;
    let f1 = counts.iter().filter(|&&c| c == 1).count() as f64;
    let f2 = counts.iter().filter(|&&c| c == 2).count() as f64;

    v.check_pass("chao1_classic ≥ S_obs", chao1_int >= s_obs);
    println!("  S_obs = {s_obs}, f1 = {f1}, f2 = {f2}");
    println!("  chao1_classic = {chao1_int:.2}");

    let expected_chao1 = s_obs + (f1 * f1) / (2.0 * f2);
    v.check("Matches analytic formula", chao1_int, expected_chao1, 1e-12);

    // chao1 (f64) uses bias-corrected formula: f1*(f1-1)/(2*(f2+1))
    // chao1_classic (u64) uses original Chao 1984: f1^2/(2*f2)
    // Both are valid; classic is the original paper, float version is bias-corrected.
    let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let chao1_float = barracuda::stats::chao1(&counts_f64);
    v.check_pass(
        "Both estimators > S_obs (unseen species detected)",
        chao1_float > s_obs && chao1_int > s_obs,
    );
    println!(
        "  chao1(f64, bias-corrected) = {chao1_float:.2}, chao1_classic(u64, Chao 1984) = {chao1_int:.2}"
    );

    let no_singles: Vec<u64> = vec![10, 5, 3, 2, 20, 7];
    let chao1_no_f1 = barracuda::stats::chao1_classic(&no_singles);
    let s_obs_no = no_singles.iter().filter(|&&c| c > 0).count() as f64;
    v.check(
        "No singletons: chao1 == S_obs",
        chao1_no_f1,
        s_obs_no,
        1e-12,
    );

    let no_doubles: Vec<u64> = vec![10, 5, 3, 1, 1, 1, 20];
    let chao1_no_f2 = barracuda::stats::chao1_classic(&no_doubles);
    let s_obs_nd = no_doubles.iter().filter(|&&c| c > 0).count() as f64;
    let f1_nd = no_doubles.iter().filter(|&&c| c == 1).count() as f64;
    let expected_no_f2 = s_obs_nd + f1_nd * (f1_nd - 1.0) / 2.0;
    v.check(
        "No doubletons: f1*(f1-1)/2 formula",
        chao1_no_f2,
        expected_no_f2,
        1e-12,
    );
    println!("  No-doubleton chao1 = {chao1_no_f2:.2} (expected {expected_no_f2:.2})");

    // ═══ S8: Cross-Validation — Detection Power + Jackknife for Rare Biosphere ══
    v.section("S8: Cross-validation — Rare Biosphere Depth Design via S70+ Stats");

    let rare_abundances = [0.001, 0.005, 0.01, 0.05, 0.1];
    let target_power = 0.95;

    for &p in &rare_abundances {
        let depth = barracuda::stats::detection_threshold(p, target_power);
        let actual_power = barracuda::stats::detection_power(p, depth);
        v.check_pass(
            &format!("p={p}: achieved ≥ target power"),
            actual_power >= target_power,
        );
        println!("  p={p:.3}: D*={depth}, achieved_power={actual_power:.4}");
    }

    // ═══ Summary ════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("ToadStool S70+++ Rewire: New primitives validated");
    println!("  evolution:   kimura_fixation_prob, error_threshold, detection_power/threshold");
    println!("  jackknife:   jackknife_mean_variance, jackknife (generalized)");
    println!("  diversity:   chao1_classic (u64 counts, Chao 1984)");
    println!("  Pin:         S70+++ (1dd7e338)");
    println!("═══════════════════════════════════════════════════════════════");

    v.finish();
}
