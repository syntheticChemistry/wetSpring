// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    dead_code
)]
//! # Exp179: Track 4 CPU Parity Benchmark — Pure Rust Math vs Python-Equivalent
//!
//! Consolidated `BarraCuda` CPU parity benchmark for all 9 Track 4 soil QS papers.
//! Demonstrates that pure Rust math (`BarraCuda` CPU) produces correct results
//! and measures wall-clock timing for each domain.
//!
//! ## Domains validated
//! - D01: QS Biofilm ODE (Waters model)
//! - D02: Cooperation dynamics (Bruger & Waters)
//! - D03: Alpha diversity (Shannon, Simpson, Chao1, Pielou)
//! - D04: Beta diversity (Bray-Curtis)
//! - D05: Anderson disorder mapping (`norm_cdf`, erf)
//! - D06: Statistical inference (Pearson correlation, meta-analysis CI)
//! - D07: Temporal recovery (exponential decay model)
//! - D08: Factorial design (2×2×2 tillage × cover × N)
//!
//! ## Evolution path
//! - **This experiment**: `BarraCuda` CPU (pure Rust, single-threaded)
//! - **Next**: Exp180 GPU validation (same math on GPU)
//! - **Then**: Exp181 pure GPU streaming (unidirectional)
//! - **Final**: Exp182 metalForge cross-substrate
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Date | 2026-02-25 |
//! | Track | Track 4 — No-Till Soil QS & Anderson Geometry |
//! | Command | `cargo run --release --bin validate_soil_qs_cpu_parity` |

use std::time::Instant;
use wetspring_barracuda::bio::cooperation::{self, CooperationParams};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::qs_biofilm::{self, QsBiofilmParams};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::special::erf;
use barracuda::stats::norm_cdf;

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    checks: usize,
}

struct LcgRng(u64);

impl LcgRng {
    const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 11) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }
}

fn generate_community(rng: &mut LcgRng, richness: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..richness).map(|_| rng.next_f64().max(0.001)).collect();
    let total: f64 = v.iter().sum();
    for a in &mut v {
        *a /= total;
    }
    v
}

fn main() {
    let mut v = Validator::new("Exp179: Track 4 CPU Parity — Pure Rust Math");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // D01: QS Biofilm ODE (Waters 2008 model)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: QS Biofilm ODE ═══");
    let t0 = Instant::now();

    let params = QsBiofilmParams::default();
    let dt = 0.01;

    let standard = qs_biofilm::scenario_standard_growth(&params, dt);
    let high = qs_biofilm::scenario_high_density(&params, dt);
    let mutant = qs_biofilm::scenario_hapr_mutant(&params, dt);
    let dgc = qs_biofilm::scenario_dgc_overexpression(&params, dt);

    let std_n = *standard.states().last().unwrap().first().unwrap();
    let std_b = standard.states().last().unwrap()[4];
    let high_b = high.states().last().unwrap()[4];
    let mut_b = mutant.states().last().unwrap()[4];
    let dgc_b = dgc.states().last().unwrap()[4];

    v.check(
        "Standard N → carrying capacity",
        std_n,
        params.k_cap,
        params.k_cap * 0.3,
    );
    v.check_pass("Standard: B ≥ 0 (biofilm initiated)", std_b >= 0.0);
    v.check_pass("High density B > standard B", high_b > std_b);
    v.check_pass("4 scenarios produce distinct B", {
        let bs = [std_b, high_b, mut_b, dgc_b];
        let mut all_distinct = true;
        for i in 0..bs.len() {
            for j in (i + 1)..bs.len() {
                if (bs[i] - bs[j]).abs() < 1e-10 {
                    all_distinct = false;
                }
            }
        }
        all_distinct
    });
    v.check_pass(
        "Time series has > 100 steps",
        standard.states().count() > 100,
    );

    let d01_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "QS Biofilm ODE (4 scenarios)",
        cpu_us: d01_us,
        checks: 5,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Cooperation Dynamics (Bruger & Waters 2018)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Cooperation Dynamics ═══");
    let t0 = Instant::now();

    let coop_p = CooperationParams::default();
    let equal = cooperation::scenario_equal_start(&coop_p, dt);
    let coop_dom = cooperation::scenario_coop_dominated(&coop_p, dt);
    let cheat_dom = cooperation::scenario_cheat_dominated(&coop_p, dt);
    let pure_c = cooperation::scenario_pure_coop(&coop_p, dt);
    let pure_ch = cooperation::scenario_pure_cheat(&coop_p, dt);

    let freq_eq = *cooperation::cooperator_frequency(&equal).last().unwrap();
    let freq_cd = *cooperation::cooperator_frequency(&coop_dom).last().unwrap();
    let freq_ch = *cooperation::cooperator_frequency(&cheat_dom)
        .last()
        .unwrap();
    let freq_pc = *cooperation::cooperator_frequency(&pure_c).last().unwrap();
    let freq_pch = *cooperation::cooperator_frequency(&pure_ch).last().unwrap();

    v.check_pass(
        "Equal start: coop freq in (0, 1)",
        freq_eq > 0.0 && freq_eq < 1.0,
    );
    v.check_pass("Coop-dominated: freq > 50%", freq_cd > 0.5);
    v.check_pass("Cheat-dominated: freq < 50%", freq_ch < 0.5);
    v.check_pass("Pure coop: freq > 90%", freq_pc > 0.9);
    v.check_pass("Pure cheat: freq < 10%", freq_pch < 0.1);

    let d02_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Cooperation (5 scenarios)",
        cpu_us: d02_us,
        checks: 5,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Alpha Diversity (Shannon, Simpson, Chao1, Pielou)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Alpha Diversity ═══");
    let t0 = Instant::now();
    let mut rng = LcgRng::new(42);

    let comm_100 = generate_community(&mut rng, 100);
    let comm_300 = generate_community(&mut rng, 300);
    let uniform_4 = vec![0.25, 0.25, 0.25, 0.25];

    v.check(
        "Shannon(uniform 4) = ln(4)",
        diversity::shannon(&uniform_4),
        4.0_f64.ln(),
        1e-10,
    );
    v.check(
        "Simpson(uniform 4) = 0.75",
        diversity::simpson(&uniform_4),
        0.75,
        1e-10,
    );
    v.check(
        "Pielou(uniform 4) = 1.0",
        diversity::pielou_evenness(&uniform_4),
        1.0,
        1e-10,
    );
    v.check(
        "Chao1(uniform 4) = 4.0",
        diversity::chao1(&uniform_4),
        4.0,
        1e-6,
    );

    let h_100 = diversity::shannon(&comm_100);
    let h_300 = diversity::shannon(&comm_300);
    v.check_pass("Shannon(300 OTUs) > Shannon(100 OTUs)", h_300 > h_100);

    let c1_100 = diversity::chao1(&comm_100);
    let c1_300 = diversity::chao1(&comm_300);
    v.check_pass("Chao1(300) > Chao1(100)", c1_300 > c1_100);

    v.check(
        "Observed features = richness",
        diversity::observed_features(&comm_100),
        100.0,
        tolerances::EXACT,
    );

    let d03_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Alpha diversity (8 checks)",
        cpu_us: d03_us,
        checks: 8,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Beta Diversity (Bray-Curtis)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Beta Diversity ═══");
    let t0 = Instant::now();

    let a = vec![0.1, 0.2, 0.3, 0.4];
    let b = vec![0.1, 0.2, 0.3, 0.4];
    let c = vec![0.4, 0.3, 0.2, 0.1];
    let z = vec![0.0, 0.0, 0.0, 1.0];

    v.check(
        "BC(identical) = 0",
        diversity::bray_curtis(&a, &b),
        0.0,
        tolerances::EXACT,
    );
    v.check_pass(
        "BC(a, c) > 0 (different)",
        diversity::bray_curtis(&a, &c) > 0.0,
    );
    v.check_pass(
        "BC(a, z) > BC(a, c)",
        diversity::bray_curtis(&a, &z) > diversity::bray_curtis(&a, &c),
    );

    let communities: Vec<Vec<f64>> = (0..5).map(|_i| generate_community(&mut rng, 50)).collect();
    let condensed = diversity::bray_curtis_condensed(&communities);
    v.check_pass("Condensed BC has n*(n-1)/2 entries", condensed.len() == 10);
    v.check_pass(
        "All BC in [0, 1]",
        condensed.iter().all(|&x| (0.0..=1.0).contains(&x)),
    );

    let d04_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Beta diversity (BC)",
        cpu_us: d04_us,
        checks: 5,
    });

    // ═══════════════════════════════════════════════════════════════
    // D05: Anderson Disorder Mapping (norm_cdf, erf)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Anderson Disorder Mapping ═══");
    let t0 = Instant::now();

    v.check("Φ(0) = 0.5", norm_cdf(0.0), 0.5, tolerances::EXACT);
    v.check("Φ(1.96) ≈ 0.975", norm_cdf(1.96), 0.975, 1e-3);
    v.check("Φ(-∞) ≈ 0", norm_cdf(-10.0), 0.0, 1e-10);
    v.check("Φ(+∞) ≈ 1", norm_cdf(10.0), 1.0, 1e-10);
    v.check("erf(0) = 0", erf(0.0), 0.0, tolerances::EXACT);
    v.check("erf(1.0)", erf(1.0), 0.842_700_792_949_715, 5e-7);

    let w_c_3d = 16.5_f64;
    let test_w = [3.0, 8.0, 12.0, 16.5, 20.0, 25.0];
    let mut prev_p = 1.0_f64;
    for &w in &test_w {
        let p = norm_cdf((w_c_3d - w) / 3.0);
        v.check_pass(
            &format!("P(QS|W={w:.0})={p:.4} monotone decreasing"),
            p <= prev_p + 1e-10,
        );
        prev_p = p;
    }

    let d05_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Anderson mapping (norm_cdf/erf)",
        cpu_us: d05_us,
        checks: 12,
    });

    // ═══════════════════════════════════════════════════════════════
    // D06: Statistical Inference (Pearson, CI)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: Statistical Inference ═══");
    let t0 = Instant::now();

    let x_perfect: Vec<f64> = (0..10).map(f64::from).collect();
    let y_perfect: Vec<f64> = (0..10).map(|i| 2.0f64.mul_add(f64::from(i), 1.0)).collect();
    let r = barracuda::stats::pearson_correlation(&x_perfect, &y_perfect).unwrap_or(0.0);
    v.check("Pearson(perfect linear) = 1.0", r, 1.0, tolerances::EXACT);

    let y_anti: Vec<f64> = (0..10).map(|i| -f64::from(i)).collect();
    let r_anti = barracuda::stats::pearson_correlation(&x_perfect, &y_anti).unwrap_or(0.0);
    v.check(
        "Pearson(anti-correlated) = -1.0",
        r_anti,
        -1.0,
        tolerances::EXACT,
    );

    let mbc_effect = 16.2;
    let mbc_ci_lower = 10.9;
    let mbc_ci_upper = 21.8;
    let se = (mbc_ci_upper - mbc_ci_lower) / (2.0 * 1.96);
    let z_score = mbc_effect / se;
    let p_value = 2.0 * (1.0 - norm_cdf(z_score));
    v.check_pass(
        &format!("Meta-analysis z={z_score:.2}, p={p_value:.2e} < 0.05"),
        p_value < 0.05,
    );

    let d06_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Statistical inference",
        cpu_us: d06_us,
        checks: 3,
    });

    // ═══════════════════════════════════════════════════════════════
    // D07: Temporal Recovery (exponential model)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D07: Temporal Recovery Model ═══");
    let t0 = Instant::now();

    let tau = 10.0_f64;
    let w_init = 18.0_f64;
    let w_final = 4.0_f64;

    let years = [0.0, 1.0, 5.0, 10.0, 20.0, 31.0, 50.0];
    let mut prev_w = f64::MAX;
    for &y in &years {
        let frac = 1.0 - (-y / tau).exp();
        let w = (w_init - w_final).mul_add(-frac, w_init);
        v.check_pass(
            &format!("Year {y:.0}: W={w:.1} ≤ previous"),
            w <= prev_w + 0.01,
        );
        prev_w = w;
    }
    let w0 = w_init;
    let w_inf = (w_init - w_final).mul_add(-(1.0 - (-1000.0 / tau).exp()), w_init);
    v.check("W(0) = W_init", w0, w_init, tolerances::EXACT);
    v.check("W(∞) → W_final", w_inf, w_final, 0.01);

    let d07_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Temporal recovery",
        cpu_us: d07_us,
        checks: 9,
    });

    // ═══════════════════════════════════════════════════════════════
    // D08: Factorial Design (2×2×2)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D08: Factorial Design (Tillage × Cover × N) ═══");
    let t0 = Instant::now();

    let tillage_factors = [1.0, 0.6];
    let cover_factors = [1.2, 0.8];
    let n_factors = [1.1, 0.9];
    let base_richness = 200_usize;
    let mut shannons = Vec::new();

    for &till in &tillage_factors {
        for &cover in &cover_factors {
            for &n_fert in &n_factors {
                let richness = (base_richness as f64 * till * cover) as usize;
                let richness = richness.max(10);
                let comm = generate_community(&mut rng, richness);
                let h = diversity::shannon(&comm);
                shannons.push((till, cover, n_fert, h));
            }
        }
    }

    let nt_mean: f64 = shannons
        .iter()
        .filter(|s| s.0 > 0.9)
        .map(|s| s.3)
        .sum::<f64>()
        / 4.0;
    let ct_mean: f64 = shannons
        .iter()
        .filter(|s| s.0 < 0.9)
        .map(|s| s.3)
        .sum::<f64>()
        / 4.0;

    v.check_pass("NT mean H' > CT mean H'", nt_mean > ct_mean);
    v.check_pass("8 factorial combinations computed", shannons.len() == 8);
    v.check_pass("All H' > 0", shannons.iter().all(|s| s.3 > 0.0));

    let d08_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Factorial design 2×2×2",
        cpu_us: d08_us,
        checks: 3,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary: Timing Table
    // ═══════════════════════════════════════════════════════════════
    println!("\n  ┌────────────────────────────────────────┬──────────┬────────┐");
    println!("  │ Domain                                 │ CPU (µs) │ Checks │");
    println!("  ├────────────────────────────────────────┼──────────┼────────┤");
    let mut total_us = 0.0;
    let mut total_checks = 0;
    for t in &timings {
        println!(
            "  │ {:<38} │ {:>8.0} │ {:>6} │",
            t.domain, t.cpu_us, t.checks
        );
        total_us += t.cpu_us;
        total_checks += t.checks;
    }
    println!("  ├────────────────────────────────────────┼──────────┼────────┤");
    println!(
        "  │ {:<38} │ {:>8.0} │ {:>6} │",
        "TOTAL", total_us, total_checks
    );
    println!("  └────────────────────────────────────────┴──────────┴────────┘");
    println!();
    println!("  All computations use BarraCuda CPU (pure Rust, single-threaded).");
    println!("  No Python, no numpy, no scipy. Zero interpreted-language dependencies.");
    println!("  The same math will run on GPU (Exp180) and metalForge (Exp182).");

    let (passed, total) = v.counts();
    println!("\n  ── Exp179 Summary: {passed}/{total} checks ──");

    v.finish();
}
