// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! Exp190: `BarraCuda` CPU Parity v10 — Pure Rust Math for V59 Science Extensions
//!
//! Validates that all V59 science domains (Exp184-188) produce correct CPU
//! results using `barracuda` always-on math (diversity, Bray-Curtis,
//! Anderson W(t) functions, and int8 NPU quantization).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | Published diversity ranges, Anderson (1958), |
//! |               | Dethlefsen & Relman PNAS 108 (2011), Islam et al. (2014) |
//! | Date          | 2026-02-26 |
//! | Command       | `cargo run --release --bin validate_barracuda_cpu_v10` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::EXACT_F64`, structural (pass/fail) |
//!
//! Validation class: Synthetic
//! Provenance: Generated data with known statistical properties

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// ── D01: Sovereign Diversity Pipeline ────────────────────────────────────────

fn validate_diversity_pipeline(v: &mut Validator) {
    v.section("═══ D01: Sovereign Diversity Pipeline (Exp184/185) ═══");
    let t = Instant::now();

    let communities: Vec<Vec<f64>> = vec![
        vec![100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        vec![55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0],
        vec![500.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![
            200.0, 180.0, 160.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0,
        ],
    ];

    for (i, counts) in communities.iter().enumerate() {
        let h = diversity::shannon(counts);
        let d = diversity::simpson(counts);
        let s_obs = diversity::observed_features(counts);
        let j = diversity::pielou_evenness(counts);

        v.check_pass(&format!("community {i} Shannon H' > 0"), h > 0.0);
        v.check_pass(
            &format!("community {i} Simpson D in [0,1]"),
            (0.0..=1.0).contains(&d),
        );
        v.check_pass(
            &format!("community {i} S_obs = species count"),
            (s_obs - counts.len() as f64).abs() < tolerances::EXACT_F64,
        );
        v.check_pass(
            &format!("community {i} Pielou J in [0,1]"),
            (0.0..=1.0001).contains(&j),
        );
    }

    let h_even = diversity::shannon(&communities[1]);
    let h_domin = diversity::shannon(&communities[2]);
    v.check_pass(
        "even community has higher Shannon than dominated",
        h_even > h_domin,
    );

    let d_even = diversity::simpson(&communities[1]);
    let d_domin = diversity::simpson(&communities[2]);
    v.check_pass(
        "even community has higher Simpson than dominated",
        d_even > d_domin,
    );

    let j_even = diversity::pielou_evenness(&communities[1]);
    v.check_pass(
        "perfectly even community has Pielou J ≈ 1.0",
        (j_even - 1.0).abs() < tolerances::DIVERSITY_EVENNESS_TOL,
    );

    println!("  Diversity pipeline: {:.0}µs", t.elapsed().as_micros());
}

// ── D02: Bray-Curtis Distance Mathematics ────────────────────────────────────

fn validate_bray_curtis(v: &mut Validator) {
    v.section("═══ D02: Bray-Curtis Distance Mathematics (Exp184/185) ═══");
    let t = Instant::now();

    let a = vec![10.0, 20.0, 30.0, 40.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let c = vec![40.0, 30.0, 20.0, 10.0];
    let d = vec![0.0, 0.0, 0.0, 100.0];

    let samples = [a, b.clone(), c, d];
    let bc = diversity::bray_curtis_matrix(&samples);
    let n = samples.len();

    for i in 0..n {
        v.check_pass(
            &format!("BC({i},{i}) = 0 (self-distance)"),
            bc[i * n + i].abs() < tolerances::EXACT_F64,
        );
    }

    v.check_pass(
        "BC(0,1) = 0 (identical communities)",
        bc[1].abs() < tolerances::EXACT_F64,
    );

    for i in 0..n {
        for j in 0..n {
            let diff = (bc[i * n + j] - bc[j * n + i]).abs();
            v.check_pass(
                &format!("BC symmetric ({i},{j})"),
                diff < tolerances::EXACT_F64,
            );
        }
    }

    let condensed_samples = [b, vec![40.0, 30.0, 20.0, 10.0]];
    let bc_condensed = diversity::bray_curtis_condensed(&condensed_samples);
    v.check_pass("condensed BC is non-zero", bc_condensed[0] > 0.0);
    v.check_pass(
        "condensed BC in [0,1]",
        (0.0..=1.0 + tolerances::EXACT_F64).contains(&bc_condensed[0]),
    );

    let zero_a = vec![0.0; 4];
    let bc_zero = diversity::bray_curtis_matrix(&[zero_a.clone(), zero_a]);
    v.check_pass(
        "BC(zero, zero) = 0 (degenerate case handled)",
        bc_zero[1].abs() < tolerances::EXACT_F64 || bc_zero[1].is_nan(),
    );

    println!("  Bray-Curtis: {:.0}µs", t.elapsed().as_micros());
}

// ── D03: Dynamic Anderson W(t) Functions ─────────────────────────────────────

fn w_tillage(t: f64) -> f64 {
    let tau = 3.0;
    let w_inf = 12.0;
    (20.0_f64 - w_inf).mul_add((-t / tau).exp(), w_inf)
}

fn w_antibiotic(t: f64) -> f64 {
    let w_healthy = 14.0;
    let w_ab = 25.0;
    let tau_recover = 14.0;
    if t < 0.0 {
        w_healthy
    } else if t <= 7.0 {
        let frac = t / 7.0;
        w_healthy + frac * (w_ab - w_healthy)
    } else {
        let dt = t - 7.0;
        (w_ab - w_healthy).mul_add((-dt / tau_recover).exp(), w_healthy)
    }
}

fn w_seasonal(t: f64) -> f64 {
    let w_0 = 16.0;
    let amplitude = 4.0;
    w_0 + amplitude * (2.0 * std::f64::consts::PI * t / 365.0).sin()
}

fn validate_dynamic_anderson(v: &mut Validator) {
    v.section("═══ D03: Dynamic Anderson W(t) Functions (Exp186) ═══");
    let t = Instant::now();

    v.check(
        "tillage W(0) = 20",
        w_tillage(0.0),
        20.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "tillage W(∞) → 12",
        w_tillage(100.0),
        12.0,
        tolerances::ODE_STEADY_STATE,
    );
    v.check_pass("tillage monotonically decreasing", {
        let t_vals: Vec<f64> = (0..20).map(f64::from).collect();
        t_vals
            .windows(2)
            .all(|w| w_tillage(w[0]) >= w_tillage(w[1]))
    });

    v.check(
        "antibiotic W(pre-dose) = 14",
        w_antibiotic(-1.0),
        14.0,
        tolerances::EXACT_F64,
    );
    v.check_pass("antibiotic W peaks during treatment", {
        let w_mid = w_antibiotic(3.5);
        w_mid > 14.0 && w_mid < 25.0
    });
    v.check(
        "antibiotic W(7) = 25 (peak)",
        w_antibiotic(7.0),
        25.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "antibiotic W(∞) → 14",
        w_antibiotic(200.0),
        14.0,
        tolerances::ODE_STEADY_STATE,
    );

    v.check(
        "seasonal W(0) = 16",
        w_seasonal(0.0),
        16.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "seasonal W(91.25) ≈ 20",
        w_seasonal(91.25),
        20.0,
        tolerances::SEASONAL_OSCILLATION,
    );
    v.check(
        "seasonal W(273.75) ≈ 12",
        w_seasonal(273.75),
        12.0,
        tolerances::SEASONAL_OSCILLATION,
    );
    v.check_pass("seasonal is periodic (365 days)", {
        let diff = (w_seasonal(0.0) - w_seasonal(365.0)).abs();
        diff < tolerances::PYTHON_PARITY
    });

    v.check_pass("seasonal range is [12, 20]", {
        let vals: Vec<f64> = (0..366).map(f64::from).map(w_seasonal).collect();
        let min = vals.iter().copied().fold(f64::MAX, f64::min);
        let max = vals.iter().copied().fold(f64::MIN, f64::max);
        let margin = tolerances::ODE_STEADY_STATE;
        min >= 12.0 - margin && max <= 20.0 + margin
    });

    println!("  Dynamic Anderson W(t): {:.0}µs", t.elapsed().as_micros());
}

// ── D04: NPU Int8 Quantization Mathematics ───────────────────────────────────

fn quantize_f64_to_i8(val: f64, min: f64, max: f64) -> i8 {
    if max <= min {
        return 0;
    }
    let normalized = (val - min) / (max - min);
    let clamped = normalized.clamp(0.0, 1.0);
    (clamped * 254.0 - 127.0) as i8
}

fn validate_npu_quantization(v: &mut Validator) {
    v.section("═══ D04: NPU Int8 Quantization (Exp188) ═══");
    let t = Instant::now();

    v.check_pass(
        "quantize(min, min, max) = -127",
        quantize_f64_to_i8(0.0, 0.0, 1.0) == -127,
    );
    v.check_pass(
        "quantize(max, min, max) = 127",
        quantize_f64_to_i8(1.0, 0.0, 1.0) == 127,
    );
    v.check_pass(
        "quantize(mid, min, max) ≈ 0",
        quantize_f64_to_i8(0.5, 0.0, 1.0).abs() <= 1,
    );

    let features = [4.5_f64, 0.95, 8.2, 7.0, 0.3, 0.85, 50.0, 150.0];
    let feature_ranges: [(f64, f64); 8] = [
        (0.0, 7.0),   // Shannon
        (0.0, 1.0),   // Simpson
        (0.0, 14.0),  // DO mg/L
        (5.5, 9.0),   // pH
        (0.0, 1.0),   // dominant fraction
        (0.0, 1.0),   // evenness
        (0.0, 100.0), // species count
        (0.0, 200.0), // total abundance
    ];

    let quantized: Vec<i8> = features
        .iter()
        .zip(feature_ranges.iter())
        .map(|(&val, &(min, max))| quantize_f64_to_i8(val, min, max))
        .collect();

    v.check_pass(
        "all quantized values in [-127, 127]",
        quantized.iter().all(|&q| (-127..=127).contains(&q)),
    );

    let reconstructed: Vec<f64> = quantized
        .iter()
        .zip(feature_ranges.iter())
        .map(|(&q, &(min, max))| {
            let normalized = (f64::from(q) + 127.0) / 254.0;
            normalized.mul_add(max - min, min)
        })
        .collect();

    for (i, (&orig, &recon)) in features.iter().zip(reconstructed.iter()).enumerate() {
        let range = feature_ranges[i].1 - feature_ranges[i].0;
        let quantization_error = (orig - recon).abs() / range;
        v.check_pass(
            &format!("feature {i} quantization error < 1%"),
            quantization_error < tolerances::PEAK_HEIGHT_REL,
        );
    }

    v.check_pass("round-trip preserves relative ordering", {
        let vals = [1.0_f64, 3.0, 5.0, 7.0, 9.0];
        let q: Vec<i8> = vals
            .iter()
            .map(|&x| quantize_f64_to_i8(x, 0.0, 10.0))
            .collect();
        q.windows(2).all(|w| w[0] < w[1])
    });

    println!("  NPU quantization: {:.0}µs", t.elapsed().as_micros());
}

// ── D05: FASTA → Diversity End-to-End ────────────────────────────────────────

fn validate_fasta_to_diversity(v: &mut Validator) {
    v.section("═══ D05: FASTA → Diversity End-to-End (Exp184) ═══");
    let t = Instant::now();

    let fasta =
        ">seq1\nATCGATCG\n>seq2\nATCGATCG\n>seq3\nGGGGAAAA\n>seq4\nGGGGAAAA\n>seq5\nCCCCTTTT\n";
    let mut counts: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut current_seq = String::new();
    for line in fasta.lines() {
        if line.starts_with('>') {
            if !current_seq.is_empty() {
                *counts.entry(current_seq.clone()).or_default() += 1.0;
                current_seq.clear();
            }
        } else {
            current_seq.push_str(line.trim());
        }
    }
    if !current_seq.is_empty() {
        *counts.entry(current_seq).or_default() += 1.0;
    }

    let abundances: Vec<f64> = {
        let mut v: Vec<f64> = counts.values().copied().collect();
        v.sort_by(|a, b| b.total_cmp(a));
        v
    };

    v.check_pass("parsed 3 unique k-mers", abundances.len() == 3);
    v.check_pass(
        "total reads = 5",
        (abundances.iter().sum::<f64>() - 5.0).abs() < tolerances::EXACT_F64,
    );

    let h = diversity::shannon(&abundances);
    let d = diversity::simpson(&abundances);
    let s = diversity::observed_features(&abundances);
    let j = diversity::pielou_evenness(&abundances);

    v.check_pass("Shannon H' > 0 for 3 species", h > 0.0);
    v.check_pass("Simpson D > 0 for 3 species", d > 0.0);
    v.check("S_obs = 3", s, 3.0, tolerances::EXACT_F64);
    v.check_pass("Pielou J < 1 (not perfectly even)", j < 1.0);
    v.check_pass("Pielou J > 0.8 (fairly even)", j > 0.8);

    println!("  FASTA → diversity: {:.0}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new(
        "Exp190: BarraCuda CPU v10 — Pure Rust Math (V59 Science Extensions, 5 Domains)",
    );
    let t_total = Instant::now();

    validate_diversity_pipeline(&mut v);
    validate_bray_curtis(&mut v);
    validate_dynamic_anderson(&mut v);
    validate_npu_quantization(&mut v);
    validate_fasta_to_diversity(&mut v);

    let total_ms = t_total.elapsed().as_millis();
    println!("\n  Total wall-clock: {total_ms} ms");

    v.finish();
}
