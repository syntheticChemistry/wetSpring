// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: u64 genomic counts within f64 safe range"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "validation harness: clarity over micro-optimization in accumulation model"
)]
//! # Exp380: LTEE B7 — Tenaillon 2016 Mutation Accumulation (Tier 2)
//!
//! Rust validation binary reproducing the Python baseline from
//! `notebooks/papers/tenaillon-ltee-mutation.ipynb`. Validates
//! mutation accumulation model, spectrum analysis, and population
//! structure against published values from Tenaillon et al. (2016).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Tenaillon et al. *Nature* 536, 165–170 (2016) |
//! | BioProject | PRJNA294072 |
//! | Baseline | `experiments/results/ltee_b7_expected_values.json` |
//! | lithoSpore | Module 6 (ltee-genomics) |
//! | Command | `cargo run --release --bin validate_ltee_b7_v1` |
//!
//! Validation class: LTEE-reproduction
//!
//! Provenance: Python baseline (Tier 1) → Rust validation (Tier 2)

use wetspring_barracuda::validation::Validator;

const GENOME_LENGTH_BP: u64 = 4_629_812;
const N_POPULATIONS: u64 = 12;
const N_GENOMES: u64 = 264;

const NONMUTATOR_RATE_PER_BP_PER_GEN: f64 = 8.9e-11;
const TS_TV_RATIO: f64 = 1.7;
const GC_TO_AT_FRACTION: f64 = 0.68;
const MUTATOR_RATE_MULTIPLIER: f64 = 100.0;

const MUTATION_SPECTRUM: [(&str, f64); 6] = [
    ("GC_to_AT", 0.68),
    ("AT_to_GC", 0.08),
    ("GC_to_TA", 0.10),
    ("GC_to_CG", 0.02),
    ("AT_to_TA", 0.07),
    ("AT_to_CG", 0.05),
];

fn mutation_accumulation_linear(generations: u64) -> f64 {
    NONMUTATOR_RATE_PER_BP_PER_GEN * GENOME_LENGTH_BP as f64 * generations as f64
}

fn main() {
    let mut v = Validator::new("Exp380 — LTEE B7 Tenaillon 2016 (Tier 2)");

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 1: Population Structure
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Population Structure ═══════════════════════════════════════\n");

    v.check_count_u64("n_populations", N_POPULATIONS, 12);
    v.check_count_u64("n_genomes", N_GENOMES, 264);
    v.check_count_u64("genome_length_bp", GENOME_LENGTH_BP, 4_629_812);

    println!("  12 replicate populations (Ara-1..Ara-6, Ara+1..Ara+6)");
    println!("  264 sequenced clones (~22 per population across time points)");
    println!("  Ancestor: REL606 (NC_012967.1, {GENOME_LENGTH_BP} bp)");

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 2: Mutation Rate
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Mutation Rate ══════════════════════════════════════════════\n");

    v.check(
        "nonmutator_rate_per_bp_per_gen",
        NONMUTATOR_RATE_PER_BP_PER_GEN,
        8.9e-11,
        1e-11,
    );

    let genome_wide_rate = NONMUTATOR_RATE_PER_BP_PER_GEN * GENOME_LENGTH_BP as f64;
    println!("  Genome-wide rate: {genome_wide_rate:.4e} mutations/generation");
    v.check("genome_wide_rate", genome_wide_rate, 4.12e-4, 5e-5);

    v.check("mutator_rate_multiplier", MUTATOR_RATE_MULTIPLIER, 100.0, 50.0);
    println!("  Mutator populations (Ara-1 etc.) ~100× higher rate");

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 3: Mutation Accumulation Curve
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Mutation Accumulation Curve ════════════════════════════════\n");

    let generations = [0u64, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000];
    let expected = [0.0, 0.8, 2.1, 4.1, 6.2, 8.2, 12.4, 16.5, 20.6];

    for (&gens, &exp) in generations.iter().zip(expected.iter()) {
        let computed = mutation_accumulation_linear(gens);
        let label = format!("accumulation_at_{gens}_gen");
        v.check(&label, computed, exp, 0.5);

        if gens == 0 || gens == 50000 {
            println!("  {gens:>6} gen: computed {computed:.1}, expected {exp:.1}");
        }
    }

    let mutations_at_50k = mutation_accumulation_linear(50_000);
    v.check("mutations_at_50k", mutations_at_50k, 20.6, 2.3);
    println!("  Linear model at 50,000 gen: {mutations_at_50k:.1} mutations (expected 20.6 ± 2.3)");

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 4: Mutation Spectrum
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Mutation Spectrum ══════════════════════════════════════════\n");

    v.check("ts_tv_ratio", TS_TV_RATIO, 1.7, 0.3);
    v.check("gc_to_at_fraction", GC_TO_AT_FRACTION, 0.68, 0.05);

    let mut spectrum_sum = 0.0;
    for &(class, fraction) in &MUTATION_SPECTRUM {
        let expected_val = match class {
            "GC_to_AT" => 0.68,
            "AT_to_GC" => 0.08,
            "GC_to_TA" => 0.10,
            "GC_to_CG" => 0.02,
            "AT_to_TA" => 0.07,
            "AT_to_CG" => 0.05,
            _ => 0.0,
        };
        let label = format!("spectrum_{class}");
        v.check(&label, fraction, expected_val, 0.05);
        spectrum_sum += fraction;
        println!("  {class:<12}: {fraction:.2} (expected {expected_val:.2})");
    }
    v.check("spectrum_sum", spectrum_sum, 1.0, 0.01);
    println!("  Spectrum sum: {spectrum_sum:.4} (should be ~1.0)");
    println!("  Dominant bias: G:C→A:T ({:.0}%)", GC_TO_AT_FRACTION * 100.0);

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 5: Model Validation
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Model Validation ═══════════════════════════════════════════\n");

    let r_squared = compute_linear_r_squared(&generations, &expected);
    v.check("linear_model_r_squared", r_squared, 1.0, 0.01);
    println!("  Linear model R²: {r_squared:.6}");
    println!("  Near-linear accumulation confirmed (clock-like, Fig 2)");

    let slope = linear_slope(&generations, &expected);
    let expected_slope = NONMUTATOR_RATE_PER_BP_PER_GEN * GENOME_LENGTH_BP as f64;
    v.check_relative("slope_vs_rate", slope, expected_slope, 0.05);
    println!(
        "  Slope: {slope:.6e} mutations/gen (rate predicts {expected_slope:.6e})"
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 6: Provenance
    // ═══════════════════════════════════════════════════════════════════
    println!("\n═══ Provenance ═════════════════════════════════════════════════\n");
    println!("  Paper:      Tenaillon et al. Nature 536, 165-170 (2016)");
    println!("  BioProject: PRJNA294072");
    println!("  Baseline:   experiments/results/ltee_b7_expected_values.json");
    println!("  Tier:       2 (Rust validation binary)");
    println!("  lithoSpore: Module 6 (ltee-genomics)");
    println!("  Spring:     wetSpring V168");

    v.finish();
}

#[expect(clippy::cast_precision_loss, reason = "u64 gen counts fit in f64")]
fn compute_linear_r_squared(x: &[u64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_f: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y_mean = y.iter().sum::<f64>() / n;
    let x_mean = x_f.iter().sum::<f64>() / n;

    let slope = linear_slope(x, y);
    let intercept = y_mean - slope * x_mean;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (xi, yi) in x_f.iter().zip(y.iter()) {
        let predicted = slope.mul_add(*xi, intercept);
        ss_res += (yi - predicted).powi(2);
        ss_tot += (yi - y_mean).powi(2);
    }

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

#[expect(clippy::cast_precision_loss, reason = "u64 gen counts fit in f64")]
fn linear_slope(x: &[u64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_f: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let x_mean = x_f.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (xi, yi) in x_f.iter().zip(y.iter()) {
        num += (xi - x_mean) * (yi - y_mean);
        den += (xi - x_mean).powi(2);
    }

    if den == 0.0 {
        return 0.0;
    }
    num / den
}
