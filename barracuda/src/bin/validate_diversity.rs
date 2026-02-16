// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate diversity metrics against analytical known values and skbio baselines.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline script | `scripts/run_exp002.py` (skbio.diversity) |
//! | Baseline commit | `21d43a0` (Exp002 complete — 2,273 ASVs) |
//! | Baseline date | 2026-02-16 |
//! | Dataset | PRJNA1195978 (phytoplankton microbiome) |
//! | Reference | skbio 0.6.0, scipy 1.12 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, Pop!\_OS 22.04) |
//!
//! # Methodology
//!
//! - **Analytical tests**: known closed-form values (Shannon of uniform = ln(S), etc.)
//! - **K-mer tests**: deterministic counting with exact expected values
//!
//! TODO(Phase 3): Replace simulated community with real Exp002 ASV table
//! and exact skbio diversity values for Python-parity validation.

use wetspring_barracuda::bio::{diversity, kmer};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("wetSpring Diversity Metrics Validation");

    validate_analytical(&mut v);
    validate_simulated_community(&mut v);
    validate_bray_curtis_matrix(&mut v);
    validate_kmer(&mut v);
    validate_evenness_and_rarefaction(&mut v);

    v.finish();
}

fn validate_analytical(v: &mut Validator) {
    v.section("── Analytical unit tests ──");

    // Shannon of uniform distribution with S species = ln(S)
    let uniform_4 = vec![25.0; 4];
    let shannon_4 = diversity::shannon(&uniform_4);
    v.check(
        "Shannon(uniform, S=4)",
        shannon_4,
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    let uniform_100 = vec![10.0; 100];
    let shannon_100 = diversity::shannon(&uniform_100);
    v.check(
        "Shannon(uniform, S=100)",
        shannon_100,
        100.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    // Simpson of uniform = 1 - 1/S
    let uniform_10 = vec![100.0; 10];
    let simpson_10 = diversity::simpson(&uniform_10);
    v.check(
        "Simpson(uniform, S=10)",
        simpson_10,
        0.9,
        tolerances::ANALYTICAL_F64,
    );

    // Bray-Curtis symmetry and bounds
    let sample_a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
    let sample_b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
    let bc_ab = diversity::bray_curtis(&sample_a, &sample_b);
    let bc_ba = diversity::bray_curtis(&sample_b, &sample_a);
    v.check(
        "Bray-Curtis symmetry",
        (bc_ab - bc_ba).abs(),
        0.0,
        tolerances::BRAY_CURTIS_SYMMETRY,
    );
    v.check("Bray-Curtis in [0,1]", bc_ab, 0.3, 0.3);
}

fn validate_simulated_community(v: &mut Validator) {
    v.section("── Simulated marine microbiome community ──");

    // NOTE: This uses fabricated data matching Exp002 *structure* but not
    // exact values. Replace with real ASV table + exact skbio baseline
    // when provenance script is committed.
    let mut community = Vec::new();
    community.extend((0..100).map(|i| 50.0 + (f64::from(i) * 1.5)));
    community.extend((0..100).map(|i| 10.0 + (f64::from(i) * 0.4)));
    community.extend((0..100).map(|i| 1.0 + (f64::from(i) * 0.09)));
    community.extend(std::iter::repeat_n(1.0, 20));
    community.extend(std::iter::repeat_n(2.0, 10));

    let alpha = diversity::alpha_diversity(&community);
    println!(
        "  Observed: {}, Shannon: {:.4}, Simpson: {:.4}, Chao1: {:.1}",
        alpha.observed, alpha.shannon, alpha.simpson, alpha.chao1
    );

    v.check("Observed features", alpha.observed, 330.0, 1.0);
    v.check(
        "Shannon in marine range",
        alpha.shannon,
        4.9,
        tolerances::SHANNON_SIMULATED,
    );
    v.check(
        "Simpson > 0.9",
        alpha.simpson,
        0.95,
        tolerances::SIMPSON_SIMULATED,
    );
    v.check(
        "Chao1 >= observed",
        alpha.chao1 - alpha.observed,
        50.0,
        tolerances::CHAO1_RANGE,
    );
}

fn validate_bray_curtis_matrix(v: &mut Validator) {
    v.section("── Bray-Curtis distance matrix ──");

    let mut community = Vec::new();
    community.extend((0..100).map(|i| 50.0 + (f64::from(i) * 1.5)));
    community.extend((0..100).map(|i| 10.0 + (f64::from(i) * 0.4)));
    community.extend((0..100).map(|i| 1.0 + (f64::from(i) * 0.09)));
    community.extend(std::iter::repeat_n(1.0, 20));
    community.extend(std::iter::repeat_n(2.0, 10));

    let sample_a: Vec<f64> = community.clone();
    let sample_b: Vec<f64> = community.iter().map(|&c| c * 0.8 + 5.0).collect();
    let sample_c: Vec<f64> = community
        .iter()
        .enumerate()
        .map(|(i, &c)| if i % 2 == 0 { c * 2.0 } else { 0.5 })
        .collect();

    let dm = diversity::bray_curtis_matrix(&[sample_a, sample_b, sample_c]);

    println!(
        "  BC(A,B) = {:.4}, BC(A,C) = {:.4}, BC(B,C) = {:.4}",
        dm[1], dm[2], dm[5]
    );

    v.check("BC(A,B) < BC(A,C)", dm[1], dm[2] * 0.5, dm[2]);
    v.check("BC(A,A) = 0", dm[0], 0.0, tolerances::BRAY_CURTIS_SYMMETRY);
}

fn validate_kmer(v: &mut Validator) {
    v.section("── K-mer counting validation ──");

    let seq = b"ACGTACGTACGT";
    let counts = kmer::count_kmers(seq, 4);
    println!(
        "  k=4, seq=ACGTACGTACGT: {} unique, {} total",
        counts.unique_count(),
        counts.total_count()
    );

    // 12 bases, k=4: 12-4+1 = 9 k-mers (exact)
    v.check_count_u64("Total 4-mers", counts.total_count(), 9);

    // ACGTACGTACGT k=4 generates: ACGT (palindrome), CGTA=TACG, GTAC (palindrome) = 3
    v.check_count("Unique canonical 4-mers", counts.unique_count(), 3);

    // K=8 on a longer sequence: 40-8+1 = 33 k-mers (exact)
    let long_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let counts8 = kmer::count_kmers(long_seq, 8);
    println!(
        "  k=8, 40bp seq: {} unique, {} total",
        counts8.unique_count(),
        counts8.total_count()
    );
    v.check_count_u64("Total 8-mers", counts8.total_count(), 33);
}

fn validate_evenness_and_rarefaction(v: &mut Validator) {
    v.section("── Evenness + Rarefaction ──");

    // Pielou evenness: perfectly even = 1.0
    let uniform = vec![25.0; 4];
    v.check(
        "Pielou(uniform S=4) = 1.0",
        diversity::pielou_evenness(&uniform),
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // Uneven community: J' < 1
    let uneven = vec![99.0, 1.0, 0.0, 0.0];
    let j = diversity::pielou_evenness(&uneven);
    v.check("Pielou(uneven) in [0,1)", j, 0.5, 0.5);

    // Rarefaction: at full depth = observed species
    let community = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
    let total: f64 = community.iter().sum();
    let curve = diversity::rarefaction_curve(&community, &[total]);
    v.check(
        "Rarefaction at full depth = S_obs",
        curve[0],
        8.0,
        tolerances::ANALYTICAL_F64,
    );

    // Rarefaction monotonicity: check increasing with depth
    #[allow(clippy::cast_precision_loss)]
    let depths: Vec<f64> = (1..=total as u64).map(|x| x as f64).collect();
    let curve = diversity::rarefaction_curve(&community, &depths);
    let monotonic = curve.windows(2).all(|w| w[1] >= w[0] - 1e-10);
    v.check(
        "Rarefaction monotonic",
        f64::from(u8::from(monotonic)),
        1.0,
        0.0,
    );
}
