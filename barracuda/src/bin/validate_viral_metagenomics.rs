// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp052 — Anderson 2014: Viral metagenomics at hydrothermal vents.
//!
//! Validates diversity metrics on viral vs. cellular functional profiles
//! and the new `bio::dnds` module (Nei-Gojobori 1986) for evolutionary
//! rate estimation. Connects to Waters (phage defense, Exp030).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Anderson et al. (2014) PLoS ONE 9:e109696 |
//! | DOI | 10.1371/journal.pone.0109696 |
//! | Faculty | R. Anderson (Carleton College) |
//! | Baseline script | `scripts/anderson2014_viral_metagenomics.py` |
//! | Baseline output | `experiments/results/052_viral_metagenomics/anderson2014_python_baseline.json` |
//! | Baseline date | 2026-02-20 |
//! | Exact command | `python3 scripts/anderson2014_viral_metagenomics.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |
//! | Python version | 3.10+ (pure Python, no external dependencies) |

use wetspring_barracuda::bio::{diversity, dnds};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp052: Anderson 2014 Viral Metagenomics Validation");

    validate_community_diversity(&mut v);
    validate_dnds_analytical(&mut v);
    validate_dnds_python_parity(&mut v);
    validate_spectral_comparison(&mut v);

    v.finish();
}

fn validate_community_diversity(v: &mut Validator) {
    v.section("── Viral vs. cellular community diversity ──");

    let viral_kegg: Vec<f64> = vec![
        120.0, 80.0, 60.0, 45.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0, 2.0, 1.0,
    ];
    let cellular_kegg: Vec<f64> = vec![
        200.0, 180.0, 150.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0,
        20.0, 15.0, 12.0, 10.0, 8.0, 5.0,
    ];

    let viral_h = diversity::shannon(&viral_kegg);
    let cellular_h = diversity::shannon(&cellular_kegg);

    let py_viral_h = 2.093_470_376_712;
    let py_cellular_h = 2.617_167_672_552;

    v.check(
        "Viral Shannon",
        viral_h,
        py_viral_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Cellular Shannon",
        cellular_h,
        py_cellular_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Cellular diversity > Viral (more functional categories)",
        f64::from(u8::from(cellular_h > viral_h)),
        1.0,
        0.0,
    );
}

fn validate_dnds_analytical(v: &mut Validator) {
    v.section("── dN/dS analytical tests ──");

    // Identical sequences → dN = dS = 0
    let result = dnds::pairwise_dnds(b"ATGATGATG", b"ATGATGATG").unwrap();
    v.check(
        "dN/dS identical: dN = 0",
        result.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS identical: dS = 0",
        result.ds,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS identical: omega = None",
        f64::from(u8::from(result.omega.is_none())),
        1.0,
        0.0,
    );

    // Synonymous only: TTT→TTC (Phe→Phe)
    let result = dnds::pairwise_dnds(b"TTTGCTAAA", b"TTCGCTAAA").unwrap();
    v.check(
        "dN/dS synonymous: dN = 0",
        result.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS synonymous: dS > 0",
        f64::from(u8::from(result.ds > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "dN/dS synonymous: omega = 0",
        result.omega.unwrap_or(f64::NAN),
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Nonsynonymous: AAA→GAA (Lys→Glu)
    let result = dnds::pairwise_dnds(b"AAAGCTGCT", b"GAAGCTGCT").unwrap();
    v.check(
        "dN/dS nonsynonymous: dN > 0",
        f64::from(u8::from(result.dn > 0.0)),
        1.0,
        0.0,
    );

    // Error cases
    v.check(
        "dN/dS: unequal length → error",
        f64::from(u8::from(dnds::pairwise_dnds(b"ATG", b"ATGA").is_err())),
        1.0,
        0.0,
    );
    v.check(
        "dN/dS: not div by 3 → error",
        f64::from(u8::from(dnds::pairwise_dnds(b"AT", b"AT").is_err())),
        1.0,
        0.0,
    );

    // Gap handling
    let result = dnds::pairwise_dnds(b"ATG---GCT", b"ATG---GCT").unwrap();
    v.check(
        "dN/dS: gaps skipped, dN = 0",
        result.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_dnds_python_parity(v: &mut Validator) {
    v.section("── dN/dS Python parity ──");

    // Synonymous case
    let r = dnds::pairwise_dnds(b"TTTGCTAAA", b"TTCGCTAAA").unwrap();
    v.check(
        "Python: syn dS",
        r.ds,
        1.207_078_434_326,
        tolerances::PYTHON_PARITY,
    );

    // Nonsynonymous case
    let r = dnds::pairwise_dnds(b"AAAGCTGCT", b"GAAGCTGCT").unwrap();
    v.check(
        "Python: nonsyn dN",
        r.dn,
        0.167_357_663_486,
        tolerances::PYTHON_PARITY,
    );

    // Mixed case
    let r = dnds::pairwise_dnds(
        b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT",
        b"ATGGCCAAATTTGCTGCTGCTGCTGCCGCT",
    )
    .unwrap();
    v.check(
        "Python: mixed dN = 0",
        r.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: mixed dS",
        r.ds,
        0.320_583_011_120,
        tolerances::PYTHON_PARITY,
    );

    // Purifying selection example
    let r = dnds::pairwise_dnds(
        b"ATGGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTGCT",
        b"ATGGCCGCTGCTGCTGCTGCTGCTGCCGCTGCTGCTGCTGCTGCTGCT",
    )
    .unwrap();
    v.check(
        "Python: purifying dS",
        r.ds,
        0.146_808_432_845,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "Python: purifying dN = 0",
        r.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_spectral_comparison(v: &mut Validator) {
    v.section("── Functional profile vector cosine ──");

    let viral: Vec<f64> = vec![120.0, 80.0, 60.0, 45.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.0];
    let cellular: Vec<f64> = vec![
        200.0, 180.0, 150.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 45.0,
    ];

    let self_sim = vector_cosine(&viral, &viral);
    v.check(
        "Cosine self-similarity = 1.0",
        self_sim,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let cross_sim = vector_cosine(&viral, &cellular);
    v.check(
        "Cosine(viral, cellular) < 1.0",
        f64::from(u8::from(cross_sim < 1.0)),
        1.0,
        0.0,
    );
    v.check(
        "Cosine(viral, cellular) > 0.5 (same domain)",
        f64::from(u8::from(cross_sim > 0.5)),
        1.0,
        0.0,
    );
}

fn vector_cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
