// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names, clippy::cast_precision_loss)]
//! Exp055 — Anderson 2017: Population genomics at hydrothermal vents.
//!
//! Validates new ANI and SNP calling modules against analytical known
//! values and Python baselines. Exercises population-level genomic
//! variation analysis for metagenome-assembled genomes.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Anderson et al. (2017) Nature Communications 8:1114 |
//! | DOI | 10.1038/s41467-017-01228-6 |
//! | Faculty | R. Anderson (Carleton College) |
//! | BioProject | PRJNA283159 |
//! | Baseline script | `scripts/anderson2017_population_genomics.py` |
//! | Baseline date | 2026-02-20 |

use wetspring_barracuda::bio::{ani, diversity, dnds, snp};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp055: Anderson 2017 Population Genomics Validation");

    validate_ani(&mut v);
    validate_snp(&mut v);
    validate_integrated_pipeline(&mut v);
    validate_python_parity(&mut v);

    v.finish();
}

fn validate_ani(v: &mut Validator) {
    v.section("── ANI (Average Nucleotide Identity) ──");

    // Self-identity
    let r = ani::pairwise_ani(b"ATGATGATG", b"ATGATGATG");
    v.check("ANI(self) = 1.0", r.ani, 1.0, tolerances::ANALYTICAL_F64);

    // Half identical
    let r = ani::pairwise_ani(b"AATT", b"AAGC");
    v.check("ANI(50%) = 0.5", r.ani, 0.5, tolerances::ANALYTICAL_F64);

    // Symmetry
    let r1 = ani::pairwise_ani(b"ATGATG", b"ATGTTG");
    let r2 = ani::pairwise_ani(b"ATGTTG", b"ATGATG");
    v.check(
        "ANI symmetric",
        (r1.ani - r2.ani).abs(),
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Gap exclusion
    let r = ani::pairwise_ani(b"A-TG", b"ACTG");
    v.check(
        "ANI: gaps excluded, 3/3 aligned",
        r.ani,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_count("ANI: aligned length = 3", r.aligned_length, 3);

    // Same species (> 95%)
    let mut seq2 = b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG".to_vec();
    seq2[0] = b'C';
    seq2[10] = b'C';
    let r = ani::pairwise_ani(
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        &seq2,
    );
    v.check(
        "ANI(same species) > 0.95",
        f64::from(u8::from(r.ani > 0.95)),
        1.0,
        0.0,
    );

    // ANI matrix
    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"CTGATG"];
    let m = ani::ani_matrix(&seqs);
    v.check_count("ANI matrix size = 3", m.len(), 3);
    v.check(
        "ANI matrix[0] = 1.0 (identical pair)",
        m[0],
        1.0,
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_snp(v: &mut Validator) {
    v.section("── SNP calling ──");

    // No variants in identical sequences
    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGATG"];
    let r = snp::call_snps(&seqs);
    v.check_count("Identical: 0 SNPs", r.variants.len(), 0);

    // Single SNP
    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGTTG"];
    let r = snp::call_snps(&seqs);
    v.check_count("Single SNP detected", r.variants.len(), 1);
    v.check_count("SNP at position 3", r.variants[0].position, 3);

    // Multiple SNPs
    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGATG", b"ATGTTG"];
    let r = snp::call_snps(&seqs);
    v.check_count("Multiple SNPs: 2", r.variants.len(), 2);

    // Allele frequency
    let seqs: Vec<&[u8]> = vec![b"A", b"A", b"A", b"T"];
    let r = snp::call_snps(&seqs);
    v.check(
        "Ref frequency = 0.75",
        r.variants[0].ref_frequency(),
        0.75,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Alt frequency = 0.25",
        r.variants[0].alt_frequency(),
        0.25,
        tolerances::ANALYTICAL_F64,
    );

    // SNP density
    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGTTG"];
    let r = snp::call_snps(&seqs);
    v.check(
        "SNP density > 0",
        f64::from(u8::from(r.snp_density() > 0.0)),
        1.0,
        0.0,
    );

    // Empty input
    let seqs: Vec<&[u8]> = vec![];
    let r = snp::call_snps(&seqs);
    v.check_count("Empty: 0 SNPs", r.variants.len(), 0);
}

fn validate_integrated_pipeline(v: &mut Validator) {
    v.section("── Integrated population pipeline ──");

    // Synthetic vent population: 4 genomes from same species
    let genomes: Vec<&[u8]> = vec![
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        b"ATGATGATGATGATGATGATGATGATGATGCTGATGATGATGATGATGATG",
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
    ];

    // ANI: all pairs should be > 95%
    let ani_mat = ani::ani_matrix(&genomes.clone());
    let all_same_sp = ani_mat.iter().all(|&a| a > 0.95);
    v.check(
        "All ANI pairs > 0.95 (same species)",
        f64::from(u8::from(all_same_sp)),
        1.0,
        0.0,
    );

    // SNPs: should find the variants
    let snps = snp::call_snps(&genomes);
    v.check(
        "Population SNPs detected",
        f64::from(u8::from(!snps.variants.is_empty())),
        1.0,
        0.0,
    );

    // dN/dS on the variant region
    let r = dnds::pairwise_dnds(genomes[0], genomes[2]).unwrap();
    v.check(
        "dN/dS computable on population pair",
        f64::from(u8::from(r.syn_sites > 0.0 || r.nonsyn_sites > 0.0)),
        1.0,
        0.0,
    );

    // Shannon diversity of allele frequencies
    let counts: Vec<f64> = vec![2.0, 1.0, 1.0]; // allele distribution
    let h = diversity::shannon(&counts);
    v.check(
        "Population allele Shannon > 0",
        f64::from(u8::from(h > 0.0)),
        1.0,
        0.0,
    );
}

fn validate_python_parity(v: &mut Validator) {
    v.section("── Python baseline parity ──");

    let r = ani::pairwise_ani(b"ATGATGATG", b"ATGATGATG");
    v.check(
        "Python: ANI identical",
        r.ani,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let r = ani::pairwise_ani(b"AATT", b"AAGC");
    v.check("Python: ANI half", r.ani, 0.5, tolerances::ANALYTICAL_F64);

    let mut seq2 = b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG".to_vec();
    seq2[0] = b'C';
    seq2[10] = b'C';
    let r = ani::pairwise_ani(
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        &seq2,
    );
    v.check("Python: ANI same species", r.ani, 0.960_784, 1e-4);

    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGTTG"];
    let r = snp::call_snps(&seqs);
    v.check_count("Python: 1 SNP", r.variants.len(), 1);
}
