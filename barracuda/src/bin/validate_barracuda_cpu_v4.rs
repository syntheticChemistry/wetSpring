// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! `BarraCUDA` CPU Parity v4 — Track 1c domains (deep-sea metagenomics).
//!
//! Extends v1-v3 (18 domains) with the 5 Track 1c modules:
//! ANI, SNP calling, dN/dS, molecular clock, and pangenome analysis.
//!
//! ```text
//! v1 (9) → v2 (+5 batch) → v3 (+9) → [THIS] v4 (+5 Track 1c) → GPU
//! ```
//!
//! Total: **23 algorithmic domains** validated in pure Rust.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline tool | Pure Python (`scripts/barracuda_cpu_v4_baseline.py`) |
//! | Baseline version | Feb 2026 |
//! | Baseline command | `python3 scripts/barracuda_cpu_v4_baseline.py` |
//! | Baseline date | 2026-02-19 |
//! | Data | Synthetic test vectors (ANI, SNP, dN/dS, clock, pangenome) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::time::Instant;
use wetspring_barracuda::bio::{ani, dnds, molecular_clock, pangenome, snp};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("BarraCUDA CPU v4 — Track 1c (5 Domains)");
    let mut timings: Vec<(&str, f64)> = Vec::new();

    // ════════════════════════════════════════════════════════════════
    //  Domain 19: ANI (Average Nucleotide Identity)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 19: ANI (Goris 2007) ═══");

    let t0 = Instant::now();

    let identical = ani::pairwise_ani(b"ATGATGATG", b"ATGATGATG");
    v.check(
        "ANI: identical → 1.0",
        identical.ani,
        1.0,
        tolerances::EXACT_F64,
    );

    let different = ani::pairwise_ani(b"AAAA", b"TTTT");
    v.check(
        "ANI: completely different → 0.0",
        different.ani,
        0.0,
        tolerances::EXACT_F64,
    );

    let half = ani::pairwise_ani(b"AATT", b"AAGC");
    v.check(
        "ANI: half-match → 0.5",
        half.ani,
        0.5,
        tolerances::EXACT_F64,
    );

    let with_gaps = ani::pairwise_ani(b"A-TG", b"ACTG");
    v.check(
        "ANI: gaps excluded",
        with_gaps.aligned_length as f64,
        3.0,
        0.0,
    );
    v.check(
        "ANI: gap-excluded still 1.0",
        with_gaps.ani,
        1.0,
        tolerances::EXACT_F64,
    );

    let seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"CTGATG"];
    let matrix = ani::ani_matrix(&seqs);
    v.check("ANI: matrix size n*(n-1)/2", matrix.len() as f64, 3.0, 0.0);

    let batch_pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"),
        (b"AAAA", b"TTTT"),
        (b"AATT", b"AAGC"),
    ];
    let batch_results = ani::pairwise_ani_batch(&batch_pairs);
    v.check(
        "ANI batch: identical → 1.0",
        batch_results[0].ani,
        1.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "ANI batch: different → 0.0",
        batch_results[1].ani,
        0.0,
        tolerances::EXACT_F64,
    );
    v.check(
        "ANI batch: half → 0.5",
        batch_results[2].ani,
        0.5,
        tolerances::EXACT_F64,
    );

    let ani_us = t0.elapsed().as_micros();
    timings.push(("ANI (pairwise + matrix + batch)", ani_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 20: SNP Calling
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 20: SNP Calling (Anderson 2017) ═══");

    let t0 = Instant::now();

    let identical_seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGATG"];
    let no_snps = snp::call_snps(&identical_seqs);
    v.check(
        "SNP: identical → 0 variants",
        no_snps.variants.len() as f64,
        0.0,
        0.0,
    );

    let one_snp_seqs: Vec<&[u8]> = vec![b"ATGATG", b"ATGATG", b"ATGTTG"];
    let one_snp = snp::call_snps(&one_snp_seqs);
    v.check(
        "SNP: single variant at pos 3",
        one_snp.variants.len() as f64,
        1.0,
        0.0,
    );
    v.check(
        "SNP: variant position = 3",
        one_snp.variants[0].position as f64,
        3.0,
        0.0,
    );

    let freq_seqs: Vec<&[u8]> = vec![b"A", b"A", b"A", b"T"];
    let freq_result = snp::call_snps(&freq_seqs);
    v.check(
        "SNP: ref freq = 0.75",
        freq_result.variants[0].ref_frequency(),
        0.75,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "SNP: alt freq = 0.25",
        freq_result.variants[0].alt_frequency(),
        0.25,
        tolerances::PYTHON_PARITY,
    );

    let multi_snp_seqs: Vec<&[u8]> = vec![b"ATGATG", b"CTGATG", b"ATGTTG"];
    let multi_density = snp::call_snps(&multi_snp_seqs);
    v.check(
        "SNP: density > 0 for polymorphic",
        f64::from(u8::from(multi_density.snp_density() > 0.0)),
        1.0,
        0.0,
    );

    let flat = snp::call_snps_flat(&one_snp_seqs);
    v.check(
        "SNP flat: same count as AoS",
        flat.positions.len() as f64,
        one_snp.variants.len() as f64,
        0.0,
    );
    v.check(
        "SNP flat: position matches",
        f64::from(flat.positions[0]),
        one_snp.variants[0].position as f64,
        0.0,
    );

    let snp_us = t0.elapsed().as_micros();
    timings.push(("SNP calling (pairwise + flat SoA)", snp_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 21: dN/dS (Nei-Gojobori 1986)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 21: dN/dS (Nei & Gojobori 1986) ═══");

    let t0 = Instant::now();

    let identical_dnds = dnds::pairwise_dnds(b"ATGATGATG", b"ATGATGATG").unwrap();
    v.check(
        "dN/dS: identical → dN=0",
        identical_dnds.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS: identical → dS=0",
        identical_dnds.ds,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // TTT→TTC: Phe→Phe (synonymous at pos 3)
    let syn_result = dnds::pairwise_dnds(b"TTTGCTAAA", b"TTCGCTAAA").unwrap();
    v.check(
        "dN/dS: syn-only → dS > 0",
        f64::from(u8::from(syn_result.ds > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "dN/dS: syn-only → dN = 0",
        syn_result.dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS: syn-only → omega = 0",
        syn_result.omega.unwrap_or(f64::NAN),
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    let mixed = dnds::pairwise_dnds(
        b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT",
        b"ATGGCCGAATTTGCTGCTGCTGCTGCCGCT",
    )
    .unwrap();
    v.check(
        "dN/dS: mixed → syn_sites > 0",
        f64::from(u8::from(mixed.syn_sites > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "dN/dS: mixed → nonsyn_sites > 0",
        f64::from(u8::from(mixed.nonsyn_sites > 0.0)),
        1.0,
        0.0,
    );

    let batch_pairs: Vec<(&[u8], &[u8])> =
        vec![(b"ATGATGATG", b"ATGATGATG"), (b"TTTGCTAAA", b"TTCGCTAAA")];
    let batch = dnds::pairwise_dnds_batch(&batch_pairs);
    v.check(
        "dN/dS batch: first identical dN=0",
        batch[0].as_ref().unwrap().dn,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "dN/dS batch: second syn-only dS>0",
        f64::from(u8::from(batch[1].as_ref().unwrap().ds > 0.0)),
        1.0,
        0.0,
    );

    let dnds_us = t0.elapsed().as_micros();
    timings.push(("dN/dS (Nei-Gojobori + batch)", dnds_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 22: Molecular Clock
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 22: Molecular Clock (Zuckerkandl & Pauling 1965) ═══");

    let t0 = Instant::now();

    let branch_lengths = vec![0.0, 0.1, 0.2, 0.05, 0.05, 0.15, 0.15];
    let parents = vec![None, Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)];

    let clock = molecular_clock::strict_clock(&branch_lengths, &parents, 3500.0, &[]).unwrap();
    v.check(
        "Clock: rate > 0",
        f64::from(u8::from(clock.rate > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Clock: root age = 3500 Ma",
        clock.node_ages[0],
        3500.0,
        1e-6,
    );
    v.check(
        "Clock: child age < root age",
        f64::from(u8::from(clock.node_ages[1] < clock.node_ages[0])),
        1.0,
        0.0,
    );
    v.check(
        "Clock: calibrations satisfied (none set)",
        f64::from(u8::from(clock.calibrations_satisfied)),
        1.0,
        0.0,
    );

    let relaxed = molecular_clock::relaxed_clock_rates(&branch_lengths, &clock.node_ages, &parents);
    let positive_rates: Vec<f64> = relaxed.iter().copied().filter(|&r| r > 0.0).collect();
    let cv = molecular_clock::rate_variation_cv(&positive_rates);
    v.check(
        "Clock: strict tree CV ≈ 0",
        cv,
        0.0,
        tolerances::PYTHON_PARITY,
    );

    let cal = molecular_clock::CalibrationPoint {
        node_id: 0,
        min_age_ma: 3000.0,
        max_age_ma: 4000.0,
    };
    let cal_clock =
        molecular_clock::strict_clock(&branch_lengths, &parents, 3500.0, &[cal]).unwrap();
    v.check(
        "Clock: calibration satisfied",
        f64::from(u8::from(cal_clock.calibrations_satisfied)),
        1.0,
        0.0,
    );

    let bad_cal = molecular_clock::CalibrationPoint {
        node_id: 0,
        min_age_ma: 5000.0,
        max_age_ma: 6000.0,
    };
    let bad_clock =
        molecular_clock::strict_clock(&branch_lengths, &parents, 3500.0, &[bad_cal]).unwrap();
    v.check(
        "Clock: violated calibration fails",
        f64::from(u8::from(!bad_clock.calibrations_satisfied)),
        1.0,
        0.0,
    );

    let clock_us = t0.elapsed().as_micros();
    timings.push(("Molecular clock (strict + relaxed + CV)", clock_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 23: Pangenome Analysis
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 23: Pangenome (Moulana & Anderson 2020) ═══");

    let t0 = Instant::now();

    let clusters = vec![
        pangenome::GeneCluster {
            id: "core1".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "core2".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "core3".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "acc1".into(),
            presence: vec![true, true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "acc2".into(),
            presence: vec![false, true, true, false, false],
        },
        pangenome::GeneCluster {
            id: "uniq1".into(),
            presence: vec![true, false, false, false, false],
        },
        pangenome::GeneCluster {
            id: "uniq2".into(),
            presence: vec![false, false, false, false, true],
        },
    ];

    let pan = pangenome::analyze(&clusters, 5);
    v.check("Pan: core = 3", pan.core_size as f64, 3.0, 0.0);
    v.check("Pan: accessory = 2", pan.accessory_size as f64, 2.0, 0.0);
    v.check("Pan: unique = 2", pan.unique_size as f64, 2.0, 0.0);
    v.check("Pan: total = 7", pan.total_size as f64, 7.0, 0.0);
    v.check(
        "Pan: Heap's alpha computed",
        f64::from(u8::from(pan.heaps_alpha.is_some())),
        1.0,
        0.0,
    );

    let flat = pangenome::presence_matrix_flat(&clusters, 5);
    v.check(
        "Pan flat: length = genes * genomes",
        flat.len() as f64,
        f64::from(7 * 5),
        0.0,
    );
    v.check(
        "Pan flat: first core gene all-1",
        f64::from(flat[0] + flat[1] + flat[2] + flat[3] + flat[4]),
        5.0,
        0.0,
    );

    let enriched_p = pangenome::hypergeometric_pvalue(8, 10, 20, 100);
    v.check(
        "Pan: enriched p < 0.05",
        f64::from(u8::from(enriched_p < 0.05)),
        1.0,
        0.0,
    );

    let not_enriched_p = pangenome::hypergeometric_pvalue(2, 10, 20, 100);
    v.check(
        "Pan: not-enriched p = 1.0",
        not_enriched_p,
        1.0,
        tolerances::PYTHON_PARITY,
    );

    let pvals = vec![0.01, 0.04, 0.03, 0.5];
    let adj = pangenome::benjamini_hochberg(&pvals);
    v.check(
        "Pan BH: all in [0,1]",
        f64::from(u8::from(adj.iter().all(|&p| (0.0..=1.0).contains(&p)))),
        1.0,
        0.0,
    );
    v.check(
        "Pan BH: smallest p adjusted up",
        f64::from(u8::from(adj[0] >= pvals[0])),
        1.0,
        0.0,
    );

    let pan_us = t0.elapsed().as_micros();
    timings.push(("Pangenome (classify + enrichment + BH)", pan_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Timing Summary
    // ════════════════════════════════════════════════════════════════
    v.section("═══ BarraCUDA CPU v4 Timing Summary ═══");
    println!("\n  {:<45} {:>12}", "Domain", "Time (µs)");
    println!("  {}", "-".repeat(60));
    for (name, us) in &timings {
        println!("  {name:<45} {us:>12.0}");
    }
    let total_us: f64 = timings.iter().map(|(_, t)| t).sum();
    println!("  {}", "-".repeat(60));
    println!("  {:<45} {:>12.0}", "TOTAL", total_us);
    println!();

    v.finish();
}
