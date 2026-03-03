// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! # Exp196b: Simulated Long-Read 16S Through `BarraCuda` Pipeline
//!
//! Validates that nanopore-length 16S reads (1400–1500 bp) can flow
//! through the existing `BarraCuda` diversity pipeline and produce
//! ecologically meaningful results. Uses synthetic communities with
//! known species abundances.
//!
//! The key insight: nanopore reads are long enough to span the full
//! 16S gene (~1500 bp), eliminating the need for paired-end merging.
//! This simplifies the pipeline: signal → basecall → quality filter →
//! dereplicate → classify → diversity.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | In-repo simulation (this binary) |
//! | Source | Synthetic 16S amplicons, deterministic RNG |
//! | Commit | wetSpring Phase 61 |
//! | Date | 2026-02-26 |
//! | Command | `cargo run --release --bin validate_nanopore_simulated_16s` |
//!
//! ## Simulation parameters
//!
//! - **Read length**: 1450 bp (target; mean ≈ 1450, tolerance 10 bp)
//! - **Substitution rate**: ~5% (1 in 20 bases)
//! - **Seed**: 42 (even community), 100 (uneven community)
//! - **Structure**: Conserved 5'/3' (20 bp each), variable middle, species-specific patterns
//!
//! Validation class: Synthetic
//! Provenance: Generated data with known statistical properties

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

/// Simulate a nanopore 16S read as a byte sequence.
///
/// Generates a synthetic 16S amplicon of `length` bp with a
/// species-specific signature region that enables classification.
fn simulate_16s_read(species_id: usize, length: usize, seed: u64) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut rng = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(species_id as u64);

    let mut read = Vec::with_capacity(length);

    // Conserved 5' region (universal primer site — first 20 bp)
    let conserved_5 = b"AGAGTTTGATCCTGGCTCAG";
    for &b in conserved_5.iter().take(length.min(20)) {
        read.push(b);
    }

    // Variable region: species-specific pattern
    let pattern_base = bases[species_id % 4];
    let pattern_alt = bases[(species_id + 1) % 4];
    while read.len() < length.saturating_sub(20) {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let r = (rng >> 33) as u32;
        if r.is_multiple_of(3) {
            read.push(pattern_base);
        } else if r % 3 == 1 {
            read.push(pattern_alt);
        } else {
            read.push(bases[(r as usize) % 4]);
        }
    }

    // Conserved 3' region (last 20 bp)
    let conserved_3 = b"AAGTCGTAACAAGGTAACC";
    for &b in conserved_3.iter().take(length.saturating_sub(read.len())) {
        read.push(b);
    }

    // Nanopore error simulation: ~5% substitution rate
    for base in &mut read {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        if (rng >> 33).is_multiple_of(20) {
            *base = bases[((rng >> 40) as usize) % 4];
        }
    }

    read
}

/// Generate a synthetic community with `n_species` at given abundances.
fn generate_community_reads(
    abundances: &[f64],
    reads_per_species: usize,
    read_length: usize,
    seed: u64,
) -> Vec<(usize, Vec<u8>)> {
    let total_abundance: f64 = abundances.iter().sum();
    let mut reads = Vec::new();
    let mut rng = seed;

    for (species_id, &abundance) in abundances.iter().enumerate() {
        let n_reads = ((abundance / total_abundance) * reads_per_species as f64).round() as usize;
        for j in 0..n_reads {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let read = simulate_16s_read(species_id, read_length, rng.wrapping_add(j as u64));
            reads.push((species_id, read));
        }
    }

    reads
}

/// Classify a read by comparing its variable region to known patterns.
#[allow(clippy::naive_bytecount)] // sovereign — no external byte-counting crate
fn classify_read(read: &[u8], n_species: usize) -> usize {
    let bases = [b'A', b'C', b'G', b'T'];
    let variable_region = if read.len() > 40 {
        &read[20..read.len().min(200)]
    } else {
        read
    };

    let mut best_species = 0;
    let mut best_score = 0usize;

    for species_id in 0..n_species {
        let pattern_base = bases[species_id % 4];
        let score = variable_region
            .iter()
            .filter(|&&b| b == pattern_base)
            .count();
        if score > best_score {
            best_score = score;
            best_species = species_id;
        }
    }

    best_species
}

fn main() {
    let mut v = Validator::new("Exp196b: Simulated Long-Read 16S Pipeline");

    // ── S1: Even community ─────────────────────────────────────

    v.section("── S1: Even community (4 species, equal abundance) ──");

    let even_abundances = vec![100.0, 100.0, 100.0, 100.0];
    let reads = generate_community_reads(&even_abundances, 400, 1450, 42);

    // Classify reads and build OTU table
    let n_species = even_abundances.len();
    let mut otu_counts = vec![0.0_f64; n_species];
    let mut correct = 0usize;

    for (true_species, read) in &reads {
        let predicted = classify_read(read, n_species);
        otu_counts[predicted] += 1.0;
        if predicted == *true_species {
            correct += 1;
        }
    }

    let total_reads = reads.len();
    let classification_accuracy = correct as f64 / total_reads as f64;
    println!(
        "  Classification: {correct}/{total_reads} ({:.1}%)",
        classification_accuracy * 100.0
    );

    v.check_pass(
        "classification accuracy > 50%",
        classification_accuracy > 0.50,
    );

    let h_even = diversity::shannon(&otu_counts);
    let expected_h = (n_species as f64).ln();
    println!("  Shannon H': {h_even:.4} (expected ~{expected_h:.4} for {n_species} even species)");

    v.check(
        "Shannon(even,4) ≈ ln(4)",
        h_even,
        expected_h,
        tolerances::NANOPORE_DIVERSITY_TOLERANCE,
    );

    let d_even = diversity::simpson(&otu_counts);
    let expected_d = 1.0 - 1.0 / n_species as f64;
    println!("  Simpson D: {d_even:.4} (expected ~{expected_d:.4})");
    v.check(
        "Simpson(even,4) ≈ 0.75",
        d_even,
        expected_d,
        tolerances::NANOPORE_DIVERSITY_TOLERANCE,
    );

    // ── S2: Uneven community ───────────────────────────────────

    v.section("── S2: Uneven community (dominant + rare) ──");

    let uneven_abundances = vec![800.0, 100.0, 50.0, 30.0, 20.0];
    let reads_uneven = generate_community_reads(&uneven_abundances, 500, 1450, 100);

    let mut otu_uneven = vec![0.0_f64; uneven_abundances.len()];
    for (_, read) in &reads_uneven {
        let predicted = classify_read(read, uneven_abundances.len());
        otu_uneven[predicted] += 1.0;
    }

    let h_uneven = diversity::shannon(&otu_uneven);
    println!("  Shannon H' (uneven): {h_uneven:.4}");

    v.check_pass("Shannon(uneven) < Shannon(even)", h_uneven < h_even);

    let dominant_frac = otu_uneven[0] / otu_uneven.iter().sum::<f64>();
    println!("  Dominant species fraction: {dominant_frac:.3}");
    v.check_pass("dominant species > 40% of reads", dominant_frac > 0.40);

    // ── S3: Read length distribution ───────────────────────────

    v.section("── S3: Read length characteristics ──");

    let read_lengths: Vec<usize> = reads.iter().map(|(_, r)| r.len()).collect();
    let mean_length = read_lengths.iter().sum::<usize>() as f64 / read_lengths.len() as f64;

    println!("  Mean read length: {mean_length:.0} bp");
    v.check(
        "mean read length ≈ 1450 bp",
        mean_length,
        1450.0,
        tolerances::NANOPORE_MEAN_READ_LENGTH_TOL,
    );

    v.check_pass(
        "all reads > 1000 bp (full-length 16S)",
        read_lengths.iter().all(|&l| l > 1000),
    );

    // ── S4: Pipeline throughput ────────────────────────────────

    v.section("── S4: Pipeline integration ──");

    let start = std::time::Instant::now();
    let n_iterations = 10_u32;
    for _ in 0..n_iterations {
        let batch = generate_community_reads(&even_abundances, 200, 1450, 42);
        let mut counts = vec![0.0_f64; n_species];
        for (_, read) in &batch {
            let sp = classify_read(read, n_species);
            counts[sp] += 1.0;
        }
        let _h = diversity::shannon(&counts);
        let _d = diversity::simpson(&counts);
    }
    let elapsed = start.elapsed();
    let reads_per_sec = f64::from(n_iterations * 200) / elapsed.as_secs_f64();
    println!("  Throughput: {reads_per_sec:.0} reads/sec (classify + diversity)");
    v.check_pass("throughput > 100 reads/sec", reads_per_sec > 100.0);

    // ── S5: Bray-Curtis between communities ────────────────────

    v.section("── S5: Community comparison (Bray-Curtis) ──");

    let bc_self = diversity::bray_curtis(&otu_counts, &otu_counts);
    v.check(
        "Bray-Curtis(self) = 0.0",
        bc_self,
        0.0,
        tolerances::NANOPORE_SIGNAL_STATS,
    );

    // Pad shorter vector with zeros for Bray-Curtis comparison
    let max_len = otu_counts.len().max(otu_uneven.len());
    let mut otu_even_padded = otu_counts.clone();
    let mut otu_uneven_padded = otu_uneven.clone();
    otu_even_padded.resize(max_len, 0.0);
    otu_uneven_padded.resize(max_len, 0.0);

    let bc_diff = diversity::bray_curtis(&otu_even_padded, &otu_uneven_padded);
    println!("  Bray-Curtis(even vs uneven): {bc_diff:.4}");
    v.check_pass("Bray-Curtis(even vs uneven) > 0.0", bc_diff > 0.0);
    v.check_pass("Bray-Curtis(even vs uneven) < 1.0", bc_diff < 1.0);

    v.finish();
}
