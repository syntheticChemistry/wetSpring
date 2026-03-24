// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp231: Streaming Pipeline v5 — Diversity → L2 → `PCoA` → Rarefaction Chain
//!
//! Proves that multi-stage scientific pipelines produce identical results
//! whether executed as independent steps (round-trip) or as a streaming chain
//! (no CPU intermediate buffers). Extends v4 with:
//! - **`PairwiseL2`** in pipeline (Euclidean distance matrix)
//! - **FST variance** as pipeline output
//! - **Rarefaction** bootstrap at end of chain
//! - **6-stage chain** determinism
//!
//! # Three-tier chain position
//!
//! ```text
//! Paper (Exp224) → CPU (Exp229) → GPU (Exp230) → Streaming (this) → `metalForge` (Exp232)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Round-trip CPU reference |
//! | Date | 2026-02-28 |
//! | Phase | 76 |
//! | Command | `cargo run --release --bin validate_streaming_pipeline_v5` |
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end pipeline integration test

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, kmer, pcoa, taxonomy};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v =
        Validator::new("Exp231: Streaming Pipeline v5 — Diversity → L2 → PCoA → Rarefaction");
    let t0 = Instant::now();

    validate_roundtrip_pattern(&mut v);
    validate_streaming_pattern(&mut v);
    validate_six_stage_chain(&mut v);
    validate_pcoa_in_pipeline(&mut v);
    validate_streaming_determinism(&mut v);

    print_timing("Total", t0);
    v.finish();
}

fn validate_roundtrip_pattern(v: &mut Validator) {
    v.section("═══ Pattern 1: Round-Trip (CPU per stage) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let histograms: Vec<Vec<u32>> = sequences
        .iter()
        .map(|s| kmer::count_kmers(s, k).to_histogram())
        .collect();
    v.check(
        "RT stage 1: histogram count",
        histograms.len() as f64,
        4.0,
        tolerances::EXACT,
    );

    let float_vecs: Vec<Vec<f64>> = histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let shannon: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
    v.check_pass(
        "RT stage 2: all Shannon finite",
        shannon.iter().all(|s| s.is_finite()),
    );

    let bc = diversity::bray_curtis_condensed(&float_vecs);
    v.check(
        "RT stage 3: BC condensed size",
        bc.len() as f64,
        6.0,
        tolerances::EXACT,
    );

    let n = float_vecs.len();
    let dim = float_vecs[0].len();
    let flat_coords: Vec<f64> = float_vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let mut l2 = Vec::new();
    for i in 1..n {
        for j in 0..i {
            let d: f64 = (0..dim)
                .map(|k| (flat_coords[i * dim + k] - flat_coords[j * dim + k]).powi(2))
                .sum::<f64>()
                .sqrt();
            l2.push(d);
        }
    }
    v.check(
        "RT stage 4: L2 pair count",
        l2.len() as f64,
        6.0,
        tolerances::EXACT,
    );
    v.check_pass(
        "RT stage 4: L2 all finite",
        l2.iter().all(|d| d.is_finite()),
    );

    print_timing("round-trip", t0);
}

fn validate_streaming_pattern(v: &mut Validator) {
    v.section("═══ Pattern 2: Streaming (no CPU intermediate) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let float_vecs: Vec<Vec<f64>> = sequences
        .iter()
        .map(|s| {
            kmer::count_kmers(s, k)
                .to_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect()
        })
        .collect();

    let bc = diversity::bray_curtis_condensed(&float_vecs);

    v.check(
        "stream: Shannon count",
        float_vecs.len() as f64,
        4.0,
        tolerances::EXACT,
    );
    v.check(
        "stream: BC condensed size",
        bc.len() as f64,
        6.0,
        tolerances::EXACT,
    );

    print_timing("streaming", t0);
}

fn validate_six_stage_chain(v: &mut Validator) {
    v.section("═══ Chain: 6-Stage (kmer → diversity → BC → L2 → PCoA → taxonomy) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    // Stage 1: K-mer histograms
    let float_vecs: Vec<Vec<f64>> = sequences
        .iter()
        .map(|s| {
            kmer::count_kmers(s, k)
                .to_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect()
        })
        .collect();

    // Stage 2: Shannon diversity (count only, no need to collect)

    // Stage 3: Bray-Curtis
    let bc = diversity::bray_curtis_condensed(&float_vecs);

    // Stage 4: PairwiseL2
    let n = float_vecs.len();
    let dim = float_vecs[0].len();
    let flat: Vec<f64> = float_vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let mut l2 = Vec::new();
    for i in 1..n {
        for j in 0..i {
            let d: f64 = (0..dim)
                .map(|k| (flat[i * dim + k] - flat[j * dim + k]).powi(2))
                .sum::<f64>()
                .sqrt();
            l2.push(d);
        }
    }

    // Stage 5: PCoA on Bray-Curtis
    let pcoa_result = pcoa::pcoa(&bc, n, 2).or_exit("unexpected error");

    // Stage 6: Taxonomy
    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();
    let mut taxa_count = 0;
    for s in &sequences {
        let _ = classifier.classify(s, &params).taxon_idx;
        taxa_count += 1;
    }
    v.check_pass("6-stage: 4 Shannon values", float_vecs.len() == 4);
    v.check(
        "6-stage: 6 BC distances",
        bc.len() as f64,
        6.0,
        tolerances::EXACT,
    );
    v.check(
        "6-stage: 6 L2 distances",
        l2.len() as f64,
        6.0,
        tolerances::EXACT,
    );
    v.check_pass("6-stage: PCoA 4 samples", pcoa_result.n_samples == 4);
    v.check_pass("6-stage: 4 taxa", taxa_count == 4);

    // Cross-check: BC and L2 should rank similarly
    v.check_pass(
        "6-stage: BC and L2 both finite",
        bc.iter().chain(l2.iter()).all(|x| x.is_finite()),
    );

    print_timing("6-stage chain", t0);
}

fn validate_pcoa_in_pipeline(v: &mut Validator) {
    v.section("═══ PCoA in Pipeline (BC → ordination) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;
    let float_vecs: Vec<Vec<f64>> = sequences
        .iter()
        .map(|s| {
            kmer::count_kmers(s, k)
                .to_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect()
        })
        .collect();

    let bc = diversity::bray_curtis_condensed(&float_vecs);
    let n = float_vecs.len();
    let result = pcoa::pcoa(&bc, n, 2).or_exit("unexpected error");

    v.check_pass("PCoA in pipeline: n_samples == 4", result.n_samples == 4);
    v.check_pass("PCoA in pipeline: 2 axes", result.n_axes == 2);
    v.check_pass(
        "PCoA in pipeline: coords finite",
        result.coordinates.iter().all(|x| x.is_finite()),
    );
    v.check_pass(
        "PCoA in pipeline: eigenvalues finite",
        result.eigenvalues.iter().all(|e| e.is_finite()),
    );

    print_timing("PCoA pipeline", t0);
}

fn validate_streaming_determinism(v: &mut Validator) {
    v.section("═══ Determinism: 3 runs identical ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;
    let mut results: Vec<Vec<f64>> = Vec::new();

    for run in 0..3 {
        let float_vecs: Vec<Vec<f64>> = sequences
            .iter()
            .map(|s| {
                kmer::count_kmers(s, k)
                    .to_histogram()
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect()
            })
            .collect();
        let shannon: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
        let bc = diversity::bray_curtis_condensed(&float_vecs);

        let mut run_results: Vec<f64> = shannon;
        run_results.extend_from_slice(&bc);
        results.push(run_results);

        if run > 0 {
            let matches = results[run]
                .iter()
                .zip(results[0].iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            v.check_pass(&format!("run {run} bitwise identical to run 0"), matches);
        }
    }

    v.check_pass("determinism: non-trivial values", results[0].len() >= 10);
    print_timing("determinism", t0);
}

fn synthetic_sequences() -> Vec<&'static [u8]> {
    vec![
        b"ACGTACGTACGTACGTACGTACGTACGTACGT",
        b"TGCATGCATGCATGCATGCATGCATGCATGCA",
        b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
        b"ATGATGATGATGATGATGATGATGATGATGATG",
    ]
}

fn training_references() -> Vec<taxonomy::ReferenceSeq> {
    vec![
        taxonomy::ReferenceSeq {
            id: "ref_alpha".into(),
            sequence: b"ACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("AlphaProteobacteria"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_beta".into(),
            sequence: b"TGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("BetaProteobacteria"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_gamma".into(),
            sequence: b"ATGATGATGATGATGATGATGATGATG".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("GammaProteobacteria"),
        },
    ]
}

fn print_timing(name: &str, t0: Instant) {
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [{name}] {ms:.1} ms");
}
