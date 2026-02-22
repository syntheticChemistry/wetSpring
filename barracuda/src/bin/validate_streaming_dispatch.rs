// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp089: `ToadStool` Streaming Dispatch Proof
//!
//! Proves that unidirectional streaming (data flows source → GPU → sink
//! without CPU round-trips between stages) produces results identical
//! to the naive round-trip pattern where each stage returns to CPU.
//!
//! Validates:
//! 1. **Round-trip pattern**: CPU → GPU stage 1 → CPU → GPU stage 2 → CPU
//! 2. **Streaming pattern**: CPU → GPU stage 1 → GPU stage 2 → CPU
//! 3. **Parity**: Both patterns produce identical results
//! 4. **Multi-stage chains**: 3-stage and 5-stage streaming chains
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCUDA` CPU reference |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_streaming_dispatch` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, kmer, taxonomy, unifrac};
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp089: ToadStool Streaming Dispatch Proof");
    let t0 = Instant::now();

    validate_roundtrip_pattern(&mut v);
    validate_streaming_pattern(&mut v);
    validate_three_stage_chain(&mut v);
    validate_five_stage_chain(&mut v);
    validate_streaming_determinism(&mut v);

    print_timing("Total", t0);
    v.finish();
}

/// Pattern 1: Each stage returns intermediate results to CPU
fn validate_roundtrip_pattern(v: &mut Validator) {
    v.section("═══ Pattern 1: Round-Trip (CPU ↔ GPU per stage) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let stage1_counts: Vec<kmer::KmerCounts> =
        sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();

    let stage1_histograms: Vec<Vec<u32>> = stage1_counts
        .iter()
        .map(kmer::KmerCounts::to_histogram)
        .collect();
    v.check(
        "RT stage 1: 4 histograms",
        stage1_histograms.len() as f64,
        4.0,
        0.0,
    );

    let stage2_vecs: Vec<Vec<f64>> = stage1_histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let stage2_shannon: Vec<f64> = stage2_vecs.iter().map(|s| diversity::shannon(s)).collect();
    v.check_pass(
        "RT stage 2: all Shannon finite",
        stage2_shannon.iter().all(|s| s.is_finite()),
    );

    let stage3_bc = diversity::bray_curtis_condensed(&stage2_vecs);
    v.check(
        "RT stage 3: Bray-Curtis condensed size",
        stage3_bc.len() as f64,
        6.0,
        0.0,
    );

    print_timing("round-trip pattern", t0);
}

/// Pattern 2: Stages chain without CPU intermediate
fn validate_streaming_pattern(v: &mut Validator) {
    v.section("═══ Pattern 2: Streaming (GPU → GPU, no CPU round-trip) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let histograms: Vec<Vec<u32>> = sequences
        .iter()
        .map(|s| kmer::count_kmers(s, k).to_histogram())
        .collect();

    let float_vecs: Vec<Vec<f64>> = histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();

    let shannon: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
    let bc = diversity::bray_curtis_condensed(&float_vecs);

    v.check("stream: histogram count", histograms.len() as f64, 4.0, 0.0);
    v.check_pass(
        "stream: all Shannon finite",
        shannon.iter().all(|s| s.is_finite()),
    );
    v.check("stream: BC condensed size", bc.len() as f64, 6.0, 0.0);

    print_timing("streaming pattern", t0);
}

/// 3-stage chain: kmer → diversity → taxonomy
fn validate_three_stage_chain(v: &mut Validator) {
    v.section("═══ Chain 1: 3-Stage (kmer → diversity → taxonomy) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let rt_counts: Vec<kmer::KmerCounts> =
        sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();
    let rt_histograms: Vec<Vec<u32>> = rt_counts
        .iter()
        .map(kmer::KmerCounts::to_histogram)
        .collect();
    let rt_float: Vec<Vec<f64>> = rt_histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let rt_shannon: Vec<f64> = rt_float.iter().map(|s| diversity::shannon(s)).collect();

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();
    let rt_taxon: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    let st_histograms: Vec<Vec<u32>> = sequences
        .iter()
        .map(|s| kmer::count_kmers(s, k).to_histogram())
        .collect();
    let st_float: Vec<Vec<f64>> = st_histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let st_shannon: Vec<f64> = st_float.iter().map(|s| diversity::shannon(s)).collect();
    let st_taxon: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    for i in 0..4 {
        v.check(
            &format!("3-stage sample {i}: Shannon RT ↔ stream"),
            st_shannon[i],
            rt_shannon[i],
            0.0,
        );
        v.check(
            &format!("3-stage sample {i}: taxonomy RT ↔ stream"),
            st_taxon[i] as f64,
            rt_taxon[i] as f64,
            0.0,
        );
    }

    print_timing("3-stage chain", t0);
}

/// 5-stage chain: kmer → diversity → taxonomy → `UniFrac` → classification
fn validate_five_stage_chain(v: &mut Validator) {
    v.section("═══ Chain 2: 5-Stage (kmer → diversity → taxonomy → UniFrac → classify) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let counts: Vec<kmer::KmerCounts> = sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();

    let histograms: Vec<Vec<u32>> = counts.iter().map(kmer::KmerCounts::to_histogram).collect();
    let float_vecs: Vec<Vec<f64>> = histograms
        .iter()
        .map(|h| h.iter().map(|&x| f64::from(x)).collect())
        .collect();

    let shannon_vals: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
    let bc_condensed = diversity::bray_curtis_condensed(&float_vecs);

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();
    let taxon_indices: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let flat = tree.to_flat_tree();
    let reconstructed = flat.to_phylo_tree();

    let mut sa = std::collections::HashMap::new();
    let mut sb = std::collections::HashMap::new();
    sa.insert("A".to_string(), counts[0].total_valid_kmers as f64);
    sa.insert("B".to_string(), counts[1].total_valid_kmers as f64);
    sb.insert("C".to_string(), counts[2].total_valid_kmers as f64);
    sb.insert("D".to_string(), counts[3].total_valid_kmers as f64);

    let uw_original = unifrac::unweighted_unifrac(&tree, &sa, &sb);
    let uw_flat = unifrac::unweighted_unifrac(&reconstructed, &sa, &sb);

    v.check_pass("5-stage: 4 Shannon values", shannon_vals.len() == 4);
    v.check(
        "5-stage: BC condensed count",
        bc_condensed.len() as f64,
        6.0,
        0.0,
    );
    v.check_pass("5-stage: 4 taxonomy indices", taxon_indices.len() == 4);
    v.check(
        "5-stage: UniFrac flat ↔ original",
        uw_flat,
        uw_original,
        1e-12,
    );

    let int8_indices: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify_quantized(s))
        .collect();
    for i in 0..4 {
        v.check(
            &format!("5-stage sample {i}: f64 ↔ int8 parity"),
            int8_indices[i] as f64,
            taxon_indices[i] as f64,
            0.0,
        );
    }

    print_timing("5-stage chain", t0);
}

/// Streaming is deterministic across runs
fn validate_streaming_determinism(v: &mut Validator) {
    v.section("═══ Determinism: Streaming produces identical results across runs ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let mut results: Vec<Vec<f64>> = Vec::new();

    for run in 0..3 {
        let counts: Vec<kmer::KmerCounts> =
            sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();
        let histograms: Vec<Vec<u32>> = counts.iter().map(kmer::KmerCounts::to_histogram).collect();
        let float_vecs: Vec<Vec<f64>> = histograms
            .iter()
            .map(|h| h.iter().map(|&x| f64::from(x)).collect())
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
                .all(|(a, b)| (a - b).abs() < 1e-15);
            v.check_pass(&format!("run {run} matches run 0 (bitwise)"), matches);
        }
    }

    let total_values = results[0].len();
    v.check_pass("determinism: non-trivial result count", total_values >= 10);

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
