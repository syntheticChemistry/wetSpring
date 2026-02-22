// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp085: `BarraCUDA` CPU Parity v7 — Tier A Data Layout Fidelity
//!
//! Proves that the 3 newly Tier A modules (kmer, unifrac, taxonomy) preserve
//! mathematical correctness through their GPU/NPU-ready data layouts:
//!
//! 1. **kmer**: `count_kmers` → `to_histogram` → `from_histogram` → counts match
//! 2. **unifrac**: `PhyloTree` → `to_flat_tree` → `UniFrac` via CSR → matches original
//! 3. **taxonomy**: `train` → `to_int8_weights` → `classify_quantized` → matches f64
//!
//! This extends v1–v6 (205/205 checks) with flat-layout fidelity for GPU/NPU.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_barracuda_cpu_v7` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::collections::HashMap;
use std::time::Instant;
use wetspring_barracuda::bio::{
    kmer,
    taxonomy::{self, Lineage},
    unifrac,
};
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp085: BarraCUDA CPU v7 — Tier A Data Layout Fidelity");
    let t0 = Instant::now();

    validate_kmer_histogram(&mut v);
    validate_kmer_sorted_pairs(&mut v);
    validate_kmer_multi_sequence(&mut v);
    validate_unifrac_flat_tree(&mut v);
    validate_unifrac_weighted(&mut v);
    validate_unifrac_sample_matrix(&mut v);
    validate_taxonomy_int8(&mut v);
    validate_taxonomy_multi_taxon(&mut v);

    print_timing("Total", t0);
    v.finish();
}

fn validate_kmer_histogram(v: &mut Validator) {
    v.section("═══ K-mer: Histogram Round-Trip (k=4) ═══");
    let t0 = Instant::now();

    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let k = 4;
    let original = kmer::count_kmers(seq, k);

    let histogram = original.to_histogram();
    v.check("histogram length = 4^k", histogram.len() as f64, 256.0, 0.0);

    let total_hist: u32 = histogram.iter().sum();
    v.check(
        "histogram total = original total_valid_kmers",
        f64::from(total_hist),
        original.total_valid_kmers as f64,
        0.0,
    );

    let restored = kmer::KmerCounts::from_histogram(&histogram, k);
    v.check(
        "restored total_valid_kmers matches",
        restored.total_valid_kmers as f64,
        original.total_valid_kmers as f64,
        0.0,
    );

    let orig_top = original.top_n(5);
    let rest_top = restored.top_n(5);
    for (i, ((ok, ov), (rk, rv))) in orig_top.iter().zip(rest_top.iter()).enumerate() {
        v.check(
            &format!("top {i} kmer matches"),
            *rk as f64,
            *ok as f64,
            0.0,
        );
        v.check(
            &format!("top {i} count matches"),
            f64::from(*rv),
            f64::from(*ov),
            0.0,
        );
    }

    print_timing("kmer histogram", t0);
}

fn validate_kmer_sorted_pairs(v: &mut Validator) {
    v.section("═══ K-mer: Sorted Pairs Round-Trip (k=6) ═══");
    let t0 = Instant::now();

    let seq = b"ATGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
    let k = 6;
    let original = kmer::count_kmers(seq, k);

    let pairs = original.to_sorted_pairs();
    v.check(
        "pairs count = unique kmers",
        pairs.len() as f64,
        original.unique_count() as f64,
        0.0,
    );

    let is_sorted = pairs.windows(2).all(|w| w[0].0 <= w[1].0);
    v.check_pass("pairs are sorted by kmer value", is_sorted);

    let restored = kmer::KmerCounts::from_sorted_pairs(&pairs, k);
    v.check(
        "restored total matches",
        restored.total_valid_kmers as f64,
        original.total_valid_kmers as f64,
        0.0,
    );
    v.check(
        "restored unique_count matches",
        restored.unique_count() as f64,
        original.unique_count() as f64,
        0.0,
    );

    print_timing("kmer sorted pairs", t0);
}

fn validate_kmer_multi_sequence(v: &mut Validator) {
    v.section("═══ K-mer: Multi-Sequence Histogram (k=4) ═══");
    let t0 = Instant::now();

    let seqs: Vec<&[u8]> = vec![
        b"ACGTACGTACGT",
        b"TGCATGCATGCA",
        b"AAAACCCCGGGG",
        b"TTTTCCCCAAAA",
    ];
    let k = 4;
    let combined = kmer::count_kmers_multi(&seqs, k);

    let hist = combined.to_histogram();
    let restored = kmer::KmerCounts::from_histogram(&hist, k);

    v.check(
        "multi-seq total preserved",
        restored.total_valid_kmers as f64,
        combined.total_valid_kmers as f64,
        0.0,
    );
    v.check(
        "multi-seq unique_count preserved",
        restored.unique_count() as f64,
        combined.unique_count() as f64,
        0.0,
    );

    let pairs = combined.to_sorted_pairs();
    let restored2 = kmer::KmerCounts::from_sorted_pairs(&pairs, k);
    v.check(
        "sorted pairs total also matches",
        restored2.total_valid_kmers as f64,
        combined.total_valid_kmers as f64,
        0.0,
    );

    print_timing("kmer multi-sequence", t0);
}

fn validate_unifrac_flat_tree(v: &mut Validator) {
    v.section("═══ UniFrac: Flat Tree Round-Trip ═══");
    let t0 = Instant::now();

    let newick = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";
    let tree = unifrac::PhyloTree::from_newick(newick);

    let flat = tree.to_flat_tree();
    v.check_pass("flat tree has nodes", !flat.parent.is_empty());
    v.check_pass("flat tree has leaves", !flat.leaf_labels.is_empty());

    let reconstructed = flat.to_phylo_tree();
    v.check(
        "reconstructed node count matches",
        reconstructed.nodes.len() as f64,
        tree.nodes.len() as f64,
        0.0,
    );

    let mut sample_a: HashMap<String, f64> = HashMap::new();
    let mut sample_b: HashMap<String, f64> = HashMap::new();
    sample_a.insert("A".into(), 10.0);
    sample_a.insert("B".into(), 5.0);
    sample_a.insert("C".into(), 1.0);
    sample_b.insert("A".into(), 1.0);
    sample_b.insert("C".into(), 10.0);
    sample_b.insert("D".into(), 5.0);

    let uw_orig = unifrac::unweighted_unifrac(&tree, &sample_a, &sample_b);
    let uw_flat = unifrac::unweighted_unifrac(&reconstructed, &sample_a, &sample_b);
    v.check(
        "unweighted UniFrac: original ↔ flat round-trip",
        uw_flat,
        uw_orig,
        1e-12,
    );

    let w_orig = unifrac::weighted_unifrac(&tree, &sample_a, &sample_b);
    let w_flat = unifrac::weighted_unifrac(&reconstructed, &sample_a, &sample_b);
    v.check(
        "weighted UniFrac: original ↔ flat round-trip",
        w_flat,
        w_orig,
        1e-12,
    );

    print_timing("unifrac flat tree", t0);
}

fn validate_unifrac_weighted(v: &mut Validator) {
    v.section("═══ UniFrac: Weighted Distance Properties ═══");
    let t0 = Instant::now();

    let newick = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";
    let tree = unifrac::PhyloTree::from_newick(newick);

    let mut identical: HashMap<String, f64> = HashMap::new();
    identical.insert("A".into(), 10.0);
    identical.insert("B".into(), 5.0);

    let self_dist = unifrac::weighted_unifrac(&tree, &identical, &identical);
    v.check("weighted self-distance = 0", self_dist, 0.0, 1e-12);

    let mut diff_a: HashMap<String, f64> = HashMap::new();
    let mut diff_b: HashMap<String, f64> = HashMap::new();
    diff_a.insert("A".into(), 100.0);
    diff_b.insert("D".into(), 100.0);

    let max_dist = unifrac::weighted_unifrac(&tree, &diff_a, &diff_b);
    v.check_pass("maximally different samples > 0", max_dist > 0.0);
    v.check_pass("weighted distance is finite", max_dist.is_finite());

    let uw_self = unifrac::unweighted_unifrac(&tree, &identical, &identical);
    v.check("unweighted self-distance = 0", uw_self, 0.0, 1e-12);

    print_timing("unifrac weighted", t0);
}

fn validate_unifrac_sample_matrix(v: &mut Validator) {
    v.section("═══ UniFrac: Sample Matrix Layout ═══");
    let t0 = Instant::now();

    let newick = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);";
    let tree = unifrac::PhyloTree::from_newick(newick);
    let flat = tree.to_flat_tree();

    let mut samples = unifrac::AbundanceTable::new();
    samples.insert(
        "sample1".into(),
        [("A".into(), 10.0), ("B".into(), 5.0)]
            .into_iter()
            .collect(),
    );
    samples.insert(
        "sample2".into(),
        [("C".into(), 8.0), ("D".into(), 3.0)].into_iter().collect(),
    );

    let (matrix, n_samples, n_leaves) = unifrac::to_sample_matrix(&flat, &samples);
    v.check("n_samples = 2", n_samples as f64, 2.0, 0.0);
    v.check(
        "n_leaves matches flat tree",
        n_leaves as f64,
        flat.leaf_labels.len() as f64,
        0.0,
    );
    v.check(
        "matrix size = n_samples × n_leaves",
        matrix.len() as f64,
        (n_samples * n_leaves) as f64,
        0.0,
    );

    let row_sum_0: f64 = matrix[..n_leaves].iter().sum();
    let row_sum_1: f64 = matrix[n_leaves..].iter().sum();
    v.check_pass("sample 1 has abundance > 0", row_sum_0 > 0.0);
    v.check_pass("sample 2 has abundance > 0", row_sum_1 > 0.0);

    print_timing("unifrac sample matrix", t0);
}

fn validate_taxonomy_int8(v: &mut Validator) {
    v.section("═══ Taxonomy: Int8 Quantization Parity ═══");
    let t0 = Instant::now();

    let refs = vec![
        taxonomy::ReferenceSeq {
            id: "ref_vibrio".into(),
            sequence: b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(),
            lineage: Lineage::from_taxonomy_string("Vibrio"),
        },
        taxonomy::ReferenceSeq {
            id: "ref_pseudo".into(),
            sequence: b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA".to_vec(),
            lineage: Lineage::from_taxonomy_string("Pseudomonas"),
        },
    ];

    let k = 4;
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);

    let test_seqs: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"TGCATGCATGCATGCA"];

    let params = taxonomy::ClassifyParams::default();
    for (i, seq) in test_seqs.iter().enumerate() {
        let f64_result = classifier.classify(seq, &params);
        let int8_result_idx = classifier.classify_quantized(seq);

        v.check(
            &format!("seq {i}: int8 argmax matches f64 taxon_idx"),
            int8_result_idx as f64,
            f64_result.taxon_idx as f64,
            0.0,
        );
    }

    let weights = classifier.to_int8_weights();
    v.check_pass("int8 weights are non-empty", !weights.weights_i8.is_empty());
    v.check_pass("scale is positive", weights.scale > 0.0);
    v.check_pass("zero_point is finite", weights.zero_point.is_finite());

    let expected_size = refs.len() * (1_usize << (2 * k));
    v.check(
        "weights size = n_taxa × 4^k",
        weights.weights_i8.len() as f64,
        expected_size as f64,
        0.0,
    );

    print_timing("taxonomy int8", t0);
}

fn validate_taxonomy_multi_taxon(v: &mut Validator) {
    v.section("═══ Taxonomy: Multi-Taxon Classification ═══");
    let t0 = Instant::now();

    let lineages = [
        Lineage::from_taxonomy_string("Bacteroidetes"),
        Lineage::from_taxonomy_string("Firmicutes"),
        Lineage::from_taxonomy_string("Proteobacteria"),
    ];

    let refs = vec![
        taxonomy::ReferenceSeq {
            id: "ref_bact".into(),
            sequence: b"AAAAAACCCCCCGGGGGGTTTTTTAAAAAACCCCCC".to_vec(),
            lineage: lineages[0].clone(),
        },
        taxonomy::ReferenceSeq {
            id: "ref_firm".into(),
            sequence: b"ACACACACACACACACACACACACACACACACACAC".to_vec(),
            lineage: lineages[1].clone(),
        },
        taxonomy::ReferenceSeq {
            id: "ref_prot".into(),
            sequence: b"ATGATGATGATGATGATGATGATGATGATGATGATG".to_vec(),
            lineage: lineages[2].clone(),
        },
    ];

    let k = 4;
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);

    let test_seqs: Vec<&[u8]> = vec![
        b"AAAAAACCCCCCGGGGGG",
        b"ACACACACACACACAC",
        b"ATGATGATGATGATGATG",
    ];

    let params = taxonomy::ClassifyParams::default();
    for (i, seq) in test_seqs.iter().enumerate() {
        let result = classifier.classify(seq, &params);
        v.check_pass(
            &format!("taxon {i}: f64 confidence > 0"),
            result.confidence.iter().any(|&c| c > 0.0),
        );

        let q_idx = classifier.classify_quantized(seq);
        v.check(
            &format!("taxon {i}: int8 matches f64"),
            q_idx as f64,
            result.taxon_idx as f64,
            0.0,
        );
    }

    let weights = classifier.to_int8_weights();
    v.check(
        "priors count = n_taxa",
        weights.priors_i8.len() as f64,
        3.0,
        0.0,
    );

    print_timing("taxonomy multi-taxon", t0);
}

fn print_timing(name: &str, t0: Instant) {
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [{name}] {ms:.1} ms");
}
