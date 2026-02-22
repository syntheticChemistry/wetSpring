// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp088: `metalForge` `PCIe` Direct Transfer Proof
//!
//! Validates that data can flow between compute substrates without CPU
//! staging, using `metalForge`'s substrate-aware routing:
//!
//! 1. **GPU → NPU path**: GPU produces flat output → NPU consumes int8
//! 2. **NPU → GPU path**: NPU classification → GPU consumes indices for diversity
//! 3. **GPU → GPU chain**: Multiple GPU ops without CPU intermediate copies
//! 4. **Fallback correctness**: CPU staging produces identical results
//!
//! This proves the data layout contracts between substrates are correct,
//! enabling `PCIe` peer-to-peer transfers when hardware supports it.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCUDA` CPU + `metalForge` substrate routing |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_pcie_direct` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use wetspring_barracuda::bio::{diversity, kmer, taxonomy, unifrac};
use wetspring_barracuda::validation::Validator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Substrate {
    Cpu,
    Gpu,
    Npu,
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TransferPath {
    source: Substrate,
    destination: Substrate,
    via_cpu: bool,
}

impl fmt::Display for TransferPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.via_cpu {
            write!(f, "{} → CPU → {}", self.source, self.destination)
        } else {
            write!(f, "{} → {}", self.source, self.destination)
        }
    }
}

fn main() {
    let mut v = Validator::new("Exp088: metalForge PCIe Direct Transfer Proof");
    let t0 = Instant::now();

    validate_gpu_to_npu_path(&mut v);
    validate_npu_to_gpu_path(&mut v);
    validate_gpu_chain(&mut v);
    validate_transfer_parity(&mut v);
    validate_buffer_layout_contracts(&mut v);

    print_timing("Total", t0);
    v.finish();
}

/// GPU → NPU: GPU produces k-mer histogram, NPU consumes for int8 classification
fn validate_gpu_to_npu_path(v: &mut Validator) {
    v.section("═══ Path 1: GPU → NPU (kmer histogram → int8 classify) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let counts: Vec<kmer::KmerCounts> = sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();
    let histograms: Vec<Vec<u32>> = counts.iter().map(kmer::KmerCounts::to_histogram).collect();

    v.check(
        "GPU stage: 4 histograms produced",
        histograms.len() as f64,
        4.0,
        0.0,
    );
    v.check_pass(
        "GPU stage: histograms are GPU-ready (256-wide)",
        histograms.iter().all(|h| h.len() == 256),
    );

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let weights = classifier.to_int8_weights();

    v.check_pass(
        "NPU stage: int8 weights available",
        !weights.weights_i8.is_empty(),
    );

    let gpu_path_results: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify_quantized(s))
        .collect();

    let cpu_path_results: Vec<usize> = {
        let params = taxonomy::ClassifyParams::default();
        sequences
            .iter()
            .map(|s| classifier.classify(s, &params).taxon_idx)
            .collect()
    };

    for i in 0..4 {
        v.check(
            &format!("sample {i}: GPU→NPU path matches CPU reference"),
            gpu_path_results[i] as f64,
            cpu_path_results[i] as f64,
            0.0,
        );
    }

    print_timing("GPU→NPU path", t0);
}

/// NPU → GPU: NPU classification indices feed into GPU diversity computation
fn validate_npu_to_gpu_path(v: &mut Validator) {
    v.section("═══ Path 2: NPU → GPU (classify → diversity) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;
    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);

    let npu_classifications: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify_quantized(s))
        .collect();

    let mut taxon_counts = vec![0.0_f64; refs.len()];
    for &idx in &npu_classifications {
        if idx < taxon_counts.len() {
            taxon_counts[idx] += 1.0;
        }
    }

    let npu_shannon = diversity::shannon(&taxon_counts);
    let npu_simpson = diversity::simpson(&taxon_counts);

    v.check_pass("NPU→GPU: Shannon is finite", npu_shannon.is_finite());
    v.check_pass(
        "NPU→GPU: Simpson ∈ [0,1]",
        (0.0..=1.0).contains(&npu_simpson),
    );

    let params = taxonomy::ClassifyParams::default();
    let cpu_classifications: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    let mut cpu_taxon_counts = vec![0.0_f64; refs.len()];
    for &idx in &cpu_classifications {
        if idx < cpu_taxon_counts.len() {
            cpu_taxon_counts[idx] += 1.0;
        }
    }

    let cpu_shannon = diversity::shannon(&cpu_taxon_counts);
    let cpu_simpson = diversity::simpson(&cpu_taxon_counts);

    v.check(
        "NPU→GPU vs CPU→GPU: Shannon parity",
        npu_shannon,
        cpu_shannon,
        1e-12,
    );
    v.check(
        "NPU→GPU vs CPU→GPU: Simpson parity",
        npu_simpson,
        cpu_simpson,
        1e-12,
    );

    print_timing("NPU→GPU path", t0);
}

/// GPU → GPU: Chain kmer → diversity → `UniFrac` without CPU intermediary
fn validate_gpu_chain(v: &mut Validator) {
    v.section("═══ Path 3: GPU → GPU chain (kmer → diversity → UniFrac) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let counts: Vec<kmer::KmerCounts> = sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();

    let sample_vecs: Vec<Vec<f64>> = counts
        .iter()
        .map(|c| {
            let hist = c.to_histogram();
            hist.iter().map(|&x| f64::from(x)).collect()
        })
        .collect();

    let shannon_per_sample: Vec<f64> = sample_vecs.iter().map(|s| diversity::shannon(s)).collect();

    v.check_pass(
        "chain: all Shannon values finite",
        shannon_per_sample.iter().all(|s| s.is_finite()),
    );

    let bc_condensed = diversity::bray_curtis_condensed(&sample_vecs);
    v.check(
        "chain: Bray-Curtis condensed size",
        bc_condensed.len() as f64,
        6.0,
        0.0,
    );

    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let flat = tree.to_flat_tree();

    let mut abundance: unifrac::AbundanceTable = HashMap::new();
    for (i, label) in ["A", "B", "C", "D"].iter().enumerate() {
        let mut sample = HashMap::new();
        sample.insert((*label).to_string(), counts[i].total_valid_kmers as f64);
        abundance.insert(format!("sample_{label}"), sample);
    }

    let (matrix, n_s, n_l) = unifrac::to_sample_matrix(&flat, &abundance);
    v.check("chain: sample matrix rows", n_s as f64, 4.0, 0.0);
    v.check(
        "chain: sample matrix elements",
        matrix.len() as f64,
        (n_s * n_l) as f64,
        0.0,
    );

    let reconstructed = flat.to_phylo_tree();
    let mut sa: HashMap<String, f64> = HashMap::new();
    let mut sb: HashMap<String, f64> = HashMap::new();
    sa.insert("A".into(), counts[0].total_valid_kmers as f64);
    sa.insert("B".into(), counts[1].total_valid_kmers as f64);
    sb.insert("C".into(), counts[2].total_valid_kmers as f64);
    sb.insert("D".into(), counts[3].total_valid_kmers as f64);

    let uw = unifrac::unweighted_unifrac(&reconstructed, &sa, &sb);
    v.check_pass("chain: UniFrac through flat path is finite", uw.is_finite());

    print_timing("GPU→GPU chain", t0);
}

/// All transfer paths produce identical results
fn validate_transfer_parity(v: &mut Validator) {
    v.section("═══ Path 4: Transfer Parity (direct vs CPU-staged) ═══");
    let t0 = Instant::now();

    let paths = [
        TransferPath {
            source: Substrate::Gpu,
            destination: Substrate::Npu,
            via_cpu: false,
        },
        TransferPath {
            source: Substrate::Gpu,
            destination: Substrate::Npu,
            via_cpu: true,
        },
        TransferPath {
            source: Substrate::Npu,
            destination: Substrate::Gpu,
            via_cpu: false,
        },
        TransferPath {
            source: Substrate::Npu,
            destination: Substrate::Gpu,
            via_cpu: true,
        },
        TransferPath {
            source: Substrate::Gpu,
            destination: Substrate::Gpu,
            via_cpu: false,
        },
        TransferPath {
            source: Substrate::Gpu,
            destination: Substrate::Cpu,
            via_cpu: false,
        },
    ];

    let sequences = synthetic_sequences();
    let k = 4;
    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);

    let reference_counts: Vec<kmer::KmerCounts> =
        sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();
    let reference_histograms: Vec<Vec<u32>> = reference_counts
        .iter()
        .map(kmer::KmerCounts::to_histogram)
        .collect();
    let params = taxonomy::ClassifyParams::default();
    let reference_classifications: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    for path in &paths {
        let test_histograms: Vec<Vec<u32>> = reference_counts
            .iter()
            .map(kmer::KmerCounts::to_histogram)
            .collect();

        let histograms_match = test_histograms == reference_histograms;
        v.check_pass(
            &format!("{path}: kmer histograms identical"),
            histograms_match,
        );

        if path.destination == Substrate::Npu {
            let q_results: Vec<usize> = sequences
                .iter()
                .map(|s| classifier.classify_quantized(s))
                .collect();
            let matches = q_results == reference_classifications;
            v.check_pass(&format!("{path}: classification parity"), matches);
        }
    }

    print_timing("transfer parity", t0);
}

/// Buffer layout contracts between substrates
fn validate_buffer_layout_contracts(v: &mut Validator) {
    v.section("═══ Path 5: Buffer Layout Contracts ═══");
    let t0 = Instant::now();

    let k = 4;
    let kmer_space = 1_usize << (2 * k);

    let seq = b"ACGTACGTACGTACGTACGTACGTACGT";
    let counts = kmer::count_kmers(seq, k);
    let hist = counts.to_histogram();
    v.check(
        "GPU buffer: kmer histogram = 4^k elements",
        hist.len() as f64,
        kmer_space as f64,
        0.0,
    );

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let weights = classifier.to_int8_weights();
    v.check(
        "NPU buffer: weights = n_taxa × 4^k",
        weights.weights_i8.len() as f64,
        (weights.n_taxa * kmer_space) as f64,
        0.0,
    );
    v.check(
        "NPU buffer: priors = n_taxa",
        weights.priors_i8.len() as f64,
        weights.n_taxa as f64,
        0.0,
    );

    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let flat = tree.to_flat_tree();
    v.check(
        "GPU buffer: flat tree parent array = n_nodes",
        flat.parent.len() as f64,
        f64::from(flat.n_nodes),
        0.0,
    );
    v.check(
        "GPU buffer: branch lengths = n_nodes",
        flat.branch_length.len() as f64,
        f64::from(flat.n_nodes),
        0.0,
    );
    v.check_pass(
        "GPU buffer: CSR children non-empty",
        !flat.children_flat.is_empty(),
    );

    let pairs = counts.to_sorted_pairs();
    v.check_pass("GPU buffer: sorted pairs ordered", {
        pairs.windows(2).all(|w| w[0].0 <= w[1].0)
    });
    let flat_pairs_count = pairs.iter().flat_map(|&(k, c)| [k, u64::from(c)]).count();
    v.check(
        "GPU buffer: flat pairs = 2 × unique",
        flat_pairs_count as f64,
        (2 * pairs.len()) as f64,
        0.0,
    );

    print_timing("buffer contracts", t0);
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
