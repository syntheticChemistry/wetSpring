// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
//! Exp086: `metalForge` End-to-End Pipeline Proof
//!
//! Chains 5 bioinformatics stages through simulated `metalForge` dispatch,
//! proving the full evolution path works as a pipeline:
//!
//! 1. **K-mer counting** (CPU → flat histogram for GPU)
//! 2. **Taxonomy classification** (CPU f64 ↔ NPU int8 parity)
//! 3. **`UniFrac` distance** (CPU ↔ flat CSR round-trip)
//! 4. **Dispatch routing** (5 workload classes → correct substrates)
//! 5. **Pipeline output is substrate-independent** (CPU-only = mixed path)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCUDA` CPU + `metalForge` dispatch |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_metalforge_pipeline` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use wetspring_barracuda::bio::{kmer, taxonomy, unifrac};
use wetspring_barracuda::validation::Validator;

// ═══════════════════════════════════════════════════════════
//  Local substrate router (mirrors metalForge/forge dispatch)
// ═══════════════════════════════════════════════════════════

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadClass {
    BatchParallel,
    QuantizedInference,
    TreeTraversal,
    SequentialIo,
    MatrixReduce,
}

struct SubstrateRouter {
    gpu_available: bool,
    npu_available: bool,
}

impl SubstrateRouter {
    const fn new(gpu: bool, npu: bool) -> Self {
        Self {
            gpu_available: gpu,
            npu_available: npu,
        }
    }

    #[allow(clippy::match_same_arms)]
    const fn route(&self, class: WorkloadClass) -> Substrate {
        match class {
            WorkloadClass::BatchParallel | WorkloadClass::MatrixReduce => {
                if self.gpu_available {
                    Substrate::Gpu
                } else {
                    Substrate::Cpu
                }
            }
            WorkloadClass::QuantizedInference => {
                if self.npu_available {
                    Substrate::Npu
                } else {
                    Substrate::Cpu
                }
            }
            WorkloadClass::TreeTraversal => {
                if self.gpu_available {
                    Substrate::Gpu
                } else {
                    Substrate::Cpu
                }
            }
            WorkloadClass::SequentialIo => Substrate::Cpu,
        }
    }
}

fn main() {
    let mut v = Validator::new("Exp086: metalForge End-to-End Pipeline Proof");
    let t0 = Instant::now();

    validate_dispatch_routing(&mut v);
    validate_pipeline_cpu_only(&mut v);
    validate_pipeline_parity(&mut v);
    validate_flat_layout_chain(&mut v);
    validate_dispatch_fallback(&mut v);

    print_timing("Total", t0);
    v.finish();
}

fn validate_dispatch_routing(v: &mut Validator) {
    v.section("═══ Stage 1: Pipeline Dispatch Routing ═══");
    let t0 = Instant::now();

    let full = SubstrateRouter::new(true, true);

    v.check_pass(
        "kmer → GPU (batch parallel)",
        full.route(WorkloadClass::BatchParallel) == Substrate::Gpu,
    );
    v.check_pass(
        "taxonomy → NPU (int8 inference)",
        full.route(WorkloadClass::QuantizedInference) == Substrate::Npu,
    );
    v.check_pass(
        "unifrac → GPU (tree traversal)",
        full.route(WorkloadClass::TreeTraversal) == Substrate::Gpu,
    );
    v.check_pass(
        "FASTQ parse → CPU (sequential I/O)",
        full.route(WorkloadClass::SequentialIo) == Substrate::Cpu,
    );
    v.check_pass(
        "diversity → GPU (matrix reduce)",
        full.route(WorkloadClass::MatrixReduce) == Substrate::Gpu,
    );

    let cpu_only = SubstrateRouter::new(false, false);
    v.check_pass(
        "CPU-only: kmer → CPU",
        cpu_only.route(WorkloadClass::BatchParallel) == Substrate::Cpu,
    );
    v.check_pass(
        "CPU-only: taxonomy → CPU",
        cpu_only.route(WorkloadClass::QuantizedInference) == Substrate::Cpu,
    );
    v.check_pass(
        "CPU-only: unifrac → CPU",
        cpu_only.route(WorkloadClass::TreeTraversal) == Substrate::Cpu,
    );

    let gpu_no_npu = SubstrateRouter::new(true, false);
    v.check_pass(
        "GPU-no-NPU: taxonomy → CPU (no NPU fallback)",
        gpu_no_npu.route(WorkloadClass::QuantizedInference) == Substrate::Cpu,
    );

    print_timing("dispatch routing", t0);
}

fn validate_pipeline_cpu_only(v: &mut Validator) {
    v.section("═══ Stage 2: CPU-Only Pipeline (reference truth) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");

    let k = 4;
    let kmer_counts: Vec<kmer::KmerCounts> =
        sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();

    v.check_pass("kmer: 4 sample counts", kmer_counts.len() == 4);

    let total_kmers: u64 = kmer_counts.iter().map(|c| c.total_valid_kmers).sum();
    v.check_pass("kmer: total > 0", total_kmers > 0);

    let histograms: Vec<Vec<u32>> = kmer_counts
        .iter()
        .map(kmer::KmerCounts::to_histogram)
        .collect();
    v.check_pass(
        "kmer→histogram: all 256-wide",
        histograms.iter().all(|h| h.len() == 256),
    );

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();
    let classifications: Vec<taxonomy::Classification> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params))
        .collect();
    v.check_pass(
        "taxonomy: 4 classifications",
        classifications.len() == 4 && classifications.iter().all(|c| !c.confidence.is_empty()),
    );

    let mut sample_a: HashMap<String, f64> = HashMap::new();
    let mut sample_b: HashMap<String, f64> = HashMap::new();
    sample_a.insert(
        "A".into(),
        f64::from(kmer_counts[0].total_valid_kmers as u32),
    );
    sample_a.insert(
        "B".into(),
        f64::from(kmer_counts[1].total_valid_kmers as u32),
    );
    sample_b.insert(
        "C".into(),
        f64::from(kmer_counts[2].total_valid_kmers as u32),
    );
    sample_b.insert(
        "D".into(),
        f64::from(kmer_counts[3].total_valid_kmers as u32),
    );

    let uw = unifrac::unweighted_unifrac(&tree, &sample_a, &sample_b);
    let ww = unifrac::weighted_unifrac(&tree, &sample_a, &sample_b);
    v.check_pass("unifrac: unweighted is finite", uw.is_finite());
    v.check_pass("unifrac: weighted is finite", ww.is_finite());
    v.check_pass("unifrac: unweighted ∈ [0,1]", (0.0..=1.0).contains(&uw));
    v.check_pass("unifrac: weighted ≥ 0", ww >= 0.0);

    print_timing("CPU-only pipeline", t0);
}

fn validate_pipeline_parity(v: &mut Validator) {
    v.section("═══ Stage 3: Pipeline Parity (f64 CPU ↔ int8 NPU path) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let cpu_counts: Vec<kmer::KmerCounts> =
        sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);

    let params = taxonomy::ClassifyParams::default();
    let cpu_classifications: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();
    let int8_classifications: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify_quantized(s))
        .collect();

    for i in 0..4 {
        v.check(
            &format!("sample {i}: f64 ↔ int8 classification parity"),
            int8_classifications[i] as f64,
            cpu_classifications[i] as f64,
            0.0,
        );
    }

    let cpu_histograms: Vec<Vec<u32>> = cpu_counts
        .iter()
        .map(kmer::KmerCounts::to_histogram)
        .collect();
    for (i, hist) in cpu_histograms.iter().enumerate() {
        let restored = kmer::KmerCounts::from_histogram(hist, k);
        v.check(
            &format!("sample {i}: kmer histogram round-trip"),
            restored.total_valid_kmers as f64,
            cpu_counts[i].total_valid_kmers as f64,
            0.0,
        );
    }

    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let flat = tree.to_flat_tree();
    let reconstructed = flat.to_phylo_tree();

    let mut sa: HashMap<String, f64> = HashMap::new();
    let mut sb: HashMap<String, f64> = HashMap::new();
    sa.insert("A".into(), 10.0);
    sa.insert("B".into(), 5.0);
    sb.insert("C".into(), 8.0);
    sb.insert("D".into(), 3.0);

    let uw_orig = unifrac::unweighted_unifrac(&tree, &sa, &sb);
    let uw_flat = unifrac::unweighted_unifrac(&reconstructed, &sa, &sb);
    v.check(
        "UniFrac parity: original ↔ CSR round-trip",
        uw_flat,
        uw_orig,
        1e-12,
    );

    let ww_orig = unifrac::weighted_unifrac(&tree, &sa, &sb);
    let ww_flat = unifrac::weighted_unifrac(&reconstructed, &sa, &sb);
    v.check(
        "weighted UniFrac parity: original ↔ CSR round-trip",
        ww_flat,
        ww_orig,
        1e-12,
    );

    print_timing("pipeline parity", t0);
}

fn validate_flat_layout_chain(v: &mut Validator) {
    v.section("═══ Stage 4: Flat Layout Chain (GPU buffer readiness) ═══");
    let t0 = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;

    let counts: Vec<kmer::KmerCounts> = sequences.iter().map(|s| kmer::count_kmers(s, k)).collect();
    let histograms: Vec<Vec<u32>> = counts.iter().map(kmer::KmerCounts::to_histogram).collect();
    let all_flat_count: usize = histograms.iter().map(Vec::len).sum();
    v.check(
        "kmer: flattened buffer = 4 × 256",
        all_flat_count as f64,
        1024.0,
        0.0,
    );

    let tree = unifrac::PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let flat_tree = tree.to_flat_tree();
    v.check_pass(
        "unifrac: flat tree CSR non-empty",
        !flat_tree.children_flat.is_empty(),
    );
    v.check_pass(
        "unifrac: branch lengths present",
        flat_tree.branch_length.iter().any(|&b| b > 0.0),
    );

    let mut abundance_table: unifrac::AbundanceTable = HashMap::new();
    for (i, label) in ["A", "B", "C", "D"].iter().enumerate() {
        let mut sample = HashMap::new();
        sample.insert((*label).to_string(), counts[i].total_valid_kmers as f64);
        abundance_table.insert(format!("sample_{label}"), sample);
    }

    let (sample_matrix, n_samples, n_leaves) =
        unifrac::to_sample_matrix(&flat_tree, &abundance_table);
    v.check("unifrac: n_samples = 4", n_samples as f64, 4.0, 0.0);
    v.check(
        "unifrac: matrix = n_samples × n_leaves",
        sample_matrix.len() as f64,
        (n_samples * n_leaves) as f64,
        0.0,
    );

    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let weights = classifier.to_int8_weights();
    v.check(
        "taxonomy: int8 weights = n_taxa × 4^k",
        weights.weights_i8.len() as f64,
        (weights.n_taxa * weights.kmer_space) as f64,
        0.0,
    );
    v.check_pass("taxonomy: scale > 0", weights.scale > 0.0);
    v.check(
        "taxonomy: priors = n_taxa",
        weights.priors_i8.len() as f64,
        weights.n_taxa as f64,
        0.0,
    );

    let sorted_pairs: Vec<Vec<(u64, u32)>> = counts
        .iter()
        .map(kmer::KmerCounts::to_sorted_pairs)
        .collect();
    let total_pairs: usize = sorted_pairs.iter().map(Vec::len).sum();
    v.check_pass("kmer: sorted pairs total > 0", total_pairs > 0);
    v.check_pass(
        "kmer: all sorted-pair vecs are ordered",
        sorted_pairs
            .iter()
            .all(|p| p.windows(2).all(|w| w[0].0 <= w[1].0)),
    );

    print_timing("flat layout chain", t0);
}

fn validate_dispatch_fallback(v: &mut Validator) {
    v.section("═══ Stage 5: Dispatch Fallback Correctness ═══");
    let t0 = Instant::now();

    let configs: Vec<(&str, SubstrateRouter)> = vec![
        ("Full (GPU+NPU+CPU)", SubstrateRouter::new(true, true)),
        ("GPU+CPU (no NPU)", SubstrateRouter::new(true, false)),
        ("NPU+CPU (no GPU)", SubstrateRouter::new(false, true)),
        ("CPU-only", SubstrateRouter::new(false, false)),
    ];

    let sequences = synthetic_sequences();
    let k = 4;
    let refs = training_references();
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, k);
    let params = taxonomy::ClassifyParams::default();

    let reference_results: Vec<usize> = sequences
        .iter()
        .map(|s| classifier.classify(s, &params).taxon_idx)
        .collect();

    for (label, router) in &configs {
        let tax_target = router.route(WorkloadClass::QuantizedInference);
        let results: Vec<usize> = if tax_target == Substrate::Npu {
            sequences
                .iter()
                .map(|s| classifier.classify_quantized(s))
                .collect()
        } else {
            sequences
                .iter()
                .map(|s| classifier.classify(s, &params).taxon_idx)
                .collect()
        };

        let all_match = results == reference_results;
        v.check_pass(
            &format!("{label}: taxonomy parity across dispatch paths"),
            all_match,
        );
    }

    let reference_counts: Vec<u64> = sequences
        .iter()
        .map(|s| kmer::count_kmers(s, k).total_valid_kmers)
        .collect();

    for (label, _) in &configs {
        let counts: Vec<u64> = sequences
            .iter()
            .map(|s| kmer::count_kmers(s, k).total_valid_kmers)
            .collect();
        v.check_pass(
            &format!("{label}: kmer counts deterministic"),
            counts == reference_counts,
        );
    }

    print_timing("dispatch fallback", t0);
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
