// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp059: 23-Domain Head-to-Head Timing Benchmark
//!
//! Runs all 23 BarraCUDA CPU domains with wall-clock timing,
//! matching the Python `benchmark_rust_vs_python.py` workloads
//! for direct Rust vs Python comparison.
//!
//! Run:
//! ```text
//! cargo run --release --bin benchmark_23_domain_timing
//! python3 scripts/benchmark_rust_vs_python.py
//! ```

use std::time::Instant;
use wetspring_barracuda::bio::{
    alignment, ani, bootstrap, cooperation, decision_tree::DecisionTree, diversity, dnds,
    felsenstein, gillespie, hmm, kmer, molecular_clock, multi_signal, pangenome, phage_defense,
    placement, qs_biofilm, robinson_foulds, signal, snp, spectral_match, unifrac::PhyloTree,
};

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp059: 23-Domain Rust Timing Benchmark (BarraCUDA CPU)");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let mut timings: Vec<(&str, f64)> = Vec::new();

    // D01: ODE Integration (RK4)
    let t0 = Instant::now();
    let _traj =
        qs_biofilm::scenario_standard_growth(&qs_biofilm::QsBiofilmParams::default(), 0.001);
    timings.push((
        "D01: ODE Integration (RK4)",
        t0.elapsed().as_micros() as f64,
    ));

    // D02: Gillespie SSA (100 reps)
    let t0 = Instant::now();
    let reactions = vec![
        gillespie::Reaction {
            propensity: Box::new(|state: &[i64]| {
                #[allow(clippy::cast_precision_loss)]
                {
                    0.5 * state[0] as f64
                }
            }),
            stoichiometry: vec![1],
        },
        gillespie::Reaction {
            propensity: Box::new(|state: &[i64]| {
                #[allow(clippy::cast_precision_loss)]
                {
                    0.1 * state[0] as f64
                }
            }),
            stoichiometry: vec![-1],
        },
    ];
    for seed in 0..100_u64 {
        let mut rng = gillespie::Lcg64::new(seed);
        let _r = gillespie::gillespie_ssa(&[100], &reactions, 10.0, &mut rng);
    }
    timings.push((
        "D02: Gillespie SSA (100 reps)",
        t0.elapsed().as_micros() as f64,
    ));

    // D03: HMM (Forward + Viterbi)
    let t0 = Instant::now();
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let _fwd = hmm::forward(&model, &[0, 1, 0, 1]);
    let _vit = hmm::viterbi(&model, &[0, 1, 0, 1]);
    timings.push((
        "D03: HMM (Forward + Viterbi)",
        t0.elapsed().as_micros() as f64,
    ));

    // D04: Smith-Waterman (40bp)
    let t0 = Instant::now();
    let _sw = alignment::smith_waterman(
        b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        &alignment::ScoringParams::default(),
    );
    timings.push((
        "D04: Smith-Waterman (40bp)",
        t0.elapsed().as_micros() as f64,
    ));

    // D05: Felsenstein (20bp, 3 taxa)
    let t0 = Instant::now();
    let tree = build_simple_felsenstein_tree();
    let _ll = felsenstein::log_likelihood(&tree, 1.0);
    timings.push((
        "D05: Felsenstein (20bp, 3 taxa)",
        t0.elapsed().as_micros() as f64,
    ));

    // D06: Diversity (Shannon + Simpson)
    let t0 = Instant::now();
    let counts = [10.0, 20.0, 30.0, 15.0, 25.0];
    let _s = diversity::shannon(&counts);
    let _si = diversity::simpson(&counts);
    let _e = diversity::pielou_evenness(&counts);
    timings.push((
        "D06: Diversity (Shannon+Simpson)",
        t0.elapsed().as_micros() as f64,
    ));

    // D07: Signal Processing
    let t0 = Instant::now();
    let test_signal: Vec<f64> = (0..256)
        .map(|i| {
            let x = i as f64 / 256.0;
            (x * 2.0 * std::f64::consts::PI).sin() + 0.5 * (x * 4.0 * std::f64::consts::PI).sin()
        })
        .collect();
    let _peaks = signal::find_peaks(&test_signal, &signal::PeakParams::default());
    timings.push(("D07: Signal Processing", t0.elapsed().as_micros() as f64));

    // D08: Cooperation ODE (100h)
    let t0 = Instant::now();
    let _coop =
        cooperation::scenario_equal_start(&cooperation::CooperationParams::default(), 0.001);
    timings.push((
        "D08: Cooperation ODE (100h)",
        t0.elapsed().as_micros() as f64,
    ));

    // D09: Robinson-Foulds (4 taxa)
    let t0 = Instant::now();
    let tree1 = PhyloTree::from_newick("((A,B),(C,D));");
    let tree2 = PhyloTree::from_newick("((A,C),(B,D));");
    let _rf = robinson_foulds::rf_distance(&tree1, &tree2);
    timings.push((
        "D09: Robinson-Foulds (4 taxa)",
        t0.elapsed().as_micros() as f64,
    ));

    // D10: Multi-Signal QS (48h)
    let t0 = Instant::now();
    let _ms = multi_signal::scenario_wild_type(&multi_signal::MultiSignalParams::default(), 0.001);
    timings.push((
        "D10: Multi-Signal QS (48h)",
        t0.elapsed().as_micros() as f64,
    ));

    // D11: Phage Defense (48h)
    let t0 = Instant::now();
    let _pd =
        phage_defense::scenario_phage_attack(&phage_defense::PhageDefenseParams::default(), 0.001);
    timings.push(("D11: Phage Defense (48h)", t0.elapsed().as_micros() as f64));

    // D12: Bootstrap (100 reps)
    let t0 = Instant::now();
    let tree_bs = build_simple_felsenstein_tree();
    let alignment_bs = bootstrap::Alignment::from_rows(&[
        felsenstein::encode_dna("ACGTACGT"),
        felsenstein::encode_dna("ACGTACGT"),
        felsenstein::encode_dna("ACGTACGT"),
    ]);
    let _boots = bootstrap::bootstrap_likelihoods(&tree_bs, &alignment_bs, 100, 1.0, 42);
    timings.push(("D12: Bootstrap (100 reps)", t0.elapsed().as_micros() as f64));

    // D13: Placement (3 taxa, 12bp)
    let t0 = Instant::now();
    let ref_tree = build_simple_felsenstein_tree();
    let _pl = placement::placement_scan(&ref_tree, "ACGTACGTACGT", 0.05, 1.0);
    timings.push((
        "D13: Placement (3 taxa, 12bp)",
        t0.elapsed().as_micros() as f64,
    ));

    // D14: Decision Tree (4 samples)
    let t0 = Instant::now();
    let dt = DecisionTree::from_arrays(
        &[0, 1, -1, -1],
        &[5.0, 3.0, 0.0, 0.0],
        &[1, 2, -1, -1],
        &[3, -1, -1, -1],
        &[None, None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let _p1 = dt.predict(&[3.0, 0.0]);
    let _p2 = dt.predict(&[7.0, 0.0]);
    let _pb = dt.predict_batch(&[
        vec![3.0, 0.0],
        vec![7.0, 0.0],
        vec![4.0, 2.0],
        vec![6.0, 4.0],
    ]);
    timings.push((
        "D14: Decision Tree (4 samples)",
        t0.elapsed().as_micros() as f64,
    ));

    // D15: Spectral Match (5 peaks)
    let t0 = Instant::now();
    let _cs = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0, 400.0, 500.0],
        &[1000.0, 500.0, 800.0, 300.0, 100.0],
        &[100.0, 200.5, 300.0, 400.0, 500.0],
        &[950.0, 520.0, 780.0, 310.0, 90.0],
        0.5,
    );
    timings.push((
        "D15: Spectral Match (5 peaks)",
        t0.elapsed().as_micros() as f64,
    ));

    // D16: Extended Diversity (Pielou, Bray-Curtis, Chao1)
    let t0 = Instant::now();
    let _p = diversity::pielou_evenness(&[25.0, 25.0, 25.0, 25.0]);
    let _bc = diversity::bray_curtis(&[10.0, 20.0, 30.0, 40.0], &[10.0, 20.0, 30.0, 40.0]);
    let _ch = diversity::chao1(&[10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    timings.push(("D16: Extended Diversity", t0.elapsed().as_micros() as f64));

    // D17: K-mer Counting (16bp, k=4)
    let t0 = Instant::now();
    let _km = kmer::count_kmers(b"ACGTACGTACGTACGT", 4);
    timings.push((
        "D17: K-mer Counting (16bp, k=4)",
        t0.elapsed().as_micros() as f64,
    ));

    // D18: Integrated Pipeline (NJ + diversity + spectral)
    let t0 = Instant::now();
    let _h = diversity::shannon(&[10.0, 20.0, 30.0, 15.0, 25.0]);
    let _bc = diversity::bray_curtis(&[10.0, 20.0, 30.0], &[15.0, 25.0, 35.0]);
    let _cs2 = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0],
        &[1000.0, 500.0, 800.0],
        &[100.0, 200.0, 300.0],
        &[950.0, 520.0, 780.0],
        0.5,
    );
    timings.push(("D18: Integrated Pipeline", t0.elapsed().as_micros() as f64));

    // D19: ANI (3 seqs, 50bp)
    let t0 = Instant::now();
    let seq_a = b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG";
    let seq_b = b"ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG";
    let seq_c = b"CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG";
    let _ani1 = ani::pairwise_ani(seq_a, seq_b);
    let _ani2 = ani::pairwise_ani(seq_a, seq_c);
    let _ani3 = ani::pairwise_ani(seq_b, seq_c);
    timings.push(("D19: ANI (3 seqs, 50bp)", t0.elapsed().as_micros() as f64));

    // D20: SNP Calling (4 seqs, 50bp)
    let t0 = Instant::now();
    let snp_seqs: Vec<&[u8]> = vec![
        b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
        b"ATGATGATGATGATGATCATGATGATGATGATGATGATGATGATGATGATG",
        b"CTGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGCTG",
        b"ATGATCATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG",
    ];
    let _snps = snp::call_snps(&snp_seqs);
    timings.push((
        "D20: SNP Calling (4 seqs, 50bp)",
        t0.elapsed().as_micros() as f64,
    ));

    // D21: dN/dS (10 codons)
    let t0 = Instant::now();
    let _dd = dnds::pairwise_dnds(
        b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT",
        b"ATGGCCGAATTTGCTGCTGCTGCTGCCGCT",
    );
    timings.push(("D21: dN/dS (10 codons)", t0.elapsed().as_micros() as f64));

    // D22: Molecular Clock (7 nodes)
    let t0 = Instant::now();
    let bl = [0.0, 0.1, 0.2, 0.05, 0.05, 0.15, 0.15];
    let parents: Vec<Option<usize>> =
        vec![None, Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)];
    let sc = molecular_clock::strict_clock(&bl, &parents, 3500.0, &[]).unwrap();
    let rates = molecular_clock::relaxed_clock_rates(&bl, &sc.node_ages, &parents);
    let _cv = molecular_clock::rate_variation_cv(&rates);
    timings.push((
        "D22: Molecular Clock (7 nodes)",
        t0.elapsed().as_micros() as f64,
    ));

    // D23: Pangenome (7 genes, 5 genomes)
    let t0 = Instant::now();
    let clusters = vec![
        pangenome::GeneCluster {
            id: "g1".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g2".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g3".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g4".into(),
            presence: vec![true, true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "g5".into(),
            presence: vec![false, true, true, false, false],
        },
        pangenome::GeneCluster {
            id: "g6".into(),
            presence: vec![true, false, false, false, false],
        },
        pangenome::GeneCluster {
            id: "g7".into(),
            presence: vec![false, false, false, false, true],
        },
    ];
    let _pc = pangenome::analyze(&clusters, 5);
    let pvals = vec![
        pangenome::hypergeometric_pvalue(8, 10, 20, 100),
        pangenome::hypergeometric_pvalue(2, 10, 20, 100),
        pangenome::hypergeometric_pvalue(5, 10, 20, 100),
    ];
    let _adj = pangenome::benjamini_hochberg(&pvals);
    timings.push(("D23: Pangenome (7 genes)", t0.elapsed().as_micros() as f64));

    // Summary
    println!("  {:<40} {:>12}", "Domain", "Time (µs)");
    println!("  {}", "─".repeat(55));
    let mut total = 0.0;
    for (name, us) in &timings {
        println!("  {name:<40} {us:>12.0}");
        total += us;
    }
    println!("  {}", "─".repeat(55));
    println!("  {:<40} {total:>12.0}", "TOTAL");
    println!();
    println!("  23/23 domains timed. All pure Rust, zero unsafe, zero dependencies.");
    println!();
}

fn build_simple_felsenstein_tree() -> felsenstein::TreeNode {
    felsenstein::TreeNode::Internal {
        left: Box::new(felsenstein::TreeNode::Leaf {
            name: "A".into(),
            states: felsenstein::encode_dna("ACGTACGTACGTACGTACGT"),
        }),
        right: Box::new(felsenstein::TreeNode::Internal {
            left: Box::new(felsenstein::TreeNode::Leaf {
                name: "B".into(),
                states: felsenstein::encode_dna("ACGTACGTACGTACGTACGT"),
            }),
            right: Box::new(felsenstein::TreeNode::Leaf {
                name: "C".into(),
                states: felsenstein::encode_dna("ACGTACGTACGTACGTACGT"),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        left_branch: 0.1,
        right_branch: 0.2,
    }
}
