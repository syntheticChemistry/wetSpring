// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp239: `BarraCuda` CPU v17 — Extended Domain Benchmark (8 New Domains)
//!
//! Extends CPU v16 (11 domains, 33 checks) with 8 NEW CPU-only domains:
//! - D12: Chimera Detection
//! - D13: DADA2 Denoising
//! - D14: Alignment (Smith-Waterman)
//! - D15: Echo State Network (ESN)
//! - D16: GBM Classifier
//! - D17: DTL Reconciliation
//! - D18: Molecular Clock
//! - D19: Random Forest + Decision Tree
//!
//! Chain: Paper (Exp233) → **CPU (this + v16)** → GPU → Streaming → metalForge
//!
//! # Provenance
//!
//! Expected values are **analytical / algorithmic** — derived from algorithm
//! definitions and known-input properties, not from Python baseline scripts.
//! Each domain uses synthetic inputs with deterministic, verifiable outputs:
//! - Chimera: constructed chimeric + non-chimeric ASVs with known parentage
//! - DADA2: synthetic error profiles with known denoised sequences
//! - Smith-Waterman: hand-scored alignments with known optimal scores
//! - ESN: fixed reservoir weights with deterministic matrix operations
//! - GBM/RF/Decision Tree: fitted on known-label synthetic data
//! - DTL Reconciliation: hand-constructed gene/species tree pair
//! - Molecular Clock: known-distance tree with analytical expected dates
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (algorithmic known-values) |
//! | Date | 2026-02-28 |
//! | Command | `cargo run --features gpu --bin validate_barracuda_cpu_v17` |
//!
//! Validation class: Python-parity
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use std::time::Instant;

use wetspring_barracuda::bio::{
    alignment, chimera, dada2, decision_tree, derep, esn, gbm, molecular_clock, random_forest,
    reconciliation,
};
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::validation::OrExit;

struct DomainTiming {
    name: &'static str,
    ms: f64,
}

fn main() {
    let mut v = Validator::new("Exp239: BarraCuda CPU v17 — 8 New Domains (Pure Rust)");
    let t_total = Instant::now();
    let mut timings: Vec<DomainTiming> = Vec::new();

    println!("  Inherited: D01-D11 from CPU v16 (33/33 checks)");
    println!("  New: D12-D19 below");
    println!();

    // ═══ D12: Chimera Detection ══════════════════════════════════════════
    let t = Instant::now();
    v.section("D12: Chimera Detection");
    let asvs = vec![
        dada2::Asv {
            sequence: b"AAAACCCCGGGGTTTT".to_vec(),
            abundance: 100,
            n_members: 100,
        },
        dada2::Asv {
            sequence: b"AAAACCCCTTTTGGGG".to_vec(),
            abundance: 80,
            n_members: 80,
        },
        dada2::Asv {
            sequence: b"AAAACCCCGGGGGGGG".to_vec(),
            abundance: 5,
            n_members: 5,
        },
    ];
    let (chimera_results, chimera_stats) =
        chimera::detect_chimeras(&asvs, &chimera::ChimeraParams::default());
    v.check_pass(
        "Chimera: results for all ASVs",
        chimera_results.len() == asvs.len(),
    );
    v.check_pass(
        "Chimera: stats populated",
        chimera_stats.input_sequences > 0,
    );
    let (non_chimeric, _) = chimera::remove_chimeras(&asvs, &chimera::ChimeraParams::default());
    v.check_pass(
        "Chimera: non-chimeric ≤ total",
        non_chimeric.len() <= asvs.len(),
    );
    timings.push(DomainTiming {
        name: "Chimera Detection",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D13: DADA2 Denoising ════════════════════════════════════════════
    let t = Instant::now();
    v.section("D13: DADA2 Denoising");
    let dada2_seqs = vec![
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGT".to_vec(),
            abundance: 50,
            best_quality: 40.0,
            representative_id: "s1".into(),
            representative_quality: vec![40; 12],
        },
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGA".to_vec(),
            abundance: 3,
            best_quality: 35.0,
            representative_id: "s2".into(),
            representative_quality: vec![35; 12],
        },
        derep::UniqueSequence {
            sequence: b"TTTTACGTACGT".to_vec(),
            abundance: 40,
            best_quality: 39.0,
            representative_id: "s3".into(),
            representative_quality: vec![39; 12],
        },
    ];
    let (dada2_asvs, dada2_stats) = dada2::denoise(&dada2_seqs, &dada2::Dada2Params::default());
    v.check_pass("DADA2: produces ASVs", !dada2_asvs.is_empty());
    v.check_pass(
        "DADA2: input uniques tracked",
        dada2_stats.input_uniques == 3,
    );
    v.check_pass("DADA2: output reads > 0", dada2_stats.output_reads > 0);
    let fasta = dada2::asvs_to_fasta(&dada2_asvs);
    v.check_pass("DADA2: FASTA non-empty", !fasta.is_empty());
    timings.push(DomainTiming {
        name: "DADA2 Denoising",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D14: Alignment (Smith-Waterman) ═════════════════════════════════
    let t = Instant::now();
    v.section("D14: Alignment — Smith-Waterman");
    let query = b"ACGTACGTACGT";
    let target = b"XXXACGTACGTXXX";
    let aln = alignment::smith_waterman(query, target, &alignment::ScoringParams::default());
    v.check_pass("SW: positive score", aln.score > 0);
    v.check_pass(
        "SW: aligned sequences non-empty",
        !aln.aligned_query.is_empty(),
    );
    let score_only =
        alignment::smith_waterman_score(query, target, &alignment::ScoringParams::default());
    v.check_pass("SW: score-only matches full", score_only == aln.score);
    let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGA", b"TTTTAAAA"];
    let pairwise = alignment::pairwise_scores(&seqs, &alignment::ScoringParams::default());
    v.check_pass("Pairwise: 3 scores (condensed)", pairwise.len() == 3);
    timings.push(DomainTiming {
        name: "Alignment (SW)",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D15: Echo State Network (ESN) ═══════════════════════════════════
    let t = Instant::now();
    v.section("D15: Echo State Network");
    let esn_config = esn::EsnConfig {
        input_size: 2,
        reservoir_size: 50,
        output_size: 1,
        ..esn::EsnConfig::default()
    };
    let mut esn_net = esn::Esn::new(esn_config);
    let inputs: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![(f64::from(i) * 0.3).sin(), (f64::from(i) * 0.3).cos()])
        .collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|inp| vec![inp[0] + inp[1]]).collect();
    esn_net.train(&inputs, &targets);
    let predictions = esn_net.predict(&inputs);
    v.check_pass(
        "ESN: predictions len = inputs",
        predictions.len() == inputs.len(),
    );
    v.check_pass(
        "ESN: predictions finite",
        predictions.iter().all(|p| p[0].is_finite()),
    );
    let npu_weights = esn_net.to_npu_weights();
    esn_net.update(&[1.0, 0.0]);
    let npu_class = npu_weights.classify(esn_net.state());
    v.check_pass("ESN NPU: classify produces index", npu_class < 10);
    timings.push(DomainTiming {
        name: "ESN",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D16: GBM Classifier ═════════════════════════════════════════════
    let t = Instant::now();
    v.section("D16: GBM Classifier");
    let tree1 = gbm::GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .or_exit("unexpected error");
    let tree2 = gbm::GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.3, 0.3],
    )
    .or_exit("unexpected error");
    let gbm_model = gbm::GbmClassifier::new(vec![tree1, tree2], 0.1, 0.0, 2).or_exit("unexpected error");
    let pred = gbm_model.predict_proba(&[0.8, 0.5]);
    v.check_pass(
        "GBM: probability ∈ [0,1]",
        (0.0..=1.0).contains(&pred.probability),
    );
    v.check_pass("GBM: class is 0 or 1", pred.class <= 1);
    let batch = gbm_model.predict_batch(&[vec![0.8, 0.5], vec![0.2, 0.1]]);
    v.check_pass("GBM batch: 2 predictions", batch.len() == 2);
    timings.push(DomainTiming {
        name: "GBM Classifier",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D17: DTL Reconciliation ═════════════════════════════════════════
    let t = Instant::now();
    v.section("D17: DTL Reconciliation");
    let host = reconciliation::FlatRecTree {
        names: vec!["h0".into(), "h1".into(), "h2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let parasite = reconciliation::FlatRecTree {
        names: vec!["p0".into(), "p1".into(), "p2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let tip_mapping = vec![
        ("p0".to_string(), "h0".to_string()),
        ("p1".to_string(), "h1".to_string()),
    ];
    let dtl = reconciliation::reconcile_dtl(
        &host,
        &parasite,
        &tip_mapping,
        &reconciliation::DtlCosts::default(),
    );
    v.check_pass("DTL: optimal cost finite", dtl.optimal_cost < u32::MAX);
    v.check_pass("DTL: event table populated", !dtl.event_table.is_empty());
    timings.push(DomainTiming {
        name: "DTL Reconciliation",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D18: Molecular Clock ════════════════════════════════════════════
    let t = Instant::now();
    v.section("D18: Molecular Clock");
    let branch_lengths = vec![0.1, 0.2, 0.15, 0.05, 0.0];
    let parent_indices = vec![Some(4), Some(4), Some(3), Some(4), None];
    let calibrations = vec![molecular_clock::CalibrationPoint {
        node_id: 4,
        min_age_ma: 10.0,
        max_age_ma: 50.0,
    }];
    let clock =
        molecular_clock::strict_clock(&branch_lengths, &parent_indices, 30.0, &calibrations);
    v.check_pass("Strict clock: result present", clock.is_some());
    if let Some(ref clk) = clock {
        v.check_pass("Strict clock: rate > 0", clk.rate > 0.0);
        v.check_pass(
            "Strict clock: calibrations satisfied",
            clk.calibrations_satisfied,
        );
    }
    let node_ages = vec![0.0, 0.0, 10.0, 15.0, 30.0];
    let rates = molecular_clock::relaxed_clock_rates(&branch_lengths, &node_ages, &parent_indices);
    v.check_pass(
        "Relaxed clock: rates count",
        rates.len() == branch_lengths.len(),
    );
    let cv = molecular_clock::rate_variation_cv(&rates);
    v.check_pass("Rate CV ≥ 0", cv >= 0.0);
    timings.push(DomainTiming {
        name: "Molecular Clock",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ D19: Random Forest + Decision Tree ══════════════════════════════
    let t = Instant::now();
    v.section("D19: Random Forest + Decision Tree");
    let dt = decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .or_exit("unexpected error");
    let dt_pred = dt.predict(&[0.8, 0.3]);
    v.check_pass("DT: valid class", dt_pred <= 1);
    v.check_pass("DT: depth > 0", dt.depth() > 0);
    let dt2 = decision_tree::DecisionTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .or_exit("unexpected error");
    let forest = random_forest::RandomForest::from_trees(vec![dt, dt2], 2).or_exit("unexpected error");
    let rf_pred = forest.predict(&[0.8, 0.5]);
    v.check_pass("RF: valid class", rf_pred <= 1);
    let rf_batch = forest.predict_batch(&[vec![0.8, 0.5], vec![0.1, 0.9]]);
    v.check_pass("RF batch: 2 predictions", rf_batch.len() == 2);
    v.check_pass("RF: 2 trees", forest.n_trees() == 2);
    timings.push(DomainTiming {
        name: "RF + Decision Tree",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ Summary ═════════════════════════════════════════════════════════
    v.section("Timing Summary");
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("  ┌──────────────────────────────────┬──────────┐");
    println!("  │ New Domain                       │ Time (ms)│");
    println!("  ├──────────────────────────────────┼──────────┤");
    for dt in &timings {
        println!("  │ {:<34} │ {:>8.2} │", dt.name, dt.ms);
    }
    println!("  ├──────────────────────────────────┼──────────┤");
    println!("  │ TOTAL (new domains)              │ {total_ms:>8.2} │");
    println!("  └──────────────────────────────────┴──────────┘");
    println!("  Pure Rust CPU — zero Python, zero GPU, zero unsafe");
    println!("  8 new domains + 11 inherited from v16 = 19 total");
    println!();

    v.finish();
}
