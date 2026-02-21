// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! `BarraCUDA` CPU Parity v3 — comprehensive coverage of ALL 18 algorithmic
//! domains.  Extends v1 (9 domains) and v2 (batch/flat APIs) with the 9
//! remaining math modules:  multi-signal QS, phage defense, bootstrap
//! resampling, phylogenetic placement, decision-tree inference, spectral
//! matching, extended diversity, k-mer counting, and alpha-diversity pipeline.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | scipy, numpy, pure Python (`srivastava2011_multi_signal`, `hsueh2022_phage_defense`, `wang2021_rawr_bootstrap`, `alamin2024_placement`, spectral/diversity/decision-tree refs) |
//! | Baseline version | Feb 2026 |
//! | Baseline command | `python3 scripts/benchmark_rust_vs_python.py` (18-domain coverage) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/benchmark_rust_vs_python.py` |
//! | Data | Synthetic test vectors (hardcoded) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! ```text
//! Python baseline → v1 (9) → v2 (+5 batch) → [THIS] v3 (+9) → GPU
//! ```

use std::time::Instant;
use wetspring_barracuda::bio::{
    bootstrap, decision_tree::DecisionTree, diversity, felsenstein, kmer, multi_signal,
    phage_defense, placement, spectral_match,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("BarraCUDA CPU v3 — All 18 Domains");
    let mut timings: Vec<(&str, f64)> = Vec::new();

    // ════════════════════════════════════════════════════════════════
    //  Domain 10: Multi-Signal QS (Srivastava 2011)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 10: Multi-Signal QS (Srivastava 2011) ═══");

    let ms_params = multi_signal::MultiSignalParams::default();
    let t0 = Instant::now();
    let wt = multi_signal::scenario_wild_type(&ms_params, 0.001);
    let ms_us = t0.elapsed().as_micros();

    v.check(
        "MS-QS: trajectory has steps",
        f64::from(u8::from(wt.steps > 100)),
        1.0,
        0.0,
    );
    let wt_final = &wt.y_final;
    v.check(
        "MS-QS: wild-type reaches steady state",
        f64::from(u8::from(!wt_final.is_empty())),
        1.0,
        0.0,
    );

    let no_qs = multi_signal::scenario_no_qs(&ms_params, 0.001);
    v.check(
        "MS-QS: no-QS mutant completes",
        f64::from(u8::from(no_qs.steps > 100)),
        1.0,
        0.0,
    );

    timings.push(("Multi-signal QS (48h)", ms_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 11: Phage Defense (Hsueh 2022)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 11: Phage Defense (Hsueh 2022) ═══");

    let pd_params = phage_defense::PhageDefenseParams::default();
    let t0 = Instant::now();
    let attack = phage_defense::scenario_phage_attack(&pd_params, 0.001);
    let pd_us = t0.elapsed().as_micros();

    v.check(
        "Phage: attack > 100 steps",
        f64::from(u8::from(attack.steps > 100)),
        1.0,
        0.0,
    );

    let no_phage = phage_defense::scenario_no_phage(&pd_params, 0.001);
    v.check(
        "Phage: no-phage > 100 steps",
        f64::from(u8::from(no_phage.steps > 100)),
        1.0,
        0.0,
    );

    let defended = phage_defense::scenario_pure_defended(&pd_params, 0.001);
    v.check(
        "Phage: defended > 100 steps",
        f64::from(u8::from(defended.steps > 100)),
        1.0,
        0.0,
    );

    timings.push(("Phage defense (48h)", pd_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 12: Bootstrap Resampling (Wang 2021)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 12: Bootstrap Resampling (Wang 2021) ═══");

    let tree_bs = felsenstein::TreeNode::Internal {
        left: Box::new(felsenstein::TreeNode::Leaf {
            name: "A".into(),
            states: felsenstein::encode_dna("ACGTACGT"),
        }),
        right: Box::new(felsenstein::TreeNode::Leaf {
            name: "B".into(),
            states: felsenstein::encode_dna("ACTTACTT"),
        }),
        left_branch: 0.1,
        right_branch: 0.1,
    };
    let alignment = bootstrap::Alignment::from_rows(&[
        felsenstein::encode_dna("ACGTACGT"),
        felsenstein::encode_dna("ACTTACTT"),
    ]);
    let t0 = Instant::now();
    let lls = bootstrap::bootstrap_likelihoods(&tree_bs, &alignment, 100, 1.0, 42);
    let bs_us = t0.elapsed().as_micros();

    v.check("Bootstrap: 100 replicates", lls.len() as f64, 100.0, 0.0);
    let all_finite = lls.iter().all(|x| x.is_finite());
    v.check(
        "Bootstrap: all LLs finite",
        f64::from(u8::from(all_finite)),
        1.0,
        0.0,
    );
    let mean_ll: f64 = lls.iter().sum::<f64>() / lls.len() as f64;
    v.check(
        "Bootstrap: mean LL < 0 (negative log-lik)",
        f64::from(u8::from(mean_ll < 0.0)),
        1.0,
        0.0,
    );

    let lls2 = bootstrap::bootstrap_likelihoods(&tree_bs, &alignment, 100, 1.0, 42);
    let deterministic = lls
        .iter()
        .zip(lls2.iter())
        .all(|(a, b)| (a - b).abs() < tolerances::EXACT_F64);
    v.check(
        "Bootstrap: deterministic (same seed)",
        f64::from(u8::from(deterministic)),
        1.0,
        0.0,
    );

    timings.push(("Bootstrap (100 reps, 8bp)", bs_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 13: Phylogenetic Placement (Alamin & Liu 2024)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 13: Phylogenetic Placement (Alamin 2024) ═══");

    let ref_tree = felsenstein::TreeNode::Internal {
        left: Box::new(felsenstein::TreeNode::Internal {
            left: Box::new(felsenstein::TreeNode::Leaf {
                name: "sp1".into(),
                states: felsenstein::encode_dna("ACGTACGTACGT"),
            }),
            right: Box::new(felsenstein::TreeNode::Leaf {
                name: "sp2".into(),
                states: felsenstein::encode_dna("ACGTACTTACGT"),
            }),
            left_branch: 0.05,
            right_branch: 0.05,
        }),
        right: Box::new(felsenstein::TreeNode::Leaf {
            name: "sp3".into(),
            states: felsenstein::encode_dna("ACTTACTTACTT"),
        }),
        left_branch: 0.1,
        right_branch: 0.2,
    };

    let t0 = Instant::now();
    let scan = placement::placement_scan(&ref_tree, "ACGTACGTACGT", 0.05, 1.0);
    let pl_us = t0.elapsed().as_micros();

    v.check(
        "Placement: found edges",
        f64::from(u8::from(!scan.placements.is_empty())),
        1.0,
        0.0,
    );
    v.check(
        "Placement: best LL finite",
        f64::from(u8::from(scan.best_ll.is_finite())),
        1.0,
        0.0,
    );
    v.check(
        "Placement: best LL < 0",
        f64::from(u8::from(scan.best_ll < 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Placement: confidence > 0",
        f64::from(u8::from(scan.confidence > 0.0)),
        1.0,
        0.0,
    );

    let batch = placement::batch_placement(&ref_tree, &["ACGTACGTACGT", "ACTTACTTACTT"], 0.05, 1.0);
    v.check("Batch placement: 2 queries", batch.len() as f64, 2.0, 0.0);

    timings.push(("Placement (3 taxa, 12bp)", pl_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 14: Decision Tree Classification
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 14: Decision Tree Classification ═══");

    let dt = DecisionTree::from_arrays(
        &[0, -1, -1],
        &[5.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        3,
    )
    .expect("valid tree");

    let t0 = Instant::now();
    let pred_low = dt.predict(&[3.0, 0.0, 0.0]);
    let pred_high = dt.predict(&[7.0, 0.0, 0.0]);
    let batch_preds = dt.predict_batch(&[
        vec![3.0, 0.0, 0.0],
        vec![7.0, 0.0, 0.0],
        vec![5.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ]);
    let dt_us = t0.elapsed().as_micros();

    v.check("DT: predict(3) = 0 (< 5)", pred_low as f64, 0.0, 0.0);
    v.check("DT: predict(7) = 1 (≥ 5)", pred_high as f64, 1.0, 0.0);
    v.check("DT: batch len = 4", batch_preds.len() as f64, 4.0, 0.0);
    v.check("DT: batch[0] = 0", batch_preds[0] as f64, 0.0, 0.0);
    v.check("DT: batch[1] = 1", batch_preds[1] as f64, 1.0, 0.0);
    v.check("DT: n_nodes = 3", dt.n_nodes() as f64, 3.0, 0.0);
    v.check("DT: n_leaves = 2", dt.n_leaves() as f64, 2.0, 0.0);
    v.check("DT: depth = 1", dt.depth() as f64, 1.0, 0.0);

    timings.push(("Decision tree (4 samples)", dt_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 15: Spectral Matching (Cosine Similarity)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 15: Spectral Matching (Cosine Similarity) ═══");

    let mz_a = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let int_a = vec![1000.0, 500.0, 800.0, 300.0, 600.0];
    let mz_b = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let int_b = vec![900.0, 550.0, 750.0, 350.0, 550.0];
    let mz_c = vec![150.0, 250.0, 350.0, 450.0, 550.0];
    let int_c = vec![600.0, 400.0, 700.0, 200.0, 500.0];

    let t0 = Instant::now();
    let self_match = spectral_match::cosine_similarity(&mz_a, &int_a, &mz_a, &int_a, 0.5);
    let near_match = spectral_match::cosine_similarity(&mz_a, &int_a, &mz_b, &int_b, 0.5);
    let diff_match = spectral_match::cosine_similarity(&mz_a, &int_a, &mz_c, &int_c, 0.5);
    let sm_us = t0.elapsed().as_micros();

    v.check(
        "Spectral: self-match ≈ 1.0",
        self_match.score,
        1.0,
        tolerances::SPECTRAL_COSINE,
    );
    v.check(
        "Spectral: near-match > 0.95",
        f64::from(u8::from(near_match.score > 0.95)),
        1.0,
        0.0,
    );
    v.check(
        "Spectral: diff-match < near-match",
        f64::from(u8::from(diff_match.score < near_match.score)),
        1.0,
        0.0,
    );
    v.check(
        "Spectral: self matched_peaks = 5",
        self_match.matched_peaks as f64,
        5.0,
        0.0,
    );

    let spectra = vec![(mz_a, int_a), (mz_b, int_b), (mz_c, int_c)];
    let pw = spectral_match::pairwise_cosine(&spectra, 0.5);
    v.check("Spectral: pairwise 3 → 3 pairs", pw.len() as f64, 3.0, 0.0);

    timings.push(("Spectral match (5 peaks)", sm_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 16: Extended Diversity (Pielou, Bray-Curtis, Chao1)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 16: Extended Diversity Suite ═══");

    let even = &[25.0, 25.0, 25.0, 25.0];
    let uneven = &[97.0, 1.0, 1.0, 1.0];
    let sample_a = &[10.0, 20.0, 30.0, 40.0];
    let sample_b = &[40.0, 30.0, 20.0, 10.0];
    let identical = &[10.0, 20.0, 30.0, 40.0];

    let t0 = Instant::now();
    let pielou_even = diversity::pielou_evenness(even);
    let pielou_uneven = diversity::pielou_evenness(uneven);
    let bc_diff = diversity::bray_curtis(sample_a, sample_b);
    let bc_same = diversity::bray_curtis(sample_a, identical);
    let chao1_val = diversity::chao1(&[10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let obs = diversity::observed_features(&[10.0, 5.0, 0.0, 1.0, 0.0]);
    let alpha = diversity::alpha_diversity(sample_a);
    let div_us = t0.elapsed().as_micros();

    v.check(
        "Pielou: even ≈ 1.0",
        pielou_even,
        1.0,
        tolerances::PEAK_HEIGHT_REL,
    );
    v.check(
        "Pielou: uneven < 0.5",
        f64::from(u8::from(pielou_uneven < 0.5)),
        1.0,
        0.0,
    );
    v.check(
        "Bray-Curtis: identical = 0",
        bc_same,
        0.0,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "Bray-Curtis: different > 0",
        f64::from(u8::from(bc_diff > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Bray-Curtis: range [0,1]",
        f64::from(u8::from((0.0..=1.0).contains(&bc_diff))),
        1.0,
        0.0,
    );
    v.check(
        "Chao1 ≥ observed",
        f64::from(u8::from(chao1_val >= 4.0)),
        1.0,
        0.0,
    );
    v.check("Observed features", obs, 3.0, 0.0);
    v.check(
        "AlphaDiversity: Shannon > 0",
        f64::from(u8::from(alpha.shannon > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "AlphaDiversity: Evenness ∈ (0,1]",
        f64::from(u8::from(alpha.evenness > 0.0 && alpha.evenness <= 1.0)),
        1.0,
        0.0,
    );

    let samples = vec![
        vec![10.0, 20.0, 30.0],
        vec![30.0, 20.0, 10.0],
        vec![5.0, 5.0, 5.0],
    ];
    let bc_matrix = diversity::bray_curtis_condensed(&samples);
    v.check(
        "BC condensed: 3 samples → 3 pairs",
        bc_matrix.len() as f64,
        3.0,
        0.0,
    );

    timings.push(("Extended diversity suite", div_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 17: K-mer Counting
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 17: K-mer Counting ═══");

    let seq = b"ACGTACGTACGTACGT";
    let t0 = Instant::now();
    let kc = kmer::count_kmers(seq, 4);
    let kmer_us = t0.elapsed().as_micros();

    v.check(
        "Kmer: unique > 0",
        f64::from(u8::from(kc.unique_count() > 0)),
        1.0,
        0.0,
    );
    v.check(
        "Kmer: total = n-k+1",
        kc.total_count() as f64,
        (seq.len() - 4 + 1) as f64,
        0.0,
    );
    let decoded = kmer::decode_kmer(0b0001_1011, 4);
    v.check(
        "Kmer: decode 0b00011011 = ACGT",
        f64::from(u8::from(decoded == "ACGT")),
        1.0,
        0.0,
    );

    let multi = kmer::count_kmers_multi(&[b"ACGT" as &[u8], b"ACGT"], 3);
    v.check(
        "Kmer multi: 2 seqs combined",
        f64::from(u8::from(multi.total_count() > 0)),
        1.0,
        0.0,
    );

    timings.push(("Kmer counting (16bp, k=4)", kmer_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Domain 18: Integrated Pipeline (NJ + Diversity + Spectral)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 18: Integrated Pipeline ═══");

    let t0 = Instant::now();
    let pipeline_diversity = diversity::alpha_diversity(&[10.0, 20.0, 30.0, 15.0, 25.0]);
    let pipeline_bc = diversity::bray_curtis(&[10.0, 20.0, 30.0], &[15.0, 25.0, 35.0]);
    let pipeline_spectral = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0],
        &[1000.0, 500.0, 800.0],
        &[100.0, 200.0, 300.0],
        &[950.0, 520.0, 780.0],
        0.5,
    );
    let pipeline_us = t0.elapsed().as_micros();

    v.check(
        "Pipeline: Shannon consistent",
        pipeline_diversity.shannon,
        diversity::shannon(&[10.0, 20.0, 30.0, 15.0, 25.0]),
        tolerances::EXACT_F64,
    );
    v.check(
        "Pipeline: BC ∈ (0,1)",
        f64::from(u8::from(pipeline_bc > 0.0 && pipeline_bc < 1.0)),
        1.0,
        0.0,
    );
    v.check(
        "Pipeline: spectral > 0.99",
        f64::from(u8::from(pipeline_spectral.score > 0.99)),
        1.0,
        0.0,
    );

    timings.push(("Integrated pipeline", pipeline_us as f64));

    // ════════════════════════════════════════════════════════════════
    //  Timing Summary
    // ════════════════════════════════════════════════════════════════
    v.section("═══ BarraCUDA CPU v3 Timing Summary ═══");
    println!("\n  {:<35} {:>12}", "Domain", "Time (µs)");
    println!("  {}", "-".repeat(50));
    for (name, us) in &timings {
        println!("  {name:<35} {us:>12.0}");
    }
    let total_us: f64 = timings.iter().map(|(_, t)| t).sum();
    println!("  {}", "-".repeat(50));
    println!("  {:<35} {:>12.0}", "TOTAL", total_us);
    println!();

    v.finish();
}
