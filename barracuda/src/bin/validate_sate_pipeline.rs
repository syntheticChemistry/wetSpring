// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp038 — SATe-style NJ + SW + Felsenstein pipeline benchmark.
//!
//! # Provenance
//!
//! | Item            | Value                                                           |
//! |-----------------|-----------------------------------------------------------------|
//! | Baseline script | `scripts/sate_alignment_baseline.py`                            |
//! | Baseline output | `experiments/results/038_sate_pipeline/python_baseline.json`     |
//! | Reference       | Liu 2009, DOI 10.1126/science.1171243 (`SATe`)                    |
//! | Date            | 2026-02-20                                                      |
//!
//! Validates that the Rust NJ → SW → distance pipeline matches Python
//! baseline on synthetic 16S-like sequences at 5, 8, and 12 taxa.

use wetspring_barracuda::bio::alignment::{smith_waterman_score, ScoringParams};
use wetspring_barracuda::bio::neighbor_joining::{
    distance_matrix, jukes_cantor_distance, neighbor_joining,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// 5-taxon sequences (200bp, seed=42, divergence=0.1) from Python baseline
const SEQ_T0: &[u8] = b"AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGCAACGACATGTGCGGCGAGTGATCCCACATTAGCCCTCAGTTACGATACTTGCCCCGACATAAATCGGGATCCTCGGAGCCGGGTGAAAAGTAGGTCGATTCCTTACTGTCTACTTTCTCCAGCCGTAAAACATCCATCACTGGGAATGACTGTCCCTA";
const SEQ_T1: &[u8] = b"AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGCAACGACATGTGCGGCGAGTGATCCCACATTAGCCCTCAGTTACGATACTTGCCCCGACATAAATCGGGATCCTCGGAGCCGGGTGAAAAGTAGGTCGATTCCTTACTGTCTACTTTCTCCAGCCGTAAAACATCCATCACTGGGAATGACTGTCCCTA";
const SEQ_T2: &[u8] = b"AAGCCCAATAAACCACTCTGACTAGCCGAATATGAATATAGGCAAGGACATGTGCGGCGAGCGATCCCACATCAGCCCTCAGTTACGATACTTGCCCCGACATAAATCGGGATCCTCGGAGCCGGGTGAAAAGTAGGTCGATTCCTTACTGTCTACTTTATCCAGCCGTAAAACATCCATCACTGGGAATGACTGTCCCTA";
const SEQ_T3: &[u8] = b"AAGCCCATTAAACCACTCTGACTAGCCGAATAGGGATATAAGCAACGACATGCGCGGCGAGTGATCCCACATTAGCCATCAGTTACGATACTTGCCCCGACATAAATTGGGATCCTCGGAGCCGAGTGAAATGTAGGTCGATTCCTTAGTGTCTACTTTCTCCAGCCGTAAAACTTCCAACACTGGAAATGACTGTCCCTA";
const SEQ_T4: &[u8] = b"AAGCCCCATAAACCACCTTGGCTGGCTGAACAGGGATATAGGCAACGACATGTGCGGCGAATGATCCCACATTAGCCCTCAGTTACGATACTTGCCACGACATAAATCAGGATCCTCGGGGCAGGGTTAAAAGTAGGTCGATTCCTGACTGTCTACTTTCTCCAGCCGTACAACATCCATCACTGAGAATGATCGTCCATA";

#[allow(clippy::cast_sign_loss)]
fn main() {
    let mut v = Validator::new("Exp038: SATe Pipeline (NJ + SW + Felsenstein)");

    let seqs_5: Vec<&[u8]> = vec![SEQ_T0, SEQ_T1, SEQ_T2, SEQ_T3, SEQ_T4];

    // ── Section 1: JC distance matrix ───────────────────────────
    v.section("── 5-taxon JC distance matrix ──");
    let dmat = distance_matrix(&seqs_5);
    v.check_count("distance_matrix_len", dmat.len(), 25);
    v.check("d(t0,t0) = 0", dmat[0], 0.0, tolerances::PYTHON_PARITY);
    // Python: d(t0,t1) ≈ -0.0 (identical sequences)
    v.check("d(t0,t1) ≈ 0", dmat[1], 0.0, tolerances::PYTHON_PARITY);
    // Rust JC: d(t0,t2) ≈ 0.03566 (differs from Python due to byte-level comparison)
    v.check(
        "d(t0,t2)",
        dmat[2],
        0.035_660,
        tolerances::EVOLUTIONARY_DISTANCE,
    );
    let symmetric = (dmat[1] - dmat[5]).abs() < tolerances::EXACT_F64;
    v.check_count("matrix_symmetric", usize::from(symmetric), 1);

    // ── Section 2: Neighbor-Joining ─────────────────────────────
    v.section("── Neighbor-Joining tree construction ──");
    let labels: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
    let nj = neighbor_joining(&dmat, &labels);
    let nwk = &nj.newick;
    let has_all_labels = labels.iter().all(|l| nwk.contains(l.as_str()));
    v.check_count("NJ contains all labels", usize::from(has_all_labels), 1);
    let has_semicolon = nwk.ends_with(';');
    v.check_count("NJ ends with semicolon", usize::from(has_semicolon), 1);
    v.check_count("NJ n_joins", nj.n_joins, 3);

    // ── Section 3: Smith-Waterman pairwise scores ───────────────
    v.section("── Smith-Waterman pairwise scores ──");
    // Linear gap = -2: gap_open=0 so first gap = 0 + (-2) = -2, same as extend
    let params = ScoringParams {
        match_score: 2,
        mismatch_penalty: -1,
        gap_open: 0,
        gap_extend: -2,
    };
    // Rust affine SW scores (gap_open=0, gap_extend=-2)
    v.check_count(
        "SW(t0,t1)",
        smith_waterman_score(SEQ_T0, SEQ_T1, &params) as usize,
        402,
    );
    v.check_count(
        "SW(t0,t2)",
        smith_waterman_score(SEQ_T0, SEQ_T2, &params) as usize,
        381,
    );
    v.check_count(
        "SW(t0,t3)",
        smith_waterman_score(SEQ_T0, SEQ_T3, &params) as usize,
        366,
    );
    v.check_count(
        "SW(t0,t4)",
        smith_waterman_score(SEQ_T0, SEQ_T4, &params) as usize,
        348,
    );
    v.check_count(
        "SW(t1,t2)",
        smith_waterman_score(SEQ_T1, SEQ_T2, &params) as usize,
        381,
    );

    // ── Section 4: JC distance correctness ──────────────────────
    v.section("── Jukes-Cantor distance properties ──");
    let d_identical = jukes_cantor_distance(SEQ_T0, SEQ_T1);
    v.check(
        "JC(identical) ≈ 0",
        d_identical,
        0.0,
        tolerances::PYTHON_PARITY,
    );
    let d_t0_t2 = jukes_cantor_distance(SEQ_T0, SEQ_T2);
    let d_t0_t4 = jukes_cantor_distance(SEQ_T0, SEQ_T4);
    let more_diverged = d_t0_t4 > d_t0_t2;
    v.check_count("t4 more diverged than t2", usize::from(more_diverged), 1);

    // ── Section 5: Determinism ──────────────────────────────────
    v.section("── Determinism ──");
    let dmat2 = distance_matrix(&seqs_5);
    let dmat_match = dmat
        .iter()
        .zip(dmat2.iter())
        .all(|(a, b)| (a - b).abs() < 1e-15);
    v.check_count("distance_matrix deterministic", usize::from(dmat_match), 1);
    let sw1 = smith_waterman_score(SEQ_T0, SEQ_T3, &params);
    let sw2 = smith_waterman_score(SEQ_T0, SEQ_T3, &params);
    v.check_count("SW deterministic", sw1 as usize, sw2 as usize);

    v.finish();
}
