// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Smith-Waterman local alignment (Exp028).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Algorithm | Smith & Waterman 1981, *J Mol Biol* 147:195-197 |
//! | Baseline script | `scripts/smith_waterman_baseline.py` |
//! | Baseline commit | `e4358c5` |
//! | Date | 2026-02-21 |
//! | Exact command | `python3 scripts/smith_waterman_baseline.py` |

use wetspring_barracuda::bio::alignment::{
    AlignmentResult, ScoringParams, pairwise_scores, smith_waterman, smith_waterman_score,
};
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp028: Smith-Waterman Local Alignment");
    let params = ScoringParams::default();

    // ── Identical sequences ─────────────────────────────────────────
    v.section("── Identical sequences ──");
    let r = smith_waterman(b"ACGTACGT", b"ACGTACGT", &params);
    v.check("Identical: score", f64::from(r.score), 16.0, 0.0);
    check_alignment_valid(&mut v, &r, "Identical");

    // ── Simple mismatch ─────────────────────────────────────────────
    v.section("── Simple mismatch ──");
    let r = smith_waterman(b"ACGT", b"ACTT", &params);
    v.check(
        "Mismatch: score matches Python",
        f64::from(r.score),
        5.0,
        0.0,
    );

    // ── Gap alignment ───────────────────────────────────────────────
    v.section("── Gap alignment ──");
    let r = smith_waterman(b"ACGTACGT", b"ACGACGT", &params);
    v.check("Gap: score matches Python", f64::from(r.score), 10.0, 0.0);
    check_alignment_valid(&mut v, &r, "Gap");

    // ── Local alignment ─────────────────────────────────────────────
    v.section("── Local alignment (embedded match) ──");
    let r = smith_waterman(b"XXXACGTACGTXXX", b"ACGTACGT", &params);
    v.check(
        "Local: score (finds 8bp match)",
        f64::from(r.score),
        16.0,
        0.0,
    );

    // ── No match ────────────────────────────────────────────────────
    v.section("── No match ──");
    let harsh = ScoringParams {
        match_score: 1,
        mismatch_penalty: -3,
        gap_open: -5,
        gap_extend: -2,
    };
    let r = smith_waterman(b"AAAA", b"CCCC", &harsh);
    v.check("NoMatch: score = 0", f64::from(r.score), 0.0, 0.0);

    // ── 16S fragment alignment ──────────────────────────────────────
    v.section("── 16S fragment (40bp, 2 mismatches) ──");
    let q = b"GATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATAC";
    let t = b"GATCCTGGCTCAGAATGAACGCTGGCGGCATGCCTAATAC";
    let r = smith_waterman(q, t, &params);
    v.check("16S: score matches Python", f64::from(r.score), 74.0, 0.0);
    check_alignment_valid(&mut v, &r, "16S");

    // ── Score-only consistency ──────────────────────────────────────
    v.section("── Score-only matches full alignment ──");
    let q = b"ACGTACGTACGT";
    let t = b"ACTTACGTACTT";
    let full_score = smith_waterman(q, t, &params).score;
    let fast_score = smith_waterman_score(q, t, &params);
    v.check(
        "ScoreOnly == Full",
        f64::from(u8::from(full_score == fast_score)),
        1.0,
        0.0,
    );

    // ── Batch pairwise ──────────────────────────────────────────────
    v.section("── Batch pairwise scores ──");
    let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACTT", b"GGGG"];
    let scores = pairwise_scores(&seqs, &params);
    v.check(
        "Pairwise: 3 sequences → 3 pairs",
        #[allow(clippy::cast_precision_loss)]
        {
            scores.len() as f64
        },
        3.0,
        0.0,
    );
    v.check("Pairwise: ACGT-ACTT score", f64::from(scores[0]), 5.0, 0.0);

    // ── Case insensitive ────────────────────────────────────────────
    v.section("── Case insensitive ──");
    let s1 = smith_waterman(b"ACGT", b"acgt", &params).score;
    let s2 = smith_waterman(b"ACGT", b"ACGT", &params).score;
    v.check("Case insensitive", f64::from(u8::from(s1 == s2)), 1.0, 0.0);

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let r1 = smith_waterman(b"ACGTACGTACGT", b"ACTTACGTACTT", &params);
    let r2 = smith_waterman(b"ACGTACGTACGT", b"ACTTACGTACTT", &params);
    v.check(
        "Deterministic score",
        f64::from(u8::from(r1.score == r2.score)),
        1.0,
        0.0,
    );
    v.check(
        "Deterministic alignment",
        f64::from(u8::from(r1.aligned_query == r2.aligned_query)),
        1.0,
        0.0,
    );

    v.finish();
}

fn check_alignment_valid(v: &mut Validator, r: &AlignmentResult, pre: &str) {
    let same_len = r.aligned_query.len() == r.aligned_target.len();
    v.check(
        &format!("{pre}: aligned lengths match"),
        f64::from(u8::from(same_len)),
        1.0,
        0.0,
    );
}
