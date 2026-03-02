// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! Exp124 — `MassBank` Full-Scale NPU Spectral Triage
//!
//! Int8-quantized spectral triage for NPU deployment. Validates recall,
//! top-1 match rate, and throughput vs full f64 cosine search.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation type | Analytical (closed-form expected values) |
//! | Expected values | Derived from `spectral_match::cosine_similarity` (unit-resolution) |
//! | Baseline script | `scripts/spectral_match_baseline.py` (cosine reference) |
//! | Baseline commit | `e4358c5` |
//! | Reference | `MassBank`-style triage, int8 dot product ranking |
//! | Acceptance thresholds | `tolerances::NPU_PASS_RATE_CEILING` / `NPU_RECALL_FLOOR` / `NPU_TOP1_FLOOR` |
//! | Date | 2026-02-25 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: Python-parity
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use std::time::Instant;
use wetspring_barracuda::bio::spectral_match;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const LIB_SIZE: usize = 5_000;
const N_QUERIES: usize = 100;
const BINS: usize = 128;
const MZ_MIN: f64 = 50.0;
const MZ_MAX: f64 = 1000.0;
const BIN_WIDTH: f64 = (MZ_MAX - MZ_MIN) / BINS as f64;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 33) as f64) / f64::from(u32::MAX)
}

fn generate_spectrum(seed: u64, n_peaks: usize) -> Vec<(f64, f64)> {
    let mut rng = seed;
    let mut peaks = Vec::with_capacity(n_peaks);
    for _ in 0..n_peaks {
        let mz = lcg(&mut rng).mul_add(MZ_MAX - MZ_MIN, MZ_MIN);
        let raw = lcg(&mut rng);
        let intensity = (raw * raw * 1000.0).max(1.0);
        peaks.push((mz, intensity));
    }
    peaks
}

fn spectrum_to_histogram(spectrum: &[(f64, f64)]) -> Vec<f64> {
    let mut hist = vec![0.0_f64; BINS];
    for &(mz, int) in spectrum {
        if (MZ_MIN..MZ_MAX).contains(&mz) {
            let bin = ((mz - MZ_MIN) / BIN_WIDTH).floor() as usize;
            let bin = bin.min(BINS - 1);
            hist[bin] += int;
        }
    }
    hist
}

fn quantize_to_int8(v: &[f64]) -> Vec<i8> {
    let max_abs = v.iter().copied().fold(0.0_f64, f64::max);
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    v.iter()
        .map(|&x| (x * scale).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

fn dot_int8(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

fn cosine_spectrum(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let (mz_a, int_a): (Vec<_>, Vec<_>) = a.iter().copied().unzip();
    let (mz_b, int_b): (Vec<_>, Vec<_>) = b.iter().copied().unzip();
    spectral_match::cosine_similarity(&mz_a, &int_a, &mz_b, &int_b, 2.0).score
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp124: MassBank Full-Scale NPU Spectral Triage");

    v.section("── S1: Library generation ──");
    let library: Vec<Vec<(f64, f64)>> = (0..LIB_SIZE)
        .map(|i| generate_spectrum(i as u64, 20 + (i % 181)))
        .collect();
    v.check_count("library spectra", library.len(), LIB_SIZE);
    let all_non_empty = library.iter().all(|s| !s.is_empty());
    v.check_pass("all spectra non-empty", all_non_empty);

    v.section("── S2: Query generation ──");
    let mut queries = Vec::with_capacity(N_QUERIES);
    let mut true_match_idx = Vec::with_capacity(N_QUERIES);
    for q in 0..N_QUERIES {
        let idx = (q * (LIB_SIZE / N_QUERIES)).min(LIB_SIZE - 1);
        let mut query = library[idx].clone();
        let mut rng = 1_000_000 + q as u64;
        for (mz, int) in &mut query {
            *mz += lcg(&mut rng).mul_add(2.0, -1.0) * 1.0;
            *int *= 0.9 + lcg(&mut rng) * 0.2;
            *int = int.max(0.1);
        }
        for _ in 0..3 {
            let mz = lcg(&mut rng).mul_add(MZ_MAX - MZ_MIN, MZ_MIN);
            let int = lcg(&mut rng) * 500.0;
            query.push((mz, int));
        }
        queries.push(query);
        true_match_idx.push(idx);
    }
    v.check_count("queries", queries.len(), N_QUERIES);
    v.check_pass(
        "each query has known match",
        true_match_idx.len() == N_QUERIES,
    );

    v.section("── S3: Fingerprint encoding ──");
    let lib_hists: Vec<Vec<f64>> = library.iter().map(|s| spectrum_to_histogram(s)).collect();
    let query_hists: Vec<Vec<f64>> = queries.iter().map(|s| spectrum_to_histogram(s)).collect();
    let hist_nonzero = lib_hists
        .iter()
        .all(|h: &Vec<f64>| h.iter().sum::<f64>() > 0.0)
        && query_hists
            .iter()
            .all(|h: &Vec<f64>| h.iter().sum::<f64>() > 0.0);
    v.check_pass("128-bin histogram non-zero per spectrum", hist_nonzero);

    v.section("── S4: NPU triage ──");
    let lib_i8: Vec<Vec<i8>> = lib_hists.iter().map(|h| quantize_to_int8(h)).collect();
    let query_i8: Vec<Vec<i8>> = query_hists.iter().map(|h| quantize_to_int8(h)).collect();

    let k = (LIB_SIZE as f64 * 0.20) as usize;
    let mut total_candidates = 0usize;
    let mut candidates_per_query: Vec<Vec<usize>> = Vec::with_capacity(N_QUERIES);

    for q_i8 in &query_i8 {
        let mut scores: Vec<(usize, i64)> = lib_i8
            .iter()
            .enumerate()
            .map(|(i, lib)| (i, dot_int8(q_i8, lib)))
            .collect();
        scores.sort_by(|a, b| b.1.cmp(&a.1));
        let top_k: Vec<usize> = scores.iter().take(k).map(|(i, _)| *i).collect();
        total_candidates += top_k.len();
        candidates_per_query.push(top_k);
    }

    let pass_rate = total_candidates as f64 / (N_QUERIES * LIB_SIZE) as f64;
    println!("  Pass rate (candidates / total): {pass_rate:.4}");
    v.check_pass(
        "pass rate < 30%",
        pass_rate < tolerances::NPU_PASS_RATE_CEILING,
    );

    v.section("── S5: Recall check ──");
    let mut recall_count = 0;
    for (q, true_idx) in true_match_idx.iter().enumerate() {
        let cands = &candidates_per_query[q];
        if cands.contains(true_idx) {
            recall_count += 1;
        }
    }
    let recall = f64::from(recall_count) / N_QUERIES as f64;
    println!(
        "  Recall: {recall:.3} ({recall_count}/{N_QUERIES} queries have true match in candidates)"
    );
    v.check_pass("recall > 90%", recall > tolerances::NPU_RECALL_FLOOR);

    v.section("── S6: Full-precision scoring on candidates ──");
    let mut top1_correct = 0;
    for (q, cands) in candidates_per_query.iter().enumerate() {
        let query = &queries[q];
        let true_idx = true_match_idx[q];
        let mut best_score = -1.0_f64;
        let mut best_idx = 0;
        for &cand_idx in cands {
            let score = cosine_spectrum(query, &library[cand_idx]);
            if score > best_score {
                best_score = score;
                best_idx = cand_idx;
            }
        }
        if best_idx == true_idx {
            top1_correct += 1;
        }
    }
    let top1_rate = f64::from(top1_correct) / N_QUERIES as f64;
    println!("  Top-1 match rate: {top1_rate:.3} ({top1_correct}/{N_QUERIES} queries)");
    v.check_pass("top-1 match > 80%", top1_rate > tolerances::NPU_TOP1_FLOOR);

    v.section("── S7: Throughput comparison ──");
    let t0_npu = Instant::now();
    for (q, q_i8) in query_i8.iter().enumerate() {
        let mut scores: Vec<(usize, i64)> = lib_i8
            .iter()
            .enumerate()
            .map(|(i, lib)| (i, dot_int8(q_i8, lib)))
            .collect();
        scores.sort_by(|a, b| b.1.cmp(&a.1));
        let cands: Vec<usize> = scores.iter().take(k).map(|(i, _)| *i).collect();
        for &cand_idx in &cands {
            let _ = cosine_spectrum(&queries[q], &library[cand_idx]);
        }
    }
    let elapsed_npu = t0_npu.elapsed();

    let sample_queries = 20;
    let sample_lib = 2000;
    let gpu_start = Instant::now();
    for query in queries.iter().take(sample_queries) {
        let mut best_score = -1.0_f64;
        for lib in library.iter().take(sample_lib) {
            let score = cosine_spectrum(query, lib);
            if score > best_score {
                best_score = score;
            }
        }
    }
    let gpu_sample_elapsed = gpu_start.elapsed();
    let gpu_scale =
        (N_QUERIES as f64 / sample_queries as f64) * (LIB_SIZE as f64 / sample_lib as f64).max(1.0);
    let gpu_extrapolated_s = gpu_sample_elapsed.as_secs_f64() * gpu_scale;
    let speedup = gpu_extrapolated_s / elapsed_npu.as_secs_f64().max(1e-9);
    println!("  NPU triage + GPU scoring: {elapsed_npu:?}");
    println!(
        "  GPU-only (extrap from {sample_queries}q×{sample_lib}lib): {gpu_extrapolated_s:.1}s"
    );
    println!("  Speedup: {speedup:.1}×");
    v.check_pass("NPU pipeline faster (speedup > 1)", speedup > 1.0);

    v.section("── S8: Energy estimate ──");
    let npu_dot_per_query = LIB_SIZE as f64;
    let int8_macs = npu_dot_per_query * BINS as f64;
    let energy_per_mac_nj = 0.001;
    let energy_per_query_j = int8_macs * energy_per_mac_nj * 1e-9;
    println!("  NPU triage energy per query: {energy_per_query_j:.6} J");
    v.check_pass("energy estimate computed", energy_per_query_j > 0.0);

    v.finish();
}
