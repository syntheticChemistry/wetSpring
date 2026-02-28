#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-27
# Commit: wetSpring Phase 66
"""
Spectral match baseline — NPU int8 triage reference (Exp124).

Generates Python reference values for `validate_npu_spectral_triage`.
Uses the same LCG RNG and histogram binning to produce library spectra,
query spectra, int8 quantisation, and cosine similarity ground truth.

This script independently reproduces the full NPU triage pipeline in
Python to validate the Rust implementation.

Pipeline:
  1. Generate 5000 library spectra (LCG RNG, matching Rust)
  2. Generate 100 query spectra (perturbed copies of known matches)
  3. 128-bin histogram encoding (m/z 50–1000)
  4. Int8 quantisation (scale to [-128, 127])
  5. Int8 dot-product triage (top-20% candidates)
  6. Full f64 cosine scoring on candidate set
  7. Recall and top-1 match metrics

Reproduction:
    python3 scripts/spectral_match_baseline.py

Requires: numpy
Python: 3.10+
"""

import json
import math
import os
from pathlib import Path

import numpy as np

LIB_SIZE = 5_000
N_QUERIES = 100
BINS = 128
MZ_MIN = 50.0
MZ_MAX = 1000.0
BIN_WIDTH = (MZ_MAX - MZ_MIN) / BINS
LCG_MULT = 6_364_136_223_846_793_005
LCG_MOD = 2**64
U32_MAX = 4_294_967_295


def lcg(seed):
    seed = (seed * LCG_MULT + 1) % LCG_MOD
    u = (seed >> 33) / U32_MAX
    return seed, u


def generate_spectrum(seed, n_peaks):
    rng = seed
    peaks = []
    for _ in range(n_peaks):
        rng, u1 = lcg(rng)
        mz = MZ_MIN + u1 * (MZ_MAX - MZ_MIN)
        rng, u2 = lcg(rng)
        intensity = max(u2 * u2 * 1000.0, 1.0)
        peaks.append((mz, intensity))
    return peaks


def spectrum_to_histogram(spectrum):
    hist = [0.0] * BINS
    for mz, intensity in spectrum:
        if MZ_MIN <= mz < MZ_MAX:
            b = int((mz - MZ_MIN) / BIN_WIDTH)
            b = min(b, BINS - 1)
            hist[b] += intensity
    return hist


def quantize_int8(v):
    max_abs = max(v) if v else 0.0
    scale = 127.0 / max_abs if max_abs > 0 else 1.0
    return [max(-128, min(127, round(x * scale))) for x in v]


def dot_int8(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity_spectra(spec_a, spec_b, tolerance_da=2.0):
    """Greedy cosine similarity matching peaks within tolerance."""
    matched_a = []
    matched_b = []
    used_b = set()
    for mz_a, int_a in spec_a:
        best_j = -1
        best_diff = tolerance_da + 1.0
        for j, (mz_b, _) in enumerate(spec_b):
            if j in used_b:
                continue
            diff = abs(mz_a - mz_b)
            if diff < best_diff:
                best_diff = diff
                best_j = j
        if best_j >= 0 and best_diff <= tolerance_da:
            matched_a.append(int_a)
            matched_b.append(spec_b[best_j][1])
            used_b.add(best_j)
    if not matched_a:
        return 0.0
    dot = sum(a * b for a, b in zip(matched_a, matched_b))
    norm_a = math.sqrt(sum(a ** 2 for a in matched_a))
    norm_b = math.sqrt(sum(b ** 2 for b in matched_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    results = {}

    # S1: Generate library
    library = []
    for i in range(LIB_SIZE):
        n_peaks = 20 + (i % 181)
        library.append(generate_spectrum(i, n_peaks))
    results["library_size"] = LIB_SIZE

    # S2: Generate queries
    queries = []
    true_match_idx = []
    for q in range(N_QUERIES):
        idx = min(q * (LIB_SIZE // N_QUERIES), LIB_SIZE - 1)
        query = list(library[idx])  # copy
        rng = 1_000_000 + q
        perturbed = []
        for mz, intensity in query:
            rng, u1 = lcg(rng)
            mz_new = mz + (u1 * 2.0 - 1.0) * 1.0
            rng, u2 = lcg(rng)
            int_new = max(intensity * (0.9 + u2 * 0.2), 0.1)
            perturbed.append((mz_new, int_new))
        for _ in range(3):
            rng, u1 = lcg(rng)
            mz_noise = MZ_MIN + u1 * (MZ_MAX - MZ_MIN)
            rng, u2 = lcg(rng)
            int_noise = u2 * 500.0
            perturbed.append((mz_noise, int_noise))
        queries.append(perturbed)
        true_match_idx.append(idx)
    results["n_queries"] = N_QUERIES

    # S3: Histogram encoding
    lib_hists = [spectrum_to_histogram(s) for s in library]
    query_hists = [spectrum_to_histogram(s) for s in queries]

    # S4: Int8 quantisation and triage
    lib_i8 = [quantize_int8(h) for h in lib_hists]
    query_i8 = [quantize_int8(h) for h in query_hists]
    k = int(LIB_SIZE * 0.20)

    recall_count = 0
    top1_correct = 0
    total_candidates = 0

    for q in range(N_QUERIES):
        scores = [(i, dot_int8(query_i8[q], lib_i8[i])) for i in range(LIB_SIZE)]
        scores.sort(key=lambda x: -x[1])
        top_k = [s[0] for s in scores[:k]]
        total_candidates += len(top_k)

        # Recall
        if true_match_idx[q] in top_k:
            recall_count += 1

        # Top-1 from full cosine
        best_score = -1.0
        best_idx = -1
        for cand_idx in top_k:
            score = cosine_similarity_spectra(queries[q], library[cand_idx])
            if score > best_score:
                best_score = score
                best_idx = cand_idx
        if best_idx == true_match_idx[q]:
            top1_correct += 1

    pass_rate = total_candidates / (N_QUERIES * LIB_SIZE)
    recall = recall_count / N_QUERIES
    top1_rate = top1_correct / N_QUERIES

    results["triage"] = {
        "k": k,
        "pass_rate": pass_rate,
        "recall": recall,
        "recall_count": recall_count,
        "top1_rate": top1_rate,
        "top1_correct": top1_correct,
    }

    # S5: Verification values
    self_sim = cosine_similarity_spectra(library[0], library[0])
    cross_sim = cosine_similarity_spectra(library[0], library[LIB_SIZE // 2])
    results["verification"] = {
        "self_similarity": self_sim,
        "cross_similarity": cross_sim,
    }

    out_dir = Path("experiments/results/124_npu_spectral_triage")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spectral_match_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nNPU Spectral Triage (lib={LIB_SIZE}, queries={N_QUERIES}, bins={BINS}):")
    print(f"  Pass rate: {pass_rate:.4f} (target < 0.30)")
    print(f"  Recall:    {recall:.3f} ({recall_count}/{N_QUERIES})")
    print(f"  Top-1:     {top1_rate:.3f} ({top1_correct}/{N_QUERIES})")
    print(f"  Self-sim:  {self_sim:.6f}")
    print(f"  Cross-sim: {cross_sim:.6f}")
    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
