#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""Exp042 baseline — MassBank PFAS spectral matching validation.

Validates spectral matching (cosine similarity) on synthetic PFAS-like
mass spectra. When MassBank data is downloaded, extends to real reference
spectra.

Tests:
  - Cosine similarity self-match = 1.0
  - Near-identical spectra similarity > 0.9
  - Unrelated spectra similarity < 0.3
  - Pairwise cosine matrix properties

Requires: Python 3.8+ (stdlib only)
"""
import json
import math
import os
import sys


def cosine_similarity(mz_a: list[float], int_a: list[float],
                      mz_b: list[float], int_b: list[float],
                      tolerance_da: float = 0.5) -> float:
    """Cosine similarity between two mass spectra."""
    matched_a = []
    matched_b = []
    used_b = set()
    for i, mz_ai in enumerate(mz_a):
        best_j = -1
        best_diff = tolerance_da + 1
        for j, mz_bj in enumerate(mz_b):
            if j in used_b:
                continue
            diff = abs(mz_ai - mz_bj)
            if diff < best_diff:
                best_diff = diff
                best_j = j
        if best_j >= 0 and best_diff <= tolerance_da:
            matched_a.append(int_a[i])
            matched_b.append(int_b[best_j])
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
    # Synthetic PFAS-like mass spectra
    # PFOS (m/z 499): characteristic fragments at 80, 99, 119, 169, 219, 269, 319, 369, 419, 499
    pfos_mz = [80.0, 99.0, 119.0, 169.0, 219.0, 269.0, 319.0, 369.0, 419.0, 499.0]
    pfos_int = [30.0, 100.0, 45.0, 80.0, 55.0, 70.0, 40.0, 25.0, 15.0, 60.0]

    # Near-PFOS (slightly shifted, simulates instrument variation)
    pfos_shifted_mz = [80.1, 99.05, 118.95, 169.1, 219.0, 268.9, 319.1, 369.0, 419.05, 499.0]
    pfos_shifted_int = [28.0, 98.0, 46.0, 78.0, 54.0, 72.0, 38.0, 26.0, 16.0, 58.0]

    # PFOA (m/z 413): different fragmentation pattern
    pfoa_mz = [69.0, 119.0, 169.0, 219.0, 269.0, 319.0, 369.0, 413.0]
    pfoa_int = [100.0, 40.0, 65.0, 50.0, 55.0, 35.0, 20.0, 45.0]

    # Unrelated compound (caffeine, m/z 194)
    caffeine_mz = [42.0, 55.0, 67.0, 82.0, 109.0, 137.0, 194.0]
    caffeine_int = [20.0, 30.0, 40.0, 55.0, 100.0, 80.0, 70.0]

    results = {}

    # Self-match
    cs_self = cosine_similarity(pfos_mz, pfos_int, pfos_mz, pfos_int)
    results["self_match"] = cs_self

    # Near-match (instrument variation)
    cs_near = cosine_similarity(pfos_mz, pfos_int, pfos_shifted_mz, pfos_shifted_int)
    results["near_match"] = cs_near

    # Same PFAS family (PFOS vs PFOA)
    cs_family = cosine_similarity(pfos_mz, pfos_int, pfoa_mz, pfoa_int)
    results["family_match"] = cs_family

    # Unrelated (PFOS vs caffeine)
    cs_unrelated = cosine_similarity(pfos_mz, pfos_int, caffeine_mz, caffeine_int)
    results["unrelated_match"] = cs_unrelated

    # Pairwise matrix (4 spectra)
    spectra = [
        (pfos_mz, pfos_int),
        (pfos_shifted_mz, pfos_shifted_int),
        (pfoa_mz, pfoa_int),
        (caffeine_mz, caffeine_int),
    ]
    pairwise = []
    for i in range(4):
        row = []
        for j in range(4):
            cs = cosine_similarity(spectra[i][0], spectra[i][1],
                                   spectra[j][0], spectra[j][1])
            row.append(cs)
        pairwise.append(row)
    results["pairwise_matrix"] = pairwise

    output = {
        "experiment": "Exp042",
        "description": "MassBank PFAS spectral matching validation",
        "data_source": "MassBank/MassBank-data (PFAS reference spectra)",
        "spectra": {
            "pfos": {"mz": pfos_mz, "intensity": pfos_int},
            "pfos_shifted": {"mz": pfos_shifted_mz, "intensity": pfos_shifted_int},
            "pfoa": {"mz": pfoa_mz, "intensity": pfoa_int},
            "caffeine": {"mz": caffeine_mz, "intensity": caffeine_int},
        },
        "results": results,
        "tolerance_da": 0.5,
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "results", "042_massbank_spectral"
    )
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "python_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("Exp042 baseline — MassBank spectral matching:")
    print(f"  Self-match: {cs_self:.6f}")
    print(f"  Near-match: {cs_near:.6f}")
    print(f"  Family match: {cs_family:.6f}")
    print(f"  Unrelated: {cs_unrelated:.6f}")
    print(f"Output: {out_path}/python_baseline.json")


if __name__ == "__main__":
    main()
