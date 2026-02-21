#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anderson 2015 — Rare biosphere at deep-sea hydrothermal vents.

Generates Python baselines for Exp051 validation. Uses synthetic vent
community profiles modeled after Anderson, Sogin & Baross (2015) FEMS
Microbiol Ecol 91:fiu016.

Three synthetic communities:
  - Piccard: high-temperature, low diversity (dominated by 2-3 taxa)
  - Von Damm: moderate diversity (Campylobacteria-dominated)
  - Background seawater: high diversity, many rare lineages

Usage:
    python scripts/anderson2015_rare_biosphere.py

Requires: numpy (no external bioinformatics tools)
Python: 3.10+
Date: 2026-02-20
"""

import json
import math
import os
from pathlib import Path


def shannon(counts):
    """Shannon entropy H' = -sum(p_i * ln(p_i))."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def simpson(counts):
    """Simpson's diversity 1 - sum(p_i^2)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    s = 0.0
    for c in counts:
        p = c / total
        s += p * p
    return 1.0 - s


def chao1(counts):
    """Chao1 richness estimator."""
    observed = sum(1 for c in counts if c > 0)
    singletons = sum(1 for c in counts if c == 1)
    doubletons = sum(1 for c in counts if c == 2)
    if doubletons == 0:
        if singletons > 0:
            return observed + singletons * (singletons - 1) / 2.0
        return float(observed)
    return observed + (singletons * singletons) / (2.0 * doubletons)


def bray_curtis(a, b):
    """Bray-Curtis dissimilarity between two count vectors."""
    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    a_ext = list(a) + [0] * (max_len - len(a))
    b_ext = list(b) + [0] * (max_len - len(b))
    num = sum(abs(a_ext[i] - b_ext[i]) for i in range(max_len))
    den = sum(a_ext[i] + b_ext[i] for i in range(max_len))
    if den == 0:
        return 0.0
    return num / den


def observed(counts):
    """Count of non-zero features."""
    return sum(1 for c in counts if c > 0)


def pielou(counts):
    """Pielou's evenness J' = H' / ln(S)."""
    s = observed(counts)
    if s <= 1:
        return 0.0
    return shannon(counts) / math.log(s)


def rarefaction_curve(counts, depths):
    """Expected species at each rarefaction depth (analytical)."""
    n = sum(counts)
    s_obs = observed(counts)
    curve = []
    for d in depths:
        if d >= n:
            curve.append(float(s_obs))
            continue
        expected = 0.0
        for c in counts:
            if c > 0:
                numerator = 1.0
                for j in range(d):
                    numerator *= (n - c - j) / (n - j)
                expected += 1.0 - numerator
        curve.append(expected)
    return curve


def rare_lineage_count(counts, threshold=0.001):
    """Count lineages below threshold relative abundance."""
    total = sum(counts)
    if total == 0:
        return 0
    return sum(1 for c in counts if 0 < c / total < threshold)


# ── Synthetic vent communities ──────────────────────────────────────
# Modeled after Anderson 2015 Table S1 community structure

# Piccard: high-temperature black smoker, low diversity
# Dominated by Nautiliales and Campylobacteria
piccard = [
    500, 350, 80, 30, 15, 10, 5, 3, 2, 2, 1, 1, 1,
]

# Von Damm: moderate-temperature, Campylobacteria-dominated
von_damm = [
    300, 200, 150, 100, 80, 60, 40, 30, 25, 20, 15, 12, 10,
    8, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
]

# Background seawater: high diversity, many rare lineages
background = [
    50, 45, 40, 38, 35, 33, 30, 28, 26, 25,
    23, 22, 20, 19, 18, 17, 16, 15, 14, 13,
    12, 11, 10, 9, 8, 7, 6, 5, 5, 4,
    4, 3, 3, 3, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]


def main():
    communities = {
        "piccard": piccard,
        "von_damm": von_damm,
        "background": background,
    }

    results = {}

    for name, counts in communities.items():
        results[name] = {
            "shannon": shannon(counts),
            "simpson": simpson(counts),
            "chao1": chao1(counts),
            "observed": observed(counts),
            "pielou": pielou(counts),
            "rare_lineage_count": rare_lineage_count(counts),
            "total_reads": sum(counts),
        }

    # Bray-Curtis pairwise
    names = list(communities.keys())
    bc_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i <= j:
                bc = bray_curtis(communities[n1], communities[n2])
                bc_matrix[f"{n1}_vs_{n2}"] = bc

    results["bray_curtis"] = bc_matrix

    # Rarefaction curves
    depths = [10, 50, 100, 200, 500]
    for name, counts in communities.items():
        results[name]["rarefaction"] = rarefaction_curve(counts, depths)
        results[name]["rarefaction_depths"] = depths

    # Write output
    out_dir = Path("experiments/results/051_rare_biosphere")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anderson2015_python_baseline.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")

    # Print summary
    for name in communities:
        r = results[name]
        print(f"\n{name}:")
        print(f"  Shannon:  {r['shannon']:.12f}")
        print(f"  Simpson:  {r['simpson']:.12f}")
        print(f"  Chao1:    {r['chao1']:.6f}")
        print(f"  Observed: {r['observed']}")
        print(f"  Pielou:   {r['pielou']:.12f}")
        print(f"  Rare (<0.1%): {r['rare_lineage_count']}")

    print("\nBray-Curtis distances:")
    for pair, bc in bc_matrix.items():
        print(f"  {pair}: {bc:.12f}")

    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
