#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-27
# Commit: wetSpring Phase 66
"""
Feng et al. 2024 — Pore-scale microbial diversity baseline.

Generates Python baselines for Exp171 validation. Reproduces the key
finding: large pores (30–150 µm) harbor higher microbial diversity than
small pores (4–10 µm) due to habitat heterogeneity.

Reference: Feng et al. (2024) Nature Communications 15:3578

Models:
  - Synthetic community generation (LCG RNG matching Rust)
  - Shannon, Simpson, Bray–Curtis diversity metrics
  - Pore size → Anderson disorder → QS probability

Reproduction:
    python3 scripts/feng2024_pore_diversity.py

Requires: numpy, scipy
Python: 3.10+
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm


LCG_MULT = 6_364_136_223_846_793_005
LCG_ADD = 1
LCG_MOD = 2**64
U32_MAX = 4_294_967_295


def lcg_next(state):
    state = (state * LCG_MULT + LCG_ADD) % LCG_MOD
    u = (state >> 33) / U32_MAX
    return state, u


def generate_community(richness, evenness, seed):
    """Generate synthetic abundance vector matching Rust's LCG."""
    state = seed % LCG_MOD
    abundances = []
    for _ in range(richness):
        state, u = lcg_next(state)
        raw = max(u * (1 - evenness) + evenness, 0.001)
        abundances.append(raw)
    total = sum(abundances)
    return [a / total for a in abundances]


def shannon(probs):
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def simpson(probs):
    return 1.0 - sum(p * p for p in probs)


def bray_curtis(a, b):
    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    a_ext = list(a) + [0.0] * (max_len - len(a))
    b_ext = list(b) + [0.0] * (max_len - len(b))
    num = sum(min(a_ext[i], b_ext[i]) for i in range(max_len))
    den_a = sum(a_ext)
    den_b = sum(b_ext)
    if den_a + den_b == 0:
        return 0.0
    return 1.0 - 2.0 * num / (den_a + den_b)


PORE_CLASSES = [
    {"name": "Macro (100-150 µm)", "size_um": 125.0, "richness": 200, "evenness": 0.8},
    {"name": "Meso (30-100 µm)", "size_um": 65.0, "richness": 150, "evenness": 0.7},
    {"name": "Micro (10-30 µm)", "size_um": 20.0, "richness": 80, "evenness": 0.5},
    {"name": "Nano (4-10 µm)", "size_um": 7.0, "richness": 30, "evenness": 0.3},
]

N_REPS = 3


def main():
    results = {}

    all_communities = {}
    class_shannons = []

    for ci, pc in enumerate(PORE_CLASSES):
        rep_shannons = []
        rep_simpsons = []
        communities = []

        for rep in range(N_REPS):
            seed = ci * 1000 + rep + 42
            comm = generate_community(pc["richness"], pc["evenness"], seed)
            communities.append(comm)
            h = shannon(comm)
            d = simpson(comm)
            rep_shannons.append(h)
            rep_simpsons.append(d)

        mean_h = np.mean(rep_shannons)
        mean_d = np.mean(rep_simpsons)
        class_shannons.append(mean_h)
        all_communities[pc["name"]] = communities

        # Anderson mapping
        connectivity = pc["size_um"] / 150.0
        effective_w = 25.0 * (1.0 - connectivity)
        qs_prob = norm.cdf((16.5 - effective_w) / 3.0)

        results[pc["name"]] = {
            "size_um": pc["size_um"],
            "richness": pc["richness"],
            "evenness": pc["evenness"],
            "mean_shannon": float(mean_h),
            "mean_simpson": float(mean_d),
            "rep_shannons": [float(x) for x in rep_shannons],
            "rep_simpsons": [float(x) for x in rep_simpsons],
            "anderson_w": effective_w,
            "qs_probability": float(qs_prob),
        }

    # Diversity ordering check
    results["diversity_ordering"] = {
        "macro_gt_meso": bool(class_shannons[0] > class_shannons[1]),
        "meso_gt_micro": bool(class_shannons[1] > class_shannons[2]),
        "micro_gt_nano": bool(class_shannons[2] > class_shannons[3]),
        "macro_nano_ratio": class_shannons[0] / class_shannons[3],
    }

    # Beta diversity (within vs between)
    within_bcs = []
    between_bcs = []
    for ci, pc in enumerate(PORE_CLASSES):
        comms = all_communities[pc["name"]]
        for i in range(N_REPS):
            for j in range(i + 1, N_REPS):
                within_bcs.append(bray_curtis(comms[i], comms[j]))

    names = [pc["name"] for pc in PORE_CLASSES]
    for ci in range(len(PORE_CLASSES)):
        for cj in range(ci + 1, len(PORE_CLASSES)):
            comms_i = all_communities[names[ci]]
            comms_j = all_communities[names[cj]]
            for i in range(N_REPS):
                for j in range(N_REPS):
                    between_bcs.append(bray_curtis(comms_i[i], comms_j[j]))

    results["beta_diversity"] = {
        "mean_within": float(np.mean(within_bcs)),
        "mean_between": float(np.mean(between_bcs)),
        "ratio": float(np.mean(between_bcs) / np.mean(within_bcs)) if np.mean(within_bcs) > 0 else float("inf"),
    }

    # Math verification
    results["math_verification"] = {
        "norm_cdf_0": float(norm.cdf(0.0)),
        "shannon_uniform_4": math.log(4),
        "simpson_uniform_4": 0.75,
    }

    out_dir = Path("experiments/results/171_soil_pore_diversity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feng2024_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    for pc in PORE_CLASSES:
        r = results[pc["name"]]
        print(f"\n{pc['name']}:")
        print(f"  Shannon: {r['mean_shannon']:.12f}")
        print(f"  Simpson: {r['mean_simpson']:.12f}")
        print(f"  Anderson W: {r['anderson_w']:.3f}")
        print(f"  P(QS): {r['qs_probability']:.6f}")
    do = results["diversity_ordering"]
    print(f"\nMacro/Nano Shannon ratio: {do['macro_nano_ratio']:.4f}")
    bd = results["beta_diversity"]
    print(f"Beta: within={bd['mean_within']:.6f}, between={bd['mean_between']:.6f}, ratio={bd['ratio']:.4f}")
    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
