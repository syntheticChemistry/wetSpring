#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-27
# Commit: wetSpring Phase 66
"""
Rabot et al. 2018 — Soil structure–function baseline.

Generates Python baselines for Exp177 validation. Reproduces the key
finding: soil structure properties (aggregate stability, porosity, pore
connectivity) determine biological function through an Anderson-like
disorder mechanism.

Reference: Rabot et al. (2018) Geoderma 314:122–137

Models:
  - Structure → Anderson disorder mapping
  - Soil type diversity gradient
  - Management practice → QS probability

Reproduction:
    python3 scripts/rabot2018_structure_function.py

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
W_C_3D = 16.5


def lcg_next(state):
    state = (state * LCG_MULT + LCG_ADD) % LCG_MOD
    u = (state >> 33) / U32_MAX
    return state, u


def generate_community(richness, evenness, seed):
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


STRUCTURE_PROPERTIES = {
    "aggregate_stability": {"good": 3.0, "poor": 20.0},
    "porosity": {"good": 5.0, "poor": 15.0},
    "pore_connectivity": {"good": 4.0, "poor": 18.0},
    "pore_size_distribution": {"good": 5.0, "poor": 16.0},
}

SOIL_TYPES = [
    {"name": "Well-structured clay loam", "effective_w": 4.0, "base_richness": 250},
    {"name": "Moderate sandy loam", "effective_w": 10.0, "base_richness": 200},
    {"name": "Degraded silt", "effective_w": 18.0, "base_richness": 100},
    {"name": "Compacted clay", "effective_w": 22.0, "base_richness": 60},
]

MANAGEMENT = [
    {"practice": "Native forest", "agg_pct": 92, "porosity_pct": 65},
    {"practice": "No-till + cover crop", "agg_pct": 78, "porosity_pct": 55},
    {"practice": "Reduced tillage", "agg_pct": 55, "porosity_pct": 48},
    {"practice": "Conventional tillage", "agg_pct": 35, "porosity_pct": 42},
    {"practice": "Intensive + bare fallow", "agg_pct": 18, "porosity_pct": 35},
]


def main():
    results = {}

    # Structure → Anderson mapping reference
    results["structure_properties"] = STRUCTURE_PROPERTIES

    # Soil type diversity gradient
    soil_results = []
    for i, st in enumerate(SOIL_TYPES):
        eff_richness = max(5, int(st["base_richness"] * max(1.0 - st["effective_w"] / 30.0, 0.1)))
        evenness = 0.8 - st["effective_w"] / 50.0

        shannons = []
        for rep in range(3):
            seed = 42 + i * 10 + rep
            comm = generate_community(eff_richness, evenness, seed)
            shannons.append(shannon(comm))

        mean_h = float(np.mean(shannons))
        qs = norm.cdf((W_C_3D - st["effective_w"]) / 3.0)

        soil_results.append({
            "name": st["name"],
            "effective_w": st["effective_w"],
            "base_richness": st["base_richness"],
            "effective_richness": eff_richness,
            "evenness": evenness,
            "mean_shannon": mean_h,
            "qs_probability": float(qs),
        })
    results["soil_types"] = soil_results

    # Management practice gradient
    mgmt_results = []
    for m in MANAGEMENT:
        w = 25.0 * (1.0 - m["agg_pct"] / 100.0)
        qs = norm.cdf((W_C_3D - w) / 3.0)
        mgmt_results.append({
            "practice": m["practice"],
            "agg_stability_pct": m["agg_pct"],
            "porosity_pct": m["porosity_pct"],
            "effective_w": w,
            "qs_probability": float(qs),
        })
    results["management"] = mgmt_results

    # Math verification
    results["math_verification"] = {
        "norm_cdf_0": float(norm.cdf(0.0)),
        "shannon_uniform_5": math.log(5),
    }

    out_dir = Path("experiments/results/177_soil_structure_function")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rabot2018_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nSoil type diversity:")
    for s in soil_results:
        print(f"  {s['name']:30s}: W={s['effective_w']:5.1f}, H'={s['mean_shannon']:.12f}, P(QS)={s['qs_probability']:.6f}")
    print(f"\nManagement gradient:")
    for m in mgmt_results:
        print(f"  {m['practice']:30s}: W={m['effective_w']:.2f}, P(QS)={m['qs_probability']:.6f}")
    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
