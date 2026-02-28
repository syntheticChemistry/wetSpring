#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-27
# Commit: wetSpring Phase 66
"""
Zuber & Villamil 2016 — No-till meta-analysis baseline.

Generates Python baselines for Exp174 validation. Reproduces meta-analysis
effect sizes for no-till vs conventional tillage microbial indicators.

Reference: Zuber & Villamil (2016) Soil Biology and Biochemistry 97:176–187

Published effect sizes:
  - MBC: +16.2% (CI 10.9–21.8%, n=56)
  - MBN: +12.4% (CI 5.2–20.1%, n=34)
  - β-glucosidase: +15.0% (CI 8.0–22.5%, n=28)
  - Phosphatase: +18.0% (CI 10.0–27.0%, n=24)

Models:
  - Anderson disorder prediction of enrichment
  - Depth-stratified no-till effects
  - Temporal recovery function

Reproduction:
    python3 scripts/zuber2016_meta_analysis.py

Requires: numpy, scipy
Python: 3.10+
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import norm


W_C_3D = 16.5

EFFECT_SIZES = [
    {"indicator": "MBC", "effect_pct": 16.2, "ci_lower": 10.9, "ci_upper": 21.8, "n_studies": 56},
    {"indicator": "MBN", "effect_pct": 12.4, "ci_lower": 5.2, "ci_upper": 20.1, "n_studies": 34},
    {"indicator": "beta_glucosidase", "effect_pct": 15.0, "ci_lower": 8.0, "ci_upper": 22.5, "n_studies": 28},
    {"indicator": "phosphatase", "effect_pct": 18.0, "ci_lower": 10.0, "ci_upper": 27.0, "n_studies": 24},
]

DEPTH_LAYERS = [
    {"label": "0-5 cm", "depth_cm": 2.5, "notill_w": 4.0, "tilled_w": 18.0},
    {"label": "5-15 cm", "depth_cm": 10.0, "notill_w": 6.0, "tilled_w": 14.0},
    {"label": "15-30 cm", "depth_cm": 22.5, "notill_w": 10.0, "tilled_w": 12.0},
    {"label": ">30 cm", "depth_cm": 40.0, "notill_w": 11.0, "tilled_w": 11.5},
]


def main():
    results = {}

    # Anderson model — aggregate-level
    notill_w = 5.0
    tilled_w = 15.0
    notill_qs = norm.cdf((W_C_3D - notill_w) / 3.0)
    tilled_qs = norm.cdf((W_C_3D - tilled_w) / 3.0)
    predicted_enrichment = (notill_qs / tilled_qs - 1.0) * 100.0

    results["anderson_aggregate"] = {
        "notill_w": notill_w,
        "tilled_w": tilled_w,
        "notill_qs": float(notill_qs),
        "tilled_qs": float(tilled_qs),
        "predicted_enrichment_pct": predicted_enrichment,
    }

    # Meta-analysis effect sizes with statistical tests
    meta_results = []
    for es in EFFECT_SIZES:
        se = (es["ci_upper"] - es["ci_lower"]) / (2 * 1.96)
        z = es["effect_pct"] / se
        p = 2.0 * (1.0 - norm.cdf(z))
        meta_results.append({
            "indicator": es["indicator"],
            "effect_pct": es["effect_pct"],
            "ci_lower": es["ci_lower"],
            "ci_upper": es["ci_upper"],
            "n_studies": es["n_studies"],
            "se": se,
            "z_score": z,
            "p_value": float(p),
            "significant": bool(p < 0.05),
        })
    results["effect_sizes"] = meta_results

    # Depth-stratified analysis
    depth_results = []
    for layer in DEPTH_LAYERS:
        nt_qs = norm.cdf((W_C_3D - layer["notill_w"]) / 3.0)
        ti_qs = norm.cdf((W_C_3D - layer["tilled_w"]) / 3.0)
        enrichment = (nt_qs / ti_qs - 1.0) * 100.0 if ti_qs > 0 else float("inf")
        depth_results.append({
            "label": layer["label"],
            "depth_cm": layer["depth_cm"],
            "notill_w": layer["notill_w"],
            "tilled_w": layer["tilled_w"],
            "notill_qs": float(nt_qs),
            "tilled_qs": float(ti_qs),
            "enrichment_pct": enrichment,
        })
    results["depth_layers"] = depth_results

    # Temporal recovery
    recovery_tau = 10.0
    w_initial = 18.0
    w_final = 4.0
    years = [1, 5, 10, 20, 40]
    recovery = []
    for y in years:
        frac = 1.0 - math.exp(-y / recovery_tau)
        w = w_initial - (w_initial - w_final) * frac
        qs = norm.cdf((W_C_3D - w) / 3.0)
        recovery.append({
            "years": y,
            "fraction_recovered": frac,
            "w": w,
            "qs_prob": float(qs),
        })
    results["temporal_recovery"] = recovery

    # Math verification
    results["math_verification"] = {
        "erf_1": float(erf(1.0)),
        "norm_cdf_0": float(norm.cdf(0.0)),
        "norm_cdf_1_96": float(norm.cdf(1.96)),
    }

    out_dir = Path("experiments/results/174_notill_meta_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zuber2016_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nAnderson model: NT_W={notill_w}, Tilled_W={tilled_w}")
    print(f"  Predicted enrichment: {predicted_enrichment:.2f}%")
    print(f"\nMeta-analysis effects:")
    for m in meta_results:
        print(f"  {m['indicator']}: {m['effect_pct']}% (p={m['p_value']:.2e}) {'*' if m['significant'] else ''}")
    print(f"\nDepth stratification:")
    for d in depth_results:
        print(f"  {d['label']}: enrichment={d['enrichment_pct']:.2f}%")
    print(f"\nTemporal recovery (tau={recovery_tau}):")
    for r in recovery:
        print(f"  {r['years']:2d}yr: W={r['w']:.3f}, P(QS)={r['qs_prob']:.6f}")
    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
