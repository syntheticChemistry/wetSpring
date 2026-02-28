#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-27
# Commit: wetSpring Phase 66
"""
Martínez-García et al. 2023 — Pore geometry → QS activation baseline.

Generates Python baselines for Exp170 validation. Reproduces the key
finding: spatial structure (pore geometry) + chemotaxis + quorum sensing
jointly determine bacterial biomass accumulation in complex porous media.

Reference: Martínez-García et al. (2023) Nature Communications 14:8332

Models:
  - Pore size → Anderson effective dimension → QS probability
  - Chemotaxis disorder reduction (15% shift of W_c)
  - Integrated pore geometry + QS + cooperation predictions

Reproduction:
    python3 scripts/martinez2023_pore_geometry.py

Requires: numpy, scipy
Python: 3.10+
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm


W_C_3D = 16.5
CHEMOTAXIS_REDUCTION = 0.15


def pore_to_anderson(pore_um):
    """Map pore size (µm) to Anderson disorder W and QS probability."""
    connectivity = min((pore_um / 75.0) ** 2, 1.0)
    effective_w = 25.0 * (1.0 - connectivity)
    qs_prob = norm.cdf((W_C_3D - effective_w) / 3.0)
    return connectivity, effective_w, qs_prob


def chemotaxis_benefit(w, w_c=W_C_3D, reduction=CHEMOTAXIS_REDUCTION):
    """QS probability gain from chemotaxis at disorder W."""
    p_no = norm.cdf((w_c - w) / 3.0)
    p_yes = norm.cdf((w_c - w * (1.0 - reduction)) / 3.0)
    return p_yes - p_no


def main():
    results = {}

    # S3: Pore geometry → Anderson disorder mapping
    pore_sizes = [4.0, 10.0, 30.0, 50.0, 100.0, 150.0]
    pore_results = []
    for pore in pore_sizes:
        conn, w, qs = pore_to_anderson(pore)
        pore_results.append({
            "pore_um": pore,
            "connectivity": conn,
            "effective_w": w,
            "qs_probability": qs,
        })
    results["pore_mapping"] = pore_results

    # S4: Chemotaxis disorder reduction
    w_eff_chemo = W_C_3D * (1.0 - CHEMOTAXIS_REDUCTION)
    disorder_values = [10.0, 14.0, 16.5, 20.0, 25.0]
    chemo_results = []
    for w in disorder_values:
        benefit = chemotaxis_benefit(w)
        p_no = norm.cdf((W_C_3D - w) / 3.0)
        p_yes = norm.cdf((W_C_3D - w * (1.0 - CHEMOTAXIS_REDUCTION)) / 3.0)
        chemo_results.append({
            "w": w,
            "p_no_chemotaxis": p_no,
            "p_with_chemotaxis": p_yes,
            "benefit": benefit,
        })
    results["chemotaxis"] = {
        "w_effective_with_chemotaxis": w_eff_chemo,
        "reduction_pct": CHEMOTAXIS_REDUCTION * 100,
        "disorder_sweep": chemo_results,
    }

    # S6: Integrated predictions
    scenarios = [
        {"name": "Sandy loam (large pores, 100µm)", "pore_um": 100.0},
        {"name": "Clay (small pores, 5µm)", "pore_um": 5.0},
        {"name": "No-till aggregate (mixed, 80µm)", "pore_um": 80.0},
        {"name": "Tilled (destroyed aggregates, 15µm)", "pore_um": 15.0},
    ]
    integrated = []
    for s in scenarios:
        conn, w, qs = pore_to_anderson(s["pore_um"])
        coop = 0.6 + 0.3 * conn if qs > 0.5 else 0.2 * conn
        integrated.append({
            "name": s["name"],
            "pore_um": s["pore_um"],
            "connectivity": conn,
            "effective_w": w,
            "qs_probability": qs,
            "qs_active": bool(qs > 0.5),
            "coop_survival": coop,
        })
    results["integrated"] = integrated

    # S7: CPU math verification
    from scipy.special import erf as scipy_erf
    results["math_verification"] = {
        "erf_1": float(scipy_erf(1.0)),
        "norm_cdf_0": float(norm.cdf(0.0)),
        "norm_cdf_1_96": float(norm.cdf(1.96)),
    }

    out_dir = Path("experiments/results/170_soil_qs_pore_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "martinez2023_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"\nPore mapping (W_c = {W_C_3D}):")
    for p in pore_results:
        print(f"  {p['pore_um']:6.0f}µm → W={p['effective_w']:.3f}, P(QS)={p['qs_probability']:.6f}")
    print(f"\nChemotaxis (15% reduction):")
    print(f"  W_eff = {w_eff_chemo:.4f}")
    for c in chemo_results:
        print(f"  W={c['w']:5.1f}: benefit={c['benefit']:.6f}")
    print(f"\nerf(1.0) = {results['math_verification']['erf_1']:.15f}")
    print(f"Φ(0)     = {results['math_verification']['norm_cdf_0']:.15f}")
    print(f"Φ(1.96)  = {results['math_verification']['norm_cdf_1_96']:.15f}")
    print("\nAll baselines computed successfully.")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
