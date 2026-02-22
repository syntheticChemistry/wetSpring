#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""
Srivastava 2011 multi-input QS network — Python baseline.

Dual-signal (CAI-1 + AI-2) QS model with LuxO phosphorelay.
5 scenarios: wild type, CAI-1 only, AI-2 only, no QS, exogenous CAI-1.

Usage:
    python srivastava2011_multi_signal.py

References:
    Srivastava et al. 2011, J Bacteriology 193:6331-41
"""

import json
import os
import numpy as np
from scipy.integrate import odeint


PARAMS = {
    "mu_max": 0.8, "k_cap": 1.0, "death_rate": 0.02,
    "k_cai1_prod": 3.0, "d_cai1": 1.0, "k_cqs": 0.5,
    "k_ai2_prod": 3.0, "d_ai2": 1.0, "k_luxpq": 0.5,
    "k_luxo_phos": 2.0, "d_luxo_p": 0.5,
    "k_hapr_max": 1.0, "n_repress": 2.0, "k_repress": 0.5, "d_hapr": 0.5,
    "k_dgc_basal": 2.0, "k_dgc_rep": 0.8,
    "k_pde_basal": 0.5, "k_pde_act": 2.0, "d_cdg": 0.3,
    "k_bio_max": 1.0, "k_bio_cdg": 1.5, "n_bio": 2.0, "d_bio": 0.2,
}


def hill(x, k, n):
    if x <= 0: return 0.0
    xn = x**n
    return xn / (k**n + xn)


def hill_repress(x, k, n):
    if x <= 0: return 1.0
    kn = k**n
    return kn / (kn + x**n)


def multi_rhs(state, t, p):
    cell = max(state[0], 0)
    cai1 = max(state[1], 0)
    ai2 = max(state[2], 0)
    luxo_p = max(state[3], 0)
    hapr = max(state[4], 0)
    cdg = max(state[5], 0)
    bio = max(state[6], 0)

    d_cell = p["mu_max"] * cell * (1 - cell/p["k_cap"]) - p["death_rate"] * cell
    d_cai1 = p["k_cai1_prod"] * cell - p["d_cai1"] * cai1
    d_ai2 = p["k_ai2_prod"] * cell - p["d_ai2"] * ai2

    dephos_cai1 = hill(cai1, p["k_cqs"], 2.0)
    dephos_ai2 = hill(ai2, p["k_luxpq"], 2.0)
    total_dephos = dephos_cai1 + dephos_ai2
    d_luxo_p = p["k_luxo_phos"] - (p["d_luxo_p"] + total_dephos) * luxo_p

    d_hapr = p["k_hapr_max"] * hill_repress(luxo_p, p["k_repress"], p["n_repress"]) - p["d_hapr"] * hapr

    dgc_rate = p["k_dgc_basal"] * max(1 - p["k_dgc_rep"] * hapr, 0)
    pde_rate = p["k_pde_basal"] + p["k_pde_act"] * hapr
    d_cdg = dgc_rate - pde_rate * cdg - p["d_cdg"] * cdg
    if cdg < 1e-12 and d_cdg < 0:
        d_cdg = 0

    bio_promote = p["k_bio_max"] * hill(cdg, p["k_bio_cdg"], p["n_bio"])
    d_bio = bio_promote * (1 - bio) - p["d_bio"] * bio

    return [d_cell, d_cai1, d_ai2, d_luxo_p, d_hapr, d_cdg, d_bio]


def run_ss(y0, params, t_end=24.0, dt=0.001):
    t = np.arange(0, t_end, dt)
    sol = odeint(multi_rhs, y0, t, args=(params,))
    tail = int(len(t) * 0.1)
    return sol[-tail:].mean(axis=0)


def main():
    y0 = [0.01, 0.0, 0.0, 2.0, 0.0, 2.0, 0.5]
    results = {}

    # Wild type
    ss = run_ss(y0, PARAMS)
    results["wild_type"] = {k: float(v) for k, v in zip(
        ["N", "CAI1", "AI2", "LuxOP", "HapR", "CdG", "B"], ss)}
    print(f"WT:     N={ss[0]:.4f} H={ss[4]:.4f} C={ss[5]:.4f} B={ss[6]:.4f}")

    # CAI-1 only
    p = dict(PARAMS); p["k_ai2_prod"] = 0.0
    ss = run_ss(y0, p)
    results["cai1_only"] = {k: float(v) for k, v in zip(
        ["N", "CAI1", "AI2", "LuxOP", "HapR", "CdG", "B"], ss)}
    print(f"CAI1:   N={ss[0]:.4f} H={ss[4]:.4f} C={ss[5]:.4f} B={ss[6]:.4f}")

    # AI-2 only
    p = dict(PARAMS); p["k_cai1_prod"] = 0.0
    ss = run_ss(y0, p)
    results["ai2_only"] = {k: float(v) for k, v in zip(
        ["N", "CAI1", "AI2", "LuxOP", "HapR", "CdG", "B"], ss)}
    print(f"AI2:    N={ss[0]:.4f} H={ss[4]:.4f} C={ss[5]:.4f} B={ss[6]:.4f}")

    # No QS
    p = dict(PARAMS); p["k_cai1_prod"] = 0.0; p["k_ai2_prod"] = 0.0
    ss = run_ss(y0, p)
    results["no_qs"] = {k: float(v) for k, v in zip(
        ["N", "CAI1", "AI2", "LuxOP", "HapR", "CdG", "B"], ss)}
    print(f"No QS:  N={ss[0]:.4f} H={ss[4]:.4f} C={ss[5]:.4f} B={ss[6]:.4f}")

    # Exogenous CAI-1
    y0_exo = [0.01, 5.0, 0.0, 2.0, 0.0, 2.0, 0.5]
    ss = run_ss(y0_exo, PARAMS)
    results["exogenous_cai1"] = {k: float(v) for k, v in zip(
        ["N", "CAI1", "AI2", "LuxOP", "HapR", "CdG", "B"], ss)}
    print(f"ExoCAI: N={ss[0]:.4f} H={ss[4]:.4f} C={ss[5]:.4f} B={ss[6]:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "024_multi_signal")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "srivastava2011_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")


if __name__ == "__main__":
    main()
