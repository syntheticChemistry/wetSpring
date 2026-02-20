#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fernandez 2020 bistable phenotypic switching — Python baseline.

Generates steady-state values and bifurcation diagram for the bistable
QS/c-di-GMP/biofilm model from Fernandez et al. 2020 (PNAS 117:29046-29054).

Extends the Waters 2008 model with positive feedback from biofilm state (B)
onto DGC production, creating bistability and hysteresis.

Usage:
    python fernandez2020_bistable.py

Outputs:
    experiments/results/023_bistable/fernandez2020_python_baseline.json
"""

import json
import os
import numpy as np
from scipy.integrate import odeint

# ── Model parameters (Waters 2008 base + Fernandez 2020 feedback) ──

BASE_PARAMS = {
    "mu_max": 0.8,
    "k_cap": 1.0,
    "death_rate": 0.02,
    "k_ai_prod": 5.0,
    "d_ai": 1.0,
    "k_hapr_max": 1.0,
    "k_hapr_ai": 0.5,
    "n_hapr": 2.0,
    "d_hapr": 0.5,
    "k_dgc_basal": 2.0,
    "k_dgc_rep": 0.3,
    "k_pde_basal": 0.5,
    "k_pde_act": 0.5,
    "d_cdg": 0.3,
    "k_bio_max": 1.0,
    "k_bio_cdg": 1.5,       # high c-di-GMP required for biofilm → low-B attractor exists
    "n_bio": 4.0,            # steep switch (ultrasensitivity)
    "d_bio": 0.2,
    # Fernandez 2020 extensions
    "alpha_fb": 3.0,
    "n_fb": 4.0,             # steep cooperative feedback
    "k_fb": 0.6,             # moderate feedback threshold
}


def hill(x, k, n):
    if x <= 0:
        return 0.0
    xn = x**n
    return xn / (k**n + xn)


def bistable_rhs(state, t, params):
    cell = max(state[0], 0.0)
    ai = max(state[1], 0.0)
    hapr = max(state[2], 0.0)
    cdg = max(state[3], 0.0)
    bio = max(state[4], 0.0)

    p = params

    d_cell = p["mu_max"] * cell * (1.0 - cell / p["k_cap"]) - p["death_rate"] * cell
    d_ai = p["k_ai_prod"] * cell - p["d_ai"] * ai
    d_hapr = p["k_hapr_max"] * hill(ai, p["k_hapr_ai"], p["n_hapr"]) - p["d_hapr"] * hapr

    basal_dgc = p["k_dgc_basal"] * max(1.0 - p["k_dgc_rep"] * hapr, 0.0)
    feedback_dgc = p["alpha_fb"] * hill(bio, p["k_fb"], p["n_fb"])
    dgc_rate = basal_dgc + feedback_dgc

    pde_rate = p["k_pde_basal"] + p["k_pde_act"] * hapr
    d_cdg = dgc_rate - pde_rate * cdg - p["d_cdg"] * cdg
    if cdg < 1e-12 and d_cdg < 0:
        d_cdg = 0.0

    bio_promote = p["k_bio_max"] * hill(cdg, p["k_bio_cdg"], p["n_bio"])
    d_bio = bio_promote * (1.0 - bio) - p["d_bio"] * bio

    return [d_cell, d_ai, d_hapr, d_cdg, d_bio]


def run_to_steady_state(y0, params, t_end=48.0, dt=0.001):
    t = np.arange(0, t_end, dt)
    sol = odeint(bistable_rhs, y0, t, args=(params,))
    tail = int(len(t) * 0.1)
    ss = sol[-tail:].mean(axis=0)
    return ss, sol


def bifurcation_scan(base_params, lo=0.0, hi=6.0, n_steps=20, t_settle=48.0, dt=0.001):
    alphas = np.linspace(lo, hi, n_steps + 1)

    # Forward: start from motile state
    y = np.array([0.9, 4.0, 1.8, 0.1, 0.02])
    b_forward = []
    for alpha in alphas:
        p = dict(base_params)
        p["alpha_fb"] = alpha
        ss, sol = run_to_steady_state(y, p, t_settle, dt)
        b_forward.append(float(ss[4]))
        y = sol[-1]

    # Backward: start from sessile state
    y = np.array([0.9, 4.0, 1.8, 3.0, 0.9])
    b_backward = []
    for alpha in reversed(alphas):
        p = dict(base_params)
        p["alpha_fb"] = alpha
        ss, sol = run_to_steady_state(y, p, t_settle, dt)
        b_backward.append(float(ss[4]))
        y = sol[-1]
    b_backward.reverse()

    # Hysteresis width
    diverge = 0.1
    first, last = None, None
    for i, (f, b) in enumerate(zip(b_forward, b_backward)):
        if abs(f - b) > diverge:
            if first is None:
                first = i
            last = i
    width = float(alphas[last] - alphas[first]) if first is not None else 0.0

    return {
        "alphas": [float(a) for a in alphas],
        "b_forward": b_forward,
        "b_backward": b_backward,
        "hysteresis_width": width,
    }


def main():
    results = {}

    # ── Scenario 1: Zero feedback (recover Waters 2008) ──
    p_zero = dict(BASE_PARAMS)
    p_zero["alpha_fb"] = 0.0
    ss, _ = run_to_steady_state([0.01, 0.0, 0.0, 2.0, 0.5], p_zero)
    results["zero_feedback"] = {
        "N_ss": float(ss[0]),
        "A_ss": float(ss[1]),
        "H_ss": float(ss[2]),
        "C_ss": float(ss[3]),
        "B_ss": float(ss[4]),
    }
    print(f"Zero feedback: N={ss[0]:.6f}, B={ss[4]:.6f}")

    # ── Scenario 2: Default feedback (bistable) ──
    ss_motile, _ = run_to_steady_state([0.01, 0.0, 0.0, 0.1, 0.02], BASE_PARAMS)
    ss_sessile, _ = run_to_steady_state([0.01, 0.0, 0.0, 3.0, 0.9], BASE_PARAMS)
    results["default_motile"] = {
        "N_ss": float(ss_motile[0]),
        "B_ss": float(ss_motile[4]),
        "C_ss": float(ss_motile[3]),
    }
    results["default_sessile"] = {
        "N_ss": float(ss_sessile[0]),
        "B_ss": float(ss_sessile[4]),
        "C_ss": float(ss_sessile[3]),
    }
    print(f"Motile start:  N={ss_motile[0]:.6f}, B={ss_motile[4]:.6f}, C={ss_motile[3]:.6f}")
    print(f"Sessile start: N={ss_sessile[0]:.6f}, B={ss_sessile[4]:.6f}, C={ss_sessile[3]:.6f}")

    # ── Scenario 3: Strong feedback ──
    p_strong = dict(BASE_PARAMS)
    p_strong["alpha_fb"] = 8.0
    ss_strong, _ = run_to_steady_state([0.01, 0.0, 0.0, 2.0, 0.8], p_strong, t_end=48.0)
    results["strong_feedback"] = {
        "N_ss": float(ss_strong[0]),
        "B_ss": float(ss_strong[4]),
        "C_ss": float(ss_strong[3]),
    }
    print(f"Strong fb:     N={ss_strong[0]:.6f}, B={ss_strong[4]:.6f}, C={ss_strong[3]:.6f}")

    # ── Bifurcation scan ──
    bif = bifurcation_scan(BASE_PARAMS, lo=0.0, hi=10.0, n_steps=50)
    results["bifurcation"] = bif
    print(f"Hysteresis width: {bif['hysteresis_width']:.3f}")

    # ── Write output ──
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "experiments", "results", "023_bistable"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fernandez2020_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")


if __name__ == "__main__":
    main()
