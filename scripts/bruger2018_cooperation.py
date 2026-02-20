#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Bruger & Waters 2018 — cooperative QS game theory Python baseline.

Two-population (cooperator/cheater) ODE model with frequency-dependent fitness.
5 scenarios: equal start, coop-dominated, cheat-dominated, pure coop, pure cheat.

References:
    Bruger & Waters 2018, AEM 84:e00402-18
"""

import json
import os
import numpy as np
from scipy.integrate import odeint


PARAMS = {
    "mu_coop": 0.7, "mu_cheat": 0.75,
    "k_cap": 1.0, "death_rate": 0.02,
    "k_ai_prod": 5.0, "d_ai": 1.0,
    "benefit": 0.3, "k_benefit": 0.5, "cost": 0.05,
    "k_bio": 1.0, "k_bio_ai": 0.5,
    "dispersal_bonus": 0.2, "d_bio": 0.3,
}


def hill(x, k):
    if x <= 0: return 0.0
    x2 = x**2
    return x2 / (k**2 + x2)


def coop_rhs(state, t, p):
    nc = max(state[0], 0)
    nd = max(state[1], 0)
    ai = max(state[2], 0)
    bio = max(state[3], 0)

    n_total = nc + nd
    crowding = max(1 - n_total / p["k_cap"], 0)

    signal_benefit = p["benefit"] * hill(ai, p["k_benefit"])
    dispersal = p["dispersal_bonus"] * (1 - bio)

    fitness_coop = (p["mu_coop"] - p["cost"] + signal_benefit + dispersal) * crowding
    fitness_cheat = (p["mu_cheat"] + signal_benefit + dispersal) * crowding

    d_nc = fitness_coop * nc - p["death_rate"] * nc
    d_nd = fitness_cheat * nd - p["death_rate"] * nd
    d_ai = p["k_ai_prod"] * nc - p["d_ai"] * ai
    d_bio = p["k_bio"] * hill(ai, p["k_bio_ai"]) * (1 - bio) - p["d_bio"] * bio

    return [d_nc, d_nd, d_ai, d_bio]


def run_ss(y0, params, t_end=48.0, dt=0.001):
    t = np.arange(0, t_end, dt)
    sol = odeint(coop_rhs, y0, t, args=(params,))
    tail = int(len(t) * 0.1)
    ss = sol[-tail:].mean(axis=0)
    freq = [nc / max(nc + nd, 1e-15) for nc, nd in zip(sol[:, 0], sol[:, 1])]
    return ss, freq[-1]


def main():
    results = {}

    scenarios = {
        "equal_start": [0.01, 0.01, 0.0, 0.0],
        "coop_dominated": [0.09, 0.01, 0.0, 0.0],
        "cheat_dominated": [0.01, 0.09, 0.0, 0.0],
        "pure_coop": [0.01, 0.0, 0.0, 0.0],
        "pure_cheat": [0.0, 0.01, 0.0, 0.0],
    }

    for name, y0 in scenarios.items():
        ss, final_freq = run_ss(y0, PARAMS)
        results[name] = {
            "Nc_ss": float(ss[0]),
            "Nd_ss": float(ss[1]),
            "AI_ss": float(ss[2]),
            "B_ss": float(ss[3]),
            "coop_freq": float(final_freq),
            "total_N": float(ss[0] + ss[1]),
        }
        print(f"{name:20s}: Nc={ss[0]:.4f} Nd={ss[1]:.4f} AI={ss[2]:.4f} "
              f"B={ss[3]:.4f} f_coop={final_freq:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "025_cooperation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bruger2018_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")


if __name__ == "__main__":
    main()
