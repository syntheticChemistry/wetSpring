#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Hsueh/Severin 2022 phage defense deaminase — Python baseline.

References:
    Hsueh, Severin et al. 2022, Nature Microbiology 7:1210-1220
"""

import json, os
import numpy as np
from scipy.integrate import odeint

PARAMS = {
    "mu_max": 1.0, "defense_cost": 0.15, "k_resource": 0.5,
    "yield_coeff": 0.5, "adsorption_rate": 1e-7, "burst_size": 50.0,
    "defense_efficiency": 0.9, "phage_decay": 0.1,
    "resource_inflow": 10.0, "resource_dilution": 0.1, "death_rate": 0.05,
}

def monod(r, k):
    return r / (k + r)

def defense_rhs(state, t, p):
    bd, bu, phage, r = [max(s, 0) for s in state]
    gl = monod(r, p["k_resource"])
    mu_d = p["mu_max"] * (1 - p["defense_cost"]) * gl
    mu_u = p["mu_max"] * gl
    inf_d = p["adsorption_rate"] * bd * phage
    inf_u = p["adsorption_rate"] * bu * phage
    kill_d = inf_d * (1 - p["defense_efficiency"])
    d_bd = mu_d * bd - kill_d - p["death_rate"] * bd
    d_bu = mu_u * bu - inf_u - p["death_rate"] * bu
    burst_u = p["burst_size"] * inf_u
    burst_d = p["burst_size"] * (1 - p["defense_efficiency"]) * inf_d
    d_phage = burst_u + burst_d - p["phage_decay"] * phage - p["adsorption_rate"] * (bd + bu) * phage
    consumption = p["yield_coeff"] * (mu_d * bd + mu_u * bu)
    d_r = p["resource_inflow"] - consumption - p["resource_dilution"] * r
    return [d_bd, d_bu, d_phage, d_r]

def run_ss(y0, params, t_end=48.0, dt=0.001):
    t = np.arange(0, t_end, dt)
    sol = odeint(defense_rhs, y0, t, args=(params,))
    tail = int(len(t) * 0.1)
    return sol[-tail:].mean(axis=0)

def main():
    names = ["Bd", "Bu", "P", "R"]
    results = {}
    scenarios = {
        "no_phage": ([1e6, 1e6, 0, 10], PARAMS),
        "phage_attack": ([1e6, 1e6, 1e4, 10], PARAMS),
        "pure_defended": ([1e6, 0, 1e4, 10], PARAMS),
        "pure_undefended": ([0, 1e6, 1e4, 10], PARAMS),
        "high_cost": ([1e6, 1e6, 1e4, 10], {**PARAMS, "defense_cost": 0.5}),
    }
    for name, (ic, p) in scenarios.items():
        ss = run_ss(ic, p)
        results[name] = {n: float(v) for n, v in zip(names, ss)}
        vals = " ".join(f"{n}={v:.2f}" for n, v in zip(names, ss))
        print(f"{name:20s}: {vals}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "030_phage_defense")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hsueh2022_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")

if __name__ == "__main__":
    main()
