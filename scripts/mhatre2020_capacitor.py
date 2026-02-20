#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Mhatre 2020 phenotypic capacitor — Python baseline.

References:
    Mhatre et al. 2020, PNAS 117:21647-21657
"""

import json, os
import numpy as np
from scipy.integrate import odeint

PARAMS = {
    "mu_max": 0.8, "k_cap": 1.0, "death_rate": 0.02,
    "k_cdg_prod": 2.0, "d_cdg": 0.5,
    "k_vpsr_charge": 1.0, "k_vpsr_discharge": 0.3,
    "n_vpsr": 3.0, "k_vpsr_cdg": 1.0,
    "w_biofilm": 0.8, "w_motility": 0.6, "w_rugose": 0.4,
    "d_bio": 0.3, "d_mot": 0.3, "d_rug": 0.3,
    "stress_factor": 1.0,
}

def hill(x, k, n):
    if x <= 0: return 0.0
    xn = x**n
    return xn / (k**n + xn)

def cap_rhs(state, t, p):
    cell, cdg, vpsr, bio, mot, rug = [max(s, 0) for s in state]
    d_cell = p["mu_max"] * cell * (1 - cell/p["k_cap"]) - p["death_rate"] * cell
    d_cdg = p["stress_factor"] * p["k_cdg_prod"] * cell - p["d_cdg"] * cdg
    charge = p["k_vpsr_charge"] * hill(cdg, p["k_vpsr_cdg"], p["n_vpsr"]) * (1 - vpsr)
    discharge = p["k_vpsr_discharge"] * vpsr
    d_vpsr = charge - discharge
    d_bio = p["w_biofilm"] * vpsr * (1 - bio) - p["d_bio"] * bio
    d_mot = p["w_motility"] * (1 - vpsr) * (1 - mot) - p["d_mot"] * mot
    d_rug = p["w_rugose"] * vpsr * vpsr * (1 - rug) - p["d_rug"] * rug
    return [d_cell, d_cdg, d_vpsr, d_bio, d_mot, d_rug]

def run_ss(y0, params, t_end=48.0, dt=0.001):
    t = np.arange(0, t_end, dt)
    sol = odeint(cap_rhs, y0, t, args=(params,))
    tail = int(len(t) * 0.1)
    return sol[-tail:].mean(axis=0)

def main():
    y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0]
    results = {}
    names = ["N", "CdG", "VpsR", "B", "M", "R"]

    for scenario, mods in [("normal", {}), ("stress", {"stress_factor": 3.0}),
                            ("low_cdg", {"k_cdg_prod": 0.3}),
                            ("vpsr_ko", {"k_vpsr_charge": 0.0})]:
        p = dict(PARAMS); p.update(mods)
        ic = list(y0)
        if scenario == "low_cdg":
            ic[1] = 0.1
        ss = run_ss(ic, p)
        results[scenario] = {n: float(v) for n, v in zip(names, ss)}
        vals = " ".join(f"{n}={v:.4f}" for n, v in zip(names, ss))
        print(f"{scenario:10s}: {vals}")

    out_dir = os.path.join(os.path.dirname(__file__),
                           "..", "experiments", "results", "027_capacitor")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mhatre2020_python_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written to {out_path}")

if __name__ == "__main__":
    main()
