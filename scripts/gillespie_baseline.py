#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-19
"""Gillespie SSA baseline — Python/numpy reference.

Implements a simple birth-death process for c-di-GMP signal modeling
(Massie et al. 2012 simplified). Produces ensemble statistics for Rust validation.

Requires: pip install numpy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent

# Simplified c-di-GMP birth-death model (Massie 2012 reduced)
# Reaction 1: ∅ → cdGMP (synthesis by DGC, rate k_dgc)
# Reaction 2: cdGMP → ∅ (degradation by PDE, rate k_pde * cdGMP)

K_DGC = 10.0    # DGC synthesis rate (molecules/time)
K_PDE = 0.1     # PDE degradation rate (per molecule per time)
T_MAX = 100.0   # simulation duration
N_RUNS = 1000   # ensemble size


def gillespie_birth_death(k_dgc, k_pde, t_max, seed):
    """Run one Gillespie SSA trajectory for birth-death process."""
    rng = np.random.RandomState(seed)
    t = 0.0
    cdgmp = 0  # start with zero molecules

    times = [t]
    counts = [cdgmp]

    while t < t_max:
        # Propensities
        a1 = k_dgc              # synthesis
        a2 = k_pde * cdgmp      # degradation
        a0 = a1 + a2

        if a0 == 0:
            break

        # Time to next reaction (exponential)
        tau = -np.log(rng.random()) / a0
        t += tau
        if t > t_max:
            break

        # Choose reaction
        r = rng.random() * a0
        if r < a1:
            cdgmp += 1  # synthesis
        else:
            cdgmp -= 1  # degradation

        cdgmp = max(cdgmp, 0)  # safety floor
        times.append(t)
        counts.append(cdgmp)

    return times, counts


def main():
    print("=" * 70)
    print("  Exp022: Gillespie SSA — Python Baseline")
    print("  Massie 2012 c-di-GMP Birth-Death Model")
    print("=" * 70)

    t0 = time.time()

    # Analytical steady state for birth-death: mean = k_dgc / k_pde
    analytical_mean = K_DGC / K_PDE
    print(f"\n  Analytical steady state: {analytical_mean:.1f} molecules")

    # Run ensemble
    print(f"  Running {N_RUNS} trajectories...")
    final_counts = []
    for i in range(N_RUNS):
        _, counts = gillespie_birth_death(K_DGC, K_PDE, T_MAX, seed=42 + i)
        final_counts.append(counts[-1])

    final_counts = np.array(final_counts)
    ensemble_mean = float(np.mean(final_counts))
    ensemble_std = float(np.std(final_counts))
    ensemble_var = float(np.var(final_counts))

    print(f"  Ensemble mean:  {ensemble_mean:.2f} (analytical: {analytical_mean:.1f})")
    print(f"  Ensemble std:   {ensemble_std:.2f}")
    print(f"  Ensemble var:   {ensemble_var:.2f}")
    print(f"  CV² = var/mean: {ensemble_var/ensemble_mean:.3f} "
          f"(Poisson: ~{analytical_mean:.1f})")

    # Determinism check: same seed → same result
    _, c1 = gillespie_birth_death(K_DGC, K_PDE, T_MAX, seed=42)
    _, c2 = gillespie_birth_death(K_DGC, K_PDE, T_MAX, seed=42)
    deterministic = (c1 == c2)

    # Single trajectory for detailed validation
    times_single, counts_single = gillespie_birth_death(K_DGC, K_PDE, T_MAX, seed=42)

    checks = []

    def check(label, condition):
        status = "PASS" if condition else "FAIL"
        checks.append({"label": label, "status": status})
        print(f"  [{status}] {label}")
        return condition

    check("Ensemble mean within 10% of analytical",
          abs(ensemble_mean - analytical_mean) < 0.1 * analytical_mean)
    check("Ensemble mean within 20% of analytical (conservative)",
          abs(ensemble_mean - analytical_mean) < 0.2 * analytical_mean)
    check("All final counts non-negative",
          all(c >= 0 for c in final_counts))
    check("Ensemble std > 0 (stochastic variability)",
          ensemble_std > 0)
    check(f"Deterministic with same seed: {deterministic}",
          deterministic)
    check(f"Single trajectory final count >= 0: {counts_single[-1]}",
          counts_single[-1] >= 0)
    check(f"Single trajectory has events: {len(counts_single)} steps",
          len(counts_single) > 10)

    # Poisson-like: for birth-death, var ≈ mean at steady state → CV² ≈ 1.0
    cv_sq = ensemble_var / max(ensemble_mean, 1e-10)
    check(f"Var/mean ~ 1.0 (Poisson): {cv_sq:.3f}",
          0.5 < cv_sq < 2.0)

    elapsed = time.time() - t0

    n_pass = sum(1 for c in checks if c["status"] == "PASS")

    out_dir = WORKSPACE / "experiments/results/022_gillespie"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gillespie_python_baseline.json"

    output = {
        "experiment": "022_massie2012_gillespie",
        "model": "birth-death (c-di-GMP simplified)",
        "parameters": {
            "k_dgc": K_DGC,
            "k_pde": K_PDE,
            "t_max": T_MAX,
            "n_runs": N_RUNS,
        },
        "analytical_steady_state": analytical_mean,
        "ensemble": {
            "mean": round(ensemble_mean, 6),
            "std": round(ensemble_std, 6),
            "variance": round(ensemble_var, 6),
            "cv_squared": round(cv_sq, 6),
        },
        "single_trajectory_seed42": {
            "n_events": len(counts_single),
            "final_count": int(counts_single[-1]),
            "final_time": round(times_single[-1], 6),
        },
        "checks": checks,
        "total_pass": n_pass,
        "total_checks": len(checks),
        "elapsed_seconds": round(elapsed, 4),
        "python_version": sys.version.split()[0],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {n_pass}/{len(checks)} PASS")
    print(f"  Results: {out_path}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print(f"{'=' * 70}")

    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
