#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-03-10
# Commit: wetSpring V107
# Hardware: Any (CPU-only, pure Python + numpy + scipy)
# SHA-256: 43904432c151f80bdfff62a7a4c18b8c90b7edf6feac3438041ea33a91bafffb
#
# Track 6 Baseline — Anaerobic Biogas Kinetics & Community Diversity
#
# Computes reference values for:
# 1. Modified Gompertz biogas production model
# 2. First-order kinetics model
# 3. Monod growth kinetics
# 4. Haldane substrate inhibition
# 5. Diversity indices on anaerobic community compositions
# 6. Anderson disorder mapping for anaerobic vs aerobic comparison
#
# References:
# - Yang et al. 2016, Adv Microbiol 6:879-897
# - Chen et al. 2016, Biomass Bioenergy 85:84-93
# - Rojas-Sossa et al. 2017, Bioresour Technol 245:714-723
# - Rojas-Sossa et al. 2019, Biomass Bioenergy 127:105263
# - Zhong et al. 2016, Biotechnol Biofuels 9:253
#
# Reproduction:
#   python3 scripts/python_anaerobic_biogas_baseline.py
#
# Output:
#   experiments/results/track6_anaerobic/biogas_kinetics_baseline.json

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import braycurtis


def _script_sha256() -> str:
    """SHA-256 of this script file for provenance tracking."""
    script_path = Path(__file__).resolve()
    return hashlib.sha256(script_path.read_bytes()).hexdigest()[:16]


def _git_commit() -> str:
    """Current git HEAD short hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# §1 Modified Gompertz: H(t) = P * exp(-exp((Rm*e/P)*(lambda_-t) + 1))
# ─────────────────────────────────────────────────────────────────────────────
def gompertz(t: float, P: float, Rm: float, lambda_: float) -> float:
    """Modified Gompertz biogas production (mL/gVS)."""
    inner = (Rm * math.e / P) * (lambda_ - t) + 1
    return P * math.exp(-math.exp(inner))


# ─────────────────────────────────────────────────────────────────────────────
# §2 First-order: B(t) = B_max * (1 - exp(-k*t))
# ─────────────────────────────────────────────────────────────────────────────
def first_order(t: float, B_max: float, k: float) -> float:
    """First-order biogas production (mL/gVS)."""
    return B_max * (1 - math.exp(-k * t))


# ─────────────────────────────────────────────────────────────────────────────
# §3 Monod: mu = mu_max * S / (Ks + S)
# ─────────────────────────────────────────────────────────────────────────────
def monod(S: float, mu_max: float, Ks: float) -> float:
    """Monod growth rate (day^-1)."""
    if S <= 0:
        return 0.0
    return mu_max * S / (Ks + S)


# ─────────────────────────────────────────────────────────────────────────────
# §4 Haldane: mu = mu_max * S / (Ks + S + S²/Ki)
# ─────────────────────────────────────────────────────────────────────────────
def haldane(S: float, mu_max: float, Ks: float, Ki: float) -> float:
    """Haldane substrate inhibition growth rate (day^-1)."""
    if S <= 0:
        return 0.0
    return mu_max * S / (Ks + S + S**2 / Ki)


# ─────────────────────────────────────────────────────────────────────────────
# §5 Diversity indices
# ─────────────────────────────────────────────────────────────────────────────
def shannon(counts: list[float]) -> float:
    """Shannon diversity H' = -sum(p*ln(p))."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    return -np.sum(p * np.log(p))


def simpson(counts: list[float]) -> float:
    """Simpson diversity D = 1 - sum(p²)."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    return 1.0 - np.sum(p**2)


def pielou(counts: list[float]) -> float:
    """Pielou evenness J = H' / ln(S)."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    S = len(arr)
    if S <= 1:
        return 1.0
    H = shannon(counts)
    return H / math.log(S)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_points = [0, 5, 10, 15, 20, 30, 40, 50]
    S_points = [0, 50, 100, 200, 500, 1000, 5000]

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": __import__("scipy").__version__,
            "command": " ".join(sys.argv) or "python3 scripts/python_anaerobic_biogas_baseline.py",
            "script_sha256": _script_sha256(),
            "git_commit": _git_commit(),
        },
        "gompertz": {},
        "first_order": {},
        "monod": {},
        "haldane": {},
        "diversity": {},
        "anderson_w": {},
    }

    # ── §1 Modified Gompertz ─────────────────────────────────────────────────
    print("═══════════════════════════════════════════════════════════════")
    print("  Track 6 Baseline — Anaerobic Biogas Kinetics")
    print("═══════════════════════════════════════════════════════════════\n")
    print("§1 Modified Gompertz: H(t) = P * exp(-exp((Rm*e/P)*(λ-t) + 1))")
    print("   Test case 1 (Yang 2016 manure): P=350, Rm=25, λ=3")
    print("   Test case 2 (corn stover): P=280, Rm=18, λ=5")
    print("   Time points: [0, 5, 10, 15, 20, 30, 40, 50] days\n")

    g1 = {"P": 350, "Rm": 25, "lambda": 3, "t": t_points, "H": []}
    g2 = {"P": 280, "Rm": 18, "lambda": 5, "t": t_points, "H": []}
    for t in t_points:
        H1 = gompertz(t, 350, 25, 3)
        H2 = gompertz(t, 280, 18, 5)
        g1["H"].append(H1)
        g2["H"].append(H2)
    results["gompertz"]["case1_yang_manure"] = g1
    results["gompertz"]["case2_corn_stover"] = g2

    H0_case1 = gompertz(0, 350, 25, 3)
    H0_expected = 350 * math.exp(-math.exp(25 * math.e * 3 / 350 + 1))
    H_inf_case1 = gompertz(1000, 350, 25, 3)  # t→∞ proxy
    print(f"   Case 1 (t=0): H(0)={H0_case1:.10f}  (expected ≈ P*exp(-exp(Rm*e*λ/P+1)))")
    print(f"   Case 1 (t=0) expected: {H0_expected:.10f}")
    print(f"   Case 1 (t→∞): H(1000)={H_inf_case1:.10f}  (→ P=350)")
    print(f"   Case 1 H(t): {[round(h, 6) for h in g1['H']]}")
    print(f"   Case 2 H(t): {[round(h, 6) for h in g2['H']]}\n")

    # ── §2 First-order kinetics ──────────────────────────────────────────────
    print("§2 First-order: B(t) = B_max * (1 - exp(-k*t))")
    print("   k=0.08 day^-1, B_max=320 mL/gVS\n")
    B_max, k = 320, 0.08
    fo = {"B_max": B_max, "k": k, "t": t_points, "B": []}
    for t in t_points:
        fo["B"].append(first_order(t, B_max, k))
    results["first_order"] = fo
    B0 = first_order(0, B_max, k)
    B_inf = first_order(1000, B_max, k)
    print(f"   B(0)={B0:.10f}  (must be 0)")
    print(f"   B(∞)={B_inf:.10f}  (→ B_max={B_max})")
    print(f"   B(t): {[round(b, 6) for b in fo['B']]}\n")

    # ── §3 Monod kinetics ────────────────────────────────────────────────────
    print("§3 Monod: μ = μ_max * S / (Ks + S)")
    print("   μ_max=0.4 day^-1, Ks=200 mg/L")
    print("   S: [0, 50, 100, 200, 500, 1000, 5000] mg/L\n")
    mu_max, Ks = 0.4, 200
    mon = {"mu_max": mu_max, "Ks": Ks, "S": S_points, "mu": []}
    for S in S_points:
        mon["mu"].append(monod(S, mu_max, Ks))
    results["monod"] = mon
    mu_0 = monod(0, mu_max, Ks)
    mu_Ks = monod(Ks, mu_max, Ks)
    mu_inf = monod(100000, mu_max, Ks)
    print(f"   μ(0)={mu_0:.10f}  (must be 0)")
    print(f"   μ(Ks)={mu_Ks:.10f}  (μ_max/2={mu_max/2})")
    print(f"   μ(∞)≈{mu_inf:.10f}  (→ μ_max={mu_max})")
    print(f"   μ(S): {[round(m, 6) for m in mon['mu']]}\n")

    # ── §4 Haldane inhibition ───────────────────────────────────────────────
    print("§4 Haldane: μ = μ_max * S / (Ks + S + S²/Ki)")
    print("   μ_max=0.4, Ks=200, Ki=3000 mg/L")
    Ki_val = 3000
    S_opt = math.sqrt(Ks * Ki_val)
    # Validate S_opt via scipy.optimize
    res = minimize_scalar(
        lambda S: -haldane(S, mu_max, Ks, Ki_val) if S > 0 else 0,
        bounds=(1, 10000),
        method="bounded",
    )
    S_opt_numerical = res.x
    print(f"   S_opt = sqrt(Ks*Ki) = {S_opt:.4f} mg/L")
    print(f"   S_opt (numerical) = {S_opt_numerical:.4f} mg/L\n")
    hal = {
        "mu_max": mu_max,
        "Ks": Ks,
        "Ki": Ki_val,
        "S_opt": S_opt,
        "S_opt_numerical": S_opt_numerical,
        "S": S_points,
        "mu": [],
    }
    for S in S_points:
        hal["mu"].append(haldane(S, mu_max, Ks, Ki_val))
    results["haldane"] = hal
    mu_opt = haldane(S_opt, mu_max, Ks, 3000)
    print(f"   μ(S_opt)={mu_opt:.10f}  (peak rate)")
    print(f"   μ(S): {[round(m, 6) for m in hal['mu']]}\n")

    # ── §5 Diversity comparison ─────────────────────────────────────────────
    print("§5 Diversity (aerobic vs anaerobic)")
    aerobic = [35, 22, 16, 12, 8, 5, 3, 2, 1, 0.5]
    anaerobic = [45, 25, 15, 8, 3, 2, 1, 0.5, 0.3, 0.2]
    H_aer = shannon(aerobic)
    H_ana = shannon(anaerobic)
    D_aer = simpson(aerobic)
    D_ana = simpson(anaerobic)
    J_aer = pielou(aerobic)
    J_ana = pielou(anaerobic)
    bc = float(braycurtis(aerobic, anaerobic))
    results["diversity"] = {
        "aerobic": {
            "counts": aerobic,
            "shannon": H_aer,
            "simpson": D_aer,
            "pielou": J_aer,
        },
        "anaerobic": {
            "counts": anaerobic,
            "shannon": H_ana,
            "simpson": D_ana,
            "pielou": J_ana,
        },
        "bray_curtis": bc,
    }
    print(f"   Aerobic:   H'={H_aer:.6f}, D={D_aer:.6f}, J={J_aer:.6f}")
    print(f"   Anaerobic: H'={H_ana:.6f}, D={D_ana:.6f}, J={J_ana:.6f}")
    print(f"   Bray-Curtis(aerobic, anaerobic)={bc:.6f}\n")

    # ── §6 Anderson W mapping ───────────────────────────────────────────────
    print("§6 Anderson W mapping: W = W_max * (1 - evenness)")
    W_max = 20
    W_aer = W_max * (1 - J_aer)
    W_ana = W_max * (1 - J_ana)
    results["anderson_w"] = {
        "W_max": W_max,
        "W_aerobic": float(W_aer),
        "W_anaerobic": float(W_ana),
        "W_anaerobic_gt_W_aerobic": bool(W_ana > W_aer),
    }
    print(f"   W_aerobic={W_aer:.6f}, W_anaerobic={W_ana:.6f}")
    print(f"   W_anaerobic > W_aerobic: {W_ana > W_aer}\n")

    # ── Write JSON ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent.parent / "experiments" / "results" / "track6_anaerobic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "biogas_kinetics_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Output: {out_path}")
    print(f"  numpy {results['metadata']['numpy_version']}, scipy {results['metadata']['scipy_version']}")
    print("All baselines computed successfully.")


if __name__ == "__main__":
    main()
