#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-18
"""Python baseline benchmark for the same workloads as benchmark_cpu_gpu.

Measures wall-clock time for numpy/scipy implementations of the same
scientific operations. Emits JSON matching the PhaseResult schema used
by the Rust harness so that results can be merged into a unified
three-tier comparison (Python → Rust CPU → Rust GPU).

Requires: pip install numpy scipy
"""
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import braycurtis

WARMUP = 3
MIN_TIME_S = 0.1


# ═══════════════════════════════════════════════════════════════════
#  Hardware inventory (matches Rust HardwareInventory)
# ═══════════════════════════════════════════════════════════════════

def detect_hardware(gate_name="wetSpring Python Baseline"):
    hw = {
        "gate_name": gate_name,
        "cpu_model": "unknown",
        "cpu_cores": os.cpu_count() or 0,
        "cpu_threads": os.cpu_count() or 0,
        "cpu_cache_kb": 0,
        "ram_total_mb": 0,
        "gpu_name": "N/A",
        "gpu_vram_mb": 0,
        "gpu_driver": "N/A",
        "gpu_compute_cap": "N/A",
        "os_kernel": platform.release(),
        "rust_version": "",
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu_model"] = line.split(":")[1].strip()
                elif line.startswith("cache size"):
                    hw["cpu_cache_kb"] = int(line.split(":")[1].strip().replace(" KB", ""))
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    hw["ram_total_mb"] = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        parts = out.split(", ")
        if len(parts) >= 4:
            hw["gpu_name"] = parts[0].strip()
            hw["gpu_vram_mb"] = int(parts[1].strip())
            hw["gpu_driver"] = parts[2].strip()
            hw["gpu_compute_cap"] = parts[3].strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return hw


# ═══════════════════════════════════════════════════════════════════
#  RAPL energy reading
# ═══════════════════════════════════════════════════════════════════

RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"


def read_rapl_uj():
    try:
        with open(RAPL_PATH) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_rapl_max_uj():
    try:
        with open(RAPL_MAX_PATH) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def rapl_delta_joules(start_uj, end_uj):
    if start_uj is None or end_uj is None:
        return 0.0
    if end_uj >= start_uj:
        return (end_uj - start_uj) / 1e6
    max_uj = read_rapl_max_uj() or (1 << 63)
    return (max_uj - start_uj + end_uj) / 1e6


# ═══════════════════════════════════════════════════════════════════
#  Benchmark helper
# ═══════════════════════════════════════════════════════════════════

def bench(fn, *args):
    """Benchmark a function, returning (microseconds/call, energy report dict)."""
    for _ in range(WARMUP):
        fn(*args)

    iters = 5
    while True:
        rapl_start = read_rapl_uj()
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        elapsed = time.perf_counter() - start
        rapl_end = read_rapl_uj()
        if elapsed > MIN_TIME_S or iters >= 1000:
            us_per_call = elapsed * 1e6 / iters
            energy = {
                "cpu_joules": rapl_delta_joules(rapl_start, rapl_end),
                "gpu_joules": 0.0,
                "gpu_watts_avg": 0.0,
                "gpu_watts_peak": 0.0,
                "gpu_temp_peak_c": 0.0,
                "gpu_vram_peak_mib": 0.0,
                "gpu_samples": 0,
            }
            return us_per_call, energy
        iters = min(iters * 3, 1000)


def fmt(us):
    if us < 1:
        return f"{us*1000:.1f}ns"
    if us < 1000:
        return f"{us:.1f}µs"
    return f"{us/1000:.2f}ms"


def peak_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def report(label, n, us):
    print(f"│ {label:<26} {n:>8} {fmt(us):>11}│")


def gen_counts(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1001, size=n).astype(np.float64)


def gen_f64(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random(n)


# ═══════════════════════════════════════════════════════════════════
#  Workload implementations
# ═══════════════════════════════════════════════════════════════════

def shannon(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log(p))


def simpson(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p ** 2)


def variance(data):
    return np.var(data)


def dot(a, b):
    return np.dot(a, b)


def bray_curtis_matrix(samples):
    n = len(samples)
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(braycurtis(samples[i], samples[j]))
    return result


def pairwise_cosine(spectra):
    n = len(spectra)
    norms = [np.linalg.norm(s) for s in spectra]
    result = []
    for i in range(n):
        for j in range(i+1, n):
            cos = np.dot(spectra[i], spectra[j]) / (norms[i] * norms[j])
            result.append(cos)
    return result


def pcoa_simple(dist_condensed, n, k=3):
    D = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = dist_condensed[idx]
            D[j, i] = dist_condensed[idx]
            idx += 1
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:k]] * np.sqrt(np.abs(eigvals[order[:k]]))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    hw = detect_hardware()
    phases = []

    print("╔════════════════════════════════════════════════════╗")
    print("║  Python Baseline Benchmark (numpy/scipy + RAPL)   ║")
    print(f"║  CPU: {hw['cpu_model'][:44]:<44} ║")
    print("╚════════════════════════════════════════════════════╝")

    def record(label, n, us_val, energy):
        report(label, n, us_val)
        phases.append({
            "phase": f"{label} N={n}",
            "substrate": "Python (numpy/scipy)",
            "wall_time_s": us_val / 1e6,
            "per_eval_us": us_val,
            "n_evals": max(1, int(1e6 / max(us_val, 1))),
            "energy": energy,
            "peak_rss_mb": peak_rss_mb(),
            "notes": "",
        })

    # ── Single-vector ──────────────────────────────────
    print("\n┌──────────────────────────────────────────────────┐")
    print("│ SINGLE-VECTOR REDUCTIONS                          │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│ {'Workload':<26} {'N':>8} {'Python':>11}│")
    print("├──────────────────────────────────────────────────┤")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_counts(n)
        t, e = bench(shannon, data)
        record("Shannon entropy", n, t, e)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_counts(n)
        t, e = bench(simpson, data)
        record("Simpson diversity", n, t, e)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_f64(n)
        t, e = bench(variance, data)
        record("Variance", n, t, e)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        a, b = gen_f64(n, 11), gen_f64(n, 22)
        t, e = bench(dot, a, b)
        record("Dot product", n, t, e)

    # ── Pairwise N×N ──────────────────────────────────
    print("\n┌──────────────────────────────────────────────────┐")
    print("│ PAIRWISE N×N WORKLOADS                            │")
    print("├──────────────────────────────────────────────────┤")

    for ns in [10, 20, 50, 100]:
        samples = [gen_counts(500, seed=42+i) for i in range(ns)]
        t, e = bench(bray_curtis_matrix, samples)
        record(f"Bray-Curtis {ns}x{ns}", ns*(ns-1)//2, t, e)

    for ns in [10, 50, 100, 200]:
        spectra = [gen_f64(500, seed=300+i) for i in range(ns)]
        t, e = bench(pairwise_cosine, spectra)
        record(f"Cosine {ns}x{ns}", ns*(ns-1)//2, t, e)

    # ── Matrix algebra ────────────────────────────────
    print("\n┌──────────────────────────────────────────────────┐")
    print("│ MATRIX ALGEBRA                                    │")
    print("├──────────────────────────────────────────────────┤")

    for ns in [10, 20, 30]:
        samples = [gen_counts(200, seed=100+i) for i in range(ns)]
        dist = bray_curtis_matrix(samples)
        t, e = bench(pcoa_simple, dist, ns, 3)
        record(f"PCoA {ns}x{ns}", ns, t, e)

    print("\n══════════════════════════════════════════════════")

    # ── Emit JSON ─────────────────────────────────────
    result = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": hw,
        "phases": phases,
    }

    out_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = result["timestamp"].replace(":", "-")
    out_path = out_dir / f"python_baseline_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON results saved to {out_path}")

    # Also write to a stable path for the runner script to find
    latest = out_dir / "python_baseline_latest.json"
    with open(latest, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
