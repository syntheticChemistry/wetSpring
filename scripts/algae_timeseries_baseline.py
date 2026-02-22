#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-20
"""Exp039 baseline — Algal pond time-series diversity analysis.

Simulates the longitudinal analysis pattern for PRJNA382322 (128-sample
Nannochloropsis raceway pond, 4-month time series). Uses synthetic
diversity data matching published ranges to establish baselines for:
  - Shannon diversity tracking over time
  - Bray-Curtis beta diversity between consecutive timepoints
  - Anomaly detection via Z-score on rolling window

This serves as proxy for Cahill (#13) phage biocontrol monitoring.

Requires: Python 3.8+ (stdlib only)
"""
import json
import math
import os
import random
import sys


def shannon(counts: list[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def bray_curtis(a: list[float], b: list[float]) -> float:
    num = sum(abs(x - y) for x, y in zip(a, b))
    den = sum(x + y for x, y in zip(a, b))
    return num / den if den > 0 else 0.0


def simulate_timeseries(n_timepoints: int, n_taxa: int, seed: int,
                        crash_at: int = -1) -> list[list[float]]:
    """Simulate abundance time series with optional crash event."""
    rng = random.Random(seed)
    base = [rng.uniform(10, 100) for _ in range(n_taxa)]
    series = []
    for t in range(n_timepoints):
        sample = []
        for i in range(n_taxa):
            drift = rng.gauss(0, 5)
            seasonal = 10 * math.sin(2 * math.pi * t / 30)
            val = base[i] + drift + seasonal
            if crash_at >= 0 and t == crash_at:
                val *= 0.1 if i < n_taxa // 2 else 2.0
            sample.append(max(0.1, val))
        series.append(sample)
    return series


def rolling_zscore(values: list[float], window: int) -> list[float]:
    """Z-score anomaly detection on rolling window."""
    zscores = [0.0] * len(values)
    for i in range(window, len(values)):
        w = values[i - window:i]
        mu = sum(w) / len(w)
        std = (sum((x - mu) ** 2 for x in w) / len(w)) ** 0.5
        if std > 0:
            zscores[i] = (values[i] - mu) / std
    return zscores


def main():
    n_timepoints = 60
    n_taxa = 20
    crash_at = 40

    # Scenario 1: Normal operation
    series_normal = simulate_timeseries(n_timepoints, n_taxa, seed=42)
    shannon_normal = [shannon(s) for s in series_normal]
    bc_consecutive = [bray_curtis(series_normal[i], series_normal[i + 1])
                      for i in range(len(series_normal) - 1)]

    # Scenario 2: Crash event at timepoint 40
    series_crash = simulate_timeseries(n_timepoints, n_taxa, seed=42, crash_at=crash_at)
    shannon_crash = [shannon(s) for s in series_crash]
    bc_crash = [bray_curtis(series_crash[i], series_crash[i + 1])
                for i in range(len(series_crash) - 1)]

    # Anomaly detection
    zscores_normal = rolling_zscore(shannon_normal, window=10)
    zscores_crash = rolling_zscore(shannon_crash, window=10)

    # The crash should produce a Z-score spike
    max_z_normal = max(abs(z) for z in zscores_normal[10:])
    max_z_crash = max(abs(z) for z in zscores_crash[10:])
    crash_z = abs(zscores_crash[crash_at]) if crash_at < len(zscores_crash) else 0

    output = {
        "experiment": "Exp039",
        "description": "Algal pond time-series diversity surveillance",
        "proxy_for": "Cahill #13 (phage biocontrol monitoring)",
        "data_source": "PRJNA382322 (128 samples, Nannochloropsis raceway)",
        "n_timepoints": n_timepoints,
        "n_taxa": n_taxa,
        "crash_timepoint": crash_at,
        "normal": {
            "shannon_mean": sum(shannon_normal) / len(shannon_normal),
            "shannon_std": (sum((s - sum(shannon_normal) / len(shannon_normal)) ** 2
                               for s in shannon_normal) / len(shannon_normal)) ** 0.5,
            "bc_mean": sum(bc_consecutive) / len(bc_consecutive),
            "max_abs_zscore": max_z_normal,
            "shannon_first5": shannon_normal[:5],
            "bc_first5": bc_consecutive[:5],
        },
        "crash": {
            "shannon_mean": sum(shannon_crash) / len(shannon_crash),
            "crash_zscore": crash_z,
            "max_abs_zscore": max_z_crash,
            "shannon_at_crash": shannon_crash[crash_at],
            "bc_at_crash": bc_crash[crash_at - 1] if crash_at > 0 else 0,
            "shannon_first5": shannon_crash[:5],
        },
        "anomaly_detected": crash_z > 2.0,
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "results", "039_algae_timeseries"
    )
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "python_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("Exp039 baseline — Algal pond time-series:")
    print(f"  Normal: Shannon mean={output['normal']['shannon_mean']:.4f}, "
          f"BC mean={output['normal']['bc_mean']:.4f}")
    print(f"  Crash: Z-score at crash={crash_z:.2f}, "
          f"anomaly detected={output['anomaly_detected']}")
    print(f"Output: {out_path}/python_baseline.json")


if __name__ == "__main__":
    main()
