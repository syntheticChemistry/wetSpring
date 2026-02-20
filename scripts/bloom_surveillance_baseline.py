#!/usr/bin/env python3
"""Exp040 baseline — Bloom event detection and surveillance pipeline.

Simulates cyanobacterial bloom surveillance pattern for PRJNA1224988
(175-sample multi-year bloom time series). Uses synthetic community data
to establish baselines for:
  - Bloom detection via dominance index spike
  - Community evenness collapse during bloom
  - Recovery trajectory post-bloom

This serves as proxy for Smallwood (#14) raceway metagenomic surveillance.

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


def simpson(counts: list[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts)


def pielou_evenness(counts: list[float]) -> float:
    h = shannon(counts)
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    return h / math.log(s)


def dominance_index(counts: list[float]) -> float:
    """Berger-Parker dominance: fraction of most abundant taxon."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return max(counts) / total


def bray_curtis(a: list[float], b: list[float]) -> float:
    num = sum(abs(x - y) for x, y in zip(a, b))
    den = sum(x + y for x, y in zip(a, b))
    return num / den if den > 0 else 0.0


def simulate_bloom(n_timepoints: int, n_taxa: int, bloom_start: int,
                   bloom_peak: int, bloom_end: int, seed: int) -> list[list[float]]:
    """Simulate community with a bloom event (taxon 0 dominates)."""
    rng = random.Random(seed)
    base = [rng.uniform(20, 80) for _ in range(n_taxa)]
    series = []
    for t in range(n_timepoints):
        sample = list(base)
        for i in range(n_taxa):
            sample[i] += rng.gauss(0, 3)
            sample[i] = max(0.1, sample[i])

        if bloom_start <= t <= bloom_end:
            progress = (t - bloom_start) / max(1, bloom_peak - bloom_start)
            if t <= bloom_peak:
                bloom_factor = 1 + 15 * min(1.0, progress)
            else:
                recovery = (t - bloom_peak) / max(1, bloom_end - bloom_peak)
                bloom_factor = 1 + 15 * (1 - recovery)
            sample[0] *= bloom_factor
            for i in range(1, n_taxa):
                sample[i] *= max(0.1, 1 - 0.5 * (bloom_factor - 1) / 15)

        series.append(sample)
    return series


def main():
    n_timepoints = 90
    n_taxa = 25
    bloom_start, bloom_peak, bloom_end = 30, 45, 60

    series = simulate_bloom(n_timepoints, n_taxa, bloom_start, bloom_peak,
                            bloom_end, seed=777)

    shannon_ts = [shannon(s) for s in series]
    simpson_ts = [simpson(s) for s in series]
    evenness_ts = [pielou_evenness(s) for s in series]
    dominance_ts = [dominance_index(s) for s in series]
    bc_consecutive = [bray_curtis(series[i], series[i + 1])
                      for i in range(len(series) - 1)]

    # Pre-bloom baseline (first 25 timepoints)
    pre_shannon = shannon_ts[:bloom_start]
    pre_mean = sum(pre_shannon) / len(pre_shannon)
    pre_std = (sum((s - pre_mean) ** 2 for s in pre_shannon) / len(pre_shannon)) ** 0.5

    # Bloom peak metrics
    peak_shannon = shannon_ts[bloom_peak]
    peak_dominance = dominance_ts[bloom_peak]
    peak_evenness = evenness_ts[bloom_peak]
    peak_bc = bc_consecutive[bloom_peak - 1] if bloom_peak > 0 else 0

    # Post-bloom recovery
    post_shannon = shannon_ts[bloom_end:bloom_end + 10] if bloom_end + 10 <= n_timepoints else []
    recovery_mean = sum(post_shannon) / len(post_shannon) if post_shannon else 0

    # Bloom detection: Shannon drops below 2 std from pre-bloom mean
    bloom_detected = peak_shannon < pre_mean - 2 * pre_std

    output = {
        "experiment": "Exp040",
        "description": "Bloom event detection and surveillance pipeline",
        "proxy_for": "Smallwood #14 (raceway metagenomic surveillance)",
        "data_source": "PRJNA1224988 (175 samples, cyanobacterial bloom)",
        "n_timepoints": n_timepoints,
        "n_taxa": n_taxa,
        "bloom_window": [bloom_start, bloom_peak, bloom_end],
        "pre_bloom": {
            "shannon_mean": pre_mean,
            "shannon_std": pre_std,
            "evenness_mean": sum(evenness_ts[:bloom_start]) / bloom_start,
        },
        "bloom_peak": {
            "shannon": peak_shannon,
            "simpson": simpson_ts[bloom_peak],
            "dominance": peak_dominance,
            "evenness": peak_evenness,
            "bray_curtis_shift": peak_bc,
        },
        "recovery": {
            "shannon_mean": recovery_mean,
            "recovery_fraction": recovery_mean / pre_mean if pre_mean > 0 else 0,
        },
        "bloom_detected": bloom_detected,
        "shannon_first5": shannon_ts[:5],
        "dominance_first5": dominance_ts[:5],
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "results", "040_bloom_surveillance"
    )
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "python_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("Exp040 baseline — Bloom surveillance:")
    print(f"  Pre-bloom Shannon: {pre_mean:.4f} ± {pre_std:.4f}")
    print(f"  Peak bloom Shannon: {peak_shannon:.4f}, dominance: {peak_dominance:.4f}")
    print(f"  Recovery Shannon: {recovery_mean:.4f}")
    print(f"  Bloom detected: {bloom_detected}")
    print(f"Output: {out_path}/python_baseline.json")


if __name__ == "__main__":
    main()
