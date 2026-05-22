#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-05-22
"""
MATRIX Pharmacophenomics — Python Control (Exp158)

Reproduces the MATRIX framework for systematic drug repurposing
as described in Fajgenbaum et al. Lancet Haematology 2025 (Paper 40).

MATRIX = Mechanism-centric Analysis of Therapeutic Repurposing
Integrating X-omic data. Uses NMF decomposition on synthetic
drug-disease-pathway data to identify repurposing candidates.

Uses a pure-numpy NMF implementation (multiplicative update rules)
to match the Rust in-tree NMF rather than depend on sklearn.

This is the Python baseline. Rust: wetspring validate --scenario matrix_pharmacophenomics

Date: 2026-05-22
Paper: 40 (Fajgenbaum et al. Lancet Haematology 2025)

Reproduction:
    python3 scripts/matrix_pharmacophenomics_baseline.py
"""

import time
import numpy as np

EPS = 1e-12


def nmf_euclidean(V, k, max_iter=200, seed=42):
    """Multiplicative-update NMF (Euclidean objective)."""
    rng = np.random.default_rng(seed)
    m, n = V.shape
    W = rng.uniform(0.01, 1.0, (m, k))
    H = rng.uniform(0.01, 1.0, (k, n))
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + EPS)
        W *= (V @ H.T) / (W @ H @ H.T + EPS)
    return W, H


def nmf_kl(V, k, max_iter=200, seed=42):
    """Multiplicative-update NMF (KL-divergence objective)."""
    rng = np.random.default_rng(seed)
    m, n = V.shape
    W = rng.uniform(0.01, 1.0, (m, k))
    H = rng.uniform(0.01, 1.0, (k, n))
    for _ in range(max_iter):
        WH = W @ H + EPS
        H *= (W.T @ (V / WH)) / (np.sum(W, axis=0, keepdims=True).T + EPS)
        WH = W @ H + EPS
        W *= ((V / WH) @ H.T) / (np.sum(H, axis=1, keepdims=True).T + EPS)
    return W, H


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def top_k_cosine(W, H, top_k):
    """Find top-k (drug, disease) pairs by cosine similarity in latent space."""
    m, k = W.shape
    n = H.shape[1]
    pairs = []
    for i in range(m):
        for j in range(n):
            sim = cosine_sim(W[i, :], H[:, j])
            pairs.append((i, j, sim))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


def run():
    checks_passed = 0
    checks_total = 0
    timings = []

    print("=" * 60)
    print("MATRIX Pharmacophenomics — Python Control (Exp158)")
    print("=" * 60)

    # §1 Synthetic drug-disease matrix
    t0 = time.perf_counter_ns()
    rng = np.random.default_rng(42)
    n_drugs = 15
    n_diseases = 10

    W_true = rng.uniform(0, 1, (n_drugs, 3))
    H_true = rng.uniform(0, 1, (3, n_diseases))
    noise = rng.uniform(0, 0.05, (n_drugs, n_diseases))
    data = W_true @ H_true + noise
    data = np.clip(data, 0, None)

    checks_total += 2
    if data.shape == (n_drugs, n_diseases):
        checks_passed += 1
        print(f"  ✓ Data matrix: {data.shape}")
    else:
        print(f"  ✗ Shape mismatch: {data.shape}")

    if np.all(data >= 0):
        checks_passed += 1
        print("  ✓ All values non-negative")
    else:
        print("  ✗ Negative values found")

    d01_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Data generation", d01_us, 2))

    # §2 NMF decomposition (Euclidean)
    t0 = time.perf_counter_ns()
    k = 3
    W, H = nmf_euclidean(data, k, max_iter=200, seed=42)

    checks_total += 2
    if np.all(W >= 0):
        checks_passed += 1
        print(f"\n  ✓ W non-negative: shape {W.shape}")
    else:
        print("  ✗ W has negative values")

    if np.all(H >= 0):
        checks_passed += 1
        print(f"  ✓ H non-negative: shape {H.shape}")
    else:
        print("  ✗ H has negative values")

    reconstruction = W @ H
    recon_error = np.mean((data - reconstruction) ** 2)
    checks_total += 1
    if recon_error < 0.1:
        checks_passed += 1
        print(f"  ✓ Reconstruction MSE = {recon_error:.6f} < 0.1")
    else:
        print(f"  ✗ Reconstruction MSE = {recon_error:.6f}")

    d02_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("NMF decomposition", d02_us, 3))

    # §3 Cosine similarity scoring
    t0 = time.perf_counter_ns()
    top_pairs = top_k_cosine(W, H, 10)

    checks_total += 1
    if len(top_pairs) == 10:
        checks_passed += 1
        print(f"\n  ✓ Top 10 drug-disease pairs computed")
    else:
        print(f"  ✗ Expected 10 pairs, got {len(top_pairs)}")

    checks_total += 1
    if all(0.0 <= p[2] <= 1.0 for p in top_pairs):
        checks_passed += 1
        print("  ✓ All cosine similarities in [0, 1]")
    else:
        print("  ✗ Cosine similarity out of range")

    checks_total += 1
    if top_pairs[0][2] >= top_pairs[-1][2]:
        checks_passed += 1
        print(f"  ✓ Sorted: top={top_pairs[0][2]:.4f} ≥ bottom={top_pairs[-1][2]:.4f}")
    else:
        print("  ✗ Not properly sorted")

    print(f"\n  Top 5 repurposing candidates:")
    print(f"  {'Drug':>6} {'Disease':>8} {'Cosine':>8}")
    for drug_i, disease_j, sim in top_pairs[:5]:
        print(f"  {drug_i:>6} {disease_j:>8} {sim:>8.4f}")

    d03_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Cosine scoring", d03_us, 3))

    # §4 KL-divergence NMF
    t0 = time.perf_counter_ns()
    W_kl, H_kl = nmf_kl(data, k, max_iter=200, seed=42)

    checks_total += 2
    if np.all(W_kl >= 0) and np.all(H_kl >= 0):
        checks_passed += 2
        print(f"\n  ✓ KL-NMF: W, H non-negative")
    else:
        print("  ✗ KL-NMF produced negative values")

    top_pairs_kl = top_k_cosine(W_kl, H_kl, 10)
    overlap = len(
        set((p[0], p[1]) for p in top_pairs[:5])
        & set((p[0], p[1]) for p in top_pairs_kl[:5])
    )
    print(f"  Overlap in top-5 (Euclidean vs KL): {overlap}/5")

    d04_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("KL-NMF", d04_us, 2))

    # Summary
    total_us = sum(t[1] for t in timings)
    print(f"\n  Timing:")
    for domain, us, n in timings:
        print(f"    {domain:<20}: {us:>10.0f} µs ({n} checks)")
    print(f"    {'Total':<20}: {total_us:>10.0f} µs")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    return checks_passed == checks_total


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
