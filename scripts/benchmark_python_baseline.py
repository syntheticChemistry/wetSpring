#!/usr/bin/env python3
"""Python baseline benchmark for the same workloads as benchmark_cpu_gpu.

Measures wall-clock time for numpy/scipy implementations of the same
scientific operations. Results are compared against Rust CPU and Rust GPU
to demonstrate the full three-tier performance ladder:
  Python → Rust CPU → Rust GPU

Requires: pip install numpy scipy scikit-bio
"""
import time
import numpy as np
from scipy.spatial.distance import braycurtis
from scipy.stats import pearsonr

WARMUP = 3
MIN_TIME_S = 0.1


def bench(fn, *args):
    """Benchmark a function, returning microseconds per call."""
    for _ in range(WARMUP):
        fn(*args)

    iters = 5
    while True:
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        elapsed = time.perf_counter() - start
        if elapsed > MIN_TIME_S or iters >= 1000:
            return elapsed * 1e6 / iters
        iters = min(iters * 3, 1000)


def fmt(us):
    if us < 1:
        return f"{us*1000:.1f}ns"
    if us < 1000:
        return f"{us:.1f}µs"
    return f"{us/1000:.2f}ms"


def report(label, n, us):
    print(f"│ {label:<26} {n:>8} {fmt(us):>11}│")


def gen_counts(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1001, size=n).astype(np.float64)


def gen_f64(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random(n)


# ── Shannon ──────────────────────────────────────────────────────────

def shannon(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log(p))


# ── Simpson ──────────────────────────────────────────────────────────

def simpson(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p ** 2)


# ── Variance ─────────────────────────────────────────────────────────

def variance(data):
    return np.var(data)


# ── Dot product ──────────────────────────────────────────────────────

def dot(a, b):
    return np.dot(a, b)


# ── Bray-Curtis matrix ───────────────────────────────────────────────

def bray_curtis_matrix(samples):
    n = len(samples)
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(braycurtis(samples[i], samples[j]))
    return result


# ── Spectral cosine ──────────────────────────────────────────────────

def pairwise_cosine(spectra):
    n = len(spectra)
    norms = [np.linalg.norm(s) for s in spectra]
    result = []
    for i in range(n):
        for j in range(i+1, n):
            cos = np.dot(spectra[i], spectra[j]) / (norms[i] * norms[j])
            result.append(cos)
    return result


# ── PCoA ─────────────────────────────────────────────────────────────

def pcoa_simple(dist_condensed, n, k=3):
    """Classical MDS / PCoA via eigendecomposition of centered distance matrix."""
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


def main():
    print("╔════════════════════════════════════════════╗")
    print("║  Python Baseline Benchmark (numpy/scipy)   ║")
    print("╚════════════════════════════════════════════╝")

    # ── Single-vector ──────────────────────────────────
    print("\n┌──────────────────────────────────────────────┐")
    print("│ SINGLE-VECTOR REDUCTIONS                      │")
    print("├──────────────────────────────────────────────┤")
    print(f"│ {'Workload':<26} {'N':>8} {'Python':>11}│")
    print("├──────────────────────────────────────────────┤")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_counts(n)
        t = bench(shannon, data)
        report("Shannon entropy", n, t)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_counts(n)
        t = bench(simpson, data)
        report("Simpson diversity", n, t)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        data = gen_f64(n)
        t = bench(variance, data)
        report("Variance", n, t)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        a, b = gen_f64(n, 11), gen_f64(n, 22)
        t = bench(dot, a, b)
        report("Dot product", n, t)

    # ── Pairwise N×N ──────────────────────────────────
    print("\n┌──────────────────────────────────────────────┐")
    print("│ PAIRWISE N×N WORKLOADS                        │")
    print("├──────────────────────────────────────────────┤")

    for ns in [10, 20, 50, 100]:
        samples = [gen_counts(500, seed=42+i) for i in range(ns)]
        t = bench(bray_curtis_matrix, samples)
        report(f"Bray-Curtis {ns}×{ns}", ns*(ns-1)//2, t)

    for ns in [10, 50, 100, 200]:
        spectra = [gen_f64(500, seed=300+i) for i in range(ns)]
        t = bench(pairwise_cosine, spectra)
        report(f"Cosine {ns}×{ns}", ns*(ns-1)//2, t)

    # ── Matrix algebra ────────────────────────────────
    print("\n┌──────────────────────────────────────────────┐")
    print("│ MATRIX ALGEBRA                                │")
    print("├──────────────────────────────────────────────┤")

    for ns in [10, 20, 30]:
        samples = [gen_counts(200, seed=100+i) for i in range(ns)]
        dist = bray_curtis_matrix(samples)
        t = bench(pcoa_simple, dist, ns, 3)
        report(f"PCoA {ns}×{ns}", ns, t)

    print("\n══════════════════════════════════════════════")
    print("Compare with: cargo run --release --features gpu --bin benchmark_cpu_gpu")


if __name__ == "__main__":
    main()
