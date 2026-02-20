#!/usr/bin/env python3
"""Python 16S control baseline for public BioProject data.

Runs a pure-Python 16S pipeline (no QIIME2 dependency) on the same FASTQ
files that the Rust validators process, producing reference values for
head-to-head Rust-vs-Python comparison.

Datasets:
  - PRJNA488170 / SRR7760408: Nannochloropsis sp. outdoor 16S (Wageningen)
  - PRJNA382322 / SRR5452557:  AlgaeParc 2013 bacterial community

Requires: pip install numpy scipy
"""
import gzip
import json
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import braycurtis

WORKSPACE = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
#  FASTQ parsing (standalone, mirrors Rust io::fastq)
# ═══════════════════════════════════════════════════════════════════

def parse_fastq_gz(path, max_reads=None):
    """Parse gzipped FASTQ, return list of (id, sequence, quality) tuples."""
    records = []
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", errors="replace") as fh:
            while True:
                header = fh.readline().rstrip("\n")
                if not header:
                    break
                if not header.startswith("@"):
                    continue
                seq = fh.readline().rstrip("\n")
                fh.readline()  # +
                qual = fh.readline().rstrip("\n")
                if seq and qual:
                    rid = header[1:].split()[0]
                    records.append((rid, seq, qual))
                    if max_reads and len(records) >= max_reads:
                        break
    except EOFError:
        pass  # truncated gzip — keep what we got
    return records


# ═══════════════════════════════════════════════════════════════════
#  Quality filtering (mirrors Rust bio::quality)
# ═══════════════════════════════════════════════════════════════════

def phred_scores(qual_str):
    """Convert ASCII quality string to Phred scores (Phred33)."""
    return [ord(c) - 33 for c in qual_str]


def quality_filter(records, min_qual=20, window_size=4, min_len=50):
    """Sliding-window quality trim + min-length filter."""
    filtered = []
    for rid, seq, qual in records:
        scores = phred_scores(qual)
        trim_pos = len(scores)
        for i in range(len(scores) - window_size + 1):
            window_mean = sum(scores[i:i + window_size]) / window_size
            if window_mean < min_qual:
                trim_pos = i
                break
        if trim_pos >= min_len:
            filtered.append((rid, seq[:trim_pos], qual[:trim_pos]))
    return filtered


# ═══════════════════════════════════════════════════════════════════
#  Dereplication (mirrors Rust bio::derep)
# ═══════════════════════════════════════════════════════════════════

def dereplicate(records, min_abundance=1):
    """Collapse identical sequences, return sorted by abundance desc."""
    counts = Counter()
    for _rid, seq, _qual in records:
        counts[seq] += 1
    uniques = [(seq, count) for seq, count in counts.most_common()
               if count >= min_abundance]
    return uniques


# ═══════════════════════════════════════════════════════════════════
#  Diversity metrics (mirrors Rust bio::diversity)
# ═══════════════════════════════════════════════════════════════════

def shannon(counts):
    """Shannon entropy H' = -sum(p_i * ln(p_i))."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def simpson(counts):
    """Simpson's diversity index 1 - sum(p_i^2)."""
    total = sum(counts)
    if total <= 1:
        return 0.0
    d = 0.0
    for c in counts:
        p = c / total
        d += p * p
    return 1.0 - d


def observed_features(counts):
    """Number of non-zero features."""
    return sum(1 for c in counts if c > 0)


def chao1(counts):
    """Chao1 richness estimator."""
    s_obs = observed_features(counts)
    f1 = sum(1 for c in counts if c == 1)
    f2 = sum(1 for c in counts if c == 2)
    if f2 == 0:
        return s_obs + (f1 * (f1 - 1)) / 2.0 if f1 > 0 else float(s_obs)
    return s_obs + (f1 * f1) / (2.0 * f2)


def bray_curtis_matrix(samples):
    """All-pairs Bray-Curtis distance matrix for list of count vectors."""
    n = len(samples)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = braycurtis(samples[i], samples[j])
            dm[i, j] = d
            dm[j, i] = d
    return dm


# ═══════════════════════════════════════════════════════════════════
#  Pipeline: FASTQ → QC → Derep → Diversity
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(name, fastq_path, max_reads=50000, min_abund=2):
    """Run the full Python 16S pipeline on a single FASTQ file."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  Input: {fastq_path}")
    print(f"{'=' * 70}")

    t0 = time.time()

    # Step 1: Parse
    print(f"  [1/4] Parsing FASTQ (max {max_reads} reads)...")
    records = parse_fastq_gz(fastq_path, max_reads=max_reads)
    n_raw = len(records)
    print(f"         Parsed {n_raw} reads")
    if n_raw == 0:
        print("  ERROR: No reads parsed")
        return None

    mean_len = sum(len(s) for _, s, _ in records) / n_raw
    print(f"         Mean read length: {mean_len:.0f} bp")

    # Step 2: Quality filter
    print("  [2/4] Quality filtering...")
    filtered = quality_filter(records, min_qual=20, window_size=4, min_len=50)
    n_filt = len(filtered)
    retention = n_filt / n_raw
    print(f"         Retained {n_filt}/{n_raw} ({retention * 100:.1f}%)")

    # Step 3: Dereplicate
    print(f"  [3/4] Dereplicating (min abundance {min_abund})...")
    uniques = dereplicate(filtered, min_abundance=min_abund)
    print(f"         {len(uniques)} unique sequences")

    # Step 4: Diversity
    print("  [4/4] Computing diversity metrics...")
    abundances = [count for _, count in uniques]

    if not abundances:
        print("  WARNING: No sequences after dereplication")
        return None

    h = shannon(abundances)
    s = simpson(abundances)
    obs = observed_features(abundances)
    c1 = chao1(abundances)

    elapsed = time.time() - t0

    results = {
        "name": name,
        "fastq": str(fastq_path),
        "reads_parsed": n_raw,
        "mean_read_length_bp": round(mean_len, 1),
        "reads_after_qc": n_filt,
        "quality_retention_pct": round(retention * 100, 2),
        "unique_sequences": len(uniques),
        "min_abundance_threshold": min_abund,
        "diversity": {
            "shannon": round(h, 6),
            "simpson": round(s, 6),
            "observed_features": obs,
            "chao1": round(c1, 2),
        },
        "elapsed_seconds": round(elapsed, 2),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }

    print(f"\n  Results:")
    print(f"    Reads parsed:       {n_raw}")
    print(f"    Mean read length:   {mean_len:.0f} bp")
    print(f"    QC retention:       {retention * 100:.1f}%")
    print(f"    Unique sequences:   {len(uniques)}")
    print(f"    Shannon:            {h:.4f}")
    print(f"    Simpson:            {s:.4f}")
    print(f"    Observed features:  {obs}")
    print(f"    Chao1:              {c1:.2f}")
    print(f"    Elapsed:            {elapsed:.2f}s")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  wetSpring Python 16S Control Baseline")
    print("  Pure-Python pipeline (no QIIME2) on public BioProject data")
    print("=" * 70)

    all_results = {}

    # ── Dataset 1: PRJNA488170 / SRR7760408 ──
    prjna488170_r1 = (
        WORKSPACE
        / "data/paper_proxy/nannochloropsis_16s/SRR7760408/SRR7760408_1.fastq.gz"
    )
    if prjna488170_r1.exists():
        result = run_pipeline(
            "PRJNA488170 / SRR7760408 (Nannochloropsis outdoor 16S, Wageningen)",
            prjna488170_r1,
            max_reads=50000,
            min_abund=2,
        )
        if result:
            all_results["PRJNA488170_SRR7760408"] = result
    else:
        print(f"\n  SKIP: {prjna488170_r1} not found")

    # ── Dataset 2: PRJNA382322 / SRR5452557 ──
    prjna382322_r1 = (
        WORKSPACE
        / "data/ncbi_bulk/PRJNA382322/SRR5452557/SRR5452557_1.fastq.gz"
    )
    if prjna382322_r1.exists():
        result = run_pipeline(
            "PRJNA382322 / SRR5452557 (AlgaeParc 2013 bacterial community)",
            prjna382322_r1,
            max_reads=50000,
            min_abund=2,
        )
        if result:
            all_results["PRJNA382322_SRR5452557"] = result
    else:
        print(f"\n  SKIP: {prjna382322_r1} not found")

    # ── Dataset 3: PRJNA1114688 (first sample) ──
    prjna1114688_dir = WORKSPACE / "data/public_benchmarks/PRJNA1114688"
    if prjna1114688_dir.exists():
        subdirs = sorted(prjna1114688_dir.iterdir())
        for sd in subdirs[:2]:  # first 2 samples
            r1 = sd / f"{sd.name}_1.fastq.gz"
            if r1.exists():
                result = run_pipeline(
                    f"PRJNA1114688 / {sd.name} (N. oculata + B. plicatilis)",
                    r1,
                    max_reads=50000,
                    min_abund=2,
                )
                if result:
                    all_results[f"PRJNA1114688_{sd.name}"] = result

    # ── Save results ──
    out_dir = WORKSPACE / "experiments/results/python_16s_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_16s_baselines.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Python 16S Control Baselines")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':<40} {'Reads':>8} {'QC%':>6} {'Uniq':>6} {'Shannon':>8} {'Simpson':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8}")
    for key, r in all_results.items():
        d = r["diversity"]
        print(
            f"  {key:<40} {r['reads_parsed']:>8} "
            f"{r['quality_retention_pct']:>5.1f}% "
            f"{r['unique_sequences']:>6} "
            f"{d['shannon']:>8.4f} "
            f"{d['simpson']:>8.4f}"
        )

    print(f"\n  Total datasets processed: {len(all_results)}")
    print("  These values serve as Python control baselines for Rust validators")
    print("  (validate_algae_16s, validate_extended_algae, validate_public_benchmarks)")

    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
