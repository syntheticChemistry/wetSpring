# Exp069: Python → Rust CPU → GPU Three-Tier Benchmark

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross/benchmark
**Binary:** `benchmark_three_tier`
**Command:** `cargo run --features gpu --release --bin benchmark_three_tier`

---

## Objective

Formalize the full value chain: Python (numpy/scipy) → Rust CPU (BarraCUDA)
→ GPU (ToadStool + local WGSL). This proves the Write → Absorb → Lean
pattern delivers measurable performance at each step.

---

## Value Chain

```
Python (numpy/scipy)  ─── 1× baseline
        │ 10-25× speedup
        ▼
Rust CPU (BarraCUDA)  ─── pure math, no interpreter overhead
        │ 1.7× at large N (dispatch breakeven dependent)
        ▼
Rust GPU (ToadStool)  ─── parallel compute, streaming dispatch
```

---

## Workloads Benchmarked

| Workload | Python | Rust CPU | Rust GPU | Notes |
|----------|--------|----------|----------|-------|
| Shannon entropy | numpy | `bio::diversity` | `FusedMapReduceF64` | ToadStool primitive |
| Simpson diversity | numpy | `bio::diversity` | `FusedMapReduceF64` | ToadStool primitive |
| Bray-Curtis | scipy | `bio::diversity` | `BrayCurtisF64` | ToadStool primitive |
| Variance | numpy | manual | `stats_gpu` | Local extension |
| PCoA | numpy.linalg | `bio::pcoa` | `BatchedEighGpu` | ToadStool primitive |

---

## Protocol

1. Run Python baseline via `scripts/benchmark_python_baseline.py` (JSON output)
2. Run Rust CPU benchmarks (same workloads, same data sizes)
3. Run Rust GPU benchmarks (same workloads, same data sizes)
4. Display three-column comparison table

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline tool | Python 3.x (numpy/scipy) |
| Exact command | `cargo run --features gpu --release --bin benchmark_three_tier` |
| Data | Synthetic vectors at N=1K, 10K, 100K, 1M |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!_OS 22.04 |
