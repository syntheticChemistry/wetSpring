# Exp067: ToadStool Dispatch Overhead Profiling

**Date:** February 21, 2026
**Status:** DONE
**Track:** cross/GPU
**Binary:** `benchmark_dispatch_overhead`
**Command:** `cargo run --features gpu --release --bin benchmark_dispatch_overhead`

---

## Objective

Measure the actual ToadStool/wgpu dispatch overhead for each GPU domain.
This separates "time to set up the dispatch" from "time to compute" —
the data that drives metalForge routing decisions.

For each domain, measures:
1. **Upload**: CPU → GPU buffer creation and data transfer
2. **Dispatch**: Shader compilation (cached) + dispatch call
3. **Compute**: Actual GPU execution time
4. **Readback**: GPU → CPU buffer readback

---

## Why This Matters

GPU dispatch has fixed overhead (~0.5-2ms). For small inputs, this overhead
dominates and CPU wins. For large batches, GPU compute scales better and
GPU wins. The crossover point is different for each algorithm.

This experiment produces the **dispatch overhead budget** that metalForge
uses to decide: "Should this workload go to CPU or GPU?"

---

## Provenance

| Field | Value |
|-------|-------|
| Baseline tool | wgpu/ToadStool dispatch timing |
| Exact command | `cargo run --features gpu --release --bin benchmark_dispatch_overhead` |
| Data | Minimal test vectors (measures overhead, not throughput) |
| Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!_OS 22.04 |
