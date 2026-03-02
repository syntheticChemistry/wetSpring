# Exp288: CPU vs GPU v8 — ToadStool Compute Dispatch + Pure Math

**Status:** PASS (16/16 CPU-only, GPU checks added with `--features gpu`)
**Date:** 2026-03-02
**Binary:** `validate_cpu_vs_gpu_v8`
**Command:** `cargo run --release [--features gpu] --bin validate_cpu_vs_gpu_v8`
**Feature gate:** gpu (optional, CPU-only mode available)

## Purpose

Deep validation of pure Rust CPU math against ToadStool GPU dispatch.
Extends v7 (D01–D27) with V92D compute-dispatch-level validation.
Runs CPU structural checks without GPU feature; adds GPU parity checks
when `--features gpu` is enabled.

## New Domains (D28–D32)

| Domain | Checks | Description |
|--------|--------|-------------|
| D28 | 6+2 | FusedMapReduce — Shannon/Simpson CPU reference + GPU parity |
| D29 | 4+11 | Anderson spectral — CPU eigenvalues + GPU determinism |
| D30 | 3+1 | Bray-Curtis pairwise matrix — CPU reference + GPU check |
| D31 | 5 | Statistics — mean, variance, min/max analytical |
| D32 | 4 | Streaming determinism — bitwise rerun identical |

## Tolerance

CPU reference is exact (analytical). GPU parity uses
`tolerances::GPU_VS_CPU_F64`. Spectral determinism is bitwise identical.

## Chain

CPU v21 (Exp287) → **Parity v8 (this)** → metalForge v13 (Exp289)
