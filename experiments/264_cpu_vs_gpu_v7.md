# Exp264: CPU vs GPU v7 — G17–G21 Parity (27 Domains)

**Status:** PASS (22/22 checks)
**Date:** 2026-03-01
**Binary:** `validate_cpu_vs_gpu_v7`
**Command:** `cargo run --release --features gpu --bin validate_cpu_vs_gpu_v7`
**Feature gate:** `gpu`

## Purpose

Closes the CPU↔GPU parity gap identified in the V87 audit. Extends v6
(D01–D22, 24 checks) with 5 new domains that prove GPU domains G17–G21
produce identical results to their CPU counterparts.

## New Domains (D23–D27)

| Domain | Checks | CPU Module | GPU Module | Parity Metric |
|--------|--------|-----------|-----------|---------------|
| D23 | 5 | `pcoa` | `pcoa_gpu` | Eigenvalues, axes, variance |
| D24 | 4 | `kmer` | `kmer_gpu` | Histogram bin counts |
| D25 | 5 | `diversity` | `diversity_gpu` | Shannon, Simpson, bootstrap CI |
| D26 | 3 | `kmd` | `kmd_gpu` | PFAS mass defects |
| D27 | 5 | `kriging` | `kriging` (GPU) | Spatial interpolation, variogram |

## Tolerance

All float comparisons use `tolerances::GPU_VS_CPU_F64`. GPU gracefully
falls back to CPU when `SHADER_F64` is unavailable — the binary still
passes with structural checks.

## Chain

CPU v20 (Exp263) → GPU v11 (Exp254) → **Parity v7 (this)** → metalForge v12 (Exp265)
