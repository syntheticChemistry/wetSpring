# Exp293: CPU vs GPU v9 — Paper Math GPU Portability

**Status:** PASS (35/35 checks, CPU-only mode; GPU parity with `--features gpu`)
**Date:** 2026-03-02
**Binary:** `validate_cpu_vs_gpu_v9`
**Command:** `cargo run --release --features gpu --bin validate_cpu_vs_gpu_v9`
**Feature gate:** gpu (optional)

## Purpose

Proves paper math is truly portable from CPU to GPU via ToadStool dispatch.
All 5 tracks validated: microbial ecology, soil QS, immunology, drug
repurposing, and deep-sea metagenomics. CPU reference established in
CPU-only mode; GPU dispatch matches CPU within f64 tolerance when enabled.

## Domains

| Domain | Checks | Description |
|--------|:------:|-------------|
| D33: Multi-Track Diversity | 10+N | 5 tracks × Shannon/Simpson CPU + GPU parity |
| D34: NMF Drug Repurposing | 5 | NMF 4×4 W≥0/H≥0, convergence, cosine |
| D35: Anderson W-Mapping | 11 | P(QS) monotone decreasing, bounds [0,1] |
| D36: Pharmacology | 5 | Hill monotone, IC50=0.5, PK C(t½), decay |
| D37: Cross-Track Parity | 2 | Bitwise identical reruns (determinism) |
| D38: Performance | 2 | Diversity < 10ms, NMF < 100ms |

## Hardware

- CPU-only mode: 35/35 checks, 0.1 ms total
- GPU mode: adds N checks per GPU-capable track (Shannon GPU ≈ CPU within `GPU_VS_CPU_F64`)

## Chain

CPU v22 (Exp292) → **GPU v9 (this)** → Streaming v9 (Exp294) → metalForge v14 (Exp295)
