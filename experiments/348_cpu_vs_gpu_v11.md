# Exp348: CPU vs GPU v11

**Date:** March 2026
**Track:** V109 — Upstream Rewire + NUCLEUS Atomics
**Binary:** `validate_cpu_vs_gpu_v11`
**Status:** PASS (19 checks)

---

## Hypothesis

GPU portability is preserved after V109 upstream changes: shannon_gpu is now synchronous, GPU_VS_CPU_F64 tolerance replaces GPU_CPU_PARITY. CPU and GPU produce identical results within tolerance.

## Method

4 domains: sync diversity GPU (D43), biogas kinetics GPU (D44), Anderson W GPU (D45), cross-track GPU composition (D46). CPU reference always runs; GPU checks activate with `--features gpu`.

## Results

All 19 checks PASS (7 GPU parity checks included). See `cargo run --release --features gpu --bin validate_cpu_vs_gpu_v11`.

## Key Finding

Sync GPU API produces identical results to CPU. Shannon GPU parity confirmed at GPU_VS_CPU_F64 (1e-6) tolerance. Track 6 math is substrate-independent.
