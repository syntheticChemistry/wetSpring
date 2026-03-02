# Exp295: metalForge v14 — Paper-Math Cross-System Validation

**Status:** ALL PASS (28/28 checks)
**Date:** 2026-03-02
**Binary:** `validate_metalforge_v14_paper_chain`
**Command:** `cargo run -p wetspring-forge --bin validate_metalforge_v14_paper_chain`
**Feature gate:** none

## Purpose

Validates the complete paper-math chain across mixed hardware: GPU → NPU → CPU
substrate transitions. Proves the same math produces the same results
regardless of which hardware executes it. NUCLEUS atomics coordinate routing.

## Sections

| Section | Checks | Description |
|---------|:------:|-------------|
| S1: Paper Workloads | 6 | 47 workloads, 45 absorbed, diversity/spectral/linalg registered |
| S2: Track Routing | 2 | All 6 tracks routed (4 GPU-preferred, 2 CPU-preferred) |
| S3: PCIe Streaming | 6 | 4-stage GPU pipeline, 3 GPU-chained, 0 CPU round-trips |
| S4: Mixed Pipeline | 2 | GPU→CPU→GPU→CPU→CPU chain, 4 cross-substrate transitions |
| S5: Cross-Substrate Parity | 4 | GPU vs CPU routing, bandwidth-aware 50MB → GPU |
| S6: Sovereign Paper | 8 | No NestGate/Songbird required, 45/47 route locally |

## Hardware Discovered

- 3 GPUs: RTX 4070, TITAN V (NVK), RTX 4070/PCIe/SSE2
- PCIe transfer: 322 µs (RTX 4070, 10MB), 640 µs (TITAN V, 10MB)
- Sovereign mode: NestGate absent, Songbird absent — local-only

## Chain

Streaming v9 (Exp294) → **metalForge v14 (this)**
