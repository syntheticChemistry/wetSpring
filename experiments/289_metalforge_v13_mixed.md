# Exp289: metalForge v13 — Mixed Hardware + NUCLEUS Atomics + PCIe Bypass

**Status:** PASS (30/30 checks)
**Date:** 2026-03-02
**Binary:** `validate_metalforge_v13_mixed`
**Command:** `cargo run -p wetspring-forge --bin validate_metalforge_v13_mixed`
**Feature gate:** none

## Purpose

Validates mixed-hardware dispatch and NUCLEUS atomics at the V92D state.
Tower discovery, workload catalog absorption tracking, capability-based
routing, bandwidth-aware routing (PCIe bypass vs CPU round-trip), streaming
pipeline analysis, node dispatch, mixed pipeline topology (NPU→GPU→CPU),
and sovereign mode (zero external dependencies).

## Sections (S1–S8)

| Section | Checks | Description |
|---------|--------|-------------|
| S1 | 5 | Tower discovery — substrates, bandwidth tiers, F64 GPUs |
| S2 | 5 | Workload catalog — 47 workloads, absorption tracking, lean |
| S3 | 4 | Capability routing — GPU, CPU-preferred, NPU conditional |
| S4 | 4 | Bandwidth-aware routing — small vs large workloads, transfer cost |
| S5 | 6 | Streaming analysis — GPU-chained vs CPU round-trip topologies |
| S6 | 2 | Node dispatch — majority of workloads routed |
| S7 | 2 | Mixed pipeline — GPU→CPU handoff present |
| S8 | 4 | Sovereign mode — operates without NestGate/Songbird |

## NUCLEUS Atomics

| Atomic | Role | Validated |
|--------|------|-----------|
| **Tower** | Substrate discovery + capability graph | S1, S3 |
| **Node** | Workload→substrate routing | S6, S7 |
| **Nest** | Storage protocol (NestGate or sovereign) | S8 |

## Chain

Parity v8 (Exp288) → **metalForge v13 (this)** → biomeOS Graph v2 (Exp290)
