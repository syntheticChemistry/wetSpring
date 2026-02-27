# Exp214: NUCLEUS Mixed Hardware V8 — V66 I/O Evolution

**Track:** cross (IPC)
**Phase:** 66
**Status:** COMPLETE — 49/49 checks PASS
**Binary:** `validate_nucleus_v8_mixed`
**Features:** `ipc`

## Purpose

Validates the NUCLEUS mixed hardware model through the IPC dispatch layer,
ensuring V66-evolved I/O (byte-native FASTQ, bytemuck nanopore, streaming
MS2) produces correct results when routed through JSON-RPC 2.0 dispatch.

## What It Tests

- Tower capabilities (crypto, TLS, HTTP, discovery)
- Byte-native FASTQ → IPC dispatch → diversity (Shannon parity)
- Bytemuck nanopore → IPC dispatch → signal stats (mean, std_dev)
- MS2 streaming → IPC dispatch → spectral cosine match
- Nest metrics (direct `Metrics` struct usage)
- CPU fallback parity (GPU unavailable path)
- Full pipeline chain: FASTQ → quality → diversity → QS ODE (4-stage)
- General dispatch routing (12 methods verified)

## Key Findings

All 49 checks pass. The IPC layer preserves math fidelity through the
V66-evolved I/O stack. The `dispatch::dispatch` function correctly routes
to CPU implementations when GPU feature is not enabled. Metrics struct
works as direct API (not dispatchable via IPC). Full 4-stage pipeline
chain produces correct final QS state from raw FASTQ input.
