# Exp206: BarraCuda CPU Parity v11 — IPC Dispatch Math Fidelity

**Date:** 2026-02-27
**Status:** PASS (64/64 checks)
**Track:** cross/IPC
**Binary:** `cargo run --features ipc --release --bin validate_barracuda_cpu_v11`

## Hypothesis

The IPC dispatch layer is purely structural — calling barracuda science
functions through `ipc::dispatch::dispatch()` produces bit-identical results
to calling them directly. No math is duplicated; no numeric drift.

## Method

Seven validation domains:

| Domain | Checks | What it proves |
|--------|:------:|----------------|
| D01: Dispatch↔Direct Diversity | 25 | 5 communities × 5 metrics (Shannon, Simpson, Chao1, S_obs, Pielou) — all `EXACT_F64` |
| D02: Dispatch↔Direct Bray-Curtis | 4 | 3 pairs + identical community (BC=0) — all `EXACT_F64` |
| D03: Dispatch↔Direct QS ODE | 12 | 4 scenarios × 3 checks (t_end, peak_biofilm, steps) — all `EXACT_F64` |
| D04: Full Pipeline Chaining | 5 | Pipeline has diversity + QS stages, Shannon parity, completion |
| D05: Three-Tier NestGate | 4 | `discover_biomeos_socket`, `is_enabled`, `discover_socket` (no panic), error codes |
| D06: NUCLEUS Atomics | 10 | Tower health, 6 capabilities, Node anderson GPU-required, unknown method -32601 |
| D07: Error Handling | 4 | Empty counts, missing counts, unknown scenario, missing id → proper JSON-RPC codes |

## Results

- 64/64 checks passed
- All dispatch↔direct comparisons at `EXACT_F64` (zero drift)
- Total wall-clock: 24ms (debug), ~5ms (release)

## Key Finding

The IPC dispatch layer adds **zero numeric overhead**. Every f64 value through
dispatch matches the direct function call bit-for-bit. This proves the dispatch
is purely structural routing — safe for NUCLEUS Tower→Node coordination.

## Modules Validated

`ipc::dispatch`, `bio::diversity`, `bio::qs_biofilm`, `ncbi::nestgate`,
`ipc::protocol` (structural), `tolerances::EXACT_F64`
