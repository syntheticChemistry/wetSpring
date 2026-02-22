# Experiment 078 — ODE GPU Sweep Readiness

**Date:** February 22, 2026
**Track:** cross/GPU (evolution readiness)
**Status:** COMPLETE

## Purpose

Refactor all ODE-based biological models to expose GPU-compatible parameter
flattening APIs, preparing them for batched GPU parameter sweep via
ToadStool's `BatchedOdeRK4F64` primitive (once the `enable f64;` blocker
is resolved upstream).

## Modules Refactored

| Module | N_VARS | N_PARAMS | Paper | Tier |
|--------|:------:|:--------:|-------|------|
| `qs_biofilm` | 5 | 18 | Waters 2008 | A (existing GPU shader) |
| `bistable` | 5 | 21 | Fernandez 2020 | A (maps to ODE sweep) |
| `multi_signal` | 7 | 24 | Srivastava 2011 | B → A |
| `phage_defense` | 4 | 11 | Hsueh 2022 | B → A |
| `cooperation` | 4 | 13 | Bruger & Waters 2018 | C (now B) |

## API Added

Each module now exposes:

- `N_VARS: usize` — state variable count
- `N_PARAMS: usize` — flat parameter count
- `Params::to_flat(&self) -> [f64; N_PARAMS]` — serialize for GPU buffer
- `Params::from_flat(&[f64]) -> Self` — deserialize from GPU readback

All `to_flat` methods are `const fn` for compile-time evaluation.

## Tests Added (10)

Each module gained 2 new tests:

- `flat_params_round_trip` — serialize/deserialize bitwise identity
- `flat_params_gpu_parity` — ODE results identical through flat round-trip

## GPU Dispatch Pattern

When ToadStool's `BatchedOdeRK4F64` supports variable-width systems:

```text
CPU: params.to_flat() → [f64; N_PARAMS] → GPU param buffer
GPU: batched RK4 integration (N workgroups × N_VARS state × N_PARAMS params)
CPU: read output buffer → OdeResult
```

The flat layout is `#[repr(C)]`-compatible: contiguous f64 array matching
the WGSL storage buffer layout used by `batched_qs_ode_rk4_f64.wgsl`.

## Validation

```
cargo fmt --check      → clean
cargo clippy (ped+nur) → 0 warnings
cargo test --lib       → 645 passed (was 635, +10 new)
cargo test --tests     → 60 passed
cargo test --doc       → 14 passed
```

## Blocker

ToadStool `BatchedOdeRK4F64` contains `enable f64;` on line 35 of its WGSL
shader, which naga rejects. The existing local ODE sweep shader works around
this for the 5-variable QS system. Once ToadStool fixes the directive, all
5 ODE modules can use the upstream batched primitive directly.

## Evolution Impact

- `multi_signal` and `phage_defense` promoted from Tier B to Tier A
  (GPU-ready, pending upstream fix)
- `cooperation` promoted from Tier C to Tier B (now has GPU-compatible API)
- `bistable` was already Tier A; now has explicit flattening
- `qs_biofilm` was already GPU-dispatched; now has round-trip API
