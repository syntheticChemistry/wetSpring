# Exp079: BarraCUDA CPU v6 — ODE Flat Param Fidelity

**Status**: COMPLETE
**Date**: 2026-02-22
**Binary**: `validate_barracuda_cpu_v6`
**Checks**: 48 (all PASS)

## Purpose

Validates that the GPU-compatible flat parameter APIs (`to_flat`/`from_flat`)
introduced in Exp078 produce bitwise-identical ODE integration results across
all 6 biological ODE models. This proves the serialization path required for
GPU dispatch preserves pure Rust math fidelity.

## Evolution Chain Position

```
Python baseline → Rust validation → [THIS] flat API parity → GPU dispatch
```

This experiment sits between Rust CPU validation (individual ODE binaries) and
GPU parameter sweep dispatch (Exp049/050). It proves the flat array layout
introduces zero numerical error.

## Modules Validated

| Module | Paper | Vars | Params | Flat RT | ODE Bitwise | Python Parity |
|--------|-------|------|--------|---------|-------------|---------------|
| qs_biofilm | Waters 2008 | 5 | 18 | PASS | PASS | N=0.975, B<0.05 |
| bistable | Fernandez 2020 | 5 | 21 | PASS | PASS | B=0.746 (sessile), hysteresis |
| multi_signal | Srivastava 2011 | 7 | 24 | PASS | PASS | B=0.413, HapR>0.3 |
| phage_defense | Hsueh 2022 | 4 | 11 | PASS | PASS | Bd > Bu |
| cooperation | Bruger 2018 | 4 | 13 | PASS | PASS | freq<0.5 (cheater) |
| capacitor | Mhatre 2020 | 5 | struct | PASS | PASS | stress > normal |

## Three-Stage Validation

1. **Flat round-trip**: `to_flat() → from_flat() → to_flat()` produces
   bitwise-identical arrays (0 ULP drift).
2. **ODE bitwise parity**: Integrating with the round-tripped struct produces
   steady-state values with zero floating-point difference from the direct struct.
3. **Python baseline parity**: The flat API path matches documented Python
   scipy `odeint` steady-state values within justified tolerances.

## Key Results

- **Zero ULP drift**: All flat round-trips are bitwise identical.
- **Zero ODE error**: `from_flat()` structs integrate identically to originals.
- **Bistable hysteresis**: Confirmed via bifurcation scan (20-point sweep).
- **Capacitor determinism**: 5-variable deterministic rerun confirmed.
- **Runtime**: ~2ms total (6 modules, debug build).

## Provenance

| Field | Value |
|-------|-------|
| Baseline commit | `e4358c5` |
| Baseline tool | scipy `odeint` (LSODA) |
| Baseline version | scipy 1.12.0, numpy 1.26.4 |
| Baseline date | 2026-02-22 |
| Exact command | `cargo run --release --bin validate_barracuda_cpu_v6` |
