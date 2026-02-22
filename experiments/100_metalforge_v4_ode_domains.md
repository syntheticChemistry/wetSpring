# Exp100: metalForge Cross-Substrate v4 — ODE Domains + NPU Dispatch

| Field    | Value                                       |
|----------|---------------------------------------------|
| Script   | `validate_metalforge_v4`                    |
| Command  | `cargo run --features gpu --bin validate_metalforge_v4` |
| Status   | **PASS** (28/28)                            |
| Phase    | 27                                          |
| Depends  | Exp093, Exp099                              |

## Purpose

Validates three new local WGSL GPU shaders for ODE integration, metalForge
NPU-aware dispatch routing, and GPU→GPU→CPU PCIe pipeline patterns.

## New Local WGSL Shaders

| Shader                           | Vars | Params | CPU Parity | Notes |
|----------------------------------|------|--------|------------|-------|
| `phage_defense_ode_rk4_f64.wgsl`| 4    | 11     | Exact      | Monod phage-bacteria defense |
| `bistable_ode_rk4_f64.wgsl`     | 5    | 21     | Exact      | QS + cooperative feedback hysteresis |
| `multi_signal_ode_rk4_f64.wgsl` | 7    | 24     | Exact      | V. cholerae dual-signal (CAI-1 + AI-2) |

All shaders use established f64 patterns:
- `fmax`/`fclamp`/`fpow` polyfills (naga lacks f64 builtins)
- `(zero + literal)` for explicit f64 constant typing
- `exp_f64`/`log_f64` via `compile_shader_f64()` preamble

## Results

| Section                  | Checks | Status | Notes                               |
|--------------------------|--------|--------|-------------------------------------|
| Phage Defense (4v, 11p)  | 5/5    | PASS   | All vars exact, 8 batches consistent|
| Bistable QS (5v, 21p)    | 8/8    | PASS   | All vars exact, finite, non-negative|
| Multi-Signal QS (7v, 24p)| 10/10  | PASS   | All vars exact, 8 batches consistent|
| NPU Routing              | 3/3    | PASS   | Batch threshold, argmax classify    |
| PCIe GPU→GPU→CPU         | 2/2    | PASS   | Biofilm fraction in [0,1]           |
| **Total**                | **28/28** | **PASS** |                                 |

## metalForge Mixed-Hardware Patterns

1. **GPU→GPU Pipeline**: Phage defense → bistable QS (no CPU roundtrip)
2. **NPU-Aware Routing**: AKD1000 detected → inference path; else CPU fallback
3. **Batch Threshold Routing**: batch > 64 → GPU; batch ≤ 64 → CPU

## ToadStool Absorption Path

These 3 local shaders demonstrate the generalized batched ODE RK4 pattern:
- Arbitrary N_VARS (4, 5, 7 proven)
- Arbitrary N_PARAMS (11, 21, 24 proven)
- Shared clamping, f64 polyfill, and RK4 infrastructure

ToadStool should absorb these into a parametric `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`
primitive, with the derivative function as a pluggable shader module.
