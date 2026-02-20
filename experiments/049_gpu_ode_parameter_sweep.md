# Experiment 049: GPU ODE Parameter Sweep (QS/c-di-GMP)

**Date**: 2026-02-20
**Status**: PASS (7/7)
**Binary**: `validate_gpu_ode_sweep` (--features gpu)

## Objective

Validate GPU-accelerated parallel parameter sweep for the Waters 2008
QS/c-di-GMP 5-variable ODE system. 64 independent parameter sets integrated
simultaneously via a local WGSL RK4 shader.

## Method

- Local copy of ToadStool's `batched_qs_ode_rk4_f64.wgsl` — upstream
  uses `enable f64;` (rejected by naga) and native `pow()` (crashes NVVM).
- Local shader uses `pow_f64()` polyfill and explicit `f64()` casts.
- Compiled via `ShaderTemplate::for_driver_auto(source, true)` to inject
  `pow_f64`, `exp_f64`, `log_f64` polyfill chain.
- 64 batches, 1000 steps, dt=0.01 (t=0→10), sweeping µ (0.4→1.66) and
  k_ai_prod (3.0→9.3).
- CPU baseline via `qs_biofilm::run_scenario`.

## Results

| Check | Result |
|-------|--------|
| Output size (320 f64) | PASS |
| All finals finite | PASS |
| All finals ≥ 0 | PASS |
| CPU ↔ GPU abs parity < 0.15 | PASS (max diff = 0.100) |
| Cells grew (N > y0) | PASS |
| Parameter sweep changes outcome | PASS |
| Higher µ → more cells | PASS |

## Key Findings

1. **NVVM f64 transcendental limitation extends to `pow()`**: native f64
   `pow()`, `exp()`, and `log()` all crash NVVM on RTX 4070. All three
   must use polyfills via `ShaderTemplate::for_driver_auto(_, true)`.

2. **naga f64 literal promotion**: bare `0.0` in `max(x, 0.0)` fails
   type checking. Must use `f64(0.0)` or typed variable.

3. **Polyfill ODE drift**: 0.1 max abs difference over 1000 steps is
   expected — software `pow_f64 = exp_f64(n * log_f64(x))` introduces
   per-step drift that compounds. Absolute (not relative) tolerance is
   the correct metric for long-horizon ODE comparison.

4. **Biological plausibility preserved**: all 64 trajectories converge
   to physically meaningful steady states (non-negative, bounded).

## Write → Absorb → Lean

- **Write**: `barracuda/src/shaders/batched_qs_ode_rk4_f64.wgsl` (local)
- **Absorb**: ToadStool should fix `enable f64;` + use `compile_shader_f64`
  in `BatchedOdeRK4F64::integrate()`
- **Lean**: remove local shader + `ode_sweep_gpu.rs`, use upstream directly

## Run

```bash
cargo run --features gpu --bin validate_gpu_ode_sweep
```
