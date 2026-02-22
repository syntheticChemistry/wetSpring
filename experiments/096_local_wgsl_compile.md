# Experiment 096 — Local WGSL Shader Compile + Dispatch Validation

| Field   | Value |
|---------|-------|
| Script  | `validate_local_wgsl_compile` |
| Binary  | `cargo run --features gpu --bin validate_local_wgsl_compile` |
| Status  | **PASS** (10/10 checks) |
| Date    | 2026-02-22 |
| Phase   | 22 |
| GPU     | RTX 4070 (Ada Lovelace, f64 1:2) |

## Purpose

Compiled and dispatch-tested all 4 local WGSL shaders before absorption.
All 4 were subsequently absorbed by ToadStool S39-41; Lean phase complete.

## Results

| Shader | Compile | Dispatch | f64 Preamble | Checks |
|--------|---------|----------|-------------|--------|
| `kmer_histogram_f64.wgsl` | PASS | PASS (4 bins verified) | No (f32/atomic) | 4 |
| `taxonomy_fc_f64.wgsl` | PASS | PASS (2 scores verified) | Yes | 3 |
| `unifrac_propagate_f64.wgsl` | PASS (2 entry points) | — | Yes | 2 |
| `batched_qs_ode_rk4_f64.wgsl` | PASS | — (tested in Exp046/092) | Yes | 1 |

### Key Findings

- All 4 shaders compile successfully on RTX 4070 via naga/Vulkan
- The f64 shaders use `compile_shader_f64()` which injects the f64 preamble
  and `pow_f64`/`exp_f64`/`log_f64` polyfills for Ada Lovelace GPUs
- `kmer_histogram` uses atomic u32 operations — no f64 needed
- `taxonomy_fc` produces exact log-posterior scores matching CPU baseline
- `unifrac_propagate` has 2 entry points (leaf_init + propagate_level)
  that both compile; full dispatch test deferred to Exp082 integration

## Checks

- 10 checks, all PASS
- Covers compile + dispatch for kmer/taxonomy, compile-only for unifrac/ODE
