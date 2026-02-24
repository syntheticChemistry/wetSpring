# wetSpring → ToadStool Handoff v15 — ODE Generic Absorption

**Date:** February 22, 2026
**Phase:** 28
**Author:** wetSpring validation pipeline
**Previous:** [v14 — Upstream GPU Fixes](WETSPRING_TOADSTOOL_V14_FEB22_2026.md)

---

## Executive Summary

wetSpring has 5 local WGSL ODE RK4 f64 shaders that all follow the same
pattern: flat parameter arrays, loop-unrolled state updates, `compile_shader_f64()`
with `fmax`/`fclamp`/`fpow` polyfills. All achieve exact CPU ↔ GPU parity.

These 5 shaders should be absorbed into a single ToadStool generic:
`BatchedOdeRK4Generic<N_VARS, N_PARAMS>` — a parameterized ODE RK4 integrator
that generates the correct shader for any (vars, params) combination.

**Key metrics:**

| Metric | Value |
|--------|-------|
| Local WGSL shaders | 5 |
| Total ODE vars across all shaders | 26 (4+5+7+4+6) |
| Total ODE params across all shaders | 95 (11+21+24+13+16+10) |
| CPU ↔ GPU parity checks | 82 (all exact) |
| Validation experiments | Exp099, Exp100, Exp101 |
| Tier B/C remaining after promotion | 0 |

---

## Part 1: The 5 ODE WGSL Shaders

All shaders live in `barracuda/src/shaders/` and follow identical conventions:

### Common Pattern

```wgsl
@group(0) @binding(0) var<storage, read> params: array<f64>;
@group(0) @binding(1) var<storage, read_write> state: array<f64>;
@group(0) @binding(2) var<uniform> meta: OdeMeta;  // dt, n_steps, n_batches

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= meta.n_batches) { return; }
    let p_off = batch * N_PARAMS;
    let s_off = batch * N_VARS;
    // RK4 loop (unrolled state access)
    for (var step = 0u; step < meta.n_steps; step++) {
        // k1 = f(state)
        // k2 = f(state + 0.5*dt*k1)
        // k3 = f(state + 0.5*dt*k2)
        // k4 = f(state + dt*k3)
        // state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    }
}
```

### Shader Details

| # | Shader File | N_VARS | N_PARAMS | Biology | CPU Reference |
|---|-------------|:------:|:--------:|---------|---------------|
| 1 | `phage_defense_ode_rk4_f64.wgsl` | 4 | 11 | Hsueh 2022 phage defense deaminase | `bio::phage_defense::derivatives()` |
| 2 | `bistable_ode_rk4_f64.wgsl` | 5 | 21 | Fernandez 2020 bistable switching | `bio::bistable::derivatives()` |
| 3 | `multi_signal_ode_rk4_f64.wgsl` | 7 | 24 | Srivastava 2011 multi-signal QS | `bio::multi_signal::derivatives()` |
| 4 | `cooperation_ode_rk4_f64.wgsl` | 4 | 13 | Bruger & Waters 2018 game theory | `bio::cooperation::derivatives()` |
| 5 | `capacitor_ode_rk4_f64.wgsl` | 6 | 16 | Mhatre 2020 phenotypic capacitor | `bio::capacitor::derivatives()` |

### Binding Layout (uniform across all 5)

| Group | Binding | Type | Content |
|:-----:|:-------:|------|---------|
| 0 | 0 | `storage, read` | `array<f64>` — flat params, `N_PARAMS` per batch |
| 0 | 1 | `storage, read_write` | `array<f64>` — flat state, `N_VARS` per batch |
| 0 | 2 | `uniform` | `OdeMeta { dt: f64, n_steps: u32, n_batches: u32 }` |

### Dispatch Geometry

```
x = ceil(n_batches / 64)
y = 1
z = 1
```

One workgroup thread per batch instance. Each thread integrates one ODE
trajectory independently.

---

## Part 2: Proposed Generic — `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`

### Why Generic

The 5 shaders are structurally identical. They differ only in:
1. `N_VARS` and `N_PARAMS` constants
2. The `derivatives()` function body (the ODE right-hand side)

A generic that takes `N_VARS`, `N_PARAMS`, and a derivatives function
template would replace all 5 local shaders with a single upstream primitive.

### Suggested API

```rust
pub struct BatchedOdeRK4Generic {
    n_vars: u32,
    n_params: u32,
    derivatives_wgsl: String,  // WGSL function body
}

impl BatchedOdeRK4Generic {
    pub fn new(n_vars: u32, n_params: u32, derivatives_wgsl: &str) -> Self;
    pub fn dispatch(&self, dev: &WgpuDevice, params: &[f64], state: &mut [f64],
                    dt: f64, n_steps: u32, n_batches: u32) -> Result<()>;
}
```

The `derivatives_wgsl` string is spliced into the shader template at compile
time. Each Spring provides its own ODE right-hand side; ToadStool provides
the RK4 integration machinery and f64 polyfill injection.

### Flat Array Convention

All 5 wetSpring ODE modules implement `to_flat()` / `from_flat()` on their
parameter and state structs:

```rust
impl CooperationParams {
    pub const N_PARAMS: usize = 13;
    pub fn to_flat(&self) -> [f64; Self::N_PARAMS] { ... }
}
impl CooperationState {
    pub const N_VARS: usize = 4;
    pub fn to_flat(&self) -> [f64; Self::N_VARS] { ... }
    pub fn from_flat(arr: &[f64]) -> Self { ... }
}
```

This convention aligns directly with the GPU buffer layout.

---

## Part 3: Validation References

| Exp | Binary | Checks | Shaders Validated |
|-----|--------|:------:|-------------------|
| 099 | `validate_cpu_vs_gpu_expanded` | phage exact | phage_defense |
| 100 | `validate_metalforge_v4` | 28/28 | bistable, multi_signal |
| 101 | `validate_pure_gpu_complete` | 52 | cooperation, capacitor (+ all 5 re-validated) |
| 102 | `validate_barracuda_cpu_v8` | 175/175 | CPU baselines for all ODE domains |
| 103 | `validate_metalforge_v5` | 58 | Cross-substrate parity for 29 domains |

### f64 Polyfill Requirements

All 5 shaders require f64 transcendental polyfills on Ada Lovelace (RTX 40-series):

| Polyfill | Used By | Implementation |
|----------|---------|---------------|
| `exp_f64(x)` | All 5 | Taylor series + range reduction |
| `log_f64(x)` | multi_signal, cooperation | Newton iteration |
| `pow_f64(x,y)` | bistable, capacitor | `exp_f64(y * log_f64(x))` |
| `fmax(a,b)` | All 5 | `select(a, b, a > b)` |
| `fclamp(x,lo,hi)` | All 5 | `fmax(lo, min(x, hi))` |

These are injected by `ShaderTemplate::for_driver_auto(source, true)` which
calls `compile_shader_f64()`. The generic must use `compile_shader_f64()` —
the existing `BatchedOdeRK4F64` calls `compile_shader()` which breaks on
naga/Vulkan backends (see v14 Bug 2).

---

## Part 4: Compose and Passthrough GPU Wrappers (Phase 28)

In addition to the 5 ODE shaders, Phase 28 promoted 10 more modules to
GPU-capable via Compose (wiring ToadStool primitives) and Passthrough
(accepting GPU buffers). These do NOT need new ToadStool primitives —
they wire existing ones:

### Compose (7 modules)

| Module | Wires | ToadStool Primitive |
|--------|-------|-------------------|
| `kmd_gpu` | Kendrick mass → k-mer histogram | `KmerHistogramGpu` |
| `merge_pairs_gpu` | Overlap scoring | `PairwiseHammingGpu` |
| `robinson_foulds_gpu` | Bipartition distance | `PairwiseHammingGpu` |
| `derep_gpu` | Sequence hashing | `KmerHistogramGpu` |
| `neighbor_joining_gpu` | Distance matrix ops | `GemmCachedF64` |
| `reconciliation_gpu` | DTL cost inference | `TreeInferenceGpu` |
| `molecular_clock_gpu` | Rate matrix ops | `GemmCachedF64` |

### Passthrough (3 modules — new primitives needed)

| Module | Needed Primitive | Notes |
|--------|-----------------|-------|
| `gbm_gpu` | `GbmBatchInferenceGpu` | Sequential boosting across rounds |
| `feature_table_gpu` | `FeatureExtractionGpu` | LC-MS feature pipeline |
| `signal_gpu` | `PeakDetectGpu` | 1D peak detection |

---

## Part 5: Recommended Actions for ToadStool

1. **Implement `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`** — parameterized
   ODE integrator that accepts a derivatives WGSL snippet. This absorbs
   all 5 wetSpring shaders and any future ODE models from other Springs.

2. **Use `compile_shader_f64()`** — not `compile_shader()`. The existing
   `BatchedOdeRK4F64` (line 209 in `batched_ode_rk4.rs`) must switch.

3. **Consider the 3 Passthrough primitives** (GBM, feature extraction,
   peak detection) as future absorption candidates. These are lower
   priority since the CPU kernels are fast and the workloads are small.

4. **Handoff validation**: Run `validate_pure_gpu_complete` and
   `validate_metalforge_v5` after absorption to verify parity.
