# wetSpring V97e — Cross-Spring Provenance Rewire Handoff

**Date:** March 7, 2026
**From:** wetSpring (life science biome)
**To:** barraCuda / toadStool / coralReef teams
**Phase:** V97e (follows V97d+ ecosystem sync)

---

## Executive Summary

wetSpring has completed the **full rewire to modern barraCuda APIs**, migrating
from positional-arg dispatch to builder-pattern structs, wiring fine-grained
precision routing, and integrating the `shaders::provenance` registry. All
breaking API changes from barraCuda HEAD (`2a6c072`) are now absorbed.

**Key metrics:**
- 1,346 tests pass (0 failures)
- 31 provenance checks pass
- Zero clippy warnings (default + GPU features)
- Zero doc warnings

---

## Rewire Details

### 1. Builder Pattern Migration (Complete)

| Dispatch | Old Signature | New Struct | Files Changed |
|----------|--------------|------------|:-------------:|
| HMM Forward | 10 positional args | `HmmForwardArgs` | 1 |
| DADA2 E-step | 9 positional args | `Dada2DispatchArgs` (`Dada2Dimensions` + `Dada2Buffers`) | 1 |
| Gillespie SSA | 7 positional args | `GillespieModel` + 4 args | 4 |

**Impact:** Named fields improve crash diagnostics and prepare for
`BatchedOdeRK45F64` adoption (adaptive step needs `Rk45DispatchArgs`).

### 2. Precision Routing (Complete)

`GpuF64::optimal_precision()` now routes through `PrecisionRoutingAdvice`
instead of the coarser `Fp64Strategy`:

| Advice | Precision | When |
|--------|-----------|------|
| `F64Native` | `Precision::F64` | Compute-class GPUs (Titan V, V100, MI250X) |
| `F64NativeNoSharedMem` | `Precision::F64` | NVK with f64 compute but broken shared-mem f64 reductions |
| `Df64Only` | `Precision::Df64` | Consumer GPUs with throttled f64 (RTX 4070, RDNA2+) |
| `F32Only` | `Precision::F32` | No f64 support (Apple M, Intel iGPU) |

New `Fp64Strategy::Sovereign` variant also handled.

### 3. Provenance API (New)

New `wetspring_barracuda::provenance` module (GPU-gated) wires
`barracuda::shaders::provenance` with wetSpring-specific views:

- `shaders_authored()` — 5 bio shaders originated by wetSpring
- `shaders_consumed()` — 17 shaders consumed total
- `shaders_from_other_springs()` — 12 shaders from other springs
- `wetspring_provenance_summary()` — formatted provenance report

### 4. Error Handling Improvements

8 pre-existing `.submit()` calls that silently ignored `Result` now properly
propagate errors via `map_err()?`. Affected modules: bistable, capacitor,
cooperation, gemm_cached (2), multi_signal, pairwise_l2, phage_defense.

---

## Cross-Spring Evolution — What We Validated

### wetSpring → Other Springs (Outbound)

| Shader | Consumers | Impact |
|--------|-----------|--------|
| `bio/smith_waterman_banded_f64.wgsl` | neuralSpring | Protein folding alignment |
| `bio/gillespie_ssa_f64.wgsl` | neuralSpring | Evolutionary dynamics simulation |
| `bio/hmm_forward_f64.wgsl` | neuralSpring | Batched inference |
| `bio/felsenstein_f64.wgsl` | (wetSpring only) | Phylogenetic likelihood |
| `reduce/fused_map_reduce_f64.wgsl` | airSpring, hotSpring | Universal map-reduce |

### Other Springs → wetSpring (Inbound)

| Shader | Origin | wetSpring Use |
|--------|--------|---------------|
| `math/df64_core.wgsl` | hotSpring | DF64 precision on consumer GPUs |
| `math/df64_transcendentals.wgsl` | hotSpring | exp/log/sin/cos for f64-class work |
| `md/stress_virial_f64.wgsl` | hotSpring | Bio-material mechanical properties |
| `md/verlet_neighbor_f64.wgsl` | hotSpring | Bio-molecular force calculations |
| `ml/esn_readout_f64.wgsl` | hotSpring | Environmental time-series |
| `special/fused_kl_divergence_f64.wgsl` | neuralSpring | Cross-entropy testing |
| `special/fused_chi_squared_f64.wgsl` | neuralSpring | Enrichment testing |
| `grid/hargreaves_et0_f64.wgsl` | airSpring | ET₀ evapotranspiration |
| `science/seasonal_pipeline.wgsl` | airSpring | Environmental monitoring |
| `stats/moving_window_f64.wgsl` | airSpring | Sensor stream statistics |
| `reduce/welford_mean_variance_f64.wgsl` | groundSpring | Universal mean+variance |
| `special/chi_squared_f64.wgsl` | groundSpring | Statistical testing |

---

## For barraCuda Team

### Re-export Suggestion

The following types are used by wetSpring but require deep `ops::bio::*` paths:

- `barracuda::ops::bio::hmm::HmmForwardArgs`
- `barracuda::ops::bio::dada2::{Dada2Buffers, Dada2Dimensions, Dada2DispatchArgs}`
- `barracuda::ops::bio::gillespie::GillespieModel`
- `barracuda::device::driver_profile::PrecisionRoutingAdvice`

Consider re-exporting these from `barracuda::` or `barracuda::device::` for
spring convenience (same pattern as `HmmBatchForwardF64`, `GillespieGpu`).

### Provenance Registry

wetSpring's Exp312 binary validates the registry and confirms:
- 28 shaders tracked, all with created+absorbed dates
- 22 are cross-spring (consumed by spring ≠ origin)
- 10 evolution timeline events with bidirectional flows
- 5×5 dependency matrix is non-trivial

---

## For toadStool Team

wetSpring has zero remaining local WGSL — fully lean. The provenance module
can now generate runtime reports for toadStool's cross-spring tracking
(`toadStool::provenance` namespace).

---

## Next Priorities

| Priority | Task | Status |
|----------|------|--------|
| P1 | `BatchedOdeRK45F64` adoption (adaptive RK45 GPU) | Ready — struct in barraCuda |
| P2 | `mean_variance_to_buffer()` for chained GPU stats | Ready — API in barraCuda |
| P3 | coralReef shader compilation proxy test | Waiting for IPC integration |
