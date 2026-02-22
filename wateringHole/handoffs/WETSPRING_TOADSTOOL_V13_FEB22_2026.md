# wetSpring → ToadStool/BarraCuda Handoff v13

**Date:** February 22, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-or-later
**Context:** Phase 25 — ToadStool absorption Lean phase, BarraCuda rename, 4 local WGSL retired

---

## Executive Summary

wetSpring has completed **97 experiments, 2,229+ validation checks, 740 Rust
tests, and 87 binaries** — all passing. **All 4 local WGSL shaders retired** —
ToadStool Sessions 39-42 absorbed every wetSpring shader, and this handoff
completes the Lean phase by rewiring to upstream ops. Zero local WGSL files
remain. BarraCuda rename (S42) applied across all active code and docs.

**New in v13 (since v12):**

1. **Full Lean phase for 4 WGSL shaders** — `batched_qs_ode_rk4_f64.wgsl`,
   `kmer_histogram_f64.wgsl`, `taxonomy_fc_f64.wgsl`, `unifrac_propagate_f64.wgsl`
   all retired; wetSpring now delegates to ToadStool upstream ops
2. **ODE sweep rewired** — `ode_sweep_gpu.rs` is now a thin wrapper around
   `barracuda::ops::BatchedOdeRK4F64` (ToadStool S41 `compile_shader_f64` fix)
3. **Exp096 rewired** — `validate_local_wgsl_compile` now exercises upstream
   `KmerHistogramGpu`, `TaxonomyFcGpu`, `UniFracPropagateGpu`, `BatchedOdeRK4F64`
4. **BarraCUDA → BarraCuda** — display name aligned with ToadStool S42 across
   42 `.rs` files and 20 active `.md` files; archive fossils left unchanged
5. **Rust 2024 pattern fixes** — `random_forest_gpu.rs`, `validate_gpu_track1c.rs`
   updated for edition 2024 implicit-borrow pattern rules
6. **740 tests pass** — zero regressions after rewire

---

## Part 1: What Changed — Write → Absorb → Lean Complete

### Shaders Retired (4 → 0 local WGSL)

| Local Shader | ToadStool Op | Session | Lean Status |
|-------------|--------------|---------|-------------|
| `batched_qs_ode_rk4_f64.wgsl` | `BatchedOdeRK4F64` | S41 (f64 fix) | ✅ Thin wrapper in `ode_sweep_gpu.rs` |
| `kmer_histogram_f64.wgsl` | `KmerHistogramGpu` | S39 | ✅ Validated via Exp096 |
| `taxonomy_fc_f64.wgsl` | `TaxonomyFcGpu` | S39 | ✅ Validated via Exp096 |
| `unifrac_propagate_f64.wgsl` | `UniFracPropagateGpu` | S39 | ✅ Validated via Exp096 |

### ODE Sweep Rewire Detail

Before (v12):
```
ode_sweep_gpu.rs → include_str!("batched_qs_ode_rk4_f64.wgsl")
                 → ShaderTemplate::for_driver_auto (manual f64 preamble)
                 → dev.compile_shader (manual pipeline setup)
                 → manual buffer management + readback
```

After (v13):
```
ode_sweep_gpu.rs → barracuda::ops::BatchedOdeRK4F64::new(device, config)
                 → integrator.integrate(initial_states, batch_params)
                 → upstream handles f64 preamble, pipeline, buffers, readback
```

Net: -155 lines of manual GPU plumbing, +15 lines of wrapper code.

---

## Part 2: ToadStool Evolution Observed (S25-S42)

wetSpring reviewed ToadStool commits d45fdfb3..5437c170 (Sessions 25-42):

| Session | Key Additions | Impact on wetSpring |
|---------|---------------|---------------------|
| S25 | FFT f64, error system debt | Infrastructure improvement |
| S27 | wetSpring v5 absorption, 16 new WGSL | Bio primitives expanded |
| S31a-h | Orphan shader wiring, RF inference, clippy clean | Quality |
| S39 | **3 wetSpring WGSL absorbed** (kmer, taxonomy, unifrac), FlatTree, math module | **Direct enabler** |
| S40 | Richards PDE, moving window stats | New capabilities |
| S41 | **6 f64 shader compile fixes**, APIs exposed for Springs | **ODE blocker resolved** |
| S42 | BarraCUDA → BarraCuda rename, 19 new WGSL | Naming alignment |

### New ToadStool Primitives Available

These are now available at `barracuda::` crate root (25 bio ops total):

```
AniBatchF64, BatchFitnessGpu, Dada2EStepGpu, DnDsBatchF64, FelsensteinGpu,
FlatTree, GillespieGpu, HillGateGpu, HmmBatchForwardF64, KmerHistogramGpu,
LocusVarianceGpu, MultiObjFitnessGpu, PairwiseHammingGpu, PairwiseJaccardGpu,
PairwiseL2Gpu, PangenomeClassifyGpu, QualityFilterGpu, RfBatchInferenceGpu,
SmithWatermanGpu, SnpCallingF64, SpatialPayoffGpu, TaxonomyFcGpu,
TreeInferenceGpu, UniFracPropagateGpu
```

New math module: `barracuda::math::{erf, erfc, gamma, ln_gamma, digamma, beta, ln_beta, regularized_gamma_p, regularized_gamma_q}`

---

## Part 3: BarraCuda Rename

ToadStool S42 renamed the display name from "BarraCUDA" to "BarraCuda".
Scope: documentation/branding only — the crate name remains `barracuda` (lowercase).

wetSpring changes:
- 42 `.rs` files: comments and docstrings updated
- 20 active `.md` files: prose updated
- Archive handoffs and historical experiments: left as fossils

---

## Part 4: Metrics After Rewire

| Metric | v12 | v13 |
|--------|-----|-----|
| Local WGSL shaders | 4 (Write phase) | **0** (all retired) |
| ToadStool primitives consumed | 28 | **32** (28 + 4 newly leaned) |
| Rust tests | 740 | 740 |
| Validation checks | 2,229+ | 2,229+ |
| `BarraCUDA` references (active code) | ~200 | **0** |
| `BarraCuda` references (active code) | 0 | ~200 |
| GPU compilation | ✅ | ✅ |
| CPU compilation | ✅ | ✅ |

---

## Part 5: Next Steps

### For ToadStool/BarraCuda team:
1. **Confirm `FlatTree` compatibility** — wetSpring's CSR tree matches ToadStool's `FlatTree`; verify field names
2. **Consider `barracuda::math` as CPU fallback** — currently gated behind GPU feature; wetSpring's `crate::special` fills this gap for CPU-only builds
3. **UniFracConfig re-export** — `UniFracConfig` is not at crate root; consider adding to bio re-exports

### For wetSpring:
1. **Wire `KmerHistogramGpu` into production kmer path** — currently CPU-only; ToadStool op ready
2. **Wire `UniFracPropagateGpu` into production unifrac path** — currently CPU-only; ToadStool op ready
3. **Wire `TaxonomyFcGpu` as alternative to GemmF64** — taxonomy_gpu.rs currently uses GemmF64; TaxonomyFcGpu may be more direct
4. **Use `barracuda::FlatTree`** instead of local CSR tree structs where applicable

---

## Appendix: File Changes

### Deleted
- `barracuda/src/shaders/batched_qs_ode_rk4_f64.wgsl` (4,555 bytes)
- `barracuda/src/shaders/kmer_histogram_f64.wgsl` (1,263 bytes)
- `barracuda/src/shaders/taxonomy_fc_f64.wgsl` (2,289 bytes)
- `barracuda/src/shaders/unifrac_propagate_f64.wgsl` (2,867 bytes)

### Rewritten
- `barracuda/src/bio/ode_sweep_gpu.rs` — thin wrapper around `BatchedOdeRK4F64`
- `barracuda/src/bin/validate_local_wgsl_compile.rs` — validates ToadStool upstream ops

### Fixed (Rust 2024 patterns)
- `barracuda/src/bio/random_forest_gpu.rs`
- `barracuda/src/bin/validate_gpu_track1c.rs`

### Renamed (BarraCUDA → BarraCuda)
- 42 `.rs` files, 20 active `.md` files
