# Absorption Manifest: wetSpring → ToadStool/BarraCuda

**Date:** February 22, 2026
**Pattern:** Write → Absorb → Lean (adopted from hotSpring)
**Status:** 28 absorbed + 4 fully leaned (32 total), 0 local WGSL shaders, 7 Tier A candidates

---

## Methodology: Write → Absorb → Lean

wetSpring follows hotSpring's proven absorption cycle:

```
Write → Validate → Hand off → Absorb → Lean
─────────────────────────────────────────────
Implement     Test against    Document in    ToadStool adds    Rewire to
on CPU +      Python +        wateringHole/  shaders/ops       upstream,
WGSL          known physics   handoffs/                        delete local
```

### Principles

1. **Biome model** — Springs don't import each other. wetSpring, hotSpring,
   neuralSpring each lean on ToadStool/BarraCuda independently. ToadStool
   absorbs what works; all Springs benefit.

2. **Shaders in `.wgsl` files** — `include_str!` for large, `pub const` for small.
   No opaque blobs. Every shader has documented binding layouts and dispatch geometry.

3. **CPU reference first** — Every GPU/NPU path has a validated CPU baseline
   (Python → Rust CPU → Rust GPU → streaming).

4. **Handoffs via wateringHole** — Structured documents with code locations,
   binding indices, workgroup sizes, test results, and tolerance rationale.

5. **Backward-compatible lean** — Re-exports and type aliases (not breaking changes)
   when switching from local to upstream.

6. **Named tolerances** — Central constants in `tolerances.rs`, not magic numbers.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Write | Local WGSL shaders for Tier A modules | **4 shaders** (ODE, kmer, unifrac, taxonomy) |
| Validate | CPU ↔ GPU parity for all shaders | ODE: 7/7; others: pending |
| Hand off | wateringHole/handoffs/ documents | v8 active, 6 archived |
| Absorb | ToadStool integrates as `ops::bio::*` | **24 primitives** absorbed |
| Lean | Rewire to upstream, delete local code | 24 primitives lean (19 wetSpring + 5 neuralSpring) |

---

## Absorbed (Lean Phase — 24 primitives)

These modules consume upstream ToadStool/BarraCuda primitives.
Local WGSL deleted; wetSpring imports from `barracuda::*`.

| wetSpring Module / Primitive | Upstream Primitive | Absorbed Date | Exp |
|------------------|-------------------|---------------|-----|
| `alignment` | `SmithWatermanGpu` | Feb 20 | 044 |
| `ani_gpu` | `AniBatchF64` | Feb 22 | 058 |
| `dada2_gpu` | `Dada2EStepGpu` | Feb 22 | — |
| `decision_tree` | `TreeInferenceGpu` | Feb 20 | 044 |
| `diversity_gpu` | `FusedMapReduceF64`, `BrayCurtisF64` | Feb 22 | 016 |
| `dnds_gpu` | `DnDsBatchF64` | Feb 22 | 058 |
| `eic_gpu` | `FusedMapReduceF64`, `WeightedDotF64` | Feb 22 | 087 |
| `felsenstein` | `FelsensteinGpu` | Feb 20 | 046 |
| `gemm_cached` | `GemmF64::WGSL` | Feb 20 | 016 |
| `gillespie` | `GillespieGpu` | Feb 20 | 044 |
| `hmm_gpu` | `HmmBatchForwardF64` | Feb 22 | 047 |
| `kriging` | `KrigingF64` (via GEMM) | Feb 22 | 087 |
| `pangenome_gpu` | `PangenomeClassifyGpu` | Feb 22 | 058 |
| `pcoa_gpu` | `BatchedEighGpu` | Feb 22 | 087 |
| `quality_gpu` | `QualityFilterGpu` | Feb 22 | — |
| `random_forest_gpu` | `RfBatchInferenceGpu` | Feb 22 | 063 |
| `rarefaction_gpu` | `FusedMapReduceF64`, `PrngXoshiro` | Feb 22 | 087 |
| `snp_gpu` | `SnpCallingF64` | Feb 22 | 058 |
| `spectral_match_gpu` | `FusedMapReduceF64`, `GemmF64` | Feb 22 | 016 |
| (cross-spring) | `PairwiseHammingGpu` | Feb 22 | 094 |
| (cross-spring) | `PairwiseJaccardGpu` | Feb 22 | 094 |
| (cross-spring) | `SpatialPayoffGpu` | Feb 22 | 094 |
| (cross-spring) | `BatchFitnessGpu` | Feb 22 | 094 |
| (cross-spring) | `LocusVarianceGpu` | Feb 22 | 094 |

---

## Local WGSL Shaders (Write Phase — 4 shaders)

Shaders written locally, pending ToadStool absorption.

| Shader | File | Domain | GPU Checks | Blocker |
|--------|------|--------|:----------:|---------|
| `batched_qs_ode_rk4_f64.wgsl` | `src/shaders/` | QS/c-di-GMP ODE | 7 (Exp049) | Upstream uses `compile_shader` not `compile_shader_f64` |
| `kmer_histogram_f64.wgsl` | `src/shaders/` | K-mer counting | pending | Needs validation binary |
| `unifrac_propagate_f64.wgsl` | `src/shaders/` | UniFrac distance | pending | Multi-pass tree levels |
| `taxonomy_fc_f64.wgsl` | `src/shaders/` | Taxonomy scoring | pending | NPU int8 variant |

### Shader Conventions (matching hotSpring)

- File: `src/shaders/<domain>_<op>_f64.wgsl`
- Entry point: `@compute @workgroup_size(N) fn <name>(...)`
- Binding layout documented in shader header
- CPU reference documented in shader header
- `f64()` explicit casts (naga type promotion bug)
- `pow_f64()` polyfill for Ada Lovelace (injected by `ShaderTemplate::for_driver_auto`)

---

## Tier A Candidates (7 modules — GPU/NPU ready)

| Module | Domain | Flat API | WGSL Shader | Priority | Notes |
|--------|--------|:--------:|:-----------:|:--------:|-------|
| `ode` | RK4 integrator | ✅ | ✅ `batched_qs_ode_rk4_f64.wgsl` | P1 | Blocked on ToadStool |
| `qs_biofilm` | QS/c-di-GMP | ✅ | ✅ (shares ODE shader) | P1 | Same shader, different params |
| `multi_signal` | 7-var ODE | ✅ | Maps to ODE sweep | P2 | Flat API via Exp078 |
| `phage_defense` | CRISPR/RM ODE | ✅ | Maps to ODE sweep | P2 | Flat API via Exp078 |
| `kmer` | K-mer histogram | ✅ | ✅ `kmer_histogram_f64.wgsl` | P2 | 4^k flat buffer |
| `unifrac` | UniFrac distance | ✅ | ✅ `unifrac_propagate_f64.wgsl` | P2 | CSR flat tree |
| `taxonomy` | Naive Bayes | ✅ | ✅ `taxonomy_fc_f64.wgsl` | P3 | GPU f64 + NPU int8 |

### Tier B (1 module — needs refactoring)

| Module | Domain | Status | Notes |
|--------|--------|--------|-------|
| `cooperation` | Game theory QS | Flat API (Exp078) | Maps to ODE sweep once ODE blocker clears |

---

## CPU Math Extraction Candidates

Local Rust implementations that duplicate barracuda upstream. Pending
`barracuda [features] math = []` for lean migration.

| Local Function | File | Upstream Target | Status |
|----------------|------|-----------------|--------|
| `erf()` | `bio/special.rs` | `barracuda::special::erf` | Shaped, FMA-optimized |
| `ln_gamma()` | `bio/special.rs` | `barracuda::special::ln_gamma` | Lanczos, Horner form |
| `regularized_gamma_lower()` | `bio/special.rs` | `barracuda::special::regularized_gamma_p` | Series expansion |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` | Trapezoidal rule |

**Blocker:** barracuda requires wgpu+akida+toadstool-core as mandatory deps.
Proposed: `[features] math = []` gates CPU-only modules without GPU stack.

---

## metalForge Forge Crate

The `metalForge/forge/` crate (`wetspring-forge` v0.2.0) provides:

| Module | Purpose | Absorption Path |
|--------|---------|-----------------|
| `probe` | GPU (wgpu) + CPU (/proc) + NPU (/dev) discovery | `barracuda::device::discovery` |
| `inventory` | Unified substrate inventory | `barracuda::device::inventory` |
| `dispatch` | Capability-based workload routing | `barracuda::dispatch` |
| `bridge` | forge ↔ barracuda device bridge | Integration seam |

**Bridge docstring:** "When ToadStool absorbs forge, this bridge becomes the
integration point — substrate discovery feeds directly into device creation."

---

## New Upstream Primitives Available (ToadStool Session 39)

5 neuralSpring-evolved bio primitives are now absorbed and consumed by
wetSpring (Exp094, 39 checks PASS). See "neuralSpring-evolved (5)" above.

### ToadStool ODE Status (Session 39)

`BatchedOdeRK4F64` exists in `ops::batched_ode_rk4` with the correct API
shape (5 vars, 17 params, same as wetSpring). The `enable f64;` directive
was removed from the WGSL shader (line 35 is now a comment). **However**,
`batched_ode_rk4.rs:209` calls `compile_shader()` instead of
`compile_shader_f64()`, which means the f64 preamble is not injected.
Without this, the shader fails on naga/Vulkan backends.

**Feedback for ToadStool**: change line 209 from `dev.compile_shader(...)` to
`dev.compile_shader_f64(...)` (or use `ShaderTemplate::for_driver_auto`).
Once fixed, wetSpring's `ode_sweep_gpu.rs` becomes a thin wrapper.

---

## Cross-Spring Contributions

Patterns from hotSpring and neuralSpring that wetSpring leans on:

| Contribution | Source | wetSpring Usage |
|-------------|--------|----------------|
| `ShaderTemplate::for_driver_auto` | hotSpring NVK workaround | All f64 WGSL compilation |
| `GpuDriverProfile` | hotSpring device capabilities | Driver-specific polyfill selection |
| `ReduceScalarPipeline` | hotSpring pipeline feedback | FMR dispatch optimization |
| `BatchedEighGpu` (NAK-optimized) | hotSpring eigensolve | PCoA ordination |
| `(zero + literal)` f64 pattern | neuralSpring naga fix | All f64 shader constants |
| `GemmCached` 60× speedup | wetSpring → absorbed | hotSpring HFB uses this |

---

## Validation Summary

| Category | Checks | Status |
|----------|:------:|--------|
| CPU parity (Python → Rust) | 1,392 | ALL PASS |
| GPU parity (CPU → GPU) | 609 | ALL PASS |
| Streaming dispatch | 80 | ALL PASS |
| Layout fidelity (Tier A) | 35 | ALL PASS |
| Transfer/streaming | 57 | ALL PASS |
| **Total** | **2,219+** | **ALL PASS** |
