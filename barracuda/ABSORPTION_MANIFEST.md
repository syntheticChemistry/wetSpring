# Absorption Manifest: wetSpring → ToadStool/BarraCuda

**Date:** February 25, 2026 (V37 revalidation)
**Pattern:** Write → Absorb → Lean (adopted from hotSpring)
**ToadStool pin:** `02207c4a` (S62+DF64 expansion, Feb 24 2026)
**Status:** 44 ToadStool primitives + 2 BGL helpers consumed, 1 local WGSL extension (Write phase — diversity fusion), 5 GPU ODE modules use trait-generated WGSL via `BatchedOdeRK4<S>::generate_shader()`, 42 GPU modules + 1 Write-phase extension, 0 Tier B/C remaining, 806 tests (759 barracuda + 47 forge), 95.75% library coverage, ToadStool S62+DF64 aligned, 167 experiments, 3,279+ checks

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
| Write | Local WGSL ODE shaders | **5 shaders deleted** — GPU modules use `generate_shader()` from `OdeSystem` trait impls (`bio/ode_systems.rs`) |
| Compose | GPU wrappers wiring ToadStool primitives | **7 modules** (kmd, merge_pairs, RF, derep, NJ, reconciliation, molecular_clock) |
| Passthrough | Accept GPU buffers, CPU kernel | **3 modules** (gbm, feature_table, signal) |
| Validate | CPU ↔ GPU parity for all shaders | All 5 ODE: exact parity (Exp099/100/101) |
| Hand off | wateringHole/handoffs/ documents | **V37** active (revalidation + deep debt cleanup), V7-V36 archived |
| Absorb | ToadStool integrates as `ops::bio::*` | **44 items** absorbed (ToadStool S62+DF64: all DONE, +46 cross-spring total) |
| Lean | Rewire to upstream, delete local code | 44 primitives + 2 BGL helpers lean, 5 `OdeSystem` trait rewires, BGL boilerplate removed |

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

## Local WGSL Shaders (Write Phase — LEAN COMPLETE)

All 5 ODE shaders have been **deleted** (30,424 bytes). GPU modules now use
`BatchedOdeRK4::<S>::generate_shader()` at runtime, producing WGSL from the
`OdeSystem` trait implementations in `bio/ode_systems.rs`.

| System | Struct | Vars | Params | WGSL Lines | CPU Parity |
|--------|--------|:----:|:------:|:----------:|:----------:|
| Phage Defense | `PhageDefenseOde` | 4 | 11 | 142 | Derivative-level exact |
| Bistable | `BistableOde` | 5 | 21 | 169 | Exact (0.00) |
| Multi-Signal | `MultiSignalOde` | 7 | 24 | 199 | Exact (4.44e-16) |
| Cooperation | `CooperationOde` | 4 | 13 | 148 | Exact (4.44e-16) |
| Capacitor | `CapacitorOde` | 6 | 16 | 170 | Exact (0.00) |

### GPU Module Pattern (post-lean)

```rust
let wgsl = BatchedOdeRK4::<XxxOde>::generate_shader();
let module = device.compile_shader_f64(&wgsl, Some("Xxx ODE"));
// Pipeline setup unchanged, dispatch_workgroups(n.div_ceil(64), 1, 1)
```

---

## Compose Phase (7 GPU Wrappers)

GPU wrappers that wire existing ToadStool primitives for GPU-accelerated
workflows. No local WGSL needed — these compose upstream ops.

| Module | ToadStool Primitive | Strategy | Exp |
|--------|-------------------|----------|-----|
| `kmd_gpu` | `KmerHistogramGpu` | Kendrick mass defect via k-mer histogram | 101 |
| `merge_pairs_gpu` | `PairwiseHammingGpu` | Overlap scoring via Hamming distance | 101 |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Bipartition distance via Hamming | 101 |
| `derep_gpu` | `KmerHistogramGpu` | Sequence hashing via k-mer histogram | 101 |
| `neighbor_joining_gpu` | `GemmCachedF64` | Distance matrix operations | 101 |
| `reconciliation_gpu` | `TreeInferenceGpu` | DTL cost inference via tree traversal | 101 |
| `molecular_clock_gpu` | `GemmCachedF64` | Rate matrix operations | 101 |

---

## Passthrough Phase (3 GPU Wrappers)

Accept GPU buffers but run CPU kernels. Pending ToadStool primitives for
full GPU dispatch.

| Module | CPU Kernel | Needed Primitive | Exp |
|--------|-----------|-----------------|-----|
| `gbm_gpu` | Sequential boosting | `GbmBatchInferenceGpu` | 101 |
| `feature_table_gpu` | Feature extraction pipeline | `FeatureExtractionGpu` | 101 |
| `signal_gpu` | Peak detection (1D) | `PeakDetectGpu` | 101 |

---

## Tier B/C — All Promoted (Phase 28)

All 13 former Tier B/C modules have been promoted to GPU-capable:

| Former Tier | Module | Promoted To | Notes |
|-------------|--------|-------------|-------|
| B | `cooperation` | Write (local WGSL 4v/13p) | ODE RK4 f64 shader |
| C | `capacitor` | Write (local WGSL 6v/16p) | ODE RK4 f64 shader |
| C | `kmd` | Compose (`KmerHistogramGpu`) | Kendrick mass defect |
| C | `gbm` | Passthrough | Sequential boosting |
| C | `merge_pairs` | Compose (`PairwiseHammingGpu`) | Overlap merging |
| C | `signal` | Passthrough | Peak detection |
| C | `feature_table` | Passthrough | Feature extraction |
| C | `robinson_foulds` | Compose (`PairwiseHammingGpu`) | Tree distance |
| C | `derep` | Compose (`KmerHistogramGpu`) | Dereplication |
| C | `chimera` | Compose (`GemmCachedF64`) | Chimera scoring (upgraded) |
| C | `neighbor_joining` | Compose (`GemmCachedF64`) | NJ tree construction |
| C | `reconciliation` | Compose (`TreeInferenceGpu`) | DTL reconciliation |
| C | `molecular_clock` | Compose (`GemmCachedF64`) | Molecular clock |

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
| `cholesky_factor()` | `bio/esn.rs` | `barracuda::linalg::cholesky_solve` | SPD system solve (ridge regression, kriging, GP) |
| `solve_ridge()` | `bio/esn.rs` | `barracuda::linalg::ridge_regression` | Cholesky-based ridge with flat buffer layout |

**Blocker:** barracuda requires wgpu+akida+toadstool-core as mandatory deps.
Proposed: `[features] math = []` gates CPU-only modules without GPU stack.

---

## metalForge Forge Crate

The `metalForge/forge/` crate (`wetspring-forge` v0.3.0, 47 tests) provides:

| Module | Purpose | Absorption Path |
|--------|---------|-----------------|
| `probe` | GPU (wgpu) + CPU (/proc) + NPU (/dev) discovery | `barracuda::device::discovery` |
| `inventory` | Unified substrate inventory | `barracuda::device::inventory` |
| `dispatch` | Capability-based workload routing | `barracuda::dispatch` |
| `bridge` | forge ↔ barracuda device bridge | Integration seam |

**Bridge docstring:** "When ToadStool absorbs forge, this bridge becomes the
integration point — substrate discovery feeds directly into device creation."

---

## New Upstream Primitives Available (ToadStool S54-S57)

### S54 Primitives (neuralSpring baseCamp → ToadStool)

| Primitive | Module | Tests | Potential wetSpring Use |
|-----------|--------|:-----:|------------------------|
| `graph_laplacian` | `barracuda::linalg` | 3 | Community network analysis |
| `effective_rank` | `barracuda::linalg` | 3 | Spectral diagnostics |
| `numerical_hessian` | `barracuda::numerical` | 3 | Optimization landscape |

5 new WGSL shaders: `symmetrize.wgsl`, `laplacian.wgsl` (linalg),
`hessian_column.wgsl` (numerical), `histogram.wgsl` (stats),
`metropolis.wgsl` (sample).

GPU fixes from airSpring also landed: `pow_f64` fractional exponent,
`acos_simple` → `acos_f64`, `FusedMapReduceF64` buffer conflict resolution.

### S56 Primitives (neuralSpring → ToadStool)

| Primitive | Module | Tests | Potential wetSpring Use |
|-----------|--------|:-----:|------------------------|
| `belief_propagation_chain` | `barracuda::linalg::graph` | 3 | Chain PGM forward pass |
| `boltzmann_sampling` | `barracuda::sample::metropolis` | 3 | CPU MCMC sampling |
| `disordered_laplacian` | `barracuda::linalg::graph` | 3 | Anderson disorder — **directly relevant to QS-disorder coupling** |

### S57 Status

4,224 core tests, 650+ WGSL shaders, 46 cross-spring absorption items
complete. All root docs synced. 222 lines commented-out code removed.

### wetSpring Consumption of S54-S56 Primitives (Exp162)

| Primitive | Module | Session | wetSpring Use | Exp |
|-----------|--------|---------|---------------|-----|
| `graph_laplacian` | `barracuda::linalg` | S54 | Community network spectral analysis | 162 |
| `effective_rank` | `barracuda::linalg` | S54 | Diversity matrix diagnostics | 162 |
| `numerical_hessian` | `barracuda::numerical` | S54 | ML model curvature analysis | 162 |
| `disordered_laplacian` | `barracuda::linalg` | S56 | QS-disorder coupling on community graphs | 162 |
| `belief_propagation_chain` | `barracuda::linalg` | S56 | Hierarchical taxonomy classification | 162 |
| `boltzmann_sampling` | `barracuda::sample` | S56 | MCMC parameter optimization | 162 |

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
| CPU parity (Python → Rust) | 1,476 | ALL PASS |
| GPU parity (CPU → GPU) | 702+ | ALL PASS |
| BarraCuda CPU parity (v1-v8) | 380/380 | ALL PASS |
| Streaming dispatch | 80 | ALL PASS |
| Layout fidelity | 35 | ALL PASS |
| Transfer/streaming | 57 | ALL PASS |
| Finite-size scaling (Exp150) | 14 | ALL PASS |
| Correlated disorder (Exp151) | 8 | ALL PASS |
| ODE lean benchmark | 11 | ALL PASS |
| Paper queue extensions (Exp152-156) | 42 | ALL PASS |
| Drug repurposing track (Exp157-161) | 40 | ALL PASS |
| **Total** | **3,132+** | **ALL PASS** |

**Phase 39 additions (Feb 24, 2026):**
- Exp150: W_c = 16.26 (disorder-averaged, L=6–12, 8 realizations)
- Exp151: Correlated disorder pushes W_c > 28 (biofilm clustering facilitates QS)
- Exp152: Physical communication pathways — 8 modes, 6/8 Anderson-susceptible (9 checks)
- Exp153: Nitrifying QS — R:P = 2.3:1 matches eavesdropper prediction (12 checks)
- Exp154: Marine interkingdom QS — 10/10 Anderson predictions correct (6 checks)
- Exp155: Myxococcus — critical density + NP geometry bootstrap (7 checks)
- Exp156: Dictyostelium — non-Hermitian relay defeats localization (8 checks)
- Exp157-161: Drug repurposing track — NMF, pathway scoring, KG embedding (40 checks)
  - New local module: `bio::nmf` (Lee & Seung multiplicative updates, cosine sim, top-K)
  - ToadStool absorption targets: NMF update shader, sparse GEMM, weighted NMF mask
- ODE lean benchmark: 5 systems × generate_shader(), upstream 20–33% faster

**V37 revalidation (Feb 25, 2026):**
- Revalidated against ToadStool `02207c4a` (S62+DF64 expansion)
- 806 tests pass (759 barracuda + 47 forge), 0 failures
- `cargo clippy --all-targets` clean (pedantic + nursery)
- 95.75% library coverage (llvm-cov)
- Deep debt cleanup: ncbi.rs sovereignty (capability-based HTTP), I/O parser lean
  (deprecated buffering removed), 56 binaries modernized (NaN-safe sorts, descriptive expects),
  tolerance provenance (5 constants with commit hashes), CI expanded (coverage gate + forge jobs)
