# wetSpring → ToadStool Handoff V36: Write-Phase Extensions + Full Absorption Accounting

**Date:** February 25, 2026
**From:** wetSpring (Phase 44, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-or-later
**Purpose:** Comprehensive handoff covering wetSpring's first Write-phase WGSL extension
(following hotSpring's absorption pattern), full absorption accounting, lessons learned
relevant to barracuda evolution, and recommendations for the ToadStool team.

---

## Executive Summary

wetSpring has reproduced **43 scientific papers** across microbial ecology, phylogenetics,
analytical chemistry, deep-sea metagenomics, and drug repurposing — all on consumer
hardware (RTX 4070 + AKD1000 NPU). The BarraCuda crate provides the compute substrate;
wetSpring wraps it with bio-specific modules, validation harnesses, and tolerance frameworks.

**Key numbers:** 812 tests, 167 experiments, 3,279+ validation checks, 157 binaries,
44 ToadStool primitives + 2 BGL helpers consumed, 1 new WGSL extension written,
0 clippy warnings, 0 unsafe blocks. All 30 actionable papers have full three-tier
validation (CPU, GPU, metalForge).

wetSpring is now entering the **Write phase** for new bio-specific GPU shaders —
following hotSpring's proven pattern of writing absorbable extensions that ToadStool
can integrate as `ops::bio::*` primitives.

---

## Part 1: What wetSpring Contributed Back to BarraCuda

### Already Absorbed (Confirmed in ToadStool S58-S62)

| Contribution | ToadStool Session | Impact |
|-------------|------------------|--------|
| 5 bio ODE systems (phage, bistable, multi-signal, cooperation, capacitor) | S58 | `BatchedOdeRK4` trait + 5 `OdeSystem` impls → generic GPU ODE sweeps |
| NMF (multiplicative update, Lee & Seung 1999) | S58 | `barracuda::linalg::nmf` — drug repurposing matrices |
| Ridge regression (Cholesky-based) | S59 | `barracuda::linalg::ridge` — ESN readout training |
| Trapezoidal integration (`trapz`) | S59 | `barracuda::numerical::trapz` — numerical quadrature |
| `erf` / `ln_gamma` / `normal_cdf` special functions | S59 | `barracuda::special` — statistical computing |
| Correlated Anderson disorder | S59 | `barracuda::spectral::anderson::correlated_disorder` |
| `GemmCached` → `GemmCachedF64` pattern | S60 | Cached GEMM pipeline for repeated dispatches |
| TransE knowledge graph scoring | S60 | `barracuda::ops::transe_score_f64` — drug repurposing KGs |
| Sparse GEMM (CSR × Dense) | S60 | `barracuda::ops::sparse_gemm_f64` — sparse drug-disease matrices |
| PeakDetect f64 | S62 | `barracuda::ops::peak_detect_f64` — GPU parallel local maxima |
| BGL helper adoption feedback | S62+DF64 | `storage_bgl_entry`/`uniform_bgl_entry` validated in 6 modules |

### New Write-Phase Extension (Ready for Absorption)

| Module | WGSL Shader | Binding Layout | Tests | Validation |
|--------|-------------|---------------|:-----:|------------|
| `diversity_fusion_gpu` | `shaders/diversity_fusion_f64.wgsl` | 3 bindings: uniform(params) + storage(abundances) + storage(results) | 6 CPU + Exp167 GPU | 18/18 checks PASS |

**What it does:** Fused Shannon entropy + Simpson index + Pielou evenness in a single
GPU dispatch. Each thread processes one sample (all species), computing all three
diversity metrics in one pass. Replaces three separate `FusedMapReduceF64` dispatches.

**Binding layout:**

| Binding | Type | Content |
|---------|------|---------|
| 0 | uniform | `{ n_samples: u32, n_species: u32 }` |
| 1 | storage, read | `abundances: array<f64>` (`n_samples` × `n_species`) |
| 2 | storage, read_write | `results: array<f64>` (`n_samples` × 3) |

**Dispatch geometry:** `ceil(n_samples / 64)` workgroups × 1 × 1, `@workgroup_size(64)`.

**Absorption target:** `ops::bio::diversity_fusion` — benefits ecology (wetSpring),
environmental monitoring (airSpring), and any domain computing community diversity metrics.

---

## Part 2: How wetSpring Uses BarraCuda — Complete API Map

### Always-On CPU Primitives (no `gpu` feature required)

| Primitive | Module | Where Used | Checks |
|-----------|--------|-----------|:------:|
| `erf` | `barracuda::special::erf` | Statistics, normal distribution | 6+ |
| `ln_gamma` | `barracuda::special::ln_gamma` | Gamma functions, Chao1 | 4+ |
| `normal_cdf` | `barracuda::special::normal_cdf` | Confidence intervals | 2+ |
| `trapz` | `barracuda::numerical::trapz` | EIC peak integration | 8+ |
| `ridge_regression` | `barracuda::linalg::ridge` | ESN readout training | 13+ |
| `nmf` | `barracuda::linalg::nmf` | Drug repurposing NMF | 16+ |
| `dot`, `l2_norm` | `barracuda::special` | Spectral cosine similarity | 40+ |
| `correlated_disorder` | `barracuda::spectral::anderson` | QS-disorder analogy | 25+ |

### GPU Primitives (44 total, all via `barracuda::ops::*`)

| Primitive | Usage Pattern | Modules Using It |
|-----------|--------------|-----------------|
| `FusedMapReduceF64` | Shannon, Simpson, Bray-Curtis, KMD, merge scoring | 8 GPU modules |
| `BrayCurtisF64` | Pairwise community dissimilarity | `diversity_gpu` |
| `BatchedEighGpu` | PCoA eigendecomposition, bifurcation analysis | `pcoa_gpu`, `ode_sweep_gpu` |
| `BatchedOdeRK4<S>` | 5 bio ODE systems via `OdeSystem` trait | 5 ODE GPU modules |
| `SmithWatermanGpu` | Pairwise local alignment | `alignment` → GPU |
| `FelsensteinGpu` | Phylogenetic likelihood, bootstrap, placement | 3 modules |
| `GillespieGpu` | Stochastic simulation | `gillespie` → GPU |
| `HmmBatchForwardF64` | HMM forward algorithm in log-space | `hmm_gpu` |
| `KmerHistogramGpu` | Dense k-mer counting | `kmer_gpu`, `derep_gpu` |
| `UniFracPropagateGpu` | Phylogenetic UniFrac distances | `unifrac_gpu` |
| `TaxonomyFcGpu` | FC layer classification + NPU int8 | `taxonomy_gpu` |
| `TreeInferenceGpu` | Decision tree / RF / GBM batch inference | 3 ML modules |
| `AniBatchF64` | Average Nucleotide Identity | `ani_gpu` |
| `DnDsBatchF64` | Synonymous/non-synonymous substitution rates | `dnds_gpu` |
| `SnpCallingF64` | SNP variant calling | `snp_gpu` |
| `PangenomeClassifyGpu` | Core/accessory/unique gene classification | `pangenome_gpu` |
| `GemmF64` / `GemmCachedF64` | Matrix multiplication (drug repurposing, chimera) | `gemm_cached`, `chimera_gpu` |
| `TranseScoreF64` | Knowledge graph triple scoring | `validate_kg_embedding` |
| `SparseMmF64` | CSR × Dense for sparse drug-disease matrices | `validate_gpu_drug_repurposing` |
| `PeakDetectF64` | GPU parallel local maxima + prominence | `validate_gpu_drug_repurposing` |
| `storage_bgl_entry` / `uniform_bgl_entry` | BGL boilerplate helpers | 6 GPU modules |

### Infrastructure Primitives

| Primitive | Usage |
|-----------|-------|
| `WgpuDevice` | All GPU modules |
| `compile_shader_f64` | Shader compilation with f64 polyfills |
| `read_buffer_f64` | GPU → CPU readback |
| `BufferPool` / `PooledBuffer` | GEMM cached pipeline |
| `TensorContext` | GEMM cached pipeline |
| `GpuF64` | Convenient GPU initialization wrapper |

---

## Part 3: Lessons Learned (Relevant to BarraCuda Evolution)

### 3.1 `log_f64` Polyfill Precision

The `compile_shader_f64` pipeline applies `apply_transcendental_workaround` only when
`needs_f64_exp_log_workaround()` returns true (driver-dependent). On Ada (RTX 4070),
this returns false, but native WGSL `log()` is f32-only — calling `log(x)` where `x: f64`
silently narrows to f32, computes, and promotes back, losing ~7 digits of precision.

**Recommendation:** Always use `log_f64()` / `exp_f64()` / `pow_f64()` in f64 shaders,
never the bare WGSL builtins. The `inject_missing_math_f64` path correctly injects
polyfills when these function names are used. Consider making `for_driver_auto` always
replace bare `log(`/`exp(`/`pow(` with polyfill equivalents in f64 shaders, regardless
of driver workaround flag.

**Evidence:** Shannon entropy GPU vs CPU diff was 3.8e-9 with `log_f64()` polyfill,
but ~1e-7 with bare `log()` (f32 truncation). Documented in Exp167.

### 3.2 BGL Helpers Save Massive Boilerplate

Adopting `storage_bgl_entry(binding, read_only)` and `uniform_bgl_entry(binding)` from
`ComputeDispatch` eliminated ~258 lines across 6 files. Every Spring writing GPU modules
should use these instead of manual `BindGroupLayoutEntry` construction.

### 3.3 DF64 Auto-Selection Blocked by Private API

`GemmF64::wgsl_shader_for_device()` is private, preventing downstream crates from
auto-selecting between native f64 and DF64 (double-float f32-pair) GEMM shaders based
on hardware capabilities. wetSpring's `GemmCached` had to hardcode the native path.

**Recommendation:** Make `wgsl_shader_for_device()` public, or provide a
`GemmF64::cached_pipeline(device)` that returns a pre-built pipeline with auto-selected
shader strategy.

### 3.4 PeakDetectF64 WGSL Bug (Still Present)

`shaders/signal/peak_detect_f64.wgsl` line 49: `prominence[idx] = 0.0;` assigns an
f32 literal to an f64 array. Should be `0.0lf` or `f64(0.0)`. wetSpring works around
this with `catch_unwind` in validation code, but the upstream fix is a one-line change.

### 3.5 `OdeSystem` Trait Is Extremely Effective

The `OdeSystem` trait pattern (implement `fn derivatives()` + `fn n_vars()` + `fn n_params()`
→ get `generate_shader()` for free) is the most effective absorption pattern we've seen.
5 bio ODE systems were absorbed with minimal code — just the trait impl. Consider
expanding this pattern to other domain-specific compute (e.g., `ReactionNetwork` trait
for chemical kinetics, `PopulationDynamics` trait for ecology).

### 3.6 Cross-Spring Evolution Benefits

| Origin Spring | Contribution to ToadStool | Downstream Benefit |
|--------------|--------------------------|-------------------|
| **hotSpring** | f64 precision infrastructure, ShaderTemplate, Fp64Strategy, DF64 core-streaming, lattice QCD, spectral module | wetSpring uses compile_shader_f64, BatchedEighGpu, Anderson primitives |
| **wetSpring** | Bio ODE trait system, NMF, ridge, special functions, diversity metrics, drug repurposing primitives | neuralSpring benefits from NMF/ridge; airSpring from diversity metrics |
| **neuralSpring** | PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance | wetSpring uses PairwiseHamming for Robinson-Foulds |
| **ToadStool native** | ComputeDispatch, BGL helpers, unified_hardware, DF64 shaders | All Springs benefit from reduced boilerplate |

---

## Part 4: Absorption Recommendations

### Tier 1 — Ready for Absorption Now

| Module | Source | What It Does | Tests | Priority |
|--------|--------|-------------|:-----:|----------|
| `diversity_fusion_gpu` | `bio/diversity_fusion_gpu.rs` + `bio/shaders/diversity_fusion_f64.wgsl` | Fused Shannon + Simpson + evenness | 6 + Exp167 (18) | **P1** |
| forge `bridge.rs` | `metalForge/forge/src/bridge.rs` | Substrate ↔ WgpuDevice bridge | 5 | P2 |
| forge `dispatch.rs` | `metalForge/forge/src/dispatch.rs` | Capability-based workload routing | 10 | P2 |

### Tier 2 — Structurally Ready, Needs WGSL

| Module | Source | What It Does | Compose Pattern |
|--------|--------|-------------|----------------|
| `gbm_gpu` | `bio/gbm_gpu.rs` | GBM batch inference | Composes TreeInferenceGpu |
| `random_forest_gpu` | `bio/random_forest_gpu.rs` | RF batch inference (SoA) | Composes RfBatchInferenceGpu |
| `eic_gpu` | `bio/eic_gpu.rs` | Extracted ion chromatogram | Composes FMR |
| `streaming_gpu` | `bio/streaming_gpu.rs` | Pipeline composition | Multi-primitive orchestration |

### Bug Fixes Needed Upstream

| Issue | File | Fix |
|-------|------|-----|
| PeakDetectF64 f32 literal | `shaders/signal/peak_detect_f64.wgsl:49` | `0.0` → `0.0lf` |
| `log()` narrowing in f64 shaders | `shaders/precision/mod.rs` | Always replace `log(` → `log_f64(` in f64 context |
| Private `wgsl_shader_for_device()` | `ops/linalg/gemm_f64.rs` | Make `pub` or add `cached_pipeline()` |

---

## Part 5: Full Metrics

| Metric | Value |
|--------|-------|
| Papers reproduced | 43 (5 tracks) |
| Experiments | 167 |
| Validation checks | 3,279+ |
| Rust tests | 812 (765 barracuda + 47 forge) |
| Binaries | 157 (146 validate + 11 benchmark) |
| ToadStool primitives consumed | 44 + 2 BGL helpers |
| Local WGSL extensions (Write phase) | 1 (`diversity_fusion_f64.wgsl`) |
| GPU modules | 42 (all lean) + 1 Write-phase |
| CPU modules | 46 |
| Lines removed (lean) | ~1,500+ (ODE WGSL, dual-path fallback, BGL boilerplate) |
| `cargo clippy --pedantic --nursery` | 0 warnings |
| `cargo fmt` | 0 diffs |
| `#![deny(unsafe_code)]` | Enforced crate-wide |
| Named tolerance constants | 59 |
| External C dependencies | 0 |
| Coverage | 95.67% |
| Three-tier papers (CPU + GPU + metalForge) | 30/30 |

---

## Supersedes

This handoff supersedes V34 and V35 for the purpose of comprehensive absorption
accounting. V34 remains the detailed absorption evolution record; V35 remains the
detailed DF64 lean record.
