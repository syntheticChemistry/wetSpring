# wetSpring — Evolution Readiness

**Date:** February 2026
**Status:** 23 CPU modules, 12 GPU modules, 11 ToadStool primitives validated

---

## Evolution Path

```
Python baseline → Rust CPU validation → GPU acceleration → sovereign pipeline
```

wetSpring consumes BarraCUDA/ToadStool GPU primitives and feeds back
requirements, bug reports, and evolution lessons via unidirectional handoffs.

---

## GPU Promotion Tiers

- **Tier A (Rewire):** GPU shader exists and is validated; wire into pipeline.
- **Tier B (Adapt):** Could use modified ToadStool patterns or small new shaders.
- **Tier C (New):** Needs new ToadStool primitives or substantial new shaders.

### Tier A — Ready to Rewire (10 modules)

| CPU Module | GPU Module | ToadStool Primitives | Checks |
|------------|-----------|---------------------|--------|
| `diversity` | `diversity_gpu` | `FusedMapReduceF64`, `BrayCurtisF64` | 38 |
| `pcoa` | `pcoa_gpu` | `BatchedEighGpu` | — |
| `spectral_match` | `spectral_match_gpu` | `GemmF64`, `FusedMapReduceF64` | — |
| `eic` | `eic_gpu` | `FusedMapReduceF64`, `WeightedDotF64` | — |
| `taxonomy` | `taxonomy_gpu` | `GemmF64` | — |
| `dada2` | `dada2_gpu` | Custom `dada2_e_step.wgsl` | — |
| `quality` | `quality_gpu` | Custom `quality_filter.wgsl` | — |
| `rarefaction` | `rarefaction_gpu` | `FusedMapReduceF64` | — |
| `kriging` | `kriging` | `KrigingF64` | — |
| `streaming` | `streaming_gpu` | `FusedMapReduceF64`, `GemmCached` | 88 |

### Tier B — Adapt Existing Patterns (5 modules)

| CPU Module | GPU Path | Blocker |
|------------|----------|---------|
| `chimera` | GPU k-mer scoring kernel | Currently CPU pass-through |
| `tolerance_search` | Adapt `batched_bisection_f64.wgsl` | ppm/Da bounds mapping |
| `kmd` | `FusedMapReduceF64` for mass-defect grouping | Pattern adaptation |
| `signal` | Map-reduce / parallel scan | Peak detection as GPU primitive |
| `qs_biofilm` | Batched parameter sweeps via RK4 | Depends on `ode` GPU |

### Tier C — New Primitives Needed (8 modules)

| CPU Module | Needed Primitive | Priority | Notes |
|------------|-----------------|----------|-------|
| `ode` | `rk4_batch_f64.wgsl` / `BatchedRK4F64` | P0 | Batched ODE for parameter sweeps |
| `gillespie` | PRNG + exponential sampling | P1 | Parallel trajectory ensembles |
| `decision_tree` | `rf_inference_f64.wgsl` / `RandomForestGpu` | P2 | Tree traversal on GPU |
| `kmer` | GPU lock-free hash table | P2 | Hash-heavy, no current primitive |
| `derep` | GPU lock-free hash table | P2 | Same blocker as kmer |
| `unifrac` | `TreeTraversalF64` | P2 | Phylogenetic tree traversal |
| `robinson_foulds` | Bipartition set ops | P3 | Pure set logic, low GPU benefit |
| `merge_pairs` | Overlap scoring shader | P3 | Branching-heavy, new design needed |

---

## ToadStool Primitives Currently Used

11 primitives from the shared `barracuda` crate:

| Primitive | Usage |
|-----------|-------|
| `FusedMapReduceF64` | Diversity, EIC, rarefaction, spectral norms, stats |
| `BrayCurtisF64` | Bray-Curtis condensed distance matrix |
| `BatchedEighGpu` | PCoA eigendecomposition |
| `GemmF64` | Taxonomy scoring, spectral cosine |
| `KrigingF64` | Spatial interpolation |
| `VarianceF64` | Population statistics |
| `CorrelationF64` | Pearson correlation |
| `CovarianceF64` | Population covariance |
| `WeightedDotF64` | EIC integration, weighted sums |
| `GemmCached` | Pipeline-cached GEMM (streaming) |
| `FusedMapReduceF64` (streaming) | Streaming pipeline quality/diversity |

Plus 2 custom WGSL shaders written by wetSpring:
- `quality_filter.wgsl` — per-base quality filtering
- `dada2_e_step.wgsl` — DADA2 E-step error model

---

## New Primitives Requested (for ToadStool/BarraCUDA)

### P0: Batched RK4 ODE Solver
- **Module:** `bio::ode`, `bio::qs_biofilm`
- **What:** `BatchedRK4F64` — batch integrate N parameter sets in parallel
- **Why:** Waters 2008 (5-variable, 4 scenarios) and future ODE models
- **Shader spec:** `rk4_batch_f64.wgsl` — fixed-step RK4, N×M state vectors

### P1: Stochastic Simulation Primitives
- **Module:** `bio::gillespie`
- **What:** GPU PRNG (LCG/xoshiro) + exponential sampling
- **Why:** Massie 2012 ensemble (1000 trajectories, embarrassingly parallel)
- **Shader spec:** Parallel birth-death SSA per workgroup

### P1: Log-Sum-Exp / HMM Forward-Backward
- **Module:** Future `bio::hmm`
- **What:** Numerically stable log-sum-exp reduction
- **Why:** PhyloNet-HMM introgression detection (Liu 2014)
- **Shader spec:** `log_sum_exp_f64.wgsl`

### P1: Smith-Waterman Alignment
- **Module:** Future `bio::alignment`
- **What:** Banded SW with affine gap penalties
- **Why:** SATe/phylogenetic alignment (Liu 2009)
- **Shader spec:** `smith_waterman_f64.wgsl`

### P2: Decision Tree / Random Forest GPU
- **Module:** `bio::decision_tree`
- **What:** Batch inference over N samples × M trees
- **Why:** PFAS monitoring ML at field scale
- **Shader spec:** `rf_inference_f64.wgsl`

### P2: Phylogenetic Likelihood / Felsenstein Pruning
- **Module:** Future `bio::phylo_likelihood`
- **What:** Postorder tree traversal with site likelihood
- **Why:** Maximum-likelihood phylogenetics
- **Shader spec:** `felsenstein_f64.wgsl`

---

## Local Extensions (Candidates for ToadStool Absorption)

These are wetSpring-specific GPU code that should be generalized:

| Local Extension | Current Location | Recommended Generalization |
|----------------|-----------------|---------------------------|
| `QualityFilterCached` | `quality_gpu.rs` | `ParallelFilter<T>` pattern |
| `Dada2Gpu` (E-step) | `dada2_gpu.rs` | `BatchPairReduce<f64>` pattern |
| `GemmCached` pipeline | `gemm_cached.rs` | `GemmF64::cached()` builder |
| `StreamingGpu` pipeline | `streaming_gpu.rs` | Pipeline composition pattern |

---

## Completed Evolution Items

| Version | Item | Date |
|---------|------|------|
| v0.1.0 | Initial 16S pipeline (CPU) | Feb 2026 |
| v0.1.0 | mzML/MS2 sovereign parsers | Feb 2026 |
| v0.1.0 | PFAS screening (KMD + spectral match) | Feb 2026 |
| v0.1.0 | GPU diversity (FusedMapReduceF64) | Feb 2026 |
| v0.1.0 | GPU streaming pipeline (88/88 parity) | Feb 2026 |
| v0.1.0 | RK4 ODE solver (Waters 2008) | Feb 2026 |
| v0.1.0 | Gillespie SSA (Massie 2012) | Feb 2026 |
| v0.1.0 | Robinson-Foulds distance | Feb 2026 |
| v0.1.0 | Decision tree inference (100% parity) | Feb 2026 |
| v0.1.0 | Newick parsing validation | Feb 2026 |

---

## Gaps and Blockers

| Gap | Impact | Status |
|-----|--------|--------|
| No GPU hash table primitive | Blocks kmer, derep GPU promotion | Waiting on ToadStool |
| No GPU tree traversal primitive | Blocks unifrac, RF GPU promotion | Waiting on ToadStool |
| `GPU_DISPATCH_THRESHOLD` = 10,000 | Small datasets fall back to CPU | By design |
| Inline tolerances in binaries | Not blocking, but debt | Deferred refactor |
| `include_str!` path to ToadStool shader | Brittle cross-crate path | ToadStool team to stabilize |
