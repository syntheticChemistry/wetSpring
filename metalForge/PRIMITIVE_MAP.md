# Primitive Map: wetSpring Rust → ToadStool GPU

**Date:** February 26, 2026
**Purpose:** Map every wetSpring Rust module to its ToadStool/BarraCuda
GPU primitive (or explain why it stays CPU-only). This guides the
absorption pipeline and identifies what ToadStool needs to build next.

> **Feb 26 update (latest):** 44 ToadStool primitives + 2 BGL helpers + 5 ODE `cpu_derivative` consumed.
> 0 local WGSL, 0 local derivative math (fully lean). diversity_fusion absorbed S63.
> All ODE shaders use `BatchedOdeRK4<S>::generate_shader()` (Absorbed).
> S68+ aligned (`e96576ee`). 79 primitives consumed. BGL boilerplate removed (~258 lines).
> Forge crate v0.3.0. 200 experiments, 4,748+ checks (60 NPU), 1,008 tests. Phase 60, 882 barracuda lib + 60 integration + 19 doc + 47 forge, 95.46% line / 93.54% fn / 94.99% branch, 86 named tolerances, 0 ad-hoc magic numbers, clippy pedantic CLEAN.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| **Lean** | Using upstream ToadStool primitive |
| **Local** | wetSpring-authored WGSL shader (absorption candidate) |
| **Compose** | Combining existing ToadStool primitives |
| **CPU** | No GPU path (I/O-bound, sequential, or too small) |
| **NPU** | Neural Processing Unit candidate |
| **Blocked** | Needs new ToadStool primitive |

---

## Full Mapping

### Diversity & Statistics (GPU-validated)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `diversity` (Shannon) | Lean | `FusedMapReduceF64::shannon_entropy()` | 004 |
| `diversity` (Simpson) | Lean | `FusedMapReduceF64::simpson_index()` | 004 |
| `diversity` (Bray-Curtis) | Lean | `BrayCurtisF64` | 004 |
| `diversity` (Pielou, Chao1) | Lean | `FusedMapReduceF64` | 044 |
| `diversity_gpu` | Lean | `BrayCurtisF64` + `FMR` | 016 |
| `spectral_match_gpu` | Lean | `FMR` (spectral cosine) | 016 |
| `stats_gpu` | Lean | `FMR` (variance, correlation) | 016 |
| `pcoa_gpu` | Lean | `BatchedEighGpu` | 016 |
| `tolerance_search` | Lean | `BatchTolSearchF64` | 016 |

### Phylogenetics (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `felsenstein` | Lean | `FelsensteinGpu` | 046 |
| `bootstrap` | Compose | `FelsensteinGpu` per replicate | 046 |
| `placement` | Compose | `FelsensteinGpu` per edge | 046 |
| `neighbor_joining` / `neighbor_joining_gpu` | Compose | GPU distance matrix + CPU NJ loop (`FMR`) | Pure GPU |
| `robinson_foulds` / `robinson_foulds_gpu` | Compose | `PairwiseHammingGpu` bipartition bit-vectors | Pure GPU |
| `reconciliation` / `reconciliation_gpu` | Compose | Batch workgroup-per-family | Pure GPU |

### ODE / Dynamical Systems (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `ode` (RK4) | Lean | `BatchedOdeRK4F64` (ToadStool S41) | 049 |
| `qs_biofilm` | Lean | `BatchedOdeRK4F64` (ToadStool S41) | 049 |
| `bistable` / `bistable_gpu` | Lean | `BatchedOdeRK4<BistableOde>::generate_shader()` | 099 |
| `multi_signal` / `multi_signal_gpu` | Lean | `BatchedOdeRK4<MultiSignalOde>::generate_shader()` | 100 |
| `phage_defense` / `phage_defense_gpu` | Lean | `BatchedOdeRK4<PhageDefenseOde>::generate_shader()` | 099 |
| `cooperation` / `cooperation_gpu` | Lean | `BatchedOdeRK4<CooperationOde>::generate_shader()` | 101 |
| `capacitor` / `capacitor_gpu` | Lean | `BatchedOdeRK4<CapacitorOde>::generate_shader()` | 100 |
| `gillespie` | Lean | `GillespieGpu` | 044 |

### Sequence Analysis (mixed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `alignment` (SW) | Lean | `SmithWatermanGpu` | 044 |
| `hmm` | Lean | `HmmBatchForwardF64` (ToadStool, absorbed Feb 22) | 047 |
| `kmer` | Lean | `KmerHistogramF64` (ToadStool S40) | 081 |
| `unifrac` | Lean | `UniFracPropagateF64` (ToadStool S40) | 082 |
| `decision_tree` | Lean | `TreeInferenceGpu` | 044 |
| `random_forest` / `random_forest_gpu` | Lean | `RfBatchInferenceGpu` (ToadStool, absorbed Feb 22) | 063 |
| `gbm` / `gbm_gpu` | Compose | `TreeInferenceGpu` batch inference | Pure GPU |

### 16S Pipeline (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `quality` / `quality_gpu` | Lean | `QualityFilterGpu` (ToadStool, absorbed Feb 22) | 016 |
| `dada2` / `dada2_gpu` | Lean | `Dada2EStepGpu` (ToadStool, absorbed Feb 22) | 016 |
| `chimera` / `chimera_gpu` | Lean | `FMR` | 016 |
| `taxonomy` / `taxonomy_gpu` | Lean | `TaxonomyFcF64` (ToadStool S40) + NPU int8 (`to_int8_weights`) | 016/083 |
| `streaming_gpu` | Lean | Multiple primitives | 016 |
| `rarefaction_gpu` | Lean | `PrngXoshiro` | 016 |

### Analytical Chemistry (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `eic` / `eic_gpu` | Lean | `FMR` | 016 |
| `kmd` / `kmd_gpu` | Compose | `FusedMapReduceF64` element-wise | Pure GPU |
| `signal` / `signal_gpu` | Compose | `FusedMapReduceF64` batch | Pure GPU |
| `feature_table` / `feature_table_gpu` | Compose | Chains `eic_gpu` + `signal_gpu` | Pure GPU |

### Lean (absorbed S63): Former diversity_fusion Extension

| Rust Module | GPU Strategy | ToadStool Primitive | Status |
|-------------|-------------|---------------------|--------|
| `diversity_fusion_gpu` | **Lean** | Absorbed by ToadStool S63 | Fused Shannon + Simpson + evenness in single dispatch |

### Track 1c: Deep-Sea Metagenomics (ToadStool-absorbed, Exp058)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `ani` / `ani_gpu` | Lean | `AniBatchF64` (ToadStool, absorbed Feb 22) | 058 |
| `snp` / `snp_gpu` | Lean | `SnpCallingF64` (ToadStool, absorbed Feb 22) | 058 |
| `dnds` / `dnds_gpu` | Lean | `DnDsBatchF64` (ToadStool, absorbed Feb 22) | 058 |
| `pangenome` / `pangenome_gpu` | Lean | `PangenomeClassifyGpu` (ToadStool, absorbed Feb 22) | 058 |
| `molecular_clock` / `molecular_clock_gpu` | Compose | `FusedMapReduceF64` relaxed rates | Pure GPU |

### Infrastructure (GPU-promoted)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `phred` | CPU | Per-base quality lookup (no parallelism benefit) | — |
| `derep` / `derep_gpu` | Compose | `KmerHistogramGpu` parallel hashing | Pure GPU |
| `merge_pairs` / `merge_pairs_gpu` | Compose | `FusedMapReduceF64` batch overlap | Pure GPU |

---

## Substrate Decision Rules

```
Is the workload batch-parallel (N > 64)?
  YES → Does it use f64 transcendentals?
    YES → Force polyfill via ShaderTemplate::for_driver_auto(_, true)
    NO  → Standard compile_shader_f64
  NO → CPU (dispatch overhead exceeds compute)

Is the data size < L2 cache (36 MB RTX 4070)?
  YES → GPU friendly (single dispatch)
  NO  → Stream via ToadStool unidirectional pipeline

Is the model a neural network or lookup table?
  YES → NPU candidate (AKD1000, int8 quantized)
  NO  → GPU or CPU
```

---

## Absorption Status

All 8 bio WGSL shaders were absorbed by ToadStool (sessions 31d/31g) and
rewired on Feb 22, 2026. The CPU-side batch APIs and `repr(C)` patterns
remain as the bridge between domain types and ToadStool's raw buffer API.

| Module | ToadStool Primitive | Batch API | Rewired |
|--------|-------------------|-----------|---------|
| `dada2` | `Dada2EStepGpu` | GPU module bridges | Feb 22 |
| `hmm` | `HmmBatchForwardF64` | `forward_batch` | Feb 22 |
| `ani` | `AniBatchF64` | `ani_matrix` | Feb 22 |
| `snp` | `SnpCallingF64` | `call_snps_batch` | Feb 22 |
| `dnds` | `DnDsBatchF64` | `pairwise_dnds_batch` | Feb 22 |
| `pangenome` | `PangenomeClassifyGpu` | `analyze_batch` | Feb 22 |
| `quality` | `QualityFilterGpu` | `filter_reads_flat` | Feb 22 |
| `random_forest` | `RfBatchInferenceGpu` | `predict_batch` | Feb 22 |
| `felsenstein` | `FelsensteinGpu` | Per-site (already absorbed) | Earlier |

All shaders absorbed; ODE blocker resolved (ToadStool S41 fixed `compile_shader_f64`).

### Shared Math (`crate::special`) — Extracted

Promoted from `bio::special` to top-level `crate::special` module.
The `bio::special` re-export shim has been removed (Phase 24).

| Function | Consumers | Location |
|----------|-----------|----------|
| `erf()` | `normal_cdf()` | `crate::special::erf` |
| `normal_cdf()` | `bio::pangenome` (enrichment FDR) | `crate::special::normal_cdf` |
| `ln_gamma()` | `regularized_gamma_lower()` | `crate::special::ln_gamma` |
| `regularized_gamma_lower()` | `bio::dada2` (Poisson p-value) | `crate::special::regularized_gamma_lower` |

---

## Counts

| Category | Count |
|----------|-------|
| **Lean** (upstream ToadStool) | 25 modules (16 original + 8 bio absorbed Feb 22 + diversity_fusion S63) |
| **Local** (WGSL shader) | 0 modules (fully lean) |
| **Compose** (existing primitives) | 16 modules (5 original + 11 pure GPU promotion) |
| **CPU** (no GPU path) | 1 module (phred) |
| **NPU** (candidate) | 1 module |
| **Local WGSL** | 0 (fully lean) |
