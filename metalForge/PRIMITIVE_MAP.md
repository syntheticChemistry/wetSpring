# Primitive Map: wetSpring Rust → ToadStool GPU

**Date:** February 20, 2026
**Purpose:** Map every wetSpring Rust module to its ToadStool/BarraCUDA
GPU primitive (or explain why it stays CPU-only). This guides the
absorption pipeline and identifies what ToadStool needs to build next.

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
| `neighbor_joining` | CPU | Sequential algorithm | — |
| `robinson_foulds` | CPU | Per-node comparison | — |
| `reconciliation` | CPU | Tree traversal | — |

### ODE / Dynamical Systems (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `ode` (RK4) | **Local** | `batched_qs_ode_rk4_f64.wgsl` | 049 |
| `qs_biofilm` | **Local** | `batched_qs_ode_rk4_f64.wgsl` | 049 |
| `bistable` | Compose | Same ODE sweep shader | — |
| `multi_signal` | Compose | Same ODE sweep shader | — |
| `phage_defense` | Compose | Same ODE sweep shader | — |
| `gillespie` | Lean | `GillespieGpu` | 044 |
| `cooperation` | CPU | Game-theoretic model | — |

### Sequence Analysis (mixed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `alignment` (SW) | Lean | `SmithWatermanGpu` | 044 |
| `hmm` | **Local** | `hmm_forward_f64.wgsl` | 047 |
| `kmer` | **Blocked** | Needs lock-free hash table (P3) | — |
| `unifrac` | **Blocked** | Needs tree traversal (P3) | — |
| `decision_tree` | Lean | `TreeInferenceGpu` | 044 |
| `random_forest` / `random_forest_gpu` | **Local** | `rf_batch_inference.wgsl` (13 checks) | 063 |
| `gbm` | CPU | Sequential boosting (batch-parallel within rounds) | 062 |

### 16S Pipeline (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `quality` / `quality_gpu` | **Local** | `quality_filter.wgsl` | 016 |
| `dada2` / `dada2_gpu` | **Local** | `dada2_e_step.wgsl` | 016 |
| `chimera` / `chimera_gpu` | Lean | `FMR` | 016 |
| `taxonomy` / `taxonomy_gpu` | Lean / **NPU** | `FMR` / FC model | 016 |
| `streaming_gpu` | Lean | Multiple primitives | 016 |
| `rarefaction_gpu` | Lean | `PrngXoshiro` | 016 |

### Analytical Chemistry (CPU-focused)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `eic` / `eic_gpu` | Lean | `FMR` | 016 |
| `kmd` | CPU | Lookup table | — |
| `signal` | CPU | FFT-based, small data | — |
| `capacitor` | CPU | Peak detection | — |
| `feature_table` | CPU | Sparse matrix | — |

### Track 1c: Deep-Sea Metagenomics (GPU-promoted, Exp058)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `ani` / `ani_gpu` | **Local** | `ani_batch_f64.wgsl` (7 checks) | 058 |
| `snp` / `snp_gpu` | **Local** | `snp_calling_f64.wgsl` (5 checks) | 058 |
| `dnds` / `dnds_gpu` | **Local** | `dnds_batch_f64.wgsl` (9 checks) | 058 |
| `pangenome` / `pangenome_gpu` | **Local** | `pangenome_classify.wgsl` (6 checks) | 058 |
| `molecular_clock` | CPU | Small calibration data, tree traversal | 053/054 |

### Infrastructure (CPU-only)

| Rust Module | Reason for CPU |
|-------------|----------------|
| `phred` | Per-base quality lookup |
| `derep` | Hash-based dereplication |
| `merge_pairs` | Sequential per-pair |

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

## Absorption Readiness Gaps

Modules with local WGSL shaders (Tier A) that are missing absorption-friendly
patterns in their CPU implementations:

| Module | Has Batch API | Has `repr(C)` | Has Flat Layout | Gap |
|--------|:------------:|:------------:|:---------------:|-----|
| `dada2` | No | No (GPU module does) | No | Denoise is single-sample; GPU module bridges |
| `hmm` | Yes | No (GPU module does) | No | `forward_batch` present; GPU module has `repr(C)` |
| `ani` | Yes | Yes | No | Ready — `ani_matrix`, `AniParams` |
| `snp` | Partial | Yes | Yes (`SnpFlatResult`) | No batch over multiple alignments |
| `dnds` | Yes | Yes | No | Ready — `pairwise_dnds_batch`, `DnDsParams` |
| `pangenome` | Partial | Yes | Yes (`presence_matrix_flat`) | No batch for many cluster sets |
| `quality` | No | No (GPU module does) | No | `filter_reads` is single-sample |
| `random_forest` | Yes | No (GPU module does) | No | `predict_batch` present; GPU module bridges |
| `felsenstein` | Per-site | No | Yes (`FlatTree`) | Already absorbed; `FlatTree` is the pattern |

For Tier A modules, the GPU companion modules (`*_gpu`) provide the `repr(C)`
bridge layer between CPU structs and WGSL bindings. This is the correct
architecture: CPU modules own the algorithm, GPU modules own the dispatch.

### Shared Math (bio::special) — Ready for Extraction

| Function | Consumers | Absorption Target |
|----------|-----------|-------------------|
| `erf()` | `normal_cdf()` | `barracuda::special::erf` |
| `normal_cdf()` | `bio::pangenome` (enrichment FDR) | `barracuda::special::normal_cdf` |
| `ln_gamma()` | `regularized_gamma_lower()` | `barracuda::special::ln_gamma` |
| `regularized_gamma_lower()` | `bio::dada2` (Poisson p-value) | `barracuda::special::regularized_gamma_p` |

---

## Counts

| Category | Count |
|----------|-------|
| **Lean** (upstream ToadStool) | 16 modules |
| **Local** (WGSL shader) | 9 modules (4 original + 4 Track 1c + 1 RF) |
| **Compose** (existing primitives) | 5 modules |
| **CPU** (no GPU path) | 13 modules |
| **NPU** (candidate) | 1 module |
| **Blocked** (needs new primitive) | 2 modules |
