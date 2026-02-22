# Primitive Map: wetSpring Rust → ToadStool GPU

**Date:** February 22, 2026
**Purpose:** Map every wetSpring Rust module to its ToadStool/BarraCUDA
GPU primitive (or explain why it stays CPU-only). This guides the
absorption pipeline and identifies what ToadStool needs to build next.

> **Feb 22 update (latest):** 19 modules lean on upstream. 4 local WGSL shaders
> in Write phase: ODE sweep, kmer histogram, unifrac propagate, taxonomy FC.
> Forge crate v0.2.0 adds streaming dispatch module.

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
| `hmm` | Lean | `HmmBatchForwardF64` (ToadStool, absorbed Feb 22) | 047 |
| `kmer` | **Local** | `kmer_histogram_f64.wgsl` — atomic 4^k histogram + sorted pairs | 081 |
| `unifrac` | **Local** | `unifrac_propagate_f64.wgsl` — CSR tree propagation + pairwise | 082 |
| `decision_tree` | Lean | `TreeInferenceGpu` | 044 |
| `random_forest` / `random_forest_gpu` | Lean | `RfBatchInferenceGpu` (ToadStool, absorbed Feb 22) | 063 |
| `gbm` | CPU | Sequential boosting (batch-parallel within rounds) | 062 |

### 16S Pipeline (GPU-composed)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `quality` / `quality_gpu` | Lean | `QualityFilterGpu` (ToadStool, absorbed Feb 22) | 016 |
| `dada2` / `dada2_gpu` | Lean | `Dada2EStepGpu` (ToadStool, absorbed Feb 22) | 016 |
| `chimera` / `chimera_gpu` | Lean | `FMR` | 016 |
| `taxonomy` / `taxonomy_gpu` | Lean / **Local** | `FMR` / `taxonomy_fc_f64.wgsl` + NPU int8 (`to_int8_weights`) | 016/083 |
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

### Track 1c: Deep-Sea Metagenomics (ToadStool-absorbed, Exp058)

| Rust Module | GPU Strategy | ToadStool Primitive | Exp |
|-------------|-------------|-------------------|-----|
| `ani` / `ani_gpu` | Lean | `AniBatchF64` (ToadStool, absorbed Feb 22) | 058 |
| `snp` / `snp_gpu` | Lean | `SnpCallingF64` (ToadStool, absorbed Feb 22) | 058 |
| `dnds` / `dnds_gpu` | Lean | `DnDsBatchF64` (ToadStool, absorbed Feb 22) | 058 |
| `pangenome` / `pangenome_gpu` | Lean | `PangenomeClassifyGpu` (ToadStool, absorbed Feb 22) | 058 |
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

Only the ODE sweep shader remains local (blocked: upstream uses `compile_shader` not `compile_shader_f64`).

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
| **Lean** (upstream ToadStool) | 24 modules (16 original + 8 bio absorbed Feb 22) |
| **Local** (WGSL shader) | 1 module (ODE sweep, blocked: upstream uses `compile_shader` not `compile_shader_f64`) |
| **Compose** (existing primitives) | 5 modules |
| **CPU** (no GPU path) | 13 modules |
| **NPU** (candidate) | 1 module |
| **Local WGSL** (Write phase — pending absorption) | 4 shaders (ODE, kmer, unifrac, taxonomy) |
