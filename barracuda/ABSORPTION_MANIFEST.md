# Absorption Manifest — wetSpring BarraCUDA

**Date:** February 22, 2026
**Pattern:** Write → Absorb → Lean (following hotSpring)

Springs write validated extensions as proposals to ToadStool/BarraCUDA.
Once absorbed upstream, Springs rewire to lean on the shared primitive
and delete local code. This manifest tracks that lifecycle.

---

## Already Absorbed (Lean)

Modules that were written locally, absorbed by ToadStool, and now
consumed as upstream primitives via `barracuda::ops::*`.

### Original Shared Primitives (15)

| Primitive | Upstream Module | Absorbed In | wetSpring Consumer |
|-----------|----------------|-------------|-------------------|
| `FusedMapReduceF64` | `barracuda::ops::fmr` | v0.4 | `diversity_gpu`, `stats_gpu`, `chimera_gpu` |
| `BrayCurtisF64` | `barracuda::ops::bray_curtis` | v0.4 | `diversity_gpu` |
| `BatchedEighGpu` | `barracuda::ops::eigh` | v0.4 | `pcoa_gpu` |
| `GemmF64` | `barracuda::ops::gemm` | v0.4 | `spectral_match_gpu`, `gemm_cached` |
| `SmithWatermanGpu` | `barracuda::ops::bio::sw` | v0.5 | via `alignment` |
| `GillespieGpu` | `barracuda::ops::bio::gillespie` | v0.5 | via `gillespie` |
| `TreeInferenceGpu` | `barracuda::ops::bio::tree` | v0.5 | via `decision_tree` |
| `FelsensteinGpu` | `barracuda::ops::bio::felsenstein` | v0.5 | `bootstrap`, `placement` |
| `KrigingF64` | `barracuda::ops::kriging` | v0.4 | `kriging` |
| `WeightedDotF64` | `barracuda::ops::fmr` | v0.4 | `eic_gpu`, `stats_gpu` |
| `VarianceF64` | `barracuda::ops::stats` | v0.4 | `stats_gpu` |
| `CorrelationF64` | `barracuda::ops::stats` | v0.4 | `stats_gpu` |
| `CovarianceF64` | `barracuda::ops::stats` | v0.4 | `stats_gpu` |
| `BatchTolSearchF64` | `barracuda::ops::search` | v0.4 | `tolerance_search` |
| `PrngXoshiro` | `barracuda::ops::prng` | v0.4 | `rarefaction_gpu` |

### Bio Primitives — Absorbed Feb 2026 (Session 31d/31g), Rewired Feb 22 (8)

These were written as local WGSL shaders in wetSpring, absorbed by
ToadStool in sessions 31d (HMM, ANI, SNP, dN/dS, Pangenome, Quality,
DADA2) and 31g (RF), and **rewired on Feb 22, 2026**: local shaders
deleted, modules now delegate to `barracuda::ops::bio::*`.

| Primitive | Upstream Module | wetSpring Wrapper | Local WGSL (deleted) |
|-----------|----------------|-------------------|---------------------|
| `HmmBatchForwardF64` | `barracuda::ops::bio::hmm` | `hmm_gpu::HmmGpuForward` | `hmm_forward_f64.wgsl` |
| `AniBatchF64` | `barracuda::ops::bio::ani` | `ani_gpu::AniGpu` | `ani_batch_f64.wgsl` |
| `SnpCallingF64` | `barracuda::ops::bio::snp` | `snp_gpu::SnpGpu` | `snp_calling_f64.wgsl` |
| `DnDsBatchF64` | `barracuda::ops::bio::dnds` | `dnds_gpu::DnDsGpu` | `dnds_batch_f64.wgsl` |
| `PangenomeClassifyGpu` | `barracuda::ops::bio::pangenome` | `pangenome_gpu::PangenomeGpu` | `pangenome_classify.wgsl` |
| `QualityFilterGpu` | `barracuda::ops::bio::quality_filter` | `quality_gpu::QualityFilterCached` | `quality_filter.wgsl` |
| `Dada2EStepGpu` | `barracuda::ops::bio::dada2` | `dada2_gpu::Dada2Gpu` | `dada2_e_step.wgsl` |
| `RfBatchInferenceGpu` | `barracuda::ops::bio::rf_inference` | `random_forest_gpu::RandomForestGpu` | `rf_batch_inference.wgsl` |

**Total Lean primitives:** 23 (15 original + 8 bio).

---

## Ready for Absorption (Write)

### Tier A: Local WGSL Shaders (1 remaining)

| Shader | File | Checks | Target Primitive | Blocker |
|--------|------|:------:|-----------------|---------|
| ODE sweep | `batched_qs_ode_rk4_f64.wgsl` | 7 | `BatchedOdeRK4F64` | ToadStool `enable f64;` in shader line 35 |

### Tier B: CPU Math Functions (4)

| Function | Module | Target |
|----------|--------|--------|
| `erf()` | `crate::special` | `barracuda::math::erf` |
| `normal_cdf()` | `crate::special` | `barracuda::math::normal_cdf` |
| `ln_gamma()` | `crate::special` | `barracuda::math::ln_gamma` |
| `regularized_gamma_lower()` | `crate::special` | `barracuda::math::regularized_gamma_p` |

Blocked on proposed `barracuda::math` feature (CPU-only, no wgpu dependency).

---

## Local Only (No Absorption Path)

Modules that stay in wetSpring because they are domain-specific or
sequential algorithms without GPU benefit.

| Module | Reason |
|--------|--------|
| `neighbor_joining` | Sequential O(n³) algorithm |
| `robinson_foulds` | Per-node tree comparison |
| `reconciliation` | Tree traversal |
| `cooperation` | Game-theoretic ODE (small system) |
| `molecular_clock` | Small calibration data |
| `kmd` | Lookup table |
| `signal` | FFT-based, small data |
| `feature_table` | Sparse matrix pipeline |
| `phred` | Per-base lookup |
| `derep` | Hash-based |
| `merge_pairs` | Sequential per-pair |
| `encoding` | Base64 codec |
| I/O parsers | Streaming, CPU-bound |

---

## Blocked

| Module | Blocker | Priority |
|--------|---------|----------|
| `kmer` | Needs lock-free GPU hash table | P3 |
| `unifrac` | Needs GPU tree traversal | P3 |
| `taxonomy` (NPU) | Needs AKD1000 int8 FC model support | P3 |

---

## Absorption Readiness Summary

| Category | Count | Change |
|----------|-------|--------|
| **Lean** (consumed upstream) | 23 primitives | +8 (Feb 22 rewire) |
| **Write** (local WGSL, ready) | 1 shader | -8 (absorbed) |
| **Write** (CPU math, ready) | 4 functions | — |
| **Local** (no GPU path) | 13 modules | — |
| **Blocked** | 3 modules | — |

## Process

1. Validate locally (Python baseline → Rust CPU → GPU shader)
2. Document binding layout, dispatch geometry, CPU reference
3. Submit handoff to `../wateringHole/handoffs/`
4. ToadStool team absorbs as shared primitive
5. wetSpring rewires: `use barracuda::ops::new_primitive`
6. Delete local WGSL + GPU module code
7. Run full validation suite to confirm parity
