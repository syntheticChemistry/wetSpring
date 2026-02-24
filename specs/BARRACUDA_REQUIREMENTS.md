# wetSpring — BarraCuda Requirements

**Last Updated**: February 23, 2026
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

### Rust CPU Modules (45 modules, 759 tests, ~97% bio+io coverage)

| Module Domain | Modules | Status |
|--------------|---------|--------|
| I/O | fastq, mzml, ms2, xml, encoding | Sovereign |
| 16S Pipeline | quality, merge_pairs, derep, dada2, chimera, taxonomy, kmer | Sovereign |
| Diversity | diversity, pcoa, unifrac | Sovereign |
| LC-MS | eic, signal, feature_table, spectral_match, tolerance_search, kmd | Sovereign |
| Spatial | kriging | Sovereign |
| Math Biology | ode, qs_biofilm, gillespie, bistable, multi_signal, cooperation, capacitor, phage_defense | Sovereign |
| Phylogenetics | felsenstein, robinson_foulds, hmm, alignment, bootstrap, placement, neighbor_joining, reconciliation | Sovereign |
| Track 1c | ani, snp, dnds, molecular_clock, pangenome | Sovereign |
| ML | decision_tree, random_forest, gbm | Sovereign |

### GPU Primitives (30 ToadStool primitives + 5 local WGSL + 12 GPU wrappers, 702 checks)

| ToadStool Primitive | wetSpring Use | Checks | Performance |
|-------------------|---------------|--------|-------------|
| `FusedMapReduceF64` | Shannon, Simpson, alpha diversity, spectral norms | 12/12 | CPU-competitive at small N |
| `BrayCurtisF64` | All-pairs Bray-Curtis distance matrix | 6/6 | 0.40x at 100×100 |
| `BatchedEighGpu` | PCoA ordination via eigendecomposition | 5/5 | Validated at f64 |
| `GemmF64` + `FusedMapReduceF64` | Spectral cosine matching | 8/8 | **926× speedup** |
| `VarianceF64` | Variance / standard deviation | 3/3 | Validated |
| `CorrelationF64` + `CovarianceF64` | Pearson r / covariance | 2/2 | Validated |
| `WeightedDotF64` | Weighted dot product | 2/2 | Validated |
| `SmithWatermanGpu` | Banded wavefront alignment | 3/3 | **Absorbed Feb 20** |
| `TreeInferenceGpu` | Decision tree inference | 6/6 | **Absorbed Feb 20** |
| `GillespieGpu` | Parallel SSA trajectories | skip | NVVM driver issue |
| `FelsensteinGpu` | Phylogenetic pruning likelihood | 15/15 | **Absorbed + composed** |
| `GemmF64::WGSL` | Eliminates fragile include_str! path | — | **Absorbed Feb 20** |

### Local WGSL Shaders (5 — Write phase active)

Original 12 shaders absorbed by ToadStool (S31d/31g + S39-41). Write cycle
covers all ODE domains not covered by the existing `BatchedOdeRK4F64` (4v/17p):

| Shader | Vars | Params | CPU ↔ GPU | Exp |
|--------|------|--------|-----------|-----|
| `phage_defense_ode_rk4_f64.wgsl` | 4 | 11 | Exact parity | 099 |
| `bistable_ode_rk4_f64.wgsl` | 5 | 21 | Exact parity | 100 |
| `multi_signal_ode_rk4_f64.wgsl` | 7 | 24 | Exact parity | 100 |
| `cooperation_ode_rk4_f64.wgsl` | 4 | 13 | Exact parity | 101 |
| `capacitor_ode_rk4_f64.wgsl` | 6 | 16 | Exact parity | 101 |

Absorption target: ToadStool `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`.

### GPU Wrappers (12 — Compose/Passthrough)

Pure GPU promotion (Exp101) added 12 GPU wrappers via Compose strategy
(wire existing ToadStool primitives) or Passthrough (GPU buffers + CPU core):

| Module | ToadStool Primitive | Strategy |
|--------|-------------------|----------|
| `kmd_gpu` | `FusedMapReduceF64` | Compose |
| `gbm_gpu` | `TreeInferenceGpu` | Compose |
| `merge_pairs_gpu` | `FusedMapReduceF64` | Compose |
| `signal_gpu` | `FusedMapReduceF64` | Compose |
| `feature_table_gpu` | `FMR + WeightedDotF64` | Compose |
| `robinson_foulds_gpu` | `PairwiseHammingGpu` | Compose |
| `derep_gpu` | `KmerHistogramGpu` | Compose |
| `chimera_gpu` | `GemmCachedF64` | Compose |
| `neighbor_joining_gpu` | `FusedMapReduceF64` | Compose |
| `reconciliation_gpu` | Batch workgroup | Passthrough |
| `molecular_clock_gpu` | `FusedMapReduceF64` | Compose |

---

## GPU Promotion Status (Feb 22, 2026)

### Resolved (ToadStool absorbed)

| Need | Resolution | Date |
|------|-----------|------|
| ~~ODE solver~~ | `RkIntegrator`, `numerical::rk45` | Feb 19 |
| ~~Gillespie SSA~~ | `GillespieGpu` (NVVM driver skip on RTX 4070) | Feb 20 |
| ~~Smith-Waterman~~ | `SmithWatermanGpu` (3/3 GPU checks) | Feb 20 |
| ~~Felsenstein pruning~~ | `FelsensteinGpu` (15/15 composed) | Feb 20 |
| ~~Decision tree~~ | `TreeInferenceGpu` (6/6 GPU parity) | Feb 20 |
| ~~GemmF64::WGSL~~ | Public const, fragile path eliminated | Feb 20 |

### Resolved (Local WGSL, Exp046-063)

| Need | Resolution | Exp |
|------|-----------|-----|
| ~~Bootstrap resampling~~ | GPU Felsenstein per replicate (15/15) | 046 |
| ~~Phylogenetic placement~~ | GPU Felsenstein per edge (15/15) | 046 |
| ~~HMM forward~~ | Local WGSL `hmm_forward_f64.wgsl` (13/13) | 047 |
| ~~ODE parameter sweeps~~ | Local WGSL `batched_qs_ode_rk4_f64.wgsl` (7/7) | 049 |
| ~~Bifurcation analysis~~ | `BatchedEighGpu` eigenvalues (5/5, bit-exact) | 050 |
| ~~ANI pairwise~~ | Local WGSL `ani_batch_f64.wgsl` (7/7) | 058 |
| ~~SNP calling~~ | Local WGSL `snp_calling_f64.wgsl` (5/5) | 058 |
| ~~dN/dS~~ | Local WGSL `dnds_batch_f64.wgsl` (9/9) | 058 |
| ~~Pangenome classify~~ | Local WGSL `pangenome_classify.wgsl` (6/6) | 058 |
| ~~RF batch inference~~ | Local WGSL `rf_batch_inference.wgsl` (13/13) | 063 |

### Remaining GPU Work

| Operation | Strategy | Priority | Effort |
|-----------|----------|----------|--------|
| ~~K-mer counting GPU~~ | ✅ `kmer_gpu` wraps `KmerHistogramGpu` (Exp099) | Done | — |
| ~~UniFrac GPU~~ | ✅ `unifrac_gpu` wraps `UniFracPropagateGpu` (Exp099) | Done | — |
| ~~Cooperation GPU~~ | ✅ Local WGSL ODE shader (Exp101) | Done | — |
| ~~Capacitor GPU~~ | ✅ Local WGSL ODE shader (Exp101) | Done | — |
| ~~13 Tier B/C modules~~ | ✅ Pure GPU promotion complete (Exp101) | Done | — |
| Taxonomy NPU | Naive Bayes → FC model → int8 | **P3** | NPU candidate |
| ODE generic absorption | 5 local shaders → `BatchedOdeRK4Generic` | **P2** | ToadStool generalization |

### BarraCuda Evolution Path

```
DONE                                     DONE                              GOAL
──────────────────────────               ────────────────────             ──────────────────
Python baseline (40 scripts)  ────────→  Rust CPU parity (380/380) ────→  ✓ DONE (v1–v8)
GPU diversity (38/38)         ────────→  GPU Parity (29 domains)  ──────→  ✓ DONE (Exp101)
GPU pipeline (88/88)          ────────→  GPU RF inference (13/13) ──────→  NPU for low-power inference
CPU 22.5× faster than Python  ────────→  GPU math PROVEN portable ─────→  Streaming v2: 10 domains (Exp105+106)
12 shaders absorbed (S31d/g + S39-41) ─→  42 GPU modules + 5 local ────→  Full Write→Absorb→Lean cycle
37 MF domains validated       ────────→  metalForge PROVEN (Exp104) ───→  ✓ 25/25 papers three-tier
```

---

## ToadStool Handoff Notes

- `log_f64` bug found by wetSpring (coefficients halved) — fixed in ToadStool Feb 16
- Native `log(f64)` crashes NVIDIA NVVM compiler — all transcendentals must use portable implementations
- **NVVM workaround**: force `ShaderTemplate::for_driver_auto(source, true)` for shaders using exp/log
- Spectral cosine achieves 926× GPU speedup — the first "GPU wins decisively" benchmark from any spring
- 45 CPU + 42 GPU Rust modules with 1 runtime dependency (flate2) — highest sovereignty ratio in the ecosystem
- **12 shaders absorbed + 5 local WGSL (Write phase) + 12 composed wrappers** — see `barracuda/EVOLUTION_READINESS.md` for status
- **Rust edition 2024**, MSRV 1.85 — `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
- **`#![deny(unsafe_code)]`** — edition 2024 makes `std::env::set_var` unsafe; `#[allow]` confined to test env-var calls
