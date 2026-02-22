# wetSpring — BarraCUDA Requirements

**Last Updated**: February 22, 2026
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

### Rust CPU Modules (41 modules, 740 tests, ~97% bio+io coverage)

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

### GPU Primitives (28 ToadStool + 4 local WGSL shaders, 609 checks)

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

### Local WGSL Shaders (4 — Write phase, pending absorption)

8 of 9 original local shaders absorbed by ToadStool (sessions 31d/31g, Feb 22).
4 remain in Write phase:

| Shader | Domain | GPU Checks | Blocker |
|--------|--------|:----------:|---------|
| `batched_qs_ode_rk4_f64.wgsl` | ODE parameter sweep | 7 | Upstream `compile_shader` needs `compile_shader_f64` |
| `kmer_histogram_f64.wgsl` | K-mer counting | — | Pending absorption handoff |
| `unifrac_propagate_f64.wgsl` | UniFrac distance | — | Pending absorption handoff |
| `taxonomy_fc_f64.wgsl` | Taxonomy scoring (NPU) | — | Pending absorption handoff |

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
| K-mer counting GPU | Lock-free hash table primitive | **P3** | High — new ToadStool primitive |
| UniFrac GPU | Tree traversal primitive | **P3** | High — new ToadStool primitive |
| Taxonomy NPU | Naive Bayes → FC model → int8 | **P3** | NPU candidate |

### BarraCUDA Evolution Path

```
DONE                                     DONE/CURRENT                     GOAL
──────────────────────────               ────────────────────             ──────────────────
Python baseline (35 scripts)  ────────→  Rust CPU parity (205/205) ────→  ✓ DONE
GPU diversity (38/38)         ────────→  GPU Parity v1 (Exp064)  ──────→  ✓ DONE (8 domains)
GPU pipeline (88/88)          ────────→  GPU RF inference (13/13) ──────→  NPU for low-power inference
CPU 22.5× faster than Python  ────────→  GPU math PROVEN portable ─────→  Scale via streaming
8 bio shaders absorbed Feb 22  ────────→  28 ToadStool primitives ────→  Full Write→Absorb→Lean cycle
25 CPU domains validated      ────────→  metalForge PROVEN (Exp065) ───→  CPU/GPU/NPU routing
```

---

## ToadStool Handoff Notes

- `log_f64` bug found by wetSpring (coefficients halved) — fixed in ToadStool Feb 16
- Native `log(f64)` crashes NVIDIA NVVM compiler — all transcendentals must use portable implementations
- **NVVM workaround**: force `ShaderTemplate::for_driver_auto(source, true)` for shaders using exp/log
- Spectral cosine achieves 926× GPU speedup — the first "GPU wins decisively" benchmark from any spring
- 41 CPU + 25 GPU Rust modules with 1 runtime dependency (flate2) — highest sovereignty ratio in the ecosystem
- **8 shaders absorbed, 4 in Write phase (ODE, kmer, unifrac, taxonomy)** — see `barracuda/EVOLUTION_READINESS.md` for status
- **Rust edition 2024**, MSRV 1.85 — `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
- **`#![deny(unsafe_code)]`** — edition 2024 makes `std::env::set_var` unsafe; `#[allow]` confined to test env-var calls
