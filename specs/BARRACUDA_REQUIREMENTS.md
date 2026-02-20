# wetSpring — BarraCUDA Requirements

**Last Updated**: February 20, 2026
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

### Rust CPU Modules (34 modules, 465 tests)

| Module Domain | Modules | Status |
|--------------|---------|--------|
| I/O | fastq, mzml, ms2, xml, encoding | Sovereign |
| 16S Pipeline | quality, merge_pairs, derep, dada2, chimera, taxonomy, kmer | Sovereign |
| Diversity | diversity, pcoa, unifrac | Sovereign |
| LC-MS | eic, signal, feature_table, spectral_match, tolerance_search, kmd | Sovereign |
| Spatial | kriging | Sovereign |
| Math Biology | ode, qs_biofilm, gillespie, bistable, multi_signal, cooperation, capacitor, phage_defense | Sovereign |
| Phylogenetics | felsenstein, robinson_foulds, hmm, alignment, bootstrap, placement, neighbor_joining, reconciliation | Sovereign |
| ML | decision_tree | Sovereign |

### GPU Primitives (15 ToadStool + 4 local WGSL, 200 checks)

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
| `TreeInferenceGpu` | Decision tree/RF inference | 6/6 | **Absorbed Feb 20** |
| `GillespieGpu` | Parallel SSA trajectories | skip | NVVM driver issue |
| `FelsensteinGpu` | Phylogenetic pruning likelihood | avail | Ready to wire |
| `GemmF64::WGSL` | Eliminates fragile include_str! path | — | **Absorbed Feb 20** |

---

## GPU Promotion Status (Feb 20, 2026)

### Resolved (ToadStool absorbed)

| Need | Resolution | Date |
|------|-----------|------|
| ~~ODE solver (Runge-Kutta)~~ | `RkIntegrator` (ToadStool), `numerical::rk45` | Feb 19 |
| ~~Gillespie stochastic simulation~~ | `GillespieGpu` (NVVM driver skip on RTX 4070) | Feb 20 |
| ~~Smith-Waterman alignment~~ | `SmithWatermanGpu` (3/3 GPU checks) | Feb 20 |
| ~~Felsenstein pruning~~ | `FelsensteinGpu` (available, needs wiring) | Feb 20 |
| ~~Decision tree inference~~ | `TreeInferenceGpu` (6/6 GPU parity) | Feb 20 |
| ~~GemmF64::WGSL~~ | Public const, fragile path eliminated | Feb 20 |

### Resolved (Exp046-050, Feb 20)

| Need | Resolution | Exp |
|------|-----------|-----|
| ~~Bootstrap resampling~~ | GPU Felsenstein per replicate (15/15) | Exp046 |
| ~~Phylogenetic placement~~ | GPU Felsenstein per edge (15/15) | Exp046 |
| ~~HMM forward/backward~~ | Local WGSL shader `hmm_forward_f64.wgsl` (13/13) | Exp047 |
| ~~ODE parameter sweeps~~ | Local WGSL shader `batched_qs_ode_rk4_f64.wgsl` (7/7) | Exp049 |
| ~~Bifurcation analysis~~ | `BatchedEighGpu` eigenvalues (5/5, bit-exact) | Exp050 |

### Remaining GPU Work

| Operation | Strategy | Priority | Effort |
|-----------|----------|----------|--------|
| K-mer counting GPU | Lock-free hash table primitive | **P3** | High — new ToadStool primitive |
| UniFrac GPU | Tree traversal primitive | **P3** | High — new ToadStool primitive |

### BarraCUDA Evolution Path

```
DONE                                NOW                              GOAL
─────────────────────────           ──────────────────               ──────────────────
Python baseline (28 scripts) ────→  Rust CPU parity (84/84) ─────→  PURE GPU (all domains)
GPU diversity (38/38)        ────→  Phylo GPU composed (15/15) ──→  metalForge cross-system
GPU pipeline (88/88)         ────→  HMM GPU shader (13/13) ─────→  NPU for low-power inference
CPU ~20× faster than Python  ────→  GPU math PROVEN portable ───→  Scale via streaming
ODE sweep GPU (7/7)          ────→  Bifurcation GPU (5/5) ──────→  Full landscape on GPU
Local shaders: 4 WGSL        ────→  ToadStool absorbs all ──────→  Full Write→Absorb→Lean cycle
```

---

## ToadStool Handoff Notes

- `log_f64` bug found by wetSpring (coefficients halved) — fixed in ToadStool Feb 16
- Native `log(f64)` crashes NVIDIA NVVM compiler — all transcendentals must use portable implementations
- **NVVM workaround**: force `ShaderTemplate::for_driver_auto(source, true)` for shaders using exp/log
- Spectral cosine achieves 926× GPU speedup — the first "GPU wins decisively" benchmark from any spring
- Sovereign XML parser (`io::xml`) eliminates `quick-xml` dependency — pattern for other I/O modules
- 34 CPU + 15 GPU Rust modules with 1 runtime dependency (flate2) — highest sovereignty ratio in the ecosystem
- **HMM absorption candidate**: `hmm_forward_f64.wgsl` (batch forward, 13/13 checks) — ready for ToadStool
