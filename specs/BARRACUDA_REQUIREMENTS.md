# wetSpring — BarraCUDA Requirements

**Last Updated**: February 19, 2026
**Purpose**: GPU kernel requirements, gap analysis, and evolution priorities

---

## Current Kernel Usage (Validated)

### Rust CPU Modules (31 modules, 361 tests)

| Module Domain | Modules | Tests | Status |
|--------------|---------|-------|--------|
| I/O | fastq, mzml, ms2, xml, encoding | ~60 | Sovereign |
| 16S Pipeline | quality, merge_pairs, derep, dada2, chimera, taxonomy, kmer | ~90 | Sovereign |
| Diversity | diversity, pcoa, unifrac | ~40 | Sovereign |
| LC-MS | eic, signal, feature_table, spectral_match, tolerance_search, kmd | ~50 | Sovereign |
| Spatial | kriging | ~10 | Sovereign |
| GPU | diversity_gpu, pcoa_gpu, spectral_match_gpu, stats_gpu, eic_gpu, rarefaction_gpu | ~30 | Validated |

### GPU Shaders (Validated Phase 3)

| ToadStool Primitive | wetSpring Use | Checks | Performance |
|-------------------|---------------|--------|-------------|
| `FusedMapReduceF64` | Shannon entropy, Simpson index, alpha diversity | 12/12 | CPU-competitive at small N |
| `BrayCurtisF64` | All-pairs Bray-Curtis distance matrix | 6/6 | 0.40x at 100x100 (CPU faster) |
| `BatchedEighGpu` | PCoA ordination via eigendecomposition | 5/5 | Validated at f64 |
| `GemmF64` + `FusedMapReduceF64` | Spectral cosine matching | 8/8 | **1,077x speedup** at 200x200 |
| `VarianceF64` | Variance / standard deviation | 3/3 | Validated |
| `CorrelationF64` + `CovarianceF64` | Pearson r / covariance | 2/2 | Validated |
| `WeightedDotF64` | Weighted dot product | 2/2 | Validated |

---

## GPU Promotion Priorities (Next Phase)

### Ready for GPU (validated Rust exists)

| Operation | Current | GPU Target | Benefit |
|-----------|---------|------------|---------|
| DADA2 denoising | Rust CPU | Parallel error model evaluation | 100x sample throughput |
| Chimera detection | Rust CPU | Parallel k-mer comparison | Scales with database size |
| Taxonomy classification (RDP Bayes) | Rust CPU | Parallel k-mer scoring | Largest single bottleneck in 16S pipeline |
| UniFrac distance | Rust CPU | Tree traversal + matrix ops | Phylogenetic beta diversity |
| Rarefaction with bootstrap CI | Rust CPU template | Embarrassingly parallel resampling | 1000 rarefaction curves in one dispatch |

### Gaps for Faculty Extension Papers

| Need | Paper(s) | Priority | Effort |
|------|----------|----------|--------|
| **ODE solver (Runge-Kutta)** | Waters 2008 (c-di-GMP dynamics), Waters 2020 (bifurcation) | **P0** | Medium — `rk4_f64.wgsl` for signaling pathway ODEs |
| **Gillespie stochastic simulation** | Massie 2012 (signaling specificity) | **P1** | Medium — PRNG + exponential sampling. GPU-parallelizable across trajectory ensemble |
| **HMM forward/backward/Viterbi** | Liu 2014 (PhyloNet-HMM) | **P1** | Medium — matrix chain multiplication in log-space. Need log-sum-exp shader |
| **Sequence alignment (Smith-Waterman)** | Liu 2009 (SATé), Alamin 2024 (metagenomics) | **P1** | High — dynamic programming on GPU. Well-studied problem with known GPU solutions |
| **Phylogenetic likelihood (Felsenstein pruning)** | Liu 2023 (cophylogenetics), SATé | **P2** | High — parallel tree evaluation. GEMM-heavy at each internal node |
| **Bifurcation / continuation** | Fernandez 2020 (phenotypic switching) | **P2** | Medium — parameter sweeps + eigenvalue tracking. `BatchedEighGpu` handles eigenvalues |

### Existing ToadStool Kernels That Apply to New Papers

| ToadStool Kernel | Extension Use |
|-----------------|---------------|
| `FusedMapReduceF64` | Gillespie trajectory statistics, ODE observable reduction |
| `BatchedEighGpu` | Bifurcation analysis eigenvalues, covariance matrix decomposition |
| `GemmF64` | HMM transition probability chains, phylogenetic likelihood |
| `CorrelationF64` | Signaling pathway cross-correlation analysis |

---

## BarraCUDA Evolution Path for wetSpring

```
Phase 2-3 (DONE)              Faculty Extension (NEXT)
────────────────              ──────────────────────
Shannon/Simpson GPU ────────→  Stochastic diversity (Gillespie)
Spectral cosine GPU ────────→  Alignment scoring on GPU
PCoA (eigensolve)   ────────→  Bifurcation eigenvalues
16S pipeline (CPU)  ────────→  Parallel DADA2/chimera/taxonomy GPU
N/A                 ────────→  ODE solver for signaling (NEW)
N/A                 ────────→  HMM Viterbi for genomics (NEW)
N/A                 ────────→  Smith-Waterman alignment (NEW)
```

---

## ToadStool Handoff Notes

- `log_f64` bug found by wetSpring (coefficients halved) — fixed in ToadStool Feb 16
- Native `log(f64)` crashes NVIDIA NVVM compiler — all transcendentals must use portable implementations
- Spectral cosine achieves 1,077x GPU speedup — the first "GPU wins decisively" benchmark from any spring
- Sovereign XML parser (`io::xml`) eliminates `quick-xml` dependency — pattern for other I/O modules
- 30 Rust modules with 1 runtime dependency (flate2) — highest sovereignty ratio in the ecosystem
