# Experiment 120: Cross-Spring Evolution Benchmark

**Binary**: `benchmark_cross_spring_evolution`
**Status**: PASS — 9/9 checks

## Purpose

Benchmark and document the cross-spring shader evolution in ToadStool
BarraCuda. Tracks which primitives evolved from each spring, measures
CPU and NPU performance, and validates the rewired modern import paths
(16 files migrated from deep `ops::bio::` to crate-root re-exports).

## Cross-Spring Provenance

| Origin | Shaders | Ops | Key Domains |
|--------|---------|-----|-------------|
| **hotSpring** | ~35 | ~25 | Nuclear HFB, lattice QCD, MD, ESN, f64 precision |
| **wetSpring** | ~22 | ~18 | Metagenomics, DADA2, ANI, dN/dS, Gillespie, Felsenstein |
| **neuralSpring** | ~14 | ~12 | metalForge, pairwise distances, evolutionary, IPR |
| **airSpring** | ~5 | ~8 | IoT, precision agriculture, Richards, Kriging |
| **ToadStool-native** | 100+ | 200+ | Math, linalg, NN, FHE, attention |
| **Total** | **612** | **265+** | All domains |

## Cross-Spring Synergy (how each spring benefits the others)

### hotSpring → wetSpring
- **f64 precision shaders**: RK4/RK45 ODE integration precision used for QS biofilm
  and Gillespie SSA models (10+ experiments)
- **Jacobi eigh**: PCoA eigendecomposition for beta-diversity (Exp101-102, 106)
- **ESN reservoir (esn_v2)**: Reservoir computing for NPU deployment (Exp114-119)
- **Broyden mixing**: HFB density mixing patterns → potential for SCF-style
  iterative diversity convergence

### neuralSpring → wetSpring
- **Spectral theory (Anderson/Hofstadter)**: QS-disorder phase classification
  (Exp113, Exp119)
- **pairwise_l2**: Cross-substrate distance metrics in metalForge
- **batch_fitness, pairwise_hamming, pairwise_jaccard**: Community evolution
  and metagenome comparison (Exp037-042)

### airSpring → wetSpring
- **moving_window_stats**: Environmental monitoring → bloom time-series
- **Kriging**: Spatial interpolation → diversity mapping across sampling sites

### wetSpring → all springs
- **Bray-Curtis (bray_curtis_f64.wgsl)**: Absorbed into ToadStool, used by
  airSpring for sensor network similarity
- **Felsenstein (felsenstein_f64.wgsl)**: Available to neuralSpring for
  phylogenetic neural priors
- **Gillespie (gillespie_ssa_f64.wgsl)**: Stochastic simulation available to
  hotSpring for nuclear decay and neuralSpring for evolutionary dynamics
- **Quality filter / DADA2 / SNP calling**: Bioinformatics pipeline primitives
  available ecosystem-wide

## Benchmark Results

| Domain | Operation | Time | Source |
|--------|-----------|------|--------|
| Diversity | Shannon (500 communities) | 0.13 ms | wetSpring |
| Diversity | Simpson (500 communities) | 0.04 ms | wetSpring |
| Diversity | Bray-Curtis (500 pairs) | 0.01 ms | wetSpring |
| QS ODE | Single integration (2000 steps) | 0.02 ms | hotSpring precision |
| QS ODE | Parameter sweep (200 configs) | <0.01 ms | hotSpring precision |
| ESN | Train (300×200 reservoir) | 30.02 ms | hotSpring ESN |
| ESN | f64 inference (100 samples) | 2.99 ms | hotSpring ESN |
| NPU | int8 inference (100 samples) | 3.72 ms | hotSpring ESN → wetSpring |

## Import Modernization

16 files migrated from deep `barracuda::ops::bio::module::Type` paths to
crate-root re-exports (`barracuda::Type`). Two config types (`QualityConfig`,
`UniFracConfig`) remain on deep paths as ToadStool S42 does not re-export them.

## Reproduction

```bash
cargo run --release --bin benchmark_cross_spring_evolution
```
