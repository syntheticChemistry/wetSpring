# Cross-System Benchmark Status

**Date:** February 20, 2026
**Purpose:** Track which algorithms are validated on which substrates

---

## Algorithm × Substrate Matrix

| Algorithm | CPU (Rust) | GPU (wgpu) | NPU (AKD1000) | Status |
|-----------|:----------:|:----------:|:--------------:|--------|
| Shannon entropy | ✓ | FusedMapReduceF64 | — | CPU+GPU |
| Simpson diversity | ✓ | FusedMapReduceF64 | — | CPU+GPU |
| Pielou evenness | ✓ | FusedMapReduceF64 | — | CPU+GPU |
| Bray-Curtis (pairwise) | ✓ | BrayCurtisF64 | — | CPU+GPU |
| Observed features | ✓ | FusedMapReduceF64 | — | CPU+GPU |
| PCoA ordination | ✓ | BatchedEighGpu | — | CPU+GPU |
| Spectral cosine | ✓ | GemmF64 + FMR | — | CPU+GPU |
| Variance/StdDev | ✓ | VarianceF64 | — | CPU+GPU |
| Correlation | ✓ | CorrelationF64 | — | CPU+GPU |
| Covariance | ✓ | CovarianceF64 | — | CPU+GPU |
| Dot product | ✓ | WeightedDotF64 | — | CPU+GPU |
| Quality filtering | ✓ | Custom WGSL | — | CPU+GPU |
| DADA2 E-step | ✓ | Custom WGSL | — | CPU+GPU |
| Taxonomy GEMM | ✓ | GemmCached | Candidate | CPU+GPU |
| Smith-Waterman | ✓ | **SmithWatermanGpu** (3/3) | — | **CPU+GPU** |
| Felsenstein pruning | ✓ | **FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| Decision tree | ✓ | **TreeInferenceGpu** (6/6) | Candidate | **CPU+GPU** |
| Gillespie SSA | ✓ | **GillespieGpu** (driver skip) | — | CPU+GPU* |
| HMM forward/Viterbi | ✓ | **Local WGSL** (13/13) | — | **CPU+GPU** |
| Bootstrap resampling | ✓ | **Compose FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| K-mer counting | ✓ | Blocked (lock-free hash) | — | CPU only |
| ODE integration (RK4) | ✓ | **Local WGSL** (7/7) | — | **CPU+GPU** |
| Phylogenetic placement | ✓ | **Compose FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| Bifurcation eigenvalues | ✓ | **BatchedEighGpu** (5/5) | — | **CPU+GPU** |
| Neighbor-Joining | ✓ | Tier C | — | CPU only |
| DTL Reconciliation | ✓ | Tier C | — | CPU only |
| Robinson-Foulds | ✓ | Tier C | — | CPU only |
| Signal processing | ✓ | Tier C | — | CPU only |
| **— Track 1c —** | | | | |
| ANI (pairwise identity) | 24/24 | **Local WGSL** (7/7) | — | **CPU+GPU** |
| SNP calling | 24/24 | **Local WGSL** (5/5) | — | **CPU+GPU** |
| dN/dS (Nei-Gojobori) | 22/22 | **Local WGSL** (9/9) | — | **CPU+GPU** |
| Molecular clock | 15/15 | Tier C | — | CPU only |
| Pangenome analysis | 24/24 | **Local WGSL** (6/6) | — | **CPU+GPU** |
| Rare biosphere diversity | 35/35 | Lean (diversity) | — | CPU+GPU via existing |
| **— ML Ensembles —** | | | | |
| Random Forest | 29/29 | **Local WGSL** (13/13) | Candidate | **CPU+GPU** |
| GBM (binary + multi) | 29/29 | CPU (sequential) | — | CPU only |

*\* = Shader absorbed but driver skip on RTX 4070 (Gillespie uses native f64 exp)*

### Tier Legend
- **CPU+GPU**: Validated on both, identical results
- **✓**: Validated in 157/157 CPU parity battery (25 domains)
- **Tier C**: CPU-only (sequential/branching algorithms)
- **Candidate**: NPU deployment possible with quantization

---

## Totals

| Substrate | Validated Checks | Algorithms |
|-----------|:----------------:|:----------:|
| CPU (Rust) | 1,241 | 25 domains (18 original + 5 Track 1c + 2 ML ensemble) |
| GPU (wgpu) | 260 | 30 promoted (15 ToadStool + 9 local WGSL + 6 composed) |
| NPU | 0 | 0 (3 candidates) |
| **Total** | **1,501** | — |

---

## Cross-System Workflow

```
               ┌──────────────────────────────────────────────┐
               │           wetSpring Pipeline                  │
               └──────────────────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────┐      ┌──────────────┐       ┌──────────────┐
│   CPU   │      │     GPU      │       │     NPU      │
│ i9-12900│      │ RTX 4070     │       │ AKD1000      │
│         │      │ Titan V      │       │              │
│ Parse   │      │ Diversity    │       │ Taxonomy     │
│ NJ/DTL  │  ──→ │ Spectral     │  ──→  │ Anomaly      │
│ Chimera │      │ PCoA/Eigen   │       │ PFAS screen  │
│ Signal  │      │ QF + DADA2   │       │              │
│ MolClk  │      │ Phylo/HMM   │       │              │
│         │      │ ODE sweep    │       │              │
│         │      │ ANI/SNP/Pan  │       │              │
│         │      │ dN/dS        │       │              │
│         │      │ RF ensemble  │       │              │
└─────────┘      └──────────────┘       └──────────────┘
  125-241W            200W                   ~1W
  Sequential         Batch-parallel         Ultra-low-power
  Reference truth    20-926× speedup        Field deployment
```
