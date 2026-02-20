# Cross-System Benchmark Status

**Date:** February 20, 2026
**Purpose:** Track which algorithms are validated on which substrates

---

## Algorithm × Substrate Matrix

| Algorithm | CPU (Rust) | GPU (wgpu) | NPU (AKD1000) | Status |
|-----------|:----------:|:----------:|:--------------:|--------|
| Shannon entropy | 84/84 | FusedMapReduceF64 | — | CPU+GPU |
| Simpson diversity | 84/84 | FusedMapReduceF64 | — | CPU+GPU |
| Pielou evenness | 84/84 | FusedMapReduceF64 | — | CPU+GPU |
| Bray-Curtis (pairwise) | 84/84 | BrayCurtisF64 | — | CPU+GPU |
| Observed features | 84/84 | FusedMapReduceF64 | — | CPU+GPU |
| PCoA ordination | 84/84 | BatchedEighGpu | — | CPU+GPU |
| Spectral cosine | 84/84 | GemmF64 + FMR | — | CPU+GPU |
| Variance/StdDev | 84/84 | VarianceF64 | — | CPU+GPU |
| Correlation | 84/84 | CorrelationF64 | — | CPU+GPU |
| Covariance | 84/84 | CovarianceF64 | — | CPU+GPU |
| Dot product | 84/84 | WeightedDotF64 | — | CPU+GPU |
| Quality filtering | 84/84 | Custom WGSL | — | CPU+GPU |
| DADA2 E-step | 84/84 | Custom WGSL | — | CPU+GPU |
| Taxonomy GEMM | 84/84 | GemmCached | Candidate | CPU+GPU |
| Smith-Waterman | 84/84 | **SmithWatermanGpu** (3/3) | — | **CPU+GPU** |
| Felsenstein pruning | 84/84 | **FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| Decision tree | 84/84 | **TreeInferenceGpu** (6/6) | Candidate | **CPU+GPU** |
| Gillespie SSA | 84/84 | **GillespieGpu** (driver skip) | — | CPU+GPU* |
| HMM forward/Viterbi | 84/84 | **Local WGSL** (13/13) | — | **CPU+GPU** |
| Bootstrap resampling | 84/84 | **Compose FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| K-mer counting | 84/84 | Blocked (lock-free hash) | — | CPU only |
| ODE integration (RK4) | 84/84 | **Local WGSL** (7/7) | — | **CPU+GPU** |
| Phylogenetic placement | 84/84 | **Compose FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| Bifurcation eigenvalues | 84/84 | **BatchedEighGpu** (5/5) | — | **CPU+GPU** |
| Neighbor-Joining | 84/84 | Tier C | — | CPU only |
| DTL Reconciliation | 84/84 | Tier C | — | CPU only |
| Robinson-Foulds | 84/84 | Tier C | — | CPU only |
| Signal processing | 84/84 | Tier C | — | CPU only |

*\* = Shader absorbed but driver skip on RTX 4070 (Gillespie uses native f64 exp)*

### Tier Legend
- **CPU+GPU**: Validated on both, identical results
- **Tier B**: CPU-validated, GPU-ready data layouts, needs ToadStool shader
- **Tier C**: CPU-only (sequential/branching algorithms)
- **Candidate**: NPU deployment possible with quantization

---

## Totals

| Substrate | Validated Checks | Algorithms |
|-----------|:----------------:|:----------:|
| CPU (Rust) | 1,035 | 18 domains (all) |
| GPU (wgpu) | 200 | 25 promoted (15 ToadStool + 4 local WGSL + 6 composed) |
| NPU | 0 | 0 (2 candidates) |
| **Total** | **1,235** | — |

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
│         │      │ Phylo/HMM   │       │              │
│         │      │ ODE sweep    │       │              │
└─────────┘      └──────────────┘       └──────────────┘
  125-241W            200W                   ~1W
  Sequential         Batch-parallel         Ultra-low-power
  Reference truth    20-926× speedup        Field deployment
```
