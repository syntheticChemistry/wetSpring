# Cross-System Benchmark Status

**Date:** March 11, 2026 (V115)
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
| K-mer counting | ✓ | **KmerHistogramGpu** (Lean) | — | **CPU+GPU** |
| ODE integration (RK4) | ✓ | **Local WGSL** (5 shaders) | — | **CPU+GPU** |
| Phylogenetic placement | ✓ | **Compose FelsensteinGpu** (15/15) | — | **CPU+GPU** |
| Bifurcation eigenvalues | ✓ | **BatchedEighGpu** (5/5) | — | **CPU+GPU** |
| Neighbor-Joining | ✓ | **Compose GemmCachedF64** | — | **CPU+GPU** |
| DTL Reconciliation | ✓ | **Compose TreeInferenceGpu** | — | **CPU+GPU** |
| Robinson-Foulds | ✓ | **Compose PairwiseHammingGpu** | — | **CPU+GPU** |
| Signal processing | ✓ | **Lean** (PeakDetectF64 S62) | — | **CPU+GPU** |
| **— Track 1c —** | | | | |
| ANI (pairwise identity) | 24/24 | **Local WGSL** (7/7) | — | **CPU+GPU** |
| SNP calling | 24/24 | **Local WGSL** (5/5) | — | **CPU+GPU** |
| dN/dS (Nei-Gojobori) | 22/22 | **Local WGSL** (9/9) | — | **CPU+GPU** |
| Molecular clock | 15/15 | **Compose GemmCachedF64** | — | **CPU+GPU** |
| Pangenome analysis | 24/24 | **Local WGSL** (6/6) | — | **CPU+GPU** |
| Rare biosphere diversity | 35/35 | Lean (diversity) | — | CPU+GPU via existing |
| **— ML Ensembles —** | | | | |
| Random Forest | 29/29 | **Local WGSL** (13/13) | Candidate | **CPU+GPU** |
| GBM (binary + multi) | 29/29 | **Compose** (TreeInferenceGpu) | — | **CPU+GPU** |

*\* = Shader absorbed but driver skip on RTX 4070 (Gillespie uses native f64 exp)*

### Tier Legend
- **CPU+GPU**: Validated on both, identical results
- **✓**: Validated in 380/380 CPU parity battery (v1-v8, 31+ domains)
- **Tier C**: All promoted (0 remaining as of Phase 28)
- **Candidate**: NPU deployment possible with quantization

---

## Totals

| Substrate | Validated Checks | Algorithms |
|-----------|:----------------:|:----------:|
| CPU (Rust) | 1,476 | 31+ domains (v1-v8) |
| GPU (wgpu) | 702+ | 47 modules (34 Lean + 5 Write + 8 Compose, 0 Passthrough) |
| NPU | 0 | 0 (3 candidates) |
| **Total** | **2,178+** | — |

### Cross-System Proof (Exp064–067)

| Experiment | What It Proves |
|-----------|---------------|
| Exp064 (GPU Parity v1) | Pure GPU math correct across 8 consolidated domains |
| Exp065 (metalForge Full) | CPU↔GPU substrate-independence: same input → same output |
| Exp066 (Scaling) | GPU crossover: dN/dS at ~200 pairs; most domains CPU-optimal at small N |
| Exp067 (Dispatch) | GPU dispatch overhead: ~5ms average (local WGSL), ~1ms (ToadStool) |

### GPU Dispatch Overhead (Exp067 — measured on RTX 4070)

| Domain | CPU (µs) | GPU (µs) | Fixed Overhead |
|--------|:--------:|:--------:|:--------------:|
| Shannon (FMR) | <1 | 994 | ~1ms |
| Bray-Curtis | <1 | 335 | ~0.3ms |
| ANI (local WGSL) | <1 | 5,855 | ~6ms |
| SNP (local WGSL) | <1 | 10,169 | ~10ms |
| dN/dS (local WGSL) | 1 | 9,900 | ~10ms |
| Pangenome (local WGSL) | <1 | 5,788 | ~6ms |
| RF (local WGSL) | <1 | 2,493 | ~2.5ms |
| HMM (local WGSL) | 1 | 5,086 | ~5ms |

**Routing rule**: GPU wins when `batch_compute_time > dispatch_overhead`.
At small N, CPU always wins. ToadStool streaming amortizes dispatch across
chained stages (1 upload + N dispatches + 1 readback).

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
│ Chimera │  ──→ │ Spectral     │  ──→  │ Anomaly      │
│         │      │ PCoA/Eigen   │       │ PFAS screen  │
│         │      │ QF + DADA2   │       │              │
│         │      │ Phylo/HMM   │       │              │
│         │      │ ODE sweep    │       │              │
│         │      │ ANI/SNP/Pan  │       │              │
│         │      │ dN/dS + RF   │       │              │
│         │      │ NJ/DTL/MolClk│       │              │
│         │      │ Signal/KMD   │       │              │
└─────────┘      └──────────────┘       └──────────────┘
  125-241W            200W                   ~1W
  Sequential         Batch-parallel         Ultra-low-power
  Reference truth    20-926× speedup        Field deployment
```
