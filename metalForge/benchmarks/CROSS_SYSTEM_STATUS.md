# Cross-System Benchmark Status

**Date:** February 21, 2026
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
- **✓**: Validated in 205/205 CPU parity battery (25 domains + 6 ODE flat)
- **Tier C**: CPU-only (sequential/branching algorithms)
- **Candidate**: NPU deployment possible with quantization

---

## Totals

| Substrate | Validated Checks | Algorithms |
|-----------|:----------------:|:----------:|
| CPU (Rust) | 1,291 | 25 domains (18 original + 5 Track 1c + 2 ML ensemble) |
| GPU (wgpu) | 345 | 30 promoted (23 ToadStool + 4 local WGSL shaders + 6 composed + consolidated) |
| NPU | 0 | 0 (3 candidates) |
| **Total** | **1,636** | — |

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
