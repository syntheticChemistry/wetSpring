# wetSpring V114 → BarraCUDA/ToadStool Absorption Handoff

**Date:** March 15, 2026
**From:** wetSpring V114 (374 experiments, 5,707+ checks, 1,326 tests)
**To:** BarraCUDA/ToadStool team
**Authority:** wateringHole (ecoPrimals Core Standards)

---

## Executive Summary

wetSpring has consumed 150+ BarraCUDA primitives and achieved fully lean status
(0 local WGSL, 0 local ODE math, 0 local regression math). All 47 GPU modules
consume upstream primitives exclusively.

This handoff documents **8 GPU primitive opportunities** discovered through
wetSpring's 19 biomeOS IPC capabilities — algorithms that work today on CPU but
would benefit from GPU acceleration via new BarraCUDA primitives.

## New Primitive Opportunities

### Priority 1: High Impact

#### 1. `MatMulF64` / `SparseGemmF64` — NMF GPU Acceleration

**Source:** `ipc/handlers/expanded.rs` → `nmf_mu()` (multiplicative update NMF)
**Current:** CPU-only Lee & Seung algorithm with `matmul`, `matmul_t_a`, `matmul_t_b`
**Need:** Dense and sparse GEMM for W×H, Wᵀ×V, W×Hᵀ decompositions
**Precision:** F64 required (convergence-sensitive iterative algorithm)
**Workload:** Matrices typically 100-10,000 rows × 2-50 rank, 200 iterations
**GPU benefit:** NMF is embarrassingly parallel — each matrix multiply is independent
**BarraCUDA exists:** `GemmF64` exists but not `SparseGemmF64`; NMF needs both
**Test:** `test_handle_nmf_basic` validates convergence, reconstruction error

```
V_approx = W × H
W_update: W ← W * (Vᵀ×H) / (W×H×Hᵀ)   — needs matmul + element-wise
H_update: H ← H * (Wᵀ×V) / (Wᵀ×W×H)   — needs matmul_t_a + element-wise
```

#### 2. `AdaptiveOdeGpu` — General-Purpose ODE Solver

**Source:** `ipc/handlers/expanded.rs` → `handle_kinetics()` (Gompertz, first-order)
**Current:** CPU analytical solutions for specific kinetics models
**Need:** GPU RK45 adaptive-step ODE solver for arbitrary reaction networks
**Precision:** F64 required (stiff systems, long integration windows)
**Workload:** Batch 10-10,000 parameter sets simultaneously
**GPU benefit:** Parameter sweeps are independent — perfect for batched dispatch
**BarraCUDA exists:** `BatchedOdeRK4` exists for fixed-step; no adaptive variant
**Note:** wetSpring's 5 ODE systems (bistable, capacitor, cooperation, multi-signal,
defense) already use `BatchedOdeRK4` via `generate_shader()`. An adaptive variant
would enable stiff system support without manual step tuning.

### Priority 2: Medium Impact

#### 3. `SmithWatermanBatchGpu` — Batched Local Alignment

**Source:** `ipc/handlers/expanded.rs` → `handle_alignment()`
**Current:** Uses `bio::alignment::SmithWaterman::align()` (CPU, single-pair)
**Need:** Batched alignment for IPC workloads (many pairs per request)
**Precision:** Integer scoring matrix (BLOSUM/custom)
**GPU benefit:** O(m×n) per pair — batch amortizes dispatch overhead
**BarraCUDA exists:** `SmithWatermanGpu` exists but single-pair; need batch variant

#### 4. `KmerHashBatchGpu` — Batched K-mer Hashing for Taxonomy

**Source:** `ipc/handlers/expanded.rs` → `handle_taxonomy()`
**Current:** CPU Naive Bayes with per-sequence k-mer extraction
**Need:** GPU k-mer extraction + log-probability accumulation
**Precision:** Mixed (integer hashing, F64 log-probabilities)
**GPU benefit:** K-mer extraction is embarrassingly parallel per position
**BarraCUDA exists:** `KmerHistogramF64` exists; need `KmerHashBatchGpu` for
streaming k-mer extraction + in-flight classification

#### 5. `TreeTraversalBatchGpu` — Phylogenetic Distance

**Source:** `ipc/handlers/expanded.rs` → `handle_phylogenetics()`
**Current:** Robinson-Foulds via `bio::robinson_foulds` (CPU)
**Need:** Batched tree comparison for large phylogenetic analyses
**Precision:** Integer (bipartition set operations)
**GPU benefit:** Bipartition hashing and set comparison are parallelizable
**BarraCUDA exists:** `FelsensteinGpu` exists for likelihood; RF is different

### Priority 3: Lower Impact (Already Functional)

#### 6. `DiversityStreamGpu` — Streaming Diversity on Time Series

**Source:** `ipc/timeseries.rs` → `handle_timeseries_diversity()`
**Current:** CPU Shannon/Simpson/Evenness computation
**Need:** Streaming GPU diversity for real-time dashboard integration
**BarraCUDA exists:** `BrayCurtisF64`, `FusedMapReduceF64` — mostly covered

#### 7. `StatisticsStreamGpu` — Time Series Analysis

**Source:** `ipc/timeseries.rs` → `handle_timeseries()`
**Current:** CPU mean/variance/trend computation
**Need:** GPU streaming statistics for large time series
**BarraCUDA exists:** `FusedMapReduceF64` covers reduce ops

#### 8. `SignalConvGpu` — Continuous Peak Detection

**Source:** `science.peak_detect` capability (existing, fully lean)
**Current:** Already uses `PeakDetectF64` (S62)
**Need:** Streaming convolution for continuous monitoring
**BarraCUDA exists:** `PeakDetectF64` covers batch; streaming variant would help

## F64/DF64 Precision Requirements

| Primitive | F64 Required | DF64 Viable | Rationale |
|-----------|-------------|-------------|-----------|
| NMF (matmul) | Yes | Yes | Iterative convergence sensitive to precision |
| ODE (adaptive RK45) | Yes | Yes | Stiff systems accumulate error |
| Smith-Waterman | No (integer) | N/A | Scoring is integer arithmetic |
| K-mer hashing | No (integer) | N/A | Hash functions are integer |
| Robinson-Foulds | No (integer) | N/A | Set operations are integer |
| Diversity | Yes | DF64 ok | Log/entropy calculations |
| Statistics | Yes | DF64 ok | Variance computation |
| Peak detection | Yes | DF64 ok | Already validated with F64 |

## wetSpring Test Validation Available

Every algorithm above has existing CPU validation tests in wetSpring that can
serve as ground truth for GPU implementations:

| Primitive | Test | Expected Output |
|-----------|------|-----------------|
| NMF | `test_handle_nmf_basic` | Convergence < 200 iter, error decreasing |
| Kinetics | `test_handle_kinetics_gompertz` | 5-param Gompertz curve match |
| Alignment | `test_handle_alignment_basic` | Known SW score for ACGT pairs |
| Taxonomy | `test_handle_taxonomy_basic` | Classification with log-prob |
| Phylogenetics | `test_handle_phylogenetics_basic` | RF distance for known trees |
| Timeseries | `test_handle_timeseries_basic` | Mean/variance/trend for known data |
| Diversity | `test_handle_timeseries_diversity` | Shannon/Simpson for known abundances |

## Current Absorption Status

All 150+ consumed primitives are working and lean. No regressions since V107.
The `Write → Absorb → Lean` cycle is complete for all existing modules.

New primitives from this handoff would extend coverage to the IPC science layer,
enabling GPU-accelerated responses to biomeOS `capability.call` requests.

## Upstream Requests

1. **`SparseGemmF64`** — CSR × Dense GEMM (Priority 1 for NMF)
2. **`BatchedOdeRK4Adaptive`** — RK45 with step control (Priority 1 for kinetics)
3. **`SmithWatermanBatchGpu`** — Multi-pair batch variant (Priority 2)
4. **`KmerHashBatchGpu`** — Streaming k-mer + classification (Priority 2)
5. **`TreeTraversalBatchGpu`** — Batched RF distance (Priority 3)

## Quality Status

- `cargo check --features ipc,json` — clean
- `cargo clippy --features ipc,json -- -W clippy::pedantic -W clippy::nursery` — zero warnings
- `cargo test` — 1,326 tests pass, 0 fail
- All code: `#![forbid(unsafe_code)]`, zero `unwrap` outside tests
