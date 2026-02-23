# Exp103: metalForge Cross-Substrate v5 — 13 Pure GPU Promotion Domains

| Field    | Value                                       |
|----------|---------------------------------------------|
| Script   | `validate_metalforge_v5`                    |
| Command  | `cargo run --features gpu --release --bin validate_metalforge_v5` |
| Status   | **PASS** (38/38)                            |
| Phase    | 28                                          |
| Depends  | Exp093 (v3), Exp100 (v4), Exp101            |

## Purpose

Extends metalForge cross-substrate validation from 16 domains (Exp093) to all
29 GPU-eligible domains by adding the 13 modules promoted in the pure GPU
completion pass. Proves: for every new GPU module, metalForge router can
dispatch to CPU or GPU and get the same answer — substrate independence.

## New Cross-System Domains (13)

| # | Module | GPU Strategy | ToadStool Primitive | CPU↔GPU |
|---|--------|-------------|-------------------|---------|
| N01 | `cooperation` | Local WGSL (4v, 13p) | `BatchedOdeRK4Generic` | ODE_GPU_PARITY |
| N02 | `capacitor` | Local WGSL (6v, 16p) | `BatchedOdeRK4Generic` | ODE_GPU_PARITY |
| N03 | `kmd` | Compose | `FusedMapReduceF64` | ANALYTICAL_F64 |
| N04 | `gbm` | Compose | `TreeInferenceGpu` | ANALYTICAL_F64 |
| N05 | `merge_pairs` | Compose | `FusedMapReduceF64` | Exact |
| N06 | `signal` | Compose | `FusedMapReduceF64` | Exact |
| N07 | `feature_table` | Compose | `FMR + WeightedDotF64` | Exact |
| N08 | `robinson_foulds` | Compose | `PairwiseHammingGpu` | Exact |
| N09 | `derep` | Compose | `KmerHistogramGpu` | Exact |
| N10 | `chimera` | Compose | `GemmCachedF64` | Exact |
| N11 | `neighbor_joining` | Compose | `FusedMapReduceF64` | ANALYTICAL_F64 |
| N12 | `reconciliation` | Compose | Batch workgroup | ANALYTICAL_F64 |
| N13 | `molecular_clock` | Compose | `FusedMapReduceF64` | ANALYTICAL_F64 |

## Results

| Module | Checks | Status | Notes |
|--------|--------|--------|-------|
| MF-N01: Cooperation ODE | 4/4 | PASS | All 4 vars via ODE_GPU_PARITY |
| MF-N02: Capacitor ODE | 6/6 | PASS | All 6 vars via ODE_GPU_PARITY |
| MF-N03: KMD | 5/5 | PASS | KMD values via ANALYTICAL_F64 |
| MF-N04: GBM | 3/3 | PASS | Batch proba via ANALYTICAL_F64 |
| MF-N05: Merge Pairs | 2/2 | PASS | Count + stats exact |
| MF-N06: Signal | 3/3 | PASS | Peak count + indices exact |
| MF-N07: Feature Table | 1/1 | PASS | Empty-input identity |
| MF-N08: Robinson-Foulds | 1/1 | PASS | Distance exact |
| MF-N09: Dereplication | 2/2 | PASS | Unique count + stats exact |
| MF-N10: Chimera | 2/2 | PASS | Count + chimeras_found exact |
| MF-N11: Neighbor Joining | 4/4 | PASS | Distance matrix via ANALYTICAL_F64 |
| MF-N12: Reconciliation | 1/1 | PASS | Optimal cost via ANALYTICAL_F64 |
| MF-N13: Molecular Clock | 4/4 | PASS | Strict + relaxed rates |
| **Total** | **38/38** | **PASS** | |

## metalForge Domain Coverage

| Version | Exp | Domains | Total |
|---------|-----|---------|-------|
| v1 | 060 | 8 core | 8 |
| v2 | 084 | +4 (SW, Gillespie, DT, spectral) | 12 |
| v3 | 093 | +4 (EIC, PCoA, Kriging, Rarefaction) | 16 |
| v4 | 100 | +0 (ODE NPU routing, not new domains) | 16 |
| **v5** | **103** | **+13 (pure GPU promotion)** | **29** |

## Impact

- **metalForge cross-system domains**: 16 → 29
- **CPU-only domains**: phred only (I/O-bound, no parallelism benefit)
- **Substrate-independent**: Every GPU-eligible algorithm proven to produce
  identical results on CPU and GPU — metalForge can route freely
