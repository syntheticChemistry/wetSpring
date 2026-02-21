# Experiment 076: metalForge Cross-Substrate Pipeline

**Date:** February 21, 2026
**Status:** Active
**Binary:** `validate_cross_substrate_pipeline`

---

## Objective

Demonstrate a heterogeneous compute pipeline that routes workloads across
GPU, NPU, and CPU substrates based on workload characteristics, with
end-to-end latency profiling for each substrate transition.

## Architecture

```
CPU: Generate synthetic abundance + spectral data
  │
  ├── Stage 1: GPU batch-parallel ─────────────────────────────┐
  │   Alpha diversity (FMR), Bray-Curtis (BrayCurtisF64),      │
  │   spectral cosine (GEMM), variance/correlation (FMR)       │
  │   → outputs feature vector per sample                      │
  │                                                            ▼
  ├── Stage 2: NPU inference path ─────────────────────────────┐
  │   Route: GPU output → classification                       │
  │   If NPU available: AKD1000 FC inference (~650µs)          │
  │   If NPU unavailable: CPU fallback (NB classifier)         │
  │   → classification label per sample                        │
  │                                                            ▼
  └── Stage 3: CPU aggregation ────────────────────────────────┘
      Collect GPU metrics + classification results
      Compute summary statistics
      Generate final report
```

## Protocol

### Substrate Transitions Profiled
| Transition | PCIe Path | Expected Latency |
|------------|-----------|------------------|
| CPU → GPU upload | Host → PCIe Gen4 x16 → RTX 4070 | ~100µs |
| GPU compute (diversity) | On-device | ~5-500µs |
| GPU → CPU readback | PCIe Gen4 x16 → Host | ~100µs |
| CPU → NPU inference | Host → PCIe 2.0 x1 → AKD1000 | ~650µs |
| Total pipeline | End-to-end | Profiled |

### Validation Checks
1. GPU diversity matches CPU reference for all samples
2. Classification routing is correct (NPU or CPU fallback)
3. CPU aggregation produces valid summary statistics
4. Per-stage latency is measured and reported
5. Cross-substrate results are consistent

## Expected Outcome
- Full heterogeneous pipeline demonstrated on Eastgate hardware
- Latency breakdown shows where time is spent per substrate
- NPU routing works (with CPU fallback when model unavailable)
- Foundation for production mixed-hardware deployment
