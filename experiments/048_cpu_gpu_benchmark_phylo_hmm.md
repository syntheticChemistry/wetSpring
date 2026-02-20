# Experiment 048: CPU vs GPU Benchmark (Phylo + HMM)

**Date:** February 20, 2026
**Status:** COMPLETE — 6/6 PASS (all parity checks)
**Track:** Cross-cutting (GPU, benchmarking)
**Binary:** `benchmark_phylo_hmm_gpu` (requires `--features gpu`)

---

## Objective

Head-to-head CPU vs GPU performance benchmark for the newly GPU-composed domains: Felsenstein pruning, bootstrap resampling (RAWR), and HMM batch forward. Document timing and scaling characteristics at current workload sizes.

## Method

1. **Parity verification** — All benchmark runs include CPU↔GPU math comparison; fail if max diff exceeds tolerance
2. **Timing** — Microsecond-precision wall-clock for CPU vs GPU
3. **Workload sizes** — Representative small-to-medium scales (16 taxa × 512 sites, 100 replicates, 256 sequences × 100 steps)

## Results

### Parity: 6/6 Checks Pass

All benchmarks verify CPU ≈ GPU before reporting timings. Max observed diff: 3.95e-9.

### Felsenstein: 16 taxa × 512 sites
| Platform | Time (µs) | Notes |
|----------|-----------|-------|
| CPU | 90 | Recursive prune |
| GPU | 46,676 | Dispatch overhead dominates |

**Ratio:** GPU slower — dispatch overhead dominates at small N

### Bootstrap: 100 replicates × 16 taxa × 512 sites
| Platform | Time (µs) | Notes |
|----------|-----------|-------|
| CPU | 9,922 | 100 independent Felsenstein calls |
| GPU | 272,180 | Same; 100 dispatches |

**Ratio:** 0.04× (GPU 25× slower) — same reason

### HMM Batch: 256 sequences × 100 steps × 3 states
| Platform | Time (µs) | Notes |
|----------|-----------|-------|
| CPU | 2,879 | Sequential over time, parallel over sequences |
| GPU | 58,519 | One dispatch per batch |

**Ratio:** 0.05× (GPU 20× slower)

## Key Findings

1. **Math is portable and identical** — max CPU↔GPU diff = 3.95e-9 across all runs
2. **GPU speedup requires scale** — Existing spectral cosine benchmark shows 926× at 200×200 matrix size
3. **Dispatch overhead dominates at small N** — Per-dispatch wgpu overhead (~tens of ms) exceeds compute at 16 taxa, 100 replicates, 256 sequences
4. **ToadStool unidirectional streaming** amortizes dispatch overhead for production workloads — single large batch beats many small dispatches

## References

- Exp016: GPU pipeline parity (DADA2 24× at scale)
- Exp044: BarraCUDA GPU v3 (spectral cosine 926×)
- Exp046: GPU phylo composition
- Exp047: GPU HMM forward

## Files Changed

| File | Purpose |
|------|---------|
| `barracuda/src/bin/benchmark_phylo_hmm_gpu.rs` | CPU vs GPU head-to-head benchmark |

## Run

```bash
cargo run --bin benchmark_phylo_hmm_gpu --features gpu
```
