# Experiment 095 — Cross-Spring Scaling Benchmark

| Field   | Value |
|---------|-------|
| Script  | `benchmark_cross_spring_scaling` |
| Binary  | `cargo run --release --features gpu --bin benchmark_cross_spring_scaling` |
| Status  | **PASS** |
| Date    | 2026-02-22 |
| Phase   | 22 |
| GPU     | RTX 4070 (Ada Lovelace, f64 1:2) |

## Purpose

Benchmark cross-spring evolved primitives at realistic bioinformatics
problem sizes. Demonstrates GPU scaling advantage and traces each shader's
origin through the ecoPrimals biome.

## Results (Release Mode, RTX 4070)

| Primitive | Evolved By | Problem Size | CPU (µs) | GPU (µs) | Speedup |
|-----------|-----------|-------------|----------|----------|---------|
| PairwiseHamming | neuralSpring | 500×1000 (125K pairs) | 15,999 | 978 | **16.4×** |
| PairwiseJaccard | neuralSpring | 200×2000 (20K pairs) | 41,780 | 151 | **276.7×** |
| SpatialPayoff | neuralSpring | 256×256 (65K cells) | 1,019 | 52 | **19.6×** |
| BatchFitness | neuralSpring | 4096×256 (1M elems) | 537 | 82 | **6.5×** |
| LocusVariance | neuralSpring | 100×10K (1M elems) | 1,097 | 57 | **19.2×** |
| FusedMapReduce | hotSpring | 100K f64 | <1 | 2,699 | N/A |
| GemmF64 | wetSpring | 256×256 f64 | 3,684 | 3,463 | **1.1×** |

## Key Findings

### GPU-Dominant Primitives (neuralSpring)

- **PairwiseJaccard** achieves **277×** speedup — the O(N²·G) column-major
  access pattern maps perfectly to GPU coalesced reads
- **SpatialPayoff** and **LocusVariance** both show ~20× GPU advantage
- **PairwiseHamming** at 16× despite O(N²·L) complexity
- **BatchFitness** at 6.5× — dot-product dominated, GPU memory-bound

### Transfer-Dominated Primitives

- **FusedMapReduce** at 100K elements: CPU completes in <1µs (release optimized),
  GPU overhead dominates at this size. At 1M+ elements, GPU wins (see Exp066)
- **GemmF64** at 256×256: near parity. The 60× speedup documented in Exp018
  was at 1024×1024+ where compute dominates transfer

## Cross-Spring Evolution Flow

```
neuralSpring  → wrote 5 bio/evolutionary shaders (metalForge Feb 21)
                ↓
ToadStool     → absorbed (Session 31f)
                ↓
wetSpring     → consumes at 6.5× – 277× GPU speedup
                ↓
hotSpring     → benefits from wetSpring GEMM (60×) and f64 precision fixes
```

Every Spring contributes. Every Spring benefits.
