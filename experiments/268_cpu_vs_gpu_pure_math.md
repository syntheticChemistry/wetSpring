# Exp268: CPU vs GPU Pure Math — ToadStool Primitives

| Field | Value |
|-------|-------|
| Binary | `validate_cpu_vs_gpu_pure_math` |
| Date | 2026-03-01 |
| GPU | NVIDIA GeForce RTX 4070 (f64 native) |
| Command | `cargo run --features gpu --bin validate_cpu_vs_gpu_pure_math` |
| Checks | 38/38 PASS |

## Purpose

Validates that every ToadStool GPU primitive produces identical results to its CPU
counterpart. This is the deepest layer of parity: not wetSpring bio domains, but
the underlying barracuda math operations themselves.

## Sections

| Section | Primitive | Checks |
|---------|-----------|--------|
| S1 | `FusedMapReduceF64` — Shannon, Simpson, Observed at 4 sizes | 12 |
| S2 | `BrayCurtisF64` — pairwise distance matrix (3 samples) | 4 |
| S3 | `BatchedEighGpu` — eigendecomposition via PCoA (4×4) | 4 |
| S4 | Graph Laplacian + Anderson spectral | 8 |
| S5 | DF64 pack/unpack at GPU boundary (6 values + slice) | 7 |
| S6 | `GpuPipelineSession` — streaming vs individual determinism | 3 |

## Validation Chain

CPU v20 → **CPU↔GPU Parity (this)** → ToadStool v3

## Key Results

- S1: Shannon/Simpson/Observed match CPU↔GPU at all sizes (32, 256, 1024, 4096)
- S2: Bray-Curtis condensed matrix all 3 pairs within `GPU_VS_CPU_F64` tolerance
- S3: PCoA eigenvalues and variance explained within `GPU_VS_CPU_F64`
- S5: DF64 roundtrip error exactly 0.0 for π, e, 1e-15, 1e20, −π, 0
- S6: 5 consecutive streaming runs produce bit-identical results
