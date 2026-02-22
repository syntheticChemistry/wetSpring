# Experiment 094 — Cross-Spring Evolution Validation

| Field   | Value |
|---------|-------|
| Script  | `validate_cross_spring_evolution` |
| Binary  | `cargo run --features gpu --bin validate_cross_spring_evolution` |
| Status  | **PASS** (39/39 checks) |
| Date    | 2026-02-22 |
| Phase   | 22 |
| GPU     | RTX 4070 (Ada Lovelace, f64 1:2) |

## Purpose

Validate and benchmark 5 neuralSpring-evolved primitives now consumed by
wetSpring through ToadStool absorption. Proves the ecoPrimals biome model:
each Spring evolves locally, ToadStool absorbs, all Springs benefit.

## Cross-Spring Provenance Map

| Primitive | Evolved By | Absorbed Session | Now Used By |
|-----------|-----------|-----------------|-------------|
| `PairwiseHammingGpu` | neuralSpring | Session 31f | wetSpring |
| `PairwiseJaccardGpu` | neuralSpring | Session 31f | wetSpring |
| `SpatialPayoffGpu` | neuralSpring | Session 31f | wetSpring |
| `BatchFitnessGpu` | neuralSpring | Session 31f | wetSpring |
| `LocusVarianceGpu` | neuralSpring | Session 31f | wetSpring |

## Results

All 5 neuralSpring-evolved primitives pass CPU↔GPU parity checks:

- **PairwiseHamming**: 10/10 pairs match (tol 1e-6)
- **PairwiseJaccard**: 6/6 pairs match (tol 1e-5)
- **SpatialPayoff**: 64/64 cells match (tol 1e-4)
- **BatchFitness**: 16/16 individuals match (tol 1e-5)
- **LocusVariance**: 6/6 loci match (tol 1e-5)

### Key Finding: LocusVariance Layout

The `LocusVarianceGpu` shader uses row-major layout `allele_freqs[pop * n_loci + locus]`,
matching Weir-Cockerham FST convention. CPU baselines must match this layout
for correct parity validation.

## Checks

- 39 GPU parity checks, all PASS
- CPU baselines computed inline with matching data layouts
- Buffer-based API (raw `wgpu::Buffer` in, readback for comparison)

## Evolution Lifecycle Proved

```
neuralSpring (Write) → ToadStool (Absorb) → wetSpring (Lean)
```

This is the first experiment proving the reverse flow: wetSpring consuming
primitives evolved by another Spring (neuralSpring), completing the full
biome cycle.
