# Experiment 071: BarraCUDA GPU — Math Portability Proof

**Date:** 2026-02-21
**Status:** COMPLETE — 11-domain GPU portability proof (24/24 checks PASS)
**Track:** GPU / barracuda

## Objective

Prove that all GPU-eligible domains produce identical results on GPU as on CPU.
This is the definitive "math is truly portable" proof: same Rust math,
same answers, different substrate.

Consolidates all GPU validators (Exp044, 045, 046-050, 058, 060, 063-065)
into one binary that covers every GPU-promoted domain with CPU↔GPU parity
checks and timing comparison.

## GPU Domains

| # | Domain | Primitive | Type |
|---|--------|-----------|------|
| 1 | Shannon entropy | FMR (ToadStool) | Absorbed |
| 2 | Simpson diversity | FMR (ToadStool) | Absorbed |
| 3 | Bray-Curtis | BrayCurtisF64 (ToadStool) | Absorbed |
| 4 | Spectral cosine | FMR (ToadStool) | Absorbed |
| 5 | PCoA eigenvalues | BatchedEighGpu (ToadStool) | Absorbed |
| 6 | Smith-Waterman | SmithWatermanGpu (barracuda) | Absorbed |
| 7 | Gillespie SSA | GillespieGpu (barracuda) | Absorbed |
| 8 | Decision Tree | TreeInferenceGpu (barracuda) | Absorbed |
| 9 | Felsenstein pruning | FelsensteinGpu (ToadStool) | Absorbed |
| 10 | HMM forward | Local WGSL | Handoff candidate |
| 11 | ANI pairwise | Local WGSL | Handoff candidate |
| 12 | SNP calling | Local WGSL | Handoff candidate |
| 13 | dN/dS | Local WGSL | Handoff candidate |
| 14 | Pangenome classify | Local WGSL | Handoff candidate |
| 15 | Random Forest | Local WGSL | Handoff candidate |

## Protocol

1. For each domain: compute on CPU, compute on GPU
2. Validate GPU == CPU within `tolerances::GPU_VS_CPU_*`
3. Time both CPU and GPU at representative batch sizes
4. Summary table: Domain | CPU µs | GPU µs | Speedup | Parity

## Provenance

| Field | Value |
|-------|-------|
| Baseline commit | current HEAD |
| Baseline tool | BarraCUDA CPU implementation (same crate) |
| Baseline date | 2026-02-21 |
| Exact command | `cargo run --release --features gpu --bin validate_barracuda_gpu_full` |
| Data | Synthetic test vectors (hardcoded, reproducible) |
| Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!_OS 22.04) |
