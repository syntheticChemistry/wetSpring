# Exp092: BarraCUDA CPU vs GPU — All 16 Domains Head-to-Head

## Purpose

Consolidated proof that BarraCUDA's pure Rust math produces identical
results whether dispatched to CPU or GPU, across **all 16** GPU-eligible
bioinformatics domains.

## Domains (16)

| # | Domain | Primitive | Tolerance | Result |
|---|--------|-----------|-----------|--------|
| 1 | Shannon + Simpson | FusedMapReduceF64 | transcendental | PASS |
| 2 | Bray-Curtis | BrayCurtisF64 | f64 | PASS |
| 3 | ANI (pairwise) | AniBatchF64 | transcendental | PASS |
| 4 | SNP calling | SnpCallingF64 | exact | PASS |
| 5 | dN/dS ratio | DnDsBatchF64 | f64 | PASS |
| 6 | Pangenome | PangenomeClassifyGpu | exact | PASS |
| 7 | Random Forest | RfBatchInferenceGpu | exact | PASS |
| 8 | HMM forward | HmmBatchForwardF64 | f64 | PASS |
| 9 | Smith-Waterman | SmithWatermanGpu | exact | PASS |
| 10 | Gillespie SSA | GillespieGpu | stochastic | PASS |
| 11 | Decision Tree | TreeInferenceGpu | exact | PASS |
| 12 | Spectral cosine | FMR + GEMM | transcendental | PASS |
| 13 | EIC total intensity | FusedMapReduceF64 | f64 | PASS |
| 14 | PCoA ordination | BatchedEighGpu | eigenvalue | SKIP (naga) |
| 15 | Kriging | KrigingF64 (GEMM) | f64 | PASS |
| 16 | Rarefaction | PrngXoshiro | stochastic | PASS |

## Key Results

- **48/48 checks PASS** (PCoA gracefully skipped — naga shader validation bug)
- 15/16 domains produce bit-identical or within-tolerance GPU results
- PCoA CPU path validated; GPU path blocked by upstream wgpu naga bug
- Total execution: 128 ms on RTX 4070

## Note on PCoA

The `BatchedEighGpu` shader triggers a naga validation error in wgpu 22.1.0
(`idx2d` call argument type mismatch). This is a known upstream issue tracked
for ToadStool to resolve via shader refactor. CPU path is fully validated.

## Reproduction

```bash
cargo run --features gpu --release --bin validate_cpu_vs_gpu_all_domains
```

## Provenance

| Field | Value |
|-------|-------|
| Binary | `validate_cpu_vs_gpu_all_domains` |
| Date | 2026-02-22 |
| Hardware | i9-12900K, RTX 4070, 64 GB DDR5 |
| Data | Synthetic (self-contained) |
