# Exp086: metalForge End-to-End Pipeline Proof

**Status**: PASS — 45/45 checks  
**Binary**: `validate_metalforge_pipeline`  
**Date**: 2026-02-22

## Purpose

Prove the complete evolution path works as a chained pipeline:
K-mer count → Taxonomy → UniFrac → dispatch routing, with each stage
correctly routed to the appropriate substrate and producing identical
results regardless of which substrate executes the computation.

## Evolution Chain Position

```
Python baseline → Rust CPU → Flat layouts (Exp085)
                                    ↓
                        Pipeline chaining (THIS) → GPU execution (next)
```

## Validated Stages

| Stage | Workload Class | Full Target | CPU Fallback | Checks |
|-------|---------------|-------------|--------------|--------|
| 1. Dispatch | 5 workload classes | GPU/NPU/CPU | CPU | 9 |
| 2. CPU pipeline | kmer+taxonomy+unifrac | CPU | — | 8 |
| 3. Parity | f64 ↔ int8, histogram, CSR | CPU ↔ NPU | CPU | 10 |
| 4. Flat layouts | GPU buffer sizing + structure | — | — | 10 |
| 5. Fallback | 4 hardware configs × 2 domains | All | CPU | 8 |

## Key Results

- **Dispatch correctness**: All 5 workload classes route to expected substrates
  across 4 hardware configurations (Full/GPU-only/NPU-only/CPU-only)
- **Pipeline parity**: f64 and int8 classification agree on all 4 test sequences
- **Layout fidelity**: K-mer histograms (4×256 flat), UniFrac CSR, taxonomy int8
  all produce GPU-ready buffers with correct sizes
- **Fallback safety**: CPU-only path produces identical results to mixed paths
- **Total new checks**: 45

## Reproduction

```bash
cargo run --release --bin validate_metalforge_pipeline
# Expected: 45/45 PASS, exit 0
```
