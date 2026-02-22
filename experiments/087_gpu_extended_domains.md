# Exp087: GPU Extended Domains — EIC, PCoA, Kriging, Rarefaction

**Status**: PASS (GPU-only, requires `--features gpu`)  
**Binary**: `validate_gpu_extended`  
**Date**: 2026-02-22

## Purpose

Extend the GPU cross-substrate validation to 4 previously uncovered domains,
bringing total GPU-eligible domain coverage to 16.

## Validated Domains

| Domain | GPU Primitive | CPU ↔ GPU Test | Checks |
|--------|-------------|----------------|--------|
| EIC total intensity | `FusedMapReduceF64` | intensity sum parity | 8+ |
| PCoA ordination | `BatchedEighGpu` | eigenvalues + coordinates | 15+ |
| Kriging interpolation | `KrigingF64` | ordinary + simple + variogram | 12+ |
| Rarefaction bootstrap | `FusedMapReduceF64` | Shannon/Simpson CI + batch | 15+ |

## Reproduction

```bash
cargo run --features gpu --release --bin validate_gpu_extended
# Expected: PASS, exit 0
```
