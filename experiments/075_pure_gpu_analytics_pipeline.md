# Experiment 075: Pure GPU Multi-Stage Analytics Pipeline

**Date:** February 21, 2026
**Status:** Active
**Binary:** `validate_pure_gpu_pipeline`

---

## Objective

Demonstrate a complete bioinformatics analytics pipeline running entirely
on GPU through five chained stages, with a single CPU upload at the start
and a single CPU readback at the end. Validate that the full GPU pipeline
produces identical results to the CPU-only pipeline.

## Pipeline Stages

```
Stage 1: Alpha diversity (Shannon, Simpson, Observed, Evenness, Chao1)
    ↓ abundance vectors stay on GPU
Stage 2: Beta diversity (Bray-Curtis distance matrix)
    ↓ condensed distance matrix stays on GPU
Stage 3: Ordination (PCoA from distance matrix)
    ↓ ordination coordinates
Stage 4: Statistical summary (variance, correlation on PC axes)
    ↓ summary statistics
Stage 5: Spectral cosine similarity (FMR pairwise)
    ↓ similarity scores → CPU readback
```

## Protocol

### Multi-Sample Dataset
- 8 synthetic microbial communities, each with 512 features
- Features generated from deterministic gradient to ensure reproducibility

### Path A: Full CPU Pipeline
Run all five stages on CPU. This is the reference truth.

### Path B: Full GPU Pipeline (Pre-Warmed Session)
Run all five stages on GPU using pre-compiled pipelines.
Use `GpuPipelineSession` for diversity/taxonomy and standalone
GPU modules for Bray-Curtis, PCoA, stats, and spectral.

### Validation Checks
1. Alpha diversity: all 5 metrics match CPU for each sample
2. Beta diversity: full condensed matrix matches CPU
3. PCoA: eigenvalues and coordinates match within tolerance
4. Stats: variance and correlation on PC1/PC2 match CPU
5. Spectral: pairwise cosine matches CPU
6. Total pipeline GPU time < sum of individual GPU stages

## Expected Result
- All parity checks PASS across all 5 stages
- GPU pipeline demonstrates end-to-end capability
- Foundation for ToadStool unidirectional streaming absorption
