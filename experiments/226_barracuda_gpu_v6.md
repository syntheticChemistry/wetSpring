# Exp226: BarraCuda GPU v6 — V71 Precision-Flexible Portability

**Track:** cross (GPU)
**Phase:** 71
**Status:** PASS — 28/28 checks
**Binary:** `validate_barracuda_gpu_v6`
**Features:** `gpu`

## Purpose

Validates CPU↔GPU parity for precision-flexible workloads. Proves that
diversity, Bray-Curtis, HMM, dN/dS, SNP, pangenome, spectral cosine, GEMM
with `with_precision(F64)`, and DF64 host roundtrip all produce identical
results on CPU and GPU.

## Model / Equations

| Domain | What it proves |
|--------|----------------|
| diversity(4) | Shannon, Simpson, observed, Chao1 |
| Bray-Curtis | Beta diversity matrix |
| DiversityFusion | Multi-metric fusion |
| HMM | Hidden Markov model |
| dN/dS | Ka/Ks ratio |
| SNP | Single-nucleotide polymorphism |
| Pangenome | Core/accessory |
| variance | Statistical variance |
| spectral cosine | Spectral similarity |
| GEMM with_precision(F64) | Double-precision GEMM |
| DF64 host roundtrip | Half↔full precision |
| BandwidthTier PCIe4x16 | Transfer topology |

## Validation

- 28 checks across all domains
- CPU reference vs GPU output within `GPU_VS_CPU_F64` tolerance

## Status

PASS — 28/28 checks
