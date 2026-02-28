# Exp227: Pure GPU Streaming v4 — Unidirectional Full Science Pipeline

**Track:** cross (GPU)
**Phase:** 71
**Status:** PASS — 24/24 checks
**Binary:** `validate_pure_gpu_streaming_v4`
**Features:** `gpu`

## Purpose

Validates a 7-stage unidirectional GPU pipeline with zero CPU compute
round-trips. Data flows: quality → diversity → fusion → GEMM → PCoA →
spectral → DF64. ToadStool streaming architecture.

## Model / Equations

7-stage pipeline:

1. **Quality** — Phred filtering, dereplication
2. **Diversity** — Shannon, Simpson, observed
3. **Fusion** — DiversityFusion multi-metric
4. **GEMM** — Matrix multiply (precision-flexible)
5. **PCoA** — Principal coordinates
6. **Spectral** — Spectral cosine / Anderson
7. **DF64** — Half-precision host roundtrip

All stages execute on GPU; no CPU compute in the data path.

## Validation

- 24 checks across pipeline stages
- End-to-end parity with CPU reference
- Zero CPU round-trips verified

## Status

PASS — 24/24 checks
