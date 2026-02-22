# Exp090: Pure GPU Streaming Pipeline — Zero CPU Round-Trips

## Purpose

Proves a full bioinformatics pipeline (diversity + taxonomy + beta) runs on GPU
with data flowing unidirectionally through stages. Only CPU involvement is at
input upload and final readback. Validates three execution modes produce
identical mathematical results.

## Architecture

```text
Round-trip:  CPU → GPU → CPU → GPU → CPU → GPU → CPU  (6 transfers)
Streaming:   CPU → GPU → GPU → GPU → CPU             (2 transfers)
Pure GPU:    CPU → GPU buffer → GPU → GPU → CPU       (2 transfers, 0 intermediate readback)
```

## Validated Modes

| Mode | Description | Checks |
|------|-------------|--------|
| Round-trip | Each GPU stage returns to CPU before next | 4 |
| Streaming | `GpuPipelineSession` chains stages on GPU | 39 |
| RT ↔ Stream parity | Prove identical results both paths | 32 |
| Batch scaling | Streaming amortization grows with batch | 5 |

## Key Results

- **80/80 PASS**
- Streaming 441-837× faster than round-trip at batch scale
- CPU ↔ GPU ↔ streaming: bitwise-identical diversity metrics
- All taxonomy classifications match across modes
- Pre-warmed `GpuPipelineSession` eliminates shader recompilation

## Reproduction

```bash
cargo run --features gpu --release --bin validate_pure_gpu_streaming
```

## Provenance

| Field | Value |
|-------|-------|
| Binary | `validate_pure_gpu_streaming` |
| Date | 2026-02-22 |
| Data | 8 synthetic communities × 256 features |
| Hardware | i9-12900K / RTX 4070 / 64 GB DDR5 |
