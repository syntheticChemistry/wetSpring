# Exp105: Pure GPU Streaming v2 — Multi-Domain Analytics Pipeline

| Field | Value |
|-------|-------|
| Status | **PASS** — 27/27 |
| Command | `cargo run --features gpu --release --bin validate_pure_gpu_streaming_v2` |
| Phase | 30 |
| Dependencies | Exp090, Exp091 (streaming v1), Exp104 (metalForge v6) |
| Binary | `barracuda/src/bin/validate_pure_gpu_streaming_v2.rs` |

## Purpose

Expand the `GpuPipelineSession` beyond taxonomy + diversity to cover additional
streaming domains. Proves that Bray-Curtis beta diversity, spectral cosine
similarity, and the full 16S analytics pipeline (alpha + beta diversity chained)
can run through pre-warmed GPU pipelines with zero shader recompilation.

## Session Expansion

| New Method | GPU Primitive | Pattern |
|------------|---------------|---------|
| `bray_curtis_matrix` | `BrayCurtisF64` (ToadStool absorbed) | Pre-compiled at session init |
| `spectral_cosine_matrix` | `GemmCached` + FMR norms | Reuses existing GEMM + FMR |
| `stream_full_analytics` | Taxonomy + diversity + BC chained | 3-stage streaming pipeline |

## Validation Sections

| Section | Domain | Checks | Result |
|---------|--------|--------|--------|
| S1 | Alpha diversity (Shannon, Simpson, observed) | 3 | Exact CPU ↔ GPU parity |
| S2 | Bray-Curtis condensed matrix (3 samples) | 5 | GPU ↔ CPU within 1e-6 |
| S3 | Spectral cosine matrix (3 spectra) | 5 | Identical=1.0, orthogonal=0.0 |
| S4 | Full pipeline (3-sample alpha + BC) | 14 | All alpha + BC exact parity |
| **Total** | | **27** | **ALL PASS** |

## Impact

- `GpuPipelineSession` now covers: QF, DADA2, taxonomy, Shannon, Simpson, observed,
  Bray-Curtis, spectral cosine, full analytics pipeline
- Pre-warmed streaming: all pipelines compiled once at session creation
- Streaming v2 proves the 16S pipeline analytics can run with a single GPU session

## Reproduction

```bash
cd barracuda && cargo run --features gpu --release --bin validate_pure_gpu_streaming_v2
```
