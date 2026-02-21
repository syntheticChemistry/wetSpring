# Experiment 072: ToadStool Unidirectional Streaming — Zero CPU Round-Trips

**Date:** February 21, 2026
**Status:** Active
**Binary:** `validate_gpu_streaming_pipeline`

---

## Objective

Prove that chaining multiple GPU stages via ToadStool's unidirectional streaming
eliminates CPU round-trips and delivers measurable throughput improvement over
individual GPU dispatches.

## Background

Current GPU validators dispatch each operation independently: upload → compute →
readback → upload → compute → readback. Each round-trip costs 100–500µs in PCIe
overhead. ToadStool's `GpuPipelineSession` pre-compiles all pipelines at init
and reuses buffer pools/bind-group caches across stages.

From PCIE_TOPOLOGY.md:
```
Traditional:  CPU → GPU → CPU → GPU → CPU → GPU → CPU  (6 PCIe transfers)
Streaming:    CPU → GPU ───→ GPU ───→ GPU → CPU          (2 PCIe transfers)
```

## Protocol

### Path A: CPU Baseline
Run Shannon, Simpson, Bray-Curtis, spectral cosine sequentially on CPU.
Measure total wall time.

### Path B: GPU Dispatched (Individual)
For each metric, create a new FMR/BrayCurtis instance, upload, compute, readback.
This is the "worst case" GPU path — maximum dispatch overhead.

### Path C: GPU Streaming (ToadStool Pipeline)
Use `GpuPipelineSession` with pre-warmed pipelines. All stages share the
`TensorContext` buffer pool and bind-group cache. Pipelines compiled once at
session init.

### Validation Checks
1. **Parity**: Path A == Path B == Path C (within GPU_VS_CPU_F64 tolerance)
2. **Streaming advantage**: Path C total_ms < Path B total_ms (overhead reduced)
3. **TensorContext reuse**: buffer pool hits > 0 (confirms caching)
4. **Pipeline warmup**: session.warmup_ms measured and reported

## Domains Tested
| # | Domain | Primitive |
|---|--------|-----------|
| 1 | Shannon entropy | FMR (absorbed) |
| 2 | Simpson diversity | FMR (absorbed) |
| 3 | Bray-Curtis distance | BrayCurtisF64 (absorbed) |
| 4 | Spectral cosine | FMR (absorbed) |

## Expected Result
- All parity checks PASS
- Streaming path measurably faster than individual dispatch
- Buffer pool reuse demonstrated via TensorContext stats
