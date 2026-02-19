# Experiment 016: GPU Pipeline Math Parity — ToadStool Infrastructure

**Date:** 2026-02-19
**Status:** PASS (68/68 checks)
**License:** AGPL-3.0-or-later

## Objective

Prove that the 16S amplicon pipeline produces **identical scientific results**
on CPU and GPU hardware, then benchmark all three tiers:

1. **Galaxy / QIIME2 / Python** (validation control)
2. **BarraCUDA CPU** (pure Rust, flat-array scoring)
3. **BarraCUDA GPU** (pure Rust + ToadStool compact GEMM streaming)

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel i9-12900K (24 threads, 125W TDP) |
| GPU | NVIDIA GeForce RTX 4070 (SHADER_F64, 200W TDP) |
| RAM | 64 GB DDR5 |
| OS | Pop!_OS 22.04, Linux 6.17.4 |
| Rust | release mode, LLVM -O3 |
| ToadStool | wgpu v22, Vulkan backend |

## Architecture: ToadStool Infrastructure Wiring

```
GpuPipelineSession::new(gpu)
  ├── TensorContext          ← buffer pool + bind group cache + batching
  ├── GemmCached             ← pre-compiled GEMM pipeline (local extension)
  ├── FusedMapReduceF64      ← pre-compiled FMR pipeline (ToadStool)
  └── warmup dispatches      ← prime driver caches (27ms one-time)

session.stream_sample(classifier, seqs, counts, params)
  ├── GemmCached::execute()  ← cached pipeline, pooled buffers
  ├── FMR::shannon()         ← reuses compiled pipeline
  ├── FMR::simpson()         ← reuses compiled pipeline
  └── FMR::observed()        ← reuses compiled pipeline
```

### ToadStool Systems Wired

| System | Module | Purpose |
|--------|--------|---------|
| **TensorContext** | `barracuda::device` | Buffer pool, bind group cache, batch grouping |
| **BufferPool** | `TensorContext::buffer_pool()` | Power-of-2 bucketed buffer reuse (73.8% reuse) |
| **GLOBAL_CACHE** | `barracuda::device::pipeline_cache` | Shader/pipeline cache (used by TensorContext) |
| **FusedMapReduceF64** | `barracuda::ops` | Pre-compiled FMR (Shannon, Simpson, sum) |
| **GemmF64** (reference) | `barracuda::ops::linalg` | GEMM math — extended locally as GemmCached |

### Local Extensions (ToadStool Absorption Path)

| Extension | File | Pattern for ToadStool |
|-----------|------|----------------------|
| **GemmCached** | `bio/gemm_cached.rs` | `GemmF64::new()` + cached pipeline |
| **BufferPool integration** | `bio/gemm_cached.rs` | Pool-managed A/B/C buffers via `acquire_pooled()` |
| **execute_to_buffer()** | `bio/gemm_cached.rs` | Return GPU buffer for chaining (no readback) |

### Key Design Decisions

1. **Compact GEMM**: Only k-mers present in query sequences are included
   in the GEMM matrices (~1,000 out of 65,536 for k=8), reducing GPU
   transfer from 587 MB to ~13 MB per dispatch — a 45× reduction.

2. **Stacked bootstrap**: Full classification + 50 bootstrap iterations
   are stacked into a single Q matrix, executing as one GEMM dispatch.

3. **GemmCached pipeline**: Shader compilation + pipeline creation hoisted
   to session init. Per-call overhead: only buffer management + dispatch.

4. **BufferPool**: A/B/C matrices use ToadStool's BufferPool with power-of-2
   bucketing. Across 10+ GEMM calls: 16 allocs, 45 reuses (73.8% reuse).

## Pipeline Stages

| Stage | CPU Module | GPU Module | GPU Primitive |
|-------|-----------|------------|---------------|
| Quality filter | `bio::quality` | `bio::quality_gpu` | FusedMapReduceF64 |
| Dereplication | `bio::derep` | `bio::derep` (CPU) | Hash-based (fast) |
| DADA2 denoise | `bio::dada2` | `bio::dada2` (CPU) | Iterative |
| Chimera detect | `bio::chimera` | `bio::chimera_gpu` | k-mer sketch + prefix-sum |
| Taxonomy | `bio::taxonomy` | `bio::taxonomy_gpu` | **GemmF64 compact GEMM** |
| Diversity | `bio::diversity` | `bio::diversity_gpu` | FusedMapReduceF64 |
| Streaming | — | `bio::streaming_gpu` | Batched session |

## Samples

10 samples across 4 BioProjects, 2.87M reads total:
- **PRJNA1114688**: 4 samples (full CPU+GPU chimera parity)
- **PRJNA629095**: 2 samples
- **PRJNA1178324**: 2 samples
- **PRJNA516219**: 2 samples

Tolerance: `GPU_VS_CPU_F64 = 1e-6`

## Results

### Parity: 68/68 Checks Passed

| Category | Checks | Passed | Status |
|----------|--------|--------|--------|
| Quality filter CPU == GPU | 10 | 10 | PASS |
| Chimera count CPU == GPU | 4 | 4 | PASS |
| Chimera retained CPU == GPU | 4 | 4 | PASS |
| Chimera agreement > 95% | 4 | 4 | PASS |
| GPU chimera completes | 6 | 6 | PASS |
| Shannon CPU ≈ GPU | 10 | 10 | PASS |
| Simpson CPU ≈ GPU | 10 | 10 | PASS |
| Observed CPU ≈ GPU | 10 | 10 | PASS |
| Taxonomy genus CPU == GPU | 10 | 10 | PASS |
| **TOTAL** | **68** | **68** | **PASS** |

### Math Parity Detail

| Metric | Max Δ (CPU vs GPU) | Tolerance |
|--------|-------------------|-----------|
| Shannon entropy | 0.00e0 | 1e-6 |
| Simpson diversity | 0.00e0 | 1e-6 |
| Observed features | 0.00e0 | 1e-6 |
| Chimera decisions | 100.0% agreement | 95% |
| Taxonomy genus | 100% match (50/50) | exact |
| Quality filter count | 0 difference | exact |

### Taxonomy: Compact GEMM vs Flat Array

| Metric | CPU (HashMap) | CPU (flat array) | GPU (compact GEMM) |
|--------|--------------|------------------|---------------------|
| Per-sample avg | 2,450 ms | 115 ms | **13 ms** |
| 10-sample total | 24.5 s | 1.15 s | **0.13 s** |
| vs HashMap | 1× | **21×** | **188×** |

The progression: HashMap (O(n) lookup) → flat array (O(1) lookup, 21×) →
compact GEMM (GPU parallelism + reduced transfer, 188×).

### GPU Streaming Session Breakdown (GemmCached + BufferPool)

| Sample | Tax GEMM (ms) | Div FMR (ms) | Session (ms) |
|--------|--------------|-------------|-------------|
| N.oculata D1-R1 | 9.8 | 0.0 | 9.8 |
| N.oculata D14-R1 | 10.8 | 0.0 | 10.8 |
| B.plicatilis D1-R2 | 9.0 | 0.0 | 9.0 |
| B.plicatilis D14-R1 | 10.8 | 0.0 | 10.8 |
| N.oceanica phyco-1 | 7.5 | 0.0 | 7.5 |
| N.oceanica phyco-2 | 12.3 | 0.0 | 12.3 |
| Cyano-tox-1 | 6.0 | 0.0 | 6.0 |
| Cyano-tox-2 | 21.0 | 0.0 | 21.0 |
| LakeErie-1 | 10.3 | 0.0 | 10.3 |
| LakeErie-2 | 12.6 | 0.0 | 12.6 |
| **Average** | **11.0** | **0.0** | **11.0** |

GemmCached eliminates first-sample warmup penalty (was 36ms → now 9.8ms).
Average per-sample from 14.3ms → **11.0ms** (23% improvement).

### ToadStool BufferPool Stats

| Metric | Value |
|--------|-------|
| Buffer allocations | 16 |
| Buffer reuses | 45 |
| **Reuse rate** | **73.8%** |
| Bucketing | Power-of-2 (256 byte minimum) |

## Three-Tier Benchmark

| Metric | Galaxy/Python | Rust CPU | Rust GPU | CPU/Galaxy | GPU/CPU |
|--------|--------------|----------|----------|------------|---------|
| Total (10 sam) | 95.6 s | 7.7 s | **6.1 s** | **12.4×** | **1.25×** |
| Per-sample | 9.56 s | 0.77 s | **0.61 s** | **12.4×** | **1.25×** |
| Taxonomy/sam | ~3.0 s | 0.115 s | **0.013 s** | **26×** | **8.8×** |
| DADA2/sam | 6.80 s | 0.32 s | 0.32 s | **21×** | 1× |
| Chimera/sam | ~1.0 s | 0.13 s | 0.13 s | **7.7×** | 1× |
| Dependencies | 7 + Galaxy | 1 (flate2) | 1 (flate2) | — | — |
| Docker | 4 GB | No | No | — | — |
| Binary size | N/A | ~8 MB | ~8 MB | — | — |

### Energy & Cost Estimate

| Metric | Galaxy | Rust CPU | Rust GPU |
|--------|--------|----------|----------|
| TDP | 125 W | 125 W | 200 W |
| Pipeline time (10 sam) | 95.6 s | 7.7 s | 6.1 s |
| Energy (kWh) | 0.00332 | 0.00025 | 0.00033 |
| Cost ($0.12/kWh) | $0.000398 | $0.000030 | $0.000040 |
| At 10K samples | **$0.40** | **$0.03** | **$0.04** |

**Rust CPU is cheapest ($0.03/10K). Rust GPU ($0.04/10K) isolates compute to
a single chip for dedicated hardware. Both are 10-13× cheaper than Galaxy.**

## Optimization History

| Change | Taxonomy | Chimera | Total (10s) |
|--------|----------|---------|-------------|
| Baseline (HashMap, O(n³)) | 24.5 s | 1,985 s | ~2,010 s |
| k-mer sketch chimera | 24.5 s | 1.6 s | 32.7 s |
| Flat-array taxonomy | 1.15 s | 1.6 s | 7.1 s (CPU) |
| Compact GEMM taxonomy | 0.13 s | 1.6 s | **6.1 s (GPU)** |
| **Total speedup** | **188×** | **1,256×** | **335×** |

## Scaling Benchmark — GemmCached + BufferPool

| Queries | CPU (ms) | GPU (ms) | Speedup | GPU/query |
|---------|----------|----------|---------|-----------|
| 5       | 76       | 3.9      | **19.6×** | 0.77 ms |
| 25      | 397      | 17.6     | **22.6×** | 0.70 ms |
| 100     | 1,607    | 31.0     | **51.9×** | 0.31 ms |
| 500     | 8,071    | 238      | **33.9×** | 0.48 ms |

GemmCached + BufferPool eliminates per-call pipeline creation AND buffer
allocation. At 5 queries: 3.9ms GPU (was 6.1ms — **36% faster**). At 100
queries: 31ms (was 34ms). Pipeline compiled once at init (27ms), buffers
reused via power-of-2 bucketing (73.8% reuse rate).

Pipeline totals (10 samples): CPU 7.3s, GPU 6.0s = **1.21×** overall.

## Conclusions

1. **Math is identical across hardware.** Zero error for diversity, 100%
   chimera agreement, 100% taxonomy agreement. The science is preserved.

2. **ToadStool infrastructure wired.** TensorContext (buffer pool + cache),
   GemmCached (pre-compiled pipeline), and FMR are integrated into the
   streaming session. BufferPool achieves 73.8% buffer reuse rate.

3. **Dispatch cleared.** GemmCached compiles the GEMM pipeline once at init
   (27ms). Per-call overhead: only buffer management + dispatch. Small
   workload speedup improved from 16× to **19.6×** at 5 queries.

4. **Local extensions prove ToadStool absorption path:**
   - `GemmCached`: GemmF64 should evolve to cache pipeline internally
   - `execute_to_buffer()`: GEMM should return GPU buffer for chaining
   - BufferPool integration: GEMM should use pool for buffer reuse

5. **The three tiers validate the full claim:**
   - Galaxy → Rust CPU: 12× faster, 13× cheaper, zero dependencies
   - Rust CPU → Rust GPU: 1.21× faster, 8.8× taxonomy speedup
   - **Rust abstracts across hardware** — same algorithm, any chip

6. **Isolating work to GPU is validated.** Taxonomy GEMM + diversity FMR
   run entirely on GPU. CPU handles I/O, hashing, and iterative stages.

7. **Next: GPU argmax kernel.** Currently, GEMM scores are read back to CPU
   for argmax + bootstrap confidence. A GPU argmax kernel would reduce
   readback from ~3.5MB to ~1KB per taxonomy dispatch — the key remaining
   optimization before chipset-phase work.

## Files

- `barracuda/src/bio/gemm_cached.rs` — **NEW**: Pre-compiled GEMM pipeline + BufferPool
- `barracuda/src/bio/streaming_gpu.rs` — Streaming session (TensorContext + GemmCached + FMR)
- `barracuda/src/bio/taxonomy.rs` — Dense flat-array NB classifier
- `barracuda/src/bio/taxonomy_gpu.rs` — Compact GEMM GPU taxonomy (standalone)
- `barracuda/src/bio/chimera.rs` — k-mer sketch + prefix-sum chimera
- `barracuda/src/bin/validate_16s_pipeline_gpu.rs` — Validation + benchmarking binary
- `experiments/results/016_gpu_pipeline_parity/gpu_parity_results.json`
