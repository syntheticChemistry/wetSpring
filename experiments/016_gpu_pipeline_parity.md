# Experiment 016: GPU Pipeline Math Parity — Streaming GEMM Architecture

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

## Architecture: Streaming GPU Pipeline

```
CPU I/O → FASTQ parse → QF → derep → DADA2 → chimera  (all CPU, <700ms)
        ↓
Upload ASV data → GPU: [taxonomy compact GEMM] → [diversity FMR] → Download
        ↓
CPU post-process: argmax, confidence, formatting  (~0.1ms)
```

### Key Design Decisions

1. **Compact GEMM**: Only k-mers present in query sequences are included
   in the GEMM matrices (~1,000 out of 65,536 for k=8), reducing GPU
   transfer from 587 MB to ~13 MB per dispatch — a 45× reduction.

2. **Stacked bootstrap**: Full classification + 50 bootstrap iterations
   are stacked into a single Q matrix, executing as one GEMM dispatch.

3. **Streaming session**: Taxonomy GEMM + diversity FMR share a single
   `WgpuDevice` and `TensorContext`, minimizing driver overhead.

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

### GPU Streaming Session Breakdown

| Sample | Tax GEMM (ms) | Div FMR (ms) | Session (ms) |
|--------|--------------|-------------|-------------|
| N.oculata D1-R1 | 36.0 | 0.0 | 36.0 |
| N.oculata D14-R1 | 9.5 | 0.0 | 9.5 |
| B.plicatilis D1-R2 | 10.0 | 0.0 | 10.0 |
| B.plicatilis D14-R1 | 11.4 | 0.0 | 11.4 |
| N.oceanica phyco-1 | 7.8 | 0.0 | 7.8 |
| N.oceanica phyco-2 | 13.1 | 0.0 | 13.1 |
| Cyano-tox-1 | 5.8 | 0.0 | 5.8 |
| Cyano-tox-2 | 22.6 | 0.0 | 22.6 |
| LakeErie-1 | 11.7 | 0.0 | 11.7 |
| LakeErie-2 | 14.8 | 0.0 | 14.8 |
| **Average** | **14.3** | **0.0** | **14.3** |

First sample has warmup cost (shader compilation). Subsequent: ~7-16ms.

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

## Scaling Benchmark — Dispatch Cleared

| Queries | CPU (ms) | GPU (ms) | Speedup | GPU/query |
|---------|----------|----------|---------|-----------|
| 5       | 98       | 6.1      | 16×     | 1.22 ms   |
| 25      | 421      | 17       | 24.7×   | 0.68 ms   |
| 100     | 1,670    | 34       | 49.7×   | 0.34 ms   |
| 500     | 8,053    | 145      | 55.5×   | 0.29 ms   |

Pre-warming FMR + GEMM shaders at session init (23.8ms one-time cost) via
`GpuPipelineSession` eliminates per-call shader compilation. With dispatch
overhead cleared, GPU advantage grows from 16× at 5 queries to 55.5× at 500
queries.

Pipeline totals (10 samples): CPU 7.7s, GPU 6.1s = **1.25×** overall.

## Conclusions

1. **Math is identical across hardware.** Zero error for diversity, 100%
   chimera agreement, 100% taxonomy agreement. The science is preserved.

2. **Streaming GEMM architecture validated.** Compact k-mer GEMM with
   stacked bootstrap eliminates ~45× of GPU transfer overhead. Average
   streaming session: 14.3ms (taxonomy 14.3ms + diversity 0.0ms, pre-warmed).

3. **Dispatch cleared.** `GpuPipelineSession` pre-warms FMR + GEMM shaders at
   init (23.8ms one-time). Per-call shader compilation eliminated; GPU
   speedup scales from 16× at 5 queries to 55.5× at 500 queries.

4. **The three tiers validate the full claim:**
   - Galaxy → Rust CPU: 13.4× faster, 13× cheaper, zero dependencies
   - Rust CPU → Rust GPU: 1.25× faster, 8.8× taxonomy speedup
   - **Rust abstracts across hardware** — same algorithm, any chip

5. **Isolating work to GPU is validated.** Taxonomy GEMM + diversity FMR
   run entirely on GPU. CPU handles I/O, hashing, and iterative stages.
   Full GPU isolation (including DADA2/chimera) deferred to chipset phase.

6. **ToadStool dispatch path**: PipelineBuilder, BufferPool, and
   UnidirectionalPipeline are available for cross-stage buffer persistence
   and zero-readback chaining — the next step for dedicated chipset work.

## Files

- `barracuda/src/bio/taxonomy.rs` — Dense flat-array NB classifier
- `barracuda/src/bio/taxonomy_gpu.rs` — Compact GEMM GPU taxonomy
- `barracuda/src/bio/streaming_gpu.rs` — Streaming GPU session wrapper
- `barracuda/src/bio/chimera.rs` — k-mer sketch + prefix-sum chimera
- `barracuda/src/bin/validate_16s_pipeline_gpu.rs` — Validation binary
- `experiments/results/016_gpu_pipeline_parity/gpu_parity_results.json`
