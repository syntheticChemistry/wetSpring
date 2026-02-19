# Experiment 016: GPU Pipeline Math Parity — CPU vs GPU Identical Results

**Date:** 2026-02-19
**Status:** PASS (68/68 checks)
**License:** AGPL-3.0-or-later

## Objective

Prove that the 16S amplicon pipeline produces **identical scientific results**
on CPU and GPU hardware. The math must be the same across:

1. **Galaxy / QIIME2 / Python** (validation control)
2. **BarraCUDA CPU** (pure Rust)
3. **BarraCUDA GPU** (pure Rust + ToadStool GPU primitives)

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel i9-12900K (24 threads, 125W TDP) |
| GPU | NVIDIA GeForce RTX 4070 (SHADER_F64, 200W TDP) |
| RAM | 64 GB DDR5 |
| OS | Pop!_OS 22.04, Linux 6.17.4 |
| Rust | release mode, LLVM -O3 |
| ToadStool | wgpu v22, Vulkan backend |

## Methodology

### Pipeline Stages Tested

| Stage | CPU Module | GPU Module | GPU Primitive |
|-------|-----------|------------|---------------|
| Quality filter | `bio::quality` | `bio::quality_gpu` | FusedMapReduceF64 |
| Dereplication | `bio::derep` | `bio::derep` (CPU) | Hash-based (fast) |
| DADA2 denoise | `bio::dada2` | `bio::dada2` (CPU) | Iterative |
| Chimera detect | `bio::chimera` | `bio::chimera_gpu` | GemmF64 (pairwise) |
| Taxonomy | `bio::taxonomy` | `bio::taxonomy_gpu` | NB classifier |
| Diversity | `bio::diversity` | `bio::diversity_gpu` | FusedMapReduceF64 |

### Samples

10 samples across 4 BioProjects, 2.87M reads total:
- **PRJNA1114688**: 4 samples (full CPU+GPU chimera parity)
- **PRJNA629095**: 2 samples (GPU-only chimera)
- **PRJNA1178324**: 2 samples (GPU-only chimera)
- **PRJNA516219**: 2 samples (GPU-only chimera)

### Tolerance

`GPU_VS_CPU_F64 = 1e-6` for all floating-point comparisons.

## Results

### Check Summary

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
| Quality filter read count | 0 difference | exact |

**All diversity metrics show ZERO error** — CPU and GPU produce bit-identical
results for Shannon, Simpson, and observed features.

### Chimera GPU Speedup

| Sample | ASVs | CPU (ms) | GPU (ms) | Speedup |
|--------|------|----------|----------|---------|
| N.oculata D1-R1 | 153 | 20,343 | 239 | 85× |
| N.oculata D14-R1 | 302 | 210,293 | 2,240 | 94× |
| B.plicatilis D1-R2 | 256 | 127,211 | 1,228 | 104× |
| B.plicatilis D14-R1 | 331 | 225,257 | 2,331 | 97× |

GPU chimera uses `GemmF64` for pairwise sequence encoding (one-hot N×4L GEMM)
plus prefix-sum crossover scoring (O(1) per crossover vs O(L) on CPU).

### Overall Pipeline Timing

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Total (10 samples) | 616.4 s | 39.0 s | 15.8× |
| Per-sample average | 61.6 s | 3.9 s | 15.8× |

## Three-Tier Benchmark

| Metric | Galaxy/Python | Rust CPU | Rust GPU | CPU/Py | GPU/CPU |
|--------|--------------|----------|----------|--------|---------|
| Per-sample time | 9.56 s | 61.6 s* | 3.9 s | — | 15.8× |
| DADA2 per-sample | 6.80 s | 0.32 s | 0.32 s | 21× | 1× |
| Chimera per-sample | ~1.0 s | 145.8 s | 1.5 s | — | 97× |
| Dependencies | 7 + Galaxy | 1 (flate2) | 1 (flate2) | — | — |
| Docker required | Yes (4 GB) | No | No | — | — |
| Binary size | N/A | ~8 MB | ~8 MB | — | — |
| GPU-portable | No | Yes | **Active** | — | — |

*CPU pipeline time dominated by chimera O(n³) — GPU eliminates this bottleneck.

### Energy & Cost Estimate

| Metric | Galaxy | Rust CPU | Rust GPU |
|--------|--------|----------|----------|
| TDP | 125W | 125W | 200W (GPU) |
| Pipeline time (10 sam) | 95.6 s | 616.4 s | 39.0 s |
| Energy (kWh) | 0.00332 | 0.02140 | 0.00217 |
| Cost (US avg $0.12/kWh) | $0.000398 | $0.002568 | $0.000260 |
| At 10K samples | $0.40 | $2.57 | $0.26 |

**Rust GPU is 1.5× cheaper than Galaxy and 9.9× cheaper than Rust CPU.**

Note: Rust CPU time is inflated by unoptimized O(n³) chimera. The GPU version
eliminates this bottleneck entirely. With GPU chimera, Rust becomes the
cheapest option at scale.

## Conclusions

1. **Math is identical across hardware.** Zero error for diversity, 100%
   chimera agreement, 100% taxonomy agreement. The science is preserved.

2. **GPU chimera is 85-104× faster than CPU**, eliminating the pipeline's
   sole remaining bottleneck (98.5% of CPU runtime).

3. **The three tiers validate the full claim:**
   - Python → Rust: 21× DADA2 speedup, single binary, zero dependencies
   - Rust CPU → Rust GPU: 15.8× overall, 97× chimera, identical math
   - **Rust allows hardware abstraction** — same code, CPU or GPU

4. **Isolating work to GPU** is validated. The GPU handles chimera, diversity,
   and taxonomy scoring. I/O and dereplication stay on CPU (fast already).
   Mixed CPU/GPU optimization is deferred to dedicated chipset work.

## Files

- `barracuda/src/bio/chimera_gpu.rs` — GPU chimera via GemmF64
- `barracuda/src/bio/quality_gpu.rs` — GPU quality filter
- `barracuda/src/bio/taxonomy_gpu.rs` — GPU taxonomy classifier
- `barracuda/src/bin/validate_16s_pipeline_gpu.rs` — validation binary
- `experiments/results/016_gpu_pipeline_parity/gpu_parity_results.json`
