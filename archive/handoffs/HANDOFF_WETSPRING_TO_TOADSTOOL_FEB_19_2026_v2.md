# Handoff v2: wetSpring → ToadStool / BarraCUDA Team

> **SUPERSEDED** by [`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md`](../../HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md)
> (v4 — ToadStool evolution review, updated remaining requests and shader designs).

**Date:** February 19, 2026 (evening)
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** All prior handoffs (Feb 16, 17, 19 morning)

---

## Executive Summary

wetSpring has completed a full Python/open-data control validation pass across
all tracks. The project now has head-to-head Python baselines on identical
public data, an ML baseline on real government environmental data, and an ODE
model baseline from published microbiology papers. The earlier GPU pipeline work
(88/88 parity, 2.45× speedup, 926× spectral cosine) remains intact and all
572 validation checks pass.

**New in this handoff:**

1. **Python control baselines** on identical public FASTQ data (PRJNA488170, PRJNA382322, PRJNA1114688)
2. **Michigan DEQ PFAS ML** — 3,719 real water samples, RF F1=0.978, GBM F1=0.992
3. **QS/c-di-GMP ODE model** — 5-variable system, 4 scenarios, 35/35 checks
4. **Jones Lab PFAS library** — 175 compounds, 259 ion types from Zenodo (open access)
5. **PhyloNet-HMM test data** scouted — 1,284 Newick gene trees available, Exp019 designed
6. **572/572 total checks PASS** (446 CPU + 126 GPU), up from 514

**What this means for ToadStool/BarraCUDA:**

The validation story is now complete. Every claim wetSpring makes about its Rust
pipeline has a Python baseline running on the same data, a published paper
providing ground truth, or a physical constant as reference. This clears the
way for the next phase: **pure Rust portable mathematics via WGSL shaders** —
converting the validated CPU algorithms into GPU compute that runs on any
WebGPU-capable device.

---

## 1. Validation State (All Green)

### 1.1 CPU Validators (446 checks)

| Binary | Checks | Experiment | Data Source |
|--------|:------:|------------|-------------|
| `validate_fastq` | 28 | Exp001 | Zenodo MiSeq SOP |
| `validate_diversity` | 27 | Exp002 | Analytical |
| `validate_16s_pipeline` | 37 | Exp003 | Synthetic + Galaxy |
| `validate_peaks` | 17 | Exp010 | scipy baselines |
| `validate_pfas` | 10 | Exp006 | FindPFAS/pyOpenMS |
| `validate_features` | 8 | Exp009 | asari MT02 |
| `validate_voc_peaks` | 22 | Exp013 | Reese 2019 Table 1 |
| `validate_algae_16s` | 34 | Exp012 | PRJNA488170 + **Python control** |
| `validate_extended_algae` | 35 | Exp017 | PRJNA382322 + **Python control** |
| `validate_public_benchmarks` | 202 | Exp014 | 4 BioProjects, 22 samples |
| `validate_pfas_library` | 26 | Exp018 | **Jones Lab 175 compounds** |

### 1.2 GPU Validators (126 checks)

| Binary | Checks | What |
|--------|:------:|------|
| `validate_diversity_gpu` | 38 | Shannon/Simpson/BC/PCoA/spectral/stats |
| `validate_16s_pipeline_gpu` | 88 | Full pipeline parity (10 samples, 4 BioProjects) |

### 1.3 Python Control Baselines (new)

| Phase | Script | Data | Result |
|-------|--------|------|--------|
| A | `validate_public_16s_python.py` | PRJNA488170, 382322, 1114688 | 4 datasets, Shannon 6.8–7.2 |
| B | `exp008_pfas_ml_baseline.py` | MI DEQ 3,719 records | RF F1=0.978, GBM F1=0.992 |
| C | `waters2008_qs_ode.py` | Literature ODE system | 35/35, 4 scenarios |
| D | Jones Lab PFAS library | Zenodo 14341321 | 259 ions, integrated into Rust |
| E | PhyloNet-HMM scout | PhyNetPy GitHub | 1,284 trees, Exp019 designed |

---

## 2. BarraCUDA Evolution — What We Learned

### 2.1 What BarraCUDA Does Well

1. **FusedMapReduceF64** is the single most useful primitive. It handles
   Shannon, Simpson, observed features, evenness, and alpha diversity in one
   dispatch. Every 16S sample touches it.

2. **GemmF64** for spectral cosine is the single biggest speedup (926×).
   The pairwise MS2 matching problem is embarrassingly parallel and the
   GPU matrix multiply dominates. `GemmCached` (our local extension) makes
   this practical by pre-compiling the pipeline and reusing buffers.

3. **BatchedEighGpu** for PCoA eigendecomposition works exactly as expected.
   No numerical issues. The 5×5 matrices from our beta diversity are small
   but the batching amortizes overhead well.

4. **BrayCurtisF64** produces condensed distance matrices that match CPU
   to machine precision. Zero divergence across 10 samples × 10 samples.

5. **Zero-copy GPU → CPU parity**: 88/88 pipeline checks pass with zero
   divergence. The `bytemuck` casting + uniform buffer approach is correct
   and performant.

### 2.2 What Required Local Extensions

These three local extensions (already documented in the absorption spec)
remain candidates for upstream ToadStool promotion:

| Extension | Purpose | Lines | Why Local |
|-----------|---------|------:|-----------|
| `QualityFilterCached` | Per-read parallel quality trimming | ~220 | `ParallelFilter<T>` generic needed |
| `Dada2Gpu` | DADA2 E-step batch log-p-error | ~180 | `BatchPairReduce<f64>` pattern |
| `GemmCached` | Pre-compiled GEMM + BufferPool | ~260 | Pipeline reuse (93% hit rate) |

### 2.3 What Drove the Architecture

The key insight from wetSpring is that **bioinformatics pipelines are
streaming problems, not batch problems.** Each FASTQ sample flows through
QF → derep → DADA2 → chimera → taxonomy → diversity. The GPU wins only when:

1. The per-sample cost is high enough to amortize dispatch overhead
2. Pipeline compilation happens once and reuses across samples
3. Buffer pools prevent allocation churn between stages

`GpuPipelineSession` captures this pattern. It holds precompiled pipelines
for all stages and provides `stream_sample()` which runs taxonomy + diversity
in a single session with shared TensorContext.

### 2.4 Performance Model

| Stage | CPU | GPU | Speedup | Why |
|-------|-----|-----|---------|-----|
| Quality filter | 0.9ms | 1.1ms | 0.85× | Too few reads per sample |
| DADA2 E-step | 326ms | 13ms | 24× | O(N²) pairwise — ideal for GPU |
| Chimera | 0.02ms | — | CPU-only | Sequential prefix-sum |
| Taxonomy GEMM | 2.1ms | 0.2ms | 10.5× | Matrix multiply |
| Diversity FMR | 0.05ms | 0.03ms | 1.7× | Small vector, but free on GPU |
| **Pipeline** | **7.3s** | **3.0s** | **2.45×** | DADA2 dominates |

The takeaway for BarraCUDA: **the GPU wins on O(N²) and O(N³) stages.** For
O(N) work like quality filtering, the dispatch overhead dominates until sample
sizes reach ~100K reads.

---

## 3. What We Learned Relevant to ToadStool Evolution

### 3.1 Architectural Lessons

1. **Pipeline compilation cost is the real enemy, not arithmetic.**
   First dispatch of any WGSL shader takes 50-200ms (pipeline creation +
   cache population). Subsequent dispatches take <1ms. `GemmCached` exists
   because taxonomy dispatch was 60× slower without pre-compilation.

2. **Buffer reuse is mandatory for streaming workloads.**
   Our `BufferPool` achieves 93% reuse. Without it, wgpu's buffer allocation
   adds ~2ms per stage per sample. At 10,000 samples this is 20 seconds of
   pure overhead.

3. **The `SHADER_F64` feature gate works correctly.** On the RTX 4070,
   wgpu reports f64 capability and all WGSL `f64` operations produce
   IEEE 754 results. No software emulation needed. The Titan V also works.

4. **The CPU → GPU dispatch threshold matters.** We use `GPU_DISPATCH_THRESHOLD = 10_000`.
   Below this, CPU is faster. Above it, GPU wins. This should be configurable per primitive.

5. **Streaming iterators + GPU = natural fit.** Our `FastqIter` yields records
   one at a time. Batching 1000 records, dispatching to GPU, collecting results,
   then yielding the next batch works cleanly. ToadStool should consider a
   `BatchedStreamProcessor<T>` pattern.

### 3.2 Bugs Found (Already Reported)

- **`log_f64` precision**: The wgpu `log()` builtin for f64 has precision
  issues on some drivers. We work around it by precomputing log tables on CPU.
- **`i32 %` negative values**: WGSL `%` on negative `i32` has undefined
  behavior across backends. hotSpring discovered this independently in the
  cell-list shader. Both projects now use branch-based wrapping.

### 3.3 What ToadStool Needs for the Next Phase

For wetSpring to move fully to WGSL shaders (pure Rust portable math), we need:

| Primitive | Use Case | Priority |
|-----------|----------|----------|
| `ParallelFilter<T>` | Quality filtering, chimera removal | High |
| `BatchPairReduce<f64>` | DADA2 E-step, error models | High |
| `RK4F64` | ODE integration (QS/c-di-GMP model) | Medium |
| `HillFunction<f64>` | Kinetic models (Hill activation) | Medium |
| `RandomForestGpu` | PFAS ML classification (Exp008) | Medium |
| `LogSumExpF64` | HMM forward/backward (Exp019) | Low |
| `SmithWaterman<f64>` | Sequence alignment | Low |

---

## 4. New Science Tracks Ready for GPU

### 4.1 QS/c-di-GMP ODE Model (Waters 2008)

The Python baseline is complete (35/35 checks). The ODE system is:

```
dN/dt = μ·N·(1 - N/K) - d·N                    (logistic growth)
dA/dt = k_ai·N - d_ai·A                         (autoinducer)
dH/dt = k_h·Hill(A, K_h, n_h) - d_h·H           (HapR regulation)
dC/dt = k_dgc·(1 - k_rep·H) - (k_pde + k_act·H)·C  (c-di-GMP)
dB/dt = k_bio·Hill(C, K_bio, n_bio)·(1-B) - d_bio·B  (biofilm)
```

This is a 5-variable ODE that runs in <5ms on CPU. The GPU value comes from
**parameter sweeps** — running 10,000 parameter combinations simultaneously
to map the biofilm formation landscape. This is pure `BatchedRK4F64`.

### 4.2 PFAS ML on Real Environmental Data (Exp008)

Michigan DEQ data (3,719 samples, 22 PFAS analytes, 28 features) yields
RF F1=0.978. The Rust port is straightforward:

1. Feature engineering (CPU): PFAS concentrations, ratios, geospatial
2. Decision tree ensemble (GPU): parallelized forest evaluation
3. Inference (GPU): batch classification of new samples

This is the use case for `RandomForestGpu` in BarraCUDA.

### 4.3 Jones Lab PFAS Library Screening (Exp018)

259 ion types from 175 PFAS compounds. Our tolerance search + KMD + spectral
match pipeline handles this on CPU in <1ms. The GPU value comes from:

1. **Batch screening**: matching 10,000+ environmental samples against 259
   reference ions simultaneously (tolerance_search becomes a batch GEMM)
2. **KMD group clustering**: all-pairs distance on 259 ions (BrayCurtisF64)

### 4.4 Phylogenetic HMM (Exp019, design phase)

PhyNetPy provides 1,284 Newick gene trees. The HMM-based introgression
detection (Liu 2014) requires:

1. Tree likelihood computation (parallel Felsenstein pruning)
2. HMM forward/backward (LogSumExpF64 in log-space)
3. Viterbi decoding (parallel max-scan)

This is the most GPU-intensive new workload and would exercise ToadStool's
`LogSumExpF64` and tree-parallel primitives.

---

## 5. Recommended Absorption Priority

| Priority | Item | Benefit |
|----------|------|---------|
| 1 | `GemmCached` → upstream `CachedPipeline<T>` | 60× taxonomy speedup for all consumers |
| 2 | `QualityFilterCached` → `ParallelFilter<T>` | Streaming filter pattern (QF, chimera, etc.) |
| 3 | `Dada2Gpu` → `BatchPairReduce<f64>` | O(N²) pairwise pattern (DADA2, distance matrices) |
| 4 | `BufferPool` → upstream buffer management | 93% allocation reuse, 20s saved over 10K samples |
| 5 | `BatchedRK4F64` (new) | ODE integration for science workloads |

---

## 6. File Inventory — What's New Since Last Handoff

### Python Baselines (new)

| File | Purpose |
|------|---------|
| `scripts/validate_public_16s_python.py` | Python 16S on PRJNA488170/382322/1114688 |
| `scripts/exp008_pfas_ml_baseline.py` | PFAS ML (RF/GBM) on Michigan DEQ data |
| `scripts/waters2008_qs_ode.py` | QS/c-di-GMP ODE model (4 scenarios) |

### Data (new)

| Directory | Contents |
|-----------|----------|
| `data/michigan_deq_pfas/` | 3,719 PFAS surface water records (ArcGIS REST API) |
| `data/jones_pfas_library/` | 175 PFAS, 259 ion types (Zenodo 14341321) |
| `data/phylonet_hmm/` | PhyNetPy demo .nex + DEFJ gene trees .newick |

### Results (new)

| Directory | Contents |
|-----------|----------|
| `experiments/results/python_16s_controls/` | Python diversity baseline JSON |
| `experiments/results/008_pfas_ml/` | Exp008 ML metrics JSON |
| `experiments/results/qs_ode_baseline/` | ODE steady states + parameters JSON |

### Rust Validators (updated)

| Binary | Change |
|--------|--------|
| `validate_algae_16s` | +5 checks (Python control cross-validation) |
| `validate_extended_algae` | +6 checks (Python control cross-validation) |
| `validate_pfas_library` | +5 checks (Jones library expansion) |

### Experiment Docs (new)

| File | Status |
|------|--------|
| `experiments/019_phylogenetic_validation.md` | DESIGN COMPLETE |

---

## 7. Path Forward: Pure Rust Portable Mathematics via Shaders

With all Python/open-data controls in place, the next step is promoting
validated Rust CPU algorithms to WGSL shaders. The evolution:

```
Python baseline (done) → Rust CPU (done, 446 checks) → Rust GPU (126 checks)
                                                         ↓
                                                    WGSL shaders (portable)
```

The portable shader target means: any WebGPU-capable device (consumer GPU,
integrated graphics, browser via WebGPU API, mobile via Dawn/wgpu) runs the
same validated mathematics. No CUDA dependency. No vendor lock-in.

**Priority promotions for shader portability:**

1. `bio::diversity` → WGSL (already done via FusedMapReduceF64)
2. `bio::dada2` E-step → WGSL (already done via custom shader)
3. `bio::taxonomy` GEMM → WGSL (already done via GemmCached)
4. `bio::quality` filter → WGSL (already done via QualityFilterCached)
5. `bio::spectral_match` → WGSL (already done via GemmF64)
6. **`bio::tolerance_search`** → WGSL batch tolerance search (new)
7. **`bio::kmd`** → WGSL KMD + homologue grouping (new)
8. **`bio::unifrac`** → WGSL tree traversal (new)
9. **ODE integration** → WGSL RK4 for QS/c-di-GMP (new)

Items 1–5 already have GPU paths. Items 6–9 are the new shader targets
that would make wetSpring's mathematics fully portable via ToadStool.

---

*This handoff documents the completion of wetSpring's validation phase
and the readiness for portable shader mathematics. The Python/open-data
controls provide the ground truth; BarraCUDA/ToadStool provides the engine.*
