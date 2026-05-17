<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Gen4: Compute-Aware Pipeline Scheduling and Rust-Native Bioinformatics

**Date:** May 17, 2026
**Status:** Active evolution — Exp381 (Barrick 2009) provides first real-data
profiling. Insights feed next-generation pipeline architecture.

---

## Core Insight

The math is identical across CPU and GPU. Where something runs is a
scheduling decision driven by effectiveness and efficiency, not
compatibility. This has been proven across 384 experiments, 5,967+ checks,
44 GPU modules, and 33.4x mean Rust/Python speedup.

The dispatch decision is:

```
substrate = f(data_size, latency_budget, power_cost, cohabitant_load)
```

Not a compatibility question. The same `bio::diversity` call returns
identical results on CPU, GPU, or NPU — the caller doesn't know or care.

---

## Validation Tier Ladder

Each tier proves the math, then the next tier proves the composition:

| Tier | Language | Proves | Example |
|------|----------|--------|---------|
| **0** | Published paper | Peer-reviewed science | Barrick 2009 Fig. 1 |
| **1** | Python/R baseline | Reproducibility | `scripts/*.py` (58 baselines) |
| **2** | Rust CPU | Implementation parity | `validate_*.rs` (349 binaries) |
| **3** | Rust GPU (barraCuda) | Substrate independence | `validate_*_gpu_*` (44 modules) |
| **4** | Primal composition | IPC parity | `validate_primal_parity` (18 roundtrips) |
| **5** | NUCLEUS deployment | End-to-end composition | `wetspring_guidestone` (38 checks) |

C++ tools (breseq, bowtie2, samtools) live at **Tier 1** — they are the
baseline we validate against, not the destination. Python and bash are
"jelly string languages" for prototyping. The production path is pure Rust
through barraCuda shaders.

---

## The breseq Baseline Pattern

breseq (C++) is used the same way Python is used: as a validation
reference. The pipeline for Barrick 2009 is:

```
Tier 1:  breseq 0.40.1 (C++) → mutation calls per clone
Tier 2:  Rust CPU → same mutation calls via sovereign aligner
Tier 3:  barraCuda GPU → same calls via WGSL shader pipeline
Tier 4:  Primal composition → same calls via IPC through atomics
```

We are currently at Tier 1 (proving the science matches the paper).
The Rust-native pipeline will replace breseq's internal steps:

| breseq Step | Current | Rust-Native Target |
|-------------|---------|-------------------|
| Reference alignment | bowtie2 (C++) | `bio::alignment` (Smith-Waterman already proven, Exp028) |
| Candidate junctions | breseq internal | `bio::junction_detection` (graph traversal) |
| Error calibration | breseq statistical model | `bio::error_model` (via barraCuda stats) |
| Mutation identification | breseq variant caller | `bio::variant_calling` (pattern matching + stats) |
| BAM processing | samtools (C) | `io::bam` (sovereign parser, like `io::fastq`, `io::mzml`) |

Each step follows Write → Absorb → Lean: write in Rust, validate against
breseq output, absorb into barraCuda, lean on the primal.

---

## Compute-Aware Scheduling

### The Cohabitant Problem

A NUCLEUS node is not a dedicated server. It's a personal workstation
that runs science alongside games, browsers, IDEs, and other work. The
compute scheduler must be a good neighbor:

| Scenario | Pipeline Response |
|----------|------------------|
| User launches a game | Reduce threads, pause GPU dispatch |
| User is AFK | Scale to full hardware |
| Power budget exceeded | Throttle to efficiency cores |
| Kill signal (SIGTERM) | Checkpoint DAG, stash partial state |
| Machine restart | Resume from cached outputs |

This is **not** HPC batch scheduling. This is **living-environment
scheduling** — the pipeline cohabits with the user's life.

### DAG Checkpoint Pattern (Proven by Exp381)

The Barrick 2009 pipeline (Exp381) proved the checkpoint pattern:

1. **Run 1**: Started with `-j 4`, processed 5/7 clones, killed
2. **Observation**: Only 25% CPU utilization on 16-thread machine
3. **Optimization**: Doubled to `-j 8` via `available_parallelism()`
4. **Run 2**: 5 clones loaded from cache (0s each), 2 remaining recomputed

Each breseq output directory is an **idempotent checkpoint**. If
`output/output.gd` exists, the clone is done. If the directory is
incomplete, it's cleaned and restarted. No partial state corruption.
No "resume from step 3" complexity — just presence/absence of the
final artifact.

The provenance DAG session tracks every step regardless:

```json
{
  "step": "breseq_variant_calling",
  "clone": "REL1164M",
  "mutations": 579,
  "wall_time_seconds": 0,
  "output_blake3": "..."
}
```

`wall_time_seconds: 0` = cached. The DAG records the *fact* of the
computation, not whether it was live or cached.

### Thread Budget Strategy

```rust
fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(4, NonZero::get)
}

fn breseq_threads() -> String {
    available_threads().clamp(2, 8).to_string()
}
```

The clamp at 8 is a memory safety bound — breseq scales memory linearly
with threads (~2-3 GB per thread for E. coli-sized genomes). On a 128 GB
machine this is conservative; on a 16 GB gaming rig it's essential.

**Evolution target**: query available memory and cohabitant load, then
compute the thread budget dynamically:

```
threads = min(
    available_parallelism(),
    (free_memory_gb / per_thread_memory_gb),
    user_preference_max_threads,
)
```

### GPU Dispatch Threshold

barraCuda already implements dispatch thresholds — small vectors stay CPU,
large matrices go GPU. The threshold is:

```
if data_elements > dispatch_threshold() {
    gpu_dispatch(kernel, data)
} else {
    cpu_fallback(data)
}
```

For LTEE post-processing (264 genomes × mutation matrices), the GPU wins
for aggregate statistics (diversity across all clones, distance matrices,
spectral analysis). For per-clone work, CPU is sufficient.

---

## Profiling Observations (Exp381, May 17 2026)

**Hardware:** Ryzen 7 5800X3D (8c/16t), 128 GB DDR4, RTX 4060 8GB, 4TB NVMe

| Metric | `-j 4` Run | `-j 8` Run |
|--------|-----------|-----------|
| CPU utilization | ~25% | ~50% (expected) |
| GPU utilization | 8% (idle) | 8% (idle — breseq is CPU-only) |
| Memory usage | 9.8 GB / 128 GB | ~15 GB / 128 GB (expected) |
| Per-clone time | ~15 min | ~8 min (expected) |
| Load average | 2.5 / 16 | 5 / 16 (expected) |

**Bottleneck**: Reference alignment (step 02) dominates — bowtie2 builds
the FM-index and aligns reads. This is the step that benefits most from
thread scaling.

**GPU opportunity**: Post-breseq aggregate analysis. Once all 264 Tenaillon
genomes are processed, the mutation count matrices, diversity metrics, and
spectral decomposition are batch GPU workloads via barraCuda.

---

## Evolution Roadmap

### Phase 1: Baseline Validation (Current)
- breseq (C++) proves science matches published claims
- Idempotent DAG checkpointing proven
- Thread scaling and profiling wired

### Phase 2: Concurrent Clone Processing
- Run N breseq instances simultaneously (bounded by RAM)
- Semaphore-based task runner in Rust
- Target: 3x throughput for Tenaillon 2016 (66h → 22h)

### Phase 3: Rust-Native Alignment
- Replace bowtie2 with sovereign `bio::alignment` (FM-index + BWT)
- Smith-Waterman already proven (Exp028, 625x faster than Python)
- GPU-accelerable via barraCuda WGSL kernels

### Phase 4: Full Sovereign Pipeline
- Replace all C++ dependencies with Rust modules
- Entire pipeline runs through barraCuda with GPU dispatch
- breseq becomes Tier 1 validation reference only

### Phase 5: Living-Environment Scheduling
- Query cohabitant load (games, browsers, other work)
- Dynamic thread/GPU budget based on system state
- SIGTERM → clean DAG checkpoint → resume on restart
- Power-aware scheduling (efficiency cores for background work)

---

## Upstream Value

This document feeds back to:

| Team | What They Learn |
|------|----------------|
| **biomeOS** | `signal.dispatch` should carry thread/GPU budget hints |
| **toadStool** | Substrate selection needs cohabitant awareness |
| **barraCuda** | Dispatch threshold should be queryable/tunable at runtime |
| **primalSpring** | Composition health should report resource utilization |
| **All springs** | The DAG checkpoint pattern is a composition primitive |

The rewire from hardcoded `-j 4` to dynamic `available_parallelism()` is
a small code change but a large architectural signal: NUCLEUS compositions
must be responsive to their host environment, not assume dedicated hardware.
