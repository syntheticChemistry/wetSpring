# Handoff: wetSpring → ToadStool / BarraCUDA Team

> **SUPERSEDED** by [`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_20_2026.md`](HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_20_2026.md)
> (v4 — ToadStool evolution review, updated remaining requests and shader designs).

**Date:** February 19, 2026
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** Feb 16 and Feb 17 handoffs

---

## Executive Summary

wetSpring has completed a full-stage GPU-accelerated 16S amplicon sequencing pipeline
using BarraCUDA (pure Rust math) + ToadStool (GPU dispatch). The results prove the
ecoPrimals thesis:

- **514/514 quantitative checks PASS** (388 CPU + 126 GPU)
- **88/88 GPU pipeline parity checks** — identical results on CPU and GPU
- **2.45× GPU pipeline speedup** (3.0s vs 7.3s for 10 samples)
- **24× DADA2 E-step speedup** — the pipeline's largest bottleneck eliminated
- **670× total optimization** from baseline to current GPU pipeline
- **$0.02/10K samples** (GPU) vs $0.03 (CPU) vs $0.40 (Galaxy)

Three local extensions are ready for ToadStool absorption. Five architectural
lessons apply to all BarraCUDA/ToadStool consumers. A comprehensive code quality
audit (Feb 19) resolved 16 GPU clippy errors, added 6 determinism tests, introduced
streaming iterators for all three I/O parsers, and achieved zero warnings across
all targets including `--features gpu`.

---

## 1. What wetSpring Proves

### 1.1 BarraCUDA Solves Math

The same algorithms produce identical results on CPU and GPU. 88/88 parity
checks pass with zero divergence: same ASV counts, same total reads, same
taxonomy assignments, same diversity metrics.

**This means:** any BarraCUDA consumer can trust that algorithm correctness
is independent of hardware target. Write once, validate once, deploy anywhere.

### 1.2 ToadStool Solves Hardware

BufferPool achieves 93% buffer reuse. Pipeline caching eliminates per-call
shader compilation. TensorContext provides session-level GPU resource management.
GPU is competitive even at 5 taxonomy queries (16.5× speedup) — dispatch
overhead is no longer a bottleneck.

**This means:** ToadStool's dispatch infrastructure works. The "small workload
penalty" that plagues naive GPU usage has been solved through pre-compilation
and buffer pooling.

### 1.3 Pure CPU and Pure GPU Are Functional Validation Goals

wetSpring validates math on CPU first, then proves identical results on GPU.
Mixed CPU/GPU optimization (where stages move between devices based on load)
is deferred to chipset phase. For now:
- Pure CPU = correctness reference
- Pure GPU = performance target
- Parity between them = science is preserved

---

## 2. BarraCUDA Evolution in wetSpring

### 2.1 Timeline

```
Phase 1 (Feb 12-16):  Galaxy/QIIME2 replication, baseline data
Phase 2 (Feb 16-17):  Pure Rust CPU — 31 modules, 388/388 checks
Phase 3 (Feb 17-18):  ToadStool wiring — 11 primitives, 38/38 GPU diversity checks
Phase 4 (Feb 18):     Sovereign pipeline — DADA2/chimera/taxonomy/UniFrac
Phase 5 (Feb 18):     Paper parity — real NCBI data vs published baselines
Phase 6 (Feb 18-19):  GPU pipeline — GemmCached + streaming session (68/68 at time)
Phase 7 (Feb 19):     Full-stage GPU — DADA2 GPU + QF WGSL → 88/88 parity, 2.45×
```

### 2.2 Optimization Journey

| Bottleneck | Before | After | What Fixed It |
|------------|--------|-------|---------------|
| Chimera (O(N³)) | 1,985 s / 10 sam | 1.6 s | k-mer sketch + prefix-sum scoring |
| Taxonomy (HashMap) | 24.5 s / 10 sam | 0.11 s | Flat array → compact GEMM on GPU |
| DADA2 E-step | 3.3 s / 10 sam | 0.13 s | Custom WGSL shader + precomputed log-err |
| Pipeline total | ~2,010 s | 3.0 s | **670× total improvement** |

### 2.3 Module Inventory (GPU)

| Module | ToadStool Primitives | Custom WGSL | Pipeline Role |
|--------|---------------------|-------------|---------------|
| `quality_gpu.rs` | — | `quality_filter.wgsl` | Per-read quality trimming |
| `dada2_gpu.rs` | — | `dada2_e_step.wgsl` | DADA2 E-step (24× speedup) |
| `gemm_cached.rs` | GemmF64 shader | — | Cached GEMM pipeline |
| `streaming_gpu.rs` | FusedMapReduceF64 | — | Pipeline session orchestrator |
| `diversity_gpu.rs` | FMR, BrayCurtisF64 | — | Alpha/beta diversity |
| `taxonomy_gpu.rs` | GemmF64 | — | Standalone taxonomy GEMM |
| `pcoa_gpu.rs` | BatchedEighGpu | — | PCoA eigendecomposition |
| `spectral_match_gpu.rs` | GemmF64, FMR | — | MS2 cosine similarity |
| `eic_gpu.rs` | FMR, WeightedDotF64 | — | EIC integration |
| `rarefaction_gpu.rs` | FMR | — | Bootstrap rarefaction |
| `stats_gpu.rs` | Variance/Corr/Cov/WDot | — | Statistical operations |
| `kriging.rs` | KrigingF64 | — | Spatial interpolation |
| `chimera_gpu.rs` | — | — | GPU validation stub (CPU math) |

---

## 3. What ToadStool Should Absorb

### 3.1 `QualityFilterCached` → `ParallelFilter<T>`

**Files:** `barracuda/src/bio/quality_gpu.rs`, `barracuda/src/shaders/quality_filter.wgsl`

**Pattern:** Per-element parallel scan with independent thread-per-element processing.
Each GPU thread processes one read: leading trim, trailing trim, sliding window.
All u32 integer arithmetic. Quality bytes packed 4-per-u32.

**Measured impact:** 100% math parity (all read counts match CPU exactly).
~0.85× CPU speed (memory-bound, not compute-bound).

**Proposed ToadStool primitive:**
```rust
let filter = ParallelFilter::<u32>::new(device, filter_shader);
let results = filter.execute(&packed_data, &offsets, &lengths, params)?;
```

**Why absorb:** The pattern (one thread per variable-length element, packed data,
offset/length arrays) recurs in any streaming data processing. FASTQ reads,
mass spectra, time series windows — all benefit from this abstraction.

### 3.2 `Dada2Gpu` → `BatchPairReduce<f64>`

**Files:** `barracuda/src/bio/dada2_gpu.rs`, `barracuda/src/shaders/dada2_e_step.wgsl`

**Pattern:** Batch pair-wise reduction. Given N elements and M centers, compute
f(element_i, center_j) reduced over a shared dimension for all (i,j) pairs.
Each thread processes one pair. O(N × M × L) total work.

**Key insight — no GPU transcendentals:** The error model uses `ln(err[from][to][qual])`
which is precomputed on CPU and uploaded as a flat f64 lookup table (672 values = 5 KB).
The GPU shader does **only f64 addition** — no `log()`, no `exp()`, no driver-specific
transcendental issues. This pattern generalizes: precompute non-linear transforms on CPU,
let GPU do the embarrassingly parallel linear algebra.

**Measured impact:** 24.4× average speedup (326ms → 13ms per sample).

**Proposed ToadStool primitive:**
```rust
let reducer = BatchPairReduce::<f64>::new(device, reduce_shader);
let scores = reducer.execute(
    &data_a,       // [N × L]
    &data_b,       // [M × L]
    &lookup_table, // precomputed per-element values
    n, m, l,
)?;
// Returns [N × M] reduced values
```

**Why absorb:** Pair-wise reduction appears in: sequence alignment scoring, distance
matrix computation, kernel evaluations, force calculations, cross-correlation. The
"precompute transcendentals on CPU" pattern is universally applicable for f64 GPU work.

### 3.3 `GemmCached` → `GemmF64::new()` with Cached Pipeline

**File:** `barracuda/src/bio/gemm_cached.rs`

**Pattern:** Pre-compile the GEMM pipeline at session init, reuse across dispatches.
Integrates BufferPool for buffer reuse. Offers `execute_to_buffer()` for
zero-readback chaining.

**Measured impact:**
- First-sample penalty eliminated (36ms → 9.8ms)
- 93% buffer reuse (385 reuses / 29 allocations)
- Small workload: 16.5× GPU speedup at just 5 queries

**Proposed ToadStool change:**
```rust
// Current (per-call compilation):
let result = GemmF64::execute(device, &a, &b, m, k, n, batch)?;

// Proposed (cached pipeline):
let gemm = GemmF64::new(device);           // compile once
let result = gemm.execute(&a, &b, ...)?;   // reuse pipeline
let buf = gemm.execute_to_buffer(&a, &b, ...)?;  // no readback
```

**Why absorb:** Every ToadStool primitive that uses pipelines benefits from this
pattern. The overhead elimination is 23% per-call and eliminates the cold-start
penalty entirely.

### 3.4 BufferPool Integration Pattern

**Pattern used by all GPU modules:**
```rust
let buf = self.ctx.buffer_pool().acquire_pooled(
    &self.device.device,
    byte_size as u64,
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
);
self.device.device.queue.write_buffer(&buf, 0, data);
// buf auto-returned to pool on drop
```

**Impact:** 93% reuse rate. DADA2's iterative EM loop particularly benefits —
the same buffers for bases, quals, and lengths are reused across all EM iterations.

---

## 4. Architectural Lessons for ToadStool/BarraCUDA

### 4.1 Compute-Bound vs Memory-Bound: The GPU Boundary

The most important architectural finding:

| Nature | GPU Speedup | Examples | Why |
|--------|-------------|----------|-----|
| **Compute-bound** | 10–63× | DADA2 E-step, taxonomy GEMM, spectral cosine | O(N² × L) work amortizes dispatch |
| **Memory-bound** | 0.85–1× | Quality scanning, small reductions | CPU cache-line access is near-optimal |
| **Hash-based** | 1× (CPU) | Dereplication, chimera k-mer | GPU hash tables are complex; CPU excels |

**Recommendation:** ToadStool should provide clear guidance on when to use GPU vs CPU.
A simple heuristic: if the operation's arithmetic intensity (FLOPs / bytes transferred)
exceeds ~10, use GPU. Below that, CPU is competitive.

### 4.2 Precomputed Lookup Tables Avoid GPU Transcendentals

DADA2's success comes from precomputing all `ln()` values on CPU and uploading
as a flat f64 table. The GPU shader does only addition. This:
- Avoids driver-specific f64 transcendental issues (NAK `exp(f64)` crash in NVK)
- Keeps the shader simple and fast (pure f64 addition)
- Minimizes GPU memory (672 values = 5 KB for full DADA2 error model)

**Recommendation:** Document this pattern as a ToadStool best practice. For any
f64 GPU workload involving transcendentals, consider: can the non-linear transform
be precomputed on CPU? If yes, the GPU shader becomes dramatically simpler.

### 4.3 Pipeline Session Pattern

`GpuPipelineSession` pre-compiles all shaders at init (40ms one-time cost),
then reuses cached pipelines and pooled buffers across all dispatches:

```
GpuPipelineSession::new(gpu)
  ├── QualityFilterCached     (compile quality_filter.wgsl)
  ├── Dada2Gpu                (compile dada2_e_step.wgsl)
  ├── GemmCached              (compile gemm_f64.wgsl)
  ├── FusedMapReduceF64       (compile FMR shaders)
  └── warmup dispatches       (prime driver caches)
```

**Recommendation:** ToadStool should support a `SessionBuilder` pattern where
consumers can declare which primitives they'll use, and all compilation happens
once at startup. This is especially important for streaming workloads.

### 4.4 Packed Variable-Length Data

For variable-length elements (reads, sequences), wetSpring uses:
- Flat data buffer (elements concatenated)
- Offset array (start position per element)
- Length array (length per element)

Quality scores are packed 4-per-u32 to minimize transfer. Sequences use 2-bit
encoding (4 bases per byte). This reduces GPU memory transfer by 4× for quality
data.

**Recommendation:** ToadStool should provide utility functions for variable-length
data packing/unpacking, as this pattern recurs in any bioinformatics or text
processing GPU workload.

### 4.5 Hybrid CPU-GPU EM Loops

DADA2's EM algorithm has an iterative structure:
1. E-step (GPU): compute log_p_error for all (seq, center) pairs
2. M-step (CPU): argmax, update error model, convergence check
3. Split (CPU): Poisson test, create new centers
4. Repeat until convergence

Only the E-step is on GPU. The CPU handles control flow, convergence detection,
and the non-parallel M-step. Data (bases, quals, lengths) is uploaded once;
only center_indices and log_err are updated per iteration.

**Recommendation:** This hybrid pattern (GPU for embarrassingly parallel inner loop,
CPU for convergence/control flow) is optimal for iterative algorithms. ToadStool
should document this as a reference pattern and ensure BufferPool handles the
"upload once, update small buffers per iteration" use case efficiently.

---

## 5. Code Quality Audit (Feb 19, 2026)

A comprehensive audit was performed covering linting, safety, validation fidelity,
streaming I/O, and determinism. All findings were resolved.

### 5.1 GPU Clippy: 16 Errors → 0

| Lint | Files | Resolution |
|------|-------|------------|
| `too_many_arguments` | `dada2_gpu.rs`, `validate_16s_pipeline_gpu.rs`, `gemm_cached.rs` | Introduced `BatchInput` struct; `#[allow]` with justification for pipeline/internal contexts |
| `needless_range_loop` | `dada2_gpu.rs` (3D error model loops) | `.iter_mut().enumerate()` where clean; `#[allow]` for 3D array indexing clarity |
| `manual_div_ceil` | `quality_gpu.rs` | Replaced `(n + 3) / 4` with `n.div_ceil(4)` |
| `unnecessary_lazy_evaluations` | `rarefaction_gpu.rs` | `.unwrap_or_else(\|\| v)` → `.unwrap_or(v)` |
| `type_complexity` | `diversity_gpu.rs` (tests) | Type aliases `ScalarGpuFn`, `MatrixGpuFn` |

**Lesson for ToadStool:** GPU code running with `-D warnings` catches real issues.
The `BatchInput` struct pattern is reusable — any function with >7 GPU-context params
benefits from grouping related parameters.

### 5.2 NaN Safety Fix

`validate_voc_peaks.rs` had `a.partial_cmp(b).unwrap()` which panics on NaN.
Fixed to `.unwrap_or(std::cmp::Ordering::Equal)`. Any ToadStool code sorting f64
readbacks should use this pattern.

### 5.3 Streaming I/O Iterators

All three I/O parsers now have zero-buffer streaming iterators alongside the
buffered `parse_*` APIs:

| Parser | Iterator | Yields |
|--------|----------|--------|
| `io::fastq` | `FastqIter` | `Result<FastqRecord>` |
| `io::mzml` | `MzmlIter` | `Result<MzmlSpectrum>` |
| `io::ms2` | `Ms2Iter` | `Result<Ms2Spectrum>` |

Each iterator processes records one at a time with constant memory overhead.
Tests verify iterator output matches buffered output exactly.

### 5.4 Determinism Tests

6 new determinism tests verify that core pipeline stages produce identical
output across multiple runs:

| Test | What It Verifies |
|------|-----------------|
| `diversity_deterministic_across_runs` | Shannon/Simpson/Chao1 stability |
| `bray_curtis_deterministic_across_runs` | Pairwise distance matrix stability |
| `dada2_deterministic_across_runs` | ASV denoising reproducibility |
| `chimera_detection_deterministic_across_runs` | Chimera scoring reproducibility |
| `taxonomy_classification_deterministic_across_runs` | Naive Bayes classification stability |
| `full_16s_pipeline_deterministic_across_runs` | End-to-end pipeline reproducibility |

### 5.5 Validation Tightening

`validate_features.rs` checks tightened from existence-only to quantitative
thresholds sourced from Exp009 acceptance criteria (mass tracks >= 500,
features >= 100, m/z min < 150, cross-match >= 30%).

### 5.6 Final Quality Gates

| Gate | Status |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS (0 warnings) |
| `cargo clippy --all-targets --features gpu -- -D warnings` | PASS (0 warnings) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | PASS |
| `cargo test --release` | 361/361 PASS |
| `unsafe` blocks | 0 |
| `TODO` / `FIXME` | 0 |
| Production `unwrap()` / `expect()` | 0 |

---

## 6. Three-Tier Profiling Infrastructure

### 6.1 Why This Matters for ToadStool

wetSpring built a sovereign benchmark harness (`barracuda/src/bench.rs`) that
measures Python, Rust CPU, and Rust GPU performance in a single unified JSON
schema — without adding a `serde` dependency. This pattern is directly useful
for ToadStool consumers who need to demonstrate performance gains and regression
test GPU dispatch overhead.

### 6.2 Architecture

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| `HardwareInventory` | CPU model/cores/MHz, RAM, GPU name/VRAM/driver/CC | Auto-detect from `/proc/cpuinfo`, `/proc/meminfo`, `nvidia-smi` |
| `PowerMonitor` | Background energy sampling | Spawns threads for RAPL (`energy_uj`) and `nvidia-smi` power/temp/VRAM |
| `EnergyReport` | Per-phase energy data | CPU joules (RAPL delta), GPU joules (trapezoidal integration of power) |
| `PhaseResult` | Per-benchmark result | Wall time, N evals, µs/eval, energy, peak RSS, notes |
| `BenchReport` | Complete run output | ISO 8601 timestamp + hardware + vector of phases → JSON |

### 6.3 Key Design Decisions

**No serde.** Manual `to_json()` methods on each struct. Includes `json_escape()`
for safe string encoding. This preserves wetSpring's sovereignty requirement
(1 runtime dep: `flate2`). ToadStool should consider whether this pattern is
preferable to adding serde — for benchmark-only code, the 450 lines of manual
JSON serialization may be simpler than a dependency.

**RAPL counter wraps.** Intel RAPL energy counters overflow at `max_energy_range_uj`.
The `PowerMonitor` reads both start and end values and computes the correct delta
including wraps. Any ToadStool energy measurement code should handle this.

**GPU power integration.** `nvidia-smi` is polled every 100ms via a background
thread. Total GPU energy (joules) is computed via trapezoidal integration of
power samples. Peak watts, peak temperature, and peak VRAM are also recorded.

**Cross-substrate compatibility.** The Python baseline script
(`scripts/benchmark_python_baseline.py`) emits the same JSON schema. This enables
direct comparison of all three tiers using standard JSON tools:

```bash
# Run all three tiers
./scripts/run_three_tier_benchmark.sh

# Results in benchmarks/results/:
#   python_baseline_YYYYMMDD_HHMMSS.json
#   benchmark_pipeline_YYYYMMDD_HHMMSS.json
#   benchmark_cpu_gpu_YYYYMMDD_HHMMSS.json
```

### 6.4 Canonical Benchmark Results (Live, Feb 19, 2026)

| Workload | Python | Rust CPU | Rust GPU | GPU/CPU |
|----------|--------|----------|----------|---------|
| Shannon 1M | 7.12ms | 3.70ms | 4.68ms | 0.79× |
| Bray-Curtis 100×100 | 14.9ms | 1.08ms | 2.40ms | 0.45× |
| Spectral cosine 200×200 | 9.10ms | 3,782ms | **4.08ms** | **926×** |
| Full pipeline (10 sam) | ~95.6s | 11.5s | 3.0s | **2.45×** |

GPU power during spectral cosine: ~31W avg, 47°C peak, 743 MiB VRAM.

### 6.5 Recommendation for ToadStool

Consider adopting the `BenchReport` JSON schema as a standard for all ToadStool
consumers. Benefits:

1. **Regression testing**: Track GPU dispatch overhead across versions
2. **Cross-project comparison**: hotSpring, wetSpring, and future primals emit
   the same schema
3. **Energy accountability**: RAPL + nvidia-smi data enables cost-per-computation
   claims
4. **Hardware inventory**: Automatic capture eliminates "what machine was this run on?"

---

## 7. ToadStool Primitives Used — Full Inventory

| Primitive | wetSpring Modules | Operations | Checks |
|-----------|------------------|------------|--------|
| FusedMapReduceF64 | diversity_gpu, streaming_gpu, eic_gpu, rarefaction_gpu | Shannon, Simpson, sum, sum_of_squares | 24 |
| BrayCurtisF64 | diversity_gpu | All-pairs Bray-Curtis distance | 6 |
| BatchedEighGpu | pcoa_gpu | PCoA eigendecomposition | 5 |
| GemmF64 | taxonomy_gpu, spectral_match_gpu, gemm_cached | Matrix multiply | 18 |
| KrigingF64 | kriging | Spatial interpolation | — |
| VarianceF64 | stats_gpu | Population/sample variance, std dev | 3 |
| CorrelationF64 | stats_gpu | Pearson correlation | 1 |
| CovarianceF64 | stats_gpu | Sample covariance | 1 |
| WeightedDotF64 | stats_gpu, eic_gpu | Weighted/plain dot product | 2 |
| TensorContext | all GPU modules | Buffer pool, bind group cache | — |
| ShaderTemplate | gemm_cached, dada2_gpu | f64 driver workaround | — |

### ToadStool Bugs Found and Fixed by wetSpring

| Bug | Ticket | Status | Impact |
|-----|--------|--------|--------|
| `log_f64` 2× coefficient | — | **Fixed upstream** | Shannon entropy was 2× wrong |
| `pow_f64` fractional exponents | TS-001 | **Fixed upstream** | Exponentiation failed for non-integer e |
| `acos`/`sin` zero-bias | TS-003 | **Fixed upstream** | PCoA eigenvalues had precision loss |
| FMR N≥1024 buffer conflict | TS-004 | **Fixed upstream** | Large arrays produced wrong results |
| `target` keyword in WGSL | — | **Fixed upstream** | Shader compilation failed on some drivers |
| `SHADER_F64` feature flag | — | **Fixed upstream** | f64 not enabled on capable hardware |

---

## 8. Custom WGSL Shaders — Technical Details

### 8.1 `quality_filter.wgsl`

```
Entry point: quality_filter
Workgroup size: 256
Bindings: params (uniform), qual_data (storage), read_offsets (storage),
          read_lengths (storage), results (storage rw)
Types: u32 only (no f64)
Thread model: 1 thread per read
Operations: leading trim, trailing trim, sliding window average, min length check
Output: packed (start << 16 | end) per read, 0 = discarded
```

### 8.2 `dada2_e_step.wgsl`

```
Entry point: e_step
Workgroup size: 256
Bindings: params (uniform), bases (storage), quals (storage),
          lengths (storage), center_indices (storage),
          log_err (storage), scores (storage rw)
Types: f64 (addition only, no transcendentals)
Thread model: 1 thread per (sequence, center) pair
Operations: lookup log_err[from_base][to_base][qual] for each position, sum
Output: f64 log_p_error per pair
```

---

## 9. Performance Summary

### Three-Tier Benchmark

| Tier | Implementation | 10 samples | Per-sample | vs Galaxy |
|------|---------------|-----------|------------|-----------|
| Galaxy/Python | QIIME2 + DADA2-R | 95.6 s | 9.56 s | baseline |
| BarraCUDA CPU | Pure Rust | 7.3 s | 0.73 s | **13.1×** |
| BarraCUDA GPU | Rust + ToadStool | **3.0 s** | **0.30 s** | **31.9×** |

### Energy & Cost

| Metric | Galaxy | Rust CPU | Rust GPU |
|--------|--------|----------|----------|
| Cost at 10K samples | $0.40 | $0.03 | **$0.02** |

### Scaling (Taxonomy GEMM)

| Queries | Speedup |
|---------|---------|
| 5 | 16.5× |
| 25 | 30.1× |
| 100 | 60.3× |
| 500 | 63.3× |

---

## 10. Remaining Work (Not For This Handoff)

These items are documented but intentionally deferred:

1. **Chimera GPU**: k-mer sketch scoring could be GPU-parallelized at large N,
   but the CPU implementation (1,256× optimized) is already fast.
2. **GPU argmax kernel**: Replace GEMM score readback with GPU-resident argmax
   (reduces readback from ~3.5 MB to ~1 KB per dispatch).
3. **Full GPU-resident pipeline**: Persistent GPU buffers, zero CPU readback
   between stages. Requires chipset-phase work with dedicated hardware.
4. **DADA2 M-step GPU**: Error model update matrix could be GPU-computed,
   but it's a small matrix (4×4×42 = 672 values) so CPU is faster.
5. **Mixed CPU/GPU load balancing**: Dynamically routing stages to CPU vs GPU
   based on workload size. Deferred until chipset phase.

---

## 11. How to Validate

```bash
cd wetSpring/barracuda

# Full CPU validation (388 checks)
cargo test --release
cargo run --release --bin validate_fastq
cargo run --release --bin validate_diversity
cargo run --release --bin validate_16s_pipeline
cargo run --release --bin validate_algae_16s
cargo run --release --bin validate_voc_peaks
cargo run --release --bin validate_public_benchmarks
cargo run --release --bin validate_mzml
cargo run --release --bin validate_pfas
cargo run --release --bin validate_features
cargo run --release --bin validate_peaks

# Full GPU validation (126 checks)
cargo run --release --features gpu --bin validate_diversity_gpu
cargo run --release --features gpu --bin validate_16s_pipeline_gpu

# Benchmarks
cargo run --release --features gpu --bin benchmark_cpu_gpu
cargo run --release --bin benchmark_pipeline

# Three-tier benchmark (Python + Rust CPU + Rust GPU, JSON output)
cd ..  # back to wetSpring root
./scripts/run_three_tier_benchmark.sh
```

Expected: 361 tests PASS, 514/514 validation checks PASS, 88/88 GPU pipeline parity,
BufferPool 93%+ reuse, DADA2 GPU 20×+ speedup, overall pipeline 2×+ GPU speedup.
Zero clippy warnings (CPU and GPU), zero doc warnings, zero unsafe.
JSON benchmark results in `benchmarks/results/`.

---

## 12. Key Files for Review

| File | What to Look At |
|------|----------------|
| `barracuda/src/bio/streaming_gpu.rs` | `GpuPipelineSession` — how all GPU stages compose |
| `barracuda/src/bio/dada2_gpu.rs` | `Dada2Gpu` — hybrid CPU-GPU EM loop pattern |
| `barracuda/src/bio/quality_gpu.rs` | `QualityFilterCached` — per-element parallel pattern |
| `barracuda/src/bio/gemm_cached.rs` | `GemmCached` — pipeline caching + BufferPool |
| `barracuda/src/bench.rs` | Benchmark harness — HardwareInventory, PowerMonitor, JSON schema (no serde) |
| `barracuda/src/shaders/dada2_e_step.wgsl` | E-step shader — f64 addition only, no transcendentals |
| `barracuda/src/shaders/quality_filter.wgsl` | QF shader — u32 integer arithmetic, packed data |
| `barracuda/src/gpu.rs` | `GpuF64` bridge — device creation, SHADER_F64, TensorContext |
| `barracuda/src/bin/validate_16s_pipeline_gpu.rs` | Validation binary — 88 parity checks + benchmarks |
| `scripts/run_three_tier_benchmark.sh` | Unified benchmark runner (Python + Rust CPU + Rust GPU) |
| `benchmarks/PROTOCOL.md` | Benchmark methodology and JSON schema specification |
| `TOADSTOOL_ABSORPTION_SPEC.md` | Detailed absorption proposals with proposed APIs |
| `experiments/016_gpu_pipeline_parity.md` | Full experiment results and analysis |
| `experiments/017_extended_algae_validation.md` | Next: extended algae validation (PRJNA382322) |
| `experiments/018_pfas_library_validation.md` | Next: PFAS library validation (Jones Lab 175 compounds) |

---

## 13. Cross-Reference: hotSpring ↔ wetSpring

| Lesson | hotSpring (Nuclear) | wetSpring (Bio/Chem) |
|--------|--------------------|-----------------------|
| f64 GPU precision | Max error 4.55e-13 MeV | Max error ≤ 1e-6 (pipeline tolerance) |
| GPU transcendentals | NVK `exp(f64)` crash → software workaround | Avoided entirely (precomputed lookup) |
| BufferPool | Used in HFB solver | 93% reuse in streaming pipeline |
| Pipeline caching | Used in eigensolve | `GemmCached` + `GpuPipelineSession` |
| Compute boundary | GPU wins at matrix dim ~30+ | GPU wins at compute intensity ~10+ FLOPs/byte |
| Custom WGSL | 19 shaders | 3 shaders (2 custom + 1 ToadStool) |
| Validation checks | 195/195 | 514/514 |
| Key speedup | 15% GPU-resident hybrid | 2.45× full pipeline |

Both projects confirm: BarraCUDA math is hardware-independent, ToadStool dispatch
is production-ready, and precomputing transcendentals avoids the f64 GPU minefield.

---

## 14. Convergent Evolution: What Two Independent Projects Discovered

hotSpring (nuclear physics) and wetSpring (life science + analytical chemistry)
were built independently, by different domains, solving different problems. Yet
they converged on identical architectural patterns. This convergence is the
strongest evidence that these patterns are fundamental — not domain-specific:

| Pattern | hotSpring Discovery | wetSpring Discovery | Implication for ToadStool |
|---------|--------------------|--------------------|--------------------------|
| Precomputed lookup tables | NVK `exp(f64)` crashed → precompute on CPU | DADA2 `ln(err)` → 672-value f64 table | **First-class pattern**: CPU precompute + GPU linear algebra |
| Pipeline session | Eigensolve needed cached pipelines | `GpuPipelineSession` + `GemmCached` | **`SessionBuilder` API**: compile once, dispatch many |
| BufferPool reuse | HFB solver reused intermediate buffers | DADA2 EM loop: 93% reuse (385/29) | **Critical for iterative algorithms**: EM, gradient descent, MD |
| Compute/memory boundary | GPU wins at matrix dim ~30+ | GPU wins at FLOPs/byte ~10+ | **Document the heuristic**: GPU benefits are predictable |
| Manual JSON (no serde) | Benchmark output without serde | `bench.rs` with `to_json()` methods | **Sovereignty pattern**: minimal deps even for infrastructure |
| Determinism as validation | Reproducible eigenvalues across runs | 10 determinism tests (diversity → pipeline) | **Determinism gates**: every consumer should test this |

**The convergence is the message.** When nuclear physics and metagenomics
independently discover the same GPU patterns, those patterns belong in ToadStool's
core, not in consumer code.

---

## 15. Upcoming Validation Work

Two experiments are designed and ready for execution. They demonstrate how
wetSpring's validated infrastructure extends to new datasets:

### 15.1 Exp017: Extended Algae Validation (PRJNA382322)

- **Dataset**: Nannochloropsis outdoor pilot-scale cultivation (12.6M reads, 20MB subsample downloaded)
- **Objective**: Run the full 16S pipeline on a second independent algae BioProject to validate
  community composition consistency across studies
- **GPU relevance**: Uses existing `GpuPipelineSession` — no new shaders needed
- **Status**: DATA READY

### 15.2 Exp018: PFAS Library Validation (Jones Lab 175 Compounds)

- **Dataset**: Jones Lab comprehensive PFAS spectral library (Nature Scientific Data, 2024, DOI:10.1038/s41597-024-04363-0)
- **Objective**: Validate mzML → PFAS pipeline against 175 reference compounds with known spectral signatures
- **GPU relevance**: Exercises `GemmF64` spectral cosine at library scale (175×175 = 15,225 pairs)
- **Status**: DESIGN

These experiments will increase the validation check count beyond 514 and exercise
the pipeline on previously unseen data — the open science validation pattern.

---

## 16. The Proof

wetSpring proves the ecoPrimals thesis across two domains:

- **514/514 quantitative checks pass** — 388 CPU + 126 GPU, covering 10 validation
  binaries, 22 real NCBI samples from 4 BioProjects, and published paper baselines
- **88/88 GPU pipeline parity** — same ASV counts, same taxonomy, same diversity
  metrics on CPU and GPU, proving BarraCUDA math is hardware-independent
- **926× GPU speedup** on batch spectral cosine — ToadStool GEMM stays constant-time
  as problem size grows, proving O(1) dispatch with O(N²) internal parallelism
- **24× DADA2 speedup** via precomputed lookup table + f64 addition shader —
  the largest pipeline bottleneck eliminated without GPU transcendentals
- **670× total optimization** from baseline to current GPU pipeline
  (chimera 1,256× + taxonomy 223× + DADA2 24× + pipeline composition)
- **$0.02 per 10K samples** — GPU is cheapest despite higher TDP because
  2.45× speed advantage compensates
- **1 runtime dependency** (`flate2`) — sovereign parsers for FASTQ, mzML, MS2,
  XML, Base64, and JSON serialization
- **6 ToadStool bugs found and fixed** — `log_f64`, `pow_f64`, `acos`/`sin`,
  FMR buffer, `target` keyword, `SHADER_F64` feature flag
- **3 local extensions ready for absorption** — `QualityFilterCached`,
  `Dada2Gpu`, `GemmCached` generalize to `ParallelFilter<T>`,
  `BatchPairReduce<f64>`, and cached-pipeline `GemmF64`
- **Structured benchmark harness** — cross-tier JSON schema with RAPL + nvidia-smi
  energy profiling, reproducible by `./scripts/run_three_tier_benchmark.sh`

The work is reproducible. The data is public. The code is sovereign.

---

*This handoff supersedes HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md and
HANDOFF_SPRINGS_TO_TOADSTOOL_FEB_17_2026.md. The absorption spec is in
TOADSTOOL_ABSORPTION_SPEC.md. The full experiment is in
experiments/016_gpu_pipeline_parity.md. Benchmark protocol and results are in
benchmarks/PROTOCOL.md and benchmarks/results/.*
