# wetSpring → ToadStool/BarraCuda Handoff V62

**Date:** February 27, 2026
**From:** wetSpring (life science biome)
**To:** ToadStool/BarraCuda evolution team
**Supersedes:** V61 (archived)

---

## Executive Summary

wetSpring completed Phase 62 — a comprehensive green sweep validating the full
pipeline chain from Python baselines through metalForge cross-substrate dispatch.
Key additions: biomeOS IPC science primal integration, GPU-aware dispatch routing
via lazy `OnceLock<GpuF64>` + dispatch threshold, and NUCLEUS atomics modeling
for Tower/Node/Nest coordination. **209 experiments, 5,021+ checks, 1,103 tests,
all PASS, clippy clean.**

---

## Part 1: How wetSpring Uses BarraCUDA (Current State)

### Primitive Consumption

| Metric | Count |
|--------|-------|
| ToadStool primitives consumed | 79 (via `compile_shader_universal`) |
| Local WGSL shaders | 0 (fully lean) |
| Local ODE derivative math | 0 (trait-generated WGSL) |
| Local regression/special math | 0 (delegated to upstream) |
| Passthrough wrappers | 0 (all promoted) |
| CPU bio modules | 47 |
| GPU bio modules | 42 (27 lean + 5 write→lean + 7 compose) |
| I/O modules | 5 (FASTQ, mzML, MS2, XML, nanopore) |
| IPC modules | 1 (dispatch — JSON-RPC science routing) |

### New in V62: IPC Dispatch Layer

wetSpring now includes a biomeOS-integrated IPC server (`wetspring_server`)
that exposes science capabilities via JSON-RPC 2.0 over Unix sockets:

- `science.diversity` — Shannon, Simpson, Chao1, S_obs, Pielou, Bray-Curtis
- `science.qs_model` — Waters 2008 QS/c-di-GMP ODE (4 scenarios)
- `science.anderson` — 3D Anderson spectral analysis (Lanczos eigenvalues, level spacing)
- `science.ncbi_fetch` — NCBI sequence retrieval (EFetch/SRA, NestGate fallback)
- `science.full_pipeline` — Multi-stage chained analysis

**GPU-aware routing:** When `--features gpu` is active, the dispatch layer
initializes `GpuF64` lazily via `std::sync::OnceLock`. For diversity methods,
workloads exceeding `GpuF64::dispatch_threshold()` route to GPU (`diversity_gpu::*`);
smaller workloads stay on CPU. Anderson spectral analysis always prefers GPU
context when available.

### Upstream vs Local Module Breakdown

| Category | Modules | Source |
|----------|:-------:|--------|
| **Lean** (upstream ToadStool) | 27 | `compile_shader_universal` |
| **Write→Lean** (ODE trait-generated) | 5 | `BatchedOdeRK4<S>::generate_shader()` |
| **Compose** (wiring ToadStool primitives) | 7 | kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, mol_clock |
| **CPU-only** | 2 | phred (lookup table), fastq_parsing (I/O bound) |
| **IPC dispatch** | 1 | JSON-RPC router (conditionally GPU-aware) |

---

## Part 2: Absorption Candidates

### Ready for Absorption (unchanged from V61)

| Module | What | Complexity | Benefit |
|--------|------|-----------|---------|
| `bio::esn` | Echo State Network reservoir | 1 file, 400 LOC | NPU inference across all Springs |
| `npu.rs` | NPU DMA bridge (int8 inference) | 1 file, 200 LOC | Neuromorphic dispatch for all Springs |
| `validation.rs` | `Validator` harness (check/section/summary) | 1 file, 150 LOC | Consistent validation across all Springs |

### New Observation: IPC Dispatch Pattern

The `ipc::dispatch` module demonstrates a pattern that could be valuable for
ToadStool's own capability routing:

- **Lazy GPU initialization** via `OnceLock` — one-time async setup, amortized
- **Dispatch threshold** — workload size determines CPU vs GPU routing
- **Feature-gated paths** — `#[cfg(feature = "gpu")]` cleanly separates GPU codepath
- **JSON-RPC method routing** — extensible method dispatch table

This pattern could inform how ToadStool routes compute requests from multiple
Springs, especially for biomeOS graph-coordinated workloads.

### Deferred (same as V61)

- `reconciliation_gpu` — depends on `TreeInferenceGpu` API stability
- `io::nanopore` — needs real MinION hardware validation first
- `bio::basecall` — not yet implemented (awaiting MinION)

---

## Part 3: BarraCuda Evolution — What We Learned

### V62 Findings

1. **GPU dispatch overhead is real.** At small workload sizes (< dispatch
   threshold), CPU beats GPU due to ~0.5-2ms fixed dispatch cost. The
   `dispatch_threshold()` pattern is essential for production routing.

2. **Lazy GPU initialization matters.** `GpuF64` creation involves wgpu adapter
   selection and device creation (~170ms). Using `OnceLock` amortizes this to
   zero for subsequent calls. ToadStool should consider a similar pattern for
   Springs that intermittently need GPU.

3. **Streaming vs round-trip gap widened.** Our Exp091 benchmark shows streaming
   (Str/RT = 0.68-0.77 at large batches) eliminates 23-32% of round-trip overhead.
   For the ToadStool streaming pipeline, pre-warming `GpuPipelineSession` and
   `execute_to_buffer()` are the right abstractions.

4. **Anderson spectral analysis is CPU-bound.** The Lanczos tridiagonalization
   in `barracuda::spectral::lanczos` dominates GPU validation wall-clock time
   (~457ms for L=8). A GPU Lanczos kernel would be the highest-impact
   absorption target for spectral workloads.

5. **IPC adds zero numeric drift.** Exp206 proves that serialization through
   JSON-RPC (`serde_json`) and back introduces exactly 0.0 error for all
   science capabilities. The IPC layer is purely structural.

6. **erf polynomial approximation accuracy.** The Abramowitz & Stegun `erf`
   polyfill achieves ~5e-7 accuracy, NOT machine precision (1e-12). We
   corrected a tolerance mismatch in the S68 cross-spring benchmark. Upstream
   `barracuda::special::erf` should document this limitation.

### Three-Tier Paper Controls

| Track | Papers | CPU | GPU | metalForge | Status |
|-------|:------:|:---:|:---:|:----------:|--------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | Full three-tier |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | Full three-tier |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | Full three-tier |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 4 (Soil QS/Anderson) | 9 | 9/9 | 9/9 | 9/9 | Full three-tier |
| **Total** | **39** | **39/39** | **39/39** | **39/39** | **ALL** |

---

## Part 4: Hardware Validation Matrix

| Substrate | Hardware | Experiments | Checks | Status |
|-----------|----------|:-----------:|:------:|--------|
| CPU (Rust) | i9-12900K | 209 | 5,021+ | ALL PASS |
| GPU (WGSL) | RTX 4070 | 44 | 1,759+ | ALL PASS |
| NPU (int8) | AKD1000 | 6 | 60 | ALL PASS |
| metalForge | CPU+GPU+NPU | 10 | 165+ | ALL PASS |
| Streaming | GPU pipeline | 5 | 204+ | ALL PASS |
| IPC dispatch | CPU or GPU | 6 | 321 | ALL PASS |

---

## Part 5: Comprehensive Sweep Results (Feb 27, 2026)

28 validation binaries re-run in a single session:

### CPU Tier
| Binary | Checks | Time |
|--------|--------|------|
| `validate_barracuda_cpu_full` (Exp070) | 50/50 | 2.3s |
| `validate_barracuda_cpu_v9` (Exp163) | 27/27 | 1.0s |
| `validate_barracuda_cpu_v10` (Exp190) | 75/75 | — |
| `validate_barracuda_cpu_v11` (Exp206) | 64/64 | — |
| `validate_soil_qs_cpu_parity` (Exp179) | 49/49 | — |
| `validate_nanopore_*` (Exp196a-c) | 52/52 | 1.3s |

### GPU Tier
| Binary | Checks | Time |
|--------|--------|------|
| `validate_barracuda_gpu_full` (Exp071) | 24/24 | 4.7s |
| `validate_barracuda_gpu_v1` (Exp064) | 26/26 | 4.0s |
| `validate_barracuda_gpu_v4` (Exp207) | 54/54 | 13.8s |
| `validate_soil_qs_gpu` (Exp180) | 23/23 | 3.4s |

### Pure GPU Streaming
| Binary | Checks | Time |
|--------|--------|------|
| `validate_pure_gpu_streaming` (Exp090) | 80/80 | 4.7s |
| `validate_pure_gpu_streaming_v2` (Exp105) | 27/27 | 1.2s |
| `validate_streaming_ode_phylo` (Exp106) | 45/45 | 2.1s |
| `validate_pure_gpu_complete` (Exp101) | 52/52 | 1.9s |

### metalForge
| Binary | Checks | Time |
|--------|--------|------|
| `validate_metalforge_v5` (Exp103) | 52/52 | 1.8s |
| `validate_metalforge_v6` (Exp104) | 24/24 | 1.3s |
| `validate_metalforge_v7_mixed` (Exp208) | 75/75 | 1.5s |

### Cross-Spring Benchmarks
| Binary | Checks | Status |
|--------|--------|--------|
| `benchmark_cross_spring_s68` (Exp189) | 28/28 | PASS |
| `benchmark_cross_spring_s65` (Exp183) | 36/36 | PASS |
| `benchmark_cross_spring_modern` (Exp169) | 20/20 | PASS |
| `benchmark_modern_systems_df64` (Exp166) | 19/19 | PASS |

### Benchmark Progression
```
Python (1,713 ms) → Rust CPU (51 ms, 33.4×) → GPU (portable) → Streaming (441-837× vs RT)
```

---

## Part 6: Recommended Actions for ToadStool/BarraCuda Team

### High Priority

1. **GPU Lanczos kernel** — `barracuda::spectral::lanczos` is the primary CPU
   bottleneck in GPU validation. A WGSL Lanczos tridiagonalization would
   eliminate the ~457ms CPU cost for Anderson spectral analysis. wetSpring
   and hotSpring both use this heavily.

2. **Document `erf` precision** — `barracuda::special::erf` achieves ~5e-7
   (Abramowitz & Stegun polynomial), not machine epsilon. Add a doc comment
   noting the accuracy bound. Springs should use `tolerances::ERF_PARITY`
   not `tolerances::ANALYTICAL_F64`.

3. **ESN absorption** — `bio::esn` is stable, well-tested, and used by both
   wetSpring (6 classifiers) and neuralSpring. Single-file extraction.

### Medium Priority

4. **Dispatch threshold API** — Consider exposing `dispatch_threshold()` as
   a ToadStool-level concept. Different Springs have different crossover
   points; a unified API would help biomeOS route optimally.

5. **`OnceLock<GpuF64>` pattern** — wetSpring's lazy GPU init pattern could
   be upstreamed as a `ToadStool::lazy_context()` helper. Springs that
   intermittently need GPU would benefit.

6. **NPU DMA bridge** — `npu.rs` is hardware-validated (AKD1000) and could
   serve as the reference NPU integration for other Springs.

### Low Priority

7. **BatchReconcileGpu** — reconciliation_gpu compose wrapper depends on
   `TreeInferenceGpu` stability. Not blocking.

8. **`io::nanopore` upstream** — waiting for MinION hardware validation.
   POD5/NRS parser is sovereign and dependency-free.

---

## Part 7: Code Quality Summary

| Check | Status |
|-------|--------|
| Clippy (pedantic + nursery) | CLEAN (0 warnings, all features, all targets) |
| Tests | 1,103 (977 lib + 60 integration + 19 doc + 47 forge) |
| Coverage | 95.46% line / 93.54% fn / 94.99% branch |
| Unsafe code | 0 production (deny crate-wide; allow only in test env-var helpers) |
| TODO/FIXME | 0 |
| Named tolerances | 92 (all scientifically justified) |
| Local WGSL | 0 (fully lean) |
| Passthrough | 0 (all promoted) |
| ToadStool alignment | S68+ (`e96576ee`) |

---

## Appendix: File Manifest (New in V62)

```
barracuda/src/ipc/dispatch.rs          — JSON-RPC method router (GPU-aware)
barracuda/src/ipc/mod.rs               — IPC module root
barracuda/src/bin/validate_barracuda_cpu_v11.rs  — Exp206 (64 checks)
barracuda/src/bin/validate_barracuda_gpu_v4.rs   — Exp207 (54 checks)
barracuda/src/bin/validate_metalforge_v7_mixed.rs — Exp208 (75 checks)
barracuda/src/bin/validate_science_pipeline.rs    — Exp203 (29 checks)
barracuda/src/bin/wetspring_server.rs             — biomeOS IPC server
experiments/203_biomeos_science_pipeline.md
experiments/206_barracuda_cpu_v11_ipc_dispatch.md
experiments/207_barracuda_gpu_v4_ipc_science.md
experiments/208_metalforge_v7_mixed_nucleus.md
```
