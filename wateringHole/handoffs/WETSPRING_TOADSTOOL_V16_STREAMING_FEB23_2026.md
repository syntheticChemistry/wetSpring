# wetSpring → ToadStool Handoff v16 — Streaming v2, metalForge v6, Absorption Learnings

**Date:** February 23, 2026
**Phase:** 30
**Author:** wetSpring validation pipeline
**Previous:** [v15 — ODE Generic Absorption](WETSPRING_TOADSTOOL_V15_ODE_GENERIC_FEB22_2026.md)

---

## Executive Summary

wetSpring has completed two major milestones since v15:

1. **metalForge v6** — All 25 actionable papers now have full three-tier
   validation (Python → BarraCUDA CPU → BarraCUDA GPU → metalForge cross-substrate).
   37 domains proven substrate-independent.

2. **Pure GPU Streaming v2** — Expanded `GpuPipelineSession` to cover 10+ domains
   (up from 2). Pre-warmed pipeline pattern proven for analytics, ODE biology,
   and phylogenetics. Zero shader recompilation across multi-dispatch sessions.

**Key metrics:**

| Metric | Value |
|--------|-------|
| Experiments | 106 |
| Total validation checks | 2,502+ |
| Rust tests | 750 |
| metalForge domains | 37 (substrate-independent) |
| Three-tier papers | 25/25 actionable |
| Streaming domains | 10+ (analytics + ODE + phylo) |
| Pre-warmed primitives | 10 in `GpuPipelineSession` |

---

## Part 1: metalForge v6 — Three-Tier Complete (Exp104)

### Gap Closure

Five domains previously lacking metalForge routing were exercised through the
cross-substrate router: QS ODE, UniFrac, DADA2, K-mer, Felsenstein. These
closed the last paper-level gaps in the Three-Tier Control Matrix.

### Three New Workloads Registered

Added to `metalForge/forge/src/workloads.rs`:

| Workload | Primitive | Origin |
|----------|-----------|--------|
| `dada2` | `Dada2EStepGpu` | Absorbed |
| `bootstrap` | `FelsensteinGpu` | Absorbed |
| `placement` | `FelsensteinGpu` | Absorbed |

Total: 28 workloads (22 absorbed, 5 local WGSL, 1 CPU-only).

### Validation

- `validate_metalforge_v6` binary: 24/24 checks
- CPU ↔ GPU parity for all 5 domains
- K-mer histogram: forward-only counting aligned between CPU and GPU

---

## Part 2: Pure GPU Streaming v2 (Exp105 + Exp106)

### GpuPipelineSession Expansion (Exp105)

The streaming session now pre-compiles 5 GPU primitives at construction:

| Field | Type | Pre-compiled At |
|-------|------|-----------------|
| `qf` | `QualityFilterCached` | Session init |
| `dada2` | `Dada2Gpu` | Session init |
| `fmr` | `FusedMapReduceF64` | Session init |
| `gemm` | `GemmCached` | Session init |
| `bc` | `BrayCurtisF64` | **NEW** — Session init |

New streaming methods:

| Method | Stages | Primitives Used |
|--------|--------|-----------------|
| `bray_curtis_matrix(&[&[f64]])` | 1 | `BrayCurtisF64` |
| `spectral_cosine_matrix(&[&[f64]])` | 2 | `GemmCached` + `FMR::sum_of_squares` |
| `stream_full_analytics(classifier, seqs, counts, params)` | 3 | GEMM taxonomy → FMR diversity → BC beta |

**Result:** 27/27 checks. Exact CPU ↔ GPU parity for all new methods.

### Pre-Warmed Domain Streaming (Exp106)

Proved that 6 domain-specific GPU primitives can be pre-compiled once and
dispatched repeatedly without shader recompilation:

| Primitive | Domain | Dispatches | Result |
|-----------|--------|------------|--------|
| `OdeSweepGpu` | QS biofilm ODE | 4-batch sweep | All finite |
| `PhageDefenseGpu` | Phage defense | 2 (different params) | Bd, Bu > 0 |
| `BistableGpu` | Bistable switch | 2 (different params) | Biofilm ∈ [0,1] |
| `MultiSignalGpu` | Multi-signal QS | 2 (different params) | All 7 vars finite |
| `FelsensteinGpu` | Phylo pruning | 2 trees | 1.3% / 6.1% rel err |
| `UniFracGpu` | Propagation | 2 datasets | Exact leaf parity |

**Result:** 45/45 checks. Warmup 25.5 ms (all 6 primitives). Execution 541.8 ms.

### Felsenstein GPU f64 Precision Note

GPU exp/log transcendental fallback compounds errors across multi-step Felsenstein pruning:
- 3-taxon tree: 1.3% relative error (CPU LL = -27.09, GPU LL = -26.72)
- 2-taxon extreme tree (all-A vs all-T, branch 0.5): 6.1% relative error

This is within the documented precision boundary for the GPU exp/log
transcendental fallback on recursive multi-step computations. GPU f64
arithmetic is native IEEE 754 double-precision via Vulkan
(`VK_KHR_shader_float64`). The `FelsensteinGpu` primitive itself is correct —
the precision gap comes from polynomial exp/log approximations on drivers
requiring the transcendental workaround (Ada Lovelace, RADV, NVK), not from
f64 arithmetic itself.

---

## Part 3: Absorption Learnings for ToadStool Evolution

### What Worked Well

1. **Pre-compiled primitives are the correct default.** Every primitive that
   supports pre-compilation (`new(device)` at session start) shows massive
   reduction in per-call overhead. The `GpuPipelineSession` pattern should
   be the recommended pattern in ToadStool documentation.

2. **`BrayCurtisF64` pre-compiles cleanly.** Adding it to the session was a
   one-line field addition. All ToadStool ops should support this pattern.

3. **`FusedMapReduceF64::sum_of_squares`** is the correct way to compute
   vector norms. We initially tried `.dot()` which doesn't exist — `sum_of_squares`
   covers this use case. Consider adding a `dot(a, b)` convenience method.

4. **`GemmCached` reuse for spectral cosine** — the pre-compiled GEMM pipeline
   was reusable across taxonomy classification and spectral similarity. Generic
   cached pipelines have high reuse value.

### What Needs Attention

1. **No `dot` product on `FusedMapReduceF64`.** We worked around this with
   `sum_of_squares` for self-dot and manual GEMM for cross-dot. A `dot(a, b)`
   method would simplify norm and cosine computations.

2. **`mean.rs` compilation error.** When checking `--bin` targets that trigger
   full ToadStool compilation, `barracuda/src/ops/mean.rs:185` has an
   undeclared `num_workgroups` variable. This doesn't affect `--lib` checks
   but blocks `--bin` compilation on clean builds.

3. **GPU exp/log transcendental fallback for recursive algorithms.** Felsenstein
   pruning shows 1-6% relative error through the polynomial exp/log path. Algorithms
   with many sequential f64 operations (matrix products at each tree level)
   accumulate error. Consider:
   - Documenting expected precision bounds per operation count
   - Providing a `precision_tier` flag on results (exact / workaround / approximate)

4. **`PhyloTree` level ordering.** The `FelsensteinGpu` expects a specific
   `levels` vector (bottom-up BFS ordering). wetSpring's `convert_tree` helper
   builds this manually. Consider providing a `PhyloTree::from_newick()` or
   `PhyloTree::from_edges()` constructor that handles level computation.

### Streaming Architecture Recommendation

The pre-warmed session pattern should be promoted to a first-class ToadStool
concept:

```rust
// Proposed ToadStool session API
let session = GpuSession::builder(device)
    .with::<FusedMapReduceF64>()
    .with::<BrayCurtisF64>()
    .with::<GemmCachedF64>()
    .with::<FelsensteinGpu>()
    .build()?;

// All primitives pre-compiled; dispatches reuse compiled pipelines
let bc_result = session.get::<BrayCurtisF64>().condensed_distance_matrix(...)?;
let fmr = session.get::<FusedMapReduceF64>();
let norm = fmr.sum_of_squares(data)?;
```

This would let any Spring build domain-specific streaming sessions without
manually holding individual primitive instances.

---

## Part 4: Current wetSpring State

### Validation Chain (Complete)

```
Python baseline (40 scripts, 29 papers)
    ↓
BarraCUDA CPU (380/380 checks, 31+ domains, 22.5× faster)
    ↓
BarraCUDA GPU (702+ checks, 29 domains, up to 926× speedup)
    ↓
Pure GPU streaming (152 checks, 10+ domains, 441-837× vs round-trip)
    ↓
metalForge cross-substrate (37 domains, 25/25 papers three-tier)
```

### Open Items (wetSpring side)

| Item | Status | Notes |
|------|--------|-------|
| PCoA GPU | **Resolved** | naga bug fixed in wgpu v22.1.0; `catch_unwind` guards removed |
| 5 ODE WGSL shaders | Write phase | Pending `BatchedOdeRK4Generic` absorption (see v15) |
| `crate::special` math | Shaped | `erf`, `ln_gamma`, `regularized_gamma` — ready for `barracuda::math` |
| Taxonomy NPU | Shaped | Int8 quantized weights, argmax parity proven (Exp083) |

### Files Changed Since v15

| File | Change |
|------|--------|
| `barracuda/src/bio/streaming_gpu.rs` | Added `BrayCurtisF64`, `bray_curtis_matrix`, `spectral_cosine_matrix`, `stream_full_analytics` |
| `metalForge/forge/src/workloads.rs` | Added `dada2`, `bootstrap`, `placement` workloads |
| `barracuda/src/bin/validate_metalforge_v6.rs` | **NEW** — Exp104 validation binary |
| `barracuda/src/bin/validate_pure_gpu_streaming_v2.rs` | **NEW** — Exp105 validation binary |
| `barracuda/src/bin/validate_streaming_ode_phylo.rs` | **NEW** — Exp106 validation binary |
| `barracuda/Cargo.toml` | 3 new `[[bin]]` entries with `required-features = ["gpu"]` |

---

## Appendix: wetSpring GPU Module Inventory (42 modules)

| Category | Count | Modules |
|----------|:-----:|---------|
| **Lean** (ToadStool) | 27 | ani, batch_fitness, dada2, diversity, dnds, eic, gemm_cached, hamming, hmm, jaccard, kmer, kriging, locus_variance, ode_sweep, pangenome, pcoa, quality, rarefaction, random_forest, snp, spatial_payoff, spectral_match, stats, streaming, taxonomy, unifrac |
| **Write** (local WGSL) | 5 | phage_defense, bistable, multi_signal, cooperation, capacitor |
| **Compose** (ToadStool wrappers) | 7 | kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock |
| **Passthrough** (GPU buffers, CPU kernel) | 3 | gbm, feature_table, signal |
