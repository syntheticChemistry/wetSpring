# wetSpring → ToadStool/BarraCuda V81 Handoff

**Date:** February 28, 2026
**From:** wetSpring V81 (Phase 81)
**To:** ToadStool/BarraCuda team
**Status:** 247 experiments, 6,273+ checks, 1,219 tests, ALL PASS
**Supersedes:** V76 (archived)
**ToadStool Pin:** S68+ (`e96576ee`)
**License:** AGPL-3.0-only

---

## Executive Summary

V81 completes the full evolution chain validation: CPU↔GPU parity across 22
domains, ToadStool streaming dispatch overhead proof, PCIe bypass mixed hardware
(NPU→GPU direct transfer, bandwidth-aware routing), and NUCLEUS Tower→Node→Nest
extended pipeline with 49+ workloads and biomeOS sovereign fallback.

**Key outcomes for absorption:**
- **22-domain CPU↔GPU parity** — same pure Rust equations produce identical
  results on CPU and GPU (Exp243). Math is truly portable.
- **ToadStool GpuPipelineSession** validated end-to-end: pre-warmed streaming
  eliminates CPU round-trips for diversity, Bray-Curtis, taxonomy, full analytics.
  Determinism proven (3 runs bit-identical). CPU reference parity confirmed (Exp244).
- **PCIe bypass topology** — GPU→GPU chains (0 round-trips), GPU→NPU direct
  (`accepts_gpu_buffer: true`), bandwidth-aware CPU fallback. 6-stage pipeline:
  5 PCIe transitions saved (Exp245).
- **NUCLEUS v2** — 49+ workloads registered in metalForge catalog, all
  ToadStool-absorbed. Tower discovery + Nest storage + Node dispatch validated
  for 8 new workloads (chimera, DADA2, GBM, reconciliation, clock, bootstrap,
  placement, assembly). biomeOS Songbird+NestGate sovereign fallback (Exp246).

---

## Part 1: What We Built (Exp239–246)

### Phase 80 (Exp239–242): Extended Domain Coverage

| Exp | Binary | Checks | What It Proves |
|-----|--------|:------:|----------------|
| 239 | `validate_barracuda_cpu_v17` | 29/29 | 8 new CPU-only domains (chimera, DADA2, alignment, ESN, GBM, DTL, clock, RF) |
| 240 | `validate_barracuda_gpu_v9` | 24/24 | 8 new GPU workloads (chimera, DADA2, GBM, DTL, clock, RF, rarefaction, kriging) |
| 241 | `validate_pure_gpu_streaming_v7` | 18/18 | 6-stage ToadStool pipeline (DADA2→Chimera→Diversity→Rarefaction→Kriging→DTL) |
| 242 | `validate_metalforge_v11_extended` | 43/43 | 23-workload cross-system catalog (16 GPU + 3 NPU + 4 CPU) |

### Phase 81 (Exp243–246): Parity + Dispatch + PCIe + NUCLEUS

| Exp | Binary | Checks | What It Proves |
|-----|--------|:------:|----------------|
| 243 | `validate_cpu_vs_gpu_v6_extended` | 24/24 | 22 domains CPU↔GPU head-to-head with timing |
| 244 | `validate_toadstool_dispatch_v2` | 22/22 | GpuPipelineSession streaming overhead, full analytics pipeline |
| 245 | `validate_pcie_bypass_mixed_hw` | 36/36 | PCIe tier detection, GPU→NPU bypass, bandwidth-aware routing |
| 246 | `validate_nucleus_v2_extended` | 62/62 | 49+ workloads, Tower+Nest+Node, biomeOS coordination |

---

## Part 2: BarraCuda Usage and Evolution

### How wetSpring Uses BarraCuda

wetSpring consumes 82 ToadStool primitives through barracuda. Zero local WGSL,
zero fallback code. The consumption pattern:

| Pattern | Count | Example |
|---------|:-----:|---------|
| **Lean** (delegate to ToadStool) | 27 | `diversity_gpu` → `FusedMapReduceF64` |
| **Compose** (wire primitives) | 7 | `reconciliation_gpu` → `BatchReconcileGpu` |
| **Write→Lean** (ODE trait-generated) | 5 | `bistable_gpu` → `BatchedOdeRK4<BistableOde>` |
| **NPU** (int8 quantization) | 1 | `esn` → AKD1000 via reservoir int8 |

### Primitives Consumed (82)

All via `compile_shader_universal(source, Precision::F64)`. Key primitives:

- `FusedMapReduceF64` — diversity, Shannon, Simpson, observed features
- `BrayCurtisF64` — pairwise distance matrix
- `BatchedEighGpu` — PCoA eigendecomposition
- `GemmCachedF64` — taxonomy, chimera scoring, spectral cosine
- `SmithWatermanGpu` — alignment
- `GillespieGpu` — stochastic simulation
- `TreeInferenceGpu` — RF, GBM, decision tree
- `FelsensteinGpu` — phylogenetic likelihood
- `Dada2EStepGpu` — ASV denoising
- `BatchedOdeRK4<S>::generate_shader()` — 5 biological ODE systems
- `ComputeDispatch` builder — 6 GPU modules refactored to use this pattern
- `BatchedMultinomialGpu` — rarefaction bootstrap
- `PairwiseL2Gpu` — distance computation

### BarraCuda Evolution Across Phases

| Phase | Version | Domains | Key Addition |
|-------|---------|:-------:|--------------|
| 1–6 | CPU v1–v3 | 18 | Foundation: diversity, ODE, taxonomy |
| 10 | CPU v4 | 23 | Track 1c: ANI, SNP, dN/dS, clock, pangenome |
| 12 | CPU v5 | 25 | 25-domain benchmark (22.5× vs Python) |
| 13 | CPU v6 | 31 | ML ensembles: RF, GBM, decision tree |
| 28 | CPU v8 | 31+ | 13 newly promoted GPU domains |
| 62 | CPU v11 | 31+ | IPC dispatch math fidelity (EXACT_F64) |
| 66 | CPU v12 | 31+ | Post-audit I/O evolution |
| 67 | CPU v13 | 47 | 47-domain pure Rust math proof |
| 80 | CPU v17 | 19 new | +chimera, +DADA2, +alignment, +ESN, +GBM, +DTL, +clock, +RF |
| 81 | GPU v6 | 22 h2h | 22-domain CPU↔GPU head-to-head parity proven |

### Performance Summary

| Workload | Speedup | Notes |
|----------|---------|-------|
| Rust vs Python (25 domains) | **33.4×** overall, 625× peak | Exp059 |
| Spectral cosine (2048) | **926×** GPU vs CPU | Exp087 |
| Streaming vs round-trip | 92-94% overhead eliminated | Exp091 |
| ToadStool pre-warmed session | <5ms warmup, bit-identical | Exp244 |

---

## Part 3: What ToadStool Should Absorb

### Already Absorbed (82 primitives, complete)

wetSpring has ZERO local WGSL, ZERO local derivative/regression math. Everything
routes through ToadStool. The Write→Absorb→Lean cycle is fully complete.

### Learnings for ToadStool Evolution

1. **GpuPipelineSession is the right abstraction.** Pre-warmed sessions with
   compiled pipelines eliminate cold-start latency. Our Exp244 proves streaming
   vs individual dispatch overhead reduction. ToadStool should continue investing
   in this pattern.

2. **PCIe bypass matters.** GPU→NPU direct transfer (via `accepts_gpu_buffer: true`)
   eliminates CPU round-trips in mixed pipelines. Our Exp245 shows a 6-stage
   pipeline saving 5 PCIe transitions. ToadStool's `StreamingSession` model
   correctly predicts streamability.

3. **Bandwidth-aware routing prevents perf regressions.** When PCIe transfer cost
   exceeds GPU dispatch overhead, CPU fallback is correct. `route_bandwidth_aware()`
   in metalForge dispatch uses `BandwidthTier::detect_from_adapter_name()` and
   `GPU_DISPATCH_OVERHEAD_US` from barracuda. This pattern should be first-class
   in ToadStool's dispatch layer.

4. **DF64 round-trip tolerance is 1e-13.** Double-double precision pack/unpack
   has inherent precision limits. We fixed multiple experiments that incorrectly
   used `tolerances::EXACT` for DF64 values. ToadStool's `df64_host::pack` returns
   `[f32; 2]` (not a tuple) — document this clearly.

5. **NUCLEUS Tower→Nest→Node is the right pipeline model.** Tower discovers
   substrates (local + Songbird mesh), Nest stores artifacts (NestGate JSON-RPC),
   Node dispatches computation. wetSpring's Exp246 validates this across 49+
   workloads. Sovereign fallback works correctly when biomeOS services are offline.

6. **49+ workloads all `ShaderOrigin::Absorbed`.** Every workload in the metalForge
   catalog routes through ToadStool primitives. No local WGSL. This validates
   that ToadStool's primitive library covers the full life science compute surface.

---

## Part 4: Validation Confidence

### Test Infrastructure

| Metric | Value |
|--------|-------|
| Experiments | 247 (all PASS) |
| Validation checks | 6,273+ |
| Rust tests | 1,219 (962 barracuda + 175 forge + 60 integration + 22 doc) |
| GPU checks | 1,945+ on RTX 4070 |
| NPU checks | 60 on AKD1000 |
| Line coverage | 95.86% |
| Named tolerances | 97 (with scientific provenance) |
| Clippy warnings | 0 (pedantic, both crates) |
| Unsafe code | 0 |
| Local WGSL | 0 |
| Papers reproduced | 52/52 |
| Three-tier papers | 50/50 |

### Recent Experiment Chain (Exp233–246)

All 14 experiments in the V80–V81 evolution chain pass:

```
Exp233 (Paper math)     → Exp234 (CPU v16)     → Exp235 (GPU v8)
  → Exp236 (Streaming v6) → Exp237 (metalForge v10) → Exp238 (IPC v2)
  → Exp239 (CPU v17)     → Exp240 (GPU v9)     → Exp241 (Streaming v7)
  → Exp242 (metalForge v11) → Exp243 (CPU↔GPU parity)
  → Exp244 (ToadStool dispatch) → Exp245 (PCIe bypass)
  → Exp246 (NUCLEUS v2)
```

---

## Part 5: Action Items for ToadStool Team

| Priority | Item | Details |
|----------|------|---------|
| **Info** | CPU↔GPU parity proven for 22 domains | No action needed — validates ToadStool's math portability |
| **Info** | GpuPipelineSession streaming validated | Pre-warmed + deterministic — ToadStool's design is correct |
| **Info** | PCIe bypass model validated | `accepts_gpu_buffer` pattern works for GPU→NPU chains |
| **Consider** | Document `df64_host::pack` return type | Returns `[f32; 2]`, not tuple — caused multiple confusion points |
| **Consider** | `BandwidthTier` in ToadStool dispatch | Currently in metalForge bridge; could be first-class in barracuda |
| **Consider** | `route_bandwidth_aware` pattern | PCIe cost vs dispatch overhead comparison for automatic fallback |

---

*This handoff is unidirectional: wetSpring → ToadStool. ToadStool absorbs what
it finds useful; wetSpring leans on upstream. No response expected.*
