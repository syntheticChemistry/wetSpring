# wetSpring V127 Handoff — barraCuda/toadStool Evolution

**Date:** 2026-03-18
**From:** wetSpring V127
**To:** barraCuda team, toadStool team
**Supersedes:** WETSPRING_V126_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_MAR16_2026.md

---

## Executive Summary

wetSpring V127 is **fully lean** — zero local WGSL, zero local math duplication,
all 44 GPU modules consume upstream barraCuda primitives. This handoff identifies
the remaining evolution candidates and ecosystem learnings for the upstream teams.

---

## 1. Absorption Candidate: `monod()` Function

**12 validation binaries** define a local `fn monod(s, mu_max, ks) -> f64` implementing
Monod substrate-limited growth kinetics:

```rust
fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}
```

**barraCuda has `hill()` but not `monod()`**. The Monod equation is mathematically
a special case of the Hill function with `n = 1`, but the naming and parameterization
differ in the biology literature. Adding `barracuda::stats::monod` would:
- Remove 12 local definitions across validation binaries
- Provide a canonical, tested, documented implementation
- Support anaerobic digestion (Track 6), soil QS (Track 4), and general microbiology

**Affected binaries:**
`validate_petaltongue_biogas_v1`, `validate_metalforge_v19`, `validate_cpu_vs_gpu_v11`,
`validate_toadstool_dispatch_v4`, `validate_pure_gpu_streaming_v13`,
`validate_metalforge_v18`, `validate_cpu_vs_gpu_v10`, `validate_barracuda_cpu_v27`,
`validate_paper_math_control_v6`, `validate_barracuda_cpu_v26`,
`benchmark_python_vs_rust_v5`, `validate_fungal_fermentation_digestate`

**Also useful:** `haldane(s, mu_max, ks, ki)` (Monod with substrate inhibition) —
used in 6 of the same binaries.

---

## 2. Forge Modules — Absorption Seam

The `metalForge/forge/` crate (252 tests) contains dispatch and hardware discovery
logic that overlaps with toadStool's role:

| Module | LOC | Tests | Absorption Target |
|--------|-----|-------|-------------------|
| `bridge.rs` | ~200 | 47 | toadStool device integration layer |
| `dispatch.rs` | ~300 | 47 | toadStool workload routing |
| `probe.rs` | ~150 | 20 | toadStool hardware discovery |
| `inventory.rs` | ~200 | 15 | toadStool substrate registry |
| `streaming.rs` | ~200 | 10 | toadStool pipeline analysis |

**Absorption strategy:** When toadStool absorbs forge, `bridge.rs` becomes the
integration point. The `dispatch.rs` routing logic (GPU→NPU→CPU priority,
bandwidth-aware fallback, PCIe cost model) represents validated patterns that
toadStool can adopt.

---

## 3. Unwired Upstream Primitives

Available in barraCuda but not yet consumed by wetSpring:

| Primitive | Session | Use Case | Blocker |
|-----------|---------|----------|---------|
| `SparseGemmF64` | S60 | Track 3 drug repurposing NMF | Track 3 experiments paused |
| `TranseScoreF64` | S60 | Track 3 KG embedding scoring | Track 3 experiments paused |
| `TopK` | S60 | Track 3 drug–disease ranking | Track 3 experiments paused |
| `BandwidthTier` | S62 | metalForge PCIe-aware dispatch | Forge not yet bandwidth-routed |
| `ComputeDispatch` | S65 | Reduce BGL/pipeline boilerplate | All GPU code already works |
| `LogsumexpWgsl` | Early | HMM forward numerical stability | CPU `log_sum_exp` sufficient |

**Recommendation:** Wire `SparseGemmF64`, `TranseScoreF64`, and `TopK` when
Track 3 experiments resume. These are ready — no upstream work needed.

---

## 4. Math Delegation Completed

All local math now delegates to barraCuda:

| Local Function | Upstream Target | Status |
|----------------|-----------------|--------|
| `kahan_sum(values)` | `barracuda::shaders::precision::cpu::kahan_sum` | **Delegated** (V127) |
| `erf(x)` | `barracuda::special::erf` | Delegated |
| `norm_cdf(x)` | `barracuda::stats::norm_cdf` | Delegated |
| `pearson_correlation(x, y)` | `barracuda::stats::pearson_correlation` | Delegated |
| `hill(x, k, n)` | `barracuda::stats::hill` | Delegated |
| `shannon(abundances)` | `barracuda::stats::diversity::shannon` | Delegated |
| `nmf(matrix, config)` | `barracuda::linalg::nmf` | Re-exported |

**Zero local math duplication** in library code. The only remaining local functions
are the `monod()`/`haldane()` kinetics in validation binaries (see §1).

---

## 5. Ecosystem Learnings for Upstream

### 5.1 Tolerance Architecture

wetSpring's 214+ named tolerance constants (`tolerances/` module) with scientific
justification proved essential for reproducibility. **Recommendation for barraCuda:**
consider a similar centralized tolerance registry for GPU precision thresholds
(DF64 roundtrip, GPU vs CPU parity, shader compilation timeouts).

### 5.2 Provenance Registry

The new `provenance.rs` Python baseline registry (`PythonBaseline` struct with
`binary`, `script`, `commit`, `date`, `category`) provides structured traceability.
**Recommendation for toadStool:** session-level provenance tracking (which shaders
compiled, which precision tier selected, which device dispatched) using a similar
structured approach.

### 5.3 MCP Tool Definitions

wetSpring's MCP module provides typed tool schemas for AI integration. Each tool
maps to an existing JSON-RPC method via `tool_to_method()`. **Recommendation for
barraCuda/toadStool:** if Squirrel or other AI primals need to discover compute
capabilities, an MCP layer on toadStool's dispatch API would be valuable.

### 5.4 IPC Resilience Patterns

`RetryPolicy` (exponential backoff) and `CircuitBreaker` (failure threshold,
half-open probing) from sweetGrass proved effective for toadStool dispatch.
**Already adopted in wetSpring V127** — recommend all springs adopt the same pattern.

### 5.5 `DispatchOutcome<T>` Pattern

The three-variant outcome enum (Success/Protocol/Application) cleanly separates
retriable transport failures from deterministic application rejections. This pattern
is now standard across groundSpring, airSpring, sweetGrass, and wetSpring.

---

## 6. barraCuda Consumption Summary

| Category | Count | Examples |
|----------|-------|---------|
| **stats** | 20+ functions | `norm_cdf`, `pearson_correlation`, `hill`, `shannon`, `bootstrap_ci`, `spearman_correlation`, `fao56_et0` |
| **spectral** | 8 functions | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `GOE_R`, `POISSON_R` |
| **special** | 6 functions | `erf`, `erfc_f64`, `bessel_j0_minus1_f64` |
| **linalg** | 5 functions | `nmf`, `cosine_similarity`, `graph_laplacian`, `effective_rank`, `CsrMatrix` |
| **ops** | 12 GPU ops | `GemmF64`, `FusedMapReduceF64`, `PeakDetectF64`, `GillespieGpu`, `WrightFisherGpu`, etc. |
| **device** | 8 types | `WgpuDevice`, `BufferPool`, `TensorContext`, `PrecisionBrain` |
| **shaders** | 3 paths | `Precision`, `provenance::*`, `precision::cpu::kahan_sum` |
| **numerical** | 4 functions | `BatchedOdeRK4`, `trapz`, `gradient_1d`, `numerical_hessian` |
| **Total consumed** | **150+ primitives** | |

---

## 7. Version & Quality

| Metric | Value |
|--------|-------|
| barraCuda version | v0.3.5 (path dependency) |
| Local WGSL | **0** |
| Local math duplication | **0** (library), 12 `monod()` in binaries |
| GPU modules | 44 (27 lean + 7 compose + 5 ODE lean + 5 viz) |
| Tests | 1,448+ pass |
| Clippy warnings | 0 (pedantic + nursery) |
| `#![forbid(unsafe_code)]` | All crate roots |
