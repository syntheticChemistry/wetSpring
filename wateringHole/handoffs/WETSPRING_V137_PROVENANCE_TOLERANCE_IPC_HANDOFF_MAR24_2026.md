<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V137 — barraCuda / toadStool / Spring Teams Absorption Handoff

| Field | Value |
|-------|--------|
| **Date** | 2026-03-24 |
| **From** | wetSpring **V137** |
| **To** | barraCuda + toadStool + spring teams + primal teams |
| **License** | AGPL-3.0-or-later |
| **Supersedes** | `WETSPRING_V135_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR24_2026.md` (archived) |

---

## Executive Summary

V137 completes two long-running debt items and adds a structural IPC improvement:

1. **Full provenance**: `//! Provenance:` doc headers on all 355 validation/benchmark
   binaries (333 barracuda + 22 forge). Every binary now documents its experiment
   number, baseline tool, and validation scope. Was 5 binaries in V136.

2. **Tolerance centralization**: 8 new named tolerance constants eliminate all
   remaining inline numeric literals in test assertions. Total: 242 named
   constants across 6 tolerance modules with full provenance documentation.

3. **IPC modularization**: Connection processing pipeline (`ipc/connection.rs`)
   extracted from `ipc/server.rs`, separating protocol handling (read, dispatch,
   write) from server lifecycle (bind, run, drop). Enables independent testing
   of the JSON-RPC pipeline.

V136 (preceding) was the deep debt + ecosystem absorption release: thiserror
migration, named cast helpers (~60 bare casts replaced), upstream contract
pinning for barraCuda v0.3.7, bitwise determinism tests, CI version pin,
hardcoded primal string elimination, CONTRIBUTING.md + SECURITY.md.

---

## Canonical V137 Metrics

| Metric | Value |
|--------|-------|
| Tests | **1,902** (unit + integration + property + doc, 0 failed) |
| Binaries | **355** (333 barracuda + 22 forge) |
| Experiments | 379 indexed (376 completed + 3 PROPOSED) |
| Validation checks | 5,707+ |
| GPU modules | 44 (all Lean) |
| CPU bio modules | 49 |
| Named tolerances | **242** (zero inline literals) |
| Provenance headers | **355/355** binaries |
| Coverage | 91.20% line / 90.30% function (gated at 90%) |
| barraCuda | v0.3.7 standalone (784+ WGSL shaders) |
| Primitives consumed | 150+ |
| Local WGSL | 0 |
| Duplicate math | 0 |
| `#[allow()]` | 0 (all `#[expect(reason)]`) |
| Unsafe | 0 (`forbid(unsafe_code)` at workspace level + per-crate) |
| C dependencies | 0 |
| MSRV | 1.87 (edition 2024) |

---

## What V137 Did (Relevant to Upstream)

### Provenance Headers (all 355 binaries)

Every `src/bin/*.rs` file now starts with a `//! Provenance:` doc comment that
traces the binary to its experiment, baseline tool, data source, and validation
scope. Format:

```rust
//! Provenance: Exp042 — validates Gonzales PK study S79
//! against published oral dose pharmacokinetic model (Cmax, elimination).
```

**Pattern for all springs**: This establishes a convention where every validation
binary is self-documenting. A grep for `//! Provenance:` in any spring gives a
complete audit trail. Recommend adoption across all springs.

### New Named Tolerance Constants (8 added, 242 total)

| Constant | Value | Module | Purpose |
|----------|-------|--------|---------|
| `DF64_SMALL_VALUE_ROUNDTRIP` | 1e-25 | `tolerances` | DF64 double-double roundtrip precision |
| `STABLE_IDENTITY_TINY` | 1e-25 | `tolerances` | Stable numeric identity operations |
| `ESN_PREDICTION_REASONABLE` | 10.0 | `tolerances::bio::esn` | ESN reservoir prediction magnitude |
| `PEAK_INDEX_PROXIMITY` | 2.0 | `tolerances::bio::misc` | Signal peak index detection proximity |
| `NPU_QUANTIZE_ROUNDTRIP` | 0.1 | `tolerances::bio::misc` | NPU int8 quantize-dequantize roundtrip |
| `ABUNDANCE_SUM_TO_ONE` | 0.05 | `tolerances::bio::misc` | Community abundance normalization |
| `PK_PEAK_DOSE_RELATIVE` | 0.2 | `tolerances::bio::misc` | Pharmacokinetic peak-to-dose ratio |
| `PK_ELIMINATION_FLOOR` | 0.15 | `tolerances::bio::misc` | PK elimination curve floor |

**Pattern for all springs**: Inline tolerance literals are a recurring debt
pattern. wetSpring's approach — named constants with doc comments explaining
the physical/mathematical justification — prevents silent tolerance drift.

### IPC Connection Extraction

`ipc/server.rs` → `ipc/connection.rs` + `ipc/server.rs`:

- **`connection.rs`**: `handle_connection()`, `process_single()`, `process_batch()`,
  `dispatch_request()`, `dispatch_notification()`, `execute_method()`. Pure
  protocol pipeline — reads JSON-RPC, dispatches, writes responses.
- **`server.rs`**: `Server::new()`, `Server::run()`, `Drop for Server`. Lifecycle
  management — bind socket, accept connections, cleanup.

This separation makes the JSON-RPC protocol testable without socket lifecycle
and keeps both files focused and under 250 LOC.

---

## What wetSpring Needs from barraCuda (unchanged + context)

| Priority | Ask | Why | wetSpring Module |
|----------|-----|-----|------------------|
| **High** | **`BatchReconcileGpu`** | DTL wavefront DP needs batched GPU kernel | `bio::reconciliation_gpu` |
| **High** | **DF64 GEMM pipeline** | Spectral cosine on FP32-heavy hardware | `bio::spectral_match_gpu` |
| **Medium** | **CPU Jacobi eigensolve** in `barracuda::linalg` | Local CPU Jacobi in pcoa.rs; should share | `bio::pcoa` |
| **Medium** | **`BandwidthTier` in metalForge substrate** | PCIe-aware routing | `metalForge` bridge |
| **Medium** | **`BatchMergePairsGpu`** | Overlap consensus needs dedicated kernel | `bio::merge_pairs_gpu` |
| **Low** | **`GbmBatchInferenceGpu`** | GBM GPU inference (currently CPU fallback) | `bio::gbm` |
| **Low** | **GPU NJ / Robinson-Foulds** | Sequential loop and full RF distance on GPU | `bio::neighbor_joining`, `bio::robinson_foulds` |

## What wetSpring Needs from toadStool

| Ask | Why |
|-----|-----|
| **Wire `compute.performance_surface.query`** | Client ready; need live RPCs |
| **`compute.route.multi_unit`** | Large bio pipelines need multi-unit routing |
| **`DeviceCapabilities::latency_model()` refinements** | Better precision/routing advice |

---

## Evolution Intelligence for All Teams

### For barraCuda Team

- **Provenance header pattern**: `//! Provenance:` on every binary. Self-documenting
  audit trail. Grep-friendly. Recommend for all barraCuda binaries/examples.
- **Tolerance naming convention**: `MODULE_DOMAIN_DESCRIPTION` pattern with doc
  comments explaining the physical basis. 242 constants, zero inline literals.
- **IPC connection/server separation**: Protocol pipeline testable without socket
  lifecycle. Consider for barraCuda's own IPC if/when it adds server capability.

### For Spring Teams (hotSpring, neuralSpring, airSpring, groundSpring, healthSpring, ludoSpring)

- **Full binary provenance**: wetSpring now has `//! Provenance:` on 355/355 binaries.
  Recommend adoption. The pattern: first doc line documents experiment number,
  baseline tool, validation scope. `clippy::doc_lazy_continuation` requires a
  blank `//!` line before `//! Provenance:` if preceded by markdown content.
- **Named tolerance centralization**: All inline numeric literals in test assertions
  should be named constants with documented justification. wetSpring's 242 constants
  across 6 tolerance modules (`tolerances/mod.rs`, `tolerances/bio/esn.rs`,
  `tolerances/bio/misc.rs`, etc.) is the reference implementation.
- **IPC modular decomposition**: If your spring has IPC server code, consider
  separating connection handling from server lifecycle. The pattern cleanly
  divides protocol concerns from OS resource management.

### For Primal Teams (biomeOS, petalTongue, Squirrel, BearDog, Songbird)

- **wetSpring IPC surface**: 23 JSON-RPC methods, 8 MCP tools. All methods use
  `primal_names::*` constants for identity — zero hardcoded strings.
- **Discovery cascade**: `discover_*()` functions follow env var → XDG → BIOMEOS →
  temp fallback. 7 primal discovery functions (coralReef, toadStool, petalTongue,
  rhizoCrypt, loamSpine, sweetGrass, Squirrel).
- **Capability registry**: `capability_registry.toml` declares 20 capabilities
  across 5 domains (ecology, provenance, brain, metrics, ai), 10 GPU-accelerated.
- **Connection resilience**: `ipc/resilience.rs` implements `RetryPolicy` +
  `CircuitBreaker`. The connection module extracted in V137 makes this pipeline
  more testable.

---

## What wetSpring Contributes Back

- **Complete provenance discipline** — first spring with 100% binary provenance
  headers. The `//! Provenance:` convention is now ecosystem-ready.
- **Zero inline tolerance literals** — 242 named constants with documented
  physical/mathematical justification. Pattern for tolerance management.
- **Modular IPC architecture** — connection pipeline separated from server lifecycle,
  enabling independent protocol testing.
- **8-axis audit methodology** — completion, quality, fidelity, dependency health,
  GPU readiness, coverage, ecosystem standards, primal coordination.
- **Zero-duplicate-math validation** — proves Write → Absorb → Lean completes.
- **Determinism testing pattern** — seeded stochastic algorithms (gillespie,
  bootstrap, rarefaction) verified bit-identical across runs.

---

## GPU Evolution Tier Map (current)

| Tier | Modules | Blocker |
|------|---------|---------|
| **Lean** (upstream op) | 22 modules | — |
| **Compose** (multi-op) | 11 modules | — |
| **Write → Lean** (ODE) | 5 modules (deleted local WGSL) | — |
| **Tier A** (rewire) | gemm_cached, taxonomy_gpu, stats paths | Buffer tuning |
| **Tier B** (adapt) | DF64 routing, precision workarounds | Host protocol changes |
| **Tier C** (new shader) | reconciliation, merge_pairs, gbm, NJ, RF | barraCuda kernel work |

---

## API Stability (wetSpring → barraCuda)

Active consumption surfaces (breaking changes should be staged with semver):

- **`barracuda::device`**: `WgpuDevice`, `TensorContext`, `DeviceCapabilities`, `PrecisionRoutingAdvice`
- **`barracuda::session`**: `TensorSession::with_device(Arc<WgpuDevice>)`
- **`barracuda::ops`**: `GemmF64`, `FusedMapReduceF64`, `BatchedEighGpu`, ODE systems, bio ops
- **`barracuda::linalg::nmf`**: `NmfConfig`, `NmfResult`, `nmf()`, `NmfObjective`, `cosine_similarity`
- **`barracuda::stats`**, **`barracuda::special`**, **`barracuda::spectral`**, **`barracuda::numerical`**
- **`barracuda::shaders::Precision`** — `GpuF64::optimal_precision()`, DF64 paths

---

*End of V137 barraCuda / toadStool / spring teams absorption handoff — 2026-03-24.*
