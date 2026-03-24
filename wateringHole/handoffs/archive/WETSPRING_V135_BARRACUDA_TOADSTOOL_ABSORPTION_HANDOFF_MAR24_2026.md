<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V135 — barraCuda / toadStool Absorption Handoff

| Field | Value |
|-------|--------|
| **Date** | 2026-03-24 |
| **From** | wetSpring **V135** |
| **To** | barraCuda + toadStool teams |
| **License** | AGPL-3.0-or-later |
| **Supersedes** | `WETSPRING_V134_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR23_2026.md` (archived) |

---

## Executive Summary

V135 is a documentation reconciliation and ecosystem sync release. All root docs,
whitePaper, baseCamp, experiments, barracuda docs, metalForge docs, and specs
have been updated to canonical V135 metrics. A V135 handoff is crafted for
central wateringHole, and ecoPrimals baseCamp reflects cross-ecosystem learnings.
No code changes — V134's codebase is the active implementation.

V134 (preceding) was a deep audit and debt-resolution release: drug NMF
delegated to `barracuda::linalg::nmf`, 26 clippy errors resolved, validation
harness refactored into domain submodules (`sink`, `harness`, `or_exit`,
`data_dir`, `timing`, `domain`), primal discovery extended to 7 ecosystem
primals (coralReef, toadStool, petalTongue, rhizoCrypt, loamSpine, sweetGrass,
Squirrel), SPDX headers on all specs/whitePaper markdown, CI expanded with
feature-matrix tests (`json`, `ipc`, `vault` + all-features).

---

## Canonical V135 Metrics

| Metric | Value |
|--------|-------|
| Tests | **1,891** (unit + integration + property + doc, 0 failed) |
| Binaries | **355** (333 barracuda + 22 forge) |
| Experiments | 379 indexed (376 completed + 3 PROPOSED) |
| Validation checks | 5,707+ |
| GPU modules | 44 (all Lean) |
| CPU bio modules | 49 |
| Named tolerances | 234 (zero inline literals) |
| Coverage | 91.20% line / 90.30% function (gated at 90%) |
| barraCuda | v0.3.7 standalone (784+ WGSL shaders) |
| Primitives consumed | 150+ |
| Local WGSL | 0 |
| Duplicate math | 0 |
| `#[allow()]` | 0 (all `#[expect(reason)]`) |
| Unsafe | 0 (`forbid(unsafe_code)` at workspace level + per-crate) |
| Clippy | 0 warnings (pedantic + nursery, `-D warnings`) |
| C dependencies | 0 |
| MSRV | 1.87 (edition 2024) |

---

## What V134–V135 Did (Relevant to Upstream)

### V134 Code Changes

- **Drug NMF → barracuda::linalg::nmf**: Deleted 3 local CPU matmul functions
  + `nmf_mu` (Lee & Seung) from `ipc/handlers/drug.rs`. Handler now delegates
  to `barracuda::linalg::nmf::nmf()` with `NmfConfig`. File went 229 → 82 lines.
- **Validation harness refactored**: `barracuda/src/validation/mod.rs` (767 LOC)
  decomposed into 6 focused submodules: `sink.rs` (ValidationSink trait + impls),
  `harness.rs` (Validator), `or_exit.rs` (OrExit trait), `data_dir.rs`
  (capability-based data discovery), `timing.rs` (bench helpers), `domain.rs`
  (DomainResult + summary). Mod.rs is now a 161-line facade.
- **Primal discovery extended**: `discover_coralreef()`, `discover_toadstool()`,
  `discover_petaltongue()`, `discover_rhizocrypt()`, `discover_loamspine()`,
  `discover_sweetgrass()` added to `ipc/discover.rs` with unit tests.
- **`primal_names.rs`**: Added `CORALREEF` constant; enriched doc comments for
  rhizoCrypt, loamSpine, sweetGrass.
- **CI expansion**: Feature-matrix tests for `json`, `ipc`, `vault`; all-features
  test job; coverage jobs include `--lib --tests` for integration tests.
- **SPDX headers**: `<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->` added to
  34 spec/whitePaper markdown files.
- **Provenance**: SHA-256 hashes corrected in BASELINE_MANIFEST.md; download
  scripts added to manifest.
- **Workspace lints**: `unsafe_code = "forbid"` added to `[workspace.lints.rust]`
  in root Cargo.toml — workspace-level enforcement.

### V135 Documentation Changes

- Canonical metrics aligned across README, CONTEXT, whitePaper, baseCamp,
  experiments, barracuda docs, metalForge docs, and specs.
- All status lines updated from V133 to V135.
- wateringHole handoff crafted and V134 archived.
- ecoPrimals baseCamp updated with cross-ecosystem learnings.

---

## What wetSpring Needs from barraCuda (unchanged + context)

| Priority | Ask | Why | wetSpring Module |
|----------|-----|-----|------------------|
| **High** | **`BatchReconcileGpu`** | `reconciliation_gpu` is still CPU passthrough; DTL wavefront DP needs batched GPU kernel | `bio::reconciliation_gpu` |
| **High** | **DF64 GEMM pipeline** | Spectral cosine and dot/GEMM on FP32-heavy hardware; precision routing ready | `bio::spectral_match_gpu` |
| **Medium** | **CPU Jacobi eigensolve** in `barracuda::linalg` | `pcoa.rs` has local CPU Jacobi; parity story vs `BatchedEighGpu`; should share implementation | `bio::pcoa` |
| **Medium** | **`BandwidthTier` in metalForge substrate** | PCIe-aware routing for cross-system dispatch | `metalForge` bridge |
| **Medium** | **`BatchMergePairsGpu`** | `merge_pairs_gpu` overlap consensus needs dedicated kernel | `bio::merge_pairs_gpu` |
| **Low** | **`GbmBatchInferenceGpu`** | GBM GPU inference (currently CPU fallback for parity) | `bio::gbm` |
| **Low** | **GPU NJ / Robinson-Foulds** | Sequential NJ loop and full RF distance on GPU — algorithmic kernels needed | `bio::neighbor_joining`, `bio::robinson_foulds` |

### Evolution Intelligence for barraCuda

- **Validation harness decomposition pattern**: The `validation/` refactor
  (mod.rs → 6 submodules) is a reusable pattern for any large module. Domain
  cohesion over arbitrary splitting: `sink` (output routing), `harness`
  (accumulation), `or_exit` (error handling), `data_dir` (discovery),
  `timing` (benchmarks), `domain` (summary). Consider this pattern for
  barraCuda's own large modules.

- **Primal discovery pattern**: `discover_*()` functions follow a capability-based
  cascade: env var → XDG runtime dir → BIOMEOS_SOCKET_DIR → temp fallback.
  Each primal has a `PRIMAL_NAME` constant in `primal_names.rs`. This pattern
  scales to any new primal without hardcoding.

- **`#[expect(reason)]` over `#[allow()]`**: First spring to achieve zero
  `#[allow()]` codebase-wide. Unfulfilled expectations produce clippy warnings
  via `unfulfilled_lint_expectations`, catching stale suppressions automatically.

- **Workspace-level `forbid(unsafe_code)`**: Combined with per-crate `#![forbid]`,
  this catches any accidental unsafe in new code at workspace scope. Recommend
  adoption across all springs.

---

## What wetSpring Needs from toadStool

| Ask | Why |
|-----|-----|
| **Wire `compute.performance_surface.query`** (and report) | Client is ready; need live RPCs for routing hints |
| **`compute.route.multi_unit`** | Large bio pipelines need multi-unit routing with back-pressure |
| **`DeviceCapabilities::latency_model()` refinements** | Better precision/routing advice for `GpuContext` decisions |

---

## What wetSpring Contributes Back

- **Zero-duplicate-math validation** — proves the Write → Absorb → Lean cycle
  completes: 49 CPU + 44 GPU modules, all delegating to barraCuda.
- **`barracuda::linalg::nmf` consumption proof** — drug NMF handler is a clean
  consumption example of the Feb 2026 absorption.
- **Validation harness architecture** — reusable 6-module decomposition pattern
  for large validation/test infrastructure.
- **7-primal discovery** — runtime discovery for coralReef, toadStool,
  petalTongue, Squirrel, rhizoCrypt, loamSpine, sweetGrass.
- **8-axis audit methodology** — completion, quality, fidelity, dependency health,
  GPU readiness, coverage, ecosystem standards, primal coordination.
- **Coverage discipline** — 91%+ with per-crate 90% gates in CI, local alias
  aligned. Integration tests + fuzz + proptest complement llvm-cov.
- **CI feature-matrix** — tests `json`, `ipc`, `vault` features independently
  plus all-features, catching feature-gate regressions.

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

*End of V135 barraCuda / toadStool absorption handoff — 2026-03-24.*
