# wetSpring V134 — barraCuda / toadStool Absorption Handoff

| Field | Value |
|-------|--------|
| **Date** | 2026-03-23 |
| **From** | wetSpring **V134** |
| **To** | barraCuda + toadStool teams |
| **License** | AGPL-3.0-or-later |
| **Supersedes** | `WETSPRING_V133_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR23_2026.md` (archived) |

---

## Executive Summary

V134 is a deep audit and debt-resolution release. Full ecosystem audit against
wateringHole standards (PURE_RUST_SOVEREIGN_STACK_GUIDANCE, ECOBIN_ARCHITECTURE,
SCYBORG_PROVENANCE_TRIO, PRIMAL_REGISTRY) confirmed compliance on all axes.
Remaining debt items resolved: drug NMF handler migrated from 112 lines of
local matmul/NMF to `barracuda::linalg::nmf::nmf()`, 26 clippy errors fixed,
stale lint suppressions cleaned, coverage-check threshold aligned to CI 90%
gate. **Zero duplicate math remains in the codebase.**

---

## What V134 Did (Relevant to Upstream)

- **Drug NMF → barracuda::linalg::nmf**: Deleted 3 local CPU matmul functions
  + `nmf_mu` (Lee & Seung) from `ipc/handlers/drug.rs`. Handler now delegates
  to `barracuda::linalg::nmf::nmf()` with `NmfConfig`. File went 229 → 82 lines.
  The Feb 2026 absorption into barraCuda is now the single implementation.
- **Clippy pedantic+nursery zero**: 26 errors resolved across 12 files — no
  `#[allow()]` anywhere, all suppressions use `#[expect(reason)]`.
- **Coverage**: 91.20% line / 90.30% function / 91.03% region (llvm-cov),
  gated at 90% in both CI and local alias.
- **Version drift**: stale v0.3.5 references in specs/README.md and
  EVOLUTION_READINESS.md corrected to v0.3.7.
- **Doc generation**: zero rustdoc warnings with `-D warnings`.

---

## Verified Quality State

| Check | Result |
|-------|--------|
| `cargo fmt` | Clean |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | **0 errors** |
| `cargo test --workspace` | **1,530 unit + 27 doc, 0 failed** |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | **0 warnings** |
| `cargo llvm-cov --workspace --lib` | **91.20% line** |
| `cargo check --target x86_64-unknown-linux-musl` | **PASS** |
| `#![forbid(unsafe_code)]` | All crate roots + binaries |
| `#[allow()]` in production | **0** |
| Local WGSL shaders | **0** |
| Duplicate math | **0** |
| C dependencies | **0** (flate2 rust_backend, blake3 pure) |

---

## What wetSpring Needs from barraCuda (unchanged from V133 + additions)

| Priority | Ask | Why |
|----------|-----|-----|
| **High** | **`BatchReconcileGpu`** | `reconciliation_gpu` is still CPU passthrough; DTL wavefront DP needs batched GPU kernel |
| **High** | **DF64 GEMM pipeline** | Spectral cosine and dot/GEMM on FP32-heavy hardware; precision routing ready |
| **Medium** | **CPU Jacobi eigensolve** in `barracuda::linalg` | `pcoa.rs` has local CPU Jacobi; parity story vs `BatchedEighGpu`; should share implementation |
| **Medium** | **`BandwidthTier` in metalForge substrate** | PCIe-aware routing for cross-system dispatch |
| **Medium** | **`BatchMergePairsGpu`** | `merge_pairs_gpu` overlap consensus needs dedicated kernel |
| **Low** | **`GbmBatchInferenceGpu`** | GBM GPU inference (currently CPU fallback for parity) |
| **Low** | **GPU NJ / Robinson-Foulds** | Sequential NJ loop and full RF distance on GPU — algorithmic kernels needed |

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
  consumption example of the Feb 2026 absorption. `NmfConfig` → `nmf()` → `NmfResult`.
- **`#[expect(reason)]` over `#[allow()]` pattern** — first spring to achieve
  zero `#[allow()]` codebase-wide. Unfulfilled expectations warn, stale
  suppressions caught by clippy `unfulfilled_lint_expectations`.
- **Audit methodology** — 8-axis audit pattern (completion, quality, fidelity,
  dependency health, GPU readiness, coverage, ecosystem standards, primal
  coordination) reusable across springs.
- **Coverage discipline** — 91%+ with per-crate 90% gates in CI, local alias
  aligned. Integration tests + fuzz + proptest complement llvm-cov.
- **Smart refactor patterns** — domain cohesion over arbitrary splitting; files
  max 939 lines with clean module boundaries.

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

*End of V134 barraCuda / toadStool absorption handoff — 2026-03-23.*
