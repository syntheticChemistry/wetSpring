# wetSpring V133 — Deep Evolution & Cross-Ecosystem Absorption

| Field | Value |
|-------|--------|
| **Date** | 2026-03-23 |
| **From** | wetSpring **V133** |
| **To** | All springs + all primals |
| **License** | AGPL-3.0-or-later |
| **Supersedes** | V132 central handoffs (and prior V127/V129-era central narratives) |

---

## Executive Summary

V133 is a **deep evolution sprint** focused on cross-ecosystem absorption and debt resolution: **`validate_all` meta-runner** (primalSpring-style aggregate validation), **`GpuContext` / `TensorSession`** GPU session layering (ludoSpring V29 pattern), **`check_relative` / `check_abs_or_rel`** tolerance helpers (groundSpring V120), **zero-copy I/O** (`Arc<Path>`, `Arc<str>` pools), **`performance_surface` fully implemented** (client wired to toadStool RPCs), **smart module refactoring** (domain cohesion, not mechanical splits), and **feature-gate cleanup** (7 GPU-heavy binaries gated; `--no-default-features` builds clean).

**Quality bar:** **1,781** tests, **0** clippy warnings (pedantic + nursery), **0** `unsafe`, **234** named tolerances, **307** validation binaries — each with `# Provenance` / provenance headers as applicable.

---

## What Changed

Aligned with **CHANGELOG [V133]** — formatted for handoff readers.

### Added

- `validate_all` meta-runner binary to run the full validation suite in one shot.
- `GpuContext` wrapping `Arc<WgpuDevice>` and delegating `TensorSession::with_device()` (`gpu/context.rs`).
- `check_relative` / `check_abs_or_rel` on the validation `Validator` for relative and hybrid absolute/relative checks.
- `scripts/validate_release.sh` for release-mode validation in CI-style workflows.
- Cargo coverage aliases (`coverage-check` with **80%** line floor; related workspace commands).
- `ipc/message.rs` and `ipc/dispatch_strategy.rs` for structured IPC routing.

### Changed

- **`performance_surface`:** stub replaced with a complete implementation, wired to toadStool `compute.performance_surface.*`.
- **`protocol.rs`**, **`compute_dispatch.rs`:** refactored for clarity and maintainability.
- **`dorado.rs`:** `/root`-style paths replaced with capability discovery.
- **MS2 / XML I/O:** zero-copy paths — `Arc<Path>` for shared paths; XML string pool uses `Arc<str>`.
- Broad **`println!` → `tracing`** migration; doctest `ignore` → `no_run` where appropriate.
- **11** provenance headers added workspace-wide; **7** binaries feature-gated behind GPU/heavy features.
- `barracuda/Cargo.toml` barraCuda version comment updated **v0.3.5 → v0.3.7**.

### Fixed

- `EVOLUTION_READINESS.md` metric counts corrected.
- `ABSORPTION_MANIFEST.md` tolerances and cross-references.
- Clippy issues in newly introduced code paths.

---

## Quality Metrics

| Metric | Value |
|--------|--------|
| **Tests** | 1,781 passed (1,529 barracuda + 252 forge), 0 failed |
| **Clippy** | 0 warnings (`-D warnings`, pedantic + nursery) |
| **Unsafe** | 0 (`#![forbid(unsafe_code)]` on crate roots + binaries) |
| **barraCuda** | v0.3.7; 150+ primitives consumed |
| **Debt markers** | 0 `#[allow()]` codebase-wide (see `EVOLUTION_READINESS.md`) |
| **Coverage** | `cargo coverage-check` — **80%** line floor (llvm-cov) |
| **Validation binaries** | 307 (all with provenance discipline) |
| **Named tolerances** | 234 |

---

## Patterns Worth Absorbing Upstream

1. **`validate_all` meta-runner** — single entrypoint to fan out domain validation binaries (mirrors primalSpring aggregate validation).
2. **`GpuContext` + `TensorSession`** — one `Arc<WgpuDevice>`, many `TensorSession` instances; matches ludoSpring V29 session batching.
3. **`check_relative` / `check_abs_or_rel`** — groundSpring V120-style helpers on a shared `Validator`.
4. **`cargo coverage-check` alias** — workspace-wide **80%** line coverage floor (`.cargo/config.toml`).
5. **Release-mode validation CI** — `scripts/validate_release.sh` for opt-level consistency with production.
6. **Smart module refactoring** — extract by **domain cohesion** (IPC message vs dispatch strategy), not line-count splits.

---

## Open Items for Next Session

- **Upstream wiring:** toadStool server endpoints for `compute.performance_surface.report` / `.query` (client complete in wetSpring).
- **GPU reconciliation:** `BatchReconcileGpu` — `reconciliation_gpu` still CPU passthrough pending barraCuda.
- **Precision:** DF64 GEMM path for spectral cosine and related workloads.
- **metalForge:** `BandwidthTier` in substrate model for PCIe-aware routing.
- **Dispatch:** `ComputeDispatch` → next-generation BGL model — migration guidance from barraCuda/ToadStool.
- **PCoA:** CPU Jacobi eigensolve in `pcoa.rs` vs GPU `BatchedEighGpu` — document and align tolerances.
- **ToadStool:** `compute.route.multi_unit` for large bio pipelines; `DeviceCapabilities::latency_model()` refinements.
- **Docs:** keep `EVOLUTION_READINESS.md` / `ABSORPTION_MANIFEST.md` counts in sync with automated metrics where possible.

---

*End of V133 deep evolution handoff — 2026-03-23.*
