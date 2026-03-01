# wetSpring → ToadStool/BarraCuda V76 Handoff

**Date:** February 28, 2026
**From:** wetSpring V76 (Phase 76)
**To:** ToadStool/BarraCuda team
**Status:** 229 experiments, 5,743+ checks, 1,148 tests, ALL PASS
**Supersedes:** V75 (ecoPrimals/wateringHole/) + V73 (archived)
**ToadStool Pin:** S68+ (`e96576ee`)
**License:** AGPL-3.0-only

---

## Executive Summary

V76 is a deep codebase audit and evolution pass. No new primitives requested —
this handoff documents the health of wetSpring's barracuda consumption and
identifies evolution opportunities for the ToadStool team.

**Key outcomes:**
- **82 ToadStool primitives** consumed, 0 local WGSL, 0 fallback code
- **Zero unsafe code** in the entire codebase
- **Zero todo!/unimplemented!()** anywhere
- **Zero .unwrap()/.expect()** in library code
- **97 named tolerance constants** with full scientific provenance
- **95.86% line coverage** (target was 90%)
- **Clippy pedantic CLEAN** across both crates, all targets
- **All external deps pure Rust** (only wgpu for GPU, required)
- **All files under 1000 LOC** (max 924 in a validation binary)

---

## What We Audited

### Code Quality Gates (ALL PASS)

| Gate | Status | Evidence |
|------|--------|----------|
| `cargo fmt --check` | CLEAN | 0 diffs |
| `cargo clippy --all-targets -- -D warnings -W clippy::pedantic` | CLEAN | Both crates |
| `cargo doc --no-deps` | CLEAN | 0 warnings |
| `cargo test` | 1,148 PASS | 955 lib + 60 integration + 20 doc + 113 forge |
| Line coverage | 95.86% | Exceeds 90% target |
| Unsafe code | 0 blocks | Crate-wide `deny(unsafe_code)` |
| Production mocks | 0 | All mocks isolated to `#[cfg(test)]` |
| Magic numbers | 0 | All 97 tolerances use `tolerances::` constants |
| File sizes | Max 924 LOC | Under 1000-line limit |

### Dependency Audit

| Crate | Version | Type | C Bindings? |
|-------|---------|------|-------------|
| barracuda (upstream) | path | Prod | No — pure Rust |
| akida-driver (upstream) | path | Prod (optional, `npu`) | No — pure Rust |
| wgpu | 22 | Prod (optional, `gpu`) | Yes — Vulkan/Metal/DX12 via wgpu-hal (required for GPU) |
| tokio | 1 | Prod (optional, `gpu`) | No — pure Rust |
| bytemuck | 1 | Prod | No — pure Rust |
| flate2 | 1.0 | Prod | No — `rust_backend` (miniz_oxide) |
| serde_json | 1 | Prod (optional, `json`) | No — pure Rust |
| tempfile | 3 | Dev/Prod (forge) | No — pure Rust |
| temp-env | 0.3 | Dev | No — pure Rust |

**Total external crates:** 8. **C bindings:** Only wgpu (required for GPU compute).

### Sovereignty Audit

| Check | Result |
|-------|--------|
| Hardcoded primal names in library code | 0 |
| Hardcoded primal names in binaries | Yes — provenance documentation only (cross-spring evolution lineage in `println!`) |
| Hardcoded external URLs | 2 — mzML namespace (required by spec), Zenodo DOI (data provenance) |
| Compile-time coupling to other primals | 0 |
| Runtime primal discovery | Capability-based (Songbird, NestGate socket cascade) |
| `todo!()` / `unimplemented!()` | 0 |
| `#[allow(dead_code)]` | 2 — both justified (pub on pub(crate) struct, provenance fields) |

---

## Workspace Evolution

Created `wetSpring/Cargo.toml` as virtual workspace root:

```toml
[workspace]
resolver = "2"
members = ["barracuda", "barracuda/fuzz", "metalForge/forge"]
```

Removed `[workspace]` from sub-crate Cargo.toml files. `cargo` commands now
work from `wetSpring/` root.

---

## BarraCuda Consumption Health

### Primitive Usage Summary (82 consumed)

| Category | Count | Examples |
|----------|:-----:|---------|
| GPU compute ops | 30+ | FusedMapReduceF64, GemmF64, BrayCurtisF64, BatchedOdeRK4, ComputeDispatch |
| Bio ops | 15+ | SmithWatermanGpu, GillespieGpu, FelsensteinGpu, DiversityFusionGpu |
| Stats delegations | 10+ | shannon, simpson, bray_curtis, pearson, mean, percentile |
| Spectral theory | 5+ | Anderson Hamiltonian, Lanczos, level_statistics |
| Cross-spring | 8+ | PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness |
| Device/dispatch | 4+ | GpuF64, compile_shader_universal, ComputeDispatch, BandwidthTier |

### No Duplicate Math

All math that exists in barracuda is consumed from barracuda. Local math is
limited to:
- `special::erf`, `special::ln_gamma` — CPU-only statistical functions not
  yet in barracuda (upstream extraction candidate: `barracuda::math`)
- `special::dot`, `special::l2_norm` — CPU f64 slice operations (barracuda's
  `dotproduct` is a GPU Tensor op, not equivalent)

### Evolution Opportunities for ToadStool

| Opportunity | Priority | Impact |
|-------------|----------|--------|
| `barracuda::math::{erf, ln_gamma, normal_cdf}` CPU feature | Medium | Eliminates wetSpring's `special.rs` (~200 lines). All springs would benefit. |
| GPU Lanczos kernel | Low | Currently CPU-bound (~457ms for L=8). Would accelerate Anderson spectral. |
| `BatchReconcileGpu` wavefront DP | Low | Full GPU DTL reconciliation. Current design uses FusedMapReduceF64 for batch cost aggregation (correct for parallelizable portion). |

---

## Test Coverage Detail

| Module | Lines | Coverage |
|--------|:-----:|:--------:|
| barracuda lib (955 tests) | ~12,000 | 95.86% |
| metalForge/forge (113 tests) | ~2,500 | 83.82% |
| Integration tests | 60 | — |
| Doc tests | 20 | — |
| Validation binaries | 210 | — (binary, not lib) |

### What's NOT Covered (by design)

- Validation binary `main()` functions — these are tested by running the
  binaries themselves (5,743+ checks)
- GPU code paths when `--features gpu` is not enabled
- NPU code paths when AKD1000 hardware is not present
- IPC server code when biomeOS is not running

---

## Three-Tier Validation Status

| Tier | Status | Checks |
|------|--------|:------:|
| Python baseline | 57 scripts, all reproducible | — |
| BarraCuda CPU | 1,476+ PASS | 407/407 parity |
| BarraCuda GPU | 1,833+ PASS | 29 domains |
| Pure GPU streaming | 152+ PASS | 441-837× vs round-trip |
| metalForge cross-system | 39/39 three-tier | All papers |
| NPU reservoir | 59 PASS | 6 domains on AKD1000 |

---

## For the ToadStool Team

1. **wetSpring is a clean consumer.** 82 primitives consumed with zero local
   WGSL, zero fallback code, zero unsafe. The crate compiles and passes all
   tests with ToadStool S68+ pinned.

2. **No blocking issues.** All GPU modules work. ComputeDispatch adopted in
   6 modules. BatchedMultinomialGpu, PairwiseL2Gpu, FstVariance all wired.

3. **Extraction candidate:** `special::{erf, ln_gamma, regularized_gamma_lower,
   normal_cdf}` — ~200 lines of CPU math that all springs could share via a
   `barracuda::math` feature (CPU-only, no wgpu dependency).

4. **The workspace root Cargo.toml** (`wetSpring/Cargo.toml`) now exists.
   This stops cargo from walking up to `ecoPrimals/Cargo.toml`. All sub-crates
   are workspace members.
