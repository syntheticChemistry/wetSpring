# SPDX-License-Identifier: AGPL-3.0-only

# wetSpring V97d+ → Ecosystem Sync Handoff (barraCuda + toadStool + coralReef)

**Date:** 2026-03-07
**From:** wetSpring V97d+ (286 experiments, 8,400+ checks, 1,047 + 200 tests)
**To:** barraCuda team, toadStool team, coralReef team
**License:** AGPL-3.0-only
**Covers:** Ecosystem sync: barraCuda `0bd401f` → `2a6c072`, toadStool S94b → S130, coralReef → Phase 10

---

## Executive Summary

wetSpring has pulled and validated against the latest ecosystem state:
- **barraCuda** `2a6c072`: 8 new commits since our last pin (`0bd401f`). Gains:
  provenance module, `BatchedOdeRK45F64`, `PrecisionRoutingAdvice`, builder
  patterns, `mean_variance_to_buffer()`, DF64 Hybrid error (not silent zeros).
- **toadStool** S130: 16 new commits since our last tracked session (S94b). Gains:
  cross-spring provenance, coralReef shader compile proxy, `science.*` IPC (10
  methods), 19,140+ tests, god file refactoring, C dep evolution.
- **coralReef** Phase 10: New system — sovereign Rust GPU compiler
  (WGSL/SPIR-V → native GPU binary), no C deps, SM70–SM89 + RDNA2. IPC:
  `shader.compile.spirv`, `shader.compile.wgsl`.

All 1,347 wetSpring tests PASS against the latest ecosystem. Zero API breakage.
Zero clippy warnings. Zero doc warnings.

---

## 1. What We Validated

| Check | Status | Notes |
|-------|--------|-------|
| `cargo check --workspace` | PASS | Compiles clean against barraCuda 2a6c072 |
| `cargo clippy --workspace -- -D warnings -W clippy::pedantic` | PASS | Zero warnings |
| `cargo doc --workspace --no-deps` | PASS | Zero warnings |
| `cargo test --workspace` | PASS | 1,347 pass, 0 fail, 1 ignored |

No API breakage from the 8 barraCuda commits. All 150+ consumed primitives
resolve correctly. The provenance module, RK45, and builder patterns are
available but not yet consumed — they represent future evolution targets.

---

## 2. New Primitives Available (Not Yet Consumed)

### From barraCuda `2a6c072`

| Primitive | Purpose | wetSpring Target |
|-----------|---------|-----------------|
| `shaders::provenance::evolution_report()` | Cross-spring shader provenance reporting | Validation binaries (provenance audit) |
| `shaders::provenance::shaders_consumed_by(WetSpring)` | wetSpring shader inventory | EVOLUTION_READINESS.md automation |
| `BatchedOdeRK45F64` | Adaptive Dormand-Prince ODE integration | Bio ODE systems (bistable, cooperation, phage defense) |
| `mean_variance_to_buffer()` | GPU-resident fused Welford (zero readback) | Chained GPU pipelines (diversity → stats) |
| `PrecisionRoutingAdvice` | Per-reduction precision routing | GPU validation binaries (shared-mem f64 safety) |
| `HmmForwardArgs` | Builder for HMM forward dispatch | `bio::hmm_gpu` |
| `Dada2DispatchArgs` | Builder for DADA2 dispatch | `bio::dada2_gpu` |
| `GillespieModel` | Builder for Gillespie SSA dispatch | `bio::gillespie` GPU path |
| `Rk45DispatchArgs` | Builder for RK45 per-step dispatch | ODE GPU binaries |

### From toadStool S130

| Capability | Purpose | wetSpring Status |
|-----------|---------|-----------------|
| `cross_spring_provenance.rs` | Provenance tracking across springs | Available for IPC integration |
| `science.*` IPC namespace | 10 JSON-RPC methods for science workloads | `wetspring_server` can consume |
| `shader.compile.*` proxy | coralReef shader compilation via toadStool | Indirect via barraCuda GPU path |
| `PrecisionRoutingAdvice` routing | f64 shared-memory safety | Future GPU validation |

### From coralReef Phase 10

| Capability | Purpose | wetSpring Status |
|-----------|---------|-----------------|
| `shader.compile.spirv` | SPIR-V → native GPU binary | Indirect (via barraCuda/toadStool) |
| `shader.compile.wgsl` | WGSL → native GPU binary | Indirect (via barraCuda/toadStool) |
| Sovereign compilation | No wgpu/naga for final dispatch | Future (when sovereign pipeline is priority) |

---

## 3. Rewire Priorities

### P0: Already Done (This Session)

- Pulled barraCuda `2a6c072`, toadStool S130, coralReef Phase 10
- Full validation suite passed (1,347 tests, 0 failures)
- Updated Cargo.toml comments, EVOLUTION_READINESS.md, gap analysis
- Updated Cargo.lock to resolve latest barraCuda

### P1: Next Session — Builder Pattern Migration

wetSpring GPU binaries still use positional args for HMM, DADA2, Gillespie,
and RK4 dispatch. Migrating to builder/struct patterns (`HmmForwardArgs`,
`Dada2DispatchArgs`, etc.) will:
- Track upstream API evolution
- Improve crash diagnostics (named fields vs positional)
- Prepare for `BatchedOdeRK45F64` adoption (adaptive step needs `Rk45DispatchArgs`)

### P2: Provenance Reporting

Wire `barracuda::shaders::provenance::evolution_report()` into a new
validation binary (`validate_provenance_report`) that:
- Calls `shaders_consumed_by(SpringDomain::WetSpring)` to verify shader inventory
- Generates markdown provenance report as CI artifact
- Validates cross-spring matrix matches expected flow graph

### P3: Precision Routing

Wire `PrecisionRoutingAdvice` into GPU validation binaries to:
- Detect `F64NativeNoSharedMem` (NVK) and route reductions through DF64
- Handle DF64 Hybrid error (new behavior: returns error, not silent zeros)
- Document per-GPU behavior in validation output

---

## 4. Breaking Changes Absorbed

| Change | Impact | Status |
|--------|--------|--------|
| DF64 Hybrid fallback → error | GPU bins that assumed silent zeros must handle error | No impact (wetSpring already skips DF64-failing ops on Hybrid) |
| `device_arc()` → `device_clone()` | Device handle API | Already migrated in V97 wgpu 28 rewire |
| `queue_arc()` → `queue_clone()` | Queue handle API | Already migrated |
| `inner_arc()` removed | Use `device.device.clone()` | Not used by wetSpring |
| PPPM `new()` → `from_device()` | Constructor change | Not used by wetSpring (physics, not bio) |
| Poll timeout env var | `BARRACUDA_POLL_TIMEOUT_SECS` | Informational (default 120s is fine) |

---

## 5. Audit Confidence

| Metric | Value |
|--------|-------|
| barraCuda commit | `2a6c072` (latest HEAD) |
| toadStool session | S130 (latest) |
| coralReef phase | 10 (latest) |
| Tests passing | 1,347 (0 failures) |
| API breakage | Zero |
| Clippy warnings | Zero (pedantic) |
| Doc warnings | Zero |
| Unsafe blocks | Zero |
