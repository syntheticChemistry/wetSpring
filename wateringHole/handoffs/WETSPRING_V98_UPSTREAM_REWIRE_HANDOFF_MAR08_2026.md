SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring V98+ → Upstream Rewire Handoff

**Date:** 2026-03-08
**From:** wetSpring V98+
**To:** barraCuda team + toadStool team + coralReef team
**License:** AGPL-3.0-or-later

---

## Executive Summary

- wetSpring **upstream rewire** to latest barraCuda, toadStool, and coralReef.
  **Zero API breakage.** V98 full chain re-validated: **173/173 PASS.**
- 1,047 lib tests, `cargo clippy -D warnings` ZERO WARNINGS (default + GPU),
  `cargo doc --no-deps` ZERO WARNINGS.

---

## 1. Pin Updates

| Dependency | Old Pin | New Pin | Key Changes |
|------------|---------|---------|-------------|
| **barraCuda** | `2a6c072` | `a898dee` | Deep debt: typed errors, named constants, test resilience, lint compliance. Docs refresh. |
| **toadStool** | S130 (`c470137f`) | S130+ (`bfe7977b`) | Deep debt: unsafe audit, dependency audit, hardcoding evolution, coverage expansion. Spring sync (5 springs zero breakage). Clippy pedantic clean. Docs cleanup. |
| **coralReef** | Iteration 7 (`72e6d13`) | Iteration 10 (`d29a734`) | AMD E2E GPU dispatch verified (RX 6950 XT). Push buffer fix, QMD CBUF binding. 990 tests (953 pass). |

---

## 2. Verification

| Gate | Status |
|------|--------|
| `cargo test -p wetspring-barracuda --lib` | **1,047 pass** (0 fail, 1 ignored) |
| `cargo fmt --check` | CLEAN |
| `cargo clippy -D warnings` (default) | ZERO WARNINGS |
| `cargo clippy -D warnings --features gpu` | ZERO WARNINGS |
| `cargo doc --workspace --no-deps` | ZERO WARNINGS |
| Exp313: Paper Math v5 (52 papers) | **32/32 PASS** |
| Exp314: CPU v24 (33 bio modules) | **67/67 PASS** |
| Exp316: GPU v13 (Hybrid-aware) | **25/25 PASS** |
| Exp317: Streaming v11 (zero CPU RT) | **25/25 PASS** |
| Exp318: metalForge v16 (cross-system) | **24/24 PASS** |
| **V98 total** | **173/173 PASS** |

---

## 3. What Changed in barraCuda (`2a6c072` → `a898dee`)

### `a898dee` — Deep Debt Evolution

- **Typed errors**: `BarracudaError` enriched with domain-specific variants
- **Named constants**: Magic numbers replaced with `const` in test infrastructure
- **Test resilience**: `test_harness.rs` and `test_pool.rs` expanded for better device management
- **Lint compliance**: Various clippy fixes across ops (bessel, hermite, legendre, etc.)
- **Precision test extraction**: `precision_tests.rs` split into `precision_tests_validation.rs`
- **DF64 test extraction**: `tests.rs` → `tests_nak.rs` in sovereign rewriter

### Impact on wetSpring: **None.** No public API changes in modules wetSpring consumes.

---

## 4. What Changed in toadStool (S130 → S130+)

### S130+ Commits

| Commit | Change |
|--------|--------|
| `4e575b86` | Clippy pedantic workspace-wide, doc updates, debris cleanup |
| `a7262515` | Spring sync: all 5 springs confirm zero API breakage against S130 |
| `73123cda` | Deep debt: unsafe audit, dependency audit, hardcoding evolution, coverage |
| `bfe7977b` | Root docs cleanup, test count updates, stale TESTING.md fix |

### Impact on wetSpring: **None.** No API changes. Quality-only evolution.

---

## 5. What Changed in coralReef (Iteration 7 → 10)

### Key Evolution

- **Iteration 9**: E2E wiring, push buffer fix, NVIDIA nouveau dispatch fixes
- **Iteration 10**: AMD E2E verified (RX 6950 XT RDNA2), wave32, 64-bit addressing
- **Tests**: 904 → 990 (953 pass, 37 ignored)

### Impact on wetSpring: **None.** coralReef is reached indirectly via barraCuda's
`device::coral_compiler`. IPC contract (`shader.compile.*`) unchanged.

---

## 6. New APIs Available (Not Yet Consumed)

Following groundSpring V96 guidance, these barraCuda APIs are available
for future adoption:

| API | Purpose | Priority |
|-----|---------|----------|
| `CorrelationF64::r_squared()` | R² from fused GPU correlation | P2 |
| `CorrelationF64::covariance()` | Covariance from fused GPU correlation | P2 |
| `VarianceF64::mean_variance_to_buffer()` | GPU-resident Welford (no readback) | P2 |
| `GpuView<T>` | Persistent GPU buffers for chained pipelines | P3 |
| `BatchedOdeRK45F64` | Adaptive Dormand-Prince ODE solver | P2 |

---

## 7. Ecosystem Sync Status

| Primal | Version | Status |
|--------|---------|--------|
| **barraCuda** | v0.3.3 `a898dee` | Current |
| **toadStool** | S130+ `bfe7977b` | Current |
| **coralReef** | Iteration 10 `d29a734` | Current |
| **wetSpring** | V98+ | All gates pass |

*This handoff is unidirectional: wetSpring → ecosystem. No response expected.*
