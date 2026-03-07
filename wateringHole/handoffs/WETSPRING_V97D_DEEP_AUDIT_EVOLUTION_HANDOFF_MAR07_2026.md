# SPDX-License-Identifier: AGPL-3.0-only

# wetSpring V97d → Deep Audit & Idiomatic Evolution Handoff

**Date:** 2026-03-07
**From:** wetSpring V97d (286 experiments, 8,400+ checks, 1,047 lib + 200 forge tests)
**To:** barraCuda team (absorption targets), toadStool team (dispatch evolution)
**License:** AGPL-3.0-only
**Covers:** V97c → V97d (Exp311, deep audit, idiomatic evolution, doc accuracy)

---

## Executive Summary

- Comprehensive deep audit of entire wetSpring codebase against wateringHole
  standards: zero TODO, zero unsafe, zero mocks in production, all files under
  1000 LOC, AGPL-3.0 on every `.rs` file — all confirmed clean.
- Buffering I/O APIs (`parse_fastq`, `parse_mzml`, `parse_ms2`) formally
  deprecated with `#[deprecated]` attributes pointing to streaming alternatives.
- 104 bare `.unwrap()` calls evolved to contextual `.expect()` across 17
  validation binaries (12 barracuda + 5 forge).
- Documentation accuracy fixes: MSRV 1.85→1.87, wgpu v22→v28, rustdoc bracket
  escaping, broken `DEPRECATION_MIGRATION.md` reference removed.
- Full regression GREEN: fmt, clippy (pedantic), doc, test — zero warnings.

---

## 1. What We Evolved

### 1.1 I/O API Deprecation (streaming-first)

| Parser | Deprecated API | Streaming Alternative | Migration |
|--------|---------------|----------------------|-----------|
| FASTQ | `parse_fastq()` | `FastqIter::open()` | Zero-alloc record iterator |
| mzML | `parse_mzml()` | `MzmlIter::open()` | Streaming XML spectrum iterator |
| MS2 | `parse_ms2()` | `Ms2Iter::open()` | Reusable-buffer line iterator |

All three buffering functions carry `#[deprecated(since = "0.1.0", note = "...")]`
with specific migration guidance. Consumers in test/validation code carry
`#[allow(deprecated)]` for intentional comparison testing.

### 1.2 Crash Diagnostics

104 bare `.unwrap()` → `.expect("context")` across 17 validation binaries.
Context messages describe the successful-path assumption (e.g.,
`"adapter must be available for GPU validation"`,
`"pipeline should have at least one stage"`).

### 1.3 Documentation Accuracy

| Item | Was | Now |
|------|-----|-----|
| MSRV in EVOLUTION_READINESS.md | 1.85 | 1.87 |
| wgpu version in lib.rs | v22 | v28 |
| Rustdoc E[XY] brackets | Unescaped (3 warnings) | Escaped (0 warnings) |
| `DEPRECATION_MIGRATION.md` ref | Broken (file never existed) | Replaced with `../CHANGELOG.md` |
| barracuda dep path | `phase1/toadstool/crates/barracuda` | `barraCuda/crates/barracuda` |
| Root README shaders/ dir | Phantom (no such dir) | Replaced with `vault/` |

---

## 2. barraCuda Absorption Targets

### 2.1 Primitives Still Consumed (150+)

wetSpring consumes 150+ barraCuda primitives across stats, linalg, numerical,
special, spectral, ops, and device modules. All via standalone `barraCuda`
v0.3.3 (694+ WGSL shaders, wgpu 28, Fp64Strategy dispatch).

### 2.2 Upstream Requests (existing — no new ones from V97d)

| Request | Source | Status |
|---------|--------|--------|
| DF64-aware `VarianceF64` / `CorrelationF64` / `CovarianceF64` | V97c Exp308 | Open — returns zero on Hybrid GPU |
| `WeightedDotF64` DF64 support | V97c Exp308 | Open — same Hybrid GPU zero bug |
| Public `wgsl_shader_for_device()` for GEMM DF64 | V44 Phase 43 | Open — currently private |
| `math::{dot, l2_norm}` as CPU f64 ops | V42 Phase 47 | Clarified: local `special::dot` is correct (barracuda `dotproduct` is GPU tensor) |

### 2.3 wetSpring Write-Phase Candidates for Absorption

| Module | Primitive | Lines | Notes |
|--------|-----------|:-----:|-------|
| `special::dot` | CPU f64 dot product | 8 | Simple sum-of-products, distinct from GPU tensor dotproduct |
| `special::l2_norm` | CPU f64 L2 norm | 4 | `sqrt(dot(x, x))` |
| `special::erf` | Error function | 15 | Horner-form approximation, 1e-7 max error |
| `special::ln_gamma` | Log-gamma function | 12 | Stirling series, validated against scipy |

These are small but universally useful. If barraCuda absorbs them, wetSpring
will rewire to upstream in the next lean cycle.

---

## 3. toadStool Dispatch Targets

### 3.1 metalForge Forge Status

Forge crate (`wetspring-forge`) is a substrate discovery and dispatch prototype:
`probe` → `inventory` → `dispatch` → `bridge`. When toadStool absorbs forge
concepts, the `bridge` module becomes the integration seam. 200 forge tests
validate the full discovery → routing → execution path.

### 3.2 NPU Pipeline

ESN reservoir computing → int8 quantization → NPU inference pipeline is
validated in software (Exp196a-c, 52 checks). Waiting for BrainChip AKD1500
hardware for production validation. `akida-driver` dependency remains at
toadStool path (independent from barraCuda).

### 3.3 IPC / biomeOS

JSON-RPC 2.0 server (`wetspring_server`) with Songbird registration, capability
discovery cascade (env → XDG → default), GPU-aware dispatch (OnceLock lazy init,
threshold routing). Validated in Exp203-207 (222 checks, EXACT_F64 through IPC).

---

## 4. External Dependency Health

| Category | Status |
|----------|--------|
| Direct deps | 3 (barracuda, flate2, serde_json) — all pure Rust |
| GPU path | wgpu 28 → renderdoc-sys (only C transitive, upstream) |
| NPU path | akida-driver (toadStool path, optional feature) |
| TLS/HTTP | Zero — NCBI fetches use sovereign capability cascade |
| Unsafe code | Zero — `#![deny(unsafe_code)]` crate-wide |

---

## 5. Audit Confidence

| Metric | Value |
|--------|-------|
| Experiments | 286 (Exp001-311) |
| Validation checks | 8,400+ |
| Library tests | 1,047 (barracuda) + 200 (forge) |
| Coverage | 95.86% line / 93.54% fn / 94.99% branch |
| Named tolerances | 164 (scientifically justified, hierarchy-tested) |
| Clippy | Zero warnings (pedantic + nursery) |
| Doc warnings | Zero |
| Unsafe blocks | Zero |
| TODO/FIXME | Zero |
| Inline tolerance literals | Zero |
| Files > 1000 LOC | Zero |
| SPDX headers | All .rs files |
| Provenance headers | All 290 binaries |

---

## 6. What's Next

1. **barraCuda**: Fix DF64 fused shader zero-output on Hybrid GPUs (P0 from V97c).
2. **barraCuda**: Consider absorbing `special::{erf, ln_gamma, dot, l2_norm}`.
3. **toadStool**: Evaluate forge → toadStool dispatch absorption path.
4. **wetSpring**: Next phase will focus on Track 5 extension papers and
   real-hardware NPU validation when AKD1500 arrives.
