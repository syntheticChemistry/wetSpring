# V92I Handoff: ToadStool S87 Absorption + Full Revalidation

**Date:** March 2, 2026
**From:** wetSpring
**To:** ToadStool / BarraCuda team
**ToadStool Pin:** S87 (`2dc26792`)
**Previous Pin:** S86 (`2fee1969`)
**Status:** Full 5-tier GREEN — zero regressions, zero API breakage

---

## What Changed Upstream (S87)

ToadStool session 87 is a deep debt evolution + test resilience cycle:

| Category | Detail |
|----------|--------|
| **async-trait reclassification** | 75 `TODO(afit)` → `NOTE(async-dyn)` across 52 files; conscious architectural decision documented |
| **FHE shader fixes** | `u64_mod_simple` rewritten in `fhe_ntt.wgsl`/`fhe_intt.wgsl` (bit-by-bit modular reduction); `fhe_pointwise_mul.wgsl` `mod_mul` fixed; all 19 FHE tests pass |
| **gpu_helpers refactor** | 663-line monolith → 3 submodules: `buffers.rs`, `bind_group_layouts.rs`, `pipelines.rs` |
| **Device-lost recovery** | `BarracudaError::is_device_lost()` + `with_device_retry` test helper |
| **Unsafe audit** | ~60+ unsafe sites across barracuda + runtime/gpu documented with SAFETY comments; all verified necessary |
| **Test fixes** | 9 pre-existing failures fixed (kernel router, cross-vendor adapter, fault tests, storage buffer limits) |
| **MatMul validation** | Inner-dimension validation in `MatMul::execute()` |
| **FHE NTT guard** | Minimum degree ≥ 2 check in `FheNtt::new()` |
| **Upstream tests** | 2,866+ barracuda tests pass (1 known flaky softmax under full concurrent GPU load) |

### Not Changed

- **ComputeDispatch**: stays at 144 ops (264 total). No new ops absorbed in S87.
- **Public API surface**: zero breaking changes. All wetSpring call sites compile without modification.
- **Precision strategy**: DF64 Hybrid unchanged.

---

## wetSpring Revalidation Results

### Build
- `cargo build --release --features gpu` — CLEAN (barracuda recompiled against S87)
- `cargo build -p wetspring-forge --release` — CLEAN

### Tests
- **barracuda lib**: 1,044 passed, 0 failed, 1 ignored
- **forge lib**: 175 passed, 0 failed, 0 ignored
- **Total**: 1,219 tests GREEN

### 5-Tier Validation Chain

| Tier | Binary | Result |
|------|--------|--------|
| T1 Paper Math | `validate_paper_math_control_v3` | PASS |
| T2 CPU | `validate_barracuda_cpu_v20` | PASS |
| T2b Python | `benchmark_python_vs_rust_v3` | PASS |
| T2c S86 CPU | `validate_cross_spring_s86` | PASS |
| T3 GPU | `validate_cpu_vs_gpu_v7` | PASS |
| T3b Pure Math | `validate_cpu_vs_gpu_pure_math` | PASS |
| T3c S86 GPU | `validate_cross_spring_modern_s86` | PASS |
| T3d Exp301 | `validate_cpu_gpu_full_domain_v92g` | PASS |
| T4 Streaming | `validate_pure_gpu_streaming_v8` | PASS |
| T4b S86 Stream | `validate_s86_streaming_pipeline` | PASS |
| T5 metalForge | `validate_mixed_hw_dispatch` | ALL PASS |
| T5b Exp302 | `validate_nucleus_biomeos_v92g` | ALL PASS |
| T5c Exp303 | `validate_mixed_nucleus_v92g` | ALL PASS |

**13/13 binaries GREEN.** Zero regressions from S86→S87.

---

## Relevant for ToadStool Evolution

1. **`with_device_retry` pattern**: wetSpring's GPU tests don't yet use this — our tests are validation binaries, not async tokio tests. If ToadStool exposes a sync equivalent, we could adopt it for robustness.

2. **FHE shader fixes**: wetSpring doesn't consume FHE ops directly, but the arithmetic fixes to NTT/INTT strengthen the foundation for any future lattice-crypto workloads.

3. **gpu_helpers submodule split**: No impact on wetSpring (we don't use sparse GPU helpers directly), but the cleaner structure helps future absorption work.

4. **Unsafe audit**: Confirms all unsafe in barracuda is necessary GPU/FFI — validates wetSpring's "zero unsafe" posture (we delegate all unsafe to upstream).

---

## Metrics Delta

| Metric | Before (V92H/S86) | After (V92I/S87) |
|--------|-------------------|-------------------|
| ToadStool pin | S86 (`2fee1969`) | S87 (`2dc26792`) |
| ComputeDispatch ops | 144/264 | 144/264 (unchanged) |
| wetSpring tests | 1,219 | 1,219 (unchanged) |
| Experiments | 279 | 279 (unchanged) |
| Validation checks | 8,180+ | 8,180+ (unchanged) |
| Upstream barracuda tests | 2,866 | 2,866+ |
| API breaks | — | 0 |
| Regressions | — | 0 |

---

## Absorption Opportunities (Unchanged)

Per V92H handoff, the 3 pending absorption opportunities remain:

1. **GPU-streamable spectral primitives** (spectral_match, spectral_density → WGSL)
2. **GPU Boltzmann sampling** (LHS/Sobol → batch GPU shader)
3. **GPU Hydrology ET₀** (FAO-56/Hargreaves → fused map-reduce)

These await ToadStool bandwidth. S87's internal strengthening makes absorption of these more robust when ready.
