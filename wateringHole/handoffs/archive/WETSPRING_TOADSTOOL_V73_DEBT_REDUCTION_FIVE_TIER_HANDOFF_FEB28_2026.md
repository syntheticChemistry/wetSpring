# wetSpring → ToadStool/BarraCuda V73 Handoff

**Date:** February 28, 2026
**From:** wetSpring V73 (Phase 73)
**To:** ToadStool/BarraCuda team
**Status:** 229 experiments, 5,743+ checks, 1,006 lib tests, ALL PASS
**Supersedes:** V66 (archived)

---

## Executive Summary

Two major evolutions since V66:

1. **V72 — Five-Tier Validation Chain (Exp224–228)**: Paper math → CPU → GPU → streaming → metalForge, proving the full evolution path from published equations through cross-system dispatch.

2. **V73 — Deep Technical Debt Reduction**: Error types evolved to proper enums, function decomposition, safe casts, named constants, zero `expect`/`unwrap` in production code.

**Key outcome for ToadStool**: wetSpring now exercises 79 ToadStool primitives across 50 validated domains with zero local WGSL, zero unsafe code, zero panicking casts, and proper error propagation throughout. The crate is idiomatic Rust 2024.

---

## V72: Five-Tier Validation Chain (Exp224–228)

### What Was Built

| Tier | Exp | Binary | Checks | What It Proves |
|------|-----|--------|:------:|----------------|
| **Paper math** | 224 | `validate_paper_math_control_v1` | 58/58 | 18 published papers produce their exact equations in pure Rust via barracuda |
| **CPU pure Rust** | 225 | `validate_barracuda_cpu_v14` | 58/58 | 50 domains + `df64_host` + cross-spring primitives (`graph_laplacian`, `effective_rank`, `numerical_hessian`) |
| **GPU portability** | 226 | `validate_barracuda_gpu_v6` | 28/28 | CPU==GPU parity, `GemmCached::with_precision(Precision::F64)`, DF64 roundtrip, `BandwidthTier` detection |
| **Pure GPU streaming** | 227 | `validate_pure_gpu_streaming_v4` | 24/24 | 7-stage unidirectional pipeline: quality→diversity→fusion→GEMM→PCoA→spectral→DF64 |
| **metalForge cross-system** | 228 | `validate_metalforge_v8_cross_system` | 33/33 | GPU→NPU→CPU IPC dispatch, DF64 in dispatch context, PCIe bypass validation |

### New ToadStool Primitives Exercised

| Primitive | Source Spring | How wetSpring Uses It |
|-----------|-------------|---------------------|
| `linalg::graph::graph_laplacian` | neuralSpring | Exp225 CPU v14: validates symmetric positive semi-definite output, row sums = 0 |
| `linalg::graph::effective_rank` | neuralSpring | Exp225: validates rank ≥ 1 for connected graphs |
| `numerical::numerical_hessian` | neuralSpring | Exp225: validates quadratic f(x,y)=x²+y² produces identity Hessian |
| `GemmCached::with_precision(Precision::F64)` | hotSpring | Exp226: precision-flexible GEMM, cold vs cached dispatch timing |
| `df64_host::{pack, unpack}` | hotSpring | Exp225/226/227: DF64 roundtrip at ≤ 1e-13 relative error |
| `BandwidthTier` | ToadStool | Exp226: hardware bandwidth detection for adaptive dispatch |

### Papers Validated in Exp224

18 papers across 6 tracks, each producing its published equation in pure Rust:

| Track | Paper | What's Checked |
|-------|-------|---------------|
| 1 | Waters 2008 | QS steady-state B_ss at low/high density |
| 1 | Massie 2012 | Gillespie SSA mean ≈ analytical (births/(births+deaths)) |
| 1 | Fernandez 2020 | Bistable switching ODE produces trajectories |
| 1 | Srivastava 2011 | Multi-signal crosstalk dynamics |
| 1 | Bruger & Waters 2018 | Cooperation game-theory equilibria |
| 1 | Seed 2011 | Phage defense population dynamics |
| 1 | MG2023 | Pore geometry QS spatial model |
| 1b | Felsenstein 1981 | JC69 probability P(A→A) = analytical |
| 1b | Various | Robinson-Foulds distance on known trees |
| 1b | Various | HMM Viterbi path decoding |
| 1c | Various | Pangenome core/shell/cloud partitioning |
| 2 | Jones Lab | PFAS spectral matching Tanimoto > 0.9 |
| 2 | EPA | Decision tree and ridge regression on PFAS data |
| 3 | Various | NMF convergence (error decreasing), TransE scoring |
| cross | Various | Anderson localization r-statistic GOE/Poisson regime |
| cross | Various | Pearson correlation, numerical integration |

---

## V73: Deep Technical Debt Reduction

### Error Type Evolution

**Before:** `Result<Value, (i64, String)>` throughout IPC dispatch and protocol.
**After:** `RpcError { code: i64, message: String }` with named constructors:

```rust
RpcError::method_not_found(method)    // -32601
RpcError::invalid_params(msg)         // -32602
RpcError::server_error(code, msg)     // -32000 to -32099
```

Implements `Display + Error`. Used by dispatch, protocol, server, and 7 experiment binaries.

**Also:** `Result<Self, String>` in `gbm`, `decision_tree`, `random_forest` → `error::Result<Self>` using existing `Error::InvalidInput`.

### Function Decomposition

| Module | Before | After |
|--------|--------|-------|
| `bio/dada2.rs` | 100+ line `denoise()` | `init_partition()` + `em_step()` + existing `build_asvs()` |
| `ipc/dispatch.rs` | 70-line `handle_diversity()` | 6 metric helpers: `insert_shannon_if_requested`, etc. |
| `bio/gbm.rs` | 60-line `predict_batch_proba()` | `predict_single_proba()` called per sample |

### Safe Casts

| Module | Before | After |
|--------|--------|-------|
| `bio/gemm_cached.rs` | `m as u32` (silent truncation) | `dim_u32(m, "m")?` returns `Result` |
| `ipc/metrics.rs` | `duration.as_micros() as u64` | `.try_into().unwrap_or(u64::MAX)` |
| `ipc/dispatch.rs` | `val as usize` | `usize::try_from(val).unwrap_or(8)` |

### Named Constants (Replaces Hardcoding)

| Module | Before | After |
|--------|--------|-------|
| `ipc/server.rs` | `"biomeos/wetspring-default.sock"` | `DEFAULT_SOCKET_PATH_XDG` |
| `ipc/songbird.rs` | `"biomeos/songbird-default.sock"` | `DEFAULT_SOCKET_PATH_XDG` |
| `ncbi/nestgate.rs` | 4 hardcoded paths | `DEFAULT_NESTGATE_PATH_XDG`, etc. |
| `gpu.rs` | `10_000` dispatch threshold | `DISPATCH_THRESHOLD_NATIVE`, etc. |
| `bio/feature_table.rs` | `eic_ppm: 5.0` | `DEFAULT_EIC_PPM`, etc. |
| `bio/feature_table_gpu.rs` | `ms1_count < 256` | `MIN_MS1_SCANS_FOR_GPU` |

---

## What ToadStool Should Know

### 1. wetSpring Is Fully Lean

Zero local WGSL. Zero local derivatives. All 79 primitives consumed from upstream. The Write→Absorb→Lean cycle is complete for all existing domains. New science (if any) would start a new Write phase.

### 2. DF64 Precision Observations

From extensive Exp224-228 testing:
- `df64_host` pack/unpack roundtrip achieves ≤ 1e-13 relative error for values up to ~300
- For GEMM output values, the error grows slightly — relaxed from 2e-14 to 1e-13 in streaming context
- HMM log-likelihood GPU vs CPU achieves ~1e-3 parity (transcendental function variance)
- Consider: DF64 error documentation could help downstream springs set realistic tolerances

### 3. Cross-Spring Primitives Work Well

`graph_laplacian`, `effective_rank`, and `numerical_hessian` from neuralSpring integrate cleanly into wetSpring's CPU validation suite. The only friction was module path discovery (`barracuda::linalg::graph::*` vs `barracuda::linalg::*`).

**Suggestion:** Consider re-exporting commonly used cross-spring primitives at a shorter path.

### 4. Feature Gate Coupling

wetSpring's CPU-only experiment (Exp225) requires `--features gpu` because `spectral` and `linalg::graph` modules are gated behind the `gpu` feature in ToadStool's barracuda crate. These are pure CPU math — they don't dispatch to GPU.

**Suggestion:** Consider a `linalg` or `math` feature flag that exposes CPU-only linear algebra without pulling in `wgpu`/`tokio`.

### 5. `BandwidthTier` Is Useful

Exp226 validates `BandwidthTier` detection. wetSpring uses it for adaptive dispatch decisions. This is a good abstraction — springs can make hardware-aware decisions without knowing GPU specifics.

### 6. Socket Path Discovery Pattern

wetSpring evolved from hardcoded socket paths to named constants with XDG → env → temp fallback. This pattern (capability-based discovery rather than hardcoded assumptions) aligns with the Primal self-knowledge principle. Each primal discovers others at runtime.

### 7. IPC Error Types Are Now Structured

`RpcError` is a proper struct with `Display + Error`. ToadStool's IPC-facing code could adopt similar structured errors if it handles JSON-RPC.

---

## Cumulative State

| Metric | Value |
|--------|-------|
| Experiments | 229 |
| Validation checks | 5,743+ |
| Library tests | 1,006 |
| Total tests | 1,199+ |
| CPU bio modules | 40 |
| GPU bio modules | 34 |
| ToadStool primitives | 79 |
| Local WGSL | 0 |
| Unsafe code | 0 |
| `expect`/`unwrap` in production | 0 |
| Named tolerances | 92 |
| Papers validated | 52 |
| Three-tier complete | 50/50 |
| Clippy pedantic | CLEAN |

---

## Recommended ToadStool Actions

1. **Feature gate refinement**: Expose `linalg::graph`, `spectral`, `numerical` without requiring `gpu` feature
2. **DF64 precision docs**: Document expected roundtrip error bounds for various value ranges
3. **Cross-spring re-exports**: Shorter paths for commonly used primitives (`graph_laplacian`, `effective_rank`)
4. **Absorb wetSpring patterns**: `RpcError` struct, named constants, `dim_u32()` pattern for wgpu dimension safety

---

## Next Steps for wetSpring

- **Awaiting ToadStool evolution**: As ToadStool absorbs cross-spring primitives, wetSpring will rewire
- **Paper math benchmarks**: Extend Exp224 papers to full Python-vs-Rust timing (Exp217 pattern)
- **Large lattice Anderson**: DF64 at L=14-20 when ToadStool's Lanczos GPU kernel lands
- **Field genomics**: Exp196-202 pending MinION sequencer hardware
