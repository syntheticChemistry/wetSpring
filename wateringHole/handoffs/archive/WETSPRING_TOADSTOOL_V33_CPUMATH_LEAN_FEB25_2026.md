# wetSpring → ToadStool Handoff V33: CPU-Math Lean (barracuda always-on)

**Date:** February 25, 2026
**From:** wetSpring (Phase 41)
**To:** ToadStool BarraCuda team
**ToadStool:** S62 (cpu-math feature gate leveraged)

---

## Summary

wetSpring restructured its dependency on ToadStool barracuda from **optional**
(only available with `--features gpu`) to **always-on** (`default-features = false`
for CPU builds, `barracuda/gpu` enabled when the `gpu` feature is active). This
eliminated all `#[cfg(not(feature = "gpu"))]` dual-path fallback code — ~177 lines
of local math that duplicated barracuda's CPU implementations.

## Architectural Change

### Before (V32)

```toml
# Cargo.toml
barracuda = { path = "...", optional = true }   # only with gpu
gpu = ["dep:barracuda", "dep:wgpu", "dep:tokio"]
```

Every function that used barracuda required two implementations:
- `#[cfg(feature = "gpu")]` → delegate to barracuda
- `#[cfg(not(feature = "gpu"))]` → local fallback (A&S erf, Lanczos, Cholesky, etc.)

### After (V33)

```toml
# Cargo.toml
barracuda = { path = "...", default-features = false }  # always available
gpu = ["barracuda/gpu", "dep:wgpu", "dep:tokio"]
```

Single code path — barracuda's CPU math is always available. `gpu` feature only
adds GPU dispatch capability on top.

## Code Removed (~177 lines)

| File | What was removed | Lines | Now delegates to |
|------|-----------------|:-----:|------------------|
| `src/special.rs` | Local erf (A&S 7.1.26 polynomial) | ~15 | `barracuda::special::erf` |
| `src/special.rs` | Local ln_gamma (Lanczos g=5 n=6) | ~30 | `barracuda::special::ln_gamma` |
| `src/special.rs` | Local regularized_gamma (series expansion) | ~30 | `barracuda::special::regularized_gamma_p` |
| `src/bio/esn.rs` | Local Cholesky ridge regression solver | ~95 | `barracuda::linalg::ridge_regression` |
| `src/bio/eic.rs` | Local trapezoidal integration loop | ~7 | `barracuda::numerical::trapz` |
| **Total** | **5 dual-path implementations** | **~177** | **5 upstream functions** |

## What This Means for ToadStool

### Validation coverage

These 5 functions are now exercised through wetSpring's full test suite on every
build (not just GPU builds). This gives ToadStool:

- **752 CPU-only tests** that exercise barracuda CPU math
- **759 GPU tests** that exercise barracuda GPU + CPU math
- **162 experiments, 3,198+ validation checks**

### barracuda CPU-math stability contract

wetSpring now depends on these barracuda functions being available without the
`gpu` feature gate:

| Function | Module | Used by |
|----------|--------|---------|
| `erf(f64) -> f64` | `barracuda::special` | pangenome enrichment, normal CDF |
| `ln_gamma(f64) -> Result<f64>` | `barracuda::special` | Poisson p-values (DADA2) |
| `regularized_gamma_p(f64, f64) -> Result<f64>` | `barracuda::special` | Incomplete gamma (DADA2) |
| `ridge_regression(states, targets, ...)` | `barracuda::linalg` | ESN reservoir readout |
| `trapz(y, x) -> Result<f64>` | `barracuda::numerical` | EIC peak integration |

If ToadStool moves any of these behind a feature gate, wetSpring CPU-only builds
will break.

## Cross-Spring Evolution Benefits

This lean demonstrates the full Write → Absorb → Lean lifecycle completing for
CPU math — not just GPU shaders. The same barracuda functions that wetSpring
contributed (ODE systems, NMF) and that ToadStool refined are now being consumed
back by wetSpring even in CPU-only mode.

### Measured ODE benchmark (S62)

| System | Local µs | Upstream µs | Speedup |
|--------|---------|-------------|---------|
| Capacitor | 1,165 | 774 | **1.51×** |
| Cooperation | 837 | 623 | **1.34×** |
| MultiSignal | 1,589 | 1,200 | **1.32×** |
| Bistable | 1,715 | 1,415 | **1.21×** |
| PhageDefense | 85 | 61 | **1.39×** |

Upstream integrators are 21–51% faster due to ToadStool's cross-spring optimization.

## Cumulative Lean Summary (V30–V33)

| Handoff | What | Lines removed | Lines added |
|---------|------|:------------:|:-----------:|
| V30 (S59) | NMF, ridge, ODE systems, Anderson, trapz | ~1,312 | ~54 |
| V32 (S62) | PeakDetectF64, TranseScoreF64 wired | 0 | ~254 |
| V33 (S62) | CPU-math always-on, dual-path elimination | ~177 | ~10 |
| **Total** | **Full lean cycle** | **~1,489** | **~318** |

## Validation

```
cargo fmt    — clean
cargo clippy — 0 warnings (pedantic + nursery)
cargo test   — 752 passed (CPU-only)
cargo test --features gpu — 759 passed
validate_barracuda_cpu_v8 — 84/84 PASS
validate_repodb_nmf       — 9/9 PASS
validate_knowledge_graph_embedding — 9/9 PASS (incl. GPU TransE parity)
benchmark_ode_lean_crossspring — 11/11 PASS
```

## Status

| Metric | Value |
|--------|-------|
| ToadStool alignment | **S62** |
| Primitives consumed | **44** (barracuda always-on) |
| Local WGSL shaders | **0** |
| Dual-path fallback code | **0** |
| Tests | 752 CPU / 759 GPU |
| Experiments | 162 |
| Validation checks | 3,198+ |
