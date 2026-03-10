# SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring V94 — barraCuda v0.3.1 Evolution Sync + ToadStool S93 Catch-Up

**Date:** 2026-03-04
**From:** wetSpring team (V94)
**To:** barraCuda team, toadStool team, ecosystem
**barraCuda version:** v0.3.1 (standalone primal)
**ToadStool session:** S93 (absorbed, caught up)
**wetSpring tests:** 1,057 lib + 200 forge — 0 failures
**Coverage:** 94.68% line
**License:** AGPL-3.0-or-later
**Supersedes:** V93 rewire handoff + V93+ deep debt handoff

---

## Executive Summary

- **ToadStool S88–S93 evolution absorbed**: Reviewed all ToadStool commits since
  V92F, including S88 spring absorption, S89 barraCuda extraction, S93 DF64
  transfer. wetSpring is now fully caught up.
- **barraCuda v0.3.1 rewire verified clean**: 1,257 total tests passing (1,057 lib
  + 200 forge), zero clippy warnings (pedantic), zero fmt diffs, zero doc warnings.
- **`norm_ppf` (inverse normal CDF) wired**: New delegation from barraCuda's
  `stats::norm_ppf` — enables quantile-based statistical tests and confidence
  intervals without local approximation code.
- **50+ library files cleaned**: All production module docs updated from "ToadStool"
  to "barraCuda" for current dependency references. Historical provenance
  preserved ("absorbed from ToadStool S54" etc.).
- **groundSpring rewire guidance applied**: Followed the `if let Ok` error handling
  pattern and capability-based discovery patterns documented by groundSpring V8–V10.

---

## Part 1: ToadStool/barraCuda Evolution Delta

### What ToadStool/barraCuda evolved since V92F

| Session | Key Changes | wetSpring Impact |
|---------|-------------|------------------|
| S88 | Spring absorption from all 5 springs, `anderson_4d` re-export, `SeasonalGpuParams::new()` | Already consumed |
| S89 | barraCuda extracted to standalone primal (956 .rs, 767 WGSL) | Already rewired (V93) |
| S90-S92 | Incremental fixes, neuralSpring precision sync, matmul rewiring | No wetSpring impact |
| S93 | D-DF64 transfer, precision strategy, df64_rewrite as default | DF64 path available; not blocking |

### New barraCuda v0.3.1 capabilities available

| Capability | Status | Priority |
|-----------|--------|----------|
| `norm_ppf` (inverse normal CDF) | **WIRED** (V94) | — |
| `gradient_1d` (numerical gradient) | Available | P3 |
| `rk45_solve` (adaptive ODE) | Available | P3 |
| `BatchToleranceSearchF64` (GPU PFAS search) | Available | P2 |
| `KmdGroupingF64` (GPU KMD homologues) | Available | P2 |
| `JackknifeMeanGpu` (GPU jackknife) | Available | P3 |
| `BootstrapMeanGpu` (GPU bootstrap) | Available | P3 |
| `KimuraGpu` (GPU Kimura fixation) | Available | P3 |
| `HargreavesBatchGpu` (GPU ET₀) | Available | P3 |
| `SeasonalPipelineF64` | Available | P3 |
| `HistogramGpu` (generic histogram) | Available | P3 |
| `trapz_product` (product integration) | Available | P3 |
| `tridiagonal_solve_f64` | Available | P3 |

### What wetSpring already consumes (144+ primitives)

No change from V93 — all previously wired primitives continue to work with
barraCuda v0.3.1 standalone. Zero API breakage confirmed.

---

## Part 2: Changes Made

### 2.1 New Primitive: `norm_ppf`

Added `special::norm_ppf` — inverse normal CDF delegating to
`barracuda::stats::norm_ppf`. Uses Beasley-Springer-Moro rational approximation.
3 new tests: median (exact 0), round-trip (5 quantiles), known quantiles (97.5th).

### 2.2 Documentation Evolution (50+ files)

All production library module docs updated to reference "barraCuda" instead of
"ToadStool" where describing the current dependency source. Examples:

| Before | After |
|--------|-------|
| "via `ToadStool`'s `FusedMapReduceF64`" | "via barraCuda's `FusedMapReduceF64`" |
| "`ToadStool`'s `BatchedEighGpu`" | "barraCuda's `BatchedEighGpu`" |
| "GPU device wrapper bridging to `ToadStool`" | "GPU device wrapper bridging to barraCuda" |

Historical provenance references preserved as-is. Example kept unchanged:
> "Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean)"

### 2.3 Files Modified

| Category | Files | Change |
|----------|:-----:|--------|
| `special.rs` | 1 | `norm_ppf` delegation + 3 tests + doc update |
| `lib.rs` | 1 | Module doc ToadStool → barraCuda |
| Bio GPU wrappers | 42 | Module doc ToadStool → barraCuda |
| Bio CPU modules | 8 | Module doc ToadStool → barraCuda |
| ESN modules | 5 | Module doc ToadStool → barraCuda (file `toadstool_bridge.rs` name preserved) |
| Tolerance modules | 2 | Doc update |
| Infrastructure | 2 | `df64_host.rs`, `ipc/dispatch.rs` doc update |

---

## Part 3: Architecture After Sync

```
wetSpring V94 (validation Spring)
  ├── barracuda/     → barraCuda v0.3.1 (standalone math primal, direct dep)
  │   ├── 144+ GPU primitives consumed
  │   ├── 767+ f64-canonical WGSL shaders available
  │   ├── Universal precision: f64/f32/f16/Df64 per hardware
  │   ├── Pure math shaders: math_f64.wgsl, math_f64_special.wgsl
  │   ├── Precision dispatch: GpuDriverProfile, Fp64Strategy, EigensolveStrategy
  │   └── DF64 transcendentals available (S93 transfer)
  ├── metalForge/    → barraCuda v0.3.1 (substrate routing, direct dep)
  └── akida-driver   → toadStool neuromorphic (independent, optional)
```

Dependency direction: `wetSpring → barraCuda` (direct). No toadStool dependency.

---

## Part 4: Quality Gate

| Gate | Result |
|------|--------|
| `cargo test --workspace --lib` | **1,257 passed**, 0 failed, 1 ignored |
| `cargo clippy --workspace -D warnings -W pedantic` | 0 warnings |
| `cargo fmt --check` | 0 diffs |
| `cargo doc --no-deps -D warnings` | Clean |
| `cargo llvm-cov --workspace --lib` | **94.68%** line coverage |
| Zero `unsafe` code | PASS |
| Zero `todo!()`/`unimplemented!()` | PASS |
| AGPL-3.0-or-later headers | PASS |
| Named tolerance constants | 164 (zero inline literals) |
| Validation binaries | 284 (all with provenance) |

---

## Part 5: GPU Primitive Wiring Roadmap

### Already wired (144 primitives)

Full inventory in `EVOLUTION_READINESS.md`. All bio modules are Lean on upstream
barraCuda. Zero local WGSL shaders. Zero local math.

### Next wiring candidates (ordered by impact)

| Priority | Primitive | Use Case | Effort |
|:--------:|-----------|----------|:------:|
| P2 | `BatchToleranceSearchF64` | GPU PFAS suspect screening (10K × 259 = 2.59M comparisons) | Low |
| P2 | `KmdGroupingF64` | GPU KMD homologue grouping for PFAS series detection | Low |
| P3 | `rk45_solve` | Adaptive ODE stepping for stiff bio systems | Medium |
| P3 | `JackknifeMeanGpu` | GPU jackknife for diversity confidence intervals | Low |
| P3 | `BootstrapMeanGpu` | GPU bootstrap for RAWR and diversity stats | Low |
| P3 | `KimuraGpu` | GPU Kimura fixation probability for population genetics | Low |
| P3 | `HargreavesBatchGpu` | GPU batch Hargreaves ET₀ for hydrology | Low |
| P3 | `gradient_1d` | Numerical gradients for ODE sensitivity analysis | Low |

### Not wired (intentionally)

| Primitive | Reason |
|-----------|--------|
| `ComputeDispatch` (tarpc) | wetSpring calls primitives directly; tarpc adds IPC layer not needed for validation |
| `domain-genomics` | barraCuda hasn't extracted genomics domain yet |
| `barracuda-core` | IPC/service layer; wetSpring has its own IPC (`wetspring_server`) |
| `staging::UnidirectionalPipeline` | Used via metalForge; no direct need |
| `LatencyModel` | Performance optimization; not validation-blocking |

---

## Part 6: Cross-Spring Evolution Status

### groundSpring rewire patterns applied

| Pattern | Status |
|---------|--------|
| `if let Ok` error handling (CPU fallback always compiles) | Already in use |
| Capability-based discovery (zero hardcoded primal names) | Already in use |
| Environment variable cascades for all paths | Already in use |
| Feature gates for GPU/NPU (CPU always works) | Already in use |

### What wetSpring can contribute back

| Contribution | Type |
|-------------|------|
| 47 bio algorithm implementations | Domain knowledge for `domain-genomics` |
| 164 tolerance constants with scientific justification | Quality infrastructure pattern |
| `norm_ppf` round-trip tolerance characterization | Precision data (1.3e-6 max error at ±2σ) |
| 284 experiment protocols | Validation methodology |
| metalForge substrate routing | Hardware discovery patterns |

---

## Part 7: Next Steps

1. **P2: Wire GPU PFAS primitives** — `BatchToleranceSearchF64` and `KmdGroupingF64`
   for large-scale PFAS suspect screening (Exp018 use case).
2. **P3: Wire GPU stats** — `JackknifeMeanGpu`, `BootstrapMeanGpu` for avoiding
   CPU round-trips in diversity pipelines.
3. **P3: Explore `rk45_solve`** — Adaptive stepping for stiff ODE systems where
   fixed-step RK4 requires very small dt.
4. **Monitor barraCuda evolution** — Watch for `domain-genomics` extraction,
   CPU Jacobi upstreaming, and `ComputeDispatch` tarpc stabilization.
