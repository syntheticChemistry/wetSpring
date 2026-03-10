# wetSpring V95 — Cross-Spring Evolution Complete (barraCuda v0.3.1 + ToadStool S93)

**Date**: 2026-03-04  
**Scope**: Full cross-spring rewiring, validation, benchmarking, provenance  
**Predecessor**: V94 barraCuda Evolution Sync  
**Experiment**: Exp305

---

## Summary

Completed the full rewire of wetSpring to consume modern barraCuda v0.3.1
primitives, including new GPU ops, adaptive ODE solvers, numerical utilities,
and cross-spring statistical primitives. Created validation + benchmark binary
(Exp305) documenting the full shader provenance across all springs.

---

## New GPU Modules Wired

| Module | File | barraCuda Op | Purpose |
|--------|------|-------------|---------|
| `tolerance_search_gpu` | `bio/tolerance_search_gpu.rs` | `BatchToleranceSearchF64` | GPU PFAS suspect screening (S×R batch) |
| `kmd_grouping_gpu` | `bio/kmd_grouping_gpu.rs` | `KmdGroupingF64` | GPU Kendrick mass defect + homologue grouping |
| `stats_extended_gpu` | `bio/stats_extended_gpu.rs` | `JackknifeMeanGpu` | GPU jackknife mean/variance/SE |
| | | `BootstrapMeanGpu` | GPU bootstrap resampled means |
| | | `KimuraGpu` | GPU batch Kimura fixation probability |
| | | `HargreavesBatchGpu` | GPU batch Hargreaves ET₀ |

## New CPU Delegations

| Function | File | barraCuda Source | Purpose |
|----------|------|-----------------|---------|
| `rk45_integrate` | `bio/ode.rs` | `numerical::rk45::rk45_solve` | Adaptive Dormand-Prince ODE solver |
| `gradient_1d` | `special.rs` | `numerical::gradient_1d` | Central-difference numerical gradient |

---

## Cross-Spring Shader Provenance

| Primitive | Origin Spring | Precision from | GPU from | Absorbed in |
|-----------|--------------|----------------|----------|-------------|
| FusedMapReduceF64 | wetSpring | hotSpring | neuralSpring | barraCuda S31 |
| BrayCurtisF64 | wetSpring | hotSpring | wetSpring | barraCuda S31 |
| BatchedOdeRK4 | wetSpring | hotSpring | wetSpring | barraCuda S58 |
| GemmF64 | neuralSpring | hotSpring | neuralSpring | barraCuda S31 |
| lanczos_eigenvalues | groundSpring | hotSpring | groundSpring | barraCuda S54 |
| anderson_eigenvalues | groundSpring | hotSpring | groundSpring | barraCuda S54 |
| boltzmann_sampling | groundSpring | hotSpring | groundSpring | barraCuda S56 |
| rk45_solve | hotSpring | hotSpring | hotSpring | barraCuda S58 |
| norm_ppf | barraCuda | — (CPU) | — (CPU) | barraCuda S59 |
| gradient_1d | barraCuda | — (CPU) | — (CPU) | barraCuda S54 |
| KimuraGpu | groundSpring | hotSpring | neuralSpring | barraCuda S58 |
| HargreavesBatchGpu | airSpring | hotSpring | airSpring | barraCuda S66 |
| JackknifeMeanGpu | wetSpring | hotSpring | wetSpring | barraCuda S60 |
| BootstrapMeanGpu | wetSpring | hotSpring | wetSpring | barraCuda S60 |
| BatchTolSearchF64 | wetSpring | — (GPU native) | wetSpring | barraCuda S41 |
| KmdGroupingF64 | wetSpring | — (GPU native) | wetSpring | barraCuda S41 |
| SmithWatermanGpu | neuralSpring | — (int scoring) | neuralSpring | barraCuda S31 |
| FelsensteinGpu | wetSpring | hotSpring f64 | wetSpring | barraCuda S31 |
| graph_laplacian | groundSpring | — (CPU) | — (CPU) | barraCuda S54 |

### Evolution Path

```
Python baseline → Rust validation → GPU acceleration → barraCuda absorption
→ cross-spring availability → sovereign pipeline
```

### Spring Contributions

- **hotSpring**: f64 precision polyfills, DF64 double-float emulation, Rk45 adaptive ODE,
  thermal shaders — precision backbone for all GPU ops
- **wetSpring**: Bio shaders (diversity, ODE, phylogeny, mass spec), tolerance search,
  KMD grouping, jackknife/bootstrap — biology domain
- **neuralSpring**: GEMM, Smith-Waterman alignment, batch fitness, Hamming distance,
  multi-head ESN — ML/alignment domain
- **groundSpring**: Spectral theory (Anderson, Lanczos), Kimura fixation, graph Laplacian,
  Latin hypercube/Sobol sampling — mathematical physics domain
- **airSpring**: Hargreaves ET₀, Thornthwaite PET — hydrology domain

---

## Quality Gate Results

| Check | Status | Details |
|-------|--------|---------|
| `cargo test --workspace --lib` | PASS | 1,261 tests (1,061 + 200 forge), 0 failures |
| `cargo clippy --pedantic` | PASS | 0 warnings |
| `cargo fmt --check` | PASS | Clean |
| `cargo doc --no-deps` | PASS | 0 warnings (RUSTDOCFLAGS=-D warnings) |
| `cargo llvm-cov` | **94.69%** | Line coverage (up from 94.68%) |
| Exp305 validation | PASS | 59/59 checks, all domains |

## Benchmark Results (Release Mode)

| Domain | Time (µs) | Notes |
|--------|-----------|-------|
| Shannon entropy (10K, 100×) | <1 | Vectorized reduction |
| gradient_1d (10K pts, 100×) | 7 | Central differences |
| RK45 exponential decay (10×) | 14 | 54 adaptive steps |
| norm_ppf (10K calls) | <0.01 | Rational approximation |
| Lotka-Volterra 2D t=50 | 82 | Adaptive RK45, 2 species |

### RK4 vs RK45 Comparison

- Fixed-step RK4 (dt=0.01): 1,000 steps
- Adaptive RK45 (rtol=1e-8): 54 steps — **18.5× fewer steps**, same accuracy to 1e-6

---

## Files Changed

### New files
- `barracuda/src/bio/tolerance_search_gpu.rs` — GPU batch tolerance search wrapper
- `barracuda/src/bio/kmd_grouping_gpu.rs` — GPU KMD + grouping wrapper
- `barracuda/src/bio/stats_extended_gpu.rs` — Jackknife, Bootstrap, Kimura, Hargreaves GPU
- `barracuda/src/bin/validate_cross_spring_s93.rs` — Exp305 cross-spring validation

### Modified files
- `barracuda/src/bio/mod.rs` — Register 3 new GPU modules
- `barracuda/src/bio/ode.rs` — Add `rk45_integrate` + 2 tests
- `barracuda/src/special.rs` — Add `gradient_1d` + 2 tests
- `barracuda/Cargo.toml` — Register Exp305 binary

---

## Architectural State After V95

```
wetSpring (V95) ─── depends on ──→ barraCuda v0.3.1 (standalone)
    │                                   │
    │ 0 local WGSL shaders              │ 767+ WGSL shaders
    │ 1,261 lib tests                   │ f64-canonical precision
    │ 94.69% line coverage              │ Precision::F16/F32/F64/Df64
    │ 150+ consumed primitives          │ driver-aware polyfills
    │                                   │
    │ GPU modules: 47 (was 44)          │ All springs contribute shaders
    │ CPU delegations: ~30              │ All springs consume shaders
    │ Validation binaries: 20+          │
    └───────────────────────────────────┘
```

## Roadmap (Future Wiring)

| Priority | Primitive | Status |
|----------|-----------|--------|
| P1 | `rk45_solve` + `gradient_1d` (CPU) | **DONE** (V95) |
| P1 | `norm_ppf` (CPU) | **DONE** (V94) |
| P2 | `BatchToleranceSearchF64` (GPU) | **WIRED** (V95) |
| P2 | `KmdGroupingF64` (GPU) | **WIRED** (V95) |
| P3 | `JackknifeMeanGpu` (GPU) | **WIRED** (V95) |
| P3 | `BootstrapMeanGpu` (GPU) | **WIRED** (V95) |
| P3 | `KimuraGpu` (GPU) | **WIRED** (V95) |
| P3 | `HargreavesBatchGpu` (GPU) | **WIRED** (V95) |
| Future | `ComputeDispatch` tarpc integration | Awaiting ToadStool S95+ |
| Future | DF64 GEMM public API | Awaiting barraCuda v0.4 |
| Future | `BandwidthTier` integration | Awaiting ToadStool S95+ |
| Future | `domain-genomics` crate extraction | wetSpring candidate |

---

## Handoff Prepared By

- wetSpring V95 cross-spring evolution validation
- Exp305: 59/59 checks passed (release mode)
- All quality gates green
- Cross-spring provenance fully documented
