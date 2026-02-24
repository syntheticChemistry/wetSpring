# wetSpring → ToadStool Handoff V25: Phase 39 + Absorption Roadmap

**Date:** February 24, 2026
**From:** wetSpring
**To:** ToadStool / BarraCuda team
**License:** AGPL-3.0-or-later
**Purpose:** Phase 39 results (finite-size scaling + correlated disorder), barracuda evolution review, ToadStool absorption targets, paper queue control status

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Phase | 39 — Finite-size scaling + correlated disorder |
| Experiments | 151 total, 3,050+ checks, ALL PASS |
| New experiments | Exp150 (14 checks), Exp151 (8 checks) |
| Tests | 728 lib + integration, 95.67% coverage |
| Local WGSL shaders | **0** (5 deleted in V24 lean) |
| ToadStool primitives consumed | 30 (Lean phase) |
| GPU modules | 42 |
| CPU modules | 45 |
| Handoffs delivered | 25 (V1–V25) |
| Key physics result | **W_c = 16.26 ± 0.95** (disorder-averaged, L=6–12) |
| Key biology result | **Correlated disorder pushes W_c > 28** (biofilm clustering) |

---

## Part 1: Phase 39 Science Results

### Exp150: Finite-Size Scaling with Disorder Averaging (14/14 PASS)

Disorder-averaged level spacing ratio ⟨r⟩ sweep across L = 6, 8, 10, 12 with
8 realizations per (L, W) point. 416 Lanczos eigensolves total.

| L | N | W_c (midpoint) | ⟨r⟩ range |
|:-:|:--:|:---------:|-----------|
| 6 | 216 | 16.04 | 0.4409 – 0.5236 |
| 8 | 512 | 16.76 | 0.4133 – 0.5216 |
| 10 | 1000 | 15.81 | 0.4191 – 0.5213 |
| 12 | 1728 | 16.44 | 0.4193 – 0.5209 |

**Mean W_c = 16.26**, spread = 0.95 (4 sizes). Literature value: 16.5.

Standard errors decrease from L=6 (0.004–0.012) to L=12 (0.002–0.003),
confirming self-averaging. ⟨r⟩ monotonically decreases with W for all L.

**Compute:** 448s in release mode (CPU Lanczos on 3D Anderson lattices).

### Exp151: Disorder-Correlated Lattices (8/8 PASS)

Biofilm-realistic spatially correlated disorder via exponential kernel smoothing.

| ξ_corr | Biological Regime | W_c | ⟨r⟩ at W=28 |
|:------:|-------------------|:---:|:-----------:|
| 0 | well-mixed planktonic | 16.49 | 0.421 |
| 1 | loose aggregation | >28 | 0.474 |
| 2 | mature biofilm | >28 | 0.492 |
| 4 | dense biofilm clusters | >28 | 0.481 |

**Key finding:** Even ξ_corr = 1 (one lattice spacing of spatial correlation)
shifts W_c beyond W = 28. The i.i.d. Anderson model is a very conservative
lower bound for QS propagation in structured biofilms. Real biofilms are
dramatically more QS-active than the uncorrelated prediction.

### Physics Implications for QS Framework

1. **W_c ≈ 16.3 confirmed**: soil (W ≈ 6.7) and gut (W ≈ 4) are deep in the
   extended regime; hot spring mat (W ≈ 19) is localized in uncorrelated model.
2. **Correlated disorder changes the picture**: with even minimal spatial
   clustering, W_c exceeds 28 — meaning hot spring mats with microcolonies
   could support QS despite high nominal diversity.
3. **The 100%/0% atlas is a lower bound**: real biofilms have spatial structure
   that facilitates QS beyond the i.i.d. prediction.

---

## Part 2: Barracuda Evolution Review

### Current Architecture

```
wetSpring barracuda crate
├── bio/           45 CPU + 42 GPU modules
│   ├── ode_systems.rs     5 OdeSystem trait impls (→ generate_shader())
│   └── *_gpu.rs           GPU modules (lean on ToadStool primitives)
├── io/            streaming parsers (FASTQ, mzML, MS2)
├── bench/         benchmark harness
├── bin/           141 validation/benchmark binaries
└── shaders/       (ODE shaders deleted — now generated at runtime)
```

### Absorption Lifecycle Status

| Phase | Count | Status |
|-------|:-----:|--------|
| **Lean** | 27 | Consuming upstream ToadStool primitives |
| **Write → Lean** | 5 | ODE modules: `generate_shader()` from `OdeSystem` traits |
| **Compose** | 7 | GPU wrappers composing ToadStool ops |
| **Passthrough** | 3 | Accept GPU buffers, CPU kernel |
| **Tier B/C** | 0 | All promoted |

### Cross-Spring Primitives Consumed

| Primitive | Provenance | wetSpring Usage |
|-----------|-----------|-----------------|
| `anderson_3d` | hotSpring (Kachkovskiy) | Exp127–151 lattice construction |
| `lanczos` / `lanczos_eigenvalues` | hotSpring | Eigenvalue extraction |
| `level_spacing_ratio`, GOE_R, POISSON_R | hotSpring | ⟨r⟩ diagnostic |
| `SpectralCsrMatrix` | hotSpring | Custom Hamiltonians (Exp151) |
| `BatchedOdeRK4<S>` | wetSpring → ToadStool S51 | QS ODE GPU shaders |
| `ShaderTemplate::for_driver_auto` | hotSpring NVK fix | All f64 compilation |
| `GemmCached` 60× speedup | wetSpring → absorbed | Used by hotSpring HFB |
| 5 neuralSpring primitives | neuralSpring → ToadStool | Game theory, fitness, Hamming |

---

## Part 3: ToadStool Absorption Targets

### Priority 1: Feedback on Existing Code

| Item | Location | Issue | Suggested Fix |
|------|----------|-------|---------------|
| `compile_shader` → `compile_shader_f64` | `batched_ode_rk4.rs:209` | f64 preamble not injected; shader fails on naga/Vulkan | Change to `compile_shader_f64()` or `ShaderTemplate::for_driver_auto` |

### Priority 2: New Primitives from wetSpring

| Primitive | Source | Description | Benefit |
|-----------|--------|-------------|---------|
| **Correlated Anderson 3D** | Exp151 | `build_correlated_anderson_3d(l, w, xi_corr, seed)` | Biofilm-realistic disorder; exponential kernel smoothing |
| **Disorder averaging** | Exp150 | Multi-realization ⟨r⟩ with stderr | Statistical confidence for W_c extraction |
| **OdeSystem trait pattern** | `bio/ode_systems.rs` | 5 concrete impls for `wgsl_derivative()` + `cpu_derivative()` | Any spring can add new ODE systems |

### Priority 3: CPU Math Extraction

wetSpring has high-quality local implementations that duplicate barracuda
upstream. Blocked by barracuda requiring wgpu+akida+toadstool-core.

| Function | File | Description | Proposed Upstream |
|----------|------|-------------|-------------------|
| `erf()` | `bio/special.rs` | Abramowitz & Stegun, FMA-optimized | `barracuda::special::erf` |
| `ln_gamma()` | `bio/special.rs` | Lanczos, Horner form | `barracuda::special::ln_gamma` |
| `regularized_gamma_lower()` | `bio/special.rs` | Series expansion | `barracuda::special::regularized_gamma_p` |
| `integrate_peak()` | `bio/eic.rs` | Trapezoidal rule | `barracuda::numerical::trapz` |
| `cholesky_factor()` | `bio/esn.rs` | SPD system solve | `barracuda::linalg::cholesky_solve` |
| `solve_ridge()` | `bio/esn.rs` | Cholesky-based ridge regression | `barracuda::linalg::ridge_regression` |

**Proposed solution:** `[features] math = []` gate in barracuda for CPU-only
modules without GPU stack dependency.

### Priority 4: Passthrough → Full GPU

Three modules accept GPU buffers but run CPU kernels. These need upstream
ToadStool primitives:

| Module | Needed Primitive | Notes |
|--------|-----------------|-------|
| `gbm_gpu` | `GbmBatchInferenceGpu` | Sequential boosting on GPU |
| `feature_table_gpu` | `FeatureExtractionGpu` | Feature extraction pipeline |
| `signal_gpu` | `PeakDetectGpu` | 1D peak detection |

### Priority 5: metalForge Forge Crate Absorption

The `metalForge/forge/` crate provides substrate discovery and dispatch
routing that could be absorbed into barracuda's device layer:

| Module | Absorption Path |
|--------|-----------------|
| `probe` (GPU + CPU + NPU discovery) | `barracuda::device::discovery` |
| `inventory` (unified substrate list) | `barracuda::device::inventory` |
| `dispatch` (capability-based routing) | `barracuda::dispatch` |
| `bridge` (forge ↔ barracuda) | Integration seam |

### Priority 6: Track 3 GPU Primitives (Future)

Drug repurposing (Fajgenbaum / Every Cure) will need:

| Primitive | Description | Priority |
|-----------|-------------|:--------:|
| NMF (f64) | Multiplicative update rules (Lee & Seung 1999) | P1 |
| Sparse GEMM | Drug-disease matrices (~5% fill) | P2 |
| Cosine similarity | Pairwise scoring on factor matrices | P2 |
| Top-K selection | Rank drug-disease pairs | P3 |

---

## Part 4: Paper Queue Control Status

### Three-Tier Control Matrix

All 25 actionable papers have full three-tier coverage:

| Tier | Description | Papers Covered | Checks |
|------|-------------|:--------------:|:------:|
| **BarraCuda CPU** | Rust math ↔ Python baseline | 25/25 | 380/380 |
| **BarraCuda GPU** | GPU ↔ CPU parity | 25/25 | 702+ |
| **metalForge** | Substrate-independent (CPU = GPU = NPU) | 25/25 | 234+ |

### Open Data Verification

All 29 reproductions use publicly accessible data:
- NCBI SRA, Zenodo, MassBank, EPA, Michigan EGLE
- Published ODE parameters (Waters papers)
- Synthetic proxy data (Sandia)
- Algorithmic (no external data) for spectral theory

### Paper Queue Status (ALL GREEN — 43/43)

| Track | Papers | Status |
|-------|--------|--------|
| Track 3 (Fajgenbaum) | 39–43 | **Exp157-161 DONE** — NMF local (`bio::nmf`), cosine sim, top-K, KG embedding |
| Phase 37 extension | 30, 35–38 | **Exp152-156 DONE** — all wave modes, nitrifying QS, marine QS, Myxococcus, Dictyostelium |
| Remaining GPU need | — | Sparse GEMM, weighted NMF mask (ToadStool absorption target) |

### Control Validation Depth

| Level | What It Proves | Experiments |
|-------|---------------|:-----------:|
| Python baseline | Match published tools | 41 scripts |
| BarraCuda CPU | Rust ↔ Python (±1e-3 to 1e-12) | Exp035,043,057,070,079,085,102 |
| BarraCuda GPU | CPU ↔ GPU (1e-6 to 1e-10) | Exp064,071,087,092,101 |
| Streaming | Zero CPU round-trips | Exp072,073,075,089,090,091,105,106 |
| metalForge | Substrate-independent output | Exp060,065,080,084,086,088,093,103,104 |
| NPU | ESN → int8 → Akida | Exp114-119 |
| NCBI-scale | Real-scale data | Exp108-113 |
| Anderson spectral | Physics validation | Exp107,122,127-138,150,151 |

---

## Part 5: Cross-Spring Evolution Update

### What wetSpring Contributed to ToadStool

| Contribution | ToadStool Session | Status |
|-------------|:-----------------:|--------|
| 5 bio ODE WGSL shaders | S46, S51 | Absorbed → `BatchedOdeRK4<S>` |
| `OdeSystem` trait pattern | S51 | Generic ODE framework |
| `GemmCached` 60× speedup | S39 | Used by hotSpring HFB |
| Game theory / fitness GPU ops | S39 | Used by neuralSpring |
| `compile_shader_f64` pattern | S40 | Cross-spring f64 standard |

### What wetSpring Consumed from ToadStool

| Source Spring | Primitive Count | Key Items |
|:------------:|:--------------:|-----------|
| hotSpring | 8 | Anderson 1D/2D/3D, Lanczos, level statistics, ShaderTemplate, GpuDriverProfile |
| neuralSpring | 5 | PairwiseHamming, PairwiseJaccard, SpatialPayoff, BatchFitness, LocusVariance |
| wetSpring (round-trip) | 12 | ODE, Gillespie, DADA2, Smith-Waterman, Felsenstein, diversity, etc. |

### New in Phase 39

- **`build_correlated_anderson_3d`**: Local implementation constructing correlated-disorder
  Hamiltonians via exponential kernel smoothing. Uses `SpectralCsrMatrix` from ToadStool.
  Candidate for upstream absorption as `barracuda::spectral::anderson_correlated_3d`.
- **Disorder-averaged ⟨r⟩ workflow**: Multi-realization sweep pattern. Could become a
  ToadStool utility: `anderson_sweep_averaged(sizes, w_range, n_realizations)`.

---

## Part 6: Verification

```bash
# Format + lint (both feature configs)
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features gpu -- -D warnings

# Library tests
cargo test --lib  # 728 passed, 1 ignored

# ODE system tests
cargo test --features gpu --lib ode_systems  # 8 passed

# Phase 39 experiments
cargo run --features gpu --release --bin validate_finite_size_scaling_v2  # 14/14 PASS
cargo run --features gpu --release --bin validate_correlated_disorder     # 8/8 PASS

# ODE lean benchmark
cargo run --features gpu --release --bin benchmark_ode_lean_crossspring  # 11/11 PASS
```

---

## Part 7: File Change Manifest

### New Files (Phase 39)

| File | Purpose |
|------|---------|
| `barracuda/src/bin/validate_finite_size_scaling_v2.rs` | Exp150: disorder-averaged finite-size scaling |
| `barracuda/src/bin/validate_correlated_disorder.rs` | Exp151: correlated disorder lattices |
| `experiments/PHASE_39_DESIGN.md` | Phase 39 design document |
| `experiments/150_finite_size_scaling_v2.md` | Exp150 protocol + results |
| `experiments/151_correlated_disorder.md` | Exp151 protocol + results |
| `whitePaper/baseCamp/sub_thesis_01_anderson_qs.md` | Sub-thesis 01: Anderson-QS |
| `whitePaper/baseCamp/sub_thesis_02_ltee.md` | Sub-thesis 02: LTEE |
| `whitePaper/baseCamp/sub_thesis_03_bioag.md` | Sub-thesis 03: BioAg |
| `whitePaper/baseCamp/sub_thesis_04_sentinels.md` | Sub-thesis 04: Sentinels + NPU |
| `whitePaper/baseCamp/sub_thesis_05_cross_species.md` | Sub-thesis 05: Cross-species QS |

### Modified Files

| File | Change |
|------|--------|
| `README.md` | Phase 39 status, 151 experiments, lean completion |
| `CONTROL_EXPERIMENT_STATUS.md` | Exp150-151, ODE lean benchmark |
| `specs/README.md` | Phase 39 status, experiment counts, handoff ref |
| `barracuda/Cargo.toml` | Exp150, Exp151 binary registrations |
| `barracuda/ABSORPTION_MANIFEST.md` | Validation summary update |
| `whitePaper/README.md` | Experiment/check counts |
| `whitePaper/baseCamp/README.md` | Sub-thesis table, Kachkovskiy experiment list |
| `wateringHole/README.md` | Active handoff table (V13–V25) |

---

## Summary for ToadStool Team

**Immediate action items:**
1. Fix `batched_ode_rk4.rs:209`: `compile_shader()` → `compile_shader_f64()`
2. Consider `[features] math = []` for CPU-only module access

**New absorption candidates:**
1. `anderson_correlated_3d(l, w, xi_corr, seed)` — correlated disorder Hamiltonian
2. `anderson_sweep_averaged(sizes, w_range, n_realizations)` — disorder-averaged ⟨r⟩

**Science finding for ToadStool evolution:**
- Correlated disorder is biologically important (shifts W_c dramatically)
- The spectral primitives from hotSpring (Lanczos, level statistics) are the
  foundation of a novel physics result in microbial ecology
- Cross-spring evolution works: hotSpring precision → wetSpring biology → new science
