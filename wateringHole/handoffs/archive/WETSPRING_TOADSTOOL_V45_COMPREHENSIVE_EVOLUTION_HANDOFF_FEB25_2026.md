# wetSpring → ToadStool V45 Handoff: Comprehensive Evolution & Absorption Report

**Date:** February 25, 2026
**Phase:** 48 (V45 — comprehensive evolution handoff)
**Primitives consumed:** 53 + 2 BGL helpers + 1 WGSL extension (Write phase)
**Evolution requests:** 8/9 P0-P3 delivered
**Tests:** 819 lib + 47 forge + 32 integration = 898 total
**Coverage:** 96.78% llvm-cov
**Experiments:** 169 (3,300+ checks, ALL PASS)
**Quality:** 0 clippy warnings (pedantic + nursery), 0 fmt diffs, 0 Passthrough

---

## Executive Summary

This handoff provides ToadStool/BarraCuda with a complete picture of how wetSpring
uses, validates, and benefits from barracuda — and what wetSpring has learned that
is relevant to ToadStool's evolution. It covers:

1. **Complete dependency surface** — every barracuda import, counted by module
2. **Cross-spring evolution provenance** — where every primitive came from and who benefits
3. **Evolution opportunities** — what ToadStool should build next to help all springs
4. **Lessons learned** — patterns, pitfalls, and architecture decisions from 169 experiments
5. **What remains local (and why)** — things wetSpring keeps because they serve a different purpose
6. **Bug reports and workarounds** — naga issues, driver quirks, API friction

---

## Part 1: Complete Dependency Surface

### 1.1 Cargo.toml

```toml
barracuda = { path = "../../phase1/toadstool/crates/barracuda", default-features = false }
```

`default-features = false` gives CPU-only math. `barracuda/gpu` feature activates
all GPU ops, spectral, linalg graph, device, etc.

### 1.2 Module Usage by File Count

| Module | Files | Heaviest Use |
|--------|------:|--------------|
| `barracuda::ops` | 42 | FMR (12+), GEMM, BrayCurtis, BatchedEigh, PeakDetect, TransE, SparseGemm |
| `barracuda::device` | 31 | WgpuDevice, TensorContext, GpuDriverProfile, BGL entries |
| `barracuda::spectral` | 21 | Anderson 2D/3D, Lanczos, level_spacing_ratio, find_w_c |
| `barracuda::special` | 17 | erf, ln_gamma, regularized_gamma_p (always-on CPU) |
| `barracuda::linalg` | 11 | NMF, ridge_regression, graph_laplacian, belief_propagation |
| `barracuda::numerical` | 11 | trapz, BatchedOdeRK4, 5 OdeSystem impls, numerical_hessian |
| `barracuda::stats` | 3 | norm_cdf, pearson_correlation, variance |
| `barracuda::sample` | 1 | boltzmann_sampling |

### 1.3 Consumed Primitives (53 total)

**CPU Math (always-on, 7 functions):**
- `erf`, `ln_gamma`, `regularized_gamma_p` — barracuda::special
- `norm_cdf`, `pearson_correlation` — barracuda::stats (V43-V44 rewire)
- `trapz` — barracuda::numerical
- `ridge_regression` — barracuda::linalg

**GPU Bio Ops (15 shaders):**
- `FelsensteinGpu`, `HmmBatchForwardF64`, `SmithWatermanGpu`, `GillespieGpu`
- `AniBatchF64`, `DnDsBatchF64`, `SnpCallingF64`, `PangenomeClassifyGpu`
- `QualityFilterGpu`, `Dada2EStepGpu`, `RfBatchInferenceGpu`
- `TreeInferenceGpu`, `KmerHistogramGpu`, `TaxonomyFcGpu`, `UniFracPropagateGpu`

**GPU Core Ops (11 primitives):**
- `FusedMapReduceF64` (FMR) — universal reduce (Shannon, Simpson, variance, mean)
- `GemmF64` / `GemmCachedF64` — tiled matrix multiply
- `SparseGemmF64` — sparse GEMM
- `TranseScoreF64` — knowledge graph embedding
- `PeakDetectF64` — LC-MS signal detection
- `BatchedEighGpu` — eigendecomposition for PCoA
- `BrayCurtisF64` — ecological distance
- `WeightedDotF64` — kriging weights
- `KrigingF64` — spatial interpolation
- `CorrelationF64` / `CovarianceF64` / `VarianceF64` — GPU stats

**GPU Cross-Spring (neuralSpring → ToadStool, 8 ops):**
- `PairwiseHammingGpu`, `PairwiseJaccardGpu` — SNP/gene distance
- `SpatialPayoffGpu` — cooperation game theory
- `BatchFitnessGpu`, `LocusVarianceGpu` — population genetics
- `graph_laplacian`, `disordered_laplacian`, `effective_rank` — network spectral
- `belief_propagation_chain`, `boltzmann_sampling` — PGM/MCMC

**GPU Spectral (hotSpring → ToadStool, 8 primitives):**
- `anderson_hamiltonian`, `anderson_2d`, `anderson_3d`, `anderson_3d_correlated`
- `lanczos`, `lanczos_eigenvalues`, `find_all_eigenvalues`
- `level_spacing_ratio`, `lyapunov_exponent`, `find_w_c`
- `almost_mathieu_hamiltonian`, `SpectralCsrMatrix`

**ODE Trait (5 systems via `BatchedOdeRK4::generate_shader()`):**
- `CooperationOde`, `BistableOde`, `MultiSignalOde`, `CapacitorOde`, `PhageDefenseOde`

**Infrastructure (2 BGL helpers):**
- `storage_bgl_entry`, `uniform_bgl_entry`

**Local WGSL (1 Write-phase extension):**
- `diversity_fusion_f64.wgsl` — ready for absorption as `ops::bio::diversity_fusion`

---

## Part 2: Cross-Spring Evolution Provenance

### 2.1 Where Things Evolved to Be Helpful

The ecoPrimals ecosystem demonstrates genuine cross-spring evolution: primitives
written for one scientific domain get absorbed into ToadStool and become available
to all springs. This is the "rising tide lifts all boats" pattern.

#### hotSpring (computational physics) → wetSpring (biology)

| What hotSpring Built | Why | How wetSpring Uses It |
|---------------------|-----|----------------------|
| f64 polyfills (naga workarounds) | FP32 GPU cores + f64 emulation | All 42 GPU modules need these |
| `PeakDetectF64` | LC-MS chromatographic peak detection | Signal processing, drug repurposing pipeline |
| `BatchedEighGpu` (NAK-optimized) | Eigendecomposition for molecular dynamics | PCoA ordination in all ecology pipelines |
| Anderson 2D/3D Hamiltonian + Lanczos | Quantum condensed matter (Abrahams 1979) | QS-disorder coupling in biofilms (Kachkovskiy sub-thesis) |
| `find_w_c` | Metal-insulator phase transition | Critical disorder threshold for biofilm quorum sensing |
| `level_spacing_ratio` | GOE-Poisson transition (RMT) | Classify microbial communities as "active" vs "suppressed" |
| Lyapunov exponent (transfer matrix) | Localization length | Predict cooperation range in structured environments |
| DF64 core-streaming | Double-double arithmetic on FP32 cores | Future: precision-sensitive ecology simulations |

**Key insight:** hotSpring's decades of precision physics infrastructure (Kachkovskiy
spectral theory, NAK eigensolvers, f64 polyfills) provides wetSpring with analytical
tools that no biology group would have built on their own. The Anderson localization
framework — originally for electron wavefunctions — predicts quorum sensing behavior
in 28 NCBI biomes (Exp126, 90 checks).

#### wetSpring (biology) → ToadStool → all springs

| What wetSpring Built | Why | Who Benefits |
|---------------------|-----|-------------|
| ODE trait + `generate_shader()` | 5 QS bio systems needed GPU ODE integration | airSpring (Richards PDE), neuralSpring (population dynamics), hotSpring (reaction kinetics) |
| 15 bio GPU shaders | 16S metagenomics pipeline on GPU | neuralSpring (population genetics), groundSpring (environmental monitoring) |
| `ridge_regression` | ESN readout training (Tikhonov) | All springs with reservoir computing |
| `trapz` | Trapezoidal integration for EIC peak areas | All springs with signal processing |
| `erf`, `ln_gamma`, `regularized_gamma_p` | Special functions for statistical testing | All springs |
| Tolerance constant pattern (77 constants) | Named tolerances with provenance citations | ToadStool adopted 12 named constants (S52) |
| `GemmCached` pattern | Caching A-transpose in GEMM | ToadStool `GemmCachedF64` (60× speedup from wetSpring insight) |
| `GPU_DISPATCH_THRESHOLD = 10_000` | Below 10K elements, CPU is faster than GPU dispatch overhead | ToadStool `DeviceCapabilities::gpu_dispatch_threshold()` |

**Key insight:** wetSpring's domain-specific validation found the `GemmCachedF64`
optimization pattern (cache A-transpose, reuse across samples) and the 10K dispatch
threshold — both now universal across all springs.

#### neuralSpring (ML/population) → wetSpring (biology)

| What neuralSpring Built | Why | How wetSpring Uses It |
|------------------------|-----|----------------------|
| `PairwiseHammingGpu` | Genome-wide SNP distance | Strain typing in 16S pipelines |
| `PairwiseJaccardGpu` | Gene presence/absence similarity | Pan-genome ecology |
| `SpatialPayoffGpu` | Spatial prisoner's dilemma | Cooperation game theory in biofilms |
| `BatchFitnessGpu` | Multi-objective EA fitness | Evolutionary simulation validation |
| `LocusVarianceGpu` | Per-locus FST decomposition | Population genetics QC |
| `graph_laplacian` | Graph Laplacian from adjacency | Community network spectral analysis |
| `disordered_laplacian` | Anderson disorder on networks | QS-disorder on social graphs |
| `belief_propagation_chain` | Chain PGM forward pass | Hierarchical taxonomy classification |
| `boltzmann_sampling` | MCMC thermal sampling | ODE parameter optimization |

**Key insight:** neuralSpring's ML/population genetics primitives solve problems
that wetSpring encounters in ecology. The `graph_laplacian` + `disordered_laplacian`
combination lets wetSpring model QS-disorder on actual microbial community networks,
not just lattices.

---

## Part 3: Evolution Opportunities for ToadStool

### 3.1 Priority Requests (P0-P2)

| # | Request | Rationale | Impact |
|:-:|---------|-----------|--------|
| P0 | **Absorb `diversity_fusion_f64.wgsl`** as `ops::bio::diversity_fusion` | Fused Shannon + Simpson + evenness in one pass. 18/18 parity checks (Exp167). Only remaining local WGSL. | wetSpring → 0 local WGSL |
| P1 | **Export `dot` and `l2_norm` as CPU helpers** from `barracuda::linalg` | wetSpring has 15+ binaries using local 5-line `dot`/`l2_norm`. Simple CPU functions that should live upstream. | All springs get CPU vector math |
| P2 | **Fix `BatchedEighGpu` naga validation** | Fails naga shader validation; wetSpring uses `catch_unwind` workaround. PCoA and spectral analysis affected. | Unblock eigendecomposition for all springs |

### 3.2 Medium Priority (P3-P5)

| # | Request | Rationale |
|:-:|---------|-----------|
| P3 | **`GemmCachedF64` streaming B-matrix** | wetSpring `streaming_gpu.rs` needs B-matrix to change per-sample; current API assumes stable B. Design: session-cached A + streaming B dispatch. |
| P4 | **CPU reference implementations for GPU-only ops** | wetSpring maintains `cpu_locus_variance_rowmajor` as validation baseline. If barracuda exported CPU reference impls for GPU-only ops, springs wouldn't need local reimplementations. |
| P5 | **KmdGroupingF64 GPU post-pass** | CPU post-pass is O(N²); for N>10K PFAS screening, a GPU all-pairs KMD comparison is needed. |

### 3.3 Low Priority / Future (P6-P8)

| # | Request | Rationale |
|:-:|---------|-----------|
| P6 | **`barracuda::stats` module expansion** | wetSpring could use `bootstrap_ci`, `chi2_decomposed`, `norm_ppf` — all available upstream but not yet consumed. Document as "available for springs" in barracuda docs. |
| P7 | **DF64 bio shaders** | When DF64 core-streaming matures, wetSpring bio shaders (Felsenstein, HMM) could benefit from double-double precision on consumer FP32 GPUs. |
| P8 | **`BatchPairReduceF64` outer batch** | For B>1 workloads, the outer batch loop is sequential; a third dispatch dimension would parallelize. |

---

## Part 4: Lessons Learned (Relevant to ToadStool Evolution)

### 4.1 Architecture Patterns That Work

**1. `default-features = false` for CPU-only builds:**
wetSpring's barracuda dependency with `default-features = false` gives CPU-only math
without pulling wgpu/pollster/bytemuck. This eliminated all `#[cfg(not(feature = "gpu"))]`
dual-path fallback code. **Recommendation:** Document this pattern in barracuda's README
for all springs to adopt.

**2. ODE trait + `generate_shader()` for domain ODE systems:**
The `OdeSystem` trait with `fn derivatives()` and `fn shader_source()` lets domain
scientists define ODE systems in Rust and get GPU WGSL shaders automatically.
Five wetSpring bio systems (cooperation, bistable, multi-signal, capacitor, phage defense)
went through the full Write → Absorb → Lean cycle via this pattern. The trait is
now reusable by airSpring (Richards PDE) and neuralSpring (population dynamics).

**3. Named tolerance constants with provenance:**
wetSpring's 77 named tolerance constants (e.g., `SHANNON_SIMULATED = 0.3`,
`EXACT_F64 = 1e-15`, `COMMUNITY_BETA_DIVERSITY = 0.05`) with provenance citations
eliminated all ad-hoc "magic number" tolerances. ToadStool adopted 12 named constants
in S52. **Recommendation:** Every spring should have a `tolerances.rs` module.

**4. Validation binary pattern:**
wetSpring's 158 binaries follow a consistent pattern: `Validator::new(name)` →
`v.section()` → `v.check()` / `v.check_pass()` / `v.check_count()` → `v.finish()`.
This is distinct from but complementary to barracuda's `ValidationHarness` (which
uses `check_abs`/`check_rel`/`require!`). Both serve valid purposes.

### 4.2 Patterns That Didn't Work (or Need Refinement)

**1. Early Passthrough tier was a code smell:**
wetSpring initially had 3 "Passthrough" modules (gbm, feature_table, signal) that
accepted GPU buffers but ran CPU kernels. These were all promoted (V40) — 2 to Compose
tier, 1 to Lean. **Lesson:** If a module isn't using GPU, it should be CPU-only from
the start.

**2. Inline W_c loop duplication:**
Before V44, 4 validation binaries had local `find_last_downward_crossing` functions
that duplicated `barracuda::spectral::find_w_c`. The only difference was the input
type (`(f64, f64)` tuples vs `AndersonSweepPoint` structs). **Lesson:** Small
type-conversion overhead is worth paying to avoid code duplication.

**3. `unwrap_or()` for barracuda `Result` types:**
barracuda's `ln_gamma`, `regularized_gamma_p`, `trapz`, `pearson_correlation` all
return `Result<f64, BarracudaError>`. wetSpring uses `unwrap_or(f64::NAN)` or
`unwrap_or(0.0)` at call sites. **Suggestion:** Consider offering infallible variants
(e.g., `erf` is already infallible) or a `_unchecked` suffix for validated inputs.

### 4.3 Driver/Hardware Findings

**1. RTX 4070 f64 polyfill requirement:**
Consumer GPUs have ~1/32 f64 throughput vs f32. hotSpring's polyfills (naga workarounds)
are essential for all f64 WGSL shaders. wetSpring validated this across all 42 GPU
modules.

**2. GPU dispatch threshold = 10,000 elements:**
Below this threshold, CPU is faster due to dispatch overhead (buffer creation,
shader compilation, readback). Now encoded in `DeviceCapabilities::gpu_dispatch_threshold()`.

**3. naga `BatchedEighGpu` issue:**
The Jacobi eigendecomposition shader fails naga validation with "invalid function call"
when inner loop exceeds complexity threshold. wetSpring works around this with
`std::panic::catch_unwind`. ToadStool should investigate the naga backend.

---

## Part 5: What Stays Local (and Why)

| Local Item | Why It Stays | Upstream Alternative |
|-----------|-------------|---------------------|
| `validation::Validator` | Different API from `ValidationHarness` (check, check_count, finish vs check_abs, check_rel, require!). 158 binaries use it. | `barracuda::validation::ValidationHarness` (available, not consumed) |
| `tolerances.rs` (77 constants) | Complementary to `barracuda::tolerances` struct system. wetSpring needs flat `f64` constants; barracuda needs `Tolerance { abs_tol, rel_tol, justification }`. | `barracuda::tolerances` (12 named constants, different shape) |
| `special::{dot, l2_norm}` | No upstream CPU equivalent. 15+ binaries use these. | **P1 request:** Export from `barracuda::linalg` |
| 6 `hill()` functions | CPU ODE derivative helpers. GPU equivalents generated by `BatchedOdeRK4::generate_shader()`. | `barracuda::ops::hill_f64::HillFunctionF64` (GPU-only, different purpose) |
| `bio/rarefaction_gpu.rs` inline variance | Domain-specific SE calculation, not general stats. | Would over-engineer to use `barracuda::stats::variance` |
| `cpu_locus_variance_rowmajor` | CPU validation reference for `LocusVarianceGpu`. | **P4 request:** Export CPU reference impls |

---

## Part 6: Bug Reports and Workarounds

### 6.1 Active Bugs

| Bug | Location | Workaround | Status |
|-----|----------|-----------|--------|
| `BatchedEighGpu` naga validation | `bio/pcoa_gpu.rs` | `catch_unwind` around shader compilation | **OPEN** — P2 request |
| `log_f64` precision | Found in wetSpring validation | ToadStool fixed in S41 | **RESOLVED** |

### 6.2 Driver-Specific Workarounds

| Issue | GPU | Workaround |
|-------|-----|-----------|
| f64 polyfills | RTX 4070 (consumer) | hotSpring naga workarounds |
| Dispatch overhead | All GPUs | GPU_DISPATCH_THRESHOLD = 10,000 |
| PeakDetect compile | RTX 4070 | `compile_shader_f64` helper |

### 6.3 Pre-existing Documentation Warnings

Three `cargo doc` warnings (`public doc links to private item`) in `ncbi_data.rs` —
pre-existing from V42 refactoring into private submodules. Not blocking.

---

## Part 7: Quality Evidence

### 7.1 Test Suite

| Suite | Count | Result |
|-------|------:|--------|
| Library tests | 819 | PASS (1 ignored) |
| Forge tests | 47 | PASS |
| Integration/doc tests | 32 | PASS |
| **Total** | **898** | **ALL PASS** |

### 7.2 Code Quality

| Gate | Status |
|------|--------|
| `cargo fmt --check` | 0 diffs |
| `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo test --lib` | 819/819 pass |
| `cargo doc --no-deps` | 3 pre-existing warnings (not introduced) |
| llvm-cov | 96.78% |
| Named tolerances | 77/77 (0 ad-hoc) |
| Passthrough modules | 0 (all promoted) |

### 7.3 Validation Experiments

| Category | Experiments | Checks | Representative |
|----------|:----------:|:------:|---------------|
| CPU baseline | 59 | 1,476 | Exp001-059 |
| GPU acceleration | 66 | 702 | Exp060-087, Exp091-100 |
| Pure GPU streaming | 10 | 152 | Exp091, Exp072-075 |
| metalForge cross-substrate | 25 | 650+ | Exp101-106, Exp157-165 |
| NPU reservoir | 6 | 59 | Exp114-119 |
| NCBI-scale hypothesis | 31 | 496 | Exp107-113, Exp121-156 |
| Cross-spring evolution | 3 | 46 | Exp094, Exp168, Exp169 |
| **Total** | **169** | **3,300+** | |

---

## Part 8: Recommendations for ToadStool Evolution

### 8.1 Immediate (next session)

1. **Absorb `diversity_fusion_f64.wgsl`** — 18/18 parity validated, structured for `ops::bio::diversity_fusion`. This closes the last open P0 request (8/9 → 9/9).
2. **Update ABSORPTION_TRACKER** with V40-V44 items (V43 normal_cdf delegation, V44 find_w_c/pearson_correlation consumption, Exp169 cross-spring benchmark).

### 8.2 Near-term (next 2-3 sessions)

3. **Export `dot` and `l2_norm` CPU helpers** from `barracuda::linalg` — trivial implementation, immediate value for all springs.
4. **Investigate `BatchedEighGpu` naga issue** — eigendecomposition is critical for PCoA across multiple springs.
5. **Document `default-features = false` pattern** in barracuda README — proven by wetSpring to eliminate all dual-path fallback code.

### 8.3 Strategic

6. **CPU reference impls for GPU-only ops** — reduces spring-local test code.
7. **DF64 bio shader path** — when DF64 matures, Felsenstein and HMM would benefit.
8. **Infallible variants** for checked math — `erf` is already infallible; `ln_gamma_unchecked` etc. would reduce `unwrap_or` boilerplate.

---

## Part 9: Acceptance Criteria

- [x] 53 primitives consumed and validated
- [x] 0 local code duplicating upstream (V44 rewire complete)
- [x] 0 Passthrough modules
- [x] 0 clippy warnings
- [x] 0 fmt diffs
- [x] 898 tests pass
- [x] 169 experiments, 3,300+ checks
- [x] Exp169 12/12 PASS with cross-spring provenance
- [x] All handoff docs current (V44 + V45)
- [x] Cross-spring evolution documented with provenance
- [x] Evolution opportunities filed (P0-P8)
- [x] Bug reports documented (BatchedEighGpu, log_f64)
- [x] Architecture lessons captured
