# wetSpring → ToadStool Absorption & Evolution Handoff V34

**Date:** February 25, 2026
**From:** wetSpring (Phase 41, life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-or-later
**Purpose:** Complete accounting of what wetSpring contributed, what it consumes,
what remains for absorption, and lessons learned that should inform barracuda's
evolution. This is the comprehensive "state of the union" handoff.

---

## Executive Summary

wetSpring has completed the full **Write → Absorb → Lean** cycle. barracuda is
now an always-on dependency (`default-features = false`), zero local WGSL shaders
remain, and zero `#[cfg(not(feature = "gpu"))]` fallback code exists. The codebase
consumes 44 upstream primitives and has contributed 13+ shaders that became shared
infrastructure. 806 tests pass (752 CPU / 759 GPU), 162 experiments validate
3,198+ checks across life science, mass spectrometry, PFAS, QS ecology, and drug
repurposing domains.

---

## Part 1: What wetSpring Contributed to ToadStool

### Shaders and primitives that originated in wetSpring

| Primitive | Contributed | ToadStool Session | Now used by |
|-----------|------------|------------------|-------------|
| `BrayCurtisF64` shader pattern | Feb 16 | S31 | neuralSpring (distance metrics), airSpring (similarity) |
| `HmmBatchForwardF64` | Feb 20 | S31d | neuralSpring (sequence modeling) |
| `AniBatchF64` | Feb 20 | S31d | All springs (genome comparison) |
| `SnpCallingF64` | Feb 20 | S31d | neuralSpring (variant calling) |
| `DnDsBatchF64` | Feb 20 | S31d | neuralSpring (selection analysis) |
| `PangenomeClassifyGpu` | Feb 20 | S31d | neuralSpring (genomics) |
| `QualityFilterGpu` | Feb 20 | S31d | All springs (FASTQ QC) |
| `Dada2EStepGpu` | Feb 20 | S31d | All springs (amplicon denoising) |
| `RfBatchInferenceGpu` | Feb 20 | S31g | neuralSpring (ensemble ML) |
| NMF (Euclidean + KL-divergence) | Feb 24 | S58 | neuralSpring (latent factors), airSpring (sensor decomposition) |
| 5× ODE bio systems (Capacitor, Cooperation, MultiSignal, Bistable, PhageDefense) | Feb 24 | S58 | All springs via `OdeSystem` trait |
| `ridge_regression` (Cholesky-based) | Feb 24 | S59 | All springs (linear solve) |
| `anderson_3d_correlated` | Feb 24 | S59 | hotSpring (condensed matter), neuralSpring (disorder) |

### Bug reports and fixes that improved ToadStool

| Issue | Where | Resolution |
|-------|-------|------------|
| SNP binding layout mismatch | `barracuda::ops::bio::snp_calling_f64` | Fixed in S31d — extra uniform binding in WGSL but not in Rust host |
| `AdapterInfo` propagation missing | `barracuda::ops::bio::quality_filter` | Fixed in S31d — adapter trim info not forwarded after GPU filtering |
| `compile_shader_f64` f64 preamble | `barracuda::device` | Fixed in S41 — batched ODE shaders needed f64 polyfill injection |
| `_f64` suffix rewriting in shader names | `ShaderTemplate::for_driver_auto` | Documented workaround: avoid `_f64` in WGSL function names to prevent false-positive rewriting |
| `OdeSystem` trait missing `wgsl_derivative()` | `barracuda::numerical::ode_generic` | wetSpring demonstrated the need; ToadStool added in S58 |

### Patterns that became shared infrastructure

| Pattern | Origin | Impact |
|---------|--------|--------|
| `OdeSystem` trait + `generate_shader()` | wetSpring bio ODE → ToadStool S58 | Any spring can define an ODE system and get GPU shaders for free |
| `Validator` harness (hardcoded expected, pass/fail, exit 0/1) | wetSpring Phase 1 | Inspired ToadStool's `ValidationHarness` (S59) |
| `#[cfg(feature = "gpu")]` dual-path pattern | wetSpring Phase 2 | Identified limitations → led to `cpu-math` feature gate (S62) |
| `barracuda always-on` architecture | wetSpring V33 (Feb 25) | Proved that `default-features = false` works: CPU-only builds get `special`, `linalg`, `numerical` without pulling wgpu |
| Cross-spring benchmark (ODE lean) | wetSpring Exp120 | Demonstrated 21-51% speedup from upstream optimization; validated cross-spring benefit |

---

## Part 2: What wetSpring Consumes from ToadStool (44 primitives)

### GPU ops (`barracuda::ops`)

| Primitive | wetSpring module | Usage |
|-----------|-----------------|-------|
| `FusedMapReduceF64` | diversity_gpu, kmd_gpu, spectral_match_gpu, stats_gpu, streaming_gpu, neighbor_joining_gpu, merge_pairs_gpu, molecular_clock_gpu, taxonomy_gpu | Shannon, Simpson, element-wise reduce, batch norms |
| `BrayCurtisF64` | diversity_gpu | Pairwise dissimilarity |
| `GemmF64` | spectral_match_gpu, gemm_cached | Matrix multiply |
| `PeakDetectF64` | signal_gpu | LC-MS peak detection (S62) |
| `TranseScoreF64` | validate_knowledge_graph_embedding | GPU TransE scoring (S60) |
| `BatchToleranceSearchF64` | tolerance_search | PFAS m/z batch search |

### GPU bio (`barracuda::ops::bio`)

| Primitive | wetSpring module |
|-----------|-----------------|
| `SmithWatermanGpu` | alignment (via barracuda) |
| `GillespieGpu` | gillespie (via barracuda) |
| `TreeInferenceGpu` | decision_tree, gbm_gpu |
| `FelsensteinGpu` | felsenstein, bootstrap, placement |
| `HmmBatchForwardF64` | hmm_gpu |
| `AniBatchF64` | ani_gpu |
| `SnpCallingF64` | snp_gpu |
| `DnDsBatchF64` | dnds_gpu |
| `PangenomeClassifyGpu` | pangenome_gpu |
| `QualityFilterGpu` | quality_gpu |
| `Dada2EStepGpu` | dada2_gpu |
| `RfBatchInferenceGpu` | random_forest_gpu |
| `BatchReconcileGpu` | reconciliation_gpu |
| `KmerHistogramGpu` | kmer_gpu, derep_gpu |
| `UniFracPropagateGpu` | unifrac_gpu |

### GPU infrastructure

| Primitive | Usage |
|-----------|-------|
| `BatchedOdeRK4::<S>::generate_shader()` | 5 ODE GPU wrappers |
| `BatchedEighGpu` | PCoA eigendecomposition |
| `WgpuDevice`, `GpuDriverProfile` | Device management |
| `Fp64Strategy` | f64 WGSL emulation on f32-only GPUs |
| `ShaderTemplate::for_driver_auto` | Driver-aware shader compilation |

### CPU math (always-on, `barracuda::special` / `linalg` / `numerical`)

| Function | wetSpring module |
|----------|-----------------|
| `erf(f64)` | special.rs → pangenome, normal_cdf |
| `ln_gamma(f64)` | special.rs → dada2 Poisson p-values |
| `regularized_gamma_p(a, x)` | special.rs → dada2 incomplete gamma |
| `ridge_regression(...)` | esn.rs → ESN reservoir readout |
| `trapz(y, x)` | eic.rs → EIC peak integration |
| NMF (`nmf::nmf()`) | validate_repodb_nmf, validate_nmf_drug_repurposing, validate_matrix_pharmacophenomics |

### Spectral theory (`barracuda::spectral`)

| Function | Usage |
|----------|-------|
| `anderson_2d`, `anderson_3d`, `anderson_3d_correlated` | QS-disorder ecology (20+ binaries) |
| `anderson_hamiltonian`, `find_all_eigenvalues` | Direct diagonalization |
| `lanczos`, `lanczos_eigenvalues` | Sparse eigenvalues (Anderson, spectral theory) |
| `level_spacing_ratio` | GOE/Poisson transition detection |
| `lyapunov_exponent` | Localization length |
| `almost_mathieu_hamiltonian` | Hofstadter butterfly |
| `GOE_R`, `POISSON_R` | Universal constants |

### Cross-spring ops (neuralSpring origin)

| Primitive | Usage |
|-----------|-------|
| `PairwiseHammingGpu` | SNP-based strain distance |
| `PairwiseJaccardGpu` | Gene presence/absence similarity |
| `SpatialPayoffGpu` | Spatial cooperation games |
| `BatchFitnessGpu` | Evolutionary simulations |
| `LocusVarianceGpu` | FST per-locus variance |
| `graph_laplacian` | Community network analysis |
| `effective_rank` | Gene expression rank |
| `numerical_hessian` | ODE sensitivity analysis |
| `boltzmann_sampling` | Parameter sweep MCMC |
| `belief_propagation_chain` | Network inference |
| `disordered_laplacian` | Disordered graph analysis |

---

## Part 3: What We Haven't Used Yet (Absorption Opportunities)

These ToadStool primitives exist but are not yet wired into wetSpring:

### High potential

| Primitive | Module | Why wetSpring should use it |
|-----------|--------|---------------------------|
| `SparseGemmF64` | `ops::sparse_gemm_f64` | Drug-disease matrices are ~5% fill; sparse NMF would be significantly faster |
| `BandwidthTier` | `dispatch::config` | metalForge dispatch should consider PCIe bandwidth for data-heavy workloads |
| `dispatch_with_transfer_cost` | `dispatch` | Bandwidth-aware CPU/GPU routing for streaming pipelines |
| `ComputeDispatch` builder | `device::compute_pipeline` | Replace 6× manual pipeline setup in GPU wrappers with 3-line builder chains |
| `KmdGroupingF64` | `ops::kmd_grouping_f64` | Replace local KMD homologue grouping with upstream GPU implementation |
| `BatchToleranceSearchF64` (batch) | `ops::batch_tolerance_search_f64` | Already partially wired; could expand to batch PFAS screening |

### Medium potential

| Primitive | Module | Why |
|-----------|--------|-----|
| `barracuda::esn_v2::ESN` | `esn_v2` | Replace local ESN with upstream GPU/NPU-capable version |
| `cosine_similarity_f64` | `ops` | Single-pair cosine (currently inlined in binaries) |
| `rk45_solve` | `numerical` | Adaptive ODE integration (currently only fixed RK4) |
| `chi_squared` | `special` | Statistical tests (currently not needed) |
| `HillFunctionF64` | `ops` | Hill function for dose-response curves |

### Low priority (no current need)

`latin_hypercube`, `sobol_sequence`, `bessel_*`, `hermite`, `laguerre`,
`screened_coulomb`, `hofstadter_butterfly`, most NN/vision/audio/FHE ops.

---

## Part 4: Lessons Learned (for ToadStool evolution)

### 1. `cpu-math` feature gate was the right call

Making `barracuda::special`, `barracuda::linalg`, and `barracuda::numerical`
available without `wgpu` was transformative. Before this, every spring had to
maintain local fallbacks for every math function. The `cpu-math` gate eliminated
~177 lines of dual-path code in wetSpring alone. **Recommendation**: ensure all
future CPU-only math goes behind `default-features = false` and doesn't pull in
`wgpu`, `pollster`, or GPU-only deps.

### 2. `OdeSystem` trait is the gold standard for GPU-codegen patterns

The `OdeSystem` trait + `generate_shader()` pattern is the most successful
abstraction in the codebase. Springs define a trait impl (20 lines), get CPU
integration AND GPU WGSL shader for free. **Recommendation**: replicate this
pattern for other parametric GPU compute (e.g., PDE solvers, stencil operations,
cellular automata). The key ingredients are:
- Trait methods for derivative, clamping, initial conditions
- WGSL generation from trait methods
- CPU/GPU integration from the same trait

### 3. Upstream integrators outperform local ones (21-51%)

ToadStool's `integrate_cpu()` is 21-51% faster than wetSpring's local
implementations of the same ODE systems, because ToadStool optimizes across all
springs' usage patterns. **This validates the Write → Absorb → Lean cycle**: the
shared version is better than any individual spring's version.

| System | Local µs | Upstream µs | Speedup |
|--------|---------|-------------|---------|
| Capacitor | 1,165 | 774 | **1.51×** |
| Cooperation | 837 | 623 | **1.34×** |
| MultiSignal | 1,589 | 1,200 | **1.32×** |
| Bistable | 1,715 | 1,415 | **1.21×** |
| PhageDefense | 85 | 61 | **1.39×** |

### 4. GPU TransE parity is excellent (1.78e-15 max diff)

The `TranseScoreF64` GPU implementation achieves f64-level parity with CPU
scoring across 538 triples. No precision loss from GPU computation. This validates
the `Fp64Strategy::Native` path for knowledge graph operations.

### 5. Avoid `_f64` in WGSL function names

`ShaderTemplate::for_driver_auto()` does string replacement on `_f64` suffixes
to inject f32 fallbacks. If a WGSL function is named `my_kernel_f64`, it gets
mangled. Use `my_kernel` or `my_kernel_double` instead. This has bitten multiple
springs.

### 6. `catch_unwind` is valuable for GPU binding mismatches

wetSpring's `snp_gpu` discovered that wgpu binding layout mismatches cause
panics, not errors. Wrapping GPU dispatch in `std::panic::catch_unwind` with
a CPU fallback is a battle-tested pattern that other springs should adopt.

### 7. barracuda always-on simplifies CI

With barracuda as an always-on dep, `cargo test` (no features) exercises the
full CPU math path. `cargo test --features gpu` adds GPU tests. No more
"did someone forget to test without GPU?" — the default build always tests
barracuda's CPU math through the spring's own test suite.

### 8. Cross-spring evolution is real and measurable

hotSpring's f64 WGSL emulation → ToadStool's `Fp64Strategy` → used by
wetSpring's ODE shaders. wetSpring's ODE `OdeSystem` trait → ToadStool's
`generate_shader()` → used by neuralSpring for evolutionary dynamics.
neuralSpring's graph primitives → used by wetSpring for community analysis.
The cross-pollination is not theoretical; it's measured in speedups, reduced
code, and wider GPU coverage.

---

## Part 5: Recommended ToadStool Evolution Priorities

### Priority 1: `ComputeDispatch` builder adoption across springs

wetSpring has 6 GPU modules with ~80 lines of manual pipeline boilerplate each.
`ComputeDispatch::new(device, "Bistable").shader(...).uniform(0, &buf)...submit()`
would replace this with 3-5 lines. Ship examples; springs will adopt.

### Priority 2: Sparse GEMM integration into NMF

Drug-disease matrices are ~5% fill. `SparseGemmF64` exists (S60) but NMF still
uses dense `matmul`. An `NmfConfig { sparse: true }` option that dispatches to
sparse GEMM when the input CSR matrix is sparse enough would be a big win for
pharmacogenomics workloads.

### Priority 3: `BandwidthTier` integration into `dispatch_with_transfer_cost`

metalForge needs bandwidth-aware routing. Currently routing is capability-based
only. Adding PCIe/USB bandwidth estimation to the dispatch decision would prevent
sending large datasets to slow-bus GPUs.

### Priority 4: Adaptive ODE (`rk45_solve`) trait pattern

`BatchedOdeRK4` is fixed-step. For stiff systems (PhageDefense near equilibrium),
adaptive stepping would reduce compute by 10-100×. A `BatchedOdeRK45::<S>` with
the same `OdeSystem` trait would be immediately useful to all springs.

### Priority 5: Document the `cpu-math` contract

Springs now depend on `barracuda::special::*`, `barracuda::linalg::*`, and
`barracuda::numerical::*` being available without `gpu`. If any of these move
behind a feature gate, CPU-only builds break across all springs. This contract
should be documented in barracuda's README.

---

## Validation

```
cargo fmt    — clean
cargo clippy — 0 warnings (pedantic + nursery)
cargo test   — 752 passed (CPU-only)
cargo test --features gpu — 759 passed
validate_barracuda_cpu_v8 — 84/84 PASS
validate_repodb_nmf       — 9/9 PASS
validate_knowledge_graph_embedding — 9/9 PASS (GPU TransE: 1.78e-15 max diff)
benchmark_ode_lean_crossspring — 11/11 PASS (21-51% upstream speedup)
```

## Status

| Metric | Value |
|--------|-------|
| ToadStool alignment | **S62** |
| Primitives consumed | **44** (barracuda always-on) |
| Primitives contributed | **13+** shaders/ops |
| Local WGSL shaders | **0** |
| Dual-path fallback code | **0** |
| Tests | 752 CPU / 759 GPU / 806 total |
| Experiments | 162 |
| Validation checks | 3,198+ |
| Write → Absorb → Lean | **Complete** |
