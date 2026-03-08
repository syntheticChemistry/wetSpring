# wetSpring → ecoPrimals: Cross-Spring Modern S86 Validation + Benchmark

**Date:** March 2, 2026
**Version:** V92F
**ToadStool:** S86 (`2fee1969`) — 264 ComputeDispatch ops
**Experiment:** Exp297 — 46/46 checks PASS
**Hardware:** NVIDIA GeForce RTX 4070 (Fp64Strategy::Hybrid, Precision::Df64)

---

## What happened

Completed the cross-spring evolution validation — proving that all 5 springs
plus wateringHole contribute shaders and primitives to the shared ToadStool
compute substrate, and that those contributions flow bidirectionally.

### Cross-Spring Shader Evolution Map

| Origin Spring | → ToadStool Module | Back to Springs | Sessions |
|--------------|-------------------|----------------|----------|
| **hotSpring** | DF64 precision, NVK workarounds, Anderson spectral | ALL springs use DF64; neuralSpring uses spectral phase | S58→S80 |
| **wetSpring** | Bio diversity, ODE (5 systems), alignment, phylo | neuralSpring brain diversity; groundSpring ecological | V6→S82 |
| **neuralSpring** | GemmF64, graph linalg, AlphaFold2 (17 shaders), HMM | wetSpring NMF via GEMM; airSpring via Laplacian | V64→S72 |
| **airSpring** | Hydrology (6 ET₀), seasonal pipeline, Nelder-Mead | groundSpring soil moisture; wetSpring environmental | V039→S81 |
| **groundSpring** | Bootstrap, Wright-Fisher, InterconnectTopology | ALL springs use bootstrap; neuralSpring uses topology | V54→S81 |
| **wateringHole** | Boltzmann, Sobol, LHS, chi-squared batch | ALL springs use sampling; airSpring uses chi-squared | V69→S80 |

### Key insight: bidirectional evolution

Shaders don't just flow *into* ToadStool — they flow *through* it. Examples:
- hotSpring's DF64 precision layer powers wetSpring's bio diversity GPU ops
- wetSpring's diversity fusion shader is used by neuralSpring's brain module
- neuralSpring's GemmF64 powers wetSpring's NMF drug repurposing pipeline
- airSpring's hydrology methods feed into groundSpring's soil moisture models
- wateringHole's Boltzmann sampling is used by all springs for optimization

---

## Changes made

### New binary: `validate_cross_spring_modern_s86`
- GPU validation + benchmark, 12 sections, 46 checks
- Tests every spring's contribution with provenance annotations
- Performance benchmark table sorted by execution time

### API fix: `rarefaction_gpu.rs`
- `BatchedMultinomialGpu::sample` signature changed in ToadStool S86
- `seeds` parameter: `&mut Vec<u32>` → `Option<&mut Vec<u32>>`
- New required parameter: `BatchedMultinomialConfig { cumulative_probs, seed }`

---

## Benchmark Results (RTX 4070, release)

| Operation | Time (ms) | Origin | Notes |
|-----------|----------|--------|-------|
| GEMM 128×128 GPU | 18.1 | neuralSpring | DF64 double-float on 5888 FP32 cores |
| GEMM 128×128 CPU | 3.7 | neuralSpring | Single-threaded reference |
| GemmCached 64×32×16 | 13.5 | wetSpring | B-matrix cached on device |
| DiversityFusion GPU | 74.6 | wetSpring | Includes pipeline compile (first run) |
| DiversityFusion CPU | 0.004 | wetSpring | 500 taxa, single-threaded |
| BrayCurtis 20×200 | 2.7 | wetSpring | Condensed distance matrix |
| Anderson 1D n=1000 | 348 | hotSpring | Dense eigensolve, CPU |
| Lanczos 200 steps | 9.8 | hotSpring | Sparse iterative, CPU |
| Bootstrap 200×50k | 24.2 | groundSpring | 50k resamples |
| Boltzmann 5k×2D | 0.26 | wateringHole | Metropolis MCMC |
| Sobol 10k×5D | 0.30 | wateringHole | Low-discrepancy sequence |

### GPU performance notes
- DF64 on RTX 4070: Hybrid strategy routes f64 through 5888 FP32 cores via
  double-float pairs. Max GEMM error vs CPU: 1.75e-5 (expected for DF64).
- First GPU dispatch includes wgpu pipeline compilation (~50-70 ms overhead).
  Subsequent dispatches are fast. DiversityFusion includes this overhead.
- BrayCurtis GPU excels at pairwise distance matrices (O(n²) parallelism).

---

## Quality gates

| Gate | Status |
|------|--------|
| Exp297 46/46 | PASS |
| 1,089 unit tests | PASS |
| `cargo clippy --all-features -W pedantic` | CLEAN |
| `cargo fmt --all -- --check` | CLEAN |

---

## Action items for other springs

1. **neuralSpring**: Exp297 proves GemmF64 GPU↔CPU parity at 1.75e-5. If
   stricter tolerance needed for ML training, evaluate `Fp64Strategy::Native`
   on server GPUs with real FP64 units.
2. **airSpring**: 6 ET₀ methods validated (Hargreaves, FAO-56, Thornthwaite,
   Makkink, Turc, Hamon). Seasonal GPU pipeline ready for end-to-end
   irrigation scheduling validation.
3. **groundSpring**: Bootstrap/jackknife/fit_all validated with FitResult
   named accessors (S81). InterconnectTopology ready for multi-GPU routing.
4. **hotSpring**: Anderson spectral scaling benchmarked (n=2000 in 1.3s CPU).
   SpectralBridge → NautilusBrain path validated. Consider GPU batch via
   BatchIprGpu for large-scale disorder studies.
5. **ToadStool**: `BatchedMultinomialGpu::sample` API change broke downstream.
   Consider semver-aware changelog for breaking changes.
