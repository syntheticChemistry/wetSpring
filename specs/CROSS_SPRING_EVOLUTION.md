<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Cross-Spring Shader & Primitive Evolution

> How five Springs contribute domain expertise to BarraCuda, and all benefit.

## Evolution Model

Each Spring validates a scientific domain against Python baselines, promoting
correct Rust implementations into ToadStool's shared BarraCuda crate. Once
absorbed, every Spring—and every future Spring—benefits from the shared math
engine without reimplementation.

```
  hotSpring ──── precision math, DF64, ESN, lattice QCD ────┐
  wetSpring ──── bio diversity, HMM, Felsenstein, DADA2 ────┤
  neuralSpring ─ ML ops, eigensolver, tensor session ───────┼──▶ BarraCuda ──▶ ToadStool
  groundSpring ─ evolution (Kimura), jackknife, bootstrap ──┤
  airSpring ──── hydrology, seasonal pipeline, Brent opt ───┘
```

## wetSpring Contributions (absorbed upstream)

| Primitive | ToadStool Location | Session | Notes |
|-----------|--------------------|---------|-------|
| Shannon entropy | `ops::fused_map_reduce_f64` | S-53 | 2.4× faster via `FusedMapReduceF64` |
| Simpson diversity | `ops::fused_map_reduce_f64` | S-53 | Same fused kernel |
| Bray-Curtis | `stats::bray_curtis` | S-64 | Also GPU via `BatchPairReduceF64` |
| `log_f64` precision fix | `shaders/math_f64.wgsl` | S-17 | Fixed halved coefficients; all Springs benefit |
| HMM forward (f64) | `ops::bio::HmmBatchForwardF64` | S-41 | 10⁹× precision vs f32 |
| Felsenstein pruning | `ops::bio::FelsensteinGpu` | S-45 | Phylogenetic likelihood |
| DADA2 E-step | `ops::bio::Dada2EStepGpu` | S-49 | Amplicon denoising |
| Smith-Waterman banded | `ops::bio::SmithWatermanGpu` | S-51 | Banded alignment |
| Gillespie SSA | `ops::bio::GillespieGpu` | S-52 | Stochastic simulation |
| `pow_f64` polyfill | `shaders/math_f64.wgsl` | S-17 | `HillGateGpu` support |
| Chao1 richness | `stats::chao1` | S-64 | Classic + bias-corrected |
| FST variance decomposition | `ops::bio::fst_variance_decomposition` | S-53 | Population genetics |
| Ada Lovelace workaround | WGSL precision guards | S-17 | f64 emulation on FP32 cores |

## Primitives wetSpring Consumes from Other Springs

### From hotSpring (precision math)

| Primitive | Use in wetSpring | Session |
|-----------|-----------------|---------|
| `ReduceScalarPipeline` | Two-pass f64 reduction for diversity GPU | S-60 |
| `DF64` core + transcendentals | Extended precision for Anderson lattices | S-63 |
| ESN reservoir/readout WGSL | Bio ESN GPU path via ToadStool `esn_v2` | S-66 |
| `math_f64.wgsl` patterns | All f64 shader math | S-17 |
| `GpuDriverProfile` | Adapter selection and FP64 strategy | S-58 |
| `Fp64Strategy` | Native/Hybrid/Concurrent dispatch | S-61 |

### From neuralSpring (ML/stats)

| Primitive | Use in wetSpring | Session |
|-----------|-----------------|---------|
| `pairwise_hamming` | Genomic distance matrices | S-55 |
| `pairwise_jaccard` | Pangenome presence/absence | S-55 |
| `pairwise_l2` | PCoA ordination | S-55 |
| `locus_variance` | FST computation | S-53 |
| `batch_fitness_eval` | Selection pressure in QS models | S-57 |
| `rk4_parallel` | ODE integration for QS dynamics | S-59 |
| `eigh_householder_qr` | Eigendecomposition for PCoA | S-63 |
| `batch_ipr` | Anderson spectral localization | S-63 |
| `hill_gate` | Cooperative regulation in QS | S-57 |

### From groundSpring (evolution)

| Primitive | Use in wetSpring | Session |
|-----------|-----------------|---------|
| `kimura_fixation_prob` | Population genetics validation | S-70 |
| `error_threshold` | Quasispecies error threshold | S-70 |
| `detection_power` | Statistical power for diversity studies | S-70 |
| `jackknife` / `jackknife_mean_variance` | Bias-corrected diversity estimates | S-70 |
| `chao1_classic` | Richness estimation (groundSpring variant) | S-70 |
| `bootstrap_ci` | Confidence intervals for all bio metrics | S-66 |

### From airSpring (hydrology/general)

| Primitive | Use in wetSpring | Session |
|-----------|-----------------|---------|
| `fit_exponential` | Growth curve fitting | S-66 |
| `fit_logarithmic` | Rarefaction curve fitting | S-66 |
| `fit_quadratic` | Non-linear regression | S-66 |
| `moving_window_stats_f64` | Time-series smoothing | S-66 |

## ToadStool Primitive Count

As of Phase 95 (March 3, 2026):
- **150+ ToadStool primitives** consumed by wetSpring
- **0 local WGSL shaders** (all absorbed)
- **844+ total WGSL shaders** in ToadStool across all Springs
- **8,300+ validation checks** in wetSpring (1,044 lib tests)
- **29 computational chemistry ops** mapped to BarraCUDA primitives (blueFish isomorphism proof)
- **Zero panics** in library code, **zero unsafe**, **zero clippy warnings** (`--all-features -W pedantic`)

## Cross-Spring Validation

`validate_cross_spring_evolution_modern` validates 23 checks across all five
Spring origins, proving that absorbed primitives work correctly in wetSpring's
bio context.

`benchmark_cross_spring_modern` benchmarks 12 primitives by Spring origin,
tracking throughput for the cross-spring math engine.

## ESN Cross-Spring Evolution

The Echo State Network illustrates cross-spring evolution:

1. **hotSpring** evolved the ESN for MD transport prediction (CPU f64 → GPU WGSL f32 → NPU int4)
2. **ToadStool** absorbed the ESN into `esn_v2` with multi-head support (11 heads, S-66)
3. **wetSpring** now uses `esn_v2` via `BioEsn` bridge for bio classifiers (diversity, taxonomy, AMR)
4. **neuralSpring** uses the same `esn_v2` for ML surrogate prediction

Each Spring contributed domain-specific improvements:
- hotSpring: precision (Xoshiro256pp PRNG, f64 training), 4-layer brain architecture, 36-head ESN with `HeadGroupDisagreement`
- wetSpring: bio feature extraction, multi-head bio classifiers, blueFish chemistry layer (29 comp-chem ops mapped)
- neuralSpring: tensor session integration and hardware routing

## blueFish Chemistry Layer (V87)

The blueFish whitePaper (`whitePaper/blueFish/`) extends cross-spring evolution into computational chemistry. The isomorphism proof (`02_ISOMORPHISM.md`) maps 29 comp-chem operations to BarraCUDA primitives:

- **Tier 1** (14 direct): Matrix diagonalization, GEMM, FFT, gradient descent — all exist
- **Tier 2** (9 compose): SCF iteration (CG + GEMM + reduce), MD integration (velocity Verlet + force eval) — compose existing
- **Tier 3** (6 genuinely new): ERI (Obara-Saika), Boys function, Schwarz screening, Becke partitioning, Gaussian basis eval, RI decomposition

The 6 new kernels represent a 15% expansion of BarraCUDA's primitive library for an entirely new scientific domain. ERI has structural analogy to hotSpring's 4-point correlation functions.

## hotSpring Brain Architecture Ingest (V87)

Reviewed hotSpring v0.6.15 brain architecture for wetSpring bio mapping:

| hotSpring Pattern | wetSpring Bio Analog |
|---|---|
| `CgResidualUpdate` (GPU → NPU) | `DiversityUpdate { n_species, shannon_h, evenness }` |
| `BrainInterrupt::KillCg` (NPU → GPU) | `BrainInterrupt::FlagNovelState` |
| `AttentionState` (Green/Yellow/Red) | Monitors diversity trajectory |
| `HeadGroupDisagreement` | Bio head groups: Anderson-informed, diversity-empirical, phylogeny-informed |
| `NautilusBrain` + `BetaObservation` | `NautilusBrain` + `ChemObservation` or `QsObservation` |
