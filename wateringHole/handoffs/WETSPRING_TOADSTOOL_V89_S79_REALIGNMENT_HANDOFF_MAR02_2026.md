# wetSpring → ToadStool/BarraCUDA Handoff V89 — S79 Deep Rewire + Cross-Spring Validation

**Date**: March 2, 2026
**From**: wetSpring (V89)
**To**: ToadStool/BarraCUDA team
**ToadStool pin**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later
**Supersedes**: V88 (Experiment Buildout + Barracuda API Learnings)

---

## Executive Summary

- **ToadStool S71→S79 deep rewire**: Updated pin from `1dd7e338` to `f97fc2ae`. Rewired lib-level IPC handlers and ESN bridge to consume S79 APIs natively (`SpectralAnalysis`, `MultiHeadEsn`).
- **`MultiHeadBioEsn` added**: New wrapper for `ToadStool` `MultiHeadEsn` (hotSpring 36-head → S79) with bio-specific 5-head scheme (diversity/taxonomy/AMR/bloom/disorder), per-head training, and head disagreement uncertainty.
- **IPC `SpectralAnalysis` rewire**: Anderson handler now returns `spectral_bandwidth`, `spectral_condition_number`, `spectral_phase`, and `marchenko_upper` via `SpectralAnalysis::from_eigenvalues()`.
- **Exp271: Cross-Spring S79 Validation**: 73/73 checks across 13 domains, benchmarking all cross-spring evolved primitives with provenance annotations (6 springs → ToadStool → wetSpring).
- **All quality gates green**: clippy (pedantic+nursery), fmt, doc, tests, zero warnings.

---

## Part 1: ToadStool S71→S79 Evolution Absorbed

### Commits since V88 pin

| Commit | Session | Key Changes |
|--------|---------|-------------|
| `997f3c6b` | S71 | GPU dispatch wiring, sovereignty constants, smart refactoring |
| `f4fa053c` | S71+ | Smart refactoring, DF64 transcendentals, ComputeDispatch migration |
| `3ebdc877` | S71++ | ComputeDispatch batch 2+3, DF64 gamma/erf, unsafe reduction |
| `d63cc116` | S71+++ | ComputeDispatch batches 4-6, external deps audit |
| `333a46d3` | S71+++ | Doc cleanup for accuracy |
| `8dc01a37` | S71+++ | Archive stale code, remove placeholder tests |
| `223b2007` | S78 | libc→rustix, async-trait→AFIT migration, wildcard narrowing |
| `7505b32a` | S79 | ESN V2 shape fix, MultiHeadEsn, spectral extensions, 5 ComputeDispatch |
| `f97fc2ae` | S79 | FFT buffer fix, f64 naga strip, asin_df64 iterative |

### New ToadStool features consumed

| Module | Feature | Consumption |
|--------|---------|-------------|
| `spectral::stats` | `SpectralAnalysis::from_eigenvalues()` | **IPC handler** (bandwidth, condition, phase, Marchenko) |
| `spectral::stats` | `SpectralPhase`, `detect_bands` | **Exp271 validated** |
| `esn_v2::multi_head` | `MultiHeadEsn`, `HeadConfig`, `HeadGroup` | **`MultiHeadBioEsn`** wrapper |
| `esn_v2::multi_head` | `head_disagreement()` | **`MultiHeadBioEsn`** uncertainty |
| `esn_v2::model` | `ExportedWeights.head_labels` | **`BioEsn`** auto-populates |
| `stats::hydrology` | FAO-56, Hargreaves, soil water balance | **Exp271 validated** |
| `stats::evolution` | Kimura, error threshold, detection power | **Exp271 validated** |
| `stats::jackknife` | Jackknife mean/variance, custom statistic | **Exp271 validated** |
| `stats::bootstrap` | `bootstrap_mean`, `rawr_mean` | **Exp271 validated** |
| `stats::regression` | `fit_linear`, `fit_quadratic`, `fit_all` | **Exp271 validated** |
| `sample::metropolis` | `boltzmann_sampling` (Metropolis-Hastings) | **Exp271 validated** |

### Infrastructure changes (transparent to wetSpring)

- **libc→rustix**: Pure Rust system calls (no C dependency)
- **async-trait→AFIT**: Native async fn in trait (Rust edition evolution)
- **Wildcard re-exports narrowed**: Explicit imports only
- **ComputeDispatch migration**: 76/250 ops migrated to new dispatch
- **844 WGSL shaders**: All f64-canonical, zero f32-only remain

---

## Part 2: wetSpring Changes

### `barracuda/src/bio/esn/toadstool_bridge.rs`

- **`MultiHeadBioEsn`** (new): Wraps `ToadStool` `MultiHeadEsn` with bio-specific heads
  - `new_bio5()`: Standard 5-head (diversity, taxonomy, AMR, bloom, disorder)
  - `new()`: Custom head configuration
  - `train_head()`: Per-head training via `BioHeadKind`
  - `head_disagreement()`: Uncertainty signal (mean pairwise L2 between heads)
  - `export_weights()` / `head_labels()` / `num_heads()`
- Cross-spring provenance documented: hotSpring (36-head concept) + wetSpring (bio heads)
- `export_weights()`: Auto-populates `head_labels` when `output_size > 1`
- `migrate_to_multi_head()`: Auto-populates `head_labels` on migrated weights

### `barracuda/src/ipc/handlers.rs`

- Anderson handler rewired to use `SpectralAnalysis::from_eigenvalues()`
- Response now includes: `spectral_bandwidth`, `spectral_condition_number`, `spectral_phase`, `marchenko_upper`
- Level spacing ratio and regime classification preserved

### `barracuda/src/gpu.rs`

- Doc updated: 844 shaders, S79 pin, `MultiHeadEsn` in consumed primitives list
- Removed "pending absorption" language — ODE domains use `ToadStool` runtime WGSL generation

### `barracuda/src/special.rs`

- Doc corrected: `dot`/`l2_norm` correctly documented as delegates to `barracuda::stats` (S64)

### `barracuda/Cargo.toml`

- ToadStool pin comment updated: S79 (`f97fc2ae`), 93+ primitives, 844 shaders

---

## Part 3: Exp271 — Cross-Spring S79 Validation

### 73/73 checks, 13 domains, ~21ms total

| Domain | Spring Origin | Session | Checks |
|--------|--------------|---------|--------|
| Alpha Diversity | wetSpring | S64 | 9 |
| Beta + Rarefaction | wetSpring | S64 | 7 |
| Spectral Analysis | hotSpring + neuralSpring | v0.6.0 + S79 | 9 |
| Population Genetics | groundSpring | S70 | 6 |
| Jackknife | groundSpring | S70 | 4 |
| Regression | airSpring | S66 | 5 |
| Hydrology (FAO-56) | airSpring + groundSpring | S70 | 6 |
| Special Functions | multi-spring | S64 | 6 |
| Bootstrap CI | multi-spring | S64+ | 4 |
| Linear Algebra | wetSpring | S59+ | 5 |
| Moving Window | airSpring + wetSpring | S66 | 4 |
| Boltzmann Sampling | wateringHole | V69 | 3 |
| Correlation | multi-spring | S64 | 5 |

### Cross-Spring Evolution Tree

```text
┌─ hotSpring v0.6.0 ── spectral (Anderson, Lanczos, level statistics)
├─ neuralSpring V69 ── spectral phase (Bulk/EdgeOfChaos/Chaotic)
├─ wetSpring S64 ───── diversity, Bray-Curtis, rarefaction, NMF, ridge
├─ groundSpring S70 ── Kimura fixation, error threshold, jackknife
├─ airSpring S66 ───── regression, moving window, hydrology
├─ wateringHole V69 ── Boltzmann sampling (Metropolis-Hastings MCMC)
└─ multi-spring ────── special functions, bootstrap, correlation
```

All primitives flow: spring → ToadStool BarraCUDA S79 → consumed by wetSpring.
844 WGSL shaders, all f64-canonical. Zero local shaders in wetSpring.

---

## Part 4: Bio Shader Provenance Map (ToadStool)

wetSpring has contributed the following shaders to ToadStool (now consumed back):

| Shader | Provenance |
|--------|-----------|
| `hmm_forward_f64.wgsl` | wetSpring handoff v5 |
| `dada2_e_step.wgsl` | wetSpring handoff v5 |
| `quality_filter.wgsl` | wetSpring handoff v5 |
| `snp_calling_f64.wgsl` | wetSpring handoff v5 |
| `pangenome_classify.wgsl` | wetSpring handoff v5 |
| `dnds_batch_f64.wgsl` | wetSpring handoff v5 |
| `ani_batch_f64.wgsl` | wetSpring handoff v5 |
| `diversity_fusion_f64.wgsl` | wetSpring Write phase |
| `taxonomy_fc.wgsl` | wetSpring metagenomics |
| `kmer_histogram.wgsl` | wetSpring metagenomics |
| `unifrac_propagate.wgsl` | wetSpring metagenomics |
| `gillespie_ssa_f64.wgsl` | wetSpring §P1 Gillespie |
| `tree_inference_f64.wgsl` | wetSpring §Shader Design 2 |
| `felsenstein_f64.wgsl` | wetSpring §Shader Design 3 |
| `smith_waterman_banded_f64.wgsl` | wetSpring §Shader Design 1 |
| `kmd_grouping_f64.wgsl` | wetSpring Exp018 Jones Lab |
| `batch_tolerance_search_f64.wgsl` | wetSpring Exp018 PFAS |

neuralSpring-contributed bio shaders (used by wetSpring via ToadStool):

| Shader | Provenance |
|--------|-----------|
| `hill_gate.wgsl` | neuralSpring metalForge |
| `multi_obj_fitness.wgsl` | neuralSpring metalForge |
| `swarm_nn_forward.wgsl` | neuralSpring metalForge |
| `hmm_backward_log_f64.wgsl` | neuralSpring S69 |
| `hmm_viterbi_f64.wgsl` | neuralSpring S69 |

---

## Part 5: Absorption Targets

### From V88 (still open)

1. **Brain architecture generalization** — abstract hotSpring 4-layer brain to `barracuda::brain`
2. **ERI shader class** — 4-center integral pattern for computational chemistry
3. **`FitResult` named accessors** — ergonomic layer over `params: Vec<f64>`
4. **`dispatch::route` docs** — clarify `None` return for CPU-only workloads
5. **Spectral pipeline convenience** — optional `anderson_eigenvalues(l, w, seed)` combining 3-step chain

### New from V89

6. **`HeadGroup::Bio`** — add wetSpring's 5-head bio scheme to ToadStool directly (diversity, taxonomy, amr, bloom, disorder), removing need for wrapper labels
7. ~~Spectral stats consumption~~ — **DONE**: IPC handler now uses `SpectralAnalysis::from_eigenvalues()`
8. ~~MultiHeadEsn bio integration~~ — **DONE**: `MultiHeadBioEsn` wrapper added

---

## Part 6: Quality State

| Metric | V88 | V89 |
|--------|-----|-----|
| ToadStool pin | S71+++ (`1dd7e338`) | S79 (`f97fc2ae`) |
| Experiments | 270 | 271 (+Exp271 cross-spring) |
| Validation checks | 7,083+ | 7,156+ (+73 cross-spring) |
| Tests | 1,249 | 1,274 (+25 from prior audit) |
| ToadStool primitives consumed | 93 | 93+ (deeper integration) |
| Local WGSL shaders | 0 | 0 |
| Clippy | CLEAN | CLEAN |
| Unsafe code | 0 | 0 |
| ToadStool WGSL shaders | ~700 | 844 (all f64-canonical) |

---

*Archived handoff:* `archive/WETSPRING_TOADSTOOL_V88_EXPERIMENT_BUILDOUT_HANDOFF_MAR02_2026.md`
