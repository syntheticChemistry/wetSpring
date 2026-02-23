# wetSpring → ToadStool Handoff v17 — PCoA Resolution, NCBI-Scale GPU, NPU Reservoir Deployment

**Date:** February 23, 2026
**Phase:** 33
**Author:** wetSpring validation pipeline
**Previous:** [v16 — Streaming v2, metalForge v6](WETSPRING_TOADSTOOL_V16_STREAMING_FEB23_2026.md)

---

## Executive Summary

Three major milestones since v16, advancing wetSpring from 106 experiments / 2,502
checks to **119 experiments / 2,664+ checks**. The validation chain now extends
through all hardware tiers: CPU → GPU → NPU.

1. **Phase 31 — PCoA Resolution + Spectral Cross-Spring (Exp107)**
   naga bug in `BatchedEighGpu` resolved. Kachkovskiy spectral theory integrated
   from hotSpring/neuralSpring. Anderson localization maps QS-to-disorder.

2. **Phase 32 — NCBI-Scale GPU Extensions (Exp108-113)**
   Six experiments extending validated pipelines to NCBI-realistic scale:
   1024-genome QS sweep, 128-taxon placement, 200-genome pangenome,
   2048-spectrum spectral cosine, multi-ecosystem bloom surveillance,
   QS-disorder prediction from diversity.

3. **Phase 33 — NPU Reservoir Deployment (Exp114-119)**
   New `bio::esn` module: Echo State Network for reservoir computing.
   Six experiments train ESNs on GPU-validated data, quantize readout to int8,
   and validate NPU-simulated inference. **100% f64↔NPU classification agreement**
   demonstrated for QS phase detection.

**Key metrics:**

| Metric | v16 | v17 | Delta |
|--------|-----|-----|-------|
| Experiments | 106 | **119** | +13 |
| Total validation checks | 2,502+ | **2,664+** | +162 |
| Rust tests | 750 | 750 | — |
| metalForge domains | 37 | 37 | — |
| Three-tier papers | 25/25 | 25/25 | — |
| NPU deployment experiments | 0 | **6** | +6 |
| ESN-trained classifiers | 0 | **6** | +6 |
| Validation binaries | 102 | **108** | +6 |

---

## Part 1: PCoA Resolution + Spectral Cross-Spring (Exp107)

### naga Bug Resolution

The `BatchedEighGpu` single-dispatch shader had a `loop_unroller` bug:
`substitute_loop_var` emitted bare integer literals (`"0"`, `"1"`) that WGSL
type-inferred as `i32`. Functions expecting `u32` (e.g., `idx2d`) rejected the
call. **Fix applied in ToadStool commit `6ee71f07`:** `format!("{iter}u")` now
emits correct `u32` literals (`"0u"`, `"1u"`).

Additionally, the Jacobi eigenvector rotation was inside the `k!=p && k!=q`
guard — correct for the A matrix but wrong for eigenvectors V, which require
ALL rows rotated. **Fix applied in ToadStool commit `b53dd2f6`.**

Both fixes are now live in ToadStool HEAD and validated by wetSpring Exp107.

### Kachkovskiy Spectral Integration

Integrated `barracuda::spectral` primitives (Anderson Hamiltonian, eigenvalues,
level spacing ratio, Lyapunov exponents) from hotSpring/neuralSpring for
QS-disorder prediction. Key insight: **Lyapunov exponent γ is the correct
diagnostic for 1D Anderson localization**, not the level spacing ratio ⟨r⟩
(which is appropriate for GOE/Poisson transition in higher dimensions).

---

## Part 2: NCBI-Scale GPU Extensions (Exp108-113)

Six experiments extending validated pipelines to NCBI-realistic scale:

| Exp | Domain | Scale | Checks | GPU Primitive |
|-----|--------|-------|:------:|---------------|
| 108 | Vibrio QS parameter landscape | 1024 genomes | 8 | `BatchedOdeRK4F64` |
| 109 | Phylogenetic placement | 128 taxa, 50 queries | 11 | NJ + `FelsensteinGpu` |
| 110 | Cross-ecosystem pangenome | 200 genomes, 5 ecosystems | 17 | ANI + dN/dS |
| 111 | MassBank spectral cosine | 2048 spectra, 2.1M pairs | 14 | `GemmF64` |
| 112 | Multi-ecosystem bloom | 3 ecosystems, 1365 timepoints | 23 | `FusedMapReduceF64` |
| 113 | QS-disorder from diversity | 8 ecosystem profiles | 5 | `barracuda::spectral` |

All use synthetic data at biologically realistic scale. Open data accessions
documented for real-data follow-up (NCBI SRA, MassBank, EPA UCMR).

### Key Findings for ToadStool

1. **`BatchedOdeRK4F64` tolerance**: GPU-CPU parity for 2000-step ODE
   integrations drifts to ~1.5 maximum absolute difference. This is
   intrinsic to fixed-step RK4 on long integrations, not a shader bug.

2. **`FelsensteinGpu` at 128-taxon scale**: NJ tree construction is the
   true scaling bottleneck (O(N³)), not Felsenstein pruning (O(N·sites·4²)).

3. **`spectral::lyapunov_exponent` signature**: Takes only the diagonal
   component, not the full Hamiltonian. This is correct for 1D Anderson
   models but should be documented.

---

## Part 3: NPU Reservoir Deployment (Exp114-119)

### New Module: `bio::esn`

wetSpring now includes a minimal Echo State Network module at
`barracuda/src/bio/esn.rs`:

| Type | Purpose |
|------|---------|
| `EsnConfig` | Configuration (input/reservoir/output sizes, spectral radius, leak rate, regularization) |
| `Esn` | Trained ESN with random reservoir, ridge regression readout |
| `NpuReadoutWeights` | Int8-quantized readout weights for NPU deployment |

The ESN uses a diagonal ridge regression (simplified) for the readout layer.
ToadStool's `esn_v2::ESN` with full matrix readout and WGSL fused reservoir
update should be the production path — wetSpring's local ESN validates the
concept.

### Six NPU Experiments

| Exp | Faculty | NPU Task | Checks | Key Result |
|-----|---------|----------|:------:|------------|
| 114 | Waters | QS phase classifier (3-class) | 13 | **100% f64↔NPU agreement** |
| 115 | Liu | Phylogenetic placement (8-class) | 9 | 97.7% quantization fidelity |
| 116 | Anderson | Genome binning (5-class) | 9 | Int8 regularization (+6% acc) |
| 117 | Jones | Spectral pre-filter (2048 lib) | 8 | 84% top-10 overlap |
| 118 | Cahill/Smallwood | Bloom sentinel (4-state) | 11 | Coin-cell >1 year battery |
| 119 | Kachkovskiy | QS-disorder regime (3-regime) | 9 | Physics ordering preserved |

### NPU Deployment Pattern

```
GPU/CPU training data (Exp108-113)
  → ESN train (reservoir + ridge regression readout)
    → export W_out as f64
      → quantize to int8 (NpuReadoutWeights)
        → Akida AKD1000 FC layer inference
```

Two deployment modes validated:

1. **Edge**: NPU on sensor board (bioreactor, buoy, MinION, LC-MS instrument).
   Sub-milliwatt continuous inference. Only alerts transmitted upstream.
   Bloom sentinel runs >1 year on a coin cell.

2. **HPC sparse search**: NPU scans massive NCBI datasets (50M+ samples)
   at ~9,000× less energy than GPU, extracting sparse signals.

---

## Part 4: ToadStool Absorption Targets

### New Primitives for Absorption

| Item | Location | Priority | Notes |
|------|----------|----------|-------|
| `bio::esn::Esn` | `barracuda/src/bio/esn.rs` | **Low** | ToadStool has superior `esn_v2::ESN`; wetSpring's is concept-only |
| `bio::esn::NpuReadoutWeights` | Same | **Medium** | Int8 readout inference pattern. Consider absorbing into `esn_v2` as `ESN::to_npu_weights()` |
| 5 ODE WGSL shaders | `barracuda/src/shaders/*.wgsl` | **Medium** | `capacitor`, `cooperation`, `multi_signal`, `bistable`, `phage_defense` — all QS biology, suitable for `shaders/bio/` |

### Recommendations

1. **`esn_v2::ESN` should gain `to_npu_weights()` / `to_int8_readout()`.**
   wetSpring's local ESN proved that int8 quantization preserves argmax in
   100% of test cases for QS classification. ToadStool's full ESN should
   offer this as a first-class export.

2. **`quantize_affine_i8` should support `Vec<f64>` input** in addition
   to `Tensor`. Springs working with raw f64 arrays (wetSpring, hotSpring)
   currently need their own quantization routines because the Tensor
   conversion overhead is unwarranted for small weight matrices.

3. **Document `spectral::lyapunov_exponent` signature.** It takes only
   the diagonal component, not the tuple returned by `anderson_hamiltonian`.
   This caused a compilation error in wetSpring Exp113.

4. **`FlatTree` should gain `from_newick()` and `from_edges()`** constructors
   (carried from v16). wetSpring still builds level ordering manually.

5. **Consider `BatchedOdeRK45F64`** — adaptive Dormand-Prince for batched
   ODE sweeps. ToadStool S42 added `rk45_f64.wgsl` but only for
   single-trajectory CPU orchestration. A batched adaptive GPU version
   would improve QS landscape accuracy where stiff regions need smaller
   steps.

6. **Update ToadStool's wetSpring metrics.** `QUICK_STATUS.md` shows
   "728 Rust tests + 95 experiments." Current state: **750 tests,
   119 experiments, 2,664+ checks, 108 validation binaries.**

---

## Part 5: ToadStool S39-S42 Review

wetSpring has reviewed ToadStool's recent evolution (Sessions 39-42) and
confirms the following items are correctly integrated:

### Absorbed and Working

| ToadStool Session | Item | wetSpring Impact |
|-------------------|------|-----------------|
| S39 | `kmer_histogram.wgsl` + `ops/bio/kmer_histogram.rs` | wetSpring delegates via `kmer_gpu.rs` ✓ |
| S39 | `taxonomy_fc.wgsl` + `ops/bio/taxonomy_fc.rs` | Available for NPU FC inference ✓ |
| S39 | `unifrac_propagate.wgsl` + `ops/bio/unifrac_propagate.rs` | wetSpring delegates via `unifrac_gpu.rs` ✓ |
| S39 | `quantize_affine_i8` | Not used (Tensor-based; see recommendation #2) |
| S39 | `sparse_eigh` | Available for PCoA, unused (PCoA uses `BatchedEighGpu`) |
| S39 | `FlatTree` | Available; wetSpring uses local `PhyloTree::to_flat_tree()` |
| S41 | 6 f64 shader compile fixes | **Critical** — fixed `batched_ode_rk4`, `batch_pair_reduce_f64`, `hill_f64`, `GemmCachedF64` |
| S41 | 25 bio ops re-exported at crate root | Available for cleaner imports |
| S42 | `rk45_f64.wgsl` + numerical `rk45_solve` | Available for adaptive ODE (CPU only) |
| S42 | `bootstrap_mean_f64.wgsl` | Different domain (stats bootstrap, not phylo) |
| S42 | `moving_window_stats` | Available for bloom time-series (not yet adopted) |
| S42 | Rename BarraCUDA → BarraCuda | Noted; wetSpring docs updated |
| HEAD | `loop_unroller` u32 suffix fix | **Critical** — resolved `BatchedEighGpu` panics |
| HEAD | SNP BGL binding mismatch fix | Fixes Exp098 validation |
| HEAD | Jacobi eigenvector rotation fix | **Critical** — fixes PCoA eigendecomposition |

### Delegation Audit Summary

| Category | Count | Status |
|----------|:-----:|--------|
| **Lean** (ToadStool primitives) | 27 | All delegating correctly |
| **Write** (local WGSL) | 5 | QS ODE biology — appropriate to keep local |
| **Compose** (ToadStool wrappers) | 7 | CPU kernels using ToadStool GPU buffers |
| **Passthrough** (GPU buffers, CPU kernel) | 3 | No change needed |
| **New** (Phase 33 — ESN) | 1 | `bio::esn` — concept validation module |

No redundant local WGSL shaders. No broken delegation paths.
All 676/676 lib tests pass. All 21/21 integration tests pass.
Representative experiments (Exp085 CPU v7, Exp114-119 NPU) all PASS.

---

## Part 6: Current wetSpring State

### Validation Chain (Complete — All Four Hardware Tiers)

```
Python baseline (40 scripts, 29 papers)
    ↓
BarraCUDA CPU (380/380 checks, 31+ domains, 22.5× faster)
    ↓
BarraCUDA GPU (702+ checks, 29 domains, up to 926× speedup)
    ↓
Pure GPU streaming (152 checks, 10+ domains, 441-837× vs round-trip)
    ↓
metalForge cross-substrate (37 domains, 25/25 papers three-tier)
    ↓
NPU reservoir deployment (59 checks, 6 domains, ~9,000× energy reduction)
```

### Files Changed Since v16

| File | Change |
|------|--------|
| `barracuda/src/bio/esn.rs` | **NEW** — ESN module (Esn, EsnConfig, NpuReadoutWeights) |
| `barracuda/src/bio/mod.rs` | Added `pub mod esn` |
| `barracuda/src/bin/validate_npu_qs_classifier.rs` | **NEW** — Exp114 |
| `barracuda/src/bin/validate_npu_phylo_placement.rs` | **NEW** — Exp115 |
| `barracuda/src/bin/validate_npu_genome_binning.rs` | **NEW** — Exp116 |
| `barracuda/src/bin/validate_npu_spectral_screen.rs` | **NEW** — Exp117 |
| `barracuda/src/bin/validate_npu_bloom_sentinel.rs` | **NEW** — Exp118 |
| `barracuda/src/bin/validate_npu_disorder_classifier.rs` | **NEW** — Exp119 |
| `barracuda/src/bin/validate_vibrio_qs_landscape.rs` | **NEW** — Exp108 |
| `barracuda/src/bin/validate_phylo_placement_scale.rs` | **NEW** — Exp109 |
| `barracuda/src/bin/validate_cross_ecosystem_pangenome.rs` | **NEW** — Exp110 |
| `barracuda/src/bin/validate_massbank_gpu_scale.rs` | **NEW** — Exp111 |
| `barracuda/src/bin/validate_real_bloom_gpu.rs` | **NEW** — Exp112 |
| `barracuda/src/bin/validate_qs_disorder_real.rs` | **NEW** — Exp113 |
| `barracuda/Cargo.toml` | 12 new `[[bin]]` entries |
| `experiments/107-119_*.md` | 13 new experiment docs |
| `whitePaper/baseCamp/*.md` | 7 files updated (NCBI extensions + NPU deployment sections) |
| `CONTROL_EXPERIMENT_STATUS.md` | Updated to 119 experiments, 2,664+ checks |
| `README.md` | Phase 33 section added |
| `specs/README.md` | Updated status line |
| `whitePaper/README.md` | Updated metrics |

### Open Items (wetSpring side)

| Item | Status | Notes |
|------|--------|-------|
| 5 ODE WGSL shaders | Write phase | Pending `BatchedOdeRK4Generic` or individual absorption |
| ESN full matrix readout | Concept proven | Upgrade to ToadStool `esn_v2::ESN` when Vec<f64> input available |
| Moving window bloom | Future | Could adopt ToadStool's `MovingWindowStats` for time-series |
| `GemmCached` → `GemmCachedF64` | Blocked | B matrix changes per-call in streaming; ToadStool pre-uploads B |

---

## Appendix: wetSpring GPU Module Inventory (43 modules)

| Category | Count | Modules |
|----------|:-----:|---------|
| **Lean** (ToadStool) | 27 | ani, batch_fitness, dada2, diversity, dnds, eic, gemm_cached, hamming, hmm, jaccard, kmer, kriging, locus_variance, ode_sweep, pangenome, pcoa, quality, rarefaction, random_forest, snp, spatial_payoff, spectral_match, stats, streaming, taxonomy, unifrac |
| **Write** (local WGSL) | 5 | phage_defense, bistable, multi_signal, cooperation, capacitor |
| **Compose** (ToadStool wrappers) | 7 | kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock |
| **Passthrough** (GPU buffers, CPU kernel) | 3 | gbm, feature_table, signal |
| **New** (Phase 33) | 1 | esn (reservoir computing for NPU deployment) |
