# Handoff v3: wetSpring → ToadStool / BarraCUDA Team

> **SUPERSEDED** by [`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md`](../../HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_21_2026.md)
> (v4 — ToadStool evolution review, updated remaining requests and shader designs).

**Date:** February 19, 2026 (final)
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** All prior handoffs (Feb 16, 17, 19 v1/v2)

---

## Executive Summary

wetSpring has completed its full validation buildout: 22 experiments across
three tracks, 645/645 validation checks, 430 tests, zero clippy warnings.
The project now demonstrates the complete evolution path from Python to
sovereign Rust, including mathematical models (ODE, stochastic simulation),
phylogenetic algorithms (Robinson-Foulds distance), and sovereign ML
(decision tree inference with 100% sklearn parity).

**What changed since v2 (this session):**

1. **Exp019 Phase 1** — Newick parsing validation: 30/30 checks vs dendropy
2. **Exp022** — Gillespie SSA (Massie 2012): 13/13 checks, Fano=0.976
3. **Exp008 Phase 3** — Decision tree inference: 7/7 checks, 744/744 parity
4. **Root docs rewritten** — README, BENCHMARK_RESULTS, CONTROL_EXPERIMENT_STATUS,
   EVOLUTION_READINESS now contain wetSpring content (were hotSpring)
5. **whitePaper updated** — 645 checks, 430 tests, 36 modules, serde_json noted
6. **All clippy pedantic+nursery** — zero warnings across entire codebase

**Totals:** 645 validation checks (519 CPU + 126 GPU), 430 tests, 22 experiments

---

## What Was Proven

| Claim | Evidence | Files |
|-------|----------|-------|
| Rust matches Python for 16S pipeline | 519/519 CPU checks | 17 validation binaries |
| GPU matches CPU | 126/126 checks, 88/88 pipeline parity | `validate_diversity_gpu`, `validate_16s_pipeline_gpu` |
| 926× spectral cosine GPU speedup | Benchmark with 2048 spectra | `benchmark_cpu_gpu` |
| 2.45× full pipeline GPU speedup | 10 samples, 4 BioProjects | `validate_16s_pipeline_gpu` |
| RK4 ODE matches scipy | 16/16 checks, 4 biological scenarios | `validate_qs_ode` |
| Gillespie SSA converges | 13/13 checks, mean within 0.2% of analytical | `validate_gillespie` |
| RF distance matches dendropy | 23/23 checks, 10 tree topologies | `validate_rf_distance` |
| Newick parser correct | 30/30 checks, 10 topologies, BL exact | `validate_newick_parse` |
| Decision tree 100% parity | 7/7 checks, 744/744 predictions match | `validate_pfas_decision_tree` |
| PFAS ML on real data | RF F1=0.978, GBM F1=0.992, DT F1=0.986 | `exp008_python_baseline.json` |
| Zero unsafe, zero TODO, zero clippy | Full codebase audit | `cargo clippy --pedantic --nursery` |

---

## BarraCUDA Primitives Used (11)

| Primitive | Where Used | Satisfaction |
|-----------|-----------|-------------|
| `FusedMapReduceF64` | diversity, EIC, rarefaction, spectral norms, stats, streaming | Fully validated |
| `BrayCurtisF64` | Bray-Curtis condensed distance matrix | Fully validated |
| `BatchedEighGpu` | PCoA eigendecomposition | Fully validated |
| `GemmF64` | Taxonomy scoring, spectral cosine | Fully validated |
| `KrigingF64` | Spatial interpolation | Functional |
| `VarianceF64` | Population statistics | Functional |
| `CorrelationF64` | Pearson correlation | Functional |
| `CovarianceF64` | Population covariance | Functional |
| `WeightedDotF64` | EIC integration, weighted sums | Functional |
| `GemmCached` | Pipeline-cached GEMM | Streaming validated |
| `FusedMapReduceF64` (streaming) | Quality/diversity in pipeline | Streaming validated |

**Assessment:** All 11 primitives work as documented. No bugs found during
this validation session. The streaming pipeline composition (`StreamingGpu`)
works correctly with mixed ToadStool + custom shaders.

---

## New Primitives Requested

### P0: Batched RK4 ODE Solver

- **Modules:** `bio::ode`, `bio::qs_biofilm`
- **What:** `BatchedRK4F64` — integrate N parameter sets in parallel on GPU
- **Why:** Waters 2008 model validates with 4 scenarios; parameter sweeps
  across hundreds of biological conditions are the natural GPU workload
- **Shader:** `rk4_batch_f64.wgsl` — fixed-step RK4, N threads × M state variables
- **CPU reference:** `bio::ode::rk4_integrate()` — fully tested, 7 unit tests
- **Validation data:** `experiments/results/qs_ode_baseline/` (Python scipy)

### P1: Parallel Stochastic Simulation (Gillespie)

- **Module:** `bio::gillespie`
- **What:** GPU PRNG (Lehmer LCG or xoshiro) + exponential sampling for
  parallel trajectory ensembles
- **Why:** Massie 2012 runs 1000 trajectories — embarrassingly parallel, each
  trajectory is independent with its own seed
- **Shader:** Per-thread SSA with workgroup-level ensemble reduction
- **CPU reference:** `bio::gillespie::birth_death_ensemble()` — sovereign LCG
- **Validation data:** `experiments/results/022_gillespie/` (Python numpy)
- **Note:** LCG constants (Knuth) are in `bio::gillespie::Lcg64`; GPU version
  should use same constants for reproducibility

### P1: Log-Sum-Exp / HMM Forward-Backward

- **Module:** Future `bio::hmm`
- **What:** Numerically stable log-sum-exp reduction for HMM emission/transition
- **Why:** PhyloNet-HMM introgression detection (Liu 2014, Exp019 Phase 3)
- **Paper:** Liu et al. 2014, PLoS Comp Bio, DOI 10.1371/journal.pcbi.1003649

### P1: Smith-Waterman Alignment

- **Module:** Future `bio::alignment`
- **What:** Banded SW with affine gap penalties
- **Why:** SATe 16S alignment (Liu 2009) bridges phylogenetics to Track 1

### P2: Random Forest / Decision Tree GPU Inference

- **Module:** `bio::decision_tree`
- **What:** Batch inference: N samples × M trees on GPU
- **Why:** PFAS monitoring at field scale (3,719 samples, 65-node tree)
- **CPU reference:** `bio::decision_tree::predict_batch()` — 9 unit tests
- **Validation data:** `experiments/results/008_pfas_ml/` (sklearn export)

### P2: Phylogenetic Likelihood (Felsenstein Pruning)

- **Module:** Future `bio::phylo_likelihood`
- **What:** Postorder tree traversal with per-site likelihood
- **Why:** Maximum-likelihood phylogenetics for Track 1b

---

## Local Extensions (Candidates for ToadStool Absorption)

These are wetSpring GPU implementations that should be generalized into
reusable ToadStool primitives:

| Local Code | Location | Generalization Target |
|------------|----------|----------------------|
| `QualityFilterCached` | `quality_gpu.rs` | `ParallelFilter<T>` — per-element predicate + count |
| `Dada2Gpu` (E-step) | `dada2_gpu.rs` | `BatchPairReduce<f64>` — pairwise error accumulation |
| `GemmCached` pipeline | `gemm_cached.rs` | `GemmF64::cached()` — pipeline + BGL caching |
| `StreamingGpu` | `streaming_gpu.rs` | Pipeline composition pattern (multi-stage) |

The `include_str!` path to ToadStool's `gemm_f64.wgsl` is brittle
(`../../phase1/toadstool/crates/barracuda/shaders/gemm_f64.wgsl`). When
ToadStool stabilizes shader loading, this should use the official API.

---

## Evolution Lessons

### 1. Stochastic→Deterministic Bridge

The Gillespie SSA (Exp022) and ODE (Exp020) modules validate the same
biological system (c-di-GMP signaling) at two levels of description. At
high molecule counts, SSA ensemble mean converges to ODE steady state
(100 molecules, Fano≈1.0). This is a powerful validation pattern: if both
methods agree, the math is correct regardless of implementation language.

**ToadStool implication:** Batched RK4 (P0) and parallel SSA (P1) should
produce cross-validatable results. Ship both and test convergence.

### 2. Decision Tree Portability Proves Math Is Language-Independent

The decision tree inference (Exp008) achieves **100% prediction parity** on
744 samples between sklearn (Python) and sovereign Rust. The tree structure
is a pure mathematical object (feature index, threshold, left/right child)
that transfers perfectly across languages. This is the strongest possible
evidence that "BarraCUDA solves math, ToadStool solves hardware."

**ToadStool implication:** Tree inference is a GPU-friendly workload
(embarrassingly parallel over samples). The 65-node tree × 744 samples
took <1ms on CPU; on GPU this scales to millions of samples trivially.

### 3. Robinson-Foulds Canonicalization Is Subtle

The bipartition canonicalization in `robinson_foulds.rs` required careful
handling of equal-sized splits. When both halves of a bipartition have the
same number of leaves, the canonical form must use lexicographic comparison
to avoid counting the same split twice. This was a bug that only manifested
on trees with 5+ leaves.

**ToadStool implication:** If RF distance moves to GPU, the canonicalization
must be done on CPU (string operations). Only the pairwise distance matrix
over many tree pairs benefits from GPU parallelism.

### 4. Sovereign PRNG for Reproducibility

The Gillespie module uses a sovereign Lehmer LCG (`bio::gillespie::Lcg64`)
instead of depending on `rand` crate. This is critical for GPU portability:
the same LCG constants work identically in Rust and WGSL, enabling bitwise
reproducibility of stochastic trajectories across CPU/GPU.

Constants (Knuth):
- Multiplier: `6_364_136_223_846_793_005`
- Increment: `1_442_695_040_888_963_407`

### 5. serde_json Is Acceptable for Model Import

Adding `serde_json` was the right call for decision tree import. The
alternative (manual JSON parsing of sklearn's tree structure) would have
been error-prone and added hundreds of lines of parsing code. The
sovereignty principle is about I/O parsers (FASTQ, mzML, MS2) where we
control the format — not about reimplementing JSON.

### 6. Python Baselines Are Essential Before Rust

Every experiment that started with a Python baseline succeeded on the first
Rust implementation attempt. The two that had debugging cycles (Robinson-Foulds
canonicalization, Gillespie CV² check) were both caught by the Python baseline
producing authoritative reference values.

**ToadStool implication:** Always have the Python baseline GREEN before
writing the GPU shader. The validation chain is: Python → Rust CPU → GPU.
Skipping the middle step loses the debugging signal.

---

## New Dependencies

| Dependency | Version | Why | Impact |
|------------|---------|-----|--------|
| `serde_json` | 1.0 | Decision tree model import from sklearn JSON | ~40KB compiled; used only in `validate_pfas_decision_tree` binary and `decision_tree` module deserialization path |

Existing: `flate2` (gzip), `bytemuck` (GPU casting), `barracuda` (optional, GPU),
`wgpu` (optional, GPU), `tokio` (optional, GPU), `tempfile` (dev).

---

## Data Assets Produced

| Path | Content | Size |
|------|---------|------|
| `experiments/results/019_phylogenetic/newick_parse_python_baseline.json` | dendropy Newick parse stats (10 trees) | 2KB |
| `experiments/results/021_rf_baseline/rf_python_baseline.json` | dendropy RF distances (10 pairs) | 3KB |
| `experiments/results/022_gillespie/gillespie_python_baseline.json` | numpy SSA ensemble (1000 runs) | 1KB |
| `experiments/results/008_pfas_ml/decision_tree_exported.json` | sklearn tree (65 nodes, 28 features) | 15KB |
| `experiments/results/008_pfas_ml/decision_tree_test_data.json` | 744 test samples + predictions | 1.2MB |
| `experiments/results/qs_ode_baseline/qs_ode_python_baseline.json` | scipy ODE (4 scenarios) | 2KB |

---

## Reproduction Commands

```bash
cd barracuda

# Full test suite (430 tests)
cargo test

# Lint clean
cargo fmt --check
cargo clippy -- -W clippy::pedantic -W clippy::nursery

# New validation binaries
cargo run --bin validate_gillespie
cargo run --bin validate_newick_parse
cargo run --bin validate_pfas_decision_tree
cargo run --bin validate_qs_ode
cargo run --bin validate_rf_distance

# Python baselines (from scripts/)
cd ../scripts
python3 gillespie_baseline.py
python3 rf_distance_baseline.py
python3 newick_parse_baseline.py
python3 pfas_tree_export.py
python3 waters2008_qs_ode.py
```

---

## Recommended ToadStool Evolution Priority

| Priority | Primitive | wetSpring Module | Payoff |
|----------|-----------|-----------------|--------|
| **P0** | `BatchedRK4F64` | ode, qs_biofilm | Parameter sweeps for ODE models |
| **P1** | GPU LCG + exponential | gillespie | 1000× trajectory parallelism |
| **P1** | `LogSumExpF64` | future hmm | HMM forward/backward |
| **P1** | `SmithWatermanF64` | future alignment | Phylogenetic alignment |
| **P2** | `TreeInferenceGpu` | decision_tree | ML at field scale |
| **P2** | `FelsensteinF64` | future phylo | ML phylogenetics |
| **Absorb** | `QualityFilterCached` | quality_gpu | Generalize to `ParallelFilter<T>` |
| **Absorb** | `Dada2Gpu` E-step | dada2_gpu | Generalize to `BatchPairReduce<f64>` |
| **Absorb** | `GemmCached` | gemm_cached | Pipeline cache into `GemmF64` builder |
| **Absorb** | `StreamingGpu` | streaming_gpu | Pipeline composition pattern |

---

## What's Left in wetSpring

| Item | Status | Blocking? |
|------|--------|-----------|
| Exp019 Phases 2-4 (gene tree RF, PhyloNet-HMM, SATe 16S) | Needs data download | No |
| Exp008 full RF ensemble in Rust | Future work | No |
| Tolerance centralization (inline → tolerances.rs) | Debt | No |
| Exp002 raw FASTQ download (70 pairs from SRA) | Deferred | No |
| Trimmomatic/pyteomics Python baselines | Deferred | No |

Nothing blocks ToadStool evolution. All Rust CPU modules are validated and
ready for GPU promotion when the corresponding primitives exist.
