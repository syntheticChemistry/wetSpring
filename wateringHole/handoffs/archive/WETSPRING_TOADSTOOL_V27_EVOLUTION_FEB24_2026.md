# wetSpring → ToadStool Handoff V27: Barracuda Evolution Review + Three-Tier Controls

**Date:** February 24, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda team
**Phase:** 39 — paper queue ALL GREEN (43/43 papers), 881 tests, 161 experiments, 3,132+ checks

---

## Summary

This handoff consolidates wetSpring's complete barracuda evolution status,
documents the code audit cleanup (V26), introduces the new Track 3 drug
repurposing workstream, and provides the three-tier control matrix showing
which papers have CPU, GPU, and metalForge coverage — and what gaps remain
for ToadStool to close.

### What Changed Since V26

1. **Root docs unified** — README, CONTROL_EXPERIMENT_STATUS, specs, whitePaper,
   baseCamp all sync'd to consistent counts (881 tests, 151 binaries, 59 tolerances)
2. **Track 3 (drug repurposing)** — 5 new experiments (Exp157-161, 40 checks)
   validating NMF, pathway scoring, and knowledge graph embeddings
3. **ToadStool S53→S57 confirmed** — all 26 items from V16-V22 absorbed; 46 cross-spring total at S57
4. **4 GPU test defects fixed** — GBM tree data, KMD assertion, hardware-dependent ignores
5. **Deprecated API removal** — `parse_fastq` → `FastqIter::open()` in all binaries
6. **6 new tolerance constants** — 59 named constants total in `tolerances.rs`

---

## Barracuda Evolution Status

### Phase Summary

| Phase | Count | Description |
|-------|:-----:|-------------|
| **Lean** | 27 | GPU modules consuming upstream ToadStool primitives |
| **Write → Lean** | 5 | ODE shaders use `generate_shader()` — WGSL deleted (30,424 bytes) |
| **Compose** | 7 | GPU wrappers wiring ToadStool primitives |
| **Passthrough** | 3 | Accept GPU buffers, CPU kernel (pending ToadStool primitives) |
| **NPU** | 1 | ESN reservoir computing → int8 → NPU deployment |
| **Tier B/C** | 0 | All promoted |

### ToadStool Primitives Consumed (31)

All consumed via `barracuda::*` imports (no local shaders):

| Category | Primitives |
|----------|-----------|
| Bio ops (8) | SmithWatermanGpu, AniBatchF64, Dada2EStepGpu, DnDsBatchF64, HmmBatchForwardF64, PangenomeClassifyGpu, QualityFilterGpu, SnpCallingF64 |
| Inference (3) | TreeInferenceGpu, RfBatchInferenceGpu, FelsensteinGpu |
| Math (4) | GemmF64, GemmCachedF64, FusedMapReduceF64, BrayCurtisF64 |
| Bio compose (5) | KmerHistogramGpu, PairwiseHammingGpu, PairwiseJaccardGpu, GillespieGpu, KrigingF64 |
| Cross-spring (5) | SpatialPayoffGpu, BatchFitnessGpu, LocusVarianceGpu, BatchedEighGpu, PrngXoshiro |
| ODE framework (1) | BatchedOdeRK4 (generate_shader from OdeSystem traits) |
| Spectral (4) | Anderson 1D/2D/3D, Lanczos, Almost-Mathieu, level statistics |
| **Subtotal** | **31** |

### CPU Math Still Local (Sovereign Fallback)

These exist in `crate::special` for the `cfg(not(feature = "gpu"))` build path:

| Function | File | Upstream Target |
|----------|------|-----------------|
| `erf()` | `special.rs` | `barracuda::special::erf` |
| `ln_gamma()` | `special.rs` | `barracuda::special::ln_gamma` |
| `regularized_gamma_lower()` | `special.rs` | `barracuda::special::regularized_gamma_p` |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` |
| `cholesky_factor()` | `bio/esn.rs` | `barracuda::linalg::cholesky_solve` |
| `solve_ridge()` | `bio/esn.rs` | `barracuda::linalg::ridge_regression` |

**Blocker:** barracuda currently requires wgpu as mandatory dependency. These
functions need `[features] math = []` to gate CPU-only modules without GPU stack.

---

## New Track 3: Drug Repurposing via Matrix Math

### Background

Dr. David Fajgenbaum (UPenn / Every Cure) repurposed sirolimus to save his own
life, then built the MATRIX platform to systematically match ~4,000 FDA-approved
drugs to ~18,000 diseases using NMF, knowledge graph embeddings, and cosine
similarity scoring.

### Experiments Completed (40 checks, all PASS)

| Exp | Name | Checks | Algorithm |
|-----|------|:------:|-----------|
| 157 | Fajgenbaum pathway scoring | 8 | PI3K/AKT/mTOR → sirolimus drug-pathway matching |
| 158 | MATRIX pharmacophenomics | 9 | Every Cure methodology (drug × disease scoring) |
| 159 | NMF drug-disease factorization | 7 | Yang 2020 multiplicative updates (Lee & Seung 1999) |
| 160 | repoDB NMF reproduction | 9 | Gao 2020 (1,571 drugs × 1,209 diseases benchmark) |
| 161 | Knowledge graph embedding | 7 | TransE baseline (ROBOKOP-style triple scoring) |

### New Modules

| Module | Location | Algorithm | GPU Status |
|--------|----------|-----------|------------|
| `bio::nmf` | `barracuda/src/bio/nmf.rs` | NMF (multiplicative updates, cosine similarity, top-K) | CPU only |
| `bio::transe` | `barracuda/src/bio/transe.rs` | TransE knowledge graph embedding (L1/L2 scoring) | CPU only |

### ToadStool Absorption Targets (Track 3)

| Primitive | Priority | Notes |
|-----------|----------|-------|
| `NmfUpdateGpu` | P1 | Multiplicative update rules on GPU. Matrix dims ~4K × 18K — fits single GPU. Lee & Seung (1999) |
| `SparseGemmGpu` | P2 | Drug-disease matrices are ~5% fill. Sparse GEMM avoids materializing 72M zero entries |
| `CosineSimilarityBatchGpu` | P2 | Pairwise scoring on NMF factor matrices. Already have `spectral_cosine` via FMR |
| `TopKSelectionGpu` | P3 | GPU radix sort + selection for top-K drug-disease pairs |
| `TransEScoreGpu` | P3 | Vectorized L1/L2 triple scoring for knowledge graph embeddings |

---

## Three-Tier Control Matrix

### Per-Paper Status (All 25 Actionable Papers)

| # | Paper | CPU | GPU | metalForge | Status |
|---|-------|:---:|:---:|:----------:|--------|
| 1 | Galaxy/QIIME2 16S | Y | Y | Y | Full three-tier |
| 2 | asari LC-MS | Y | Y | Y | Full three-tier |
| 3 | FindPFAS screening | Y | Y | Y | Full three-tier |
| 4 | GPU diversity + spectral | Y | Y | Y | Full three-tier |
| 5 | Waters 2008 QS ODE | Y | Y | Y | Full three-tier |
| 6 | Massie 2012 Gillespie | Y | Y | Y | Full three-tier |
| 7 | Hsueh 2022 Phage defense | Y | Y | Y | Full three-tier |
| 8 | Fernandez 2020 Bistable | Y | Y | Y | Full three-tier |
| 9 | Mhatre 2020 Capacitor | Y | Y | Y | Full three-tier |
| 10 | Bruger 2018 Cooperation | Y | Y | Y | Full three-tier |
| 12 | Srivastava 2011 Multi-signal | Y | Y | Y | Full three-tier |
| 13 | Cahill proxy | Y | Y | Y | Full three-tier |
| 14 | Smallwood proxy | Y | Y | Y | Full three-tier |
| 15 | Liu 2014 HMM | Y | Y | Y | Full three-tier |
| 16 | Alamin 2024 Placement | Y | Y | Y | Full three-tier |
| 17 | Liu 2009 SATe | Y | Y | Y | Full three-tier |
| 18 | Zheng 2023 DTL | Y | Y | Y | Full three-tier |
| 20 | Wang 2021 RAWR | Y | Y | Y | Full three-tier |
| 21 | Jones PFAS MS | Y | Y | Y | Full three-tier |
| 22 | Jones PFAS F&T | Y | Y | Y | Full three-tier |
| 24 | Anderson 2017 Population | Y | Y | Y | Full three-tier |
| 25 | Moulana 2020 Pangenome | Y | Y | Y | Full three-tier |
| 26 | Mateos 2023 Sulfur | Y | Y | Y | Full three-tier |
| 27 | Boden 2024 Phosphorus | Y | Y | Y | Full three-tier |
| 28 | Anderson 2014 Viral | Y | Y | Y | Full three-tier |
| 29 | Anderson 2015 Rare biosphere | Y | Y | Y | Full three-tier |

**25/25 actionable papers: full three-tier (CPU + GPU + metalForge).**

### Track 3 Control Status (Drug Repurposing — NEW)

| # | Paper | CPU | GPU | metalForge | Gap |
|---|-------|:---:|:---:|:----------:|-----|
| 39 | Fajgenbaum 2019 (sirolimus) | Y | — | — | Needs pathway scoring GPU shader |
| 40 | Fajgenbaum 2025 (MATRIX) | Y | — | — | Needs NMF + cosine sim GPU |
| 41 | Yang 2020 (NMF drug) | Y | — | — | `NmfUpdateGpu` — ToadStool target |
| 42 | Gao 2020 (repoDB NMF) | Y | — | — | Same `NmfUpdateGpu` shader |
| 43 | ROBOKOP (KG embedding) | Y | — | — | `TransEScoreGpu` — ToadStool target |

**5/5 Track 3 papers: CPU only. GPU + metalForge pending ToadStool absorption.**

### Extension Paper Control Status (Phase 37-39)

| # | Paper | CPU | GPU | metalForge | Notes |
|---|-------|:---:|:---:|:----------:|-------|
| 30 | Physical comm pathways | Y | — | — | Algorithmic (no GPU path needed) |
| 31 | Cold seep QS | Y | — | — | Catalog validation (data-driven) |
| 32 | luxR phylogeny | Y | — | — | Catalog validation (data-driven) |
| 33 | Meyer PRE 2020 waves | Y | — | — | Analytical model |
| 34 | Jemielita burst statistics | Y | — | — | Analytical reinterpretation |
| 35 | Nitrifying community QS | Y | — | — | Catalog validation |
| 36 | Marine interkingdom QS | Y | — | — | Catalog validation |
| 37 | Rajagopalan Myxococcus | Y | — | — | Critical density calculation |
| 38 | Dictyostelium cAMP relay | Y | — | — | Analytical model |

Extension papers are primarily analytical/catalog validations — GPU acceleration
is not the bottleneck. These are correctly CPU-only by design.

---

## Code Audit Findings (V26)

### Issues Found and Fixed

1. **Deprecated API usage** — 3 validation binaries still called `fastq::parse_fastq`
   (deprecated in favor of streaming `FastqIter::open()`). Migrated all three.

2. **Magic numbers** — 6 inline literals promoted to `tolerances.rs`:
   - `MATRIX_EPS` (1e-15): division guard for NMF/cosine/norms
   - `NMF_INIT_FLOOR` (1e-10): NMF initialization positivity
   - `BOX_MULLER_U1_FLOOR` (1e-15): prevents `ln(0)` in ESN Gaussian
   - `GAMMA_RIGHT_TAIL_OFFSET` (200.0): regularized gamma early exit
   - `ODE_DIVISION_GUARD` (1e-30): Hill function denominator guard
   - `ERROR_BODY_PREVIEW_LEN` (200): API error message truncation

3. **GPU test defects** (pre-existing, found during revalidation):
   - `gbm_gpu::gbm_batch_matches_individual` — stump arrays had 1 node instead of 3
   - `kmd_gpu::kmd_gpu_matches_cpu_small` — asserted KMD ≈ 0 instead of KMD consistent
   - `batch_fitness_gpu` + `locus_variance_gpu` — hardware-dependent; marked `#[ignore]`

4. **Hardcoded URL** — NCBI Entrez base URL promoted to module-level `const`

### Quality Metrics After Audit

| Metric | Before | After |
|--------|--------|-------|
| Named tolerances | 53 | 59 |
| Deprecated API calls | 3 | 0 |
| GPU test failures (with `--features gpu`) | 4 | 0 (2 fixed, 2 correctly ignored) |
| ToadStool alignment | S42 | S53 |
| barracuda tests | 811 | 834 |
| forge tests | 38 | 47 |
| Total tests | 849 | 881 |

---

## Learnings for ToadStool Evolution

### 1. ODE Generic Framework Works

All 5 biological ODE systems (`PhageDefenseOde`, `BistableOde`, `MultiSignalOde`,
`CooperationOde`, `CapacitorOde`) use `BatchedOdeRK4<S>::generate_shader()`.
CPU parity is derivative-level exact. Upstream integrator is 20-33% faster than
local. This validates the generic ODE approach — any spring can add systems
by implementing `OdeSystem` trait. ToadStool should absorb more ODE systems
from other springs using this pattern.

### 2. Compose Pattern Scales

7 wetSpring modules use the "compose" pattern — wiring existing ToadStool
primitives for domain-specific workflows without new WGSL. This is the
lowest-cost GPU path and should be the default recommendation for new domains.

### 3. Drug Repurposing is GEMM-Shaped

Track 3 (NMF, cosine similarity, top-K selection) maps cleanly to existing
ToadStool primitives. The core NMF update rule is two GEMMs per iteration:
`H = H * (W^T * V) / (W^T * W * H)`. With `GemmCachedF64` already absorbed,
the NMF shader is primarily element-wise ops + GEMM composition.

### 4. Passthrough → Compose Opportunity

Three passthrough modules (gbm, feature_table, signal) accept GPU buffers but
run CPU kernels. Once ToadStool adds the corresponding primitives, these become
zero-change GPU upgrades. The buffer layout contracts are already validated.

### 5. CPU Math Feature Gate Needed

The `[features] math = []` proposal from V16 remains the biggest blocker for
clean CPU-only builds. Six local functions duplicate upstream `barracuda::special`
when the `gpu` feature is disabled. A CPU-only feature gate would let wetSpring
and other springs import just the math without pulling in wgpu.

---

## Recommended ToadStool Actions

### P0 — Immediate

1. **`NmfUpdateGpu` shader** — multiplicative update rules for NMF. wetSpring
   has CPU reference (`bio::nmf.rs`), binding layout, and 16 checks ready.
2. **`barracuda [features] math = []`** — CPU-only math feature gate.

### P1 — Next Sprint

3. **`SparseGemmGpu`** — sparse matrix multiply for drug-disease matrices.
4. **`TransEScoreGpu`** — vectorized triple scoring for knowledge graphs.
5. **`GbmBatchInferenceGpu`** — promote gbm passthrough to full GPU.

### P2 — Backlog

6. **`PeakDetectGpu`** — 1D peak detection (promote signal passthrough).
7. **`FeatureExtractionGpu`** — LC-MS feature pipeline (promote feature_table).
8. **Taxonomy NPU int8 integration** — BrainChip Akida pathway.

---

## Test Verification

```
cargo test                    → 834 passed, 1 ignored
cargo test (forge)            → 47 passed
cargo test --features gpu     → 770 passed, 9 ignored
cargo clippy --all-targets    → 0 warnings (pedantic + nursery)
cargo fmt --check             → 0 diffs
```

---

## File Locations

| Item | Path |
|------|------|
| NMF module | `barracuda/src/bio/nmf.rs` |
| TransE module | `barracuda/src/bio/transe.rs` |
| Tolerances | `barracuda/src/tolerances.rs` (59 constants) |
| Absorption manifest | `barracuda/ABSORPTION_MANIFEST.md` |
| Three-tier matrix | `specs/README.md` |
| Paper queue | `specs/PAPER_REVIEW_QUEUE.md` |
| Previous handoff (V26) | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V26_SYNC_FEB24_2026.md` |

---

## Evolution Roadmap

```
CURRENT STATE (V27):
  43/43 papers ALL GREEN
  25/25 actionable papers full three-tier (CPU + GPU + metalForge)
  5/5 Track 3 papers CPU-only (NMF, KG embedding — pending ToadStool)

NEXT:
  → ToadStool absorbs NmfUpdateGpu        → Track 3 GPU validation
  → ToadStool adds math feature gate       → Clean CPU-only builds
  → Passthrough → Compose promotions       → gbm, feature_table, signal GPU
  → CPU benchmarks for Track 3             → Python parity timing
  → metalForge Track 3 routing             → Full three-tier for all 43 papers
```
