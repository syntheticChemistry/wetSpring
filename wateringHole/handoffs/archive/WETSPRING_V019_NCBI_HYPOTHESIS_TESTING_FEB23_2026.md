# wetSpring v19 — NCBI-Scale Hypothesis Testing Handoff

**Date:** February 23, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-or-later
**Previous:** V18 — Cross-Spring Rewire (Phase 34, Feb 23)
**Context:** Phase 35 extends validated pipelines to real NCBI data with
GPU-confirmed results. Six new experiments (Exp121-126) test hypotheses
against 200 Vibrio assemblies, 158 Campylobacterota assemblies, 136
BioProjects, and 2D/1D Anderson localization comparisons.

---

## Part 1: Executive Summary

| Metric | Value |
|--------|-------|
| Experiments | 126 (6 new) |
| Validation checks | 2,819+ (146 new, GPU-confirmed) |
| Rust tests | 750 (676 lib + 60 integration + 14 doc) |
| NCBI assemblies fetched | 200 Vibrio + 158 Campylobacterota + 136 BioProjects |
| New binaries | 6 (validate_ncbi_vibrio_qs, validate_anderson_2d_qs, validate_temporal_esn_bloom, validate_npu_spectral_triage, validate_ncbi_pangenome, validate_ncbi_qs_atlas) |
| New Rust module | `bio::ncbi_data` (JSON loader with synthetic fallback) |
| New ESN methods | `train_stateful`, `train_stateless` |
| GPU experiments confirmed | Exp121 (14/14), Exp122 (12/12), Exp126 (90/90) |
| cargo check --features gpu | 0 warnings |
| cargo test --lib | 676 passed, 0 failed |

### Key Phase 35 findings

1. **Real Vibrio genomes cluster in biofilm space** (Exp121) — 200/200 assemblies
   produce biofilm at ODE equilibrium; the synthetic grid's planktonic/extinction
   regions are not occupied by real genomes
2. **2D Anderson has extended QS-active plateau** (Exp122) — 8 sweep points above
   midpoint for W>2, absent in 1D; bloom communities are QS-active in 2D but
   suppressed in 1D; critical J_c ≈ 0.41
3. **NPU spectral triage achieves 100% recall** (Exp124) — 20% pass rate, 3.7×
   speedup, perfect top-1 matching in two-stage NPU→GPU pipeline
4. **28-biome QS atlas validated** (Exp126) — disorder W monotonic with Pielou J
   across all biomes; all 28 biomes correctly ordered by QS suppression potential

---

## Part 2: New ToadStool Consumption

### 2.1 New Primitives Used

| Primitive | Experiment | Usage |
|-----------|-----------|-------|
| `OdeSweepGpu` | Exp121 | 200-batch Vibrio QS parameter sweep |
| `anderson_2d` | Exp122 | 20×20 lattice construction |
| `lanczos` / `lanczos_eigenvalues` | Exp122, Exp126 | 2D spectral analysis |
| `level_spacing_ratio` | Exp122, Exp126 | QS regime classification |
| `anderson_hamiltonian` | Exp122, Exp126 | 1D baseline sweep |

### 2.2 New wetSpring Module: `bio::ncbi_data`

Loads real NCBI JSON data (produced by `scripts/fetch_ncbi_phase35.py`) with
deterministic synthetic fallback for offline/CI execution. Structs:

- `VibrioAssembly` — accession, organism, genome_size_bp, gene_count, scaffold_count, isolation_source
- `CampyAssembly` — same fields for Campylobacterota
- `BiomeProject` — accession, title, biome_type, n_samples, region

Minimal JSON parser (no `serde_json` in prod path) — `json_str_value`,
`json_int_value`, `split_json_objects`.

### 2.3 ESN Extensions

`train_stateful` and `train_stateless` methods added to `bio::esn::Esn`:
- Stateful: drives reservoir through trajectory sequences, resets only between trajectories
- Stateless: resets reservoir before each sample (independent classification)

---

## Part 3: Absorption Recommendations

### 3.1 ESN Matrix Ridge Regression

Current ESN uses diagonal ridge regression (λI). Exp123 showed stateful and
stateless ESN achieve identical accuracy (45%) — the temporal memory advantage
requires proper matrix ridge regression. **Recommend**: ToadStool ESN v2 with
full W_out = Y·X^T·(X·X^T + λI)^{-1} training.

### 3.2 Anderson 3D Extension

Exp122 demonstrated the 2D extended plateau. Exp107 Section 5 showed 3D
produces ⟨r⟩ = 0.4843 at W=2 (most GOE-like value). **Recommend**: 3D
Anderson sweep binary using `anderson_3d` to map the full 1D→2D→3D QS-active
window progression. Natural test case: hydrothermal vent chimney geometry.

### 3.3 NCBI Data Pipeline for CI

The `fetch_ncbi_phase35.py` script and `bio::ncbi_data` fallback pattern
should be standardized. **Recommend**: ToadStool `data_cache` module with
fetch→cache→fallback lifecycle for any external data source.

---

## Part 4: Handoff Artifact Inventory

| Artifact | Location | Purpose |
|----------|----------|---------|
| Python fetch script | `scripts/fetch_ncbi_phase35.py` | NCBI Datasets v2 + Entrez API |
| Rust data loader | `barracuda/src/bio/ncbi_data.rs` | JSON parse + synthetic fallback |
| Exp121 binary | `barracuda/src/bin/validate_ncbi_vibrio_qs.rs` | Vibrio QS GPU sweep |
| Exp122 binary | `barracuda/src/bin/validate_anderson_2d_qs.rs` | 2D Anderson lattice |
| Exp123 binary | `barracuda/src/bin/validate_temporal_esn_bloom.rs` | Stateful ESN bloom |
| Exp124 binary | `barracuda/src/bin/validate_npu_spectral_triage.rs` | NPU→GPU triage |
| Exp125 binary | `barracuda/src/bin/validate_ncbi_pangenome.rs` | Campylobacterota pangenome |
| Exp126 binary | `barracuda/src/bin/validate_ncbi_qs_atlas.rs` | 28-biome QS atlas |
| Phase 35 design | `experiments/PHASE_35_DESIGN.md` | Hypotheses and methodology |
| baseCamp briefings | `whitePaper/baseCamp/*.md` | Per-faculty results |

---

## Part 5: Open Questions for Next Work

1. **3D Anderson sweep**: Does the 3D metal-insulator transition create an
   even wider QS-active window? At what disorder W_c does the 3D transition
   occur, and does it map to an ecologically meaningful Pielou J?

2. **Hot spring geochemistry as 3D lattice model**: Hydrothermal vent chimney
   porosity and permeability create a natural 3D lattice. Can we map real
   chimney geometry parameters (porosity, pore connectivity) to Anderson
   lattice connectivity to predict QS propagation in vent communities?

3. **Real-time NCBI monitoring**: The fetch→cache→fallback pattern works for
   batch analysis. Can we extend to streaming NCBI updates (new assemblies
   trigger automated re-parameterization of QS landscape)?

4. **ESN temporal memory**: ToadStool ESN v2 with matrix ridge regression
   should unlock the stateful advantage for bloom detection. Predicted:
   2–4 window early detection gain.
