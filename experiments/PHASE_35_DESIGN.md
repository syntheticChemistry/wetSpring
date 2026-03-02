# Phase 35: NCBI-Scale Hypothesis Testing & Extended Systems
> *Fossil record — design completed; all experiments implemented and passing.*

**Date:** February 23, 2026
**Phase:** 35 — "Real Data, Real Hypotheses"
**Predecessor:** Phase 34 (cross-spring rewire, Exp120)
**Experiments:** Exp121–126
**NCBI API Key:** `ecoPrimals/testing-secrets/api-keys.toml` → `ncbi_api_key`

---

## Motivation

Phases 29–34 built and validated a three-tier CPU → GPU → NPU pipeline across
120 experiments. But every experiment uses synthetic data mirroring real
distributions. Phase 35 bridges to reality: real NCBI genome/metagenome
metadata drives parameter spaces, and new physics (2D Anderson) unlocks
predictions that 1D models cannot make.

Six hypotheses, each extending a validated experiment:

| Exp | Extends | Hypothesis | Hardware |
|-----|---------|-----------|----------|
| 121 | Exp108 | Real Vibrio genomic diversity produces wider, more bistable QS parameter landscapes than synthetic grids | GPU |
| 122 | Exp113 | 2D Anderson lattice exhibits a critical diversity threshold for QS activation absent in 1D | CPU |
| 123 | Exp118 | Stateful (memory-carrying) ESN detects pre-bloom 2–4 windows earlier than stateless | NPU |
| 124 | Exp111/117 | NPU spectral triage reduces GPU compute by >80% while retaining >95% recall | NPU+GPU |
| 125 | Exp110 | Real cross-ecosystem pangenomes show environment-specific accessory gene enrichment | CPU |
| 126 | Exp113/119 | Biomes cluster in Anderson disorder-space, creating a predictive global QS-activity atlas | GPU |

---

## Data Pipeline

### NCBI Access

Script: `scripts/fetch_ncbi_phase35.py`

```
NCBI Entrez (10 req/s with API key)
  → Assembly DB: Vibrio (Exp121), Campylobacterota (Exp125)
  → BioProject DB: 16S amplicon surveys from diverse biomes (Exp126)
  → SRA DB: bloom monitoring time-series (Exp123)
```

Data lands in `data/ncbi_phase35/` as JSON summaries. Rust binaries read
these summaries for real parameter derivation; fall back to enhanced synthetic
data when offline.

### Fallback Strategy

Every binary must pass with synthetic data alone (CI/offline mode). Real NCBI
data narrows parameter ranges and adds provenance, but does not change the
validation logic.

---

## Exp121: Real Vibrio QS Parameter Landscape

### Background

Exp108 swept a uniform 32×32 grid over `mu_max` and `k_ai_prod`. Real Vibrio
genomes have correlated parameters: fast growers tend to have simpler QS
circuits, and environmental strains have different AI production kinetics than
clinical isolates.

### Hypothesis

**H121:** Real Vibrio genomic diversity produces a QS parameter landscape with:
1. Wider `k_ai_prod` range than the synthetic 1.0–10.0 grid
2. >15% of parameter sets in bistable regions (vs ~10% synthetic)
3. Clinical vs environmental isolate clustering in parameter space
4. Correlation between genome size and QS circuit complexity

### Design

1. **Data source:** NCBI Assembly DB — search `Vibrio[Organism]`, filter for
   complete/chromosome-level assemblies. Extract: genome size, gene count,
   scaffold count, organism name (species/strain), isolation source.
2. **Parameter derivation:**
   - `mu_max` ∝ inverse genome size (smaller genomes → faster growth, Vieira-Silva & Rocha 2010)
   - `k_ai_prod` ∝ QS gene count (luxI/luxR homologs) — proxy from total gene count
   - `k_hapr_ai` ∝ operon density (genes/Mbp as complexity proxy)
3. **Pipeline:** Derived params → `QsBiofilmParams` → GPU ODE sweep → classify outcomes
4. **Validation:** Compare regime distribution to Exp108; test clinical/environmental clustering

### Checks (~15)

- S1: NCBI fetch returns ≥100 assemblies (or synthetic fallback activates)
- S2: Parameter derivation produces valid `QsBiofilmParams` for all entries
- S3: GPU sweep completes, all trajectories finite
- S4: Bistability fraction measured, compared to Exp108 baseline
- S5: Clinical vs environmental regime distribution tested

---

## Exp122: 2D Anderson Spatial QS Lattice

### Background

Exp113 used 1D Anderson localization as an analogy for QS signal propagation.
But 1D Anderson is trivial: all states are localized for any disorder W > 0.
In 2D, there is critical behavior at W_c ≈ 5.3 (symplectic class) or a
logarithmic localization length divergence (orthogonal class). This creates a
genuine phase transition that 1D cannot capture.

### Hypothesis

**H122:** A 2D Anderson lattice with disorder W derived from microbial
community diversity (Pielou J → W mapping) exhibits:
1. A critical diversity threshold J_c where QS signaling transitions from
   localized (suppressed) to extended (active)
2. Level spacing ratio ⟨r⟩ crossing from Poisson (0.386) to GOE (0.531) at W_c
3. Sharper QS-active/QS-suppressed transition than 1D (quantified by
   transition width ΔW)
4. The critical threshold J_c maps to ecologically meaningful diversity
   (~0.4–0.6 Pielou evenness)

### Design

1. **Build 2D lattice:** L×L tight-binding Hamiltonian with random on-site
   potential V_i ∈ [-W/2, W/2], periodic boundary conditions
2. **Disorder sweep:** W from 0.5 to 20.0 (40 points), corresponding to
   Pielou J from 0.95 to 0.05
3. **Eigenvalue statistics:** Full diagonalization for L ≤ 30 (900 sites),
   Lanczos for L > 30. Compute ⟨r⟩ at each W.
4. **Comparison:** 1D chain of same total sites (N=L²) vs 2D lattice — same
   disorder, different topology
5. **QS mapping:** Identify J_c from ⟨r⟩ crossover, compare to real ecosystem
   diversity distributions

### Checks (~12)

- S1: 2D Hamiltonian construction correct (symmetric, correct dimension L²×L²)
- S2: W=0 clean limit → ⟨r⟩ ≈ GOE (0.531)
- S3: W=20 strong disorder → ⟨r⟩ ≈ Poisson (0.386)
- S4: ⟨r⟩ monotonically decreasing with W (ergodic → localized)
- S5: 2D transition width ΔW measurably narrower than 1D
- S6: Critical J_c falls in ecologically relevant range

---

## Exp123: Temporal ESN Bloom Cascade

### Background

Exp118 showed NPU-quantized ESN can classify bloom phases from single-window
diversity snapshots. But real bloom detection depends on temporal patterns:
gradual diversity decline → dominance spike → recovery. A stateless ESN
treats each window independently, missing the trajectory.

### Hypothesis

**H123:** A stateful (memory-carrying) ESN that preserves reservoir state
across consecutive time windows:
1. Detects pre-bloom state 2–4 windows earlier than stateless ESN
2. Achieves >50% pre-bloom recall (vs ~25% stateless in Exp118)
3. Maintains <5% false-positive rate for normal-state windows
4. NPU int8 quantization preserves the early-detection advantage
   (stateful-NPU still beats stateless-f64)

### Design

1. **Temporal data:** Generate multi-window bloom trajectories (50–100 windows
   per series, labeled: normal → pre-bloom → active → post-bloom → normal)
2. **Stateful ESN:** After each window, carry reservoir state to next window.
   Training: drive entire trajectory, ridge-regress on states.
3. **Stateless baseline:** Reset reservoir before each window (Exp118 approach)
4. **Comparison metrics:**
   - Detection latency: number of windows from true pre-bloom onset to first
     correct pre-bloom prediction
   - Pre-bloom recall: fraction of pre-bloom windows correctly identified
   - Normal specificity: fraction of normal windows correctly identified
5. **NPU path:** Quantize both stateful and stateless readout weights, compare

### Checks (~14)

- S1: Stateful ESN training converges (loss decreases with epochs)
- S2: Stateful f64 pre-bloom recall > stateless f64 recall
- S3: Detection latency: stateful < stateless by ≥2 windows
- S4: Normal specificity > 90% for both (no false-alarm regression)
- S5: NPU quantized stateful > NPU quantized stateless
- S6: Stateful-NPU recall > stateless-f64 recall (key: int8 + memory > f64 + no memory)
- S7: Energy estimate: stateful-NPU on coin-cell feasible (reservoir state = 200 × i8)

---

## Exp124: MassBank Full-Scale NPU Spectral Triage

### Background

Exp117 showed NPU-quantized spectral screening works at small scale (~1000
spectra). Real MassBank has >500,000 spectra. A two-stage pipeline —
NPU pre-filter (cheap, fast) → GPU precise cosine scoring (expensive, accurate)
— could dramatically reduce GPU work.

### Hypothesis

**H124:** NPU spectral triage at MassBank scale:
1. NPU int8 pre-filter reduces candidate spectra by >80% (passes <20% to GPU)
2. True-hit recall >95% (NPU rarely misses a genuine match)
3. End-to-end throughput >10× single-GPU-only pipeline
4. NPU energy per query <1 µJ (vs GPU ~1 mJ per query)

### Design

1. **Library:** Generate 50,000 synthetic spectra mirroring MassBank structure
   (m/z 50–1000, 10–500 peaks per spectrum, intensity distribution following
   log-normal)
2. **Queries:** 500 query spectra, each with a known true match + noise
3. **NPU stage:** ESN-encoded spectral fingerprint → int8 cosine distance →
   threshold → candidate list
4. **GPU stage:** Full f64 cosine scoring on candidates only
5. **Baseline:** GPU-only scoring of all 50,000 library spectra per query

### Checks (~11)

- S1: Library generation produces 50,000 valid spectra
- S2: NPU pre-filter pass rate <20% of library
- S3: True-hit recall >95%
- S4: GPU-only vs NPU+GPU parity for top-1 match identity
- S5: Throughput improvement quantified
- S6: Energy estimate: NPU triage savings

---

## Exp125: Real NCBI Campylobacterota Cross-Ecosystem Pangenome

### Background

Exp110 used synthetic pangenomes for vent/coastal/deep-sea. Real
Campylobacterota (formerly Epsilonproteobacteria) span hydrothermal vents,
gut microbiomes, and food-borne pathogens — genuinely distinct ecosystems with
different selective pressures.

### Hypothesis

**H125:** Real cross-ecosystem Campylobacterota pangenomes exhibit:
1. Environment-specific accessory gene fractions (vent: sulfur metabolism,
   gut: adhesion/secretion, food-borne: stress response)
2. Core genome fraction ≈ 25–35% regardless of ecosystem (invariant,
   confirming Exp110 synthetic finding)
3. Cross-ecosystem accessory overlap <15% (distinct adaptive repertoires)
4. Heap's law exponent α differs by ecosystem (open vs closing pangenome)

### Design

1. **Data source:** NCBI Assembly DB — search `Campylobacterota[Organism]`,
   extract metadata (isolation source, host, environment)
2. **Parameter derivation:** From assembly summaries, derive genome sizes,
   gene counts, N50, and group by ecosystem tag
3. **Synthetic mirroring:** Generate pangenomes that mirror real genome size
   and gene count distributions (since we can't do full annotation in Rust
   without external tools)
4. **Pipeline:** Pangenome analysis → core/accessory/unique classification →
   Heap's law fitting → cross-ecosystem comparison

### Checks (~13)

- S1: NCBI fetch returns ≥50 Campylobacterota assemblies (or fallback)
- S2: Ecosystem classification produces ≥3 groups
- S3: Core genome fraction per ecosystem within 20–40%
- S4: Accessory gene fractions differ significantly across ecosystems
- S5: Heap's law fit R² > 0.85 per ecosystem
- S6: Cross-ecosystem accessory overlap measured

---

## Exp126: Global QS-Disorder Atlas from NCBI 16S Surveys

### Background

Exp113 mapped synthetic community diversity (Pielou J) to Anderson disorder
(W) and showed QS regime classification. But the mapping was tested on 7
hand-designed ecosystems. Real NCBI 16S surveys cover hundreds of biomes
with measured alpha diversity.

### Hypothesis

**H126:** A global QS-disorder atlas built from real NCBI 16S survey metadata:
1. Biomes cluster in disorder-space (similar biomes → similar W)
2. Known high-QS environments (gut, biofilm, rhizosphere) cluster at low W
   (extended/active regime)
3. Known low-QS environments (deep ocean, extreme habitats) cluster at high W
   (localized/suppressed regime)
4. The Pielou J → Anderson W mapping predicts QS activity status with >70%
   accuracy when validated against literature

### Design

1. **Data source:** NCBI BioProject DB — search for 16S amplicon studies with
   diversity metadata. Extract: BioProject ID, organism, biome, sample count,
   reported diversity metrics.
2. **Diversity derivation:** For projects with raw counts: compute Pielou J
   directly. For projects with only summary stats: estimate from Shannon/richness.
3. **Atlas construction:** Map each biome to W via `evenness_to_disorder()`,
   compute ⟨r⟩ per ecosystem, classify QS regime.
4. **GPU pipeline:** Anderson Hamiltonian + eigenvalue computation for each
   biome's disorder level.
5. **Validation:** Cross-reference predicted QS regimes with published QS
   activity data from literature.

### Checks (~10)

- S1: NCBI fetch returns ≥20 BioProjects with diversity data (or fallback)
- S2: Pielou J computed for each biome
- S3: Anderson ⟨r⟩ computed per biome via GPU
- S4: Biome clustering in disorder-space measured (variance ratio)
- S5: Known high-QS biomes correctly classified as QS-active (>70%)
- S6: Known low-QS biomes correctly classified as QS-suppressed (>70%)

---

## Implementation Plan

### Data Pipeline

```
scripts/fetch_ncbi_phase35.py
  ├── Vibrio assemblies → data/ncbi_phase35/vibrio_assemblies.json
  ├── Campylobacterota assemblies → data/ncbi_phase35/campylobacterota_assemblies.json
  └── 16S BioProjects → data/ncbi_phase35/biome_16s_projects.json
```

### Experiment Binaries

```
barracuda/src/bin/
  ├── validate_ncbi_vibrio_qs.rs          (Exp121, --features gpu)
  ├── validate_anderson_2d_qs.rs          (Exp122, CPU-only)
  ├── validate_temporal_esn_bloom.rs      (Exp123, CPU-only)
  ├── validate_npu_spectral_triage.rs     (Exp124, CPU-only)
  ├── validate_ncbi_pangenome.rs          (Exp125, CPU-only)
  └── validate_ncbi_qs_atlas.rs           (Exp126, --features gpu)
```

### New Module

```
barracuda/src/bio/ncbi_data.rs
  - load_vibrio_assemblies() → Vec<VibrioAssembly>
  - load_campylobacterota() → Vec<CampyAssembly>
  - load_biome_projects() → Vec<BiomeProject>
  - Fallback: generate synthetic equivalents when data files absent
```

### Dependencies

No new crate dependencies. NCBI fetching stays in Python scripts (stdlib
urllib). Rust reads JSON data files via manual parsing (no serde_json for
CPU-only binaries; existing `json` feature for GPU binaries that already
have it).

---

## Success Criteria

Phase 35 succeeds when:
1. All 6 experiments pass with synthetic fallback data
2. NCBI fetch script produces valid data for ≥3 of 6 data sources
3. At least 1 hypothesis shows a measurable difference between real and
   synthetic parameter spaces
4. 2D Anderson (Exp122) demonstrates sharper transition than 1D
5. Stateful ESN (Exp123) shows measurable early-detection advantage
6. Total validation checks: 2,673 + ~75 = ~2,748+

---

## Timeline

| Step | Work |
|------|------|
| 1 | Design doc (this file) |
| 2 | NCBI fetch script + data module |
| 3 | Exp121 (Vibrio QS) + Exp122 (2D Anderson) |
| 4 | Exp123 (temporal ESN) + Exp124 (spectral triage) |
| 5 | Exp125 (pangenome) + Exp126 (QS atlas) |
| 6 | Update tracking docs + baseCamp briefings |
| 7 | Handoff V19 to ToadStool |
