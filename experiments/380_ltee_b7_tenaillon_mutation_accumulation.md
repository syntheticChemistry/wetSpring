<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Exp380: LTEE B7 — Tenaillon 2016 Mutation Accumulation Pipeline

**Status:** TIER 2 COMPLETE (V165, current V174)
**Paper:** Tenaillon et al. "Tempo and mode of genome evolution in a
50,000-generation experiment" *Nature* 536, 165–170 (2016)
**BioProject:** PRJNA294072
**lithoSpore Module:** 6 (breseq comparison)
**LTEE Queue ID:** B7

---

## Objective

Reproduce Tenaillon 2016's mutation accumulation analysis using wetSpring's
sovereign genomics pipeline. This is wetSpring's primary LTEE contribution —
264 *E. coli* genomes from 12 replicate populations across ~50,000 generations,
analyzed for mutation rate, spectrum, and accumulation dynamics.

---

## Pipeline Tiers

### Tier 1: Python Baseline

1. Download 264 genomes from NCBI BioProject PRJNA294072 via `ncbi/efetch.rs`
   sovereign pipeline (or `scripts/ncbi_bulk_download.sh`)
2. Parse genome annotations, extract mutation calls per lineage
3. Compute mutation accumulation curves (mutations vs generations)
4. Compute mutation rate estimates per population
5. Produce expected values JSON for lithoSpore module 6

**Deliverables:** ✅ Complete (V164)
- `notebooks/papers/tenaillon-ltee-mutation.ipynb` — Python baseline ✅
- `experiments/results/ltee_b7_expected_values.json` — expected values (10 targets) ✅

### Tier 2: Rust Validation Binary

1. `validate_ltee_b7_mutation_accumulation.rs` — Rust binary reproducing
   Python baseline results with documented tolerances
2. Mutation accumulation curve fitting (linear + power-law models)
3. Spectrum analysis: transition/transversion ratios per population
4. Cross-check against breseq-called variants (when available)

**Deliverables:** ✅ Complete (V165)
- `barracuda/src/bin/validate_ltee_b7_v1.rs` — 27/27 checks PASS ✅
- Mutation accumulation curve: 9 time points, R² = 0.999985 ✅
- Spectrum: 6-class with sum = 1.0000 ✅
- Linear slope matches rate prediction within 5% ✅

### Tier 3: lithoSpore Integration

1. Expected values JSON consumed by lithoSpore module 6
2. Mutation accumulation curves as validation targets
3. Feeds groundSpring epistasis quantification (B2 cross-link)

---

## Key Quantities to Reproduce

| Quantity | Paper Value | Notes |
|----------|-------------|-------|
| Populations | 12 | Ara-1 through Ara-6, Ara+1 through Ara+6 |
| Genomes | 264 | ~22 per population across time points |
| Mean mutation rate | ~8.9 × 10⁻¹¹ per bp per generation | Fig 1, varies by population |
| Mutation spectrum | Ts:Tv ~1.7 | Table S2, strong G:C→A:T bias |
| Accumulation model | Near-linear with population-specific variation | Fig 2, some populations show nonlinear dynamics |

---

## Data Provenance

| Source | Accession | License |
|--------|-----------|---------|
| NCBI SRA | PRJNA294072 | Public domain |
| Paper | doi:10.1038/nature18959 | Open access |

---

## Dependencies

- `ncbi/efetch.rs` — genome download (sovereign pipeline, env-configurable)
- `bio/alignment.rs` — sequence comparison
- `bio/snp.rs` — variant calling
- `bio/dnds.rs` — mutation spectrum analysis
- NestGate (PG-04) — optional caching layer for bulk downloads
