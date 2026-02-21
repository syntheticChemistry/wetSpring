# Experiment 051 — Anderson 2015: Rare Biosphere at Deep-Sea Hydrothermal Vents

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Anderson, Sogin, Baross (2015) "Biogeography and ecology of the rare and abundant microbial lineages in deep-sea hydrothermal vents" FEMS Microbiol Ecol 91:fiu016
**DOI:** 10.1093/femsec/fiu016
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate wetSpring's diversity and rarefaction primitives against deep-sea vent
16S amplicon analysis. This paper asks: when does a microbial lineage constitute
signal vs. sampling noise? The computational methods (rarefaction, alpha/beta
diversity, rank-abundance) map directly to existing wetSpring modules.

## Data

- **Original data:** MBL darchive (https://darchive.mblwhoilibrary.org/handle/1912/7205)
  454 pyrosequencing 16S rRNA amplicons from Juan de Fuca Ridge vents
- **Proxy approach:** Use vent community profiles from supplementary Table S1
  (OTU abundance data) to validate diversity computations without requiring
  raw sequence download

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Alpha diversity (Shannon, Simpson, Chao1) | `bio::diversity` | Existing |
| Rarefaction curves | `bio::diversity::rarefaction_curve` | Existing |
| Beta diversity (Bray-Curtis) | `bio::diversity::bray_curtis_matrix` | Existing |
| PCoA ordination | `bio::pcoa` | Existing |
| Rank-abundance curves | `bio::diversity` (sorted counts) | Existing |
| Rare lineage identification (< 0.1% RA) | `bio::diversity::observed` + threshold | Existing |

## Validation Design

### Phase 1: Synthetic vent community (analytical)
- Construct 3 synthetic communities mimicking paper's vent profiles:
  - Piccard (high-temperature, low diversity)
  - Von Damm (moderate, Campylobacteria-dominated)
  - Background seawater (high diversity, many rare lineages)
- Known ground truth → exact validation of diversity metrics

### Phase 2: Rarefaction and rare lineage detection
- Rarefaction at multiple depths (100, 500, 1000, 5000, 10000 reads)
- Verify rarefaction curve saturation detection
- Rare lineage counting: lineages < 0.1% of total reads

### Phase 3: Python baseline comparison
- Python script calculates all metrics on same synthetic communities
- Rust must match within `tolerances::ANALYTICAL_F64`

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| Shannon per community (3) | Analytical | 1e-12 |
| Simpson per community (3) | Analytical | 1e-12 |
| Chao1 per community (3) | Analytical | 1e-12 |
| Observed features per community (3) | Exact | 0 |
| Rarefaction monotonic (3) | Boolean | 0 |
| Rarefaction saturation (background) | Boolean | 0 |
| Bray-Curtis matrix symmetry | Analytical | 1e-15 |
| BC(Piccard, background) > BC(VonDamm, background) | Boolean | 0 |
| PCoA eigenvalues non-negative | Boolean | 0 |
| Rare lineage count per community (3) | Exact | 0 |
| Python parity (Shannon, Simpson, BC) | Analytical | 1e-12 |

**Estimated checks:** ~25

## Python Baseline

`scripts/anderson2015_rare_biosphere.py`

## Rust Validation Binary

`barracuda/src/bin/validate_rare_biosphere.rs`
