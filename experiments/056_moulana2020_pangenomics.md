# Experiment 056 — Moulana & Anderson 2020: Sulfurovum Pangenomics

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Moulana, Anderson et al. (2020) "Selection is a significant driver of gene gain and loss in the pangenome of Sulfurovum" mSystems 5:e00673-19
**DOI:** 10.1128/mSystems.00673-19
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate wetSpring's pangenome analysis primitives against gene gain/loss
analysis in Sulfurovum populations. This paper uses gene clustering,
presence-absence matrices, and phylogenomics to show that selection drives
pangenome dynamics at deep-sea vents.

## Data

- **BioProjects:** PRJNA283159 (Mid-Cayman Rise, 9 MAGs) +
  PRJEB5293 (Axial Seamount, 13 MAGs)
- **MAGs:** 22 Sulfurovum genomes (70-97% complete)
- **Supplementary:** Gene cluster data in Data Set S1

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Gene clustering (CD-HIT style) | **NEW** `bio::pangenome` | **Needed** |
| Presence-absence matrix | **NEW** `bio::pangenome` | **Needed** |
| Core/accessory genome partitioning | **NEW** `bio::pangenome` | **Needed** |
| Phylogenomics (concatenated alignment) | `bio::felsenstein` + `bio::alignment` | Existing |
| dN/dS per gene cluster | `bio::dnds` | From Exp052 |
| COG enrichment (hypergeometric test) | **NEW** `bio::enrichment` | **Needed** |
| Binomial CDF (geographic bias) | Standard math | Simple |

## Validation Design

### Phase 1: Pangenome module
- New `bio::pangenome` module:
  - Gene identity clustering (greedy, threshold-based)
  - Presence-absence binary matrix
  - Core genome (present in all), accessory (present in some), unique (one only)
  - Heap's law fit (power law for genome openness)
- Validate against analytical known-values (synthetic gene sets)

### Phase 2: Enrichment testing
- New `bio::enrichment` module:
  - Hypergeometric test (Fisher exact)
  - Multiple testing correction (Benjamini-Hochberg)
- Validate against scipy.stats.hypergeom

### Phase 3: Integrated pangenome pipeline
- Parse Sulfurovum gene annotations
- Build presence-absence matrix
- Partition core/accessory/unique
- Verify Heap's law fit indicates open pangenome

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| Core genome size (synthetic) | Exact | 0 |
| Accessory genome size (synthetic) | Exact | 0 |
| Unique genome size (synthetic) | Exact | 0 |
| Presence-absence matrix dimensions | Exact | 0 |
| Heap's law alpha < 1 (open pangenome) | Boolean | 0 |
| Hypergeometric p-value (known) | Analytical | 1e-10 |
| BH correction monotonic | Boolean | 0 |
| dN/dS per cluster (3+ clusters) | Analytical | 1e-10 |
| Python parity (pangenome, enrichment) | Analytical | 1e-10 |
| Deterministic clustering | Boolean | 0 |

**Estimated checks:** ~20

## Python Baseline

`scripts/moulana2020_pangenomics.py`

## Rust Validation Binary

`barracuda/src/bin/validate_pangenomics.rs`
