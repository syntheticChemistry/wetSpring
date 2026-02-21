# Experiment 053 — Mateos & Anderson 2023: Sulfur-Cycling Enzyme Phylogenomics

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Mateos, Anderson et al. (2023) "The evolution and spread of sulfur-cycling enzymes reflect the redox state of the early Earth" Science Advances 9:eade4847
**DOI:** 10.1126/sciadv.ade4847
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate wetSpring's tree reconciliation and phylogenetic primitives against
deep-time enzyme evolution analysis. This paper traces sulfur-cycling enzymes
across 3+ billion years using gene-tree/species-tree reconciliation — the same
DTL framework validated in Exp034 (Zheng 2023), now applied to real
geobiological data. Undergraduate co-authors suggest reproducible methods.

## Data

- **Figshare:** Project 144267
  (https://figshare.com/projects/The_evolution_and_spread_of_sulfur-cycling_enzymes_across_the_tree_of_life_through_deep_time/144267)
- **Content:** Newick gene trees, species trees, alignments, chronograms,
  fossil calibrations
- **Source sequences:** AnnoTree (public genome database)

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Gene tree parsing (Newick) | `bio::felsenstein` (parse_newick) | Existing |
| Robinson-Foulds distance | `bio::robinson_foulds` | Existing |
| DTL reconciliation | `bio::reconciliation` | Existing |
| Tree likelihood (Felsenstein) | `bio::felsenstein` | Existing |
| Bootstrap support | `bio::bootstrap` | Existing |
| Molecular clock rate estimation | **NEW** `bio::molecular_clock` | **Needed** |

## Validation Design

### Phase 1: Tree reconciliation on sulfur enzyme families
- Parse gene trees from Figshare data
- Run DTL reconciliation against reference species tree
- Validate HGT / duplication / loss event counts against paper's Table 1
- Robinson-Foulds distance between gene trees and species tree

### Phase 2: Molecular clock module
- New `bio::molecular_clock` module: strict and relaxed clock rate estimation
- Uncorrelated relaxed clock (lognormal, CIR) — the paper uses PhyloBayes
- Validate node ages against fossil calibration constraints

### Phase 3: Python baseline comparison
- Python script using DendroPy for tree operations
- Ete3 for reconciliation verification
- Compare DTL event counts and RF distances

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| Newick parse (gene trees) | Exact | 0 |
| Newick parse (species tree) | Exact | 0 |
| RF distance (gene vs species) > 0 | Boolean | 0 |
| DTL cost optimal | Analytical | 0 |
| HGT event count | Exact | 0 |
| Duplication event count | Exact | 0 |
| Loss event count | Exact | 0 |
| Clock rate positive | Boolean | 0 |
| Node ages within calibration bounds | Boolean | 0 |
| Python parity (RF, DTL) | Analytical | 1e-12 |
| Deterministic reconciliation | Boolean | 0 |

**Estimated checks:** ~20

## Python Baseline

`scripts/mateos2023_sulfur_phylogenomics.py`

## Rust Validation Binary

`barracuda/src/bin/validate_sulfur_phylogenomics.rs`
