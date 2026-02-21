# Experiment 054 — Boden & Anderson 2024: Phosphorus-Cycling Enzyme Phylogenomics

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Boden, Anderson et al. (2024) "Timing the evolution of phosphorus-cycling enzymes through geological time" Nature Communications 15:3703
**DOI:** 10.1038/s41467-024-47914-0
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate the same tree reconciliation + molecular clock methodology as Exp053
on an independent dataset (phosphorus enzymes instead of sulfur). Shares
Exp053's computational pipeline but provides independent validation of
reconciliation math on different gene families. Data from OSF.

## Data

- **OSF:** https://osf.io/vt5rw/
- **Content:** 865 genomes, phosphorus gene trees, reconciliation outputs,
  Tara Oceans regression data
- **Source sequences:** Public genome databases

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Gene tree parsing (Newick) | `bio::felsenstein` (parse_newick) | Existing |
| Robinson-Foulds distance | `bio::robinson_foulds` | Existing |
| DTL reconciliation | `bio::reconciliation` | Existing |
| Molecular clock (CIR, LN, UGAM) | `bio::molecular_clock` | From Exp053 |
| HGT cost optimization | `bio::reconciliation` | Existing |

## Validation Design

### Phase 1: Phosphorus gene tree reconciliation
- Parse gene trees from OSF data
- Run DTL reconciliation against reference species tree
- Compare HGT / duplication / loss counts to paper's supplementary data
- RF distances between phosphorus gene trees

### Phase 2: Cross-validate with Exp053
- Same reconciliation code on different input → exercises generality
- Verify HGT costs match paper's reported optimized values

### Phase 3: Tara Oceans regression proxy
- Paper correlates enzyme abundance with ocean chemistry
- Validate linear regression on published data points

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| Newick parse (phosphorus gene trees) | Exact | 0 |
| RF distance (gene vs species) | Analytical | 1e-12 |
| DTL reconciliation cost | Exact | 0 |
| HGT event count | Exact | 0 |
| Duplication + loss counts | Exact | 0 |
| Clock node ages plausible (> 0) | Boolean | 0 |
| Regression R² > 0 | Boolean | 0 |
| Python parity (RF, DTL) | Analytical | 1e-12 |
| Cross-validation with Exp053 pipeline | Boolean | 0 |

**Estimated checks:** ~15

## Python Baseline

`scripts/boden2024_phosphorus_phylogenomics.py`

## Rust Validation Binary

`barracuda/src/bin/validate_phosphorus_phylogenomics.rs`
