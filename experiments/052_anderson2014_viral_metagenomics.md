# Experiment 052 — Anderson 2014: Viral Metagenomics at Hydrothermal Vents

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Anderson et al. (2014) "Evolutionary strategies of viruses and cells in hydrothermal systems revealed through metagenomics" PLoS ONE 9:e109696
**DOI:** 10.1371/journal.pone.0109696
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate wetSpring's diversity, k-mer, and alignment primitives against viral
vs. cellular metagenome comparison. This paper contrasts viral and cellular
fractions from the same hydrothermal vent, exercising community comparison
metrics. Connects to Waters (phage defense, Exp030) and Cahill (algae pond
phage, Exp039).

## Data

- **MG-RAST:** 4469452.3 (viral fraction), 4481541.3 (cellular fraction)
- **Location:** Hulk vent, Main Endeavour Field, Juan de Fuca Ridge
- **Technology:** 454 pyrosequencing
- **Proxy approach:** Use published functional profiles (KEGG/COG/SEED
  distributions from supplementary) as community composition vectors

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Community diversity (Shannon, Simpson) | `bio::diversity` | Existing |
| Beta diversity (viral vs. cellular) | `bio::diversity::bray_curtis` | Existing |
| K-mer frequency profiles | `bio::kmer` | Existing |
| Functional profile comparison | `bio::spectral_match::cosine_similarity` | Existing |
| dN/dS ratio estimation | **NEW** `bio::dnds` | **Needed** |
| Fragment recruitment (alignment) | `bio::alignment` | Existing |

## Validation Design

### Phase 1: Viral vs. cellular community comparison
- Synthetic functional profiles matching paper's KEGG distributions
- Shannon/Simpson for each fraction
- Bray-Curtis distance between viral and cellular
- K-mer frequency divergence

### Phase 2: dN/dS module validation
- New `bio::dnds` module: codon-aware pairwise dN/dS (Nei-Gojobori 1986)
- Validate against analytical known-values (identical sequences → dN/dS = 0)
- Validate against Python baseline (Biopython `cal_dn_ds`)

### Phase 3: Fragment recruitment scoring
- Smith-Waterman alignment scores for viral reads against reference genomes
- Validates alignment module on metagenomic-length sequences

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| Shannon (viral fraction) | Analytical | 1e-12 |
| Shannon (cellular fraction) | Analytical | 1e-12 |
| BC(viral, cellular) > 0.5 | Boolean | 0 |
| K-mer divergence detectable | Boolean | 0 |
| Cosine similarity (self) ≈ 1.0 | Analytical | 1e-12 |
| dN/dS identical = 0.0 | Analytical | 1e-12 |
| dN/dS synonymous-only = 0.0 | Analytical | 1e-12 |
| dN/dS nonsynonymous > 0 | Boolean | 0 |
| dN/dS Python parity | Analytical | 1e-10 |
| SW alignment score positive | Boolean | 0 |
| Python parity (diversity) | Analytical | 1e-12 |

**Estimated checks:** ~20

## Python Baseline

`scripts/anderson2014_viral_metagenomics.py`

## Rust Validation Binary

`barracuda/src/bin/validate_viral_metagenomics.rs`
