# Experiment 055 — Anderson 2017: Population Genomics at Hydrothermal Vents

**Track:** 1c (Deep-Sea Metagenomics)
**Paper:** Anderson et al. (2017) "Genomic variation in microbial populations inhabiting the marine subseafloor at deep-sea hydrothermal vents" Nature Communications 8:1114
**DOI:** 10.1038/s41467-017-01228-6
**Faculty:** R. Anderson (Carleton College)

## Purpose

Validate wetSpring's alignment and diversity primitives plus new population
genetics modules against metagenomic population-level analysis. This paper
performs SNP calling, dN/dS estimation, and ANI (Average Nucleotide Identity)
on metagenome-assembled genomes from the Mid-Cayman Rise — the most
computationally demanding of the Anderson Track 1c papers.

## Data

- **BioProject:** PRJNA283159 (Mid Cayman Rise Metagenome)
- **SRA:** 8 metagenomic samples from Piccard and Von Damm vent fields
- **Volume:** ~275 Gbases (subsampling required for validation)
- **MAGs:** 73 metagenome-assembled genomes

## Computational Methods (from paper)

| Method | wetSpring Module | Status |
|--------|-----------------|--------|
| Read quality filtering | `bio::quality` | Existing |
| K-mer analysis | `bio::kmer` | Existing |
| Sequence alignment | `bio::alignment` | Existing |
| dN/dS estimation | `bio::dnds` | From Exp052 |
| ANI (Average Nucleotide Identity) | **NEW** `bio::ani` | **Needed** |
| SNP calling / variant frequency | **NEW** `bio::snp` | **Needed** |
| Diversity metrics | `bio::diversity` | Existing |
| Phylogenetic placement | `bio::placement` | Existing |

## Validation Design

### Phase 1: ANI module
- New `bio::ani` module: pairwise Average Nucleotide Identity
- ANI = (sum of alignment identities * alignment lengths) / (total aligned length)
- Validate: ANI(self) = 1.0, ANI symmetric, ANI(different species) < 0.95

### Phase 2: SNP calling module
- New `bio::snp` module: variant calling from aligned reads
- Count variants per position, compute allele frequencies
- Validate against analytical (known synthetic variants)

### Phase 3: Population-level analysis on proxy data
- Subsampled PRJNA283159 reads (1000 reads per sample)
- Quality → alignment → SNP → diversity pipeline
- Compare population metrics to paper's supplementary tables

## Expected Checks

| Check | Type | Tolerance |
|-------|------|-----------|
| ANI(self) = 1.0 | Analytical | 1e-12 |
| ANI symmetric | Analytical | 1e-12 |
| ANI(same species) > 0.95 | Boolean | 0 |
| ANI(different species) < 0.95 | Boolean | 0 |
| SNP count on synthetic data | Exact | 0 |
| Allele frequency sum = 1.0 | Analytical | 1e-12 |
| dN/dS pipeline integration | Analytical | 1e-10 |
| Diversity (Shannon) per sample | Analytical | 1e-12 |
| Python parity (ANI, SNP, dN/dS) | Analytical | 1e-10 |
| Quality filter pass rate > 0 | Boolean | 0 |

**Estimated checks:** ~20

## Python Baseline

`scripts/anderson2017_population_genomics.py`

## Rust Validation Binary

`barracuda/src/bin/validate_population_genomics.rs`
