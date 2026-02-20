# Experiment 019: Phylogenetic Network / HMM Validation

**Date**: 2026-02-19
**Status**: DESIGN COMPLETE — test data identified, tools available
**Track**: 1b (Comparative Genomics, Liu)

---

## Objective

Validate phylogenetic inference algorithms against published tools and
datasets from the Liu lab (CMSE, MSU) and the NakhlehLab (Rice University).
This establishes the Python/open-data baseline for comparative genomics
before porting to Rust.

## Data Sources

### PhyNetPy (NakhlehLab, GitHub)

- **Repository**: https://github.com/NakhlehLab/PhyNetPy
- **License**: Open source
- **Content**: 27 NEXUS files + 1,284 Newick gene trees
- **Directories**:
  - `DEFJ/` — gene tree datasets (100 genes × multiple replicates)
  - `DLLS/` — additional phylogenetic test cases
  - `Bayesian/` — Bayesian inference test NEXUS files
  - `NexusFiles/` — standalone NEXUS alignment files
- **Use**: Newick tree parsing, network inference validation

### PhyloNet-HMM (Liu et al. 2014, PLoS Comp Bio)

- **Paper**: DOI 10.1371/journal.pcbi.1003649
- **Software**: Java JAR from phylogenomics.rice.edu
- **Empirical data**: Mouse chromosome 7 introgression (Mus musculus × M. spretus)
  - ~9% of chr7 sites detected as introgressive (Vkorc1 region)
- **Simulated data**: Coalescent models with recombination, isolation, migration
- **Note**: Original data downloads from Rice server redirect; may need direct
  contact or extraction from paper supplementary materials

### SATe (Liu et al. 2009, Science)

- **Paper**: DOI 10.1126/science.1171243
- **Data**: University of Illinois repository (DOI 10.13012/B2IDB-5139418_V1)
- **SATe-II data**: Dryad (DOI 10.5061/dryad.n9r3h)
  - 16S rRNA datasets (16S.B.ALL and 16S.T)
  - Directly relevant to wetSpring Track 1 (16S pipeline)
- **Software**: Available at phylo.bio.ku.edu

## Design

### Phase 1: Newick Tree Parsing Validation

1. Download PhyNetPy gene tree files (Newick format)
2. Parse with our Rust `bio::unifrac::PhyloTree` (already has Newick parser)
3. Validate: tree topology, leaf count, branch lengths match Python parsing
4. Python baseline: `ete3` or `dendropy` for reference tree statistics

### Phase 2: Gene Tree Analysis

1. Load gene trees from DEFJ/ directory (100 genes × 10 replicates)
2. Compute Robinson-Foulds distances between gene trees
3. Validate against PhyNetPy-computed distances
4. This tests the core tree comparison primitive

### Phase 3: PhyloNet-HMM Introgression Detection

1. If empirical data available: reproduce mouse chr7 introgression detection
2. If simulated data available: validate HMM transition probabilities
3. Python baseline: PhyloNet-HMM Java JAR output as reference

### Phase 4: SATe 16S Alignment (bridges to Track 1)

1. Download SATe-II 16S rRNA dataset from Dryad
2. Run SATe alignment on 16S data
3. Compare alignment quality metrics with our DADA2 ASV sequences
4. This connects Liu's phylogenetics work directly to our 16S pipeline

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Newick parse matches Python (leaf count, branch length) | 100% | PhyNetPy test data |
| Robinson-Foulds distance matches dendropy | ±0 | Gene tree pairs |
| Introgression sites detected | ~9% of chr7 | Liu 2014 paper Table 1 |
| SATe alignment score ≥ baseline | SP score parity | SATe-II Dryad data |

## Dependencies

- PhyNetPy gene tree data (GitHub, open source)
- dendropy or ete3 (Python tree comparison baseline)
- PhyloNet-HMM JAR (Java, GPL)
- SATe-II data (Dryad, CC0)
- Rust: `bio::unifrac::PhyloTree` (Newick parser already implemented)

## Relationship to Paper Queue

| Paper | Year | What We Validate |
|-------|------|-----------------|
| Liu et al. 2014 (PhyloNet-HMM) | 2014 | Introgression detection, HMM primitives |
| Liu et al. 2009 (SATe) | 2009 | Large-scale alignment + tree co-estimation |
| Alamin & Liu 2024 (TCBB) | 2024 | Metagenomic phylogenetic placement |
| Wang et al. 2021 (RAWR) | 2021 | Modern resampling for phylogenetic confidence |

## Notes

- PhyNetPy is the modern Python rewrite of PhyloNet (Java)
- Our `bio::unifrac` module already parses Newick trees — this extends it
  to full phylogenetic network analysis
- The SATe 16S dataset is a natural bridge between Liu's phylogenetics
  and our existing 16S pipeline (Track 1)
- Gene tree / species tree discordance is directly relevant to the
  "unknown unknowns" problem in pond crash forensics (Carney 2016)
