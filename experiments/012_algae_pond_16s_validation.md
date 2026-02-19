# Experiment 012: Algae Pond 16S Validation (Paper Proxy)

**Track**: 1 (Life Science)
**Date**: 2026-02-19
**Status**: ACTIVE
**Depends on**: Exp001 (Galaxy QIIME2 pipeline), Exp011 (16S pipeline validation)

---

## Objective

Validate the full Rust 16S amplicon pipeline against a Galaxy/QIIME2 baseline
on real Nannochloropsis outdoor cultivation 16S data — the closest open proxy
for Papers 1/2 (Carney 2016, Humphrey 2023), whose raw reads are not publicly
deposited.

## Data Source

| Field | Value |
|-------|-------|
| BioProject | [PRJNA488170](https://www.ncbi.nlm.nih.gov/bioproject/488170) |
| Run | SRR7760408 |
| Reads | 11.9M spots, 7.2G bases, paired-end Illumina MiSeq |
| Primers | 27F / 338R (V1-V2 region) |
| Organism | Nannochloropsis sp. CCAP211/78 outdoor pilot reactors |
| Lab | Wageningen University and Research |
| Paper | DOI 10.1007/s00253-022-11815-3 |

### Why This Proxy

Papers 1/2 (Carney 2016, Humphrey 2023) studied Nannochloropsis / Microchloropsis
salina pond microbiomes using 16S amplicon sequencing. Their raw reads are not
deposited in NCBI SRA (DOE/Sandia lab data). PRJNA488170 is the closest publicly
available analog: same genus, same outdoor cultivation setting, same sequencing
target (bacterial 16S), open access.

### Data Availability Audit (2026-02-19)

| Paper | Raw Reads in NCBI SRA | OTU Data | Conclusion |
|-------|-----------------------|----------|------------|
| Paper 1 (Carney 2016) | NOT found | In paper figures | DOE restricted |
| Paper 2 (Humphrey 2023) | NOT found | In supplementary | Zymo outsourced, no SRA deposit |

## Protocol

1. Download SRR7760408 via `scripts/download_paper_data.sh --algae-16s`
2. Run Galaxy/QIIME2 pipeline (import, quality, DADA2, taxonomy, diversity)
3. Save Galaxy baseline to `experiments/results/012_algae_pond/`
4. Run `cargo run --bin validate_algae_16s` against baseline

## Baseline Generation

If Galaxy is available:
- Import paired FASTQ into Galaxy
- DADA2 denoise-paired (trunc_len_f=250, trunc_len_r=200)
- Classify with SILVA 138.1 classifier
- Generate diversity metrics (Shannon, Simpson, Bray-Curtis)
- Export ASV table, taxonomy, diversity report

If Galaxy is NOT available (self-contained mode):
- Use Rust pipeline with synthetic reference for analytical validation
- Compare pipeline stage outputs against known analytical properties
- Cross-validate with Humphrey 2023 published results:
  - 18 OTUs identified in bacteriome
  - Core genera: Thalassospira, Marinobacter, Oceanicaulis, Robiginitalea,
    Nitratireductor, Hoeflea, Sulfitobacter
  - B. safensis dynamics in phage rescue experiments

## Acceptance Criteria

1. Rust FASTQ parser handles real MiSeq paired-end data without panic
2. Quality filtering retains >50% of reads
3. DADA2 denoising produces >1 ASV
4. Chimera removal does not eliminate all ASVs
5. Taxonomy classification assigns phylum-level for >50% of ASVs
6. Diversity metrics are within biological plausibility ranges
7. If Galaxy baseline available: Rust results within tolerance of Galaxy

## Validation Binary

`cargo run --bin validate_algae_16s`

## Outputs

- `validate_algae_16s` checks in Validator harness
- ASV count, diversity metrics, taxonomy breakdown
