# Exp185: Cold Seep Metagenomes Through Sovereign Pipeline

**Date:** February 26, 2026
**Phase:** V55 — Science extensions
**Binary:** `validate_cold_seep_pipeline` (planned)
**Command:** `cargo run --release --features gpu --bin validate_cold_seep_pipeline`
**Status:** Protocol defined, implementation pending
**Depends on:** Exp184 (pipeline validation with small test set)

## Purpose

Process 170 cold seep metagenomes from Ruff et al. through the sovereign
NCBI → diversity → Anderson pipeline. Tests the headline prediction:
"3D marine sediment habitats support extended QS (delocalized states)"
on real metagenomic data at scale.

## Hypothesis

Cold seep sediments exhibit high microbial diversity (Shannon H' > 3)
and 3D pore geometry. Anderson localization theory predicts:
- Level spacing ratio r ≈ GOE (0.531) for high-diversity 3D habitats
- QS signaling remains viable (delocalized regime)
- Disorder parameter W ~ 1/diversity maps to the extended side of W_c

## Target Datasets

| BioProject | Accessions | Description | Download |
|-----------|------------|-------------|----------|
| PRJNA315684 | 170 runs | Cold seep 16S V4 amplicons | `fasterq-dump` |
| Supplementary | Published OTU tables | Alpha/beta diversity metrics | Manual |

## Pipeline

1. **Download** (NestGate or sovereign): 170 SRA runs → FASTQ cache
2. **Quality filter**: Trim adapters, quality Q20 (existing FASTQ parser)
3. **OTU/ASV clustering**: Map reads to reference taxonomy
4. **Diversity metrics**: Shannon H', Simpson D, observed features per sample
5. **Anderson mapping**: Map diversity to disorder W, construct 3D lattice
6. **Spectral analysis**: Lanczos eigenvalues → level spacing ratio r
7. **Classification**: r vs midpoint → extended/localized per sample

## Validation Checks

### S1: Data Integrity
- [ ] All 170 FASTQ files downloaded and SHA-256 verified
- [ ] No truncated downloads (file size > 1 KB)
- [ ] Cache directory structured by accession

### S2: Diversity Metrics
- [ ] Shannon H' > 0 for all 170 samples
- [ ] Mean Shannon H' in [2, 5] (expected for marine sediment)
- [ ] Simpson D in [0.7, 1.0] (high evenness expected)
- [ ] Observed features > 50 per sample (16S amplicon richness)

### S3: Anderson Classification
- [ ] >80% of samples classified as extended (QS viable)
- [ ] Mean r > midpoint (0.458) across all samples
- [ ] r correlates positively with Shannon H' (Spearman ρ > 0.3)
- [ ] W estimates cluster below W_c ≈ 16.5

### S4: Cross-Reference
- [ ] Diversity metrics within 20% of Ruff et al. published values
- [ ] Spatial patterns match published community gradients
- [ ] Results consistent with Exp140 biome predictions

## Compute Estimate

- Download: ~5-50 GB FASTQ, ~2 hours on broadband
- Processing: ~30 minutes per sample on Strandgate
- Total: ~8 hours including download
- GPU Lanczos: ~2 hours on biomeGate RTX 4070

## Provenance

| Item | Value |
|------|-------|
| Source paper | Ruff et al., Nature Microbiology (2019) |
| BioProject | PRJNA315684 |
| Data license | Public domain (NCBI SRA) |
| Analysis code | `barracuda/src/bin/validate_cold_seep_pipeline.rs` |
