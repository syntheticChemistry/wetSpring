# Exp184: Real NCBI 16S Through Sovereign Pipeline

**Date:** February 26, 2026
**Phase:** V59 — Science extensions
**Binary:** `validate_real_ncbi_pipeline`
**Command:** `cargo run --release --bin validate_real_ncbi_pipeline`
**Status:** DONE (CPU, 25 checks PASS)

## Purpose

Validate that wetSpring's sovereign NCBI data pipeline can download real
16S rRNA sequences from NCBI, process them through the full diversity →
Anderson spectral analysis pipeline, and produce results consistent with
published community ecology metrics.

This is the first experiment to use **real NCBI data** rather than synthetic
communities or published table values. It proves the pipeline works
end-to-end: NCBI query → sequence download → FASTQ parsing → diversity
metrics → Anderson localization diagnosis.

## Target Datasets

### Tier 1: Small Test Set (validation)

| Accession | Description | Source | Approx Size |
|-----------|-------------|--------|-------------|
| SRR5314241 | Marine sediment 16S V4 | Cold seep (Ruff et al.) | ~10 MB |
| SRR5314242 | Marine sediment 16S V4 | Cold seep (Ruff et al.) | ~10 MB |
| SRR5314243 | Marine sediment 16S V4 | Cold seep (Ruff et al.) | ~10 MB |
| SRR1793429 | Deep-sea vent 16S | PRJNA283159 | ~5 MB |
| SRR1793430 | Deep-sea vent 16S | PRJNA283159 | ~5 MB |

### Tier 2: Full Cold Seep Set (science)

- 170 metagenome accessions from Exp144-145 cold seep literature
- Download to Strandgate bio node or Westgate cold storage
- Process in batches of 20

### Tier 3: LTEE Sequencing (evolution)

- Tenaillon et al. 2016 SRA data
- Tests Anderson anomaly predictions from Exp143

## Pipeline Steps

```
1. ncbi::esearch_count(db, query)     → confirm accessions exist
2. ncbi::efetch_fasta(db, id, key)    → download FASTA/FASTQ
3. ncbi::write_with_integrity(dir, f) → cache with SHA-256
4. io::fastq::for_each_record(path)   → parse sequences
5. bio::diversity::shannon(counts)     → Shannon H'
6. bio::diversity::simpson(counts)     → Simpson D
7. bio::diversity::observed_features() → S_obs
8. spectral::anderson_3d(L, W, seed)  → Anderson lattice (W = f(S_obs))
9. spectral::lanczos → eigenvalues    → level spacing ratio r
10. Classify: r > midpoint → extended (QS viable) / localized (QS suppressed)
```

## Validation Checks

### S1: Data Acquisition
- [ ] ESearch returns >0 hits for each accession
- [ ] EFetch downloads valid FASTA (starts with `>`)
- [ ] Cached files pass SHA-256 integrity verification
- [ ] Downloaded file sizes are plausible (>1 KB, <1 GB)

### S2: Diversity Pipeline
- [ ] Shannon H' > 0 for all samples
- [ ] Simpson D in [0, 1] for all samples
- [ ] Observed features > 10 for 16S metagenomes
- [ ] Bray-Curtis distance matrix is symmetric

### S3: Anderson Spectral Analysis
- [ ] Level spacing ratio r in [POISSON_R, GOE_R] for all samples
- [ ] High-diversity samples (H' > 3) tend toward GOE (extended)
- [ ] Low-diversity samples (H' < 1) tend toward Poisson (localized)
- [ ] W_c estimate consistent with Exp150 (16.5 ± 2)

### S4: Cross-Reference with Published Values
- [ ] Shannon H' within 10% of published values where available
- [ ] Community composition broadly consistent with source papers
- [ ] Anderson classification matches biome predictions from Exp140

## Dependencies

- `ncbi/efetch.rs` — EFetch FASTA download (Phase 1a)
- `ncbi/sra.rs` — SRA run download (Phase 1b)
- `ncbi/cache.rs` — integrity-verified caching (Phase 1c)
- `io/fastq/` — existing FASTQ parser
- `bio/diversity/` — existing diversity metrics
- `barracuda::spectral` — Anderson eigensolver (feature = "gpu")

## Compute Estimate

- Tier 1 (5 accessions): ~30 seconds download, ~5 seconds compute
- Tier 2 (170 accessions): ~2 hours download, ~30 minutes compute
- Tier 3 (LTEE): ~4 hours download, ~1 hour compute

## Data Provenance

All data sourced from NCBI Sequence Read Archive (SRA), a public repository.
No authentication required for data access. NCBI API key used for rate
limiting (10 req/s vs 3 req/s).

| Dataset | BioProject | Publication | License |
|---------|-----------|-------------|---------|
| Cold seep 16S | PRJNA315684 | Ruff et al. 2019 | Public domain |
| Deep-sea vent | PRJNA283159 | Reveillaud et al. 2016 | Public domain |
| LTEE sequencing | PRJNA294072 | Tenaillon et al. 2016 | Public domain |
