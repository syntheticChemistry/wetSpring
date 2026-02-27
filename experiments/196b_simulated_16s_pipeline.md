# Experiment 196b: Simulated Long-Read 16S Pipeline

**Date:** February 27, 2026
**Phase:** 61
**Track:** Field Genomics (Sub-thesis 06)
**Status:** PASS (11/11 checks)
**Binary:** `validate_nanopore_simulated_16s`

---

## Objective

Validate that synthetic nanopore-length reads (1000-5000 bp, ~10% error rate)
flow through the sovereign 16S pipeline (DADA2 → chimera → taxonomy → diversity)
and produce biologically meaningful community reconstructions. This tests the
pipeline's robustness to long-read error profiles before real MinION data arrives.

## Background

Nanopore reads are longer but noisier than Illumina. Full-length 16S (~1500 bp)
provides better taxonomic resolution but introduces insertion/deletion errors
that challenge ASV denoising. The pipeline must:

1. Handle reads 10-50× longer than Illumina 16S V4 amplicons
2. Tolerate ~10% error rate (vs ~0.5% for Illumina)
3. Reconstruct community profiles that match known input compositions
4. Detect Anderson disorder regime from reconstructed community

## Validation Sections

| Section | Checks | What It Validates |
|---------|:------:|-------------------|
| S1: ASV recovery | 3 | Expected ASVs recovered from noisy reads, abundance correlation |
| S2: Community reconstruction | 3 | Shannon diversity, Bray-Curtis to ground truth, Pielou evenness |
| S3: Anderson regime | 3 | W(disorder) computation, regime classification from long-read profile |
| S4: Pipeline throughput | 2 | Reads/sec, memory usage with streaming I/O |

**Total:** 11/11 PASS

## Tolerance Constants

- `LONG_READ_OVERLAP` — minimum overlap for merge in long-read context
- `NANOPORE_DIVERSITY_VS_ILLUMINA` — allowed diversity metric divergence

## Key Findings

- Full-length 16S reads provide genus-level classification even at ~10% error
- DADA2 denoising adapts to longer reads via quality-based error model
- Anderson regime detection is robust to nanopore noise — disorder (W) within
  2 units of Illumina-equivalent profile
- Streaming I/O handles long reads without memory spikes
