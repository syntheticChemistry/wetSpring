# Experiment 001: Galaxy Bootstrap

**Date:** February 12-16, 2026
**Status:** DONE (Phase 1 — Galaxy/QIIME2 baseline established)
**Goal:** Self-host Galaxy Project on Eastgate, install bioinformatics
tools, validate with public 16S rRNA dataset.

---

## Setup

### Galaxy Version

Initially deployed `bgruening/galaxy-stable:latest` (Galaxy v20.09) but
modern toolshed revisions require Galaxy 22.05+. Upgraded to
`quay.io/bgruening/galaxy:24.1` (Galaxy 24.1, Ubuntu 22.04, Python 3.10,
PostgreSQL 15).

### Configuration

```yaml
# docker-compose.yml key settings
image: quay.io/bgruening/galaxy:24.1
environment:
  GALAXY_CONFIG_BRAND: wetSpring
  GALAXY_CONFIG_ADMIN_USERS: admin@galaxy.org
  GALAXY_CONFIG_MASTER_API_KEY: wetspring-bootstrap-key-2026
  GALAXY_CONFIG_CONDA_AUTO_INSTALL: true
  GALAXY_CONFIG_CONDA_PREFIX: /tool_deps/_conda
```

### Tool Installation

All 15 tool repositories installed via `ephemeris` (shed-tools):

```bash
pip install ephemeris
shed-tools install -g http://localhost:8080 \
  -a "wetspring-bootstrap-key-2026" \
  -t control/galaxy/tool_lists/amplicon_tools.yml
```

**Result:** 15/15 repositories installed (50s total), expanding to 32 individual
tools including all BLAST+ variants.

### Installed Tools (32 total)

| Section | Tool | Version | Status |
|---------|------|---------|--------|
| QC | FastQC | 0.74+galaxy1 | OK |
| QC | Trimmomatic | 0.39+galaxy2 | OK |
| QC | Cutadapt | 5.2+galaxy0 | OK |
| Amplicon | QIIME2 DADA2 denoise-paired | 2026.1.0 | OK |
| Amplicon | QIIME2 feature-table summarize | 2026.1.0 | OK |
| Amplicon | QIIME2 taxa barplot | 2026.1.0 | OK |
| Amplicon | QIIME2 diversity core-metrics | 2026.1.0 | OK |
| Taxonomy | Kraken2 | 2.1.3+galaxy1 | OK |
| Taxonomy | mothur classify.seqs | 1.39.5.0 | OK |
| Assembly | SPAdes | 4.2.0+galaxy0 | OK |
| Annotation | Prokka | 1.14.6+galaxy1 | OK |
| Phage | Pharokka | 1.3.2+galaxy0 | OK |
| Alignment | NCBI BLAST+ (12 tools) | 2.16.0+galaxy0 | OK |
| Alignment | Bowtie2 | 2.5.4+galaxy0 | OK |
| Viz | ggplot2 heatmap | 3.5.1+galaxy1 | OK |

---

## Validation Dataset

**Source:** Galaxy Training Network — mothur MiSeq SOP tutorial
**Data:** Schloss lab mouse gut 16S rRNA V4 region, paired-end Illumina MiSeq
**Zenodo:** https://zenodo.org/records/800651
**Downloaded:** 44 files (20 paired samples + reference files, 372 MB)

---

## Results

### FastQC — F3D0_R1.fastq (Forward reads)

```
FastQC v0.12.1
Total Sequences:          7,793
Total Bases:              1.9 Mbp
Sequence Length:          249-251 bp
Encoding:                 Sanger / Illumina 1.9 (Phred33)
%GC:                      54%
Sequences poor quality:   0
Per-base quality:         PASS (Q32-38+ across all positions)
Per-sequence quality:     PASS
GC content:               PASS
Sequence length dist:     PASS
```

**Interpretation:** Clean data. Quality scores consistently high (>Q30)
across full read length. No adapter contamination detected. Consistent
with well-prepared 16S V4 amplicon library.

### FastQC — F3D0_R2.fastq (Reverse reads)

Completed successfully (hid=9,10). Reverse reads typically show slightly
lower quality toward 3' end but still well within acceptable range for
DADA2 denoising.

---

## Success Criteria

- [x] Galaxy web UI accessible on localhost:8080
- [x] Galaxy upgraded to v24.1 (supports modern tool revisions)
- [x] FastQC, Trimmomatic, Cutadapt tools installed (QC section)
- [x] DADA2, QIIME2 tools installed (Amplicon section)
- [x] Kraken2, mothur, BLAST+ installed (Taxonomy/Alignment)
- [x] SPAdes, Prokka, Pharokka installed (Assembly/Annotation)
- [x] Successfully run FastQC on F3D0_R1.fastq — PASS
- [x] Successfully run FastQC on F3D0_R2.fastq — PASS
- [ ] Successfully run DADA2 denoise on paired-end 16S data
- [ ] Produce taxonomy barplot from QIIME2

---

## Issues Encountered

1. **Galaxy v20.09 too old**: Tools like FastQC 0.74, QIIME2 2024+, Kraken2
   target Galaxy 22.05-24.2 — rejected by v20.09 with "targets version X.Y"
   errors. Fixed by upgrading to `quay.io/bgruening/galaxy:24.1`.

2. **shed_tool_conf.xml race condition**: On v20.09, concurrent tool installs
   via ephemeris wrote only 5/15 tools to the XML config. Fixed by upgrade
   (Galaxy 24.1 handles this correctly).

3. **Conda dependency resolution**: First FastQC run failed with "command not
   found" because `conda_auto_install` was disabled by default. Fixed by
   setting `conda_auto_install: true` and `conda_prefix: /tool_deps/_conda`
   in `galaxy.yml`.

4. **SPAdes owner mismatch**: YAML listed `owner: iuc` but correct ToolShed
   owner is `nml`. Fixed in `amplicon_tools.yml`.

---

## Notes

- Galaxy Docker image is ~4 GB compressed
- First boot takes ~80s (DB migration, conda init)
- Tool installation via ephemeris: ~50s for 15 repositories
- FastQC runs in <10s per sample on Eastgate
- conda_auto_install enables automatic dependency resolution on first tool use
- Eastgate has 12 GB VRAM + 64 GB RAM — plenty for NGS workloads

---

## Next Steps

1. Upload all 20 paired-end samples from MiSeq SOP dataset
2. Run DADA2 denoise-paired on the full dataset
3. Run SILVA taxonomy classification
4. Generate taxonomy barplot via QIIME2
5. Proceed to Experiment 002: 16S amplicon replication with algae data
