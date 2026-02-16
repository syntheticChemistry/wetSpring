# Experiment 001: Galaxy Bootstrap

**Date:** February 12, 2026
**Goal:** Self-host Galaxy Project on Eastgate, install bioinformatics
tools, validate with public 16S rRNA dataset.

---

## Setup

1. Pull and start Galaxy Docker:
   ```
   cd wetSpring/control/galaxy
   docker compose up -d
   ```

2. Access Galaxy UI at `http://localhost:8080`

3. Create admin account (admin@galaxy.org)

4. Install amplicon analysis tools from tool list:
   ```
   shed-tools install -g http://localhost:8080 \
     -a <api_key> -t tool_lists/amplicon_tools.yml
   ```

## Validation Dataset

Use the **Galaxy Training Network** 16S amplicon tutorial dataset:
- Source: https://training.galaxyproject.org/training-material/topics/metagenomics/tutorials/mothur-miso/tutorial.html
- Alternatively: Human Microbiome Project mock community (SRR**)

## Success Criteria

- [ ] Galaxy web UI accessible on localhost:8080
- [ ] FastQC, Trimmomatic, DADA2, QIIME2 tools installed
- [ ] Kraken2, BLAST+ installed
- [ ] SPAdes, Prokka installed
- [ ] Successfully run FastQC on a test FASTQ file
- [ ] Successfully run DADA2 denoise on paired-end 16S data
- [ ] Produce taxonomy barplot from QIIME2

## Notes

- Galaxy Docker image is ~4 GB compressed, ~8 GB on disk
- SILVA/Greengenes reference databases add ~2-5 GB
- Eastgate has 12 GB VRAM + 64 GB RAM — plenty for NGS workloads
- Galaxy uses PostgreSQL internally (not SQLite) in Docker mode

## Next Steps

Once validated, proceed to Experiment 002: replicate Pond Crash Forensics
pipeline with comparable public 16S amplicon data.
