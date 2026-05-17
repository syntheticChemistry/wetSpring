<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Exp381: breseq Pipeline — Barrick 2009 via Nest Atomic Composition

**Status:** IN PROGRESS (V177)
**Paper:** Barrick et al. "Genome evolution and adaptation in a long-term experiment
with *Escherichia coli*" *Nature* 461, 1243–1247 (2009)
**SRA Study:** SRP001569
**Reference Genome:** REL606 (CP000819.1, 4,629,812 bp)
**lithoSpore Module:** 6 (breseq comparison, via ferment transcript braid)
**LTEE Queue ID:** B1

---

## Objective

Execute the first real-data Nest Atomic composition: download SRA reads through
the sovereign SRA pipeline, run breseq variant calling, record provenance through
the trio (rhizoCrypt → loamSpine → sweetGrass), and export a ferment transcript
braid for lithoSpore.

This is **composition validation**, not just science reproduction. We've proven
Python/Rust math parity (Exp380 = Tier 2 COMPLETE). Now we prove the atomic
pipeline works end-to-end on real data from NCBI.

---

## Dataset

| Clone | SRA Run | Generations | Size | Platform |
|-------|---------|-------------|------|----------|
| REL1164M | SRR032370 | ~2,000 | 190 MB | Illumina GA, 36bp SE |
| REL2179M | SRR032371 | ~5,000 | 202 MB | Illumina GA, 36bp SE |
| REL4536M | SRR032372 | ~10,000 | 198 MB | Illumina GA, 36bp SE |
| REL7177M | SRR032373 | ~15,000 | 209 MB | Illumina GA, 36bp SE |
| REL8593M | SRR032374 | ~20,000 | 162 MB | Illumina GA, 36bp SE |
| REL10379 | SRR032375 | ~30,000 | 171 MB | Illumina GA, 36bp SE |
| REL10926 | SRR032376 | ~40,000 | 173 MB | Illumina GA, 36bp SE |

**Total:** 7 genomes, ~1.3 GB. All single-end 36bp Illumina Genome Analyzer.

---

## Composition Model

```
Nest Atomic (Neutron = data foundation)
├── NestGate: storage.fetch_external → SRA cache
│   (Phase 2: replaces prefetch+fasterq-dump with IPC fetch)
├── wetSpring: breseq pipeline
│   ├── provenance.begin (rhizoCrypt DAG session)
│   ├── provenance.record × N (download + alignment + variant calling)
│   └── provenance.complete (dehydrate → commit → braid)
├── provenance.export_braid → braids/barrick_2009.json
└── lithoSpore: receives braid → data.toml upstream_braid
```

**Current phase:** SRA tools on PATH (micromamba env), breseq 0.40.1 local.
NestGate IPC fetch is Phase 2 — sovereign tools first, atomic composition after.

---

## Validation Checks

| # | Check | Type |
|---|-------|------|
| 1 | Reference genome downloaded (REL606, 4,629,812 bp) | structural |
| 2 | All 7 SRA runs downloaded and FASTQ validated | structural |
| 3 | breseq completes on each clone without error | pipeline |
| 4 | Mutations detected in late clones > early clones | science |
| 5 | Provenance session recorded with trio witnesses | composition |
| 6 | Ferment transcript braid exported as valid JSON | composition |
| 7 | Braid BLAKE3 hash is non-empty | integrity |

---

## Data Provenance

| Source | Accession | License |
|--------|-----------|---------|
| NCBI SRA | SRP001569 | Public domain |
| NCBI GenBank | CP000819.1 (REL606 genome) | Public domain |
| Paper | doi:10.1038/nature08480 | Subscription (analysis is independent) |

---

## Results (V177 — In Progress)

| Clone | Mutations | Status |
|-------|----------|--------|
| REL1164M (~2k gen) | 579 | DONE |
| REL2179M (~5k gen) | — | Running |
| REL4536M (~10k gen) | — | Pending |
| REL7177M (~15k gen) | — | Pending |
| REL8593M (~20k gen) | 1108 | DONE |
| REL10379 (~30k gen) | — | Pending |
| REL10926 (~40k gen) | — | Pending |

**Observation:** REL8593M (1108) > REL1164M (579) confirms mutation accumulation
trend (Barrick 2009 Fig. 1). Remaining clones processing.

**Braid:** Exported to `provenance/braids/barrick_2009_mutations.json` (local provenance,
trio unavailable in standalone mode).

---

## Dependencies

- `sra-toolkit` 3.4.1 (prefetch + fasterq-dump)
- `breseq` 0.40.1 (variant calling)
- `samtools` 1.23.1 (BAM processing)
- `bowtie2` (alignment)
- `R` 4.5.3 (breseq HTML reports)
- Environment: `/mnt/4tb-work/micromamba/envs/breseq-env/`
- Workspace: `/mnt/4tb-work/ecoPrimals/springs/wetSpring/datasets/ltee/barrick_2009/`
