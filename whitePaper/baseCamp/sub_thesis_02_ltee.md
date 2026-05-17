<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Sub-thesis 02: LTEE Constrained Evolution and Anderson Anomalies

**Date:** February 27, 2026
**Faculty:** Kachkovskiy (MSU CMSE), Waters (MSU MMG)
**Status:** Framework established — anomaly catalog complete (9 anomalies, 3 NP solutions). W_c = 16.26 ± 0.95 quantified. Track 4 no-till data (Exp170-178) provides agricultural time-series test bed

---

## Core Claim

Where QS exists despite Anderson's prediction of signal localization,
evolution has discovered solutions to physics problems. These "Anderson
anomalies" are analogous to Lenski's LTEE discoveries: constrained
evolution finding innovations through historical contingency.

## Key Results

| Finding | Experiment | Checks |
|---------|:----------:|:------:|
| 9 Anderson anomalies catalogued (3 genuine NP solutions) | Exp143 | 5 |
| luxR phylogeny × geometry overlay (12 lineages) | Exp146 | 5 |
| V. cholerae inverted logic as NP solution | Exp143 | — |
| Myxococcus self-organized geometry (contact bypass) | Exp143 | — |
| Dictyostelium cAMP relay (signal amplification) | Exp143 | — |

## The Lenski Connection

- LTEE: constrained evolution discovers innovations through historical contingency
- Anderson anomalies: same pattern — QS where physics says it shouldn't work
- Citrate mutation (Ara-3) = one-in-a-trillion event; Anderson bypass = rare evolutionary innovation
- Both require potentiating mutations + specific genetic context

## Anomaly Classification

| Type | Example | Mechanism |
|------|---------|-----------|
| **NP solution** | V. cholerae inverted QS | Novel signaling logic |
| **NP solution** | Myxococcus C-signal | Self-organized 3D from 2D |
| **NP solution** | Dictyostelium relay | Signal amplification cascade |
| **Loophole** | V. fischeri light organ | Tissue-specific QS activation |
| **Loophole** | Roseobacter algal surface | Low-diversity 2D exploitation |

## Open Questions

1. Can we quantify the "evolutionary cost" of each NP solution?
2. Are Anderson anomalies more common in lineages with longer evolutionary histories?
3. Does the LTEE predict which anomaly types should be most frequent?

## Exp381: First Real-Data Composition (May 2026)

| Clone | Generations | Mutations | Status |
|-------|-----------|----------|--------|
| REL1164M | ~2,000 | 579 | Done |
| REL2179M | ~5,000 | 608 | Done |
| REL4536M | ~10,000 | 604 | Done |
| REL8593M | ~20,000 | 1108 | Done |
| REL10926 | ~40,000 | 2296 | Done |

Mutation accumulation trend confirmed (Barrick 2009 Fig. 1). First real-data
Nest Atomic composition through the ecosystem. breseq (C++) serves as Tier 1
validation baseline — the destination is sovereign Rust alignment + variant
calling through barraCuda. See `gen4_compute_aware_pipeline.md`.

**Profiling insight:** Hardcoded `-j 4` on 16-thread machine = 25% utilization.
DAG checkpoint pattern proven: kill → optimize → restart costs 0s for completed
clones. Living-environment scheduling is a composition primitive.

## Connection to Gen3 Thesis

Chapter 14: Constrained evolution framework links Lenski LTEE to Anderson QS.
P ≠ NP enzyme thesis: physics constraints → evolutionary NP solutions.
