<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# NCBI Anaerobic Digestion Datasets — Track 6 Extension Data

**Date:** March 10, 2026
**Source:** NCBI Entrez E-utilities search (ESearch + ESummary)
**Purpose:** Identify real community datasets for Track 6 real-data validation

---

## Liao Group Direct — No SRA Data Found

The five reproduced Liao group papers (Yang 2016, Chen 2016, Rojas-Sossa
2017/2019, Zhong 2016) do not have linked SRA BioProjects or deposited
sequence data in NCBI. Community composition data from these papers exists
only in journal supplementary materials (published tables).

| Paper | PMID | SRA Link | Supplementary |
|-------|------|----------|---------------|
| Yang et al. 2016 (co-digestion) | — | None | Journal table (Adv Microbiol 6:879-897) |
| Chen et al. 2016 (culture conditions) | — | None | Journal table (Biomass Bioenergy 85:84-93) |
| Rojas-Sossa et al. 2017 (coffee) | 28917107 | None | Bioresour Technol 245:714-723 |
| Rojas-Sossa et al. 2019 (AFEX) | — | None | Biomass Bioenergy 127:105263 |
| Zhong et al. 2016 (fungal) | 27895707 | None | Biotechnol Biofuels 9:253 |

**Action:** Extract community composition tables from supplementary materials
for real-data Track 6 validation. These replace synthetic proxy communities.

---

## Directly Relevant BioProjects (Public 16S + Anaerobic Digestion)

### Tier 0: Highest Relevance (co-digestion, manure, biogas + 16S)

| Accession | SRA Runs | Description | Relevance to Track 6 |
|-----------|----------|-------------|---------------------|
| **PRJNA951939** | **37** | Field-scale digesters: food waste, MWS, manure across 10 sites in Korea | Direct: 16S community + biogas, closest to Liao's co-digestion work |
| **PRJEB37891** | **76** | Food waste AD survey, microbial community profiling | Direct: large AD microbiome survey, diversity analysis |
| **PRJNA1128653** | **29** | Intensifying methane during AD, two key strategies | Direct: methane optimization, community-performance link |
| **PRJEB27282** | **17** | Fungal pretreatment + anaerobic digestion metagenomes | Direct: matches Zhong 2016 fungal fermentation on digestate |

**Total Tier 0:** 159 SRA runs, ~5-15GB estimated raw data

### Tier 1: Related (AD community + biogas, different substrates)

| Accession | SRA Runs | Description | Relevance |
|-----------|----------|-------------|-----------|
| PRJEB30568 | — | Wastewater treatment plant AD microbiome | AD community dynamics |
| PRJEB27206 | — | Biomethanization of CO₂ in AD | CO₂ conversion, community structure |
| PRJNA735449 | — | Biogas fermenter metagenome | Reactor community profiling |
| PRJNA752008 | — | AD microbiome exposed to ammonium | Inhibition studies (matches Haldane model) |
| PRJNA416235 | 2 | Livestock/vegetable waste AD | Co-digestion community |
| PRJNA587352 | 3 | Sewage sludge co-digestion | Active + total microbiome |

### Tier 2: Cross-Regime Comparison (aerobic vs anaerobic)

These are not AD-specific but provide aerobic reference communities for
the Paper 16 W comparison (W_anaerobic vs W_aerobic):

| Accession | Runs | Description | W Comparison |
|-----------|------|-------------|-------------|
| PRJNA315684 | 170 | Cold seep metagenomes | Anaerobic reference (deep-sea) |
| PRJNA488170 | — | Algae pond | Aerobic reference |
| PRJNA283159 | — | Deep-sea vent | Microaerophilic reference |

---

## Search Summary

| Query | Database | Count |
|-------|----------|-------|
| "anaerobic digestion" AND "16S" AND "biogas" | BioProject | 43 |
| "anaerobic digestion" AND "16S rRNA" AND "community" | BioProject | 69 |
| "co-digestion" AND "manure" AND "16S" | BioProject | 6 |
| "Liao W" AND "anaerobic" AND "16S" | PubMed | 3 |
| Liao + anaerobic digestion + biosystems + Michigan | PubMed | 19 |

---

## Recommended Download Order

1. **PRJNA951939** (37 runs) — 10-site field-scale digesters, closest match
   to Liao's co-digestion reactors. Run Track 6 Gompertz + Anderson W on
   real communities. NestGate `data.ncbi_fetch` with SRA accession list.

2. **PRJEB37891** (76 runs) — Large food waste AD survey. Statistical power
   for W distribution analysis across many AD configurations.

3. **PRJNA1128653** (29 runs) — Methane intensification strategies. Direct
   link to Gompertz/Monod kinetics validation with real community data.

4. **PRJEB27282** (17 runs) — Fungal pretreatment AD. Matches Zhong 2016
   (fungal fermentation on digestate). Test aerobic-anaerobic W shift with
   fungal preprocessing step.

5. **Journal supplementary tables** — Yang 2016, Chen 2016, Rojas-Sossa
   2017/2019, Zhong 2016 published community composition. Manual extraction,
   ~1 hour. Creates ground-truth baselines.

---

## Compute Estimate

| Dataset | Download | DADA2/Diversity | Anderson W | Total |
|---------|----------|-----------------|------------|-------|
| PRJNA951939 (37 runs) | ~30 min | ~15 min | ~1 min GPU | <1h |
| PRJEB37891 (76 runs) | ~1h | ~30 min | ~2 min GPU | <2h |
| PRJNA1128653 (29 runs) | ~20 min | ~10 min | ~1 min GPU | <1h |
| PRJEB27282 (17 runs) | ~15 min | ~10 min | <1 min GPU | <30 min |
| Supplementary tables | manual | N/A (pre-processed) | <1 min GPU | manual |
| **Total** | **~2h download** | **~1h CPU** | **~5 min GPU** | **<4h** |

All workloads fit on Eastgate alone. No LAN required for Tier 0.
