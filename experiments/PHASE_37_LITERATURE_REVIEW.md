# Phase 37 Literature Review: Novelty Assessment & Extension Papers

**Date:** February 23, 2026
**Purpose:** Assess what is novel in our Anderson-QS framework vs existing
literature, and identify papers that can extend the science.

---

## What Is NOVEL in Our Work

### 1. Anderson Localization Applied to QS Signal Propagation (CORE NOVELTY)

**Status: No prior work found.**

Web search explicitly confirms: "Anderson localization applied to biological
signal propagation or quorum sensing remains an unexplored research frontier
in the 2024-2025 literature."

Nobody has mapped the Anderson metal-insulator transition to QS signaling in
microbial communities. The closest work:
- PRE 2020 (Meyer et al.): QS traveling wave propagation in V. fischeri.
  Uses reaction-diffusion PDE, NOT Anderson/spectral theory.
- SciRep 2019 (Jemielita et al.): Spatial heterogeneity affects QS burst
  statistics. Notes disorder effects but never invokes Anderson localization.

**Our contribution**: Mapping diversity (Pielou J) → disorder (W), computing
level spacing ratio ⟨r⟩ as GOE/Poisson diagnostic, and showing the d=3
metal-insulator transition predicts where QS can/can't work.

### 2. Evenness-to-Disorder Mapping (J → W) (NOVEL)

No prior work maps ecological diversity indices to Anderson Hamiltonian
disorder parameter. This is a new bridge between ecology and condensed
matter physics.

### 3. Dimensional Phase Diagram for QS (NOVEL)

28 biomes × 1D/2D/3D tested systematically. The finding that ALL natural
biomes are QS-active in 3D and NONE in 2D (at standard diversity levels)
has not been reported.

### 4. Anderson as Null Hypothesis + NP Solution Framework (NOVEL)

V. cholerae inverted logic, Myxococcus self-organized geometry, and
Dictyostelium relay are individually well-known. What's NEW is:
- Unifying them as "Anderson anomalies" — QS where localization predicts failure
- Classifying them as NP solutions vs loopholes
- Connecting to the constrained evolution thesis

### 5. Dilution Model W_eff = W_base/occupancy (NOVEL)

No prior work models planktonic dilution as effective Anderson disorder
amplification. The prediction that QS breaks at <75% occupancy is new.

### 6. Cross-Domain QS Prediction (NOVEL)

Bacteria → yeast → protist → tissue cells following the same Anderson
scaling rules (L_eff from cell diameter) has not been proposed.

### 7. Producer/Receiver NCBI Habitat Analysis (NOVEL APPROACH)

Systematic NCBI search separating synthases from receptors by isolation
source. The concept of luxR-only "solos" is known (Subramoni & Venturi
2009), but our habitat-resolved analysis is new.

---

## What Is ESTABLISHED (Not Novel)

### Well-Known Individual Phenomena

| Finding | Status | Key References |
|---------|--------|----------------|
| V. cholerae inverted QS | Well-characterized | Hammer & Bassler 2003; Waters et al. 2008 |
| Dictyostelium cAMP relay waves | Textbook | Devreotes 1989; many subsequent |
| Myxococcus C-signal + fruiting body | Known | Julien et al. 2000; Rajagopalan 2021 PNAS |
| SAR11/Prochlorococcus streamlining | Known | Giovannoni et al. 2005; many |
| luxR orphan/solo receptors | Known | Subramoni & Venturi 2009 |
| QS regulates biofilm formation | Textbook | O'Toole et al. 2000; many |
| QS in soil metagenomes | Emerging | Cold seep 2025; functional metagenomics 2021 |

### Existing QS Spatial Models (But NOT Anderson)

| Paper | Approach | Differs From Ours |
|-------|----------|-------------------|
| Meyer et al. PRE 2020 | Reaction-diffusion PDE for QS waves | Continuum PDE, not spectral theory |
| Jemielita et al. SciRep 2019 | Agent-based biofilm + QS timing | Stochastic agents, not Anderson |
| Trovato et al. SciRep 2017 | Tube height vs cell density | Experimental, not theoretical framework |
| Hense et al. 2007 | "Efficiency sensing" vs "quorum" | Conceptual, not mathematical |

---

## Papers to Read and Extend Our Science

### Tier 1: Directly Extends Our Framework

| # | Paper | Year | Why Read |
|---|-------|------|----------|
| P1 | **"Physical communication pathways in bacteria: an extra layer to quorum sensing"** — Biophys Rev Lett | 2025 | Catalogs ALL microbial communication modes (mechanical, EM, acoustic, light) beyond QS. Directly extends our "types of microbial communication" analysis. Can we apply Anderson to these other signals? |
| P2 | **"Diverse quorum sensing systems regulate microbial communication in deep-sea cold seeps"** — Microbiome | 2025 | **299,355 QS genes** across **170 metagenomes**, **34 QS types**, **6 systems**. Massive dataset to test Anderson predictions. Habitat = deep-sea sediment (3D). We predict: high QS prevalence. Verify with their data. |
| P3 | **"In silico protein analysis, ecophysiology, and reconstruction of evolutionary history"** — BMC Genomics | 2024 | Phylogenetic reconstruction of luxR family evolution. Can we correlate QS gene gain/loss with habitat geometry transitions in their tree? |
| P4 | **"Spatially propagating activation of QS in V. fischeri"** — PRE 101:062421 | 2020 | Closest to our physics approach. They model QS as traveling waves; we model it as Anderson localization. Different regime: they study PROPAGATION in active systems; we study WHETHER propagation is possible given disorder. Complementary. |

### Tier 2: Validates or Challenges Predictions

| # | Paper | Year | Why Read |
|---|-------|------|----------|
| P5 | **"Burst statistics in biofilm QS: role of spatial colony-growth heterogeneity"** — SciRep | 2019 | Spatial disorder in colony growth affects QS timing. Their "disordered colony" = our Anderson disorder. They found clustered cells → earlier but more localized QS. This is exactly the localization-to-extended transition we predict. |
| P6 | **"Functional metagenomic analysis of QS in a nitrifying community"** — npj Biofilms | 2021 | 13 luxI + 30 luxR from activated sludge metagenome. R:P ratio = 2.3:1. We can test our eavesdropper prediction against their data. Activated sludge is 3D floc → Anderson predicts QS-active. |
| P7 | **"QS and biofilm in anaerobic bacterial communities"** — IJMS | 2024 | Anaerobic communities. Important because our model assumes AHL diffusion — anaerobic conditions may change signal chemistry. Tests our exception for Bacteroides (anaerobic gut, no classical QS). |
| P8 | **"A review of QS mediating interkingdom interactions in the ocean"** — Commun Biol | 2025 | Marine QS review. Will help us assess which marine organisms truly do QS and which don't. Can refine our "obligate plankton = no QS" prediction. |

### Tier 3: Experimental Validation Targets

| # | Paper | Year | Why Read |
|---|-------|------|----------|
| P9 | Rajagopalan et al. **"Cell density, alignment, and orientation correlate with C-signal-dependent gene expression"** — PNAS | 2021 | Experimental data on Myxococcus C-signal → 3D transition. Our Anomaly #5. Can we extract the "critical cell density" and map it to our Anderson L_min prediction? |
| P10 | Devreotes et al. (2023 Frontiers) **"Integrated cross-regulation pathway for cAMP relay in Dictyostelium"** | 2023 | Updated relay model. Our Anomaly #6. The relay amplification circuit that defeats Anderson localization — can we model the relay mathematically as a non-Hermitian Anderson system? |

---

## Novelty Summary

| Claim | Novel? | Confidence |
|-------|--------|------------|
| Anderson localization → QS signal propagation | **YES** | HIGH (no prior work found) |
| Pielou J → Anderson W mapping | **YES** | HIGH (new bridge) |
| 28-biome dimensional QS phase diagram | **YES** | HIGH (new dataset) |
| 100%/0% atlas (3D vs 2D) | **YES** | HIGH (new finding) |
| W_eff = W_base/occupancy dilution | **YES** | HIGH (new model) |
| Anderson anomalies as NP solutions | **YES** | HIGH (new framework) |
| Cross-domain L_eff scaling | **YES** | MODERATE (prediction, not validated) |
| Producer/receiver NCBI analysis | **NOVEL APPROACH** | MODERATE (data exists, analysis is new) |
| Square-cubed vs Pólya recurrence | **YES (connection)** | HIGH (Pólya known, application novel) |
| QS distance scaling table | **YES (framing)** | FUN (pedagogical contribution) |

**Bottom line**: The core contribution — applying Anderson localization
theory to microbial quorum sensing — appears to be genuinely novel.
No published work connects these two fields. The dimensional phase diagram,
the anomaly framework, and the dilution model are all new.
