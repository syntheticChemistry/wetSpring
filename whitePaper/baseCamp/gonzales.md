<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# Prof. Andrea J. Gonzales — MSU Pharmacology & Toxicology

**Track:** 5 — Immunological Anderson & Drug Repurposing
**Papers to reproduce:** 6 (Papers 53–58)
**Total checks:** 359+ (Gonzales pipeline validated — dose-response, PK decay, tissue lattice, provenance chain)
**IPC methods:** `science.gonzales.dose_response`, `science.gonzales.pk_decay`, `science.gonzales.tissue_lattice`
**Domains:** Cytokine signaling, JAK/STAT pathway, dose-response modeling,
tissue geometry, neuro-immune axis, drug repurposing

---

## Connection to wetSpring

Gonzales's 18 years at Zoetis produced the clinical and mechanistic data for
JAK inhibitors (oclacitinib/Apoquel) and anti-IL-31 monoclonal antibodies
(lokivetmab/Cytopoint) in veterinary dermatology. Her empirical data maps
directly onto the Anderson localization framework: cytokines are diffusible
signals propagating through disordered tissue — the same physics that governs
quorum sensing autoinducers (Paper 01) and soil QS (Paper 06).

The connection to Fajgenbaum (Track 3) creates a bridge: Anderson geometry
as a filtering dimension for drug repurposing scoring. MATRIX asks "does the
drug hit the right pathway?" Anderson adds "can the drug physically reach
the target cell through tissue geometry?"

---

## Papers

| # | Citation | Experiment | Checks | Status |
|---|----------|-----------|:------:|--------|
| 53 | Gonzales et al. 2013, *Vet Dermatol* 24:48-53 | Exp273 (partial) | — | Proposed |
| 54 | Gonzales et al. 2014, *J Vet Pharmacol Ther* 37:317-324 | Planned | — | Proposed |
| 55 | Gonzales et al. 2016, *Vet Dermatol* 27:34-e10 | Planned | — | Proposed |
| 56 | Fleck,...,Gonzales 2021, *Vet Dermatol* 32:681-e182 | Planned | — | Proposed |
| 57 | Gonzales et al. 2024, *J Vet Pharmacol Ther* 47:447-453 | Planned | — | Proposed |
| 58 | McCandless et al. 2014, *Vet Immunol Immunopathol* 157:42-48 | Planned | — | Proposed |

---

## Reproduction Plan

### Paper 53: Gonzales 2013 — IL-31 in Canine Atopic Dermatitis

**To reproduce:** IL-31 serum elevation data, pruritus scoring, nerve
activation profiles. Map to Anderson model: IL-31 as propagating signal,
tissue heterogeneity as disorder W, nerve activation threshold as
localization boundary.

### Paper 54: Gonzales 2014 — Oclacitinib JAK1 Selectivity

**To reproduce:** IC50 dose-response curves for JAK1 vs JAK2 vs JAK3.
Map IC50 to Anderson barrier height: drug concentration reduces effective W
by blocking signal transduction at receptor level.

### Paper 55: Gonzales 2016 — IL-31 Pruritus Model

**To reproduce:** Time-series pruritus scores at 1, 6, 11, 16 hours for
oclacitinib vs prednisolone vs dexamethasone. LSTM time-series prediction
of treatment response (neuralSpring). Anderson model: treatment as W
reduction over time.

### Paper 56: Fleck/Gonzales 2021 — Lokivetmab Pharmacodynamics

**To reproduce:** Dose-dependent duration curves (0.125/0.5/2.0 mg/kg at
14/28/42 days). Pharmacokinetic decay as signal extinction in Anderson model.
ESN regime classifier for treatment phase transitions.

### Paper 57: Gonzales 2024 — Flea Allergic Dermatitis

**To reproduce:** JAK1 selectivity in FAD model. Cross-disease validation:
same Anderson pathway (JAK1/STAT → cytokine block) works in both AD and FAD.

### Paper 58: McCandless 2014 — IL-31 Target Cells

**To reproduce:** Three-compartment cell-type mapping (immune, skin, neuronal).
Build multi-compartment Anderson lattice with cell-type-specific on-site
energies and inter-compartment hopping.

---

## The Anderson Mapping

| Anderson QS (Paper 01) | Immunological Extension |
|------------------------|------------------------|
| Lattice site | Cell position in tissue |
| On-site energy ε_i | Cell type identity (keratinocyte, Th2, neuron, mast cell) |
| Hopping parameter t | Cytokine diffusion coefficient in ECM |
| Disorder W | Cell-type heterogeneity (Pielou evenness of cell populations) |
| Dimension d | Tissue geometry (epidermis ≈ 2D; dermis ≈ 3D) |
| Level spacing ratio r | Cytokine signal extended (propagating) vs localized (confined) |

---

## Dimensional Promotion–Collapse Duality

Paper 06 (no-till soil): Tillage = dimensional COLLAPSE (3D → 2D) →
QS fails → ecosystem services collapse.

Paper 12 (AD skin): Scratching = dimensional PROMOTION (2D → 3D) →
cytokine delocalization → inflammatory cascade amplifies.

Same physics, opposite direction, opposite outcome. The Anderson framework
is agnostic — it predicts signal propagation. Whether propagation is
beneficial or pathological depends on the biological context.

---

## Cross-Spring Connections

| Spring | Component | Application |
|--------|-----------|-------------|
| wetSpring | Anderson spectral, diversity, NCBI pipeline | Core lattice simulation, tissue diversity metrics, gene expression queries |
| neuralSpring | ESN regime classifier, LSTM time series | AD state classification, treatment response prediction |
| groundSpring | Transport, uncertainty, spectral validation | Cytokine propagation distance, measurement confidence |
| hotSpring | Brain architecture, attention state | Monitoring pipeline for disease state transitions |
