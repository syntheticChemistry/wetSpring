# Sub-thesis 01: Anderson Localization as QS Null Hypothesis

**Date:** February 27, 2026
**Faculty:** Kachkovskiy (MSU CMSE), Waters (MSU MMG)
**Status:** Active — 184 experiments (Exp107-156, 170-182, 356), 3,418+ checks, W_c = 16.26 ± 0.95 (finite-size scaling L=6-12), Track 4 soil QS complete (9 papers, full three-tier), 9 extension papers validated, correlated disorder + dilution effects quantified, **V110: O₂-modulated W model (H3, r=0.851) validated against 10 environments**, clippy pedantic CLEAN, 79 named tolerances

---

## Core Claim

The Anderson metal-insulator transition, applied to microbial community
structure, provides a physics-based null hypothesis for where quorum
sensing (QS) can and cannot operate. Habitat geometry (dimensionality)
and species diversity (disorder W) determine QS viability.

## Key Results

| Finding | Experiment | Checks |
|---------|:----------:|:------:|
| 28-biome QS atlas: 100% active in 3D, 0% in 2D | Exp129 | 12 |
| W_c ≈ 16.5 (3D metal-insulator transition) | Exp131 | 11 |
| W_c = 16.26 ± 0.95 (disorder-averaged, L=6–12) | Exp150 | 14 |
| Correlated disorder pushes W_c > 28 | Exp151 | 8 |
| J → W mapping: Pielou evenness → Anderson disorder | Exp126 | 90 |
| Dilution amplifies effective disorder: W_eff = W/occupancy | Exp137 | 10 |
| Eukaryote scaling: same physics, different L_eff | Exp138 | 11 |
| Extension papers: cold seep, wave synthesis, burst stats | Exp144–149 | 36 |
| **O₂-modulated W(H',O₂) model beats single-variable (r=0.851 vs -0.575)** | **Exp356** | **18** |
| **Signal dilution: diversity IS disorder (H2 r=+0.812)** | **Exp356** | — |

## Novelty Assessment

From PHASE_37_LITERATURE_REVIEW.md:
- **Anderson localization → QS signal propagation**: No prior work found.
- **Pielou J → Anderson W mapping**: Novel bridge between ecology and condensed matter.
- **28-biome dimensional phase diagram**: New dataset.
- **Anderson anomalies as NP solutions**: New framework.

## Computational Infrastructure

| Primitive | Source | Usage |
|-----------|--------|-------|
| `anderson_3d` | ToadStool (hotSpring provenance) | Hamiltonian construction |
| `lanczos` / `lanczos_eigenvalues` | ToadStool | Eigenvalue extraction |
| `level_spacing_ratio` | ToadStool | ⟨r⟩ diagnostic (GOE vs Poisson) |
| `BatchedOdeRK4<S>` | ToadStool (wetSpring provenance) | QS ODE parameter sweeps |

## V110: Oxygen-Modulated Anderson Model (Exp356)

**Key finding (March 2026):** The original H' → W mapping (W = 20·exp(-0.3·H'))
is **wrong for cross-environment comparison**. It predicts MORE QS in diverse
aerobic soil than in a monoculture — contradicting known QS biology.

Three alternative W parameterizations were tested against literature QS
prevalence scores for 10 environments (Lab E. coli, P. aeruginosa biofilm,
human gut, anaerobic digester, oral biofilm, rhizosphere, ocean surface,
bulk soil, hot spring, deep-sea vent):

| Model | W function | Pearson r | MAE |
|-------|-----------|:---------:|:---:|
| H1 (inverse diversity) | W = 20·exp(-0.3·H') | -0.575 | 0.418 |
| H2 (signal dilution) | W = 4·H' | +0.812 | 0.331 |
| **H3 (O₂-modulated)** | **W = 3.5·H' + 8·O₂** | **+0.851** | **0.331** |

**Interpretation:** In diverse communities, QS signals from one species get
"scattered" by all the others — diversity IS the disorder, not the opposite.
Adding oxygen as a second disorder dimension captures FNR/ArcAB/Rex-mediated
QS gene regulation: anaerobic conditions reduce transcriptional noise for QS
operons, giving anaerobic communities (gut, digesters) a QS advantage over
aerobic environments (ocean, bulk soil) at similar diversity levels.

**Predictions confirmed:**
- Monoculture (E. coli) > biofilm > gut > digester > rhizosphere > ocean > soil
- Anaerobic mean P(QS) > aerobic mean P(QS)
- Known biology correlation: r = +0.851

**Experimental validation path:** This is testable with real 16S +
metatranscriptomic data. NCBI SRA has paired 16S/metatranscriptome datasets
from anaerobic gut (HMP), aerobic soil (EMP), and digester (ADREC) environments.
Quantify QS gene expression (luxS, lasI, rhlI, CSP genes) vs community diversity
and O₂ regime. If the H3 model holds, QS gene expression / diversity should
follow the W = 3.5·H' + 8·O₂ curve.

**Eventual lab confirmation:** Simple plate assays with reporter strains
(V. harveyi BB170 for AI-2, C. violaceum CV026 for AHL) in:
1. Monoculture (positive control)
2. Defined 5-species community (aerobic)
3. Same 5-species community (anaerobic)
4. Complex soil extract (aerobic)
5. Complex digester extract (anaerobic)

Prediction: reporter activation should decrease with diversity (signal dilution)
and increase under anaerobic conditions (O₂ disorder reduction), following H3.

## Open Questions

1. Can ν be extracted precisely with L > 12 (needs GPU Lanczos)?
2. Does correlated disorder explain hot spring mat QS anomalies?
3. Can ⟨r⟩ be computed from real bacterial colony coordinates (Exp149 proposal)?
4. **NEW (V110):** Does the two-dimensional W(H', O₂) model hold for real metatranscriptomic QS gene expression data?
5. **NEW (V110):** What is the oxygen coefficient in W = α·H' + β·O₂? Is β consistent across biome types?

## Connection to Gen3 Thesis

Chapter 14: Biological validation of Anderson localization framework.
The Anderson model is the null hypothesis; nature's violations are the science.
The V110 H3 model refines the null hypothesis: disorder has two sources
(community diversity and oxygen regime), not one.
