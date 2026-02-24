# Sub-thesis 01: Anderson Localization as QS Null Hypothesis

**Date:** February 24, 2026
**Faculty:** Kachkovskiy (MSU CMSE), Waters (MSU MMG)
**Status:** Active — 151 experiments, W_c quantified

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

## Open Questions

1. Can ν be extracted precisely with L > 12 (needs GPU Lanczos)?
2. Does correlated disorder explain hot spring mat QS anomalies?
3. Can ⟨r⟩ be computed from real bacterial colony coordinates (Exp149 proposal)?

## Connection to Gen3 Thesis

Chapter 14: Biological validation of Anderson localization framework.
The Anderson model is the null hypothesis; nature's violations are the science.
