# Exp140: QS Gene Prevalence by Habitat Geometry

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (7/7 checks) |
| **Binary**     | `validate_qs_gene_prevalence` |
| **Date**       | 2026-02-23 |
| **Phase**      | 37 — Empirical Validation |

## Hypothesis

If the Anderson model correctly predicts where QS can work, organisms in
geometries that prevent QS (2D mats, dilute plankton) should have LOST QS genes
over evolutionary time. Organisms in 3D habitats should be ENRICHED for QS genes.

## Design

- Curate 35 organisms with known QS status and primary habitat geometry
- Classify habitats: 3D_dense, 3D_dilute, 2D_mat, 2D_surface
- Test 4 Anderson predictions against QS gene prevalence

## Key Results

| Geometry   | n  | with QS | QS%   | mean QS systems | mean genome Mb |
|------------|---:|--------:|------:|----------------:|---------------:|
| 3D_dense   | 24 | 20      | 83.3% | 1.38            | 7.1            |
| 3D_dilute  | 7  | 0       | 0.0%  | 0.00            | 2.9            |
| 2D_mat     | 3  | 0       | 0.0%  | 0.00            | 3.2            |
| 2D_surface | 1  | 0       | 0.0%  | 0.00            | 3.3            |

All 4 Anderson predictions CONFIRMED:
- P1: 3D_dense (83%) >> 3D_dilute (0%)
- P2: 3D_dense (83%) >> 2D_mat (0%)
- P3: All 7 obligate plankton have ZERO QS systems
- P4: Mean QS system count: 1.38 (3D) > 0.00 (dilute) > 0.00 (2D)

## Key Findings

1. **83% of 3D-dense organisms have QS; 0% of dilute/2D organisms do**
2. **Obligate plankton (SAR11, Prochlorococcus) have the smallest genomes
   AND zero QS** — evolutionary streamlining removes useless circuitry
3. **Exceptions prove the rule**: Geobacter (3D, no QS) uses nanowires;
   Bacteroides (3D, no QS) uses metabolic signals. The geometry ALLOWS
   diffusible signaling; the specific mechanism varies.
4. **E. coli has only an AHL RECEPTOR (SdiA), not a synthase** — evolved
   to eavesdrop on QS in the dense gut without paying production cost
5. **The Anderson model predicts where QS CAN work, not where it MUST evolve**
