# Exp146: luxR Phylogeny × Habitat Geometry Overlay

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (5/5 checks) |
| **Binary**     | `validate_luxr_phylogeny_geometry` |
| **Date**       | 2026-02-23 |
| **Phase**      | 38 — Extension Papers |

## Core Idea

Overlay habitat geometry on the luxR receptor family evolutionary tree
(BMC Genomics 2024). Test whether QS gene gain/loss correlates with lineage
transitions between habitat geometries (biofilm → planktonic, 3D → 2D).

## 12 Lineages Analyzed

| Clade | Geometry | LuxR Status | LuxI Paired? | Cross-species? |
|-------|----------|-------------|:------------:|:--------------:|
| Vibrio (marine biofilm) | 3D_dense | intact | yes | no |
| Vibrio (marine planktonic) | 3D_dilute | INVERTED | no | yes |
| Pseudomonas (biofilm/soil) | 3D_dense | intact | yes | no |
| Pseudomonas (plant leaf) | 2D_surface | ABSENT | no | no |
| Enterobacteriaceae (gut) | 3D_dense | solo (eavesdropper) | no | yes |
| Rhizobiaceae (root nodule) | 3D_dense | paired + solo | yes | yes |
| SAR11 (open ocean) | 3D_dilute | ABSENT | no | no |
| Prochlorococcus | 3D_dilute | ABSENT | no | no |

## Key Findings

- 3D_dense: 100% retain luxR (8/8 lineages)
- 3D_dilute: 33% retain luxR (1/3, V. cholerae with inverted logic)
- 2D_surface: 0% retain luxR (0/1)
- Solo receptors (eavesdroppers) are enriched in 3D mixed-species habitats
- Lineage transitions from biofilm → planktonic show QS gene loss

## Connection to Sub-thesis 05

Solo luxR receptors in gut bacteria = interspecies eavesdropping network.
Rhizobial luxR + plant flavonoid = cross-kingdom QS bridge.
