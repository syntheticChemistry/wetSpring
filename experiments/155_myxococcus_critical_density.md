# Exp155: Myxococcus C-Signal Critical Density

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (7/7 checks) |
| **Binary**     | `validate_myxococcus_critical_density` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Paper Queue Extension |
| **Paper**      | 37 (Rajagopalan et al. PNAS 2021) |

## Core Idea

Analyze Myxococcus xanthus as NP Solution #2 — the organism that bootstraps 3D
geometry from 2D using contact-dependent C-signal (Anderson-immune), then switches
to diffusible A-signal once the 3D fruiting body enables it.

## Key Findings

- Critical density: 5×10⁵ cells/mm² (175% surface coverage — multilayering required)
- Fruiting body: 50 µm height × 100 µm diameter → L_eff = 100-200 cells (>> L_min = 4)
- Two-stage signaling: contact (Anderson-immune) → aggregation → diffusion (Anderson-subject)
- Life cycle tracks Anderson transition: 2D localized → partial 3D → fully extended
- NP solution type: "build the hardware before running the algorithm"

## Connection to Sub-thesis 02

Myxococcus demonstrates how constrained evolution solves the Anderson barrier:
by bootstrapping the geometry that makes diffusible signaling possible.

---
