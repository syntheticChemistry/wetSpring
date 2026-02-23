# Exp129: Dimensional QS Phase Diagram

**Status:** PASS (GPU) — 12/12 checks
**Binary:** `validate_dimensional_phase_diagram`
**Features:** `gpu`
**Extends:** Exp122 (2D), Exp126 (28-biome atlas)

## Hypothesis

H129: The three dimensions partition biomes into distinct QS regimes. Biomes
QS-suppressed in 1D may be active in 2D, and biomes suppressed in 2D may be
active in 3D.

## Design

- 28 biomes from Exp126's `biome_diversity_params`
- Pre-computed disorder sweeps: 1D (N=400), 2D (20×20), 3D (8×8×8)
- Classify each biome × dimension as QS-active or QS-suppressed

## Key Results (GPU confirmed)

| Dimension | QS-active biomes | QS-suppressed biomes |
|-----------|-----------------|---------------------|
| 1D        | 0/28            | 28/28               |
| 2D        | 0/28            | 28/28               |
| 3D        | 28/28           | 0/28                |

Dimensional gains: 1D→2D = 0, 2D→3D = 28.

## Key Findings

1. **All 28 biomes are QS-active in 3D, zero in 1D or 2D** — 3D is the decisive dimension
2. **Dimensional monotonicity confirmed**: active(3D) >= active(2D) >= active(1D)
3. **Soil (J=0.99, W=14.85)** — suppressed in 1D+2D, active in 3D (W < W_c ≈ 16.5)
4. **The phase boundary is binary**: the 3D metal-insulator transition at W_c ≈ 16.5
   exceeds all natural biome disorder values (max W ≈ 14.85 for soil)
5. **Prediction**: any naturally occurring microbial community with 3D structure
   (soil, biofilm, sediment, chimney) can sustain community-wide QS

## Phase Diagram

```
       1D        2D        3D
J=0.0  ACTIVE    ACTIVE    (finite-size)
J=0.2  suppressed ACTIVE   ACTIVE
J=0.4  suppressed suppressed ACTIVE
J=0.6  suppressed suppressed ACTIVE
J=0.8  suppressed suppressed ACTIVE
J=1.0  suppressed suppressed ACTIVE
```

## Implication for Ecology

This result suggests that spatial dimensionality — not diversity alone —
determines whether microbial communities achieve collective behavior.
The 3D Anderson metal-insulator transition provides a theoretical
framework for understanding why soil bacteria, despite extreme diversity,
can coordinate antibiotic resistance, nutrient cycling, and pathogen
suppression through quorum sensing.
