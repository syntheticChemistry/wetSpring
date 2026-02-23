# Exp128: Vent Chimney Geometry QS Prediction

**Status:** PASS (GPU) — 12/12 checks
**Binary:** `validate_vent_chimney_qs`
**Features:** `gpu`
**Extends:** Exp107 Section 5 (Anderson 3D baseline)

## Hypothesis

H128: Vent chimney porosity and mineral heterogeneity determine QS propagation
depth. High-porosity chimney sections support QS; dense mineral zones suppress
it in 2D but sustain it in 3D.

## Design

- 4 chimney zones: young sulfide, mature anhydrite, silica conduit, weathered exterior
- Parameters: porosity (5-40%), mineral heterogeneity (0-1), temperature (20-350°C)
- Mapping: porosity/heterogeneity/temperature → Anderson disorder W
- 3D lattice (8×8×8) and 2D lattice (20×20) comparison per zone

## Key Results (GPU confirmed)

| Chimney Zone       | Porosity | W     | 2D ⟨r⟩ | 2D Regime  | 3D ⟨r⟩ | 3D Regime  |
|-------------------|----------|-------|---------|------------|---------|------------|
| Young sulfide     | 30%      | 7.29  | 0.4448  | suppressed | 0.5237  | QS-active  |
| Mature anhydrite  | 8%       | 16.25 | 0.3795  | suppressed | 0.4760  | QS-active  |
| Silica conduit    | 15%      | 5.48  | 0.4845  | ACTIVE     | 0.5185  | QS-active  |
| Weathered exterior| 35%      | 13.49 | 0.3813  | suppressed | 0.5074  | QS-active  |

## Key Findings

1. **3 of 4 chimney zones flip from 2D-suppressed to 3D-active** — 3D is essential
2. **Mineral heterogeneity dominates disorder** — silica W < young < weathered < mature
3. **All zones are QS-active in 3D** — the 3D metal-insulator transition is high enough
4. **Young sulfide has highest QS potential** — high porosity + low heterogeneity
5. **A 2D slab model misses 75% of chimney QS capability** — depth matters

## Biological Implication

Hydrothermal vent chimneys are fundamentally 3D porous structures. Modeling them
as 2D surfaces severely underestimates their capacity for community-wide quorum
sensing. Even dense, mature anhydrite zones (W=16.25) sustain extended states in 3D.
