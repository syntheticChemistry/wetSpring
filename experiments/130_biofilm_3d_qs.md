# Exp130: Thick Biofilm 3D QS Extension

**Status:** PASS (GPU) — 9/9 checks
**Binary:** `validate_biofilm_3d_qs`
**Features:** `gpu`
**Extends:** Exp122 (2D biofilm)

## Hypothesis

H130: A 3D biofilm block sustains QS-active signaling at higher Pielou J
(greater diversity) than a 2D slab. J_c(3D) > J_c(2D).

## Design

- 2D slab: 20×20 = 400 sites (Exp122 geometry)
- 3D block: 8×8×6 = 384 sites (comparable total, 6-layer depth)
- 20-point disorder sweep W = 0.5 to 20.0
- J_c extraction via last downward midpoint crossing
- Biofilm diversity scan at J = 0.3, 0.4, 0.5, 0.6

## Key Results (GPU confirmed)

| Geometry  | Sites | Plateau (W>2) | J_c   |
|-----------|-------|--------------|-------|
| 2D slab   | 400   | 4            | 0.406 |
| 3D block  | 384   | 16           | 1.248 |

Plateau ratio: 3D/2D = 4.0× — even stronger than Exp127's cube comparison.

## Biofilm Diversity Scan

| J   | W    | 2D ⟨r⟩ | 2D Regime  | 3D ⟨r⟩ | 3D Regime  |
|-----|------|---------|------------|---------|------------|
| 0.3 | 4.85 | 0.4835  | ACTIVE     | 0.5308  | ACTIVE     |
| 0.4 | 6.30 | 0.4472  | suppressed | 0.5371  | ACTIVE     |
| 0.5 | 7.75 | 0.4478  | suppressed | 0.5621  | ACTIVE     |
| 0.6 | 9.20 | 0.4371  | suppressed | 0.5503  | ACTIVE     |

## Key Findings

1. **3D biofilm sustains QS at J=0.6 where 2D fails at J=0.4** — 3D triples the
   diversity tolerance
2. **J_c(3D block) ≈ 1.25 >> J_c(2D slab) ≈ 0.41** — 3× wider QS window
3. **Peak 3D ⟨r⟩ = 0.5621 at W=7.68** — strongly extended, near GOE
4. **3D block plateau is 4× wider than 2D slab** (16 vs 4 points)
5. **Just 6 layers of depth** transform QS capability — biofilm thickness matters

## Biological Implication

Thin biofilms (monolayer/bilayer) can only sustain QS in low-diversity
communities (J < 0.4). Adding just 6 layers of vertical structure
enables QS at diversities where 2D models predict failure. This explains
why mature, thick biofilms in clinical settings can coordinate resistance
genes even when harboring diverse species.
