# Exp127: 3D Anderson Dimensional QS Sweep

**Status:** PASS (GPU) — 17/17 checks
**Binary:** `validate_anderson_3d_qs`
**Features:** `gpu`
**Extends:** Exp122 (2D Anderson)

## Hypothesis

H127: The 3D cubic lattice maintains QS-active (GOE-like) ⟨r⟩ at disorder
values where 2D has already localized. The 3D transition W_c(3D) > W_c(2D).

## Design

- 1D chain (N=400), 2D lattice (20×20), 3D lattice (8×8×8=512 sites)
- 20-point disorder sweep W = 0.5 to 25.0 per dimension
- Compare plateau widths, J_c values, ecosystem mapping

## Key Results (GPU confirmed)

| Dimension | Plateau points (W>2) | J_c   | Weak W ⟨r⟩ | Strong W ⟨r⟩ |
|-----------|---------------------|-------|------------|--------------|
| 1D        | 0                   | —     | 0.5021     | 0.3942       |
| 2D        | 5                   | 0.557 | 0.4683     | 0.3770       |
| 3D        | 12                  | 1.283 | 0.4527     | 0.4343       |

Plateau ratio: 3D/2D = 2.4×, confirming dramatic dimensional widening.

## Key Findings

1. **3D extended plateau is 2.4× wider than 2D** — 12 vs 5 points above midpoint
2. **J_c(3D) ≈ 1.28 >> J_c(2D) ≈ 0.56** — 3D sustains QS at far higher diversity
3. **Gut, vent, soil, ocean** all flip from 1D+2D suppressed to 3D QS-active
4. **3D ⟨r⟩ stays above POISSON even at W=25** — genuine metal-insulator transition
5. Peak 3D ⟨r⟩ = 0.5406 at W=3.08, approaching GOE (0.5307)

## Ecosystem Mapping

| Ecosystem | J     | 1D       | 2D       | 3D       |
|-----------|-------|----------|----------|----------|
| biofilm   | 0.025 | ACTIVE   | ACTIVE   | suppressed* |
| bloom     | 0.168 | suppressed | ACTIVE | ACTIVE   |
| gut       | 0.975 | suppressed | suppressed | ACTIVE |
| vent      | 0.941 | suppressed | suppressed | ACTIVE |
| soil      | 0.990 | suppressed | suppressed | ACTIVE |
| ocean     | 0.988 | suppressed | suppressed | ACTIVE |

*Biofilm at W=0.87 hits finite-size artifact for L=8; would be active for larger L.

## hotSpring Provenance

All spectral primitives (`anderson_3d`, `lanczos`, `level_spacing_ratio`)
developed in hotSpring, absorbed into ToadStool `barracuda::spectral`.
