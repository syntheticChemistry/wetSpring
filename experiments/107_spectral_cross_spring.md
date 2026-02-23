# Exp107: Cross-Spring Spectral Theory — Anderson Localization for Quorum Sensing

**Date**: February 23, 2026
**Status**: PASS — 25/25 checks
**Binary**: `validate_spectral_cross_spring` (requires `gpu` feature for barracuda import path)
**Paper**: Bourgain & Kachkovskiy (2018) "Anderson localization for two interacting quasiperiodic particles" — GAFA

## Purpose

Exercises `barracuda::spectral` primitives from wetSpring's import path, bridging
Kachkovskiy/Bourgain spectral theory to the quorum-sensing domain.  Previously
spectral theory lived exclusively in hotSpring (Papers 14–22) and neuralSpring
(Papers 022–023).  This experiment brings Anderson localization into wetSpring's
validation scope as a cross-spring primitive exercise.

## Conceptual Link

Autoinducer diffusion through a heterogeneous bacterial population is analogous
to wave propagation in a disordered medium.  Anderson localization predicts when
signals stay local (localized states) vs. propagate community-wide (extended
states), depending on population heterogeneity (disorder W).

## Validation Sections

### 1. Anderson 1D (7 checks)
- Gershgorin bounds: σ(H) ⊂ [-2-W/2, 2+W/2]
- Eigenvalue count = N (500)
- Lyapunov γ(0) > 0 (all states localized in 1D)
- Kappus–Wegner: γ(0) ≈ W²/96 (relative error < 30% at W=4)
- Lyapunov γ(1.8) > 0 at band edge
- Level statistics ⟨r⟩ ≈ Poisson (0.3973 vs 0.3863)

### 2. Almost-Mathieu / Aubry–André (6 checks)
- Herman formula: γ(0) = ln(λ) for λ = 1.5, 2.0, 3.0 (all within 0.004)
- Aubry–André transition: λ=0.5 → γ ≈ 0 (extended), λ=2.0 → γ > 0 (localized)
- Spectrum bounds: σ(H) ⊂ [-2-2λ, 2+2λ]

### 3. Lanczos vs Sturm (3 checks)
- Extremal eigenvalues agree (N=200 1D Anderson)
- Full Lanczos returns ≥ N/2 eigenvalues

### 4. Anderson 2D (3 checks)
- Weak disorder (W=1): ⟨r⟩ = 0.4603 > Poisson
- Strong disorder (W=20): ⟨r⟩ = 0.4168 ≈ Poisson
- Gershgorin bounds valid

### 5. Anderson 3D (3 checks)
- Metallic regime (W=2): ⟨r⟩ = 0.4843 > Poisson
- Insulating regime (W=25): ⟨r⟩ = 0.4171 ≈ Poisson
- Gershgorin bounds valid

### 6. QS-Disorder Analogy (3 checks)
- ⟨r⟩ decreases from W=0.5 (0.4907) to W=10 (0.3850) — monotonic
- High heterogeneity ⟨r⟩ ≈ Poisson (signals localized)
- Lyapunov γ(W=10) >> γ(W=0.5) — localization increases with heterogeneity

## Primitives Exercised

All from `barracuda::spectral` (ToadStool upstream):
- `anderson_hamiltonian`, `anderson_2d`, `anderson_3d`
- `almost_mathieu_hamiltonian`
- `lyapunov_exponent`
- `lanczos`, `lanczos_eigenvalues`
- `find_all_eigenvalues`
- `level_spacing_ratio`, `POISSON_R`, `GOE_R`

## Cross-Spring Coverage

| Spring | Papers | Status |
|--------|--------|--------|
| hotSpring | 14–22 (41/41 checks) | Full reproduction |
| neuralSpring | 022–023 (33/33 checks) | Spectral commutativity + Anderson |
| wetSpring | 23 (25/25 checks) | Cross-spring exercise + QS bridge |

## Reproduction

```bash
cargo run --features gpu --bin validate_spectral_cross_spring
```

Expected output: 25/25 PASS in ~3 seconds.
