# Phase 39 Design: Finite-Size Scaling + Correlated Disorder + Paper Queue

**Date:** February 24, 2026
**Status:** Exp150-161 ALL PASS (104 checks); paper queue ALL GREEN (43/43 papers)
**Predecessor:** Phase 38 (Extension papers — COMPLETE), Phase 37 (Anderson null hypothesis)
**Connection:** Gen3 thesis Ch. 14, baseCamp Sub-thesis 01 (Anderson-QS)
**Source:** V020 handoff "Next Phase" items

---

## Objective

Extract the thermodynamic critical disorder W_c for the Anderson metal-insulator
transition in 3D and characterize how realistic biofilm disorder (spatially
correlated) modifies the transition. These results directly feed the Anderson-QS
framework by providing the physics-grade W_c value against which biological
systems are compared.

## Background

Exp131 performed an initial finite-size scaling study (L = 6–10) using a single
realization per (L, W) point. The ⟨r⟩ crossing provided a rough W_c estimate.
Phase 39 sharpens this:

1. **Disorder averaging**: Multiple realizations reduce statistical noise
2. **Larger lattices**: L = 12 provides better scaling
3. **Scaling collapse**: Extract critical exponent ν from ⟨r⟩(W, L)
4. **Correlated disorder**: Real biofilms don't have i.i.d. random disorder —
   species cluster spatially. What does this do to W_c?

---

## Experiments

### Exp150: Finite-Size Scaling with Disorder Averaging

**Goal**: Extract W_c and ν from 3D Anderson model with proper statistics.

**Method**:
- Lattice sizes: L = 6, 8, 10, 12 (cubes, L³ sites)
- Disorder sweep: W ∈ [10, 22], 13 points (focus around expected W_c ≈ 16.5)
- Realizations: 8 per (L, W) point — disorder-averaged ⟨r⟩
- For each realization: `anderson_3d(L, L, L, W, seed)` → `lanczos` → eigenvalues → ⟨r⟩`
- Extract:
  - ⟨r⟩_avg(W, L) and standard error
  - W_c from curve crossing (all L curves cross at W_c)
  - ν from scaling collapse: ⟨r⟩ = f((W − W_c)L^(1/ν))

**Checks**:
1. W_c in [14, 20] (literature: ~16.5 for standard Anderson)
2. All ⟨r⟩_avg monotonically decrease with W (for fixed L)
3. Crossing point stable across L pairs
4. ν in [1.0, 2.0] (literature: ν ≈ 1.57 for 3D Anderson)
5. Standard error decreases with realizations

**Binary**: `validate_finite_size_scaling_v2`

### Exp151: Disorder-Correlated Lattices

**Goal**: Model realistic biofilm disorder where species cluster spatially.

**Method**:
- Generate correlated random potentials using exponential spatial correlation:
  V_i = Σ_j K(|r_i − r_j|) · ε_j, where K(r) = exp(−r/ξ_corr) and ε ~ U[-W/2, W/2]
- Correlation lengths: ξ_corr = 0 (uncorrelated), 1, 2, 4 lattice spacings
- Compare ⟨r⟩(W) curves for each ξ_corr
- Biological interpretation:
  - ξ_corr = 0: well-mixed planktonic community
  - ξ_corr = 1: loosely aggregated biofilm
  - ξ_corr = 2–4: mature biofilm with spatial clustering

**Checks**:
1. ξ_corr = 0 reproduces uncorrelated results (matches Exp150)
2. Increased ξ_corr shifts W_c upward (smoother disorder ≈ less effective scattering)
3. Strong correlation (ξ_corr = 4) shows qualitatively different ⟨r⟩ behavior
4. All eigenvalues finite for all correlation lengths
5. Biological mapping: biofilm clusters reduce effective disorder, making QS easier

**Binary**: `validate_correlated_disorder`

### Exp152: IPR-Based Finite-Size Analysis

**Goal**: Use inverse participation ratio (IPR) as a complementary localization
diagnostic alongside ⟨r⟩.

**Method**:
- Same (L, W) grid as Exp150
- For each eigenstate: IPR = Σ |ψ_i|⁴
- Extended states: IPR ~ 1/N (delocalized)
- Localized states: IPR ~ O(1) (concentrated on few sites)
- Compute typical IPR (geometric mean) at band center

**Checks**:
1. IPR transition at same W_c as ⟨r⟩ transition
2. Extended regime: IPR < 0.1 / N
3. Localized regime: IPR > 0.01
4. Transition sharpens with increasing L

**Binary**: `validate_ipr_finite_size`

---

## Connection to Sub-theses

| Sub-thesis | Connection |
|------------|------------|
| 01 (Anderson-QS) | W_c defines the phase boundary for QS viability |
| 02 (LTEE) | Finite-size effects relevant to small evolving populations |
| 03 (BioAg) | Correlated disorder models rhizosphere spatial structure |
| 05 (Cross-species) | Multi-species clusters create correlated disorder patterns |

## Success Criteria

1. W_c extracted with statistical error bars (not just a single crossing)
2. ν consistent with literature (1.57 ± 0.2)
3. Correlated disorder shifts W_c in the predicted direction
4. IPR confirms ⟨r⟩ results independently
5. Results directly usable in baseCamp Sub-thesis 01
