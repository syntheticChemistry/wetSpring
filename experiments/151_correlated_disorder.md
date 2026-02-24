# Exp151: Disorder-Correlated Lattices for Biofilm Disorder

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (8/8 checks) |
| **Binary**     | `validate_correlated_disorder` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Correlated Disorder |

## Core Idea

Real biofilms have spatially clustered species (microcolonies), not i.i.d.
random disorder. This experiment tests how spatial correlation in the
Anderson potential affects the metal-insulator transition.

## Method

- L = 8 (512 sites), 4 realizations per point
- Disorder sweep: W ∈ [8, 28], 11 points
- Correlation lengths: ξ_corr = 0, 1, 2, 4 (lattice spacings)
- Exponential smoothing kernel: K(r) = exp(−r/ξ_corr)
- Variance-normalized to preserve nominal disorder strength

## Results

| ξ_corr | Biological Regime | W_c | ⟨r⟩ at W=28 |
|:------:|-------------------|:---:|:-----------:|
| 0 | well-mixed planktonic | 16.49 | 0.421 |
| 1 | loose aggregation | >28 | 0.474 |
| 2 | mature biofilm | >28 | 0.492 |
| 4 | dense biofilm clusters | >28 | 0.481 |

## Key Findings

1. **Uncorrelated (ξ=0): W_c = 16.49** — matches Exp150 and literature (16.5).

2. **Even ξ=1 pushes W_c beyond W=28.** At ξ=1, the ⟨r⟩ curve barely
   decreases from GOE — the system is almost entirely in the extended
   (metallic/QS-active) regime across the entire sweep.

3. **ξ=2 and ξ=4 are robustly extended.** ⟨r⟩ ≈ 0.50 even at W=28,
   well above the midpoint (0.459) and close to GOE (0.531).

4. **The biological implication is strong**: spatial clustering of species
   in biofilms DRAMATICALLY reduces effective scattering. The i.i.d.
   Anderson model is a very conservative lower bound for QS propagation.

## Biological Interpretation

| Regime | Anderson Prediction (i.i.d.) | With Clustering |
|--------|------------------------------|-----------------|
| Soil (W ≈ 6.7) | Extended ✓ | Extended ✓✓ |
| Gut (W ≈ 4) | Extended ✓ | Extended ✓✓ |
| Hot spring (W ≈ 19) | Localized ✗ | **Extended ✓** (if clustering present) |

This means:
- The 100%/0% QS atlas based on i.i.d. Anderson is a **lower bound** —
  real biofilms are MORE QS-active than predicted.
- Hot spring microbial mats, if they have spatial clustering (which some do),
  could support QS despite high nominal disorder. This resolves a potential
  anomaly in the framework.
- The Anderson null hypothesis becomes even stronger: only truly well-mixed,
  uncorrelated communities at high diversity should fail QS.

## Connection to Sub-theses

- **01 (Anderson-QS)**: Correlated disorder strengthens the QS prediction
  by showing the model is conservative.
- **03 (BioAg)**: Rhizosphere biofilms have strong spatial structure →
  QS is even more robust than i.i.d. predicts.
- **05 (Cross-species)**: Multispecies microcolonies create correlated
  disorder → interspecies QS facilitated by spatial clustering.
