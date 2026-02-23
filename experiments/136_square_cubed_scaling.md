# Exp136: Square-Cubed Law & Interior Fraction Scaling

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (6/6 checks, GPU-confirmed) |
| **Binary**     | `validate_square_cubed_scaling` |
| **Date**       | 2026-02-23 |
| **Phase**      | 36c — Why Analysis |

## Hypothesis

The user's intuition: 3D advantage is the square-cubed law — larger cubes
have proportionally more interior (L³ volume, L² surface). Interior cells
are shielded from boundary disorder. Is this the whole story?

## Design

- 2D lattices at W=13 for L=6 to 30 (36 to 900 sites)
- 3D lattices at W=13 for L=4 to 12 (64 to 1728 sites)
- Compute interior fraction, ⟨r⟩, and their Pearson correlation
- Multi-W scaling at W=5, 10, 15, 20 for 3D cubes

## Key Results

2D at W=13 (all suppressed):

| L  | N    | interior% | ⟨r⟩   |
|----|------|-----------|--------|
| 6  | 36   | 44.4%     | 0.4285 |
| 14 | 196  | 73.5%     | 0.4267 |
| 30 | 900  | 87.1%     | 0.4040 |

3D at W=13 (all active):

| L  | N    | interior% | ⟨r⟩   |
|----|------|-----------|--------|
| 4  | 64   | 12.5%     | 0.4667 |
| 8  | 512  | 42.2%     | 0.4757 |
| 12 | 1728 | 57.9%     | 0.4961 |

- Pearson correlation(interior_fraction, ⟨r⟩) in 3D = **0.526** (moderate)
- 2D: NEVER active at W=13, even at L=30 (900 cells)
- 3D: Active at L=4 (just 64 cells!)

## Key Findings

1. **Square-cubed law CONTRIBUTES (r=0.53) but is NOT the full story**
2. **The dominant effect is TOPOLOGICAL**: random walk recurrence
   - d≤2: return probability = 1 → signal always scatters back → localization
   - d≥3: return probability < 1 → signal can propagate indefinitely
3. **125 cells in 3D > 900 cells in 2D** — a 5×5×5 cube beats a 30×30 sheet
4. **The transition is qualitative (d=2 vs d=3), not quantitative (size)**
5. Minimum 3D colony for QS at W=13: **L=4 (64 cells)**
