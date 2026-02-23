# Exp122: 2D Anderson Spatial QS Lattice

**Status:** PASS (12/12 checks)
**Binary:** `validate_anderson_2d_qs`
**Features:** `--features gpu`
**Date:** 2026-02-23
**GPU confirmed:** Yes (release build, ~7s)

## Purpose

Builds 2D Anderson lattices (20x20 = 400 sites) with disorder derived from Pielou J, comparing 1D vs 2D localization transitions. Tests whether 2D spatial structure creates an extended QS-active plateau absent in 1D.

## Design

Construct 2D tight-binding Hamiltonian with random on-site potential V_i in [-W/2, W/2]. Map Pielou evenness J to Anderson disorder W. Compute level spacing ratio ⟨r⟩ across 20-point disorder sweep (W=0.5–15). Compare 1D chain vs 2D lattice. Classify QS regime from eigenvalue statistics (Poisson vs GOE crossover). Map 6 ecosystems to Anderson disorder space.

## Data Source

Synthetic ecosystem diversity parameters mapped to Anderson disorder. Six ecosystem types (biofilm n=5, bloom n=8, gut n=300, vent n=150, soil n=1000, ocean n=800) provide Pielou J values; linear mapping J→W drives lattice construction.

## GPU Results

- **2D has genuine extended plateau absent in 1D**: 8 sweep points above GOE/Poisson midpoint for W>2
- 1D localizes almost immediately: ⟨r⟩ drops below midpoint by W≈1.3
- 2D maintains QS-active states up to W≈5.8 (extended regime)
- **Bloom (W=2.93): QS-suppressed in 1D but QS-ACTIVE in 2D** — spatial structure preserves signaling
- Biofilm (W=0.87): QS-active in both 1D and 2D
- Gut, vent, soil, ocean (W>14): QS-suppressed in both
- Critical Pielou evenness J_c ≈ 0.41 (ecologically meaningful range 0.2–0.7)

## Key Finding

The 2D lattice creates a QS-active window for moderately dominated communities (bloom, biofilm) that 1D cannot support. In 1D, Anderson's theorem guarantees all states localize for any W>0. In 2D, the extended plateau means that spatial structure (biofilm geometry, bloom patches) preserves community-wide QS signaling at disorder levels where 1D signaling would collapse. This suggests 3D lattices (e.g., thick biofilms, hydrothermal vent chimneys) may have even wider QS-active windows — a testable prediction for Phase 36.

## Reproduction

```bash
cargo run --release --features gpu --bin validate_anderson_2d_qs
```
