# wetSpring V21 — Why Analysis: Mapping, Scaling, Dilution, Eukaryotes

**Date:** February 23, 2026
**Experiments:** 135-138 (Phase 36c)
**Checks:** 35 new (2,954+ total, all PASS)
**Binaries:** 4 new GPU-confirmed

---

## What This Handoff Adds

Phase 36c interrogates the *why* behind the clean 100%/0% atlas split from
Phase 36b. Four experiments test whether the results are modeling artifacts
or genuine physics predictions, explore planktonic/fluid systems, and extend
the model across domains of life.

## Key Findings

### 1. The 100%/0% split is NOT a modeling artifact (Exp135)

Tested 9 mapping slopes α ∈ [5, 35]. The split reflects Anderson's theorem:
- d≤2: all states localize for any W>0 (Abrahams et al. 1979)
- d≥3: genuine metal-insulator transition at W_c ≈ 16.5
- Natural biomes J ∈ [0.73, 0.99] → W ∈ [11.1, 14.9]: always below W_c(3D),
  always above W_c(2D)

Low-diversity communities (monocultures, J<0.05) CAN do 2D QS because W < 1.

### 2. Square-cubed law contributes but is secondary (Exp136)

- Interior fraction correlates r=0.53 with ⟨r⟩ (moderate)
- But 125 cells in 3D (L=5) beat 900 cells in 2D (L=30)
- Dominant effect: random walk recurrence (Pólya 1921) — topological, not geometric
- 2D NEVER active at W=13 regardless of size; 3D active at L=4 (64 cells)

### 3. Free plankton is QS-suppressed; attachment is required (Exp137)

- QS breaks at ≤75% occupancy (W_eff ≈ 17.3 exceeds W_c)
- Sea plankton at 10⁶/mL has ~0.1% occupancy → W_eff >> W_c → suppressed
- Matches marine biology: QS prevalence scales with surface attachment
- Biofilm temporal stages: early colonization (J=0.2) → 2D works; climax (J=0.95) → needs 3D

### 4. QS is universal across domains of life (Exp138)

- Bacteria (L=10), yeast (L=8), protists (L=7), tissue cells (L=5) — all
  QS-active at typical diversity in 3D
- Minimum colony: 64 cells (L=4)
- Tissue cells operate in different regime: low diversity (W<3), not geometry
- Prediction: QS scales with cell_density × colony_volume, not taxonomy

## ToadStool Primitives Consumed

| Primitive | Origin | Used In |
|-----------|--------|---------|
| `anderson_3d` | hotSpring | Exp135-138 |
| `anderson_2d` | hotSpring | Exp135-136 |
| `anderson_hamiltonian` | hotSpring | Exp135 |
| `lanczos` + `lanczos_eigenvalues` | hotSpring | Exp135-138 |
| `find_all_eigenvalues` | hotSpring | Exp135-136 |
| `level_spacing_ratio` | hotSpring | Exp135-138 |

## Novel Contributions for hotSpring

1. **Biological validation of topological argument** — Pólya recurrence theorem
   has measurable ecological consequences (QS in d=3, not d=2)
2. **Dilution model** — W_eff = W_base/occupancy maps sparse suspensions to Anderson
3. **Cross-domain size scaling** — cell diameter → effective L → QS prediction
4. **Temporal turnover mapping** — J(t) trajectory → W(t) → QS regime transitions

## Artifact Inventory

| File | Type |
|------|------|
| `barracuda/src/bin/validate_mapping_sensitivity.rs` | Exp135 binary |
| `barracuda/src/bin/validate_square_cubed_scaling.rs` | Exp136 binary |
| `barracuda/src/bin/validate_planktonic_dilution.rs` | Exp137 binary |
| `barracuda/src/bin/validate_eukaryote_scaling.rs` | Exp138 binary |
| `experiments/135_mapping_sensitivity.md` | Exp135 doc |
| `experiments/136_square_cubed_scaling.md` | Exp136 doc |
| `experiments/137_planktonic_dilution.md` | Exp137 doc |
| `experiments/138_eukaryote_scaling.md` | Exp138 doc |

## Reproduction

```bash
cd barracuda
cargo run --release --features gpu --bin validate_mapping_sensitivity
cargo run --release --features gpu --bin validate_square_cubed_scaling
cargo run --release --features gpu --bin validate_planktonic_dilution
cargo run --release --features gpu --bin validate_eukaryote_scaling
```
