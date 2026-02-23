# wetSpring → ToadStool Handoff V20: 3D Anderson Dimensional QS Phase Diagram

**Date:** February 23, 2026
**Phase:** 36
**Experiments:** 127–130 (50 GPU-confirmed checks, all PASS)
**Total:** 130 experiments, 2,869+ validation checks
**Predecessor:** V19 (NCBI-scale hypothesis testing, Phase 35)

---

## Summary

Phase 36 extends Phase 35's 2D Anderson QS findings into 3D using spectral
primitives already absorbed from hotSpring into ToadStool (`anderson_3d`,
`lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`). Four new experiments
build the complete dimensional phase diagram for microbial quorum sensing.

---

## Key Findings

### 1. 3D Anderson Plateau is 2.4× Wider Than 2D (Exp127)

| Dimension | Plateau (W>2) | J_c   | Ecosystem Gain |
|-----------|:------------:|:-----:|----------------|
| 1D        | 0            | —     | — |
| 2D        | 5            | 0.557 | bloom gains QS |
| 3D        | 12           | 1.283 | gut, vent, soil, ocean gain QS |

The 3D metal-insulator transition at W_c ≈ 16.5 (orthogonal class) exceeds
all natural biome disorder values (max W ≈ 14.85 for soil).

### 2. Vent Chimney 3D Structure Is Essential for QS (Exp128)

3 of 4 chimney zones are QS-active in 3D but suppressed in 2D:
- Young sulfide: W=7.29, 2D suppressed → 3D active
- Mature anhydrite: W=16.25, 2D suppressed → 3D active
- Weathered exterior: W=13.49, 2D suppressed → 3D active

A 2D surface model misses 75% of chimney QS capability.

### 3. All 28 Biomes QS-Active in 3D (Exp129)

The complete (J × dimension) phase diagram shows:
- 1D: 0/28 biomes QS-active
- 2D: 0/28 biomes QS-active
- 3D: 28/28 biomes QS-active

Dimensionality, not diversity, is the decisive factor.

### 4. Six Layers Transform Biofilm QS (Exp130)

| Geometry   | Plateau (W>2) | J_c   |
|-----------|:------------:|:-----:|
| 2D slab (20×20)  | 4    | 0.406 |
| 3D block (8×8×6) | 16   | 1.248 |

J_c(3D) / J_c(2D) = 3.1 — just 6 layers of depth triple the diversity tolerance.

---

## ToadStool Primitives Consumed

All from `barracuda::spectral` (absorbed from hotSpring):

| Primitive | Usage | Exp |
|-----------|-------|-----|
| `anderson_3d(lx, ly, lz, W, seed)` | 3D cubic lattice Hamiltonian | 127-130 |
| `anderson_2d(lx, ly, W, seed)` | 2D lattice for comparison | 127-130 |
| `anderson_hamiltonian(n, W, seed)` | 1D chain for baseline | 127, 129 |
| `lanczos(matrix, max_iter, seed)` | Sparse eigensolver | 127-130 |
| `lanczos_eigenvalues(tridiag)` | Eigenvalue extraction | 127-130 |
| `level_spacing_ratio(eigenvalues)` | ⟨r⟩ diagnostic | 127-130 |
| `find_all_eigenvalues(diag, off)` | Dense 1D eigensolver | 127, 129 |
| `GOE_R`, `POISSON_R` | Reference constants | 127-130 |

---

## Novel Contribution to hotSpring

Phase 36 provides the first **biological validation data** for the 3D Anderson
metal-insulator transition. The spectral primitives developed in hotSpring for
nuclear/plasma physics (Papers 14-22) now have a concrete biological application:

1. **Level spacing ratio ⟨r⟩** distinguishes QS-active (GOE-like) from
   QS-suppressed (Poisson-like) communities
2. **The dimensional phase transition** predicts which microbial communities
   can sustain collective behavior based on spatial geometry
3. **Vent chimney mapping** connects Anderson lattice parameters to measurable
   physical properties (porosity, mineral heterogeneity)

This creates a feedback loop: hotSpring develops spectral theory → ToadStool
absorbs primitives → wetSpring applies to biology → biological results
validate and motivate further spectral theory development.

---

## Absorption Recommendations

### For ToadStool Team

1. **No new primitives needed** — Phase 36 consumes existing `barracuda::spectral`
2. **Finite-size study** — L=8 cubic lattice shows finite-size effects at W ≤ 0.5.
   If hotSpring adds finite-size scaling routines, wetSpring would consume them
3. **Anderson transport** — if `barracuda::spectral` gains conductance/transport
   primitives, wetSpring could compute QS signal propagation rates, not just
   active/suppressed classification

### For wetSpring Next Phase

1. **Finite-size scaling** — run L=6,8,10,12 cubes to extract thermodynamic W_c
2. **Disorder-correlated lattices** — real biofilms have spatially correlated
   heterogeneity, not i.i.d. disorder
3. **NCBI chimney data** — fetch real vent microbial community data to validate
   Exp128 predictions against measured QS phenotypes
4. **ESN 3D classifier** — train Exp123's ESN to predict QS regime from (J, dim)

---

## Artifact Inventory

| Artifact | Path | Status |
|----------|------|--------|
| Phase 36 design | `experiments/PHASE_36_DESIGN.md` | Complete |
| Exp127 binary | `barracuda/src/bin/validate_anderson_3d_qs.rs` | PASS 17/17 |
| Exp128 binary | `barracuda/src/bin/validate_vent_chimney_qs.rs` | PASS 12/12 |
| Exp129 binary | `barracuda/src/bin/validate_dimensional_phase_diagram.rs` | PASS 12/12 |
| Exp130 binary | `barracuda/src/bin/validate_biofilm_3d_qs.rs` | PASS 9/9 |
| Exp127 doc | `experiments/127_anderson_3d_qs.md` | Complete |
| Exp128 doc | `experiments/128_vent_chimney_qs.md` | Complete |
| Exp129 doc | `experiments/129_dimensional_phase_diagram.md` | Complete |
| Exp130 doc | `experiments/130_biofilm_3d_qs.md` | Complete |
| Kachkovskiy briefing | `whitePaper/baseCamp/kachkovskiy.md` | Updated |

---

## Reproduction

```bash
cd barracuda

# All four Phase 36 experiments
cargo run --release --features gpu --bin validate_anderson_3d_qs
cargo run --release --features gpu --bin validate_vent_chimney_qs
cargo run --release --features gpu --bin validate_dimensional_phase_diagram
cargo run --release --features gpu --bin validate_biofilm_3d_qs
```
