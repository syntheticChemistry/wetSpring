# Phase 36: 3D Anderson Lattice QS — Dimensional Phase Diagram
> *Fossil record — design completed; all experiments implemented and passing.*

**Date:** February 23, 2026
**Phase:** 36
**Predecessor:** Phase 35 (NCBI-scale hypothesis testing, Exp121-126)
**Experiments:** Exp127–130
**Primitives:** `barracuda::spectral` (anderson_3d, anderson_2d, anderson_hamiltonian, lanczos, level_spacing_ratio — all absorbed from hotSpring)

---

## Motivation

Phase 35 revealed two key findings:

1. **Exp122**: The 2D Anderson lattice has a genuine extended plateau (QS-active
   window) absent in 1D. Bloom communities are QS-active in 2D but suppressed
   in 1D. Critical J_c ≈ 0.41.

2. **Exp107 Section 5**: A single 3D point (L=6, W=2) gave ⟨r⟩ = 0.4843 — the
   most GOE-like (extended/metallic) value in the entire dataset. At W=25,
   ⟨r⟩ = 0.4171 (localized).

**Physics prediction:** In 3D, Anderson theory predicts a genuine
metal-insulator transition at a critical disorder W_c ≈ 16.5 (orthogonal
symmetry class, cubic lattice, t=1). This means:

- 1D: all states localize for any W > 0 (Anderson's theorem)
- 2D: marginal — logarithmic corrections, weak localization
- 3D: genuine phase transition at W_c — extended states survive below W_c

For QS biology, this predicts that 3D structures (thick biofilms,
hydrothermal vent chimney interiors, soil pore networks) can sustain
community-wide signaling at diversity levels where 1D and 2D fail.

**hotSpring connection:** All spectral primitives (`anderson_3d`, `lanczos`,
`level_spacing_ratio`) were developed in hotSpring (Papers 14-22, 41/41
checks), absorbed into ToadStool `barracuda::spectral`, and exercised in
wetSpring since Exp107. Phase 36 leverages this cross-spring infrastructure
to produce novel ecological predictions that may feed back as biological
validation data for hotSpring's spectral theory.

---

## Experiment Summary

| Exp | Extends | Hypothesis | Hardware | Checks |
|-----|---------|-----------|----------|--------|
| 127 | Exp122 | 3D Anderson has wider QS-active window than 2D; W_c(3D) > W_c(2D) | GPU | ~18 |
| 128 | Exp107 S5 | Vent chimney porosity maps to QS propagation depth via 3D Anderson | GPU | ~14 |
| 129 | Exp122/126 | Complete dimensional phase diagram: J × dim → QS regime for 28 biomes | GPU | ~16 |
| 130 | Exp122 | 3D biofilm block sustains QS at higher diversity than 2D slab | GPU | ~12 |

---

## Exp127: 3D Anderson Dimensional Sweep

### Background
Exp122 compared 1D vs 2D with 20-point sweeps and found the 2D extended
plateau. Now we add the third dimension to complete the picture.

### Hypothesis
**H127:** The 3D cubic lattice maintains QS-active (GOE-like) ⟨r⟩ at
disorder values where 2D has already localized. The 3D transition point
W_c(3D) exceeds W_c(2D).

### Design
- **1D chain**: N=400 sites, `anderson_hamiltonian` + `find_all_eigenvalues`
- **2D lattice**: 20×20=400 sites, `anderson_2d` + `lanczos`
- **3D lattice**: 8×8×8=512 sites, `anderson_3d` + `lanczos`
- **Disorder sweep**: 20 points, W = 0.5 to 25.0 (extended range for 3D)
- **Diagnostics**: ⟨r⟩ at each point, plateau width, transition crossing

### Checks (~18)
- S1: 1D sweep (4) — count, endpoints, monotonicity
- S2: 2D sweep (4) — count, endpoints, monotonicity
- S3: 3D sweep (4) — count, endpoints, extended plateau
- S4: Dimensional comparison (3) — 3D plateau wider than 2D
- S5: Ecosystem mapping (3) — 6 ecosystems across 3 dimensions

---

## Exp128: Vent Chimney Geometry

### Background
Hydrothermal vent chimneys (Anderson's domain) have measurable porosity
(5-40%), mineral heterogeneity, and temperature gradients. These map
naturally to Anderson lattice parameters.

### Hypothesis
**H128:** Vent chimney porosity and mineral heterogeneity determine QS
propagation depth. High-porosity chimney sections (>20%) support QS;
dense mineral zones suppress it. The 3D Anderson model predicts which
chimney zones host collective microbial behavior.

### Design
- **Chimney profiles**: 4 chimney types mapping real vent parameters
  - Young sulfide (high porosity 30%, moderate disorder)
  - Mature anhydrite (low porosity 8%, high disorder)
  - Silica-lined conduit (medium porosity 15%, low disorder)
  - Weathered exterior (high porosity 35%, high disorder)
- **3D lattice**: 8×8×8 with disorder derived from mineral heterogeneity
- **Porosity mapping**: fraction of active sites in lattice
- **Prediction**: QS regime (active/suppressed) for each chimney zone

### Checks (~14)
- S1: Chimney parameter derivation (4) — valid ranges
- S2: 3D spectral analysis per chimney (4) — ⟨r⟩ computed
- S3: QS regime prediction (3) — high-porosity active, low-porosity suppressed
- S4: Comparison to 2D slab model (3) — 3D captures depth that 2D misses

---

## Exp129: Dimensional QS Phase Diagram

### Background
Exp126 mapped 28 biomes to 1D Anderson disorder. Exp122 showed 2D changes
the regime for some biomes. This experiment builds the complete
dim × J phase diagram.

### Hypothesis
**H129:** The three dimensions partition biomes into distinct QS regimes:
biomes QS-suppressed in 1D may be QS-active in 2D, and biomes suppressed
in 2D may be active in 3D. The dimensional phase diagram predicts which
biomes require spatial structure for collective behavior.

### Design
- **28 biomes** from Exp126 biome_diversity_params
- **3 dimensions**: 1D (N=400), 2D (20×20), 3D (8×8×8)
- **Pre-compute** disorder sweep for each dimension (reuse from Exp127)
- **Classify** each biome × dimension combination as QS-active or suppressed
- **Phase diagram**: biome_J × dimension → regime

### Checks (~16)
- S1: Phase diagram computed (3) — all biomes × 3 dims
- S2: Monotonicity (3) — QS-active count increases with dimension
- S3: 1D baseline (3) — matches Exp126 atlas
- S4: Known biomes (4) — biofilm active in 3D, soil suppressed in all
- S5: Dimensional gain (3) — number of biomes that flip from suppressed to active

---

## Exp130: Thick Biofilm 3D Extension

### Background
Exp122 showed biofilm (J=0.025, W=0.87) is QS-active in both 1D and 2D.
But real biofilms are 3D structures — multi-layer communities with thickness
10-100 µm. Does the 3D geometry extend the QS-active window to higher
diversity levels where 2D biofilm models would predict suppression?

### Hypothesis
**H130:** A 3D biofilm block sustains QS-active signaling at higher Pielou J
(greater diversity) than a 2D slab of the same total area. The critical
evenness J_c(3D) > J_c(2D) ≈ 0.41.

### Design
- **2D slab**: 20×20 = 400 sites (Exp122 geometry)
- **3D block**: 8×8×6 = 384 sites (comparable total, 6-layer depth)
- **Disorder sweep**: 20 points, W = 0.5 to 20.0
- **J_c extraction**: find transition point in each geometry
- **Biofilm-specific ecosystems**: test at J = 0.3, 0.4, 0.5, 0.6

### Checks (~12)
- S1: 2D slab sweep (3) — baseline from Exp122 geometry
- S2: 3D block sweep (3) — extended plateau measurement
- S3: J_c comparison (3) — J_c(3D) > J_c(2D)
- S4: Biofilm diversity scan (3) — test 4 diversity levels

---

## Implementation Plan

### New Binaries

| Binary | Path | Features |
|--------|------|----------|
| `validate_anderson_3d_qs` | `src/bin/validate_anderson_3d_qs.rs` | `gpu` |
| `validate_vent_chimney_qs` | `src/bin/validate_vent_chimney_qs.rs` | `gpu` |
| `validate_dimensional_phase_diagram` | `src/bin/validate_dimensional_phase_diagram.rs` | `gpu` |
| `validate_biofilm_3d_qs` | `src/bin/validate_biofilm_3d_qs.rs` | `gpu` |

### Dependencies

All from `barracuda::spectral` (already absorbed from hotSpring):
- `anderson_hamiltonian` — 1D
- `anderson_2d` — 2D
- `anderson_3d` — 3D
- `lanczos` / `lanczos_eigenvalues` — sparse eigensolver
- `find_all_eigenvalues` — dense eigensolver (1D)
- `level_spacing_ratio` — ⟨r⟩ diagnostic
- `GOE_R` / `POISSON_R` — reference constants

From `wetspring_barracuda`:
- `bio::diversity` — shannon, pielou_evenness
- `validation::Validator` — standard check framework

### Fallback Strategy

All binaries use `#[cfg(feature = "gpu")]` / `#[cfg(not(feature = "gpu"))]`
blocks. CPU fallback prints skip messages and runs community generation
sanity checks only. Full spectral analysis requires `--features gpu`.

---

## Success Criteria

1. All 4 binaries compile with 0 warnings (`cargo check --features gpu`)
2. All checks pass on GPU (`cargo run --release --features gpu --bin <name>`)
3. 3D plateau demonstrably wider than 2D (measured in disorder units)
4. Vent chimney QS predictions are physically reasonable
5. Phase diagram correctly partitions biomes by dimension
6. J_c(3D) > J_c(2D) > J_c(1D) — dimensional hierarchy confirmed
