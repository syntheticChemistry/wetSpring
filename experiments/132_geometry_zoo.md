# Exp132: Geometry Zoo — Lattice Shape vs QS Capability

**Status:** PASS (GPU) — 11/11 checks
**Binary:** `validate_geometry_zoo`
**Features:** `gpu`
**Extends:** Exp127, Exp130

## Hypothesis

H132: Lattice shape determines QS capability at matched site counts. Block
geometry outperforms slab, film, tube, and chain. Ecological habitats map
to specific geometries.

## Design

- Six geometries at matched site counts (~384–400): chain, slab 20×20, thin_film 14×14×2, tube, block, cube
- Disorder sweep per geometry
- Extract plateau width and J_c
- Rank geometries by QS capability

## Key Results (GPU confirmed)

| Geometry      | Sites | Plateau | J_c  |
|---------------|-------|---------|------|
| chain         | 384   | 0       | —    |
| slab_20x20    | 400   | 5       | 7.10 |
| thin_film     | 392   | 7       | 12.15|
| tube          | 384   | 5       | 8.28 |
| block         | 384   | 12      | 18.47|
| cube          | 392   | 11      | 16.15|

**Ranking:** block > cube > thin_film > slab ≈ tube > chain

## Key Findings

1. **3D block is 2.4× better than any 2D geometry** — J_c(block)=18.47 vs J_c(slab)=7.10
2. **Just 2 layers of depth (thin_film 14×14×2) adds 40% more plateau than pure 2D slab** — 7 vs 5 points
3. **Tube (cave/gut) beats 1D but barely beats 2D** — intermediate between slab and thin_film
4. **Chain has zero QS capability** — Anderson's theorem in 1D

## Ecological Mapping

| Habitat           | Geometry   | QS Regime |
|-------------------|------------|-----------|
| cave sediment     | block      | QS-CAPABLE|
| hot spring mat    | thin_film  | QS-CAPABLE|
| gut lumen         | tube       | QS-CAPABLE|
| soil              | cube       | QS-CAPABLE|
