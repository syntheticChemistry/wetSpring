# Exp134: Cross-Ecosystem QS Capability Atlas

**Status:** PASS (GPU) — 11/11 checks
**Binary:** `validate_cross_ecosystem_atlas`
**Features:** `gpu`
**Extends:** Exp129, Exp132

## Hypothesis

H134: 28 biomes × 5 geometries produces the definitive QS capability atlas.
Geometry effectiveness varies: block activates all biomes; chain, slab, tube
activate none; thin film activates only lowest-diversity biomes.

## Design

- 28 biomes from NCBI atlas (Exp129)
- 5 geometries: chain, slab, film, tube, block
- Classify each biome × geometry as QS-active or QS-suppressed
- Tabulate geometry effectiveness across biomes

## Key Results (GPU confirmed)

| Geometry | QS-active biomes | QS-suppressed |
|----------|------------------|---------------|
| chain    | 0/28             | 28/28         |
| slab     | 0/28             | 28/28         |
| film     | 3/28             | 25/28         |
| tube     | 0/28             | 28/28         |
| block    | 28/28            | 0/28          |

**3 biomes active in thin film:** deep_sea_hadal, biofilm_hospital, algal_bloom_taihu (lowest diversity)

## Key Findings

1. **3D block geometry is the ONLY shape that activates all 28 biomes**
2. **Thin film activates only the 3 lowest-diversity biomes**
3. **Chain, slab, tube activate ZERO biomes** — 1D and 2D geometries insufficient
4. **Real-world QS requires true 3D spatial structure** — surfaces, tubes, and thin films are insufficient for most microbial communities
5. **Definitive atlas:** 28 × 5 = 140 biome-geometry pairs, block-only universal QS
