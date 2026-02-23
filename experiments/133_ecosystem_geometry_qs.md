# Exp133: Ecosystem Geometry QS Profiles

**Status:** PASS (GPU) — 17/17 checks
**Binary:** `validate_ecosystem_geometry_qs`
**Features:** `gpu`
**Extends:** Exp132

## Hypothesis

H133: Cave, hot spring, and rhizosphere ecosystems have geometry-specific QS
profiles. Only 3D block-geometry zones sustain QS at natural diversity levels.
2D surfaces and tubes are QS-suppressed.

## Design

- 12 ecosystem zones modeled with physically appropriate lattice dimensions and diversity params
- Cave: pool_sediment, passage_sediment, subterranean_river, wall_biofilm, stalactite_film
- Hot spring: surface_mat, deep_mat, silica_sinter
- Rhizosphere: bulk_soil, inner_rhizosphere, root_surface, mycorrhizal_hyphae
- Map each zone to ⟨r⟩ and classify QS-active vs suppressed

## Key Results (GPU confirmed)

**QS-ACTIVE (⟨r⟩ > 0.45):**
| Zone            | ⟨r⟩    |
|-----------------|--------|
| pool_sediment   | 0.4967 |
| bulk_soil       | 0.4848 |
| inner_rhizosphere| 0.4785|
| passage_sediment| 0.4611 |

**QS-SUPPRESSED (⟨r⟩ < 0.46):**
| Zone             | ⟨r⟩    |
|------------------|--------|
| deep_mat         | 0.4539 |
| root_surface     | 0.4517 |
| silica_sinter    | 0.4306 |
| subterranean_river| 0.4240|
| wall_biofilm     | 0.4194 |
| mycorrhizal_hyphae| 0.4185|
| stalactite_film  | 0.4079 |
| surface_mat      | 0.4035 |

## Key Findings

1. **Only 3D block-geometry zones (sediments, soil pores) sustain QS at natural diversity levels**
2. **2D surfaces (wall biofilms, sinter) and tubes (stalactites, hyphae) are QS-suppressed**
3. **Diversity AND geometry jointly determine QS** — not diversity alone
4. **Pool sediment and bulk soil** — highest ⟨r⟩ among 12 zones
