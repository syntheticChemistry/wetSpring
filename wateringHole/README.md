# wetSpring wateringHole

**Date:** February 23, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V18** | `handoffs/WETSPRING_V018_CROSS_SPRING_REWIRE_HANDOFF_FEB23_2026.md` | Feb 23 | Phase 34: full rewire, cross-spring benchmark, 120 experiments |
| V17 | `handoffs/WETSPRING_TOADSTOOL_V17_NPU_RESERVOIR_FEB23_2026.md` | Feb 23 | Phase 33: NPU reservoir, NCBI-scale GPU, PCoA fix |
| V16 | `handoffs/WETSPRING_TOADSTOOL_V16_STREAMING_FEB23_2026.md` | Feb 23 | Phase 30: streaming v2, metalForge v6 |
| V15 | `handoffs/WETSPRING_TOADSTOOL_V15_ODE_GENERIC_FEB22_2026.md` | Feb 22 | 5 ODE shaders → BatchedOdeRK4Generic |
| V14 | `handoffs/WETSPRING_TOADSTOOL_V14_FEB22_2026.md` | Feb 22 | Write phase, ODE shaders, forge v0.3.0 |
| V13 | `handoffs/WETSPRING_TOADSTOOL_V13_FEB22_2026.md` | Feb 22 | Edition 2024, structural evolution |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 612 WGSL shader provenance map (35 hot, 22 wet, 14 neural, 5 air) |
| `handoffs/CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md` | wetSpring perspective on biome model |
| `handoffs/CROSS_SPRING_PROVENANCE_FEB22_2026.md` | Shader origin tracking |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V12, rewire).

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
