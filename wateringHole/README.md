# wetSpring wateringHole

**Date:** February 27, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V65** | `handoffs/WETSPRING_TOADSTOOL_V65_PROGRESSION_BENCHMARK_HANDOFF_FEB27_2026.md` | Feb 27 | **current** — V65 progression benchmark (Python → CPU → GPU → streaming → metalForge), Exp210 cross-spring evolution (24/24), Exp211 progression benchmark (16/16), submit_and_poll migration, Fp64Strategy/optimal_precision wiring, 211 experiments 5,061+ checks ALL PASS |

## NestGate Handoffs

| File | Date | Scope |
|------|------|-------|
| `handoffs/WETSPRING_NESTGATE_DATA_TYPES_EVOLUTION_HANDOFF_FEB26_2026.md` | Feb 26 | Data type profiling, P0-P3 gap analysis, evolution roadmap for biology-aware storage |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 700+ WGSL shader provenance map (cross-spring, ToadStool S68 universal precision) |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V64, API report, rewire, cross-spring provenance — 54 files).
V34-V39 archived (superseded by V40). V40-V43 archived (superseded by V47). V44-V45 archived (superseded by V48). V47-V50 archived (superseded by V52). V52 archived (superseded by V53+V54). V53-V55 archived (superseded by V56-V59). V56-V58 archived (superseded by V59+V60). V59-V64 archived (superseded by V65).
Cross-spring docs moved to archive: `CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md`, `CROSS_SPRING_PROVENANCE_FEB22_2026.md`.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
