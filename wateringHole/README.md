# wetSpring wateringHole

**Date:** February 25, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V33** | `handoffs/WETSPRING_TOADSTOOL_V33_CPUMATH_LEAN_FEB25_2026.md` | Feb 25 | Phase 41: barracuda always-on, ~177 lines dual-path removed, zero fallback code |
| V32 | `handoffs/WETSPRING_TOADSTOOL_V32_S62_LEAN_FEB24_2026.md` | Feb 24 | Phase 41: S62 lean — PeakDetectF64, TranseScoreF64 wired, paper queue fully GPU-covered |
| V31 | `handoffs/WETSPRING_TOADSTOOL_V31_ABSORPTION_TARGETS_FEB24_2026.md` | Feb 24 | Phase 41: absorption targets, cross-spring evolution insights |
| V30 | `handoffs/WETSPRING_TOADSTOOL_V30_S59_LEAN_FEB24_2026.md` | Feb 24 | Phase 41: S59 lean — NMF, ridge, ODE systems, correlated Anderson (~1,312 lines removed) |
| API | `handoffs/TOADSTOOL_PRIMITIVES_API_REPORT_FEB24_2026.md` | Feb 24 | ToadStool primitives API reference |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 660+ WGSL shader provenance map (cross-spring, ToadStool S62) |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V29, rewire, cross-spring provenance).
Cross-spring docs moved to archive: `CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md`, `CROSS_SPRING_PROVENANCE_FEB22_2026.md`.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
