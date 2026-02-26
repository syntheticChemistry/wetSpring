# wetSpring wateringHole

**Date:** February 26, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V53** | `handoffs/WETSPRING_TOADSTOOL_V53_CROSS_SPRING_EVOLUTION_HANDOFF_FEB26_2026.md` | Feb 26 | **current** — cross-spring evolution benchmarks, GPU performance data, tolerance learnings, absorption candidates for ToadStool |
| **V52** | `handoffs/WETSPRING_TOADSTOOL_V52_S66_REWIRE_HANDOFF_FEB26_2026.md` | Feb 26 | S66 rewire (`hill`, `fit_linear`, `mean`/`percentile`), V51 GPU validation (1,578 checks on RTX 4070), 79 primitives consumed |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 694+ WGSL shader provenance map (cross-spring, ToadStool S66) |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V50, API report, rewire, cross-spring provenance — 46 files).
V34-V39 archived (superseded by V40). V40-V43 archived (superseded by V47). V44-V45 archived (superseded by V48). V47-V50 archived (superseded by V52).
Cross-spring docs moved to archive: `CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md`, `CROSS_SPRING_PROVENANCE_FEB22_2026.md`.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
