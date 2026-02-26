# wetSpring wateringHole

**Date:** February 26, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V58** | `handoffs/WETSPRING_TOADSTOOL_V58_EVOLUTION_LEARNINGS_HANDOFF_FEB26_2026.md` | Feb 26 | **current** — Evolution learnings, cross-spring patterns, absorption candidates, DF64 bio opportunity, feature-gate audit, benchmark data |
| **V57** | `handoffs/WETSPRING_TOADSTOOL_V57_S68_CATCHUP_HANDOFF_FEB26_2026.md` | Feb 26 | ToadStool S66→S68 catch-up (19 commits), CPU feature-gate fix, universal precision alignment |
| **V56** | `handoffs/WETSPRING_TOADSTOOL_V56_SCIENCE_PIPELINE_HANDOFF_FEB26_2026.md` | Feb 26 | Science extension pipeline, NCBI EFetch/SRA, NestGate JSON-RPC, GPU Anderson L=14-20, biomeOS science graph |
| **V55** | `handoffs/WETSPRING_TOADSTOOL_V55_DEEP_DEBT_HANDOFF_FEB26_2026.md` | Feb 26 | Deep debt resolution, idiomatic Rust evolution, 79-primitive usage audit |
| **V54** | `handoffs/WETSPRING_TOADSTOOL_V54_CODEBASE_AUDIT_HANDOFF_FEB26_2026.md` | Feb 26 | Codebase audit, supply-chain audit, tolerance provenance hardening |
| **V53** | `handoffs/WETSPRING_TOADSTOOL_V53_CROSS_SPRING_EVOLUTION_HANDOFF_FEB26_2026.md` | Feb 26 | Cross-spring evolution benchmarks, GPU performance data, tolerance learnings |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 700+ WGSL shader provenance map (cross-spring, ToadStool S68 universal precision) |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V52, API report, rewire, cross-spring provenance — 49 files).
V34-V39 archived (superseded by V40). V40-V43 archived (superseded by V47). V44-V45 archived (superseded by V48). V47-V50 archived (superseded by V52). V52 archived (superseded by V53+V54).
Cross-spring docs moved to archive: `CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md`, `CROSS_SPRING_PROVENANCE_FEB22_2026.md`.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
