# wetSpring wateringHole

**Date:** March 15, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V120** | `handoffs/WETSPRING_V120_CROSS_SPRING_ABSORPTION_HANDOFF_MAR15_2026.md` | Mar 15 | **Cross-Spring Absorption** — typed errors completed, deploy graph hardened (`fallback = "skip"`), `live_pipeline.rs` refactored, shared Python tolerance module (120+ constants). |
| **V120** | `handoffs/WETSPRING_V120_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_MAR15_2026.md` | Mar 15 | **toadStool/barraCuda Team** — 44 GPU modules, 150+ primitives consumed, typed error patterns, tolerance centralization, deploy graph integration, ESN evolution, cross-spring learnings. |
| | *V119 → `handoffs/archive/`* | | V119 Deep Debt Evolution Sprint handoff archived. |
| | *V113–V118 → `handoffs/archive/`* | | V113–V118 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs). 127 total archived. |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V118 (126+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
