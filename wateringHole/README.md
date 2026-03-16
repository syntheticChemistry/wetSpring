# wetSpring wateringHole

**Date:** March 16, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V121** | `handoffs/WETSPRING_V121_DEEP_DEBT_EVOLUTION_HANDOFF_MAR16_2026.md` | Mar 16 | **Deep Debt Evolution + Absorption** — 214 named tolerances (zero inline literals), all primal names via `primal_names::*`, crate-level `#[allow()]` → `#[expect(reason)]`, blake3 pure Rust, tempfile test paths, baseline verification fix. Full toadStool/barraCuda absorption handoff with patterns and evolution requests. |
| | *V119–V120 → `handoffs/archive/`* | | V119–V120 handoffs archived. |
| | *V113–V118 → `handoffs/archive/`* | | V113–V118 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs). 129 total archived. |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V120 (129 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
