# wetSpring wateringHole

**Date:** March 16, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V122** | `handoffs/WETSPRING_V122_MODERN_IDIOMATIC_RUST_HANDOFF_MAR16_2026.md` | Mar 16 | **Modern Idiomatic Rust Evolution + Absorption** — all `#[allow()]` → `#[expect(reason)]` across 276+ validation binaries (298 files), 1,139 unfulfilled expectations cleaned, 18 new forge tests, idiomatic Rust fixes, unsafe eliminated from tests. Zero `#[allow()]` in entire codebase. Full toadStool/barraCuda absorption handoff with migration guide and patterns. |
| | *V119–V121 → `handoffs/archive/`* | | V119–V121 handoffs archived. |
| | *V113–V118 → `handoffs/archive/`* | | V113–V118 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs). 130 total archived. |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V121 (130 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
