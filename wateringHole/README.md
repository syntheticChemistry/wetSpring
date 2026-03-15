# wetSpring wateringHole

**Date:** March 15, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V116** | `handoffs/WETSPRING_V116_CAPABILITY_DISCOVERY_TOLERANCE_CENTRALIZATION_HANDOFF_MAR15_2026.md` | Mar 15 | **Capability discovery + tolerance centralization** — `capability.list` handler, 14 domains / 19 methods, inline tolerance centralization (15 binaries), capability-based primal discovery (3 binaries), forge lint parity. 8 GPU primitive opportunities carried forward. |
| | *V113–V115 → `handoffs/archive/`* | | V113–V115 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V115 (123+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
