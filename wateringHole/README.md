# wetSpring wateringHole

**Date:** March 15, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V113** | `handoffs/WETSPRING_V113_PROVENANCE_TRIO_CAPABILITIES_DEPLOY_HANDOFF_MAR15_2026.md` | Mar 15 | **Provenance trio + deploy graph** — rhizoCrypt/loamSpine/sweetGrass integration, 19 biomeOS capabilities (was 9), cross-spring time series exchange, biomeOS deploy graph, NMF. |
| **V112** | `handoffs/WETSPRING_V112_STREAMING_PEDANTIC_CAPABILITY_HANDOFF_MAR14_2026.md` | Mar 14 | Streaming-only I/O + capability discovery — deprecated buffering parsers removed, zero clippy warnings. |
| **V114** | `handoffs/WETSPRING_V114_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR12_2026.md` | Mar 12 | Deep audit handoff — features gate fixes, clippy, deprecated parsers, inline tolerances. |
| **V113** | `handoffs/WETSPRING_V113_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR11_2026.md` | Mar 11 | Upstream evolution — 150+ primitives, P0/P1/P2 absorption targets, hardware learning. |
| **V113** | `handoffs/WETSPRING_V113_PAPER_EXTENSION_ROADMAP_HANDOFF_MAR11_2026.md` | Mar 11 | Paper extension roadmap (Exp364-370, 67/67). |
| **V112** | `handoffs/WETSPRING_V112_NVIDIA_HARDWARE_LEARNING_HANDOFF_MAR11_2026.md` | Mar 11 | NVIDIA hardware learning prototype (Exp361-363). |
| **V111** | `handoffs/WETSPRING_V111_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR14_2026.md` | Mar 14 | barraCuda/toadStool absorption from V111 deep debt. |
| **V111** | `handoffs/WETSPRING_V111_DEEP_DEBT_EVOLUTION_HANDOFF_MAR14_2026.md` | Mar 14 | V111 deep debt evolution handoff. |
| | *V109 and earlier → `handoffs/archive/`* | | Fossil record: V7–V109 (89+ archived handoffs) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V111+ (89+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
