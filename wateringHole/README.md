# wetSpring wateringHole

**Date:** March 15, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V114** | `handoffs/WETSPRING_V114_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR15_2026.md` | Mar 15 | **BarraCUDA absorption** — 8 GPU primitive opportunities mapped (NMF, adaptive ODE, batched SW, taxonomy, phylogenetics). |
| **V114** | `handoffs/WETSPRING_V114_NICHE_SETUP_GUIDANCE_HANDOFF_MAR15_2026.md` | Mar 15 | **Niche setup guide** — 7-step checklist for springs modeling the wetSpring niche pattern. |
| **V114** | `handoffs/WETSPRING_V114_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR12_2026.md` | Mar 12 | Deep audit handoff — features gate fixes, clippy, deprecated parsers, inline tolerances. |
| **V113** | `handoffs/WETSPRING_V113_PROVENANCE_TRIO_CAPABILITIES_DEPLOY_HANDOFF_MAR15_2026.md` | Mar 15 | Provenance trio + deploy graph — rhizoCrypt/loamSpine/sweetGrass, 19 capabilities, time series, NMF. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V112 (95+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
