# wetSpring wateringHole

**Date:** April 17, 2026  
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V144** | `handoffs/WETSPRING_V144_COMPOSITION_VALIDATION_TIER_HANDOFF_APR17_2026.md` | Apr 17 | **Composition validation tier**: Python→Rust→Primal composition. Exp400 (136/136 proto-nucleate), Exp401 (43/43 IPC parity), Exp402 (63/63 niche gate). 18 IPC roundtrip tests. Ed25519→BLAKE3 keyed MAC. `metrics.snapshot` handler. biomeOS V144 composition health ownership. barraCuda v0.3.12. Patterns for primal + spring teams. |
| | *Superseded → `handoffs/archive/`* | | V143 and earlier are archived (**155** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (826+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — **V143 and earlier** are archived (**155** files).  
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:  
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).  
No reverse dependencies.
