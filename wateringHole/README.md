# wetSpring wateringHole

**Date:** August 3, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V210+ — Wave 156 Deep Debt Evolution)

| Metric | Value |
|--------|-------|
| wetSpring | V210+ (346 scenarios, 54 niche, 59 consumed, 47 dispatch, 22 domains, 2,201 tests, clippy zero) |
| Gate | **westGate** (primary dev + Data NAS); southGate (NUCLEUS); strandGate (GPU science) |
| NUCLEUS deployment | **southGate** — 13/13 processes, 11/13 health-responding, 2 BTSP-gated |
| Architecture | Capability-based discovery, SSOT primal constants, `cast::*` safety, `tracing` observability |
| Deep debt | 7-stream evolution: capability discovery, cast safety (186 casts), idiom modernization, mock isolation, dep evolution, coverage expansion, variant completion |
| Active gaps | WS-9 (L3 parity), WS-11 (MAPQ 6/8 — MOB/AMP/CON/INV cross-validation logged, Gap #11) |
| PG gaps | **0 open** — 22/22 closed |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **W156** | `handoffs/WETSPRING_WAVE156_DEEP_DEBT_AUG03_2026.md` | Aug 03 | **Deep debt evolution**: 7-stream systematic cleanup — capability discovery, cast safety (186 casts), idiom modernization, mock isolation, dep evolution, coverage, variant completion. 2,201 tests. |
| **W111** | `handoffs/WETSPRING_WAVE111_WS11_DEPTH_JUN13_2026.md` | Jun 13 | **WS-11 depth**: MAPQ calibration pipeline, cross-validation engine, variant caller parity 6/8. |
| **W76-DD** | `handoffs/WETSPRING_WAVE76_DEEP_DEBT_JUN03_2026.md` | Jun 03 | **Deep debt**: TCP transport, primal name centralization, Songbird registration fix, ionic bonding modernized. |
| **W76** | `handoffs/WETSPRING_WAVE76_PARITY_ALIGNMENT_JUN03_2026.md` | Jun 03 | **Parity alignment**: 157 warnings eliminated, clippy zero, 2,085 tests. |
| | *Superseded → `handoffs/archive/`* | | W67/W63/W60/W55 + gate deployment + V183 and earlier archived (**206** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map — 800+ barraCuda WGSL, zero local |

## Upstream Blockers (Post-Deployment Findings)

| Blocker | Owner | wetSpring Impact | Status |
|---------|-------|------------------|--------|
| `compute.fan_out` primitive | toadStool | Tenaillon 264-clone batch (590 GB) awaiting fan_out scheduler | WAITING |
| `capability.call` remote dispatch | songbird | Cross-gate ionic bond calls need TCP routing via mesh | WAITING |
| `crypto.ionic_bond.seal` (Ed25519) | bearDog | Provenance seal signing for bond termination | WAITING |
| biomeOS E2E `nest.sync` | biomeOS | WS-2 cross-spring data exchange | WAITING |
| ~~loamSpine tokio nesting panic~~ | loamSpine | ~~Provenance trio incomplete on southGate~~ | **FIXED** (Wave 53) |
| ~~petalTongue `--socket` flag~~ | petalTongue | ~~BTSP-gated, process alive via manual `--port` start~~ | **RESOLVED** (Wave 60) |
| ~~barraCuda startup crash~~ | barraCuda | ~~Alive on port 9741 (manual port assignment)~~ | **RESOLVED** (Wave 60) |
| ~~ToadStool health.liveness empty~~ | toadStool | ~~Health responds via `compute-nucleus01.sock`~~ | **RESOLVED** (Wave 60) |
| Squirrel needs Ollama endpoint | Squirrel | AI narration unavailable without local Ollama (BTSP-gated) | OPTIONAL |
| ~~fetch.sh RECENT_TAGS bug~~ | plasmidBin | ~~`local` keyword bug; workaround: use cached binaries~~ | **WORKAROUND** (Wave 60) |
| Forgejo SSH key registration | eastGate | Cannot push to Forgejo; key needs API registration | **BLOCKED** |

## Archive

Superseded handoffs in `handoffs/archive/` — gate deployment + W55 + V183 and earlier (**206** files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
