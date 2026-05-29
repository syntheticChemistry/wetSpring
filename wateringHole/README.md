# wetSpring wateringHole

**Date:** May 29, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V190 — Wave 60 Eukaryotic)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.28 (458 methods, 52 scenarios, 100% coverage, behavioral convergence) |
| wetSpring | V190 (345 scenarios, 50 niche, 59 consumed, 45 dispatch, 55 baselines, 2,085 tests) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **50** |
| NUCLEUS deployment | **southGate** — 13/13 processes (11/13 health, 2 BTSP-gated), biomeOS 1725 caps |
| Forgejo sync | Remotes configured, SSH key registration **pending** |
| plasmidBin | v5.6.0 (v2026.05.28) |
| Active gaps | WS-9 (L3 parity), WS-11 (variant caller re-measurement) |
| PG gaps | **0 open** — PG-02/PG-04 VERIFIED, 22/22 closed |
| Resolved | WS-1 (ionic — WIRED), WS-4 (WASM upstream), WS-8, WS-10 |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **W60** | `handoffs/WETSPRING_WAVE60_EUKARYOTIC_ONBOARDING_MAY28_2026.md` | May 28 | **Eukaryotic onboarding**: 13/13 NUCLEUS, Forgejo remotes, WaterFall profile, SSH key blocker. |
| | *Superseded → `handoffs/archive/`* | | W55/W48/W49/W50 + gate deployment + V183 and earlier archived (**206** files). |

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
