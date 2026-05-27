# wetSpring wateringHole

**Date:** May 26, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V188 — Wave 55 Redeploy)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.28 (458 methods, 52 scenarios, 100% coverage, behavioral convergence) |
| wetSpring | V188 (345 scenarios, 50 niche, 59 consumed, 45 dispatch, 55 baselines) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **50** |
| NUCLEUS deployment | **southGate** — 8/13 health-responding, Wave 53 hardened, Songbird stable |
| Cell deployment | `wetspring_cell.toml` validated, `server` alias ready |
| plasmidBin | v5.6.0 (v2026.05.27 release) |
| Active gaps | WS-9 (L3 parity), WS-11 (variant caller re-measurement) |
| PG gaps | **0 open** — PG-02/PG-04 VERIFIED, 22/22 closed |
| Resolved | WS-1 (ionic — WIRED), WS-4 (WASM upstream), WS-8, WS-10 |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **W55** | `handoffs/WETSPRING_WAVE55_SOUTHGATE_REDEPLOY_MAY26_2026.md` | May 26 | **southGate redeploy**: Wave 53 hardened binaries, 8/13 NUCLEUS, PG-02/PG-04 VERIFIED, mesh seeded. |
| | *Superseded → `handoffs/archive/`* | | W48/W49/W50 + gate deployment + V183 and earlier archived (**205** files). |

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
| petalTongue `--socket` flag | petalTongue | Cannot join NUCLEUS composition on southGate | **BUG** |
| barraCuda startup crash | barraCuda | No GPU tensor compute via IPC on southGate | **BUG** |
| ToadStool health.liveness empty | toadStool | Health probes fail despite socket existing | **BUG** |
| Squirrel needs Ollama endpoint | Squirrel | AI narration unavailable without local Ollama | OPTIONAL |
| fetch.sh RECENT_TAGS bug | plasmidBin | 4 primals unfetchable when not in latest release | **BUG** |

## Archive

Superseded handoffs in `handoffs/archive/` — gate deployment + V183 and earlier (**202** files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
