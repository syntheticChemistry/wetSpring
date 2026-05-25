# wetSpring wateringHole

**Date:** May 25, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V187 — Wave 50 Covalent HPC)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.28 (458 methods, 52 scenarios, 100% coverage, behavioral convergence) |
| wetSpring | V185 (345 scenarios, 50 niche, 59 consumed, 45 dispatch, 55 baselines) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **50** (was 49; +lifecycle.status) |
| NUCLEUS deployment | **southGate LIVE** — 9/9 primals, exp091+exp094 PASS, Songbird TCP :7700 |
| Cell deployment | `wetspring_cell.toml` validated, `server` alias ready |
| plasmidBin | v5.5.0, CLI fix `8c8cb44` + simplified `9231b24` |
| Active gaps | WS-9 (L3 parity), WS-11 (variant caller re-measurement) |
| Resolved | WS-1 (ionic — WIRED), WS-4 (WASM upstream), WS-8, WS-10 |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **W50** | `handoffs/WETSPRING_WAVE50_COVALENT_HPC_MAY25_2026.md` | May 25 | **Covalent HPC readiness**: NUCLEUS 7/13, cross-subnet confirmed, WS-2 nest.sync path, upstream blockers documented. |
| **W49** | `handoffs/WETSPRING_WAVE49_POST_PRIMORDIAL_MAY25_2026.md` | May 25 | **Post-primordial cleanup**: plasmidBin-only discovery, shared `primal_binary` module, composition script cleaned, legacy `phase2/` paths updated. |
| **W48** | `handoffs/WETSPRING_WAVE48_COVALENT_MESH_MAY25_2026.md` | May 25 | **Covalent mesh deployment conformance**: health.liveness shape, lifecycle.status, bonding.* dispatched, Songbird TCP :7700, cell deployment. |
| | *Superseded → `handoffs/archive/`* | | Gate deployment + V183 and earlier archived (**202** files). |

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
| loamSpine tokio nesting panic | loamSpine | Provenance trio incomplete on southGate | **BUG** |
| Health probe timing | plasmidBin/respective | rhizoCrypt/sweetGrass/toadStool sometimes exceed 8s timeout | TIMING |
| Squirrel needs Ollama endpoint | Squirrel | AI narration unavailable without local Ollama | OPTIONAL |

## Archive

Superseded handoffs in `handoffs/archive/` — gate deployment + V183 and earlier (**202** files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
