# wetSpring wateringHole

**Date:** May 23, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V185 — Wave 48 Covalent Mesh)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.28 (458 methods, 52 scenarios, 100% coverage, behavioral convergence) |
| wetSpring | V185 (345 scenarios, 49 niche, 52 consumed, 55 baselines) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **49** (was 43; +6 bonding.*) |
| NUCLEUS deployment | **southGate LIVE** — 9/9 primals, exp091+exp094 PASS, Songbird TCP :7700 |
| Cell deployment | `wetspring_cell.toml` validated, `server` alias ready |
| plasmidBin | v5.5.0, CLI fix `8c8cb44` + simplified `9231b24` |
| Active gaps | WS-9 (L3 parity), WS-11 (variant caller re-measurement) |
| Resolved | WS-1 (ionic — IMPLEMENTED + locally wired), WS-4 (WASM expanded), WS-8, WS-10 |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **Gate** | `handoffs/WETSPRING_GATE_DEPLOYMENT_MAY23_2026.md` | May 23 | **Post-primordial gate deployment EXECUTED**: southGate 9/9 primals, exp091+exp094 PASS. Deployment issues documented for upstream. |
| **V183** | `handoffs/WETSPRING_V183_DEEP_DEBT_EVOLUTION_HANDOFF_MAY22_2026.md` | May 22 | **Deep debt evolution**: Track A-E (refactoring, discovery, baselines, fan_out, notebooks). Composition patterns for NUCLEUS deployment. |
| **V180** | `handoffs/WETSPRING_UPSTREAM_ASKS_RIVER_DELTA_MAY19_2026.md` | May 19 | **River Delta**: upstream asks for WS-2 (RootPulse), WS-3 (chain anchor), WS-4 (WASM). WS-1 ionic now RESOLVED locally (V184). |
|| | *Superseded → `handoffs/archive/`* | | V182 and earlier archived (**199** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map — 800+ barraCuda WGSL, zero local |

## Upstream Blockers (Post-Deployment Findings)

| Blocker | Owner | wetSpring Impact | Status |
|---------|-------|------------------|--------|
| `compute.fan_out` primitive | toadStool | Tenaillon 264-clone batch (590 GB) awaiting fan_out scheduler | WAITING |
| `capability.call` remote dispatch | songbird | Cross-gate ionic bond calls need TCP routing over TURN relay | WAITING |
| `crypto.ionic_bond.seal` (Ed25519) | bearDog | Provenance seal signing for bond termination | WAITING |
| Nest deploy (VPS) | nestGate / projectNUCLEUS | WS-9 L3 parity requires live trio | WAITING |
| biomeOS E2E `nest.sync` | biomeOS | WS-2 cross-spring data exchange | WAITING |
| loamSpine tokio nesting panic | loamSpine | Provenance trio incomplete on southGate | **BUG** |
| toadStool/nestgate/rhizoCrypt health probe | plasmidBin/respective | Health sweep shows UNREACHABLE despite start success | TIMING |
| Squirrel needs Ollama endpoint | Squirrel | AI narration unavailable without local Ollama | OPTIONAL |

## Archive

Superseded handoffs in `handoffs/archive/` — V182 and earlier (**199** files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
