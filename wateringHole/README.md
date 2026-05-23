# wetSpring wateringHole

**Date:** May 23, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (Wave 46)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.27 (458 methods, 49 scenarios, ionic runtime live) |
| wetSpring | V184 (345 scenarios, 49 niche, 52 consumed, 55 baselines) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **49** (was 43; +6 bonding.*) |
| Active gaps | WS-9 (L3 parity), WS-11 (variant caller re-measurement) |
| Resolved | WS-1 (ionic — IMPLEMENTED + locally wired), WS-8, WS-10 |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V184** | *(this commit)* | May 23 | **Wave 46 absorption**: registry 458, ionic bonding module, primalSpring v0.9.27 pin, 49 niche. |
| **V183** | `handoffs/WETSPRING_V183_DEEP_DEBT_EVOLUTION_HANDOFF_MAY22_2026.md` | May 22 | **Deep debt evolution**: Track A-E (refactoring, discovery, baselines, fan_out, notebooks). Composition patterns for NUCLEUS deployment. |
| **V180** | `handoffs/WETSPRING_UPSTREAM_ASKS_RIVER_DELTA_MAY19_2026.md` | May 19 | **River Delta**: upstream asks for WS-2 (RootPulse), WS-3 (chain anchor), WS-4 (WASM). WS-1 ionic now RESOLVED locally (V184). |
|| | *Superseded → `handoffs/archive/`* | | V182 and earlier archived (**199** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map — 800+ barraCuda WGSL, zero local |

## Upstream Blockers (Wave 46)

| Blocker | Owner | wetSpring Impact |
|---------|-------|------------------|
| `compute.fan_out` primitive | toadStool | Tenaillon 264-clone batch (590 GB) awaiting fan_out scheduler |
| `capability.call` remote dispatch | songbird | Cross-gate ionic bond calls need TCP routing over TURN relay |
| `crypto.ionic_bond.seal` (Ed25519) | bearDog | Provenance seal signing for bond termination |
| Nest deploy (VPS) | nestGate / projectNUCLEUS | WS-9 L3 parity requires live trio |
| biomeOS E2E `nest.sync` | biomeOS | WS-2 cross-spring data exchange |

## Archive

Superseded handoffs in `handoffs/archive/` — V182 and earlier (**199** files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
