# wetSpring wateringHole

**Date:** June 3, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Current Ecosystem State (V195 — Wave 76 Deep Debt)

| Metric | Value |
|--------|-------|
| primalSpring | v0.9.28 (458 methods, 56 scenarios, 92 experiments, temporal sync spec) |
| wetSpring | V195 (345 scenarios, 50 niche, 59 consumed, 45 dispatch, 55 baselines, 2,089 tests, clippy zero) |
| Registry sync | **458** — zero drift |
| Niche capabilities | **50** |
| NUCLEUS deployment | **southGate** — 10/13 health (coralReef rename, 2 BTSP-gated) |
| Architecture | TCP transport, primal name constants, Songbird registration fixed, macOS RSS |
| Dependencies | axum 0.8.9, blake3 1.8.5, proptest 1.11, tempfile 3.27, tower-http 0.6.11 |
| Forgejo sync | Remotes configured, SSH key registration **pending** |
| Active gaps | WS-9 (L3 parity — needs FASTQ), WS-11 (MAPQ calibration — needs dataset) |
| PG gaps | **0 open** — 22/22 closed |

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **W76-DD** | `handoffs/WETSPRING_WAVE76_DEEP_DEBT_JUN03_2026.md` | Jun 03 | **Deep debt**: TCP transport, primal name centralization, Songbird registration fix, ionic bonding modernized, 5 deps bumped, 2,089 tests. |
| **W76** | `handoffs/WETSPRING_WAVE76_PARITY_ALIGNMENT_JUN03_2026.md` | Jun 03 | **Parity alignment**: bearDog w135 absorbed (no auth calls), 157 warnings eliminated, clippy zero, 2,085 tests clean. |
| **W67** | `handoffs/WETSPRING_WAVE67_GLACIAL_CUTOVER_JUN01_2026.md` | Jun 01 | **Glacial cutover**: Songbird socket fix, biomeOS capability.call proxy, bearDog S4 auth verified. |
| **W63** | `handoffs/WETSPRING_WAVE63_RIVER_DELTA_MAY30_2026.md` | May 30 | **River Delta**: PG-02/PG-04 re-verified, composition_nucleus.sh fossilized, domain_profile.toml created, temporal sync tooling confirmed. |
| **W60-S** | `handoffs/WETSPRING_WAVE60_STABILIZATION_MAY29_2026.md` | May 29 | **Stabilization**: Clippy zero, cast fixes, airSpring AAR response, steady-state. |
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
