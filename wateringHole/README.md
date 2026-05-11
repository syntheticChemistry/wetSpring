# wetSpring wateringHole

**Date:** May 11, 2026  
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V162** | `handoffs/WETSPRING_V162_TIER4_IPC_FIRST_DEFAULTS_HANDOFF_MAY11_2026.md` | May 11 | **Tier 4 IPC-first defaults**: composition/dispatch paths default to IPC where applicable; primal gap ledger updated to 4 open (all external), 18 resolved/closed. Companion to V161 upstream + deep-debt handoffs. |
| **V161** | `handoffs/WETSPRING_V161_UPSTREAM_PRIMAL_SPRING_HANDOFF_MAY11_2026.md` | May 11 | **Upstream primal & spring team handoff**: NestGate live deployment (P0), provenance trio JSON-RPC (P0), full data/compute chain dependencies. Patterns for handler-level primal-proof wiring, legacy surface separation, foundation seeding. projectNUCLEUS deployment readiness matrix. |
| V161 | `handoffs/WETSPRING_V161_DEEP_DEBT_FOUNDATION_SEED_HANDOFF_MAY11_2026.md` | May 11 | PG-12 resolved + foundation seeded: Legacy surface separated. Thread 04 seeded (36 targets). Deep debt audit clean. Zero internal gaps. 4 open (all external), 18 resolved/closed. |
| V160 | `handoffs/WETSPRING_V160_TIER4_PRIMAL_PROOF_HANDOFF_MAY11_2026.md` | May 11 | PG-09 resolved — Tier 4 handler-level primal-proof complete. All 5 handlers wired. plasmidBin binary placed (1.4M). |
| V159 | `handoffs/WETSPRING_V159_DEEP_DEBT_PRIMAL_EVOLUTION_HANDOFF_MAY11_2026.md` | May 11 | Deep debt audit + barraCuda IPC routing: `ipc/barracuda_route.rs` module, `primal-proof` handler wiring (3 science handlers), comprehensive deep debt audit (zero debt), broken provenance reference fixed, all docs synchronized. |
| V158 | `handoffs/WETSPRING_V158_GAP_CLOSURE_FOUNDATION_HANDOFF_MAY11_2026.md` | May 11 | Post-interstadial gap closure + foundation seeding: skunkBat IPC module wired, CI cross-sync updated to 413, capability-oriented discovery, 4 gaps closed + 2 advanced (8 open/14 resolved), foundation seeding targets for NCBI 16S/cold seep/Fajgenbaum. |
| V157 | `handoffs/WETSPRING_V157_DEEP_DEBT_EVOLUTION_HANDOFF_MAY10_2026.md` | May 10 | Deep debt evolution + upstream handoff: IPC timeouts centralized, GPU API evolved, shared validation harness, skunkBat wired, biomeOS v3.51 consumed, CI cross-sync, 3 gaps closed. |
| V155 | `handoffs/WETSPRING_V155_DEEP_DEBT_DOCS_HANDOFF_MAY09_2026.md` | May 9 | Deep debt resolution + docs cleanup: formal `#[expect(reason)]` on 193 lint attrs, env-configurable data-source URLs, zero clippy warnings workspace-wide, upstream primal handoff, doc version sync. |
| V154 | `handoffs/WETSPRING_V154_INTERSTADIAL_EUKARYOTIC_HANDOFF_MAY09_2026.md` | May 9 | Interstadial eukaryotic evolution: per-trio provenance split, certification module, validation scenarios, UniBin binary, primal-proof feature, fossilization. |
| V153 | `handoffs/WETSPRING_V153_DEEP_DEBT_EVOLUTION_HANDOFF_MAY08_2026.md` | May 8 | Deep debt evolution: hardcoding → env-configurable, shared helpers, doc drift fixes. |
| V153 | `handoffs/WETSPRING_V153_UPSTREAM_PRIMAL_HANDOFF_MAY08_2026.md` | May 8 | Upstream primal handoff: gaps, patterns, composition debt for primal teams. |
|| | *Superseded → `handoffs/archive/`* | | V152 and earlier archived (**168** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back (V128 fossil record, V161 metrics banner) |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map — 800+ barraCuda WGSL, zero local (V126 fossil record, V161 metrics banner) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage (SUPERSEDED — see `docs/PRIMAL_GAPS.md`) |

## Archive

Superseded handoffs in `handoffs/archive/` — V152 and earlier (**168** files).  
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → guideStone → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:  
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).  
No reverse dependencies.
