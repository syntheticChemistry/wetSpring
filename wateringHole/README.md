# wetSpring wateringHole

**Date:** April 27, 2026  
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records. Pattern library for primalSpring and primal teams.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V151** | `handoffs/WETSPRING_V151_DEEP_DEBT_EVOLUTION_HANDOFF_APR27_2026.md` | Apr 27 | **Deep debt evolution**: zero `dyn` dispatch in I/O, `Write`-based validation output, hardcoded paths removed, shared helpers extracted, concrete error types, tolerance centralization. 1209 lib tests pass. |
| V150 | `handoffs/WETSPRING_V150_PHASE46_COMPOSITION_EXPLORER_HANDOFF_APR27_2026.md` | Apr 27 | Phase 46 composition explorer: data visualization lane. Superseded by V151 debt evolution. |
| V149 | `handoffs/WETSPRING_V149_V0917_ALIGNMENT_HANDOFF_APR20_2026.md` | Apr 20 | v0.9.17 alignment: 38/38 pass (4 skip), 6 new parity checks, PG-13 resolved, `is_skip_error` adopted. Superseded by V150 composition. |
| V148 | `handoffs/WETSPRING_V148_ECOSYSTEM_EVOLUTION_HANDOFF_APR19_2026.md` | Apr 19 | Ecosystem handoff: per-primal feedback, patterns for spring teams, NUCLEUS deployment recipes. Superseded by V149 alignment. |
| V148 | `handoffs/WETSPRING_V148_GUIDESTONE_LEVEL4_NUCLEUS_HANDOFF_APR19_2026.md` | Apr 19 | **guideStone Level 4**: 31/31 live NUCLEUS, 4 primals over IPC. Handle-based matmul, sample std_dev, cross-atomic pipeline. 5 new gaps (PG-13–17). |
| V148 | `handoffs/WETSPRING_V148_PRIMALSPRING_V0916_ALIGNMENT_HANDOFF_APR20_2026.md` | Apr 20 | v0.9.16 manifest alignment: 15 validation_capabilities, BLAKE3 checksums (Property 3), family-aware discovery, Level 4 achieved. |
| V147 | `handoffs/WETSPRING_V147_ECOSYSTEM_EVOLUTION_HANDOFF_APR18_2026.md` | Apr 18 | Ecosystem evolution (pre-NUCLEUS): primal use review, composition patterns. Superseded by V148 ecosystem handoff. |
| V147 | `handoffs/WETSPRING_V147_GUIDESTONE_LEVEL3_HANDOFF_APR18_2026.md` | Apr 18 | guideStone Level 3: bare certified (9/9, exit 2). N2 expanded. CONSUMED_CAPABILITIES aligned to v0.9.15 (48 total). PG-10/11/12 documented. |
| V146 | `handoffs/WETSPRING_V146_GUIDESTONE_LEVEL2_HANDOFF_APR18_2026.md` | Apr 18 | guideStone Level 2: `wetspring_guidestone` binary via `primalspring::composition` API. Bare science + NUCLEUS IPC parity. 5 certified properties documented. |
| V145 | `handoffs/WETSPRING_V145_ECOSYSTEM_EVOLUTION_HANDOFF_APR17_2026.md` | Apr 17 | Ecosystem evolution: composition patterns, NUCLEUS deployment, per-primal feedback, 7 gaps. |
| V145 | `handoffs/WETSPRING_V145_PRIMAL_PROOF_TIER2_HANDOFF_APR17_2026.md` | Apr 17 | Primal proof Tier 2 (IPC-WIRED): Exp403, 22 consumed capabilities, PG-09. |
| V144 | `handoffs/WETSPRING_V144_COMPOSITION_VALIDATION_TIER_HANDOFF_APR17_2026.md` | Apr 17 | Composition validation tier: Exp400-402, 18 IPC roundtrip tests. |
| | *Superseded → `handoffs/archive/`* | | V143 and earlier are archived (**155** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (800+ barraCuda WGSL shaders, V126 snapshot) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — **V143 and earlier** are archived (**155** files).  
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda → NUCLEUS composition → ecoBin harvest.

## Convention

Following hotSpring's naming pattern:  
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).  
No reverse dependencies.
