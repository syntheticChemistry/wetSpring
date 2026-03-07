# wetSpring wateringHole

**Date:** March 7, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V97e** | `handoffs/WETSPRING_V97E_PROVENANCE_REWIRE_HANDOFF_MAR07_2026.md` | Mar 7 | Provenance rewire: builder patterns (HMM, DADA2, Gillespie), PrecisionRoutingAdvice, shaders::provenance API, Exp312 (31/31). 1,346 tests, zero warnings. |
| **V97d+** | `handoffs/WETSPRING_V97D_ECOSYSTEM_SYNC_HANDOFF_MAR07_2026.md` | Mar 7 | Ecosystem sync: barraCuda 2a6c072, toadStool S130, coralReef Phase 10. Zero API breakage, 1,347 tests PASS. |
| **V97d** | `handoffs/WETSPRING_V97D_DEEP_AUDIT_EVOLUTION_HANDOFF_MAR07_2026.md` | Mar 7 | Deep audit: I/O deprecation, unwrap→expect evolution, doc accuracy, broken ref cleanup (Exp311, 125 items) |
| **V97c** | `handoffs/WETSPRING_V97C_FUSED_OPS_CHAIN_HANDOFF_MAR05_2026.md` | Mar 5 | Fused ops full chain: Exp306-310 (111 checks). DF64 dispatch routing confirmed wired; DF64 fused shaders produce zero on RTX 4070 (shader validation gap, not wiring). |
| **V97** | `handoffs/WETSPRING_V97_BARRACUDA_033_WGPU28_REWIRE_HANDOFF_MAR05_2026.md` | Mar 5 | barraCuda v0.3.3 + wgpu 28 rewire: 1,247 tests, zero clippy, chi_squared upstream fix |
| **V96** | `handoffs/WETSPRING_V96_DEEP_DEBT_CHUNA_HANDOFF_MAR05_2026.md` | Mar 5 | Deep debt: silent fallback elimination, capability IPC, Chuna papers queued |
| **V95** | `handoffs/WETSPRING_V95_CROSS_SPRING_EVOLUTION_COMPLETE_MAR04_2026.md` | Mar 4 | Cross-spring evolution complete: 6 GPU ops + 2 CPU delegations wired, Exp305 (59/59), full provenance table, benchmarks |
| **V94** | `handoffs/WETSPRING_V94_BARRACUDA_EVOLUTION_SYNC_MAR04_2026.md` | Mar 4 | barraCuda evolution sync: norm_ppf wiring, 50+ doc files cleaned (ToadStool → barraCuda), gap analysis |
| **V93+** | `handoffs/WETSPRING_V93_DEEP_DEBT_TOADSTOOL_HANDOFF_MAR04_2026.md` | Mar 4 | Deep debt round 3: 164 tolerances, test extraction, provenance complete |
| **V93** | `handoffs/WETSPRING_BARRACUDA_031_REWIRE_HANDOFF_MAR03_2026.md` | Mar 3 | Standalone barraCuda v0.3.1 rewire: path swap, MSRV bump, zero API breakage |
| **V93** | `handoffs/WETSPRING_V93_BARRACUDA_EVOLUTION_FEEDBACK_MAR03_2026.md` | Mar 3 | Evolution feedback: 150+ primitives consumed, precision observations |
| **V92F** | `handoffs/WETSPRING_CROSS_SPRING_V92F_MODERN_S86_HANDOFF_MAR02_2026.md` | Mar 2 | Cross-spring modern S86 validation + benchmark |
| **V92D+** | `handoffs/WETSPRING_PAPER_CHAIN_V92D_PLUS_HANDOFF_MAR02_2026.md` | Mar 2 | Paper-math chain: 52 papers, CPU→GPU→streaming→metalForge |
| **V92B** | `handoffs/WETSPRING_GONZALES_REPRO_V92B_HANDOFF_MAR02_2026.md` | Mar 2 | Gonzales reproducibility handoff |
| **V92** | `handoffs/WETSPRING_IMMUNO_ANDERSON_V92_HANDOFF_MAR02_2026.md` | Mar 2 | Immunological Anderson handoff |
| **V91** | `handoffs/WETSPRING_DEEP_DEBT_V91_HANDOFF_MAR02_2026.md` | Mar 2 | Deep debt resolution handoff |
| **V90** | `handoffs/WETSPRING_BIO_BRAIN_V90_CROSS_SPRING_HANDOFF_MAR02_2026.md` | Mar 2 | Bio Brain cross-spring handoff |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V92J ToadStool-era handoffs (80+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
