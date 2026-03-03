# wetSpring wateringHole

**Date:** March 3, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V93** | `handoffs/WETSPRING_BARRACUDA_031_REWIRE_HANDOFF_MAR03_2026.md` | Mar 3 | Standalone barraCuda v0.3.1 rewire: path swap, MSRV bump, NPU strategy, 1,044 tests pass, zero API breakage |
| **V93** | `handoffs/WETSPRING_V93_BARRACUDA_EVOLUTION_FEEDBACK_MAR03_2026.md` | Mar 3 | Evolution feedback to barraCuda/toadStool: 144 primitives consumed, precision observations, deep debt patterns, architecture after rewire |
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

Superseded handoffs in `handoffs/archive/` — V7-V92J ToadStool-era handoffs (80+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
