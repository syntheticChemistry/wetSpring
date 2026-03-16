# wetSpring wateringHole

**Date:** March 16, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V123** | `handoffs/WETSPRING_V123_ZERO_PANIC_DUAL_DISCOVERY_HANDOFF_MAR16_2026.md` | Mar 16 | **Zero-Panic + Dual Discovery** — `OrExit` trait replaces 1,039 `.expect()` + 632 `.unwrap()` across 192 binaries (groundSpring V109 pattern). Dual-format capability discovery with `operation_dependencies`/`cost_estimates`/`semantic_mappings` (neuralSpring/ludoSpring). `extract_rpc_error()` centralized (healthSpring V29). Python deps upper-bounded. |
| | *V122 → `handoffs/archive/`* | | V122 handoff archived. |
| | *V119–V121 → `handoffs/archive/`* | | V119–V121 handoffs archived. |
| | *V113–V118 → `handoffs/archive/`* | | V113–V118 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs). 131 total archived. |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V122 (131 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
