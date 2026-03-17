# wetSpring wateringHole

**Date:** March 16, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V124** | `handoffs/WETSPRING_V124_DISPATCH_TRACING_DENY_HANDOFF_MAR16_2026.md` | Mar 16 | **compute.dispatch + Tracing + deny.toml** — Typed `compute.dispatch` IPC client for toadStool S156+ (healthSpring/ludoSpring pattern). Structured `tracing` replaces 14 `eprintln!` in server/songbird/fetch/metalForge (coralReef pattern). Workspace `deny.toml` with `wildcards=deny`, `yanked=deny` (groundSpring/airSpring pattern). |
| | *V123 → `handoffs/archive/`* | | V123 handoff archived. |
| | *V122 → `handoffs/archive/`* | | V122 handoff archived. |
| | *V119–V121 → `handoffs/archive/`* | | V119–V121 handoffs archived. |
| | *V113–V118 → `handoffs/archive/`* | | V113–V118 handoffs archived. |
| | *V112 and earlier → `handoffs/archive/`* | | Fossil record: V7–V112 (95+ archived handoffs). 132 total archived. |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V123 (132 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
