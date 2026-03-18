# wetSpring wateringHole

**Date:** March 18, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V129** | `handoffs/WETSPRING_V129_DEEP_DEBT_EVOLUTION_HANDOFF_MAR18_2026.md` | Mar 18 | **Deep Debt Evolution** — `cast` module (15 helpers, ~170 raw casts migrated), unconditional `primal_names` module (zero hardcoded primal strings), upstream `bingocube-nautilus` JSON roundtrip fix, pure Rust binary discovery, `skip_with_code()` composable exits, 2 new tolerance constants. Supersedes V128. |
| **V127** | `handoffs/WETSPRING_V127_RESILIENCE_MCP_AUDIT_HANDOFF_MAR18_2026.md` | Mar 18 | **IPC Resilience + MCP Tools + Audit Debt Resolution** — `RetryPolicy`/`CircuitBreaker`, 8 MCP tool definitions, Python baseline provenance, tolerance constants, `kahan_sum` delegation, `unlicensed = "deny"`. |
| **V126** | `handoffs/WETSPRING_V126_DISPATCH_OUTCOME_HEALTH_PROBES_HANDOFF_MAR16_2026.md` | Mar 16 | **DispatchOutcome + Health Probes + IpcError Helpers** — `DispatchOutcome<T>`, `health.liveness` + `health.readiness`, `IpcError` query helpers. 24 capabilities, 16 domains. |
| | *V125 and earlier → `handoffs/archive/`* | | Fossil record: V7–V128 (138 total archived). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | **V128** — What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V128 (138 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
