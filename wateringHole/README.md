# wetSpring wateringHole

**Date:** March 18, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V128** | `handoffs/WETSPRING_V128_ECOSYSTEM_ABSORPTION_HANDOFF_MAR18_2026.md` | Mar 18 | **Ecosystem Absorption** — `cast` module (9 helpers), `mul_add()` FMA sweep, ecoBin C-dep ban (14 crates), `PRIMAL_DOMAIN`, `FAMILY_ID` sockets, 7 IPC proptests, learnings for upstream. Supersedes V127 evolution handoff. |
| **V127** | `handoffs/WETSPRING_V127_RESILIENCE_MCP_AUDIT_HANDOFF_MAR18_2026.md` | Mar 18 | **IPC Resilience + MCP Tools + Audit Debt Resolution** — `RetryPolicy`/`CircuitBreaker`, 8 MCP tool definitions, Python baseline provenance, tolerance constants, `kahan_sum` delegation, `unlicensed = "deny"`. |
| | *V127 evolution handoff → `handoffs/archive/`* | | Superseded by V128 ecosystem absorption handoff. |
| **V126** | `handoffs/WETSPRING_V126_DISPATCH_OUTCOME_HEALTH_PROBES_HANDOFF_MAR16_2026.md` | Mar 16 | **DispatchOutcome + Health Probes + IpcError Helpers** — `DispatchOutcome<T>`, `health.liveness` + `health.readiness`, `IpcError` query helpers. 24 capabilities, 16 domains. |
| | *V125 and earlier → `handoffs/archive/`* | | Fossil record: V7–V127 (137 total archived). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | **V128** — What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V127 (137 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
