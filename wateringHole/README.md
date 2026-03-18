# wetSpring wateringHole

**Date:** March 18, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V127** | `handoffs/WETSPRING_V127_RESILIENCE_MCP_AUDIT_HANDOFF_MAR18_2026.md` | Mar 18 | **IPC Resilience + MCP Tools + Audit Debt Resolution** — `RetryPolicy`/`CircuitBreaker` (sweetGrass), 8 MCP tool definitions for Squirrel AI, Python baseline provenance registry, 7 new tolerance constants, `kahan_sum` delegation to barraCuda, `unlicensed = "deny"` policy, NPU device constant, IPC integration tests. |
| | `handoffs/WETSPRING_V127_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR18_2026.md` | Mar 18 | **barraCuda/toadStool evolution handoff** — `monod()` absorption candidate, forge dispatch module, unwired primitives roadmap, learnings for upstream evolution. |
| **V126** | `handoffs/WETSPRING_V126_DISPATCH_OUTCOME_HEALTH_PROBES_HANDOFF_MAR16_2026.md` | Mar 16 | **DispatchOutcome + Health Probes + IpcError Helpers** — `DispatchOutcome<T>` protocol vs application error separation (groundSpring/airSpring/sweetGrass). `health.liveness` + `health.readiness` probes (healthSpring V32). `IpcError` query helpers. 24 capabilities, 16 domains. |
| | *V125 + V126 evolution → `handoffs/archive/`* | | Superseded by V127 barraCuda/toadStool evolution handoff. |
| | *V124 and earlier → `handoffs/archive/`* | | Fossil record: V7–V126 (136 total archived). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | **V128** — What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V126 (136 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
