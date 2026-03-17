# wetSpring wateringHole

**Date:** March 16, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V126** | `handoffs/WETSPRING_V126_DISPATCH_OUTCOME_HEALTH_PROBES_HANDOFF_MAR16_2026.md` | Mar 16 | **DispatchOutcome + Health Probes + IpcError Helpers** — `DispatchOutcome<T>` protocol vs application error separation (groundSpring/airSpring/sweetGrass). `health.liveness` + `health.readiness` probes (healthSpring V32). `IpcError` query helpers: `is_retriable()`, `is_timeout_likely()`, `is_method_not_found()`, `is_connection_error()` (sweetGrass circuit-breaker). 24 capabilities, 16 domains. |
| | `handoffs/WETSPRING_V125_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR16_2026.md` | Mar 16 | **toadStool/barraCuda absorption handoff** — IPC patterns, primitive consumption, ecosystem patterns. |
| | *V125 → `handoffs/archive/`* | | V125 IpcError handoff archived. |
| | *V124 and earlier → `handoffs/archive/`* | | Fossil record: V7–V124 (134 total archived). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V125 (134 files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
