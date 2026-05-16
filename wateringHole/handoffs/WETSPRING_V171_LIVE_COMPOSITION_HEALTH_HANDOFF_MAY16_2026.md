# wetSpring V171 — Live Composition Health Handoff

**Date:** May 16, 2026
**From:** wetSpring
**To:** primalSpring, biomeOS, Provenance Trio teams, NestGate team, all springs
**Scope:** `composition.science_health` evolved from static deferred declarations to live runtime probing

---

## What Changed

### 1. New Module: `barracuda/src/ipc/composition_health.rs`

Four probing functions with graceful degradation:

- `probe_trio_status()` — discovers sockets for rhizoCrypt, loamSpine, sweetGrass via `discover::discover_socket()`. For each: attempts `health.liveness` JSON-RPC with 500ms timeout. Returns per-component `"live"` / `"discovered"` / `"absent"`.
- `probe_nestgate_status()` — same pattern for NestGate.
- `probe_biomeos_status()` — checks Neural API socket via `provenance::neural_api_socket()`, attempts `primal.list` (Wave 20) for live primal count, falls back to `health.liveness`.
- `probe_schema_parity()` — self-validates `capability.list` response against Wave 20 canonical shape (capabilities array + count field match).

All probes use a 500ms timeout, return typed status enums, and never panic.

### 2. Evolved Handler: `handle_composition_science_health()`

Before (V170):
```json
{
  "subsystems": { "provenance_trio": "deferred_check", "nestgate": "deferred_check" },
  "biome_os": { "composition_status": "deferred_to_live", "method_register": "deferred_to_live" }
}
```

After (V171):
```json
{
  "subsystems": {
    "provenance_trio": { "rhizocrypt": "absent", "loamspine": "absent", "sweetgrass": "absent", "summary": "absent" },
    "nestgate": "absent"
  },
  "biome_os": {
    "neural_api": "absent",
    "primal_count": null,
    "schema_parity": { "conformant": true, "has_capabilities_array": true, "has_count": true, "count_matches": true },
    "wave": 20
  }
}
```

### 3. Degradation Table

| Environment | Trio | NestGate | BiomeOS | healthy |
|---|---|---|---|---|
| Bare (CI, no sockets) | all "absent" | "absent" | "absent" | true (bare science valid) |
| Partial NUCLEUS | mixed | varies | varies | true |
| Full NUCLEUS | all "live" | "live" | "live" | true |
| Socket exists but RPC fails | "discovered" | "discovered" | "discovered" | true (degraded) |

### 4. Tests

- 7 new unit tests in `composition_health::tests`
- 1 new dispatch-level test: `composition_science_health_live_probing`
- Updated integration test: `composition_science_health_roundtrip` validates new response shape (trio is object, nestgate is string, biome_os has neural_api and schema_parity)

---

## Why This Matters for Upstream

### For primalSpring / guideStone validators
- `composition.science_health` now returns **live truth** — validators can distinguish between a spring that has never probed (V170 `"deferred_check"`) and one that has probed and found primals absent/discovered/live.
- Schema parity self-check means wetSpring validates its own Wire Standard conformance on every health call.

### For biomeOS
- The `primal_count` field (populated via `primal.list` Wave 20 RPC) gives biomeOS a feedback loop: the spring reports how many primals it sees from the biomeOS perspective.
- `neural_api` field reports actual socket reachability, not a static placeholder.

### For Provenance Trio teams (rhizoCrypt, loamSpine, sweetGrass)
- Per-component status gives visibility into which trio components are deployed. When PG-02 resolves (deployment), these fields will transition from `"absent"` to `"live"`.

### For NestGate team
- Same pattern: PG-04 deployment will be visible in the `nestgate` field transitioning to `"live"`.

### For other springs
- This pattern (live probing with graceful degradation) is recommended for all springs implementing `composition.*_health`. It replaces the Wave 17 deferred-check pattern with live truth while remaining safe for bare-science CI runs.

---

## Composition Pattern for Adoption

Springs wanting to adopt this pattern need:

1. `discover::discover_socket()` or equivalent for peer primal socket discovery
2. A short-timeout JSON-RPC probe function (500ms recommended for health endpoints)
3. Typed status enum: `Live` / `Discovered` / `Absent`
4. Schema parity self-check against Wave 20 `capability.list` canonical shape
5. `healthy: true` even when all subsystems are absent (bare science is valid)

---

## Metrics

- Build gate: `cargo clippy --features ipc --lib -- -W clippy::pedantic -W clippy::nursery` (zero warnings)
- Tests: 252 lib (composition_health: 7, dispatch: 1 new), integration roundtrip updated
- Primal gaps: 2 open (PG-02 trio deploy, PG-04 NestGate deploy), 20 resolved/closed
- Registry: 452 methods, Wave 20

---

## Deferred Items

- **Provenance-first live data chains**: Planned next phase — `nest.store`/`nest.commit` E2E validation with live trio deployment.
- **Timeout tuning**: 500ms is conservative. May lower for local UDS once trio is deployed.
- **Parallel probing**: Currently sequential. Could use `std::thread::scope` for parallel socket probes if latency matters.

---

## Upstream Observations

- biomeOS `primal.list` response format not yet confirmed in production. The probe assumes `{ "result": { "count": N } }` shape per Wave 20 spec.
- trio components discovered via `discover_socket()` use the standard env-var → XDG → temp fallback chain. No custom discovery needed.
- Schema parity self-check catches any future regression in our own `capability.list` response shape.
