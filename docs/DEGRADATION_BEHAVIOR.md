<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring — Degradation Behavior

**Last Updated**: May 17, 2026 (V177 — Wave 20 PM lithoSpore audit absorption)
**Scope**: What happens when each consumed primal is unreachable
**Contract**: Domain logic (science, validation) MUST NOT fail because a primal
is unavailable. Provenance is enrichment, not a gate.

---

## Pattern

wetSpring follows the graceful degradation pattern from
`PROVENANCE_TRIO_INTEGRATION_GUIDE.md`:

```
if socket_available {
    result = capability_call(socket, domain, operation, args);
    // use result
} else {
    // degrade: local fallback, gap report, or skip
}
```

No `has_capability()` pre-check is currently implemented — discovery is
socket-probe-based with `health.liveness` timeout (2s). This aligns with
the `CAPABILITY_BASED_DISCOVERY_STANDARD.md` tier fallback model.

---

## Per-Primal Degradation Table

| Primal | Capability | Unreachable Behavior | Returns | Science Gated? |
|--------|-----------|---------------------|---------|----------------|
| **rhizoCrypt** | `dag.session.create` | Local session ID (`local-wetspring-{ts}`) | `ProvenanceResult { available: false }` | No |
| **rhizoCrypt** | `dag.event.append` | Returns `id: "unavailable"` | `ProvenanceResult { available: false }` | No |
| **rhizoCrypt** | `dag.dehydrate` | Propagates `Err` to `complete_session` | `provenance: "unavailable"` | No |
| **loamSpine** | `ledger.commit` | Propagates `Err` to `complete_session` | `provenance: "partial"` (DAG exists, no commit) | No |
| **sweetGrass** | `braid.create` | `.ok()` + empty `braid_id` | `provenance: "complete"` (DAG+spine, no braid) | No |
| **NestGate** | `storage.store` | Best-effort return, no error | No data stored | No |
| **NestGate** | `storage.retrieve` | `GapReport` with `missing_primals` | `Ok(gap_report: true)` | No |
| **NestGate** | `storage.fetch_external` | `GapReport` with `missing_primals` | `Ok(gap_report: true)` | No |
| **barraCuda** | `tensor.*`, `stats.*`, etc. | In-process library fallback (dual-lane) | Library result | No |
| **toadStool** | `compute.dispatch` | Skip GPU dispatch | `anderson: "skipped"` | No |
| **coralReef** | `shader.compile.wgsl` | Skip shader compilation | Feature-gated out | No |
| **Squirrel** | `ai.complete` | Returns `unavailable` JSON | `Ok(status: "unavailable")` | No |
| **petalTongue** | `render.dashboard` | Facade returns 503 | HTTP `SERVICE_UNAVAILABLE` | No |
| **biomeOS** | `signal.dispatch` | Falls back to multi-call sequence | `nest.commit` fallback path | No |
| **biomeOS** | `primal.announce` | Skip registration | Silent skip | No |
| **Songbird** | `discovery.find_primals` | XDG fallback → temp dir | Socket search | No |
| **BearDog** | `crypto.hash` | N/A (Tower assumed available) | — | — |

---

## Provenance Partial Completion States

Per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` § Transaction Semantics:

| State | rhizoCrypt | loamSpine | sweetGrass | `provenance` field |
|-------|-----------|-----------|------------|-------------------|
| Full | DAG complete | Entry sealed | Braid committed | `"complete"` |
| DAG + spine | DAG complete | Entry sealed | Unreachable | `"complete"` (braid_id empty) |
| DAG only | DAG complete | Unreachable | — | `"partial"` |
| None | Unreachable | — | — | `"unavailable"` |

**Consumer rule**: wetSpring always returns `Ok(...)` from IPC handlers with
the `provenance` status field. The `session_result` JSON carries
`session_id`, `merkle_root`, `commit_id`, `braid_id` — consumers check
which fields are populated to determine completeness.

---

## Wave 17 Signal Elevation

When biomeOS v3.56+ is available, `complete_session` uses `signal.dispatch`
with `nest.commit` — biomeOS atomically manages the dehydrate → commit →
braid graph. If pre-v3.56 or signal dispatch fails, wetSpring falls back to
the three-phase multi-call sequence.

---

## Data Fetch Degradation

`data.fetch.chembl` / `data.fetch.pubchem` use a tiered acquisition path:

1. **Tier 1**: NestGate `storage.fetch_external` (HTTPS → cache)
2. **Tier 2**: NestGate `storage.retrieve` (cached data)
3. **Fallback**: `GapReport` with `missing_primals` list

The `GapReport` is returned as `Ok(json!(gap_report: true))` — never an error.
Provenance completion (`trio::complete_session`) is called regardless of
fetch success.

---

## Facade (HTTP) Degradation

The science facade (`barracuda/src/facade/`) returns:

| Condition | HTTP Status | Body |
|-----------|------------|------|
| IPC server unavailable | 503 | `{ "error": "service_unavailable" }` |
| Method not found | 404 | `{ "error": "method_not_found" }` |
| Computation error | 200 | `{ "error": "...", "degraded": true }` |
| Full success | 200 | `{ "result": ... }` |
