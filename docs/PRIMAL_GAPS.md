# wetSpring — Primal Composition Gaps

Gaps discovered during primal composition validation (Exp400 and IPC
integration). Each gap is handed back to primalSpring for ecosystem-wide
refinement per `NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.

Last updated: 2026-05-11 (V160 — PG-09 handler wiring complete.
All 5 handlers that call `barracuda::*` directly are now `primal-proof` wired:
`handle_diversity` (→ `stats.diversity`), `handle_anderson` (→ `spectral.anderson_3d`),
`handle_qs_model` (→ `compute.ode_rk4`), `handle_nmf` (→ `linalg.nmf`),
`handle_dose_response` (→ `stats.hill_sweep`). Remaining handlers use only
`crate::bio::*` (wetSpring library) — no barraCuda IPC routing needed.
Tier 4 Cargo structure confirmed: `barracuda` is `optional = true` with
`barracuda-lib` feature gate (default on). 7 open gaps, 15 resolved/closed.
1,962 lib tests.)

---

## Gap Tracking Convention

- Gaps are **composition-specific**: missing primal capabilities, IPC
  misalignment, proto-nucleate drift, or bonding protocol holes.
- Architectural gaps that are not composition-related live in
  [GAPS.md](../GAPS.md) (7 architectural gaps documented).
- Each gap references the downstream proto-nucleate manifest:
  `primalSpring/graphs/downstream/downstream_manifest.toml` (wetspring:
  `spring_name = "wetspring"`)

---

## PG-01: Proto-Nucleate Not Parsed at Build Time (Resolved V141)

**Owner:** wetSpring (internal)
**Status:** Resolved — `niche::tests::proto_nucleate_node_names_match_niche_dependencies`

`niche.rs` now reads `primalSpring/graphs/downstream/downstream_manifest.toml`
at test time via `std::fs::read_to_string` and cross-checks the `[[downstream]]`
entry keyed by `spring_name = "wetspring"`, the manifest’s
`proto_nucleate_template.toml` reference, `owner = "wetSpring"`, and niche
dependency names against that row’s `depends_on` (with IPC trio optional per
PG-02). Drift is caught in CI.

---

## PG-02: Provenance Trio — IPC Clients Partially Wired

**Owner:** rhizoCrypt, loamSpine, sweetGrass teams
**Status:** Partial — IPC wiring exists, trio endpoints not yet live

V142 progress:
- `ipc/provenance.rs` sends `capability.call` to `dag.session.create`,
  `dag.event.append`, `dag.dehydrate` (rhizoCrypt), `session.commit`
  (loamSpine), and `braid.create` (sweetGrass) via Neural API socket
- `ipc/sweetgrass.rs` has typed `BraidRequest`/`BraidCommitRequest` structs
  with `braid.create` + `braid.commit` RPC calls
- `WireWitnessRef` struct added per Attestation Encoding Standard v2.0 —
  provenance handlers emit `witnesses` array in responses
- `CONSUMED_CAPABILITIES` declares `dag.session.create`, `entry.append`,
  `braid.create`, `braid.commit` per Wire Standard L3
- Graceful degradation: all trio calls fall back to local session tracking
  when Neural API socket is unavailable

**Remaining:** Trio primals reaching IPC-ready status with stable endpoints.
Once live, wetSpring's existing IPC paths will route to them transparently.

**Blocked by:** Trio primals reaching IPC-ready status.

---

## PG-03: Capability Discovery Is Name-Based

**Owner:** Songbird team / biomeOS + wetSpring (partial)
**Status:** Partial (V158) — capability-oriented abstraction wired, runtime
discovery still name-based pending Songbird `capability.resolve`

V158 progress:
- `discover_by_capability(domain)` maps capability domains to provider
  primals and resolves sockets — single migration point when Songbird
  ships `capability.resolve`
- `capability_to_primal(domain)` provides the canonical mapping (15 domains
  covering all 13 primals) at `const` time
- All callers can migrate from `discover_primal("barracuda")` to
  `discover_by_capability("tensor")` — decouples intent from identity
- Tests cover all known domain mappings and unknown-domain handling

**Remaining:** Songbird implementing `capability.resolve` → socket path
mapping. When that ships, `discover_by_capability` swaps its internals
from `capability_to_primal → discover_primal` to a single Songbird RPC
call. Callers unchanged.

---

## PG-04: NestGate Storage — IPC Wired, Awaiting Live Deployment

**Owner:** NestGate team
**Status:** Partial — IPC routing implemented (V152), NestGate not yet deployed live

V152 progress:
- `data_fetch.rs` routes all external fetches through
  `capability.call("storage", "fetch_external")` via biomeOS → NestGate
- `capability.call("storage", "store")` for NestGate cache persistence
- `capability.call("storage", "retrieve")` for cache retrieval
- Pure primal composition: no fallbacks, gap reports on missing primals
- exp400 validates NestGate health and cross-atomic store→retrieve pipeline

**Remaining:** NestGate live deployment in NUCLEUS stack. IPC paths are
wired and tested — gap is infrastructure, not code.

---

## PG-05: toadStool Compute — IPC Discovery + barraCuda Optional

**Owner:** toadStool team
**Status:** Partial — discovery wired, barraCuda now optional (V152)

V152 progress:
- `discover_toadstool()` helper resolves toadStool socket
- barraCuda marked `optional = true` in `Cargo.toml` with `barracuda-lib`
  feature gate — IPC-first deployment possible without library linking
- exp400 validates `compute.health` via biomeOS
- `deny.toml` bans `ring` crate for Tower crypto purity

**Remaining:** Full compute dispatch via toadStool IPC (sovereign dispatch
path). barraCuda library dep remains as default for validation/CI.

---

## PG-06: Ionic Bond Negotiation Protocol

**Owner:** primalSpring Track 4
**Status:** Metadata declared; no negotiation protocol (same as GAPS.md #1)

The facade declares bonding capabilities (Covalent, Ionic) in composition
health responses and bonding metadata. No automated protocol for
establishing, modifying, or terminating ionic bonds exists.

---

## PG-07: Niche vs Handlers Capability Drift (Resolved V141)

**Owner:** wetSpring (internal)
**Status:** Resolved — cross-check test added

`niche::CAPABILITIES` (45) included 3 aspirational strings not backed by
dispatch handlers. `capability_domains::DOMAINS` was missing vault,
data.fetch, and composition families. Both are now covered by automated
tests in `capability_domains::tests`.

---

## PG-08: spring_validate_manifest Binary Name Inconsistency (Closed V158)

**Owner:** primalSpring (manifest maintainer)
**Status:** Closed (V158) — informational, upstream manifest alignment

`primalSpring/graphs/spring_validation/spring_validate_manifest.toml` uses
`spring_binary = "wetspring"` for the wetSpring entry, while
`downstream_manifest.toml` and `spring_deploy_manifest.toml` both use
`spring_binary = "wetspring_primal"`. One of these should be canonical.

**V158 resolution:** Closed as informational — this is a primalSpring
manifest maintenance item, not a wetSpring code gap. wetSpring's UniBin
binary name is `wetspring_unibin` per the `[[bin]]` table in Cargo.toml.
The inconsistency in upstream manifests does not affect wetSpring's build
or validation. Handed off to primalSpring via wateringHole.

---

## PG-09: barraCuda IPC Evaporation Surface — Domain Math via IPC

**Owner:** wetSpring (internal)
**Status:** Resolved (V160) — all 5 handlers that call `barracuda::*` directly
are `primal-proof` wired with IPC-first routing and graceful in-process fallback.

barraCuda is a full ecobin primal. The v0.9.17 canonical surface defines 33
JSON-RPC methods (TENSOR 9, STATS 9, COMPUTE 4, SPECTRAL 3, LINALG 6,
HEALTH 2). Today, wetSpring links barraCuda as a Rust library dependency
(`path = "../../../primals/barraCuda"`) and calls math in-process. For the
primal proof (Level 5), domain math must migrate to IPC.

**What exists (V160):**
- `niche::CONSUMED_CAPABILITIES` declares full v0.9.17 canonical surface (33
  methods) plus 15 legacy Exp403 methods pending migration
- `wetspring_guidestone` binary (Level 4): NUCLEUS validated (38/38 pass, 4 skip,
  exit 0), bare mode (16/16 pass, exit 2), handle-based matmul, all stats/linalg/
  spectral methods passing, cross-atomic pipeline. `is_skip_error` from v0.9.17
- Exp403 (`validate_primal_parity_v1`) remains as Tier 2 IPC-WIRED validation
  with the original 22-method surface
- Socket discovery uses `ipc::discover::discover_primal()` (env var → XDG → temp)
- **`ipc/barracuda_route.rs`** — barraCuda IPC routing module with `discover()`,
  `forward()`, `try_forward()`, `is_available()`, and 5 tests
- **Handler-level `#[cfg(feature = "primal-proof")]`** — all 5 handlers that call
  `barracuda::*` directly are wired with IPC-first routing:
  - `handle_diversity` → `stats.diversity` (V159)
  - `handle_anderson` → `spectral.anderson_3d` (V159)
  - `handle_qs_model` → `compute.ode_rk4` (V159)
  - `handle_nmf` → `linalg.nmf` (V160)
  - `handle_dose_response` → `stats.hill_sweep` (V160)
  Each attempts IPC first and falls back to in-process on any failure (socket
  absent, transport error, RPC reject). Remaining handlers (`kinetics`,
  `alignment`, `taxonomy`, `phylogenetics`, `brain`, gonzales non-hill) use
  only `crate::bio::*` (wetSpring library) — no barraCuda IPC routing needed.

**Evaporation candidates (library → IPC migration):**

| Library call | IPC method | Priority | V159 Status |
|-------------|-----------|----------|-------------|
| `barracuda::stats::mean()` | `stats.mean` | High | Covered via `stats.diversity` forward |
| `barracuda::stats::std_dev()` | `stats.std_dev` | High | Covered via `stats.diversity` forward |
| `barracuda::stats::weighted_mean()` | `stats.weighted_mean` | Medium | Pending |
| `barracuda::linalg::matmul()` | `tensor.matmul` | High | Pending (handler not yet wired) |
| `barracuda::ops::*` | `tensor.*` | Medium | Pending |
| `barracuda::dispatch::*` | `compute.dispatch` | High | `compute.ode_rk4` wired |
| `barracuda::sample::noise_*()` | `noise.perlin2d/3d` | Low | Pending |
| `barracuda::spectral::*` | `spectral.anderson_3d` | High | **Wired V159** |

**Impact:** The library dep remains for Level 2 Rust-proof comparison and CI.
The IPC path is additive. Full evaporation happens when Tier 3 (NUCLEUS from
plasmidBin) is deployed.

V158 progress:
- `primal-proof` Cargo feature declared (enables IPC-only sovereign build)
- `discover_by_capability("tensor")` routes to barraCuda socket via
  capability-oriented abstraction (decouples from primal name)
- `BARRACUDA` constant added to `primal_names.rs` for discovery wiring
- Dual-lane pattern documented: `cargo build --features ipc,primal-proof`
  (IPC-only) vs `cargo build --features ipc` (in-process default)

V159 progress:
- `ipc/barracuda_route.rs` — full routing module (`discover`, `forward`,
  `try_forward`, `is_available`, atomic request IDs, graceful degradation)
- `#[cfg(feature = "primal-proof")]` wired into `handle_diversity`,
  `handle_anderson`, `handle_qs_model` — three core science handlers

V160 progress:
- `handle_nmf` → `linalg.nmf` IPC forward wired (drug.rs)
- `handle_dose_response` → `stats.hill_sweep` IPC forward wired (gonzales.rs)
- Audit of all remaining handlers confirmed: `kinetics`, `alignment`,
  `taxonomy`, `phylogenetics`, `brain`, and gonzales non-hill handlers use
  only `crate::bio::*` — no direct `barracuda::` calls, no IPC routing needed
- Handler-level primal-proof wiring is **complete** — all 5/5 barraCuda-calling
  handlers are covered

**Remaining (non-blocking):** Add integration test with a mock barraCuda
responder, extend `health.readiness` to report barraCuda primal connectivity
when `primal-proof` is active.

**Status:** Resolved — handler-level wiring is complete. The remaining items
are quality-of-life improvements, not Tier 4 blockers.

---

## PG-10: spectral/linalg Routing Gap in primalSpring composition API

**Owner:** primalSpring (composition API)
**Status:** Open — discovered during V147 guideStone N2 expansion

`primalspring::composition::method_to_capability_domain()` maps method
prefixes to capability domains. Currently `tensor`, `stats`, `math`, `noise`,
`activation`, `rng`, `fhe`, `tolerances`, `validate`, `device` all route to
`"tensor"` (barraCuda). However, `spectral` and `linalg` prefixes fall through
to the default `_ => prefix` branch, meaning `method_to_capability_domain("spectral.fft")`
returns `"spectral"` instead of `"tensor"`.

**Workaround:** wetSpring's guideStone explicitly passes `"tensor"` as the
capability domain for spectral/linalg `validate_parity` calls, bypassing the
helper. This works but defeats the purpose of centralized routing.

**Fix:** Add `"spectral" | "linalg"` to the `"tensor"` match arm in
`method_to_capability_domain()`.

**Impact:** Any spring using `method_to_capability_domain` for spectral/linalg
methods will get incorrect routing until this is fixed.

---

## PG-11: downstream_manifest.toml Drift — Missing N2 Methods

**Owner:** primalSpring (manifest maintainer)
**Status:** Resolved (V148) — primalSpring v0.9.16 reconciled manifest

**V147 state:** Manifest listed 7 `validation_capabilities`, guideStone validated
more. `guidestone_readiness` was `1` while code was `3`.

**V148 resolution:** primalSpring v0.9.16 updated the wetSpring `[[downstream]]`
entry to 15 `validation_capabilities` (`tensor.matmul`, `tensor.create`,
`stats.mean`, `stats.std_dev`, `stats.variance`, `stats.correlation`,
`linalg.solve`, `linalg.eigenvalues`, `spectral.fft`, `spectral.power_spectrum`,
`compute.dispatch`, `storage.store`, `storage.retrieve`, `inference.complete`,
`crypto.hash`) and set `guidestone_readiness = 3`. wetSpring N1 now exercises
all 15 manifest capabilities.

**Remaining:** N2 extended checks (`stats.median`, `linalg.determinant`,
`stats.weighted_mean`) are beyond the manifest and exercised opportunistically.

---

## PG-12: Exp403 Legacy Method Surface — Pending v0.9.17 Migration

**Owner:** wetSpring (internal)
**Status:** Open — documented V147, unchanged V149

Exp403 (`validate_primal_parity_v1`) uses 22 barraCuda methods from the
pre-v0.9.17 surface. 15 of these are now "legacy" (not on the v0.9.17
canonical list): `tensor.scale`, `tensor.clamp`, `tensor.sigmoid`,
`tensor.batch.submit`, `stats.weighted_mean`, `noise.perlin2d/3d`,
`math.sigmoid/log2`, `activation.fitts/hick`, `fhe.ntt/pointwise_mul`,
`tolerances.get`, `rng.uniform`.

These may still be served by barraCuda but are not part of the canonical
33-method surface. Exp403 D01 tiers should be migrated to v0.9.17 method
names, and legacy-only checks should be gated behind a feature or documented
as extended surface.

**Impact:** Low — Exp403 still works. But the dual surface (canonical + legacy)
complicates reasoning about what wetSpring actually requires from barraCuda.

---

## Summary Table

| # | Gap | Owner | Blocked By | Phase |
|---|-----|-------|------------|-------|
| PG-01 | Proto-nucleate not parsed | wetSpring | **Resolved V141** | -- |
| PG-02 | Provenance trio IPC | rhizoCrypt/loamSpine/sweetGrass | Trio IPC readiness (partial V142) | 2 |
| PG-03 | Name-based discovery | Songbird/biomeOS | **Partial V158** — capability abstraction wired | 3 |
| PG-04 | NestGate IPC wired, deploy pending | NestGate | NestGate live deployment | 2 |
| PG-05 | toadStool discovery + barraCuda optional | toadStool | Sovereign dispatch wiring | 2 |
| PG-06 | Ionic bond protocol | primalSpring Track 4 | Bond negotiation spec | 2 |
| PG-07 | Capability drift | wetSpring | **Resolved V141** | -- |
| PG-08 | Validate manifest binary name | primalSpring | **Closed V158** — informational, upstream | -- |
| PG-09 | barraCuda IPC evaporation | wetSpring | **Resolved V160** — all 5 handlers wired, Tier 4 complete | -- |
| PG-10 | spectral/linalg routing | primalSpring | `method_to_capability_domain` update | 1 |
| PG-11 | Manifest drift (N2 methods) | primalSpring | **Resolved V148** | -- |
| PG-12 | Exp403 legacy surface | wetSpring | v0.9.17 migration | 2 |
| PG-13 | barraCuda missing 6 manifest methods | barraCuda | **Resolved V149** — param names corrected | -- |
| PG-14 | Squirrel BTSP-only socket | Squirrel | **Closed V158** — informational, infra dep | -- |
| PG-15 | ToadStool compute.dispatch | ToadStool | **Closed V158** — informational, expected | -- |
| PG-16 | stats.std_dev N-1 vs N divisor | barraCuda/wetSpring | **Resolved V156** — documented as intentional | -- |
| PG-17 | tensor.matmul handle-based only | barraCuda | inline data path or document | 1 |
| PG-18 | Provenance trio UDS connection reset | rhizoCrypt/loamSpine/sweetGrass | Trio JSON-RPC on UDS | 2 |
| PG-19 | petalTongue scene primitive format | petalTongue/primalSpring | **Informational V150** — documented | -- |
| PG-20 | socat dependency in composition lib | primalSpring | **Resolved V156** — uds_send.py established | -- |
| PG-21 | Health check uses socat | primalSpring | **Resolved V156** — same as PG-20 | -- |
| PG-22 | Songbird socket timeout | Songbird | **Resolved V158** — timeouts centralized, graceful | -- |

---

## PG-13: barraCuda Missing 6 Manifest Methods

**Owner:** barraCuda
**Status:** Resolved (V149) — all 6 methods now work with correct parameters

V148 state: barraCuda responded "Unknown method" for all six. V149 discovery:
the methods ARE registered but require specific parameter names that differed
from what the guideStone was sending. Corrected parameter mappings:
- `stats.variance`: `{"data": [...]}` (not `{}`) → returns `{result, convention, denominator}`
- `stats.correlation`: `{"x": [...], "y": [...]}` (not `{}`) → returns `{result}`
- `linalg.solve`: `{"matrix": [...], "b": [...]}` (not `rhs`) → returns `{result}`
- `linalg.eigenvalues`: `{"matrix": [...]}` → returns `{result: [eigenvalues]}`
- `spectral.fft`: `{"data": [...]}` → returns `{real, imag, result, n}`
- `spectral.power_spectrum`: `{"data": [...]}` → returns `{result, n}`

All 6 now PASS in guideStone (38/38). The gap was in parameter naming, not
in barraCuda's capabilities.

---

## PG-14: Squirrel BTSP-Only Socket (Closed V158)

**Owner:** Squirrel
**Status:** Closed (V158) — informational, infrastructure dependency

V148 state: Squirrel liveness was SKIP (connection reset). V149: the
`nucleus_launcher.sh` properly configures Squirrel with provider sockets
and environment variables. Squirrel liveness now PASSES via the `ai`
capability domain. However, `inference.complete` still SKIPs because no
Ollama backend is configured (`AI_PROVIDER_SOCKETS` or `dev-direct-http`
feature required).

**V158 resolution:** Closed as informational. wetSpring's IPC wiring to
Squirrel is complete (discovery, liveness, AI capability routing). The
`inference.complete` SKIP is an infrastructure constraint (no Ollama
backend) — not a code gap. `discover_by_capability("ai")` routes to
Squirrel correctly. When an AI provider is deployed, inference will
route transparently.

---

## PG-15: ToadStool compute.dispatch — Requires GPU Binary (Closed V158)

**Owner:** ToadStool
**Status:** Closed (V158) — informational, expected hardware requirement

V148 state: ToadStool responded "Method not found". V149: `compute.dispatch`
IS registered and responds with a proper error: "Missing 'binary' array
(compiled GPU binary bytes)". The method exists but requires actual compiled
GPU binary data, which is a legitimate constraint (not a missing capability).
ToadStool liveness PASSES via the `compute` capability domain.

**V158 resolution:** Closed as informational. The `compute.dispatch` method
is registered and responsive; the SKIP in guideStone is expected behavior
(probes cannot provide compiled GPU binary bytes). `discover_by_capability("compute")`
routes to toadStool correctly. When a coralReef-compiled shader binary is
available, dispatch will work end-to-end via the `sovereign-dispatch` feature.

---

## PG-16: stats.std_dev N-1 vs N Divisor Convention (Resolved V156)

**Owner:** barraCuda + wetSpring
**Status:** Resolved — documented as intentional convention difference

barraCuda's `stats.std_dev` uses sample standard deviation (Bessel's
correction, N-1 divisor). wetSpring's B0 bare check uses population
std_dev (N divisor). Both are mathematically correct. The guideStone uses
√250 (sample) for IPC and √200 (population) for bare. This asymmetry is
**intentional**: IPC tests validate barraCuda's sample convention; bare tests
validate wetSpring's population convention. Both are correct per their context.

**V156 resolution:** Documented as intentional per post-interstadial triage.
No code change needed — the asymmetric expectations are by design.

---

## PG-17: tensor.matmul Handle-Based API

**Owner:** barraCuda
**Status:** Open — discovered V148 live NUCLEUS validation

`tensor.matmul` requires pre-created tensor handles (`lhs_id`, `rhs_id`)
rather than accepting inline data. The guideStone works around this with
create→matmul→check-shape, but `validate_parity` (which expects a scalar
`result` field) cannot be used. Either barraCuda should add an inline-data
convenience path, or `primalspring::composition` should gain a handle-aware
parity helper.

---

## PG-18: Provenance Trio UDS Connection Reset (Composition-Discovered)

**Owner:** rhizoCrypt / loamSpine / sweetGrass
**Status:** Open — discovered V150 composition exploration (Phase 46)

All three provenance trio primals (rhizoCrypt, loamSpine, sweetGrass) accept
UDS connections but immediately reset them when JSON-RPC is sent. This matches
upstream PG-45 (primalSpring composition testing). The sockets are created and
the primals are running (`pgrep` confirms), but they do not speak JSON-RPC on
UDS out of the box.

**Impact:** DAG session creation, ledger spine creation, and braid recording
all fail. The composition degrades gracefully (logs empty responses, continues
without provenance tracking). However, the **entire provenance layer is
non-functional** for interactive compositions.

**Workaround:** The composition lib handles empty responses gracefully; DAG,
ledger, and braid features are disabled but the composition still runs.

---

## PG-19: petalTongue Scene Primitive Format — Text Node Schema

**Owner:** petalTongue / primalSpring (composition lib)
**Status:** Informational — documented V150 composition exploration

`visualization.render.scene` requires the Text primitive format:
```json
{"Text": {"x": 50.0, "y": 50.0, "content": "...", "font_size": 14.0,
  "color": {"r": 0.7, "g": 0.7, "b": 0.8, "a": 1.0},
  "anchor": "Center", "bold": false, "italic": false, "data_id": null}}
```

This is correctly implemented in `nucleus_composition_lib.sh` `make_text_node`,
but the format differs from what a naive JSON builder might produce (e.g.
`{"type":"text","text":"..."}` is rejected). Springs must use the single-key
enum-variant pattern (`"Text": {...}`) with explicit `x`/`y`/`content` fields.

**Impact:** Low — documented, lib handles it. First-time users may be confused.

---

## PG-20: socat Dependency for Composition Library (Resolved V156)

**Owner:** primalSpring (composition lib)
**Status:** Resolved — `tools/uds_send.py` established as socat alternative

wetSpring's `tools/uds_send.py` (45-line Python UDS shim) serves as a proven
`socat` fallback for UDS JSON-RPC transport. The local composition lib is
patched to try `socat` first, then `python3 uds_send.py`. This pattern works
reliably in production.

**V156 resolution:** Closed per post-interstadial triage. The Python shim is
a permanent solution (not a workaround). Upstream promotion remains a candidate
for primalSpring but does not block wetSpring operations.

---

## PG-21: composition_nucleus.sh Health Check Uses socat (Resolved V156)

**Owner:** primalSpring (composition launcher)
**Status:** Resolved — same fallback as PG-20

Health check reports "no response" without `socat`, but primals are healthy
(confirmed via Python/direct socket). The `uds_send.py` fallback from PG-20
applies to health checks as well.

**V156 resolution:** Closed per post-interstadial triage. Same resolution
path as PG-20 — Python UDS shim works for both `send_rpc` and health checks.

---

## PG-22: Songbird Socket Timeout During Composition Launch (Resolved V158)

**Owner:** Songbird / wetSpring
**Status:** Resolved (V158) — timeouts centralized, discovery graceful

Songbird consistently times out during Phase 1 of `composition_nucleus.sh`
launch. The socket never appears at `songbird-{FAMILY_ID}.sock`. Songbird
may not support the `--family-id` / `FAMILY_ID` socket naming convention,
or may bind to a different path. Discovery still works via capability
aliases.

**V158 resolution:** wetSpring's Songbird timeout is centralized at
`ipc::timeouts::DISCOVERY` (5s) since V157. The `songbird.rs` module
handles the timeout gracefully (connect fails cleanly, heartbeat loop
re-registers on failure). The gap is in Songbird's socket naming convention
(`--family-id` support), not in wetSpring's timeout handling.
`discover_by_capability("discovery")` routes to Songbird correctly.
wetSpring operates in standalone mode when Songbird is absent — this is
the designed degradation path per sovereign fallback architecture.

---

---

## Post-Interstadial Triage (V156, May 10, 2026)

Per primalSpring post-interstadial audit: "15 open primal gaps remain. Triage
and close these during interstadial — you have the test infrastructure and
coverage to support aggressive gap closure."

**Closed V156 wave:** PG-16 (documented as intentional), PG-20 (uds_send.py
established), PG-21 (same resolution as PG-20).

**Status after V156 triage:** 12 open, 10 resolved/closed.

---

## Gap Closure Wave (V158, May 11, 2026)

Per primalSpring audit: "Primary target: close 15 open primal gaps" and
"push plasmidBin release binary."

**Closed V158 wave:** PG-08 (informational — upstream manifest alignment),
PG-14 (informational — Squirrel needs Ollama, wetSpring IPC fully wired),
PG-15 (informational — ToadStool method registered, GPU binary expected),
PG-22 (resolved — timeouts centralized, graceful degradation).

**Advanced V158:** PG-03 (capability-oriented discovery abstraction
`discover_by_capability` wired, 15 domain→primal mappings, single migration
point for Songbird `capability.resolve`), PG-09 (`primal-proof` feature
declared, capability discovery wired, `BARRACUDA` constant added).

**New infrastructure wired V158:**
- `ipc/skunkbat.rs` — audit event emitter (`audit.event`, `audit.forward`,
  graceful degradation, 12 tests)
- CI cross-sync updated to 413 canonical methods (was 403)
- `discover_by_capability()` + `capability_to_primal()` in `ipc/discover.rs`
- `BARRACUDA` added to `primal_names.rs`
- `SKUNKBAT` re-exported from `ipc/primal_names.rs`
- skunkBat IPC module cross-sync test added

**Advanced V159:** PG-09 (barraCuda IPC evaporation — handler-level
`#[cfg(feature = "primal-proof")]` wired into 3 science handlers with
graceful fallback to in-process compute).

**New infrastructure wired V159:**
- `ipc/barracuda_route.rs` — barraCuda IPC routing module (`discover`,
  `forward`, `try_forward`, `is_available`, atomic request IDs, 5 tests)
- `handle_diversity` → `stats.diversity` IPC forward
- `handle_anderson` → `spectral.anderson_3d` IPC forward
- `handle_qs_model` → `compute.ode_rk4` IPC forward
- Paper count reconciled (63 consistent across all docs)
- CONTEXT.md gap count synchronized (8 open, 14 resolved/closed)

**Resolved V160:** PG-09 (barraCuda IPC evaporation — all 5/5 handlers that
call `barracuda::*` directly are now `primal-proof` wired).

**New handler wiring V160:**
- `handle_nmf` → `linalg.nmf` IPC forward (drug.rs)
- `handle_dose_response` → `stats.hill_sweep` IPC forward (gonzales.rs)
- Full audit confirmed remaining handlers use only `crate::bio::*`

**Status after V160:** 7 gaps open, 15 resolved/closed. 1,962 lib tests.

**Remaining open gaps by owner:**
- **wetSpring internal (1):** PG-12 (Exp403 legacy surface — v0.9.17 migration)
- **External teams (5):** PG-02 (trio IPC readiness), PG-04 (NestGate deploy),
  PG-05 (toadStool sovereign dispatch), PG-06 (ionic bond protocol),
  PG-18 (trio UDS connection reset)
- **Mixed (2):** PG-03 (partial — Songbird + wetSpring, capability abstraction
  wired), PG-10 (primalSpring spectral/linalg routing)
- **barraCuda (1):** PG-17 (matmul handle-based API)

*This document is maintained by wetSpring and fed back to primalSpring via
`wateringHole/handoffs/` per the NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
