# wetSpring — Primal Composition Gaps

Gaps discovered during primal composition validation (Exp400 and IPC
integration). Each gap is handed back to primalSpring for ecosystem-wide
refinement per `NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.

Last updated: 2026-04-27 (V151 — Deep debt evolution. 15 gaps open (PG-18..22
from composition testing), 7 resolved. Provenance trio UDS reset (PG-18), petalTongue
scene format documented (PG-19), socat dependency (PG-20/21), Songbird timeout (PG-22).
V151: code quality debt cleared — zero `dyn` dispatch, `Write`-based output, shared helpers.)

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

**Owner:** Songbird team / biomeOS
**Status:** Structural — socket resolution uses primal names

`ipc/discover.rs` resolves sockets via `discover_squirrel()`,
`discover_toadstool()` etc., using canonical name strings from
`primal_names.rs`. The `by_capability` field in the proto-nucleate graph
is metadata only — runtime discovery does not query Songbird for "who
provides capability X."

**Impact:** If a capability migrates between primals, wetSpring's socket
resolution breaks. True capability-based discovery would decouple from
primal identity.

**Blocked by:** Songbird implementing `capability.resolve` → socket path
mapping, and biomeOS routing capability queries to registered providers.

---

## PG-04: NestGate Storage — Declared but Not Wired

**Owner:** NestGate team
**Status:** Declared as optional niche dependency; no IPC integration

`data.fetch.*` handlers store fetched data locally with BLAKE3 hashes.
`vault.*` handlers provide consent-gated storage. Neither path calls
NestGate over IPC for persistent cross-spring storage.

**Impact:** Data provenance terminates at local filesystem boundaries. No
cross-spring data retrieval with provenance continuity (see also
GAPS.md #2: Cross-Spring Data Exchange via RootPulse).

---

## PG-05: toadStool Compute — Discovery Helper but No Active Calls

**Owner:** toadStool team
**Status:** `discover_toadstool()` helper exists; no runtime compute dispatch

wetSpring uses barraCuda directly as a path dependency for all math/GPU
operations. toadStool is declared as a Node/compute dependency in the niche,
and the proto-nucleate lists it as a Phase 2 node with `spawn = false`.

For sovereign dispatch (coralReef native), compute requests would route
through toadStool IPC, not direct barraCuda imports. This is the expected
evolution path but not yet wired.

**Impact:** None currently — barraCuda path dependency is the correct
approach for validation springs. Gap becomes relevant at NUCLEUS deployment.

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

## PG-08: spring_validate_manifest Binary Name Inconsistency

**Owner:** primalSpring (manifest maintainer)
**Status:** Informational — discovered during V144 audit

`primalSpring/graphs/spring_validation/spring_validate_manifest.toml` uses
`spring_binary = "wetspring"` for the wetSpring entry, while
`downstream_manifest.toml` and `spring_deploy_manifest.toml` both use
`spring_binary = "wetspring_primal"`. One of these should be canonical.

**Impact:** Tooling that reads the validate manifest may invoke the wrong
binary name if it differs from the deploy/downstream manifests.

---

## PG-09: barraCuda IPC Evaporation Surface — Domain Math via IPC

**Owner:** wetSpring (internal)
**Status:** In progress — guideStone Level 4, CONSUMED_CAPABILITIES aligned to v0.9.17

barraCuda is a full ecobin primal. The v0.9.17 canonical surface defines 33
JSON-RPC methods (TENSOR 9, STATS 9, COMPUTE 4, SPECTRAL 3, LINALG 6,
HEALTH 2). Today, wetSpring links barraCuda as a Rust library dependency
(`path = "../../../primals/barraCuda"`) and calls math in-process. For the
primal proof (Level 5), domain math must migrate to IPC.

**What exists (V149):**
- `niche::CONSUMED_CAPABILITIES` declares full v0.9.17 canonical surface (33
  methods) plus 15 legacy Exp403 methods pending migration
- `wetspring_guidestone` binary (Level 4): NUCLEUS validated (38/38 pass, 4 skip,
  exit 0), bare mode (16/16 pass, exit 2), handle-based matmul, all stats/linalg/
  spectral methods passing, cross-atomic pipeline. `is_skip_error` from v0.9.17
- Exp403 (`validate_primal_parity_v1`) remains as Tier 2 IPC-WIRED validation
  with the original 22-method surface
- Socket discovery uses `ipc::discover::discover_primal()` (env var → XDG → temp)

**Evaporation candidates (library → IPC migration):**

| Library call | IPC method | Priority |
|-------------|-----------|----------|
| `barracuda::stats::mean()` | `stats.mean` | High — used in diversity, QS |
| `barracuda::stats::std_dev()` | `stats.std_dev` | High — used in diversity |
| `barracuda::stats::weighted_mean()` | `stats.weighted_mean` | Medium (legacy surface) |
| `barracuda::linalg::matmul()` | `tensor.matmul` | High — core linear algebra |
| `barracuda::ops::*` | `tensor.*` | Medium — tensor operations |
| `barracuda::dispatch::*` | `compute.dispatch` | High — GPU workloads |
| `barracuda::sample::noise_*()` | `noise.perlin2d/3d` | Low — legacy surface |

**Impact:** The library dep remains for Level 2 Rust-proof comparison and CI.
The IPC path is additive. Full evaporation happens when Tier 3 (NUCLEUS from
plasmidBin) is deployed.

**Blocked by:** Nothing — barraCuda already exposes all methods. The gap is
in wetSpring's wiring, not in barraCuda's capabilities.

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
| PG-03 | Name-based discovery | Songbird/biomeOS | capability.resolve | 3 |
| PG-04 | NestGate not wired | NestGate | NestGate IPC readiness | 2 |
| PG-05 | toadStool compute IPC | toadStool | NUCLEUS deployment | 3 |
| PG-06 | Ionic bond protocol | primalSpring Track 4 | Bond negotiation spec | 2 |
| PG-07 | Capability drift | wetSpring | **Resolved V141** | -- |
| PG-08 | Validate manifest binary name | primalSpring | Manifest alignment | 1 |
| PG-09 | barraCuda IPC evaporation | wetSpring | Nothing — wiring gap | 1 |
| PG-10 | spectral/linalg routing | primalSpring | `method_to_capability_domain` update | 1 |
| PG-11 | Manifest drift (N2 methods) | primalSpring | **Resolved V148** | -- |
| PG-12 | Exp403 legacy surface | wetSpring | v0.9.17 migration | 2 |
| PG-13 | barraCuda missing 6 manifest methods | barraCuda | **Resolved V149** — param names corrected | -- |
| PG-14 | Squirrel BTSP-only socket | Squirrel | **Partial V149** — liveness ok, inference needs Ollama | 2 |
| PG-15 | ToadStool compute.dispatch missing | ToadStool | **Updated V149** — registered, needs GPU binary | 2 |
| PG-16 | stats.std_dev N-1 vs N divisor | barraCuda/wetSpring | document convention | 1 |
| PG-17 | tensor.matmul handle-based only | barraCuda | inline data path or document | 1 |
| PG-18 | Provenance trio UDS connection reset | rhizoCrypt/loamSpine/sweetGrass | Trio JSON-RPC on UDS | 2 |
| PG-19 | petalTongue scene primitive format | petalTongue/primalSpring | **Informational V150** — documented | -- |
| PG-20 | socat dependency in composition lib | primalSpring | Python shim workaround | 1 |
| PG-21 | Health check uses socat | primalSpring | Same fallback as send_rpc | 1 |
| PG-22 | Songbird socket timeout | Songbird | Socket naming convention | 2 |

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

## PG-14: Squirrel BTSP-Only Socket

**Owner:** Squirrel
**Status:** Partially resolved (V149) — liveness PASS, inference SKIP

V148 state: Squirrel liveness was SKIP (connection reset). V149: the
`nucleus_launcher.sh` properly configures Squirrel with provider sockets
and environment variables. Squirrel liveness now PASSES via the `ai`
capability domain. However, `inference.complete` still SKIPs because no
Ollama backend is configured (`AI_PROVIDER_SOCKETS` or `dev-direct-http`
feature required).

**Remaining:** inference.complete requires an Ollama instance or AI provider
socket. This is an infrastructure dependency, not a code gap.

---

## PG-15: ToadStool compute.dispatch — Requires GPU Binary

**Owner:** ToadStool
**Status:** Updated (V149) — method registered, needs compiled binary input

V148 state: ToadStool responded "Method not found". V149: `compute.dispatch`
IS registered and responds with a proper error: "Missing 'binary' array
(compiled GPU binary bytes)". The method exists but requires actual compiled
GPU binary data, which is a legitimate constraint (not a missing capability).
ToadStool liveness PASSES via the `compute` capability domain.

**Impact:** compute.dispatch SKIP in guideStone is expected — the noop probe
cannot provide compiled GPU binary bytes. A real dispatch test would require
a coralReef-compiled shader binary.

---

## PG-16: stats.std_dev N-1 vs N Divisor Convention

**Owner:** barraCuda + wetSpring
**Status:** Open — discovered V148 live NUCLEUS validation

barraCuda's `stats.std_dev` uses sample standard deviation (Bessel's
correction, N-1 divisor). wetSpring's B0 bare check uses population
std_dev (N divisor). Both are mathematically correct but test different
conventions. The guideStone now uses √250 (sample) for IPC and √200
(population) for bare — this asymmetry should be documented as intentional.

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

## PG-20: socat Dependency for Composition Library

**Owner:** primalSpring (composition lib)
**Status:** Open — discovered V150 composition exploration

`nucleus_composition_lib.sh` `send_rpc` requires `socat` for UDS JSON-RPC
transport. On systems without `socat` (no sudo access, minimal containers),
the composition cannot run without a shim.

**Workaround:** wetSpring added `tools/uds_send.py` — a 45-line Python UDS
shim used as a `socat` fallback. The local lib is patched to try `socat`
first, then `python3 uds_send.py`. Candidate for upstream promotion.

---

## PG-21: composition_nucleus.sh Health Check Uses socat

**Owner:** primalSpring (composition launcher)
**Status:** Open — discovered V150 composition exploration

`composition_nucleus.sh` health check always reports "no response" when
`socat` is unavailable, even though primals are healthy (confirmed via
Python/direct socket). The launcher should use the same fallback strategy
as `send_rpc`.

---

## PG-22: Songbird Socket Timeout During Composition Launch

**Owner:** Songbird
**Status:** Open — discovered V149/V150 composition launches

Songbird consistently times out during Phase 1 of `composition_nucleus.sh`
launch. The socket never appears at `songbird-{FAMILY_ID}.sock`. Songbird
may not support the `--family-id` / `FAMILY_ID` socket naming convention,
or may bind to a different path. Discovery still works via capability
aliases.

---

*This document is maintained by wetSpring and fed back to primalSpring via
`wateringHole/handoffs/` per the NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
