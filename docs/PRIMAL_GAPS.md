# wetSpring — Primal Composition Gaps

Gaps discovered during primal composition validation (Exp400 and IPC
integration). Each gap is handed back to primalSpring for ecosystem-wide
refinement per `NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.

Last updated: 2026-04-18 (V147 — guideStone Level 3, bare mode certified, N2 expanded to v0.9.15 surface)

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
**Status:** In progress — guideStone Level 3, CONSUMED_CAPABILITIES aligned to v0.9.15

barraCuda is a full ecobin primal. The v0.9.15 canonical surface defines 33
JSON-RPC methods (TENSOR 9, STATS 9, COMPUTE 4, SPECTRAL 3, LINALG 6,
HEALTH 2). Today, wetSpring links barraCuda as a Rust library dependency
(`path = "../../../primals/barraCuda"`) and calls math in-process. For the
primal proof (Level 5), domain math must migrate to IPC.

**What exists (V147):**
- `niche::CONSUMED_CAPABILITIES` declares full v0.9.15 canonical surface (33
  methods) plus 15 legacy Exp403 methods pending migration
- `wetspring_guidestone` binary (Level 3): bare mode certified (9/9, exit 2),
  N2 expanded with stats.variance, stats.median, stats.correlation,
  linalg.determinant, linalg.eigenvalues, spectral.fft
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
**Status:** Open — discovered during V147 guideStone expansion

wetSpring's `downstream_manifest.toml` entry lists 7 `validation_capabilities`:
`tensor.matmul`, `stats.mean`, `compute.dispatch`, `storage.store`,
`storage.retrieve`, `inference.complete`, `crypto.hash`.

The guideStone N2 layer now also validates `stats.std_dev`, `stats.variance`,
`stats.median`, `stats.correlation`, `linalg.determinant`, `linalg.eigenvalues`,
`spectral.fft`, and `stats.weighted_mean`. These are not in the manifest.

Additionally, `guidestone_readiness` in the manifest is `1` while
`niche::GUIDESTONE_READINESS` is now `3`. These should be reconciled.

**Impact:** biomeOS composition completeness checks may under-report wetSpring's
actual capability requirements if only the manifest is consulted.

---

## PG-12: Exp403 Legacy Method Surface — Pending v0.9.15 Migration

**Owner:** wetSpring (internal)
**Status:** Open — documented V147

Exp403 (`validate_primal_parity_v1`) uses 22 barraCuda methods from the
pre-v0.9.15 surface. 15 of these are now "legacy" (not on the v0.9.15
canonical list): `tensor.scale`, `tensor.clamp`, `tensor.sigmoid`,
`tensor.batch.submit`, `stats.weighted_mean`, `noise.perlin2d/3d`,
`math.sigmoid/log2`, `activation.fitts/hick`, `fhe.ntt/pointwise_mul`,
`tolerances.get`, `rng.uniform`.

These may still be served by barraCuda but are not part of the canonical
33-method surface. Exp403 D01 tiers should be migrated to v0.9.15 method
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
| PG-11 | Manifest drift (N2 methods) | primalSpring | Manifest update | 1 |
| PG-12 | Exp403 legacy surface | wetSpring | v0.9.15 migration | 2 |

---

*This document is maintained by wetSpring and fed back to primalSpring via
`wateringHole/handoffs/` per the NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
