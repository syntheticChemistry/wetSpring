# wetSpring — Primal Composition Gaps

Gaps discovered during primal composition validation (Exp400 and IPC
integration). Each gap is handed back to primalSpring for ecosystem-wide
refinement per `NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.

Last updated: 2026-04-11 (V142 Wire Standard + composition evolution)

---

## Gap Tracking Convention

- Gaps are **composition-specific**: missing primal capabilities, IPC
  misalignment, proto-nucleate drift, or bonding protocol holes.
- Architectural gaps that are not composition-related live in
  [GAPS.md](../GAPS.md) (7 documented as of V140).
- Each gap references the proto-nucleate graph:
  `primalSpring/graphs/downstream/wetspring_lifescience_proto_nucleate.toml`

---

## PG-01: Proto-Nucleate Not Parsed at Build Time (Resolved V141)

**Owner:** wetSpring (internal)
**Status:** Resolved — `niche::tests::proto_nucleate_node_names_match_niche_dependencies`

`niche.rs` now reads `primalSpring/graphs/downstream/wetspring_lifescience_proto_nucleate.toml`
at test time via `std::fs::read_to_string` and cross-checks all niche dependency names,
the `wetspring` application node, `pattern = "proto_nucleate"`, and `owner = "wetSpring"`
against the canonical graph. Drift is caught in CI.

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

## Summary Table

| # | Gap | Owner | Blocked By | Phase |
|---|-----|-------|------------|-------|
| PG-01 | Proto-nucleate not parsed | wetSpring | **Resolved V141** | -- |
| PG-02 | Provenance trio IPC | rhizoCrypt/loamSpine/sweetGrass | Trio IPC readiness (partial V142) | 2 |
| PG-03 | Name-based discovery | Songbird/biomeOS | capability.resolve | 3 |
| PG-04 | NestGate not wired | NestGate | NestGate IPC readiness | 2 |
| PG-05 | toadStool compute IPC | toadStool | NUCLEUS deployment | 3 |
| PG-06 | Ionic bond protocol | primalSpring Track 4 | Bond negotiation spec | 2 |
| PG-07 | Capability drift | wetSpring | Resolved V141 | -- |

---

*This document is maintained by wetSpring and fed back to primalSpring
(`docs/PRIMAL_GAPS.md`) and `infra/wateringHole/handoffs/` per the
NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
