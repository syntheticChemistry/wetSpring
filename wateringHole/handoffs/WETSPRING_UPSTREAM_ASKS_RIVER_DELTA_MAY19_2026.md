# wetSpring Upstream Asks — River Delta Audit Response

**Date:** May 19, 2026
**From:** wetSpring (southGate)
**To:** primalSpring (coordination), biomeOS, loamSpine, petalTongue
**Priority:** Upstream asks — no wetSpring implementation until specs land
**License:** AGPL-3.0-or-later

---

## Context

primalSpring River Delta audit (May 19, 2026) identified 4 upstream gaps
that wetSpring reported but cannot resolve locally. This document formalizes
the upstream asks with acceptance criteria.

Barrick 2009 is SEALED. Tenaillon 2016 batch 0 validated (2/5 clones).
WS-11 v2 calibration deployed locally (GPU min_depth, MAPQ filtering,
±5bp window matching, duplicate removal).

---

## WS-1: Ionic Contract Negotiation

**Owner:** primalSpring Track 4
**Priority:** HIGH

**Current state:** `BondingConstraint` + `BondingPolicy` declared;
`GET /api/v1/system/composition` exposes metadata. No automated protocol.

**Ask:** Define the negotiation protocol spec:
- Handshake sequence (discovery → propose → accept/reject → active)
- Contract serialization format (JSON-RPC or protobuf)
- Mutual verification (bond validity check during dispatch)
- Bond lifecycle: establish, modify, terminate, final provenance seal

**Acceptance:** wetSpring can programmatically create an ionic bond with
another spring's NUCLEUS for data exchange without manual maintainer
intervention.

---

## WS-2: Cross-Spring Data Exchange (RootPulse)

**Owner:** biomeOS (orchestration) + provenance trio
**Priority:** HIGH

**Current state:** NestGate stores locally. biomeOS has `nest_sync.toml`
graph spec (6-node sequential: `dag.checkout_slice` → … → `session.commit`).
No live orchestration wiring.

**Ask:** biomeOS implements `rootpulse.sync` (or equivalent) as a
NeuralAPI composition graph:
- `signal.dispatch` → trio pipeline → cross-spring braid subset delivery
- Minimum viable: one spring requests a braid subset from another spring's
  NestGate with provenance continuity
- Differential sync (not full fetch)

**Acceptance:** wetSpring can pull a Barrick 2009 braid from another
spring's NestGate via `rootpulse.sync` with provenance chain intact.

---

## WS-3: Public Chain Anchor

**Owner:** loamSpine team
**Priority:** MEDIUM

**Current state:** loamSpine has `anchor.publish` / `anchor.verify` +
`EntryType::PublicChainAnchor` — records receipts but no actual chain
submission. Provenance verifiable only within ecosystem trust boundary.

**Ask:** loamSpine explores public timestamping options:
- RFC 3161 TSA (simplest, no blockchain dependency)
- Blockchain anchor (e.g. Ethereum Attestation Service)
- Spec-level first — implementation timeline open

**Acceptance:** A sealed Barrick 2009 braid hash can be verified by an
entity outside the ecoPrimals trust boundary.

---

## WS-4: petalTongue Client-Side WASM

**Owner:** petalTongue team
**Priority:** MEDIUM

**Current state:** `petal-tongue-wasm` crate exists; not product-integrated.
Server-side SVG rendering works. All grammar rendering requires live HPC.

**Ask:** petalTongue scopes WASM compilation target:
- Phase 3 of web deployment
- Grammar engine compiled to WASM, shipped to browser
- Eliminates server round-trip for visualization
- Offline rendering with cached data

**Acceptance:** wetSpring explorer page renders grammar-of-graphics chart
with no RPC round-trip to HPC.

---

## Local Debt Resolved (this session)

| Item | Fix | Impact |
|------|-----|--------|
| WS-11 GPU min_depth | Wired `CallerConfig.min_depth` to `SnpCallingF64` (was hardcoded to 2) | GPU pre-filter now consistent with CPU |
| WS-11 position matching | `compare_calls` uses ±5bp window | Reveals true positive matches hidden by coordinate representation |
| WS-11 MAPQ filtering | `PileupConfig.min_mapq=10` in sovereign pipeline | Excludes low-confidence mappings from variant calling |
| WS-11 duplicate removal | `PileupConfig.skip_duplicates=true`, `skip_secondary=true` | Excludes PCR/optical duplicates and secondary/supplementary alignments |
| GPU mapping threshold | Raised from 100bp to 250bp | CPU handles all Illumina reads; GPU reserved for long-read tech |

---

*Filed by wetSpring. Routing via wateringHole to owning teams.*
