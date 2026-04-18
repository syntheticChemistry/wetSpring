# wetSpring GAPS — Known Architectural Gaps

Intentional gaps scaffolded by the wetSpring science NUCLEUS deployment.
Each gap is documented so it feeds back to the owning team via wateringHole
and primalSpring evolution tracking.

Last updated: 2026-04-17 (V145 — no architectural gaps closed, all 7 remain
as external team dependencies. V145: primal proof Tier 2 (IPC-WIRED), PG-09
barraCuda IPC evaporation surface. V144: composition validation tier. V143:
deploy graph canonical. Internal code quality green.)

---

## 1. Ionic Contract Negotiation

**Owner:** primalSpring Track 4 (`BondingConstraint + BondingPolicy`)

**Status:** Scaffolded — bonding metadata declared, no negotiation protocol.

The facade declares its bonding capabilities in `bonding_metadata.json` and
exposes them via `GET /api/v1/system/composition`. However, there is no
automated protocol for:

- Establishing a new ionic bond (capability scoping, duration, attribution)
- Modifying an existing bond (scope expansion/reduction)
- Terminating a bond (cleanup, final provenance seal)
- Verifying a bond's validity during request dispatch

primalSpring's `BondType::Ionic` and `TrustModel::Contractual` define the
abstract model. The concrete negotiation protocol (handshake, contract
serialization, mutual verification) is the gap.

**Impact:** External researchers cannot self-service ionic bond creation.
Manual setup required via ecosystem maintainers.

---

## 2. Cross-Spring Data Exchange via RootPulse

**Owner:** RootPulse team

**Status:** Not started — NestGate stores locally, no remote pull protocol.

The `data.fetch.*` handlers store fetched data in NestGate with BLAKE3 hashes
and provenance sessions. The `vault.*` handlers provide consent-gated storage.
However:

- No protocol for another spring's NUCLEUS to pull provenance-wrapped data
  subsets from this spring's NestGate
- No RootPulse sync mechanism for cross-spring provenance chain exchange
- No differential sync (only full fetch or nothing)

**Impact:** Each spring operates as a silo. Ionic bonds can declare shared
capabilities but cannot exchange stored data with provenance continuity.

---

## 3. Public Chain Anchor

**Owner:** loamSpine team (Paper 20 Section 6 — marked "Needed")

**Status:** Not started — provenance braids exist but have no public anchor.

The provenance trio produces:
- rhizoCrypt ephemeral DAG sessions
- loamSpine immutable ledger commits
- sweetGrass semantic braids (PROV-O aligned)

Missing: a mechanism to anchor these braids to a public verifiable ledger
(timestamping service, blockchain anchor, or similar). Without this, the
provenance chain is verifiable only within the ecosystem trust boundary.

**Impact:** External verification of provenance requires trusting the
ecosystem's own ledger. Public anchoring would provide independent
verification.

---

## 4. petalTongue Client-Side WASM

**Owner:** petalTongue team

**Status:** Server-side SVG rendering works; client WASM not yet built.

The facade currently renders grammar expressions server-side via petalTongue
RPC, returning SVG strings. The endgame (Phase 3 of web deployment) is:

- Compile petalTongue grammar engine to WASM
- Ship to browser as a client-side renderer
- Eliminate server round-trip for visualization
- Enable offline rendering with cached data

**Impact:** All grammar rendering requires live HPC connection. No offline
interactive visualization. Plotly.js remains the primary client renderer.

---

## 5. ludoSpring / esotericWebb Interactive Composition

**Owner:** ludoSpring team, esotericWebb team

**Status:** Not started — explorer uses Plotly.js; no game/composition engine.

The current explorer is a Plotly.js chart viewer with parameter sliders.
The endgame is:

- **ludoSpring:** Interactive scenes with real physics, navigation, and
  exploration (gaming engine for science)
- **esotericWebb:** Composition contracts for combining visualizations,
  data streams, and interaction modes
- **petalTongue:** Grammar-of-graphics rendering within ludoSpring scenes

Together these would replace the static chart explorer with a living,
navigable data environment.

**Impact:** Data exploration is limited to 2D charts. No 3D navigation,
no physics simulation overlays, no interactive data manipulation.

---

## 6. hotSpring Physics Simulations

**Owner:** hotSpring team

**Status:** Not started — same NUCLEUS pattern, different science domain.

hotSpring would use the same deploy graph pattern proven by wetSpring to
run real physics simulations:

- Anderson localization in physical systems (not just biological analogy)
- Wave propagation in disordered media
- Thermal transport in heterogeneous materials

The NUCLEUS deployment pattern, facade architecture, and provenance wrapping
are directly transferable.

**Impact:** Physics simulations are only available as analytical math in
wetSpring's Anderson modules. No numerical physics simulation engine.

---

## 7. Radiating Attribution Calculator

**Owner:** sweetGrass team + sunCloud team
(Paper 20 Section 6 — marked "Low, Phase 4")

**Status:** Not started.

When data flows through ionic bonds across multiple springs, attribution
must "radiate" — each downstream use must credit upstream contributors
proportionally. This requires:

- Attribution weight calculation based on derivation depth
- Credit distribution across the NFT vertex DAG
- Integration with sunCloud's economic model

**Impact:** Novel Ferment Transcript vertices record derivation chains but
do not compute or distribute attribution weights. The gAIa commons model
requires this for its economic sustainability.

---

## Summary Table

| # | Gap | Owner | Priority | Phase |
|---|-----|-------|----------|-------|
| 1 | Ionic contract negotiation | primalSpring Track 4 | High | 2 |
| 2 | Cross-spring data exchange | RootPulse | High | 2 |
| 3 | Public chain anchor | loamSpine | Medium | 3 |
| 4 | Client WASM renderer | petalTongue | Medium | 3 |
| 5 | Interactive composition | ludoSpring + esotericWebb | Low | 4 |
| 6 | Physics simulations | hotSpring | Low | 4 |
| 7 | Radiating attribution | sweetGrass + sunCloud | Low | 4 |

---

*This document is maintained by wetSpring and fed back to wateringHole
for ecosystem-wide gap tracking. Each gap discovered during the science
NUCLEUS deployment informs the primal teams responsible for resolution.*
