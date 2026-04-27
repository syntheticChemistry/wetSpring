<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V145 — Primal Proof Tier 2 (IPC-WIRED) Handoff

**Date:** 2026-04-17
**From:** wetSpring
**To:** primalSpring, barraCuda, biomeOS, NestGate, Squirrel, BearDog, toadStool

---

## Summary

V145 adds the Tier 2 (IPC-WIRED) primal proof validation harness — **Exp403**
(`validate_primal_parity_v1`). This binary calls live NUCLEUS primals over UDS
sockets and compares results against local Rust baselines. When a primal socket
is absent, checks SKIP honestly (CI-safe).

This is the first binary in wetSpring that calls primals by capability over
real Unix domain sockets rather than in-process library dispatch. It bridges
Level 4 (barraCuda GPU) to Level 5 (Primal Composition Proof).

---

## What Changed

### New: Exp403 — `validate_primal_parity_v1`

Tier 2 IPC-WIRED validation binary with 6 domains across 5 primals:

| Domain | Primal | Methods Tested |
|--------|--------|----------------|
| D01 | barraCuda | stats.mean, stats.std_dev, stats.weighted_mean, tensor.matmul, rng.uniform |
| D02 | barraCuda | health.liveness, capabilities.list, identity.get |
| D03 | NestGate | storage.store, storage.retrieve |
| D04 | Squirrel | inference.complete |
| D05 | BearDog | crypto.hash |
| D06 | toadStool | compute.dispatch |

Exit semantics: 0 = pass, 1 = fail, 2 = all skipped (no primals found).

### Updated: `niche::CONSUMED_CAPABILITIES`

Now declares 22 barraCuda domain math methods consumed over IPC:
`tensor.matmul`, `tensor.create`, `tensor.add`, `tensor.scale`,
`tensor.clamp`, `tensor.reduce`, `tensor.sigmoid`, `tensor.batch.submit`,
`stats.mean`, `stats.std_dev`, `stats.weighted_mean`, `compute.dispatch`,
`noise.perlin2d`, `noise.perlin3d`, `math.sigmoid`, `math.log2`,
`activation.fitts`, `activation.hick`, `fhe.ntt`, `fhe.pointwise_mul`,
`tolerances.get`, `rng.uniform`.

### New: PG-09 — barraCuda IPC Evaporation Surface

Documents the library → IPC migration path for all domain math calls.
The gap is in wetSpring's wiring, not in barraCuda (which already exposes
all 32 JSON-RPC methods).

---

## Verification

- `cargo clippy --features json,ipc,facade -p wetspring-barracuda -- -D warnings` → zero warnings
- `cargo test --features ipc -p wetspring-barracuda --lib` → 1,592 passed
- `cargo test --features ipc -p wetspring-barracuda --test ipc_roundtrip` → 18 passed

---

## Three-Tier Primal Composition Pattern

```
Tier 1: LOCAL_CAPABILITIES
  Exp401 (IPC parity) + Exp402 (niche parity)
  In-process dispatch() simulating JSON-RPC — always green in CI

Tier 2: IPC-WIRED  ← NEW (V145)
  Exp403 (primal parity)
  Live UDS calls to primals with check_skip for absent sockets
  CI: exit 2 (skip) when no primals deployed

Tier 3: FULL NUCLEUS  ← NEXT
  biomeos deploy --graph wetspring_science_nucleus.toml
  All primals from plasmidBin ecobins on clean machine
  Spring validates externally against the running NUCLEUS
```

---

## What primalSpring Needs to Know

1. **wetSpring now calls barraCuda by capability over IPC.** The 22 consumed
   capabilities are declared in `niche.rs` and verified by Exp402 D03.

2. **PG-09 documents the evaporation path.** Every `barracuda::stats::mean()`
   library call in production code is an IPC migration candidate. The library
   dep stays for Level 2 comparison.

3. **Tier 3 readiness requires plasmidBin ecobins.** Once `biomeos deploy`
   can stand up the proto-nucleate from ecobins, Exp403 becomes a Tier 3
   validator by changing the socket discovery from env vars to NUCLEUS paths.

---

## Composition Gaps (7 open, 2 resolved)

| # | Gap | Status |
|---|-----|--------|
| PG-01 | Proto-nucleate not parsed | Resolved V141 |
| PG-02 | Provenance trio IPC | Partial V142 |
| PG-03 | Name-based discovery | Structural |
| PG-04 | NestGate not wired | Declared |
| PG-05 | toadStool compute IPC | Declared |
| PG-06 | Ionic bond protocol | Metadata only |
| PG-07 | Capability drift | Resolved V141 |
| PG-08 | Validate manifest binary name | Informational |
| PG-09 | barraCuda IPC evaporation | **New V145** |

---

*This handoff follows the `{SPRING}_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`
convention per `wateringHole/` naming standard.*
