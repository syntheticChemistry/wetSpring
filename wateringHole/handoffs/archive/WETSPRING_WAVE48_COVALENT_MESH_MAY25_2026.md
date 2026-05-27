# wetSpring Wave 48 — Covalent Mesh Deployment Conformance

**Date**: May 25, 2026
**From**: wetSpring (southGate)
**To**: primalSpring, downstream springs
**Version**: V185 (commits `7f18b68` + `c332041`)

---

## Sound Off

| Field | Value |
|-------|-------|
| **Gate** | southGate |
| **Hardware** | AMD Ryzen 7 5800X3D 8-Core, 128GB DDR4, NVIDIA RTX 4060 |
| **Secondary** | strandGate (Dual EPYC 64-core, 256GB ECC, RTX 3090 + RX 6950 XT) |
| **Composition** | Node Atomic (9/9 primals, exp091+exp094 PASS) |
| **NUCLEUS** | operational |
| **Songbird federation** | TCP :7700 live, `discovery.peers` responding |
| **Cell graph** | `plasmidBin/cells/wetspring_cell.toml` (13 nodes) |

## Deployment Conformance (DEPLOYMENT_BEHAVIOR_STANDARD v1.0)

| Requirement | Status |
|-------------|--------|
| `--socket` / `--port` / `--family-id` | Done (V185) |
| `server` alias | Done (V185) |
| `health.liveness` → `{"status":"alive"}` | Done (V185b) |
| `lifecycle.status` → name/version/status/uptime | Done (V185b) |
| `primal.announce` on startup | Pending — Songbird `discovery.register` only |
| Bonding on wire | Done (V185b) — 6 `bonding.*` methods dispatched |

## Code Changes (V185b)

- `health.liveness` returns `{"status":"alive","alive":true,"primal":"wetspring"}` (backward compat retained)
- `lifecycle.status` handler: `{"primal":"wetspring","version":"0.1.0","status":"running","uptime_s":N}`
- `bonding.*` wired into `dispatch.rs`: propose, accept, reject, status, terminate, list
- Capability surface: 22 domains, 45 dispatch methods (was 21/38), 50 niche, 59 consumed
- `capability_registry.toml`: +`lifecycle.status`
- CI cross-sync: 7/7 PASS (added `ledger.`/`lifecycle.`/`bonding.` domain prefixes)

## primalSpring v0.9.28 Absorption

- 52 scenarios (3 new: coordination-api, health-lifecycle-surface, crypto-identity-surface)
- 458/458 methods exercised (100% coverage, was 70%)
- 787 tests
- `downstream_manifest.toml` fixes confirmed: `guidestone_binary = "wetspring"`, `guidestone_readiness = 5`

## Gap Status Updates

| Gap | Previous | Current |
|-----|----------|---------|
| PG-02 (provenance trio) | Deployment-only | Deployed on southGate, explicit roundtrip pending |
| PG-04 (NestGate) | Deployment-only | Deployed on southGate, roundtrip pending |
| PG-06 (ionic bonding) | Closed/deferred | Locally wired — 6 methods on JSON-RPC wire |
| WS-4 (petalTongue WASM) | Not built | Upstream IMPLEMENTED, wetSpring adoption pending |

## Remaining Upstream Asks

1. **loamSpine tokio panic**: `Cannot start a runtime from within a runtime` on health probe — upstream bug
2. **Health probe timing**: rhizoCrypt, sweetGrass, toadStool sometimes exceed 8s timeout in nucleus health sweep
3. **`primal.announce` vs `discovery.register`**: wetSpring still uses Songbird `discovery.register`; standard says `primal.announce` to biomeOS Neural API. Need guidance on migration path.

## Ecosystem Pull Summary

Pulled all 28 repos (May 25). Key upstream changes absorbed:
- barraCuda: `signal.rs`/`stats.rs` split from `math.rs`, `neural_announce.rs`, 87 methods
- coralReef: `ptx_emit/ray_query.rs`, deep debt cleanup
- sweetGrass: `neural_announce.rs`
- petalTongue: `petal-tongue-wasm` 8+ wasm_bindgen exports
- loamSpine: benchScale roundtrip harness (51 validations, 43 methods)
- toadStool: S274 `GuestLoadPolicy` yield-to-owner
- plasmidBin: `start_primal.sh` simplified post-convergence (`9231b24`)
