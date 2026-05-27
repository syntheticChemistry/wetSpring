# wetSpring — Wave 55: southGate Redeploy + Provenance Trio Verification

**Date:** 2026-05-26
**Version:** V188
**Gate:** southGate (5800X3D, RTX 4060 + 3090s, 128GB)
**Upstream:** primalSpring Wave 53 audit — Songbird socket hardening

---

## Summary

Redeployed NUCLEUS on southGate with Wave 53 hardened binaries. Songbird
socket crash fix confirmed. loamSpine Tokio panic fixed. Provenance trio
(PG-02/PG-04) live IPC roundtrip verified. All 22 primal gaps closed.

## Actions Taken

### 1. Binary refresh

- `plasmidBin/fetch.sh --all --force` against v2026.05.27 release
- 9/13 binaries refreshed: beardog, songbird, toadstool, barracuda, coralreef,
  nestgate, loamspine, biomeos, skunkbat
- 4 not in release (rhizocrypt, sweetgrass, petaltongue, squirrel) — existing
  binaries retained
- XDG symlinks updated to point to fresh `x86_64-unknown-linux-musl/` binaries

### 2. NUCLEUS redeploy

- Stopped previous NUCLEUS, cleaned sockets and Songbird sled DB
- Restarted with:
  ```
  SONGBIRD_FEDERATION_PORT=7700
  SONGBIRD_PEERS=192.168.1.144:7700
  NODE_ID=southgate01
  BEARDOG_NODE_ID=southgate01
  NESTGATE_JWT_SECRET=<generated>
  ```
- New env vars required by Wave 53 binaries: NODE_ID (BearDog), JWT_SECRET (NestGate)

### 3. Health status: 8/13 responding

| Primal | Status | Notes |
|--------|--------|-------|
| biomeOS | ALIVE | Neural API bootstrap |
| BearDog | ALIVE | Now requires NODE_ID env |
| Songbird | ALIVE | **Wave 53 fix confirmed — stable** |
| ToadStool | SOCKET | Socket exists, no health response |
| barraCuda | DOWN | Process crashed after launch |
| coralReef | ALIVE | Via shader.sock alias |
| NestGate | ALIVE | 66 methods, BTSP-auth gated |
| Squirrel | DOWN | Socket not at expected name |
| rhizoCrypt | ALIVE | Via permanence-nucleus01.sock |
| loamSpine | ALIVE | **Tokio panic FIXED** |
| sweetGrass | ALIVE | Braid/attribution operational |
| petalTongue | DOWN | Binary doesn't support --socket |
| skunkBat | DOWN | Socket not detected |

### 4. PG-02 provenance trio VERIFIED

Live IPC roundtrip on southGate NUCLEUS:
- `spine.create` → loamSpine: `{"spine_id":"019e66d8-decb-7c53-8e4d-570c00b4aaf4","genesis_hash":[71,66,252,...]}`
- `braid.create` → sweetGrass: Full PROV-O braid with `@context`, `@id`, DID, timestamp
- rhizoCrypt alive via permanence alias
- NestGate alive, BTSP-auth gates storage (correct)

### 5. Mesh federation

- Songbird TCP on `*:7700` (LAN-reachable)
- `mesh.init` called with `node_id=southgate01`, `bootstrap_peers=["192.168.1.144:7700"]`
- 0 peers at time of deploy (eastGate offline)

## Remaining

- 5/13 primals not health-responding (ToadStool, barraCuda, Squirrel, petalTongue, skunkBat)
- petalTongue binary needs Wave 53+ release to support --socket
- WS-2 nest.sync orchestration experiment pending
- WS-11 Tenaillon re-measurement via covalent compute pending

## Gap Status

**0 PG gaps open.** 22/22 resolved/closed. PG-02/PG-04 were the last.
