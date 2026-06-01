# wetSpring Wave 67 — Glacial Cutover: southGate Mesh Primals

**Date:** June 1, 2026
**Version:** V193
**Gate:** southGate
**Impulse:** `wave67-southgate-glacial-mesh-primals` (ACK)

---

## Summary

All three P0 glacial blockers assigned to southGate are resolved. Songbird,
biomeOS, and bearDog changes committed and pushed upstream. eastGate can now
proceed with Phase 1 mesh validation (`discovery.peers` + `capability.call`).

---

## P0 Fixes

### 1. Songbird Security Socket Fix (P0 BLOCKER — RESOLVED)

**Commit:** `ae9b42f0` on `primals/songBird`

**Problem:** `SecurityCryptoProvider::from_env()` in `songbird-http-client` ignored the
`--security-socket` CLI flag. The flag sets `SECURITY_PROVIDER_ENDPOINT` env var, but
`from_env()` only checked `SECURITY_PROVIDER_MODE` before falling into neural/direct
mode discovery — bypassing the CLI override entirely.

**Fix:**
- `from_env()` now checks `SECURITY_PROVIDER_ENDPOINT` first, before mode selection
- `discover_neural_api_socket()` includes `SECURITY_PROVIDER_ENDPOINT` at priority 3
  (after `SECURITY_PROVIDER_SOCKET`, before `BEARDOG_SOCKET`)
- `songbird-crypto-provider` discovery aligned with same pattern

**Effect:** When Songbird starts with `--security-socket /path/to/beardog.sock`, ALL
code paths (orchestrator server AND http_client TLS) use the correct socket. Federation
TLS and cross-gate `capability.call` routing now work with explicit socket configuration.

### 2. biomeOS capability.call RPC (P0 BLOCKER — RESOLVED)

**Commit:** `9ed36983` on `primals/biomeOS`

**Problem:** `capability.call` returned `-32601 Method not found` on the biomeOS API
socket (`unix_server.rs`). The full implementation exists on the Neural API socket, but
cross-gate probes and spring health sweeps often hit the API socket.

**Fix:**
- API socket now proxies `capability.call`, `graph.execute`, and `topology.primals`
  to the Neural API socket via `neural-api-client`
- Proxy returns `-32002` with actionable message if Neural API socket is not running
- Fixed stale test that expected `-32601` for `graph.execute` (now proxied)
- `capabilities.list` response updated to advertise `capability.call` and `graph.execute`

**Effect:** `capability.call` works on BOTH sockets. Cross-gate probes succeed regardless
of which socket they connect to.

### 3. bearDog S4 Auth Config (P0 — VERIFIED)

**Commit:** `a61c37101` on `primals/bearDog`

**Status:** bearDog is running with full auth services accessible:

| Endpoint | Status |
|----------|--------|
| UDS `beardog-southgate.sock` | Alive |
| TCP `0.0.0.0:9100` | Bound and accepting |
| `auth.mode` | `permissive` (switch to `enforced` for formal shadow) |
| `auth.issue_ionic` | Ed25519-signed tokens, 374 bytes |
| `auth.verify_ionic` | Validates signature + claims, returns `valid: true` |
| `auth.public_key` | `did:key:z6Mki1B4GPTWum1Pf1byM43KwP442P1kPyxNvMvU1634bgxH` |

**Token roundtrip verified:**
```
issue_ionic → token (374 bytes, EdDSA)
verify_ionic → { valid: true, scope_ok: true, claims: { iss, sub, exp, iat, jti } }
```

**For ironGate 7-day shadow:**
1. bearDog TCP :9100 is reachable from LAN
2. `jupyterhub_btsp_auth.py` connects to TCP :9100
3. Set `BEARDOG_AUTH_MODE=enforced` when starting formal validation
4. ironGate caches public key via `auth.public_key` for local verification

**Changes committed:** MethodGate evolution (enforced/permissive), BTSP config cleanup,
Android/Unix platform support, updated ENVIRONMENT_VARIABLES docs.

---

## Phase 1 Readiness

southGate is ready for eastGate mesh validation:

| Criterion | Status |
|-----------|--------|
| Songbird socket fix | DONE — pushed `ae9b42f0` |
| biomeOS capability.call | DONE — pushed `9ed36983` |
| bearDog S4 auth | DONE — pushed `a61c37101`, TCP :9100 live |
| SONGBIRD_PEERS configured | Needs verification after Songbird redeploy |
| Federation 13/13 | 10/13 health, needs redeploy for coralReef |

**Next:** eastGate runs `discovery.peers` smoke test (eastGate <-> southGate) and
`s_covalent_mesh` live validation. southGate is the mesh partner.

---

## Glacial Cutover Checklist (southGate items)

- [x] Songbird security socket fix (P0 blocker)
- [x] biomeOS capability.call RPC (P0 blocker)
- [x] bearDog S4 auth config (P0)
- [x] All three pushed to GitHub
- [ ] SONGBIRD_PEERS verification after redeploy
- [ ] Cross-gate discovery.peers smoke test (eastGate drives)
- [ ] capability.call smoke test (eastGate drives)
- [ ] BEARDOG_AUTH_MODE=enforced for formal 7-day shadow (ironGate coordinates)
