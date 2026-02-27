# NUCLEUS Local Deployment — Eastgate Tower

**Date:** February 27, 2026
**Node:** Eastgate (i9-12900, RTX 4070, AKD1000, 32GB DDR5)
**Status:** OPERATIONAL — 4 primals + Neural API live, NCBI data flowing

---

## What's Running

```
/run/user/1000/biomeos/
├── beardog-eastgate.sock          (crypto, TLS 1.3)
├── songbird-eastgate.sock         (HTTP, discovery, mesh)
├── toadstool-eastgate.jsonrpc.sock (compute, JSON-RPC)
├── toadstool-eastgate.sock        (compute, tarpc)
├── nestgate-eastgate.sock         (storage)
└── songbird.sock                  (secondary)

/tmp/biomeos-neural-api-eastgate.sock (Neural API router)
/tmp/neural-api-eastgate.sock → symlink to above
```

## Validated Capabilities

| Primal | IPC | Status | Key Test |
|--------|-----|--------|----------|
| **BearDog** | `primal.info` | production_ready 0.9.0 | 72 crypto methods, HSM init, BTSP |
| **Songbird** | `primal.info` | 0.2.1 | HTTP/HTTPS via BearDog TLS, 14 capabilities |
| **ToadStool** | `toadstool.health` | healthy 0.1.0 | 36 methods, JSON-RPC + tarpc dual socket |
| **NestGate** | `health` | healthy 0.1.0 | storage.store/retrieve/list, blob, model cache |
| **Neural API** | capability.call routing | COORDINATED | 121 translations, Tower Atomic detected |

## End-to-End Test: NCBI via Full Primal Stack

```
Query: "Vibrio harveyi 16S" → NCBI E-utilities
Path:  Songbird HTTP → BearDog TLS 1.3 → NCBI → Songbird → caller
Result: 39,598 hits, accession PX756524.1 (V. harveyi ML28, 1264bp, 2026-02-24)
Stored: NestGate key "science:ncbi:esummary:3135754935" (283 bytes)
```

## Startup Commands

```bash
# 1. BearDog (crypto foundation)
FAMILY_ID=eastgate NODE_ID=eastgate RUST_LOG=beardog=info \
  beardog server --family-id eastgate

# 2. Songbird (network + discovery)
FAMILY_ID=eastgate NODE_ID=eastgate \
  BEARDOG_SOCKET=/run/user/1000/biomeos/beardog-eastgate.sock \
  CRYPTO_PROVIDER_SOCKET=/run/user/1000/biomeos/beardog-eastgate.sock \
  SONGBIRD_SECURITY_PROVIDER=beardog RUST_LOG=songbird=info \
  songbird server --socket /run/user/1000/biomeos/songbird-eastgate.sock

# 3. ToadStool (compute)
FAMILY_ID=eastgate NODE_ID=eastgate \
  TOADSTOOL_SOCKET=/run/user/1000/biomeos/toadstool-eastgate.sock \
  BEARDOG_SOCKET=/run/user/1000/biomeos/beardog-eastgate.sock \
  SONGBIRD_SOCKET=/run/user/1000/biomeos/songbird-eastgate.sock \
  RUST_LOG=toadstool=info toadstool daemon

# 4. NestGate (storage)
NESTGATE_JWT_SECRET=$(openssl rand -base64 48) \
  FAMILY_ID=eastgate NODE_ID=eastgate \
  NESTGATE_SOCKET=/run/user/1000/biomeos/nestgate-eastgate.sock \
  BEARDOG_SOCKET=/run/user/1000/biomeos/beardog-eastgate.sock \
  SONGBIRD_SOCKET=/run/user/1000/biomeos/songbird-eastgate.sock \
  RUST_LOG=nestgate=info nestgate daemon

# 5. Neural API (biomeOS orchestrator)
cd biomeOS && FAMILY_ID=eastgate NODE_ID=eastgate RUST_LOG=info \
  nucleus serve --family eastgate
```

---

## Deployment Feedback for Team

### Issues Found (actionable)

**F1: BearDog `--family-id` CLI flag doesn't set FAMILY_ID env**

BearDog accepts `--family-id eastgate` on the CLI but then fails with
`FAMILY_ID or BEARDOG_FAMILY_ID must be set`. The CLI flag should populate
the internal identity, or at minimum the error message should mention the
CLI flag. Same issue with `NODE_ID` / `--node-id` (which doesn't exist as a
CLI flag at all).

*Affected:* `beardog server`
*Workaround:* Set `FAMILY_ID` and `NODE_ID` as environment variables.

**F2: Songbird starts without IPC unless `--socket` is passed**

Running `songbird server` starts the orchestrator but does NOT bind an IPC
socket. The startup log says "Songbird ready!" but then suggests
`--socket` or `--listen` as a tip. For NUCLEUS deployments, IPC should be
enabled by default when `FAMILY_ID` is set.

*Affected:* `songbird server`
*Workaround:* Always pass `--socket /run/user/1000/biomeos/songbird-{family}.sock`.

**F3: Neural API socket naming mismatch**

- `nucleus serve` creates: `/tmp/biomeos-neural-api-eastgate.sock`
- Songbird HTTP client looks for: `/tmp/neural-api-eastgate.sock`
- The graph env specifies: `${XDG_RUNTIME_DIR}/biomeos/neural-api-${FAMILY_ID}.sock`

Three different conventions. The HTTP client's fallback discovery path
(`/tmp/neural-api-{family}.sock`) doesn't match the actual server path
(`/tmp/biomeos-neural-api-{family}.sock`).

*Affected:* Songbird TLS → BearDog delegation via Neural API
*Workaround:* `ln -sf /tmp/biomeos-neural-api-eastgate.sock /tmp/neural-api-eastgate.sock`
*Fix:* Standardize on one path, preferably the XDG one.

**F4: ToadStool Songbird registration fails (missing `primal_id`)**

ToadStool attempts to register with Songbird on startup but the registration
call fails with `{\"code\":-32603,\"message\":\"Invalid params: missing field
'primal_id'\"}`. ToadStool is sending a registration payload that doesn't
include `primal_id`.

*Affected:* ToadStool → Songbird auto-discovery
*Workaround:* ToadStool falls back to standalone mode (IPC still works).

**F5: NestGate refuses to start without `NESTGATE_JWT_SECRET`**

For local NUCLEUS deployments over Unix sockets (no HTTP), JWT auth is not
needed. NestGate should allow socket-only mode without JWT, or auto-generate
a random secret for local-only deployments.

*Affected:* `nestgate daemon`
*Workaround:* `export NESTGATE_JWT_SECRET=$(openssl rand -base64 48)`

**F6: NestGate `storage.list` doesn't find stored keys**

After `storage.store` with `family_id=wetspring, key=test:hello`, calling
`storage.list` with the same `family_id` and `prefix=test:` returns empty.
The key exists (retrieve works), but list doesn't find it.

*Affected:* `storage.list` IPC method
*Workaround:* Use `storage.retrieve` with known keys.

### Observations (non-blocking)

- **BearDog** creates an `audit.log` in the CWD — should use XDG data dir.
- **Songbird** attempts external connectivity test to `192.168.1.144:8080` on
  startup (hardcoded?). Fails gracefully but produces alarming error messages.
- **ToadStool** has dual-socket architecture (tarpc + JSON-RPC). The graph
  references `toadstool-{family_id}.sock` but ToadStool creates both
  `toadstool-{family_id}.sock` (tarpc) and `toadstool-{family_id}.jsonrpc.sock`.
  JSON-RPC callers need to use the `.jsonrpc.sock` variant.
- **NestGate** reports `Family: default` despite `FAMILY_ID=eastgate` being set.
  The family_id is correctly handled at the storage level but not displayed.
- **Neural API** reads graphs from the biomeOS project directory, so `nucleus`
  must be run from the biomeOS project root to find `graphs/`.
- **Build times** (release, from clean): BearDog 1m34s, Songbird 3m12s,
  ToadStool 2m10s, NestGate 2m06s, biomeOS 57s. Total: ~10 minutes.

### What Works Well

- **BearDog** is rock-solid. Starts in <100ms, binds socket + TCP, full crypto
  suite available immediately. Best startup experience of all primals.
- **Neural API auto-discovery** is elegant. It found all running primals via
  socket scanning, loaded 121 translations from the graph, and entered
  COORDINATED MODE automatically.
- **NestGate storage** works cleanly. Store/retrieve/capabilities all respond
  correctly with structured JSON-RPC. Content-addressed storage with BLAKE3
  is a solid foundation.
- **Cross-primal TLS delegation** works end-to-end. Songbird → Neural API →
  BearDog TLS 1.3 → NCBI HTTPS. Real *Vibrio harveyi* 16S data retrieved from
  NCBI and stored in NestGate.
- **PID locking** (Songbird) prevents duplicate instances. Good safety feature.
- **Graceful degradation** — each primal falls back to standalone mode when
  dependencies aren't available, rather than crashing.

---

## Architecture Validated

```
biomeOS Neural API (121 capability translations)
    │
    ├── BearDog (crypto, TLS 1.3, HSM, BTSP)
    │     └── 72 JSON-RPC methods
    │
    ├── Songbird (HTTP, discovery, mesh, onion, relay)
    │     ├── http.request → BearDog TLS → external HTTPS
    │     └── 14 capability categories
    │
    ├── ToadStool (compute, GPU, workload orchestration)
    │     ├── JSON-RPC: toadstool-eastgate.jsonrpc.sock
    │     └── tarpc: toadstool-eastgate.sock
    │
    └── NestGate (storage, BLAKE3, blob, model cache)
          └── storage.store / retrieve / list / blob
```

## Next Steps

1. Register wetSpring as a science capability provider (needs Songbird registration fix F4)
2. Wire NestGate `NCBILiveProvider` for bulk data acquisition
3. SRA toolkit integration for FASTQ downloads
4. Test ToadStool compute dispatch from wetSpring science pipeline
5. Multi-tower extension (bring up Strandgate as second node)
