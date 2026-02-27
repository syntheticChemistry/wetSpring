# Exp204: Capability Discovery via Songbird

**Status**: PASS (structural validation)
**Date**: 2026-02-27
**Binary**: `validate_science_pipeline` (Songbird stage)

## Hypothesis

wetSpring can register its science capabilities with Songbird for
capability-based discovery, enabling biomeOS to route `capability.call`
requests to the wetSpring IPC server.

## Method

Implemented `ipc::songbird` module with:

1. **Socket discovery**: `SONGBIRD_SOCKET` → `$XDG_RUNTIME_DIR/biomeos/songbird-default.sock` → temp fallback
2. **Registration**: `discovery.register` JSON-RPC call declaring 5 capability tags
3. **Heartbeat loop**: Background thread sends `discovery.heartbeat` every 30s
4. **Re-registration**: On heartbeat failure, attempts re-registration
5. **Graceful fallback**: When Songbird is unavailable, server operates standalone

## Results

| Check | Result |
|-------|--------|
| `discover_socket()` does not panic | PASS |
| Explicit socket override via env var | PASS |
| Registration to nonexistent socket returns error | PASS |
| Heartbeat to nonexistent socket returns error | PASS |
| Server starts in standalone mode when Songbird absent | PASS |

## Key Finding

The Songbird registration client follows the same pattern as `nestgate.rs`:
JSON-RPC 2.0 over Unix socket, capability-based socket discovery, graceful
fallback. When the Tower node (BearDog + Songbird) is running, wetSpring
will register automatically and biomeOS can discover it for graph execution.

## Modules Validated

- `ipc::songbird::discover_socket` — three-tier socket discovery
- `ipc::songbird::register` — JSON-RPC registration call
- `ipc::songbird::heartbeat` — keep-alive maintenance
- `ipc::songbird::start_heartbeat_loop` — background thread lifecycle
