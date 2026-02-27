# Exp203: biomeOS Science Pipeline Integration

**Status**: PASS (29/29 checks)
**Date**: 2026-02-27
**Binary**: `validate_science_pipeline` (requires `--features ipc`)

## Hypothesis

wetSpring can function as a biomeOS science primal, accepting JSON-RPC 2.0
requests over a Unix domain socket and dispatching to validated barracuda
library functions with zero math duplication.

## Method

Built `wetspring-server` — a thin JSON-RPC 2.0 server over Unix socket
implementing the Primal IPC Protocol v2.0. Created validation binary that
tests the full `science_pipeline.toml` graph locally:

1. **Server lifecycle**: bind, accept, multi-threaded dispatch
2. **health.check**: returns status, version, 5 registered capabilities
3. **science.diversity**: Shannon, Simpson, Chao1, observed, Pielou, Bray-Curtis
4. **science.qs_model**: ODE integration for 4 QS/biofilm scenarios
5. **science.full_pipeline**: chains diversity → QS model → Anderson
6. **Protocol compliance**: JSON-RPC 2.0 error codes, multi-request connections
7. **Songbird registration**: graceful fallback when Songbird unavailable
8. **Neural API metrics**: per-method call counts, latency tracking

## Results

| Check | Result |
|-------|--------|
| health.check returns healthy + 5 capabilities | PASS |
| Diversity: Shannon/Simpson/Chao1/observed/Pielou exact match | PASS |
| Diversity: Bray-Curtis between sample pairs | PASS |
| QS model: 4 scenarios (standard, high density, HapR mutant, DGC) | PASS |
| Full pipeline: diversity + QS model stages chained | PASS |
| Protocol: unknown method → -32601, bad version → error | PASS |
| Multiple requests on single connection | PASS |
| Songbird graceful fallback | PASS |
| Metrics: 19 total calls, 15 successes, 4 errors tracked | PASS |

## Key Finding

wetSpring is now a fully functional biomeOS science primal. The IPC server
adds zero new math — it is a thin JSON-RPC wrapper over the same barracuda
library functions validated in Exp001–196. When biomeOS orchestrates the
`science_pipeline.toml` graph (NestGate → wetSpring → ToadStool), wetSpring
will handle the science nodes correctly.

## Modules Validated

- `ipc::protocol` — JSON-RPC 2.0 parse/format
- `ipc::dispatch` — method routing to 5 capabilities
- `ipc::server` — Unix socket listener, threaded connections
- `ipc::songbird` — Songbird discovery registration + heartbeat
- `ipc::metrics` — Neural API pathway learning metrics
- `bio::diversity` — via `science.diversity` dispatch
- `bio::qs_biofilm` + `bio::ode` — via `science.qs_model` dispatch

## Architecture

```
biomeOS Graph Executor
  ↓ capability.call("science.diversity", {...})
  ↓
wetspring-server (Unix socket, JSON-RPC 2.0)
  ↓ dispatch
  ↓
barracuda::bio::diversity::shannon() [same validated code]
  ↓
JSON-RPC response → biomeOS
```
