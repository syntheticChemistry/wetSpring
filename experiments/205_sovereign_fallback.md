# Exp205: Sovereign Fallback — wetSpring With/Without biomeOS

**Status**: PASS (structural validation)
**Date**: 2026-02-27
**Binary**: `validate_science_pipeline` + `nestgate` unit tests

## Hypothesis

wetSpring works identically with or without biomeOS. The IPC server and
all science capabilities function in standalone mode; biomeOS integration
is purely additive.

## Method

Validated the three-tier data routing strategy in `nestgate.rs`:

1. **Tier 1 (biomeOS)**: `capability.call("science.ncbi_fetch")` via Neural API
2. **Tier 2 (NestGate)**: Direct socket to NestGate cache + sovereign fallback
3. **Tier 3 (Sovereign)**: Direct NCBI HTTP fetch

Each tier falls through gracefully when the upstream service is unavailable.

## Results

| Check | Result |
|-------|--------|
| `discover_biomeos_socket()` returns None when biomeOS not running | PASS |
| `fetch_via_biomeos()` fails gracefully with nonexistent socket | PASS |
| `fetch_tiered()` falls through to sovereign when all ecosystem services absent | PASS |
| IPC server operates fully in standalone mode | PASS |
| Songbird registration fails gracefully (standalone mode) | PASS |
| All 5 science capabilities work without biomeOS | PASS |
| Metrics tracking works without biomeOS | PASS |

## Key Finding

wetSpring is fully sovereign. Every capability (diversity, QS model, Anderson,
NCBI fetch, full pipeline) works without any ecosystem service. The three-tier
routing adds biomeOS integration as a transparent optimization layer:
- With biomeOS: routes through Neural API → NestGate → cached data
- Without biomeOS: falls through to direct NCBI HTTP
- Zero behavior change in the science math

## Modules Validated

- `ncbi::nestgate::discover_biomeos_socket` — biomeOS socket discovery
- `ncbi::nestgate::fetch_via_biomeos` — Neural API capability.call routing
- `ncbi::nestgate::fetch_tiered` — three-tier fallback orchestration
- `ipc::songbird::discover_socket` — graceful Songbird fallback
- `ipc::server` — standalone operation
