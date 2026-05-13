<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V165b — Tier 2 Wave Sync + Ecosystem Response

**From:** wetSpring V165b
**To:** primalSpring (upstream audit), lithoSpore, projectNUCLEUS, all spring teams
**Date:** May 12, 2026
**Context:** Response to "Ecosystem Wave Sync" — toadStool S250, barracuda.precision.route

---

## Same-Day Response to Ecosystem Wave

The ecosystem wave announced `toadstool.validate` and `barracuda.precision.route`
as live upstream. wetSpring wired both within hours:

| Upstream Event | wetSpring Response | Version |
|---------------|-------------------|---------|
| `toadstool.validate` (S250) | `ipc/toadstool_validate.rs` — typed client, 4 tests | V165b |
| `barracuda.precision.route` (S250) | `ipc/precision_route.rs` — typed client, 2 tests | V165b |
| `toadstool.list_workloads` (S245+) | Wired in same module | V165b |
| LTEE B7 audit request | `validate_ltee_b7_v1.rs` — 27/27 PASS | V165 |
| Thread 4 expression request | `ENVIRONMENTAL_GENOMICS.md` authored | V164 |
| `--format json` request | `OutputFormat` enum on UniBin | V164 |
| Foundation 10/10 threads | All wired in `THREAD_INDEX.toml` | V164b |

---

## Tier 2 Convergence — Structurally Complete

| Component | Module | Status |
|-----------|--------|--------|
| `--format json` | `cli.rs` `OutputFormat` enum | Done (V164) |
| `toadstool.validate` | `ipc/toadstool_validate.rs` | Done (V165b) |
| `toadstool.list_workloads` | `ipc/toadstool_validate.rs` | Done (V165b) |
| `barracuda.precision.route` | `ipc/precision_route.rs` | Done (V165b) |
| `compute.dispatch.submit` | `ipc/compute_dispatch.rs` | Already wired |
| `compute.performance_surface` | `ipc/performance_surface.rs` | Already wired |
| 12 workload TOMLs | `projectNUCLEUS/workloads/wetspring/` | Done (V164) |

wetSpring is **Tier 2 structurally ready**. When toadStool's production socket
is available, the full Tier 2 pipeline works: workload pre-flight → precision
advisory → dispatch → validation → JSON output.

---

## Absorption Patterns for Other Springs

### `toadstool.validate` client pattern

```rust
// ipc/toadstool_validate.rs — reuses compute_dispatch discovery
pub fn validate(workload_path: &str, dry_run: bool) -> Result<ValidateResult, ValidateError>
pub fn list_workloads(filter: &str) -> Result<ListWorkloadsResult, ValidateError>
```

`ValidateResult`: `valid`, `gpu_available`, `precision_tier`,
`estimated_dispatch_time_ms`, `warnings`, `required_capabilities`.

### `barracuda.precision.route` client pattern

```rust
// ipc/precision_route.rs — discovers via BARRACUDA_SOCKET
pub fn route(domain: &str, hardware_hint: &str) -> Result<PrecisionAdvice, PrecisionError>
```

`PrecisionAdvice`: `recommended_tier`, `fma_safe`, `requires_compiler`, `hardware_hint`.

Both follow the `compute_dispatch.rs` pattern: `discover()` → `rpc_call()` →
typed parse. No `unsafe`, no hardcoded paths, family_id aware.

---

## Current Metrics

| Metric | Value |
|--------|-------|
| Tests | 1,962 lib + 97 integration + 18 IPC |
| Binaries | 367 (345 barracuda + 22 forge) |
| Experiments | 384 indexed |
| Papers | 63/63 + LTEE B7 TIER 2 COMPLETE |
| Coverage | 91.20% line / 90.30% function |
| GuideStone | Level 4 (38/38 pass) |
| Primal gaps | 4 open (all external) |
| Foundation | 10/10 threads active |
| Tier | 2 (structurally complete) |
