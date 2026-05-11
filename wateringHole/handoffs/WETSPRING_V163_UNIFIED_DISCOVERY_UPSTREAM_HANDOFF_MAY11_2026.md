<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# wetSpring V163 — Unified Socket Discovery & Upstream Primal/Spring Handoff

**Date:** May 11, 2026
**Version:** V163
**From:** wetSpring
**To:** All primal teams, all spring teams, projectNUCLEUS, foundation

---

## Summary

V163 unifies socket discovery across all IPC client paths, ensuring
multi-instance deployment parity via `family_id()`. Deep debt audit
confirms zero remaining internal debt across all dimensions.

---

## V163 Changes

### Socket Discovery Unified

**Bug fixed:** `facade/ipc_client.rs` was hardcoding `wetspring-default.sock`,
ignoring `FAMILY_ID` / `BIOMEOS_FAMILY_ID`. In multi-instance deployments
(`FAMILY_ID` ≠ `"default"`), the facade client would connect to the wrong
socket while the IPC server bound to `wetspring-{family_id}.sock`.

**Fix:** Facade now delegates to `ipc::discover::resolve_bind_path()` —
same path logic as the server. Same fix applied to:
- `ncbi/nestgate/discovery.rs` standalone fallback (non-`ipc` builds)
- `visualization/ipc_push.rs` petalTongue fallback (non-`ipc` builds)

**Pattern for other springs:** Any standalone socket discovery code (for
builds without the `ipc` feature) must replicate the `FAMILY_ID` /
`BIOMEOS_FAMILY_ID` → `"default"` cascade. Reference implementation:
`barracuda/src/ipc/discover.rs:family_id()`.

### Deep Debt Audit Results (V163)

| Dimension | Result |
|-----------|--------|
| `unsafe` code | **ZERO** — `forbid(unsafe_code)` enforced workspace-wide |
| `#[allow()]` | **ZERO** — all lint suppressions use `#[expect(reason = "...")]` |
| Production mocks | **ZERO** — all mock/dummy/fake code in `#[cfg(test)]` modules |
| `extern crate` / `try!` / `#[macro_use]` | **ZERO** — fully edition 2024 idiomatic |
| Hardcoded URLs | **ZERO** — all URLs have env var overrides with defaults |
| Large library files (>800L) | **ZERO** — 5 files >800L are all validation binaries |
| External C deps | **ZERO** direct — `wgpu` has unavoidable native GPU drivers; `deny.toml` bans 15 C-binding crates |
| Dead feature flags | **ZERO** — all features actively used |
| `dyn` dispatch | **ONE** justified exception: `gillespie.rs` `PropensityFn = Box<dyn Fn>` — heterogeneous reaction vec requires type erasure per SSA design. Documented with audit trail. |

---

## Remaining 4 Open Gaps (All External)

| PG | Owner | Dependency | wetSpring Status |
|----|-------|------------|-----------------|
| PG-02 | rhizoCrypt / loamSpine / sweetGrass | Provenance trio IPC endpoints not yet live | IPC paths wired, graceful degradation confirmed |
| PG-03 | Songbird / biomeOS | `capability.resolve` RPC not yet shipped | `discover_by_capability()` wired as migration point |
| PG-04 | NestGate | Live deployment needed for data routing | Standalone HTTP fallback works, NestGate routing staged |
| PG-05 | toadStool | Sovereign dispatch requires compiled GPU binary | `compute.dispatch` method responsive, binary constraint expected |

**None of these block wetSpring operations.** All have graceful degradation
(standalone mode, local compute, direct HTTP). They block full primal
composition chains — specifically:
- **PG-02 + PG-04:** Block provenance-wrapped data pipelines
- **PG-03:** Blocks runtime capability routing (currently compile-time map)
- **PG-05:** Blocks sovereign GPU dispatch (currently in-process or IPC)

---

## Patterns for Absorption by Other Springs

### 1. Handler-Level `primal-proof` Wiring (V159–V160)

Any handler that calls a math library directly can be dual-pathed:

```rust
#[cfg(feature = "primal-proof")]
{
    if let Ok(result) = barracuda_route::try_forward("stats.diversity", &params) {
        return Ok(result);
    }
}
// In-process fallback
crate::bio::diversity::shannon(data)
```

Reference: `barracuda/src/ipc/handlers/science.rs` (5 handlers wired).

### 2. IPC-First Defaults (V162b)

```toml
# Cargo.toml — default does NOT link barraCuda library
[features]
default = []
barracuda-lib = ["dep:barracuda"]
ipc = ["json", "dep:serde", "dep:serde_json"]
```

Users opt in to local compute with `--features barracuda-lib`. IPC client
works without the library. Reference: `barracuda/Cargo.toml`.

### 3. Unified Socket Discovery (V163)

All socket paths should resolve via:
1. `{PRIMAL}_SOCKET` env var
2. `$XDG_RUNTIME_DIR/biomeos/{primal}-{family_id}.sock`
3. `<temp_dir>/{primal}-{family_id}.sock`

Where `family_id` = `FAMILY_ID` || `BIOMEOS_FAMILY_ID` || `"default"`.
Never hardcode `-default.sock`. Reference: `barracuda/src/ipc/discover.rs`.

### 4. Foundation Thread Seeding

Target file: `data/targets/thread04_enviro_targets.toml` (36 validated targets).
Thread index: `THREAD_INDEX.toml`. BLAKE3 content hashes + sweetGrass braid
references make results load-bearing geological layers.

### 5. LTEE Reproduction Pattern

1. Pick paper from `specs/PAPER_REVIEW_QUEUE.md`
2. Create Python baseline in `notebooks/` (Tier 1)
3. Produce expected values JSON for lithoSpore
4. Mark queue item STARTED with experiment ID

wetSpring B7 (Tenaillon 2016) is STARTED — sovereign genomics pipeline.

---

## NUCLEUS Deployment via Neural API (biomeOS)

### Cell Membrane Architecture

```
Extracellular (CDN)     Membrane (tunnel)     Intracellular (sovereign)
─────────────────────   ──────────────────    ─────────────────────────
primals.eco/lab/        biomeos deploy         toadStool dispatch
  notebooks/              --graph ...            GPU/NPU/CPU routing
  data explorer         UniBin validate        NestGate storage
  provenance verify     composition.status     Provenance trio
```

### Deployment Flow

1. `fetch_primals.sh` downloads UniBin from plasmidBin
2. `biomeos deploy --graph graphs/wetspring_deploy.toml`
3. toadStool dispatches workloads to sovereign compute
4. Provenance trio wraps results (when live — PG-02)
5. `composition.status` reports health via Neural API

### Workload TOMLs (projectNUCLEUS)

11 wetSpring workload TOMLs in `projectNUCLEUS/workloads/wetspring/`.
All gate-agnostic (use `$SPRINGS_ROOT`). Ready for validation when
UniBin is published to plasmidBin.

---

## Next Evolution Targets

1. **Close PG-02/03/04/05:** External teams need IPC-ready endpoints
2. **LTEE B7 completion:** 264 NCBI genomes → mutation accumulation curves
3. **plasmidBin release binary:** `cargo build --release` → publish
4. **Foundation seeding:** Continue anchoring LTEE genomic data to Thread 04
5. **guidestone L4 → L5:** Expand certification organelle coverage

---

*This handoff feeds back to primalSpring via the
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol. Upstream audit requested.*
