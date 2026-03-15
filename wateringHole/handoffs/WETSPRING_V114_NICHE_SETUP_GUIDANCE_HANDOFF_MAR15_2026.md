# wetSpring V114 — Niche Setup Guidance for Springs

**Date:** March 15, 2026
**From:** wetSpring V114
**To:** All Springs modeling niche deployment
**Authority:** wateringHole (ecoPrimals Core Standards)

---

## Purpose

This handoff documents how wetSpring set itself up as a biomeOS niche,
following wateringHole standards. Other springs can use this as a reference
implementation for their own niche deployment.

## The Niche Setup Checklist

wetSpring followed this sequence to become a deployable niche. Each step
references the wateringHole standard that governs it.

### Step 1: UniBin Binary Structure

**Standard:** `UNIBIN_ARCHITECTURE_STANDARD.md`

Your spring binary must support these subcommands:

```
wetspring server   # Start JSON-RPC IPC server
wetspring status   # Report health and capabilities
wetspring version  # Print version string
```

The binary name matches the primal name. `--help` and `--version` are required.
Structured error output with exit codes (0 success, 1 failure).

### Step 2: IPC Server + Socket Binding

**Standard:** `SPRING_AS_PROVIDER_PATTERN.md`

Bind a JSON-RPC 2.0 socket at:
```
$XDG_RUNTIME_DIR/biomeos/wetspring-${FAMILY_ID}.sock
```

Socket discovery cascade (most specific → most general):
1. `WETSPRING_SOCKET` environment variable
2. `BIOMEOS_SOCKET_DIR` + primal name
3. `XDG_RUNTIME_DIR/biomeos/` + primal name
4. `/tmp/biomeos/` fallback

wetSpring implementation: `barracuda/src/ipc/mod.rs` (server), `barracuda/src/ipc/dispatch.rs` (routing)

### Step 3: Capability Registration

**Standard:** `SPRING_AS_PROVIDER_PATTERN.md`

Register capabilities with Neural API via `capability.register`:

```json
{
  "method": "capability.register",
  "params": {
    "capability": "science.diversity",
    "primal": "wetspring",
    "socket": "/run/user/1000/biomeos/wetspring-00000000.sock",
    "semantic_mappings": ["alpha_diversity", "beta_diversity", "shannon", "simpson"]
  }
}
```

wetSpring registers 19 capabilities across 4 domains:
- **Science (11):** diversity, peak_detect, ode, feature_table, spectral_match, kinetics, alignment, taxonomy, phylogenetics, nmf, denoising
- **Time series (2):** timeseries, timeseries_diversity
- **Provenance (3):** provenance.begin, provenance.record, provenance.complete
- **Infrastructure (3):** health.check, capability.list, metrics.summary

Implementation: `barracuda/src/ipc/handlers/mod.rs` → `CAPABILITIES` array

### Step 4: Deploy Graph

**Standard:** `SPRING_AS_NICHE_DEPLOYMENT_STANDARD.md`

Create `graphs/<spring>_deploy.toml` defining the germination DAG:

```toml
[graph]
name = "wetspring_niche"
version = "1.0.0"
coordination = "Sequential"

# Phase 1: Tower (required infrastructure)
[[nodes]]
id = "beardog"
primal = "bearDog"
phase = 1
capabilities_required = ["auth.session"]

# Phase 2: Optional dependencies
[[nodes]]
id = "songbird"
primal = "songBird"
phase = 2
capabilities_required = ["ipc.register"]

# Phase 3: Your spring
[[nodes]]
id = "wetspring"
primal = "wetSpring"
phase = 3
depends_on = ["beardog", "songbird"]
capabilities_provided = ["science.diversity", "science.peak_detect", ...]

# Phase 4: Validation
[[nodes]]
id = "validation"
primal = "wetSpring"
subcommand = "validate"
phase = 4
depends_on = ["wetspring"]
```

Key principles:
- **4 phases:** Tower → optional deps → spring → validation
- **Capability-based:** `depends_on` names nodes, `capabilities_required` names what you need
- **No hardcoded paths:** biomeOS resolves everything via capability routing
- **Sequential coordination:** Phases execute in order, nodes within a phase can parallelize

wetSpring implementation: `graphs/wetspring_deploy.toml`

### Step 5: Provenance Trio Integration (Optional but Recommended)

**Standard:** `SPRING_PROVENANCE_TRIO_INTEGRATION_PATTERN.md`

Wire the three-phase provenance lifecycle:
1. `begin_session(name)` → rhizoCrypt dehydrates ephemeral state
2. `record_step(session_id, data)` → append to DAG vertex
3. `complete_session(session_id)` → dehydrate → loamSpine commit → sweetGrass attribute

All interaction via `capability.call` to biomeOS — never import trio code directly.
Graceful degradation: domain logic works when trio is unavailable.

wetSpring implementation: `barracuda/src/ipc/provenance.rs`

### Step 6: Cross-Spring Data Exchange (Optional)

**Standard:** `CROSS_SPRING_DATA_FLOW_STANDARD.md`

Use the canonical time series schema for spring-to-spring data:

```json
{
  "schema": "ecoPrimals/time-series/v1",
  "source_spring": "wetspring",
  "variable": "shannon_diversity",
  "unit": "nats",
  "timestamps": ["2026-03-15T00:00:00Z"],
  "values": [2.45]
}
```

wetSpring implementation: `barracuda/src/ipc/timeseries.rs`

### Step 7: Neural API Workflow Graphs (Advanced)

**Standard:** `whitePaper/neuralAPI/03_GRAPH_EXECUTION.md`

Define workflow graphs for automated multi-step operations:
- **Sequential:** Provenance-wrapped experiment pipelines
- **Pipeline:** Cross-spring composition (wetSpring → airSpring)
- **Continuous:** Live dashboards with feedback loops

These graphs enable biomeOS to orchestrate your spring's capabilities
without manual coordination.

## Lessons Learned from wetSpring

1. **Start with `health.check` and `capability.list`** — these are mandatory and
   exercise the full IPC stack before you add science capabilities.

2. **One handler per capability** — each `science.*` method gets its own function
   with explicit `/// # Errors` documentation.

3. **Graceful degradation everywhere** — if a dependency (like the provenance trio)
   is unavailable, return partial results, not errors.

4. **Named tolerances** — all numeric thresholds in a central `tolerances.rs` with
   documented provenance (paper reference, Python baseline).

5. **Streaming I/O** — use iterators (`FastqIter`, `MzmlIter`) not buffered parsers.
   IPC payloads can be large; stream them.

6. **`#![forbid(unsafe_code)]`** — zero unsafe blocks. Use safe Rust abstractions.
   If you need unsafe for performance, push it to BarraCUDA.

7. **Archive superseded handoffs** — move old handoffs to `handoffs/archive/` when
   new versions supersede them. The fossil record is preserved.

## Reference Files

| File | Purpose |
|------|---------|
| `graphs/wetspring_deploy.toml` | Deploy graph (Phase 4 standard) |
| `barracuda/src/ipc/mod.rs` | IPC server entry point |
| `barracuda/src/ipc/dispatch.rs` | Method routing (18 routes) |
| `barracuda/src/ipc/handlers/mod.rs` | Capability registry (19 entries) |
| `barracuda/src/ipc/provenance.rs` | Provenance trio integration |
| `barracuda/src/ipc/timeseries.rs` | Cross-spring time series |

## Quality Status

- `cargo check --features ipc,json` — clean
- `cargo clippy --features ipc,json -- -W clippy::pedantic -W clippy::nursery` — zero warnings
- `cargo test` — 1,326 tests pass, 0 fail
