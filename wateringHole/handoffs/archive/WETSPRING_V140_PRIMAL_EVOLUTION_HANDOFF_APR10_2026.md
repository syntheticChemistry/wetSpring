# WETSPRING V140 — Primal Evolution + Composition Pattern Handoff

| Field | Value |
|-------|-------|
| Date | 2026-04-10 |
| From | wetSpring V140 |
| To | primalSpring, barraCuda, toadStool, biomeOS, coralReef, sweetGrass, Squirrel |
| License | AGPL-3.0-or-later |

## Purpose

This handoff documents what wetSpring learned about primal composition during
V138-V140 and hands patterns, gaps, and evolution requests back to the primal
teams. The goal: every primal team can read their section, absorb what's
relevant, and evolve their interfaces.

---

## 1. Composition Patterns Proven

wetSpring validated these NUCLEUS composition patterns end-to-end. Any spring
can adopt them directly.

### 1.1 Deploy Graph Pattern

```toml
[graph]
name = "myspring_science_nucleus"
version = "3.0.0"
coordination = "sequential"

[[graph.node]]
name = "beardog"
order = 1
required = true
by_capability = "security"
capabilities = ["crypto.sign_ed25519"]
depends_on = []
```

**Key findings:**
- Use `[[graph.node]]` not `[[nodes]]` — the `graph_validate.rs` parser expects the canonical schema
- Every node must have `by_capability` for capability-first discovery
- `order` field drives germination sequence; `depends_on` enforces dependency closure
- `required = false` with graceful degradation works for optional primals (toadStool, Squirrel, petalTongue)

### 1.2 Composition Health Pattern

Five-handler pattern proven by wetSpring:

```
composition.science_health  → domain-specific subsystem check
composition.tower_health    → BearDog + Songbird
composition.node_health     → BearDog + ToadStool
composition.nest_health     → BearDog + NestGate
composition.nucleus_health  → Full aggregate (Tower + Node + Nest + provenance trio)
```

Each handler returns:
```json
{
  "healthy": true,
  "atomic": "NUCLEUS",
  "spring": "wetSpring",
  "tiers": { "tower": true, "node": true, "nest": true, "provenance_trio": true },
  "components": { "beardog": {...}, "songbird": {...}, ... }
}
```

**Recommendation to primalSpring:** standardize this 5-handler pattern in
`PRIMALSPRING_COMPOSITION_GUIDANCE.md` so all springs implement it consistently.

### 1.3 Capability Discovery Pattern

```rust
fn probe_capability(domain: &str) -> Option<Value> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let family_id = std::env::var("FAMILY_ID").ok()?;
    let socket = PathBuf::from(runtime_dir())
        .join("biomeos")
        .join(format!("neural-api-{family_id}.sock"));

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.discover",
        "params": { "domain": domain },
        "id": 1,
    });
    // ... send/receive/parse
}
```

**Pattern:** discover by capability domain, not by primal identity. The
`by_capability` field in deploy graphs maps to `capability.discover` calls.

### 1.4 Socket Discovery Precedence

```
1. $WETSPRING_SOCKET (explicit override)
2. $XDG_RUNTIME_DIR/biomeos/wetspring-$FAMILY_ID.sock
3. $TMPDIR or /tmp (fallback)
```

### 1.5 Validation Chain (Paper → Code → Primal)

```
GET /api/v1/validation/chain/:paper_id → {
  "chain": {
    "source": { "doi": "...", "tables": [...] },
    "python_baseline": { "path": "...", "hash": "..." },
    "rust_validation": { "binary": "...", "checks": 14 },
    "nucleus_composition": { "computation": {...}, "provenance": {...} }
  }
}
```

The validation chain endpoint proves the three-tier evolution:
published paper → Python baseline → Rust validation → live NUCLEUS computation.

---

## 2. Per-Primal Findings and Requests

### 2.1 barraCuda

**What wetSpring consumes:** 150+ primitives from `barracuda` v0.3.7
(stats, linalg, ops, dispatch, nn, spectral, nautilus). Zero local math.
Zero local WGSL. Full lean achieved.

**Bug found:** `tolerances/precision.rs:138` references `crate::device`
without `#[cfg(feature = "gpu")]` gating. This causes `cargo test -p barracuda`
to fail when `gpu` feature is not active in the downstream consumer but is
active in the workspace resolver. **Workaround:** test via `cargo test --workspace`.
**Request:** gate the device import behind `#[cfg(feature = "gpu")]`.

**Upstream request:** When `PrecisionTier` gains new variants (e.g., `F16`),
downstream `match` expressions break. Consider adding `#[non_exhaustive]` to
the enum or documenting new variant additions in the changelog.

### 2.2 toadStool

**What wetSpring consumes:** `toadstool-hw-safe` for hardware discovery,
`akida-driver` for NPU access. Listed as optional dependency in niche.

**Finding:** The `akida-driver` path dependency uses `version = "*"` which
triggers `cargo-deny` wildcard warnings in downstream consumers. **Request:**
pin to a real version or use `version = "0.1"` in `akida-driver/Cargo.toml`.

### 2.3 biomeOS / Neural API

**Pattern proven:** wetSpring's `probe_capability()` successfully discovers
primals via `capability.discover` through the Neural API Unix socket.

**Gap:** No `BIOMEOS_NEURAL_API_SOCKET` environment variable override exists.
Socket path is derived from `FAMILY_ID` + `XDG_RUNTIME_DIR`. If biomeOS runs
in a non-standard location, there's no way to override the socket path.
**Request:** add `BIOMEOS_NEURAL_API_SOCKET` override to biomeOS.

**Deployment pattern confirmed:**
```bash
biomeos deploy --graph graphs/wetspring_science_nucleus.toml
```
Sequential germination: Tower → Nest → Node → Science → Validation.

### 2.4 sweetGrass / rhizoCrypt / loamSpine (Provenance Trio)

**What wetSpring does:** `provenance::envelope()` produces Tier 1 (local)
provenance. When the trio is available, Tier 3 (full braid) is attempted
with a circuit breaker (3 failures → fallback to Tier 1).

**Pattern:** Graceful degradation. The trio is optional — science works
without it, provenance degrades gracefully. This is the right pattern for
all springs.

### 2.5 Squirrel (AI)

**What wetSpring does:** Added to niche dependencies as optional. AI
capabilities (`ai.ecology_interpret`) are routed through IPC when Squirrel
is available. Falls back to deterministic computation when absent.

**Pattern:** Zero spring code changes needed. Squirrel discovers
`neuralSpring` as a provider; springs just call `capability.discover("ai")`.

### 2.6 petalTongue (Visualization)

**What wetSpring does:** Grammar-of-graphics rendering via RPC. 6 grammar
renderers (dose-response, PK decay, tissue lattice, hormesis, cross-species,
biome atlas). Falls back to returning raw JSON when petalTongue is unavailable.

### 2.7 coralReef

**What wetSpring does:** Sovereign probe check. `sovereign-dispatch` feature
gate for future coralReef native compilation path.

**Gap:** No `compute.shader.status` capability exposed by coralReef yet.
wetSpring probes for sovereign availability but can't query shader compilation
status.

---

## 3. Deploy Graph Evolution

### What We Learned

1. **Schema consistency matters.** Having 6 graphs in `[[graph.node]]` and 1
   in `[[nodes]]` broke the structural validator. All 7 are now canonical.

2. **`graph_validate.rs` catches real bugs.** Duplicate node names, missing
   `by_capability`, broken dependency closure, order inconsistencies — all
   caught before deployment.

3. **The `graph.metadata` section is useful.** Pattern version and witness
   wire format should be standardized across all spring deploy graphs.

### Recommended Deploy Graph Standard

```toml
[graph]
name = "..."
version = "X.Y.Z"
description = "..."
coordination = "sequential"

[graph.metadata]
pattern_version = "2026-04-10"
witness_wire = "WireWitnessRef"  # optional

[[graph.node]]
name = "..."
binary = "..."     # optional if spawn = false
order = N
required = true/false
depends_on = [...]
by_capability = "..."
capabilities = [...]
health_method = "health.liveness"  # optional
```

---

## 4. Gaps Handed Back

### To primalSpring

| # | Gap | Action |
|---|-----|--------|
| 1 | Deploy graph schema documentation | Document the canonical `[[graph.node]]` schema |
| 2 | 5-handler composition health standard | Add to PRIMALSPRING_COMPOSITION_GUIDANCE.md |
| 3 | `coordination.validate_composition` | Confirm `primalspring_primal` implements this |
| 4 | Tolerance provenance standard | Consider standardizing `tolerance_provenance.toml` pattern |

### To barraCuda

| # | Gap | Action |
|---|-----|--------|
| 1 | `tolerances/precision.rs` feature gate | Gate `crate::device` behind `#[cfg(feature = "gpu")]` |
| 2 | `PrecisionTier` variant additions | Document new variants; consider `#[non_exhaustive]` |

### To toadStool

| # | Gap | Action |
|---|-----|--------|
| 1 | `akida-driver` version wildcard | Pin to real semver in Cargo.toml |

### To biomeOS

| # | Gap | Action |
|---|-----|--------|
| 1 | `BIOMEOS_NEURAL_API_SOCKET` override | Add env var override for non-standard socket locations |

### To coralReef

| # | Gap | Action |
|---|-----|--------|
| 1 | `compute.shader.status` capability | Expose shader compilation status via IPC |

---

## 5. ecoBin Harvest Path

wetSpring is harvest-ready for `infra/plasmidBin/`:

```bash
cd ecoPrimals/springs/wetSpring
cargo build --release --features ipc --bin wetspring
cp target/release/wetspring ../../infra/plasmidBin/wetspring/wetspring

cd ../../infra/plasmidBin
./harvest.sh wetspring --tag v0.9.0
```

Harvested ecoBin characteristics:
- Pure Rust, zero C application dependencies
- `forbid(unsafe_code)` — no unsafe in the binary
- Cross-compile ready (linux-x86_64, linux-aarch64)
- Self-contained: IPC server, science compute, composition health, provenance
- Configurable via environment variables (no config files required)

---

## 6. Pattern Library for Other Springs

These patterns are ready for adoption by hotSpring, airSpring, groundSpring,
healthSpring, neuralSpring, ludoSpring:

1. **Niche self-knowledge** (`niche.rs`): NICHE_NAME, NICHE_DESCRIPTION, DEPENDENCIES, CAPABILITIES
2. **Capability registry** (`capability_registry.toml`): machine-readable capability surface
3. **Proto-nucleate alignment**: validation binary checking IPC surface against proto-nucleate graph
4. **Tolerance provenance trail**: machine-readable link from tolerances to derivation sources
5. **CI orchestrator**: `check_all.sh` (fmt → clippy → test → deny → coverage → baselines)
6. **Reproduction manifest**: pinned primal versions for exact reproduction
7. **Dark Forest middleware**: Axum middleware for BearDog token verification
8. **Validation chain endpoint**: paper → Python → Rust → live NUCLEUS computation

---

*License: AGPL-3.0-or-later*
