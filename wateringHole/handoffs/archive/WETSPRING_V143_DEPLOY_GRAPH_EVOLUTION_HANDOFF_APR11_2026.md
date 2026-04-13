<!--
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# wetSpring V143 — Deploy Graph Evolution + Primal Composition Patterns

| Field | Value |
|-------|-------|
| **Spring** | wetSpring |
| **Version** | V143 |
| **Date** | 2026-04-11 |
| **barraCuda** | 0.3.11 |
| **Wire Standard** | L2 + L3 |
| **Proto-nucleate** | 141/141 (D01–D07) |
| **Deploy graphs** | 7 canonical (`[[graph.nodes]]`) |
| **Status** | All quality gates green |

---

## 1. What We Learned: Python → Rust → Primal Composition Validation

wetSpring's evolution path proves a repeatable three-tier validation model:

### Tier 1: Python Validates Rust (Science Fidelity)

58 frozen Python/R/Galaxy scripts produce reference outputs with SHA-256
hashes (`scripts/BASELINE_MANIFEST.md`). 1,950 Rust tests + 360 validation
binaries compare against these baselines with named, centralized tolerances
(`tolerances/mod.rs`, 4 sub-modules: bio, instrument, gpu, spectral).

**Pattern for springs:** Freeze your Python baselines. Hash them. Pin
tolerances by category (machine, instrument, GPU reorder). The baselines
are the science — they don't change unless the paper changes.

### Tier 2: Rust + Python Validate NUCLEUS Composition (Primal Patterns)

141 programmatic checks (Exp400, 7 domains) validate that the spring's
self-knowledge (`niche.rs`), deploy graphs, and proto-nucleate are aligned:

| Domain | What It Validates |
|--------|-------------------|
| D01 | Niche capabilities, dependencies, consumed capabilities |
| D02 | Capability surface completeness vs proto-nucleate |
| D03 | Proto-nucleate primal coverage (every node represented) |
| D04 | Deploy graph structural integrity (TOML parse, deps, ordering) |
| D05 | Composition model alignment (required nodes, health handlers) |
| D06 | Bonding metadata, ecology semantic mappings |
| D07 | Deploy graph metadata compliance (schema, fragments, bonding, owner) |

**Pattern for springs:** Wire your `niche.rs` to your proto-nucleate TOML at
test time. Cross-check capability counts, dependency names, and graph node
names. Add a guard constant (`EXPECTED_CHECKS`) so check-count drift fails CI.

### Tier 3: Composition → ecoBin Harvest

When all checks are green, the spring is a plasmidBin candidate. Build
`musl` static binaries, run `harvest.sh`, submit to `infra/plasmidBin/`.

---

## 2. Deploy Graph Patterns for All Springs

### Canonical Schema: `[[graph.nodes]]` (plural)

primalSpring NA-016 established `[[graph.nodes]]` as canonical. The parser
(`deploy/mod.rs`) accepts legacy `[[graph.node]]` but all new/updated graphs
should use plural. wetSpring V143 migrated all 7 graphs.

### Required Metadata

Every graph should carry:

```toml
[graph.metadata]
pattern_version = "2026-04-11"
schema = "canonical"
composition_model = "nucleated"        # or "pure" or "validation"
science_domain = "your_domain"
owner = "yourSpring"
fragments = ["tower_atomic", ...]      # which NUCLEUS atomics are present
```

### Bonding Policy (full-NUCLEUS graphs)

Graphs that compose all three atomics should declare:

```toml
[graph.bonding_policy]
bond_type = "Metallic"
trust_model = "InternalNucleus"
tower_internal = "covalent"
tower_to_node = "metallic"
tower_to_nest = "metallic"
encryption_tiers.tower = "full"
encryption_tiers.node = "delegated"
encryption_tiers.nest = "delegated"
```

### Node Completeness

Full-NUCLEUS graphs should declare all primals as nodes, even if `spawn = false`
and `required = false`. The graph is the composition declaration — biomeOS uses
it to understand what the spring expects to exist in its environment.

### Capability Strings — Use Proto-nucleate as Source of Truth

| Domain | Canonical Strings |
|--------|-------------------|
| Security | `crypto.sign_ed25519`, `crypto.verify_ed25519`, `crypto.blake3_hash` |
| Discovery | `discovery.find_primals`, `discovery.announce` |
| Compute | `compute.dispatch.submit`, `compute.execute` |
| Shader | `shader.compile.wgsl`, `shader.compile.spirv` |
| Math | `math.tensor`, `math.stats`, `math.spectral`, `math.fft` |
| Storage | `storage.store`, `storage.retrieve`, `storage.list` |
| DAG | `dag.session.create`, `dag.event.append`, `dag.merkle.root` |
| Ledger | `spine.create`, `entry.append`, `session.commit` |
| Attribution | `braid.create`, `braid.commit`, `provenance.graph` |
| AI | `ai.query`, `inference.complete`, `inference.embed` |
| Render | `render.dashboard`, `tui.push` |

---

## 3. Primal Evolution Asks

### For barraCuda (math primal)

- **No new asks** — 0.3.11 alignment is complete, 150+ primitives consumed
- wetSpring is the largest consumer of bio-domain GPU primitives (spectral,
  FFT, statistical clustering, ODE, phylogenetic, NMF, diversity)
- **Pattern discovered:** `upstream_contract.rs` pins the barraCuda version
  with a CI test. Other springs should adopt this pattern.

### For toadStool (compute dispatch)

- PG-05 still open: `discover_toadstool()` helper exists but no active
  compute dispatch calls via IPC. This becomes relevant at NUCLEUS deployment
  when compute requests route through toadStool rather than direct barraCuda
  path dependency.
- **Pattern for toadStool:** Publish `compute.dispatch.submit` and
  `compute.execute` as stable IPC endpoints. Springs will use them when
  they move from path-dependency compute to IPC-routed compute.

### For Songbird / biomeOS (discovery + orchestration)

- PG-03: Capability discovery is still name-based. wetSpring uses
  `discover_squirrel()`, `discover_toadstool()` etc. via primal name strings.
  True capability-first routing needs `capability.resolve` → socket path.
- **`consumed_capabilities`** field is ready for biomeOS to parse for
  deploy-time composition completeness validation.
- **`methods` flat array** (Wire Standard L2) — biomeOS v2.93+ should parse
  `result.methods` first and skip legacy format detection.
- **Deploy graph `fragments` metadata** — biomeOS can use this to understand
  what NUCLEUS atomics are present without parsing every node.

### For rhizoCrypt / loamSpine / sweetGrass (provenance trio)

- IPC wiring exists in wetSpring for all three trio primals. Once trio
  endpoints are stable, wetSpring routes to them transparently.
- `WireWitnessRef` format is ready for trio to accept and carry opaquely.
- **Pattern:** Provenance always degrades gracefully. When trio is unavailable,
  wetSpring falls back to local session tracking with `provenance: "local"`.

### For NestGate (storage)

- PG-04: Declared as optional niche dependency, no IPC integration yet.
  Local storage with BLAKE3 hashes works. Cross-spring data retrieval
  with provenance continuity requires NestGate IPC.

### For Squirrel (AI)

- `ai.ecology_interpret` routes ecology queries to Squirrel via Neural API
  socket with graceful degradation. When neuralSpring provides native WGSL
  ML inference, Squirrel will route to it — no wetSpring code changes.
- **Pattern:** Declare `inference.complete` and `inference.embed` as consumed
  capabilities. When Squirrel discovers neuralSpring as a provider, inference
  upgrades transparently.

### For petalTongue (visualization)

- IPC push client wired for real-time dashboard rendering. 15+ visualization
  scenarios ready (phylogenetics, ordination, calibration, pangenome, etc.).
- **Ask:** Client-side WASM compilation (GAPS.md #4) would eliminate server
  round-trip for visualization.

### For primalSpring (coordination)

- **Binary naming:** Proto-nucleate says `wetspring_primal`, actual binary is
  `wetspring`. Recommend aligning proto-nucleate to `wetspring`.
- **Composition validation pattern:** D07 metadata compliance is a pattern
  all springs can adopt. primalSpring could provide a shared
  `validate_deploy_graph_metadata()` function that springs call.
- **Proto-nucleate evolution:** As primals add capabilities, proto-nucleate
  graphs should evolve. The feedback loop: spring discovers gap → hands back
  via `PRIMAL_GAPS.md` → primalSpring updates proto-nucleate → spring
  validates new composition → cycle continues.

---

## 4. Patterns for Sibling Springs

### niche.rs Self-Knowledge Pattern

```rust
pub const NICHE_NAME: &str = "yourspring";
pub const CAPABILITIES: &[&str] = &[ ... ];
pub const CONSUMED_CAPABILITIES: &[&str] = &[ ... ];
pub const DEPENDENCIES: &[NicheDependency] = &[ ... ];
```

Cross-check at test time against proto-nucleate TOML. Guard with constant.

### Validation Binary Pattern (hotSpring heritage)

Every validation binary: hardcoded expected values, named tolerances from
centralized module, explicit PASS/FAIL, exit 0/1. Use `Validator` harness.

### Upstream Contract Pattern

Pin barraCuda version. CI test reads barraCuda's workspace `Cargo.toml` and
asserts `version >= pin`. Prevents silent precision drift.

### Socket Discovery Cascade

```
$NEURAL_API_SOCKET → $BIOMEOS_SOCKET_DIR → $XDG_RUNTIME_DIR/biomeos/ → temp_dir()
```

Socket name: `{primal}-default.sock` or `neural-api-{family_id}.sock`

### Provenance Degradation

Always `ProvenanceResult { available: bool, data: Option<...> }`. Never
hard-fail on missing provenance. Trio is optional — degrade to local tracking.

---

## 5. NUCLEUS Deployment via Neural API from biomeOS

### The Deployment Model

```
biomeos deploy --graph graphs/wetspring_science_nucleus.toml
```

biomeOS reads the graph, checks node health in dependency order, spawns
nodes with `spawn = true`, and validates composition via health probes.

### What Springs Need

1. `health.liveness` — minimal "I'm alive" probe
2. `health.readiness` — subsystems initialized
3. `capability.list` — Wire Standard L2+L3 envelope
4. `identity.get` — `{primal, version, domain, license}`
5. Composition health handlers (optional but recommended)

### Deploy Graph → Proto-nucleate → Niche Alignment

```
Proto-nucleate (primalSpring-owned, 14 nodes)
  ↓ validated at test time by niche.rs
Niche (spring-owned, 9 dependencies)
  ↓ drives IPC discovery
Deploy graph (spring-owned, 7 variants)
  ↓ consumed by biomeOS
NUCLEUS deployment (biomeOS-orchestrated)
```

---

## Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | Clean |
| `cargo clippy --workspace -D warnings` | 0 warnings |
| `cargo test --workspace` | 1,950 passed, 0 failed |
| Exp400 composition | 141/141 (D01–D07) |
| Deploy graphs | 7 canonical, bonding + fragments |
| Wire Standard | L2 + L3 |
| `forbid(unsafe_code)` | Enforced |
| `#[allow()]` in production | 0 |

---

*This handoff is maintained by wetSpring. Archived in
`wateringHole/handoffs/`. The infra/wateringHole/ handoff
(`WETSPRING_V143_DEPLOY_GRAPH_CANONICAL_COMPOSITION_HANDOFF_APR11_2026.md`)
covers the same changes for the broader ecosystem audience.*
