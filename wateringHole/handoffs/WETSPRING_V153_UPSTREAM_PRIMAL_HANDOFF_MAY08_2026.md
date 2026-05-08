# wetSpring V153 — Upstream Primal & Cross-Spring Handoff

**Date:** May 8, 2026
**Phase:** 60+ Deep Debt Evolution — Patterns for Absorption
**From:** wetSpring
**To:** primalSpring, all primal teams, all spring teams

---

## Purpose

This handoff documents everything wetSpring has learned from building the
full Paper → Python → Rust → NUCLEUS composition pipeline. It surfaces:

1. **Open primal gaps** that block evolution
2. **Composition patterns** that worked (absorb these)
3. **Composition friction** that needs upstream fixes
4. **Neural API / biomeOS patterns** for NUCLEUS deployment
5. **Downstream patterns** for other springs to replicate

---

## Section 1: Open Primal Gaps for Upstream Teams

### Critical (blocking Level 5 guideStone)

| Gap | Owner | Impact | Details |
|-----|-------|--------|---------|
| **PG-02**: Provenance Trio UDS | rhizoCrypt/loamSpine/sweetGrass | DAG sessions, braids non-functional | Trio primals accept UDS connections but reset on JSON-RPC. `capability.call("dag", "session.create")` returns empty. All trio wiring exists in wetSpring — just needs primal-side JSON-RPC handler. |
| **PG-09**: barraCuda IPC evaporation | wetSpring (wiring) + barraCuda (surface) | Library dep remains default | 33 v0.9.17 methods available via IPC; wetSpring still links in-process. Need IPC-only benchmark to prove acceptable latency. |
| **PG-18**: Trio UDS connection reset | rhizoCrypt/loamSpine/sweetGrass | Zero provenance tracking in composition | Same as PG-02 but discovered during Phase 46 composition. All three primals running (`pgrep` confirms) but don't speak JSON-RPC on UDS. |

### Moderate (needed for production NUCLEUS)

| Gap | Owner | Impact |
|-----|-------|--------|
| **PG-03**: Name-based discovery | Songbird/biomeOS | Socket resolution uses names, not capabilities |
| **PG-10**: spectral/linalg routing | primalSpring | `method_to_capability_domain()` missing `"spectral" | "linalg"` → `"tensor"` |
| **PG-14**: Squirrel inference | Squirrel | `inference.complete` needs Ollama backend |
| **PG-16**: stats.std_dev N-1 vs N | barraCuda + wetSpring | Convention mismatch (sample vs population) |
| **PG-17**: tensor.matmul handle-only | barraCuda | No inline data path — handle-based only |
| **PG-20/21**: socat dependency | primalSpring composition lib | `send_rpc` / health check require socat |
| **PG-22**: Songbird socket timeout | Songbird | Socket never appears during composition launch |

### Informational (documented, not blocking)

| Gap | Owner | Notes |
|-----|-------|-------|
| **PG-06**: Ionic bond protocol | primalSpring Track 4 | No negotiation protocol exists |
| **PG-08**: Validate manifest binary name | primalSpring | `wetspring` vs `wetspring_primal` |
| **PG-12**: Exp403 legacy surface | wetSpring | 15 legacy methods pending v0.9.17 migration |
| **PG-15**: toadStool compute.dispatch | toadStool | Needs compiled GPU binary input |
| **PG-19**: petalTongue scene format | petalTongue | Single-key enum variant documented |

---

## Section 2: Composition Patterns That Worked

### Pattern 1: Pure Primal Composition (No Fallbacks)

wetSpring routes ALL external data through primal composition:

```
Browser → Facade → biomeOS capability.call("storage", "fetch_external") → NestGate
```

If NestGate is offline, wetSpring returns a **structured gap report** — never
falls back to direct HTTP or disk. This is the correct pattern for all springs.

**Key files:** `barracuda/src/ipc/handlers/data_fetch.rs`

### Pattern 2: Provenance Trio Wire Names

Canonical `capability.call` tuples:

| Domain | Operation | Wire method |
|--------|-----------|-------------|
| `dag` | `session.create` | `dag.session.create` |
| `dag` | `event.append` | `dag.event.append` |
| `dag` | `dehydrate` | `dag.dehydrate` |
| `session` | `commit` | `session.commit` |
| `braid` | `create` | `braid.create` |
| `storage` | `fetch_external` | `storage.fetch_external` |
| `storage` | `store` | `storage.store` |
| `storage` | `retrieve` | `storage.retrieve` |

### Pattern 3: Gap Report Structure

When a primal is missing, return this instead of failing:

```json
{
  "gap_report": true,
  "missing_primals": [
    {
      "primal": "NestGate",
      "capability": "storage.fetch_external",
      "required_for": "TLS fetch + content-addressed caching",
      "deploy": "start NestGate with fetch_external capability"
    }
  ],
  "action": "hand to primalSpring for primal evolution"
}
```

### Pattern 4: Deploy Graph v3.0

`graphs/wetspring_science_nucleus.toml` uses canonical patterns:

- `[graph.metadata]` with `health_method` per node
- `capabilities` arrays with domain aliases (bare `"dag"`, `"spine"`,
  `"entry"` alongside dotted methods)
- Phase-ordered deployment (0: biomeOS → 1: Tower → 2: Nest → 3: Node →
  4: Trio → 5: Meta → 6: Niche → 8: Facade)

### Pattern 5: Registry Cross-Sync

`tools/check_registry_sync.sh` validates:
1. Every method in `capability_registry.toml` exists in primalSpring's canonical registry
2. Every dotted method string in Rust source appears in the local registry

**Finding:** 34 wetSpring domain methods need upstream absorption into
primalSpring's canonical 389-method registry.

### Pattern 6: Feature-Gated barraCuda

`Cargo.toml` pattern for IPC-first deployment:

```toml
[features]
default = ["barracuda-lib"]
barracuda-lib = ["dep:barracuda"]

[dependencies]
barracuda = { path = "...", optional = true }
```

Disable `default-features = false` for IPC-only NUCLEUS deployment where
barraCuda runs as a separate primal.

---

## Section 3: Composition Friction — Upstream Fixes Needed

### 3.1 biomeOS capability.call routing

`discover_capability()` uses **exact key lookup**. If a primal registers
`"dag.session.create"` but `capability.call` receives `capability: "dag"`,
the lookup fails. Fix: register bare domain aliases alongside dotted methods.

**Fix applied in primalSpring V082:** Added bare domain names to all 12
deploy graphs. Springs should do the same.

### 3.2 socat dependency

`nucleus_composition_lib.sh` requires `socat` for UDS JSON-RPC. wetSpring
added `tools/uds_send.py` as fallback. Candidate for upstream promotion to
primalSpring composition lib.

### 3.3 Method name consistency

primalSpring's `method_to_capability_domain()` maps `tensor`, `stats`,
`math`, etc. to `"tensor"` but misses `spectral` and `linalg`. Springs
using these prefixes get incorrect routing.

**Fix:** Add `"spectral" | "linalg"` to the tensor match arm.

---

## Section 4: Neural API / biomeOS Patterns

### Socket Discovery

```rust
fn resolve_socket(env_var: &str, filename: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        return Some(PathBuf::from(p));
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    Some(PathBuf::from(runtime).join("biomeos").join(filename))
}
```

### JSON-RPC 2.0 over UDS

All primal communication uses newline-delimited JSON-RPC 2.0 over Unix
domain sockets. Request format:

```json
{"jsonrpc":"2.0","method":"capability.call","params":{"capability":"storage","operation":"fetch_external","args":{...}},"id":1}
```

### Health Probes

Every primal should respond to `health.check` with at minimum:
```json
{"status": "ok", "methods": <count>}
```

---

## Section 5: Downstream Patterns for Springs

### What Other Springs Should Absorb from wetSpring

| Pattern | Where | Why |
|---------|-------|-----|
| Paper notebooks (Tier 1/2/3 evolution) | `notebooks/papers/*.ipynb` | Publishable, frozen → live → primals |
| Gap report structure | `ipc/handlers/data_fetch.rs` | Don't fail on missing primals |
| Registry cross-sync test | `tools/check_registry_sync.sh` | Catch method drift |
| barraCuda optional feature | `barracuda/Cargo.toml` | IPC-first deployment |
| `ring` crate ban | `deny.toml` | Security: use BearDog crypto |
| exp400 composition experiment | `experiments/exp400_*/` | NUCLEUS parity validation |
| Shared validation timing | `validation/timing.rs` | `CpuGpuRow`, `CrossSpringEntry`, `BenchRow` |
| Env-configurable bind/CORS | `bin/wetspring_science_facade.rs` | No hardcoded server config |

### Spring-Specific Recommendations

| Spring | Recommendation |
|--------|---------------|
| **hotSpring** | Add deploy graphs (only 1 vs 7), replicate exp400 for physics niche |
| **healthSpring** | Create `capability_registry.toml` (uses Rust constants only), convert 54 Python scripts to notebooks |
| **neuralSpring** | Advance guideStone to L4-L5, resolve 18 barraCuda IPC gaps |
| **ludoSpring** | Add registry cross-sync test, create paper notebooks (0 .ipynb currently) |
| **groundSpring** | Advance guideStone to L4+, create composition experiment crates |
| **airSpring** | Create `capability_registry.toml`, add primalSpring dep, advance guideStone to L3+ |

---

## Section 6: wetSpring Primal Usage Summary

### Primals Consumed (Runtime Composition)

| Primal | Usage | Methods |
|--------|-------|---------|
| **biomeOS** | Orchestration, capability routing | `capability.call`, `health.check` |
| **BearDog** | Crypto hashing, consent verification | `crypto.hash`, `security.verify_consent` |
| **Songbird** | Discovery, registration | Socket discovery |
| **NestGate** | External fetch, storage, retrieval | `storage.fetch_external`, `storage.store`, `storage.retrieve` |
| **barraCuda** | Math (in-process + IPC) | 33 v0.9.17 canonical methods |
| **toadStool** | GPU dispatch, performance surface | `compute.dispatch`, `compute.health` |
| **rhizoCrypt** | DAG session management | `dag.session.create`, `dag.event.append`, `dag.dehydrate` |
| **loamSpine** | Ledger commits | `session.commit` |
| **sweetGrass** | Semantic braids | `braid.create`, `braid.commit` |
| **Squirrel** | AI inference | `inference.complete` |
| **petalTongue** | Visualization rendering | `visualization.render.scene` |
| **coralReef** | Shader compilation | Discovery only |

### wetSpring Domain Methods (34 registered)

Science: `science.diversity`, `science.qs_model`, `science.gonzales.*`,
`science.anderson.*`, `science.alignment`, `science.pfas_classify`, etc.

Data: `data.fetch.chembl`, `data.fetch.pubchem`, `data.fetch.ncbi`

Vault: `vault.store`, `vault.retrieve`, `vault.consent.verify`

These 34 methods need absorption into primalSpring's canonical registry.

---

## Section 7: Benchmark & Dataset Status

### Benchmarks (Complete)

- 23 Rust benchmark binaries
- 3 Python benchmark scripts
- Galaxy/QIIME2, DADA2, R/vegan, phyloseq industry parity
- 63/63 papers reproduced across 6 tracks

### Datasets (Roadmap)

- **Active**: ChEMBL, PubChem, NCBI SRA, Galaxy test data
- **P0 (blocked on NestGate)**: HMP, Tara Oceans, EMP, Lake Erie HAB, cold seep metagenomes
- **Reference-only**: Published values from Gonzales, Waters, Liao papers

### Downstream Projects

- **projectNUCLEUS**: wetSpring provides science workloads for ironGate deployment
- **foundation**: Provenance chain is canonical example of scientific knowledge lineage

---

*This handoff is the definitive upstream document for primalSpring's next
audit cycle. All gaps, patterns, and recommendations are current as of
V153 (May 8, 2026). primalSpring should absorb the 34 domain methods,
fix the 3 critical gaps (PG-02/18 trio UDS, PG-09 IPC evaporation),
and propagate the composition patterns to all springs.*
