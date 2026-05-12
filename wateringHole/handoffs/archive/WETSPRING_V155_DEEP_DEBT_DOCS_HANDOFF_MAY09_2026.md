# wetSpring V155 — Deep Debt Resolution + Documentation Sync

**Date:** 2026-05-09  
**From:** wetSpring  
**To:** primalSpring, primal teams (barraCuda, toadStool, NestGate, BearDog, sweetGrass, rhizoCrypt, loamSpine), downstream springs  
**primalSpring version:** v0.9.25  
**Commit:** 07f2ffb (code) + this handoff

---

## Summary

V155 closes the deep debt evolution wave opened by V153. All remaining lint
policy gaps, hardcoded URLs, and idiomatic Rust issues are resolved. The full
workspace passes `cargo build + fmt --check + clippy --workspace --all-targets +
test --workspace --lib` with **zero warnings** and **zero failures**.

---

## What Changed (V155)

| Item | Before | After |
|------|--------|-------|
| `#[expect()]` policy | 193 bare attributes (no `reason`) | All 193 carry formal `reason = "..."` |
| Data-source URLs | ChEMBL, PubChem hardcoded in handler code | `WETSPRING_CHEMBL_BASE_URL`, `WETSPRING_PUBCHEM_BASE_URL`, `WETSPRING_PLASMID_BIN_URL` with defaults |
| `Vec<&String>` in unifrac | Heap-allocated key references | `Vec<&str>` via `String::as_str` |
| `eprintln!` in certification | stderr in library code | `tracing::warn!` |
| exp400 signatures | `&Option<PathBuf>` (14 functions) | `&Path` (idiomatic, zero clippy warnings) |
| exp400 `resolve_socket` | Returns `Option<PathBuf>` (always `Some`) | Returns `PathBuf` directly |
| Clippy workspace | Warnings in exp400/benchmarks | **Zero warnings** workspace-wide |
| CHANGELOG | Missing V152–V154 | V152–V155 entries added |
| wateringHole archive | V151 handoffs in active dir, counts wrong | V151 archived (167 total), counts corrected |

---

## Build Gate Results

| Gate | Result |
|------|--------|
| `cargo build` | clean |
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets` | 0 warnings |
| `cargo test --workspace --lib` | 252 passed, 0 failed, 1 ignored |
| `cargo-deny check` | advisories ok, bans ok, licenses ok, sources ok |

---

## Primal Integration Status

### Discovery Patterns

| Pattern | Where | Status |
|---------|-------|--------|
| `CompositionContext::from_live_discovery_with_fallback()` | certification/, validation/scenarios/, wetspring_unibin | **Active** — new code |
| `ctx.call()` | certification/health.rs, wetspring_unibin | **Active** — primalSpring v0.9.25 pattern |
| `discover_primal()` | validate_primal_parity_v1.rs, validate_emp_anderson_atlas.rs | **Legacy** — valid low-level primitive for explicit socket resolution |
| `capability_call()` (Neural API) | ipc/provenance/{rhizocrypt,loamspine,sweetgrass}.rs | **Active** — trio witness protocol |
| `spawn_primal` / `probe_primal` / `AtomicHarness` / `PrimalClient` | nowhere | **Removed** |

### Feature Flags

| Feature | Purpose | `cfg` Wired? |
|---------|---------|:------------:|
| `ipc` | Enables IPC server, handlers, dispatch | Yes |
| `gpu` | Enables wgpu GPU modules | Yes |
| `vault` | Enables BearDog crypto vault | Yes |
| `guidestone` | Enables certification + scenarios + UniBin | Yes |
| `barracuda-lib` | Links barraCuda as library (default) | Yes |
| `primal-proof` | IPC-first compute routing | **Declared only** — no `cfg` branching yet |

### IPC Handler Architecture

wetSpring's 37 JSON-RPC handlers currently dispatch to **in-process library
calls** (`crate::bio::*`, `barracuda::*`). The `primal-proof` feature flag is
declared for future IPC-first routing but has **no `cfg` wiring** — handlers
remain library-backed regardless of feature state.

**Remaining evolution target**: gate the 15 science handlers behind
`#[cfg(not(feature = "primal-proof"))]` with IPC-first alternatives behind
`#[cfg(feature = "primal-proof")]`, allowing sovereign NUCLEUS deployments
where wetSpring routes all compute through ecobin barraCuda over JSON-RPC.

### Binary Inventory

| Category | Count |
|----------|:-----:|
| Total `[[bin]]` targets | 343 |
| Prokaryotic validation/benchmark | ~340 |
| Eukaryotic (UniBin) | 1 (`wetspring_unibin`) |
| Prokaryotic superseded (still ships) | 1 (`wetspring_guidestone`) |
| Primary IPC server | 1 (`wetspring`) |

---

## Upstream Primal Handoff

### For primalSpring

1. **`primal-proof` implementation guidance needed**: wetSpring has declared the
   feature but needs a canonical pattern for dual-lane handler dispatch
   (`library` vs `IPC`). Should this be a per-handler `#[cfg]` gate, a runtime
   strategy pattern, or a middleware layer? Request: add a reference
   implementation to primalSpring's composition examples.

2. **`CompositionContext::discovered_count()`**: wetspring_unibin `status`
   subcommand wants to report how many primals were discovered. Current
   `CompositionContext` doesn't expose this. Request: add a `discovered_count()`
   or `discovered_primals()` accessor.

3. **UniBin `serve` subcommand pattern**: wetspring_unibin's `serve` is a stub.
   Should it absorb the existing `wetspring` IPC server binary, or should it
   delegate to it? Request: document the canonical UniBin `serve` pattern.

4. **Guidestone → UniBin naming**: `niche.rs` still exports
   `GUIDESTONE_BINARY = "wetspring_guidestone"` while UniBin is
   `wetspring_unibin`. Need guidance on when to update downstream manifests
   and whether `GUIDESTONE_BINARY` should become `UNIBIN_BINARY`.

### For barraCuda

1. **IPC-first routing**: When `primal-proof` is wired, wetSpring will route
   compute through barraCuda ecobin over JSON-RPC. barraCuda's ecobin server
   needs to expose all 150+ primitives wetSpring currently consumes as library
   calls. Current barraCuda IPC surface may need expansion.

2. **GPU module coverage**: wetSpring consumes 44 GPU modules via barraCuda
   v0.3.12. All are lean (zero local WGSL). No new shader requirements from
   V155.

### For NestGate / BearDog / Provenance Trio

1. **Env-configurable endpoints**: wetSpring now supports overriding all
   external data-source URLs via env vars. NestGate fetch_external still
   receives the canonical URL as the `target` — this is correct (NestGate
   owns TLS). No action needed unless NestGate wants to validate targets
   against a registry.

2. **Provenance trio per-module split**: `ipc/provenance/{rhizocrypt,loamspine,
   sweetgrass}.rs` mirrors the trio's witness model. Each sub-module speaks
   directly to its counterpart via `capability_call`. No interface changes
   needed.

### For Other Springs

Patterns wetSpring has validated that other springs should absorb:

1. **Formal `#[expect(reason)]`**: Use `reason = "..."` on every `#[expect()]`.
   Never bare, never `#[allow()]` in production `.rs` files. Script available
   in commit history for bulk conversion.

2. **Env-configurable external URLs**: Every hardcoded URL should have an env
   var override with a sensible default. Pattern: `const DEFAULT: &str = "...";
   fn base() -> String { env::var("SPRING_URL").unwrap_or_else(|_| DEFAULT.to_owned()) }`.

3. **`&Option<T>` → `Option<&T>` or `&T`**: Clippy `ref_option` is nursery
   but catches real API design issues. If your function always has a value,
   use `&T` directly.

4. **UniBin consolidation**: Start with `certify` + `validate` + `status` +
   `version` subcommands. `serve` can be a stub initially. Fossilize the old
   guidestone to `fossilRecord/`.

---

## Downstream Absorption

Systems that consume wetSpring patterns or outputs:

| Consumer | What They Get | Absorption Status |
|----------|---------------|-------------------|
| **primals.eco** | 42 niche capabilities via IPC, 5 meta-notebooks, 11 paper notebooks | Active — science API surface stable |
| **plasmidBin** | wetSpring ecobin packaging | V154 deploy graphs ready (7 canonical) |
| **sporeprint** | Experiment results, validation reports | 380 completed experiments, 63/63 papers |
| **primalSpring** | Evolution status, gap reports, composition patterns | This handoff + GAPS.md + PRIMAL_GAPS.md |
| **biomeOS** | niche manifest, deploy graphs, capability registry | 42 niche caps, 48 consumed, 37 dispatch |

---

## Remaining Evolution Targets (V156+)

1. **Wire `primal-proof` feature**: Gate handler compute behind `cfg` for
   sovereign NUCLEUS deployment (15 science handlers).
2. **UniBin `serve`**: Absorb wetspring IPC server into `wetspring_unibin serve`.
3. **Binary consolidation**: Selectively absorb most-exercised validation
   binaries as UniBin scenarios.
4. **`GUIDESTONE_BINARY` → `UNIBIN_BINARY`**: Pending primalSpring guidance
   on downstream manifest naming.
5. **Update `ECOSYSTEM_LEVERAGE_GUIDE.md`**: Still anchored at V128 with V150
   note — needs refresh to V155 metrics.

---

*This handoff feeds into ecoPrimals/wateringHole for ecosystem-wide tracking.
primalSpring will audit and route action items to primal teams.*
