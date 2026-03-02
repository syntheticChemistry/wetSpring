# wetSpring → ecoPrimals Handoff V91 — Deep Debt Resolution & Idiomatic Rust Evolution

**Date**: March 2, 2026
**From**: wetSpring (V91)
**To**: ecoPrimals team
**ToadStool pin**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later
**Supersedes**: V90 (Bio Brain Cross-Spring Ingest)

---

## Executive Summary

- **Capability-based discovery**: Unified socket discovery into `ipc::discover` module — eliminated 3 copies of duplicated discovery logic across `server.rs`, `songbird.rs`, and `nestgate/discovery.rs`. All primal discovery now uses the same env-var + XDG + fallback protocol. Standalone fallback for non-IPC builds.
- **Handlers refactored**: Monolithic `ipc/handlers.rs` (605 lines) split into domain-focused sub-modules: `handlers/mod.rs` (health + GPU init + helpers), `handlers/brain.rs` (brain pipeline), `handlers/science.rs` (diversity + QS + NCBI + Anderson + full pipeline). GPU/CPU diversity duplication collapsed from 8 functions to 1 generic.
- **Idiomatic Rust evolution**: `#[must_use]` added to `gillespie`, `pcoa`; `as` casts replaced with `From`/`TryFrom` + explicit `#[allow]` annotations with safety comments in `unifrac/flat_tree.rs` and `ncbi_data/vibrio.rs`.
- **Test coverage expanded**: 5 new brain handler dispatch tests (22 total dispatch tests). 1,088 total lib tests passing (up from 1,083).
- **Quality gates**: `cargo fmt` ✓, `cargo clippy --pedantic --nursery` ✓ (zero warnings), `cargo test --lib --features ipc,nautilus` 1,088/1,088, Exp272 64/64.

---

## Part 1: Capability-Based Discovery

### Before (V90)

```
server.rs:      DEFAULT_SOCKET_PATH_XDG, DEFAULT_SOCKET_PATH_FALLBACK, resolve_bind_path()
songbird.rs:    DEFAULT_SOCKET_PATH_XDG, DEFAULT_SOCKET_PATH_FALLBACK, discover_socket()
nestgate/:      DEFAULT_NESTGATE_PATH_*, DEFAULT_BIOMEOS_PATH_*, resolve_socket(), discover_*()
```

Three independent implementations of the same 3-tier discovery pattern.

### After (V91)

```
ipc/discover.rs:  resolve_bind_path(env_var, primal), discover_socket(env_var, primal)
                  resolve_socket_explicit(explicit, xdg, subpath, fallback)
server.rs:        → discover::resolve_bind_path("WETSPRING_SOCKET", PRIMAL_NAME)
songbird.rs:      → discover::discover_socket("SONGBIRD_SOCKET", "songbird")
nestgate/:        → discover::discover_socket("NESTGATE_SOCKET", "nestgate")
```

Single unified implementation. Each primal discovered by `(env_var, primal_name)` pair. No hardcoded absolute paths. Self-knowledge only.

### Files

| File | Change |
|------|--------|
| `ipc/discover.rs` (NEW) | Unified discovery: `resolve_bind_path`, `discover_socket`, `resolve_socket_explicit`, 4 unit tests |
| `ipc/mod.rs` | Registered `discover` module |
| `ipc/server.rs` | Removed 2 constants, `resolve_bind_path` delegates to `discover::resolve_bind_path` |
| `ipc/songbird.rs` | Removed 2 constants + 15 lines, `discover_socket` delegates to `discover::discover_socket` |
| `ncbi/nestgate/discovery.rs` | Removed 4 constants + 70 lines of duplicated logic, delegates to `discover` with standalone fallback for non-IPC builds |

---

## Part 2: Handler Refactoring

### Before (V90)

`ipc/handlers.rs` — 605 lines, one file containing health, diversity (8 cfg-duplicated metric functions), QS model, NCBI fetch, Anderson, full pipeline, brain observe/attention/urgency, helpers.

### After (V91)

| File | Lines | Domain |
|------|:-----:|--------|
| `ipc/handlers/mod.rs` | ~105 | Health, GPU init, helpers (`extract_f64_array`, `extract_string_array`), re-exports |
| `ipc/handlers/brain.rs` | ~130 | `brain.observe`, `brain.attention`, `brain.urgency`, `parse_bio_observation` |
| `ipc/handlers/science.rs` | ~260 | `science.diversity`, `science.qs_model`, `science.ncbi_fetch`, `science.anderson`, `science.full_pipeline` |

**Key improvement**: 8 duplicated `insert_*_if_requested` functions (4 GPU + 4 CPU variants) collapsed into 1 generic `insert_metric_if_requested` that takes `cpu_fn` + conditional `gpu_fn`.

---

## Part 3: Idiomatic Rust Improvements

| Category | Files Changed | Detail |
|----------|--------------|--------|
| `#[must_use]` | `gillespie.rs`, `pcoa.rs` | Added to `Lcg64::next_u64/next_f64/exp_variate`, `pcoa()` |
| `as` casts | `unifrac/flat_tree.rs`, `ncbi_data/vibrio.rs` | Explicit `#[allow(clippy::cast_possible_truncation)]` with safety comments for usize→u32 (tree nodes always small) |
| device.clone() | N/A | Confirmed correct — `Arc<WgpuDevice>` clones are cheap ref-count ops |

---

## Part 4: Test Coverage

| Change | Before | After |
|--------|:------:|:-----:|
| Dispatch tests | 17 | 22 |
| New brain tests | 0 | 5 (observe, insufficient heads, attention, urgency, observe→attention) |
| Total lib tests | 1,083 | 1,088 |

---

## Part 5: Audit Findings (Pre-existing, Documented)

| Category | Status | Notes |
|----------|--------|-------|
| Unsafe code | CLEAN | `#![deny(unsafe_code)]` crate-wide |
| `todo!()`/`unimplemented!()` | CLEAN | Zero in production |
| `unwrap()`/`expect()` in lib | CLEAN | All in tests/binaries only |
| Production mocks | CLEAN | Zero — all mocks isolated to test |
| External C deps | MINIMAL | Only `wgpu` (optional, necessary for GPU) |
| AGPL-3.0-or-later | CLEAN | All files have SPDX headers |

---

## Part 6: Quality Gate

| Gate | Status |
|------|--------|
| `cargo fmt` | PASS |
| `cargo clippy --all-features -W pedantic -W nursery` | PASS (zero warnings) |
| `cargo test --lib --features ipc,nautilus` | 1,088 passed, 0 failed, 1 ignored |
| Exp272 (64 checks / 7 domains) | 64/64 PASS |
| Zero `unsafe` code | PASS |
| Zero `todo!()`/`unimplemented!()` | PASS |
