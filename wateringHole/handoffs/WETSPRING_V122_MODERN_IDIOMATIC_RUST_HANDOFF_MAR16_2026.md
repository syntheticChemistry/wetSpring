# WETSPRING V122 — Modern Idiomatic Rust Evolution + Absorption Handoff

**Date:** 2026-03-16
**From:** wetSpring V122
**To:** barraCuda, toadStool, All Springs
**License:** AGPL-3.0-or-later
**Covers:** V121 → V122 (full `#[expect(reason)]` migration, idiomatic Rust evolution, test coverage)

---

## Executive Summary

- **Zero `#[allow()]` in entire codebase** — production, validation, and test code
- **276+ validation binaries** migrated from `#[allow(lint)]` to `#[expect(lint, reason = "...")]`
- **1,139 stale `#[expect()]` lines** removed via automated clippy JSON analysis
- **18 new forge tests** (error Display/Error, bridge edge cases, nest discovery)
- **Idiomatic Rust evolution**: redundant closures, `Path::extension()`, struct patterns, unsafe removal
- **1,703 tests pass**, 354 binaries, 376 experiments, 5,707+ validation checks

---

## Part 1: What Changed (V121 → V122)

### `#[expect(reason)]` Migration (298 files, +3,593 −1,628 lines)

Every `#![allow(lint)]` in every validation and benchmark binary has been converted to
`#![expect(lint, reason = "...")]` with a curated reason dictionary:

| Lint | Reason |
|------|--------|
| `clippy::expect_used` | "validation harness: fail-fast on setup errors" |
| `clippy::unwrap_used` | "validation harness: fail-fast on setup errors" |
| `clippy::cast_possible_truncation` | "validation harness: small-range numeric conversions" |
| `clippy::cast_sign_loss` | "validation harness: non-negative values cast to unsigned" |
| `clippy::cast_precision_loss` | "validation harness: counter/timing values within f64 range" |
| `clippy::cast_lossless` | "validation harness: explicit cast for readability" |
| `clippy::module_name_repetitions` | "ecosystem convention: primal modules use domain-qualified names" |
| `clippy::similar_names` | "validation harness: domain-specific nomenclature" |
| `clippy::too_many_lines` | "validation harness: comprehensive multi-domain validation" |
| `clippy::cognitive_complexity` | "validation harness: sequential validation steps" |

After initial migration, `cargo clippy --message-format=json` was parsed to identify
1,139 `unfulfilled_lint_expectations` — these were `#[expect()]` attributes for lints that
didn't actually fire, indicating the original `#[allow()]` was unnecessary. All were
automatically removed, leaving only expectations that are genuinely needed.

### Test Coverage (forge: 234 → 252)

| File | Tests Added | Coverage |
|------|:-----------:|----------|
| `error.rs` | 10 | Display/Error for NestError, SongbirdError, AssemblyError, NcbiError, DataError |
| `bridge.rs` | 3 | `estimated_transfer_us` zero bytes, `detect_bandwidth_tier` unknown/non-GPU |
| `nest/tests.rs` | 3 | `default_socket_path` content, `discover_nestgate_socket` absent, env override |

The nest test suite was also cleaned: an `unsafe` env var manipulation test was replaced
with a safe capability-agnostic alternative.

### Idiomatic Rust Fixes

| Before | After | File |
|--------|-------|------|
| `.map(\|m\| m.len())` | `.map(Vec::len)` | `ipc/dispatch.rs` |
| `.ends_with(".toml")` | `Path::extension() == Some("toml")` | `niche.rs` |
| field reassign after default | `..Default::default()` | `inventory/output.rs` |
| unused struct fields | prefixed with `_` | 8 validation binaries |
| `unsafe { set_var(...) }` | removed | `nest/tests.rs` |
| `#![allow(cast_*)]` in npu.rs | 3 detailed `#![expect(reason)]` | `npu.rs` |

### Test Module Lint Evolution

6 `#[cfg(test)]` modules received explicit `#[expect()]` for test-appropriate patterns:

| Module | Lint | Reason |
|--------|------|--------|
| `bio/bootstrap.rs` | `expect_used` | "test module: assertions use expect for clarity" |
| `bio/diversity.rs` | `expect_used` | "test module: assertions use expect for clarity" |
| `ipc/handlers/ai.rs` | `expect_used` | "test module: assertions use expect for clarity" |
| `ipc/transport.rs` | `expect_used`, `unwrap_used` | "test module: assertions use unwrap/expect for clarity" |
| `visualization/ipc_push.rs` | `expect_used` | "test module: assertions use expect for clarity" |
| `ipc/timeseries.rs` | `approx_constant` | "3.14 is a Shannon diversity value, not PI" |

---

## Part 2: barraCuda Primitive Consumption (unchanged from V121)

150+ GPU primitives consumed. All lean. Zero local WGSL. See `ABSORPTION_MANIFEST.md`
and `EVOLUTION_READINESS.md` for the full ledger.

### Unwired Primitives (candidates for future wiring)

- `SparseGemmF64` — Track 3 drug repurposing sparse matrices
- `TranseScoreF64` — Knowledge graph embedding GPU acceleration
- `TopK` — Ranking acceleration for NMF/KG results

---

## Part 3: Patterns Worth Absorbing

### For barraCuda

1. **`#[expect(reason)]` as ecosystem standard**: wetSpring now has **zero** `#[allow()]`
   anywhere — production, validation, and test code. Every suppression carries a mandatory
   reason string. This makes stale suppressions detectable at compile time (Rust warns on
   unfulfilled expectations). Pattern: `#![expect(clippy::lint, reason = "justification")]`.

2. **Curated reason dictionary**: Validation binaries share a consistent vocabulary for
   lint reasons. This could become a barraCuda/ecosystem standard for all springs.

3. **Automated cleanup tooling**: Parsing `cargo clippy --message-format=json` for
   `unfulfilled_lint_expectations` enables bulk cleanup after `#[allow()]` → `#[expect()]`
   migration. Useful for any spring undertaking this evolution.

### For toadStool

1. **`#[expect(reason)]` migration guide**: wetSpring's 298-file migration serves as
   a template. Key learnings:
   - Convert `#[allow()]` → `#[expect()]` first (bulk script)
   - Run clippy to find unfulfilled expectations
   - Remove unfulfilled ones (they were never needed)
   - Fix any actual lint violations that were previously hidden
   - Add `#[expect()]` to test modules for test-appropriate patterns

2. **Test coverage patterns**: Error enum Display/Error tests, bridge edge cases, and
   discovery tests are high-value, low-effort coverage targets.

---

## Part 4: Evolution Requests for Upstream

### barraCuda (P1 — carried from V121)

- **`u64` percentile API**: Local `fn percentile(data: &[u64], pct: usize) -> u64`
  for NPU timing diagnostics. If barraCuda adds generic integer percentile, we delegate.
- **Tolerance module documentation**: Adopt wetSpring's 214 named constant pattern
  as cross-spring standard.

### barraCuda (P2 — carried from V121)

- **SparseGemmF64**: Track 3 drug repurposing sparse matrices
- **TranseScoreF64**: GPU knowledge graph scoring
- **TopK kernel**: Ranking acceleration for NMF reconstruction

### toadStool (P1 — carried from V121, updated)

- **`#[expect(reason)]` migration**: Now proven at scale (298 files, 276+ binaries).
  Recommend adopting across toadStool codebase.
- **`tempfile` for test paths**: Evolve `/tmp/` hardcoded test paths.

### All Springs (P1 — new)

- **`#[expect(reason)]` ecosystem convention**: wetSpring V122 demonstrates that
  zero `#[allow()]` is achievable across an entire codebase (production + validation +
  test). Recommend as ecosystem standard in `ECOBIN_ARCHITECTURE_STANDARD.md`.

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --all-features` | **Zero warnings** |
| `cargo test --workspace --lib` | **1,605 passed** (1,353 barracuda + 252 forge) |
| Total tests (with integration) | **1,703** |
| Validation checks | 5,707+ |
| Validation binaries | 354 (332 barracuda + 22 forge) |
| Experiments | 376 |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]`) |
| `#[allow()]` in codebase | **0** (production + validation + test) |
| `#[expect(reason)]` coverage | **100%** of lint suppressions |
| TODO/FIXME markers | 0 |
| Inline tolerance literals | 0 |
| Hardcoded primal names | 0 |
| Local WGSL shaders | 0 |
| Named tolerance constants | 214 |
| Python baselines | 58 (SHA-256 verified) |
| barraCuda version | v0.3.5 (HEAD 03986ce) |
| toadStool | S155 |
| Rust edition | 2024 |
| MSRV | 1.87 |
| Files changed (V121→V122) | 298 (+3,593 −1,628) |
