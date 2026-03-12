# Exp371: V114 Deep Audit + Build Fix + Idiomatic Evolution

**Date:** 2026-03-12
**Track:** V114
**Status:** PASS (all fixes verified)
**Binary:** N/A (audit + fixes, not a single validation binary)
**Features:** default, gpu, json

## Hypothesis

A comprehensive deep audit of the V113 codebase will identify and resolve build blockers, clippy regressions, deprecated API usage, inline magic numbers, hardcoded values, and code duplication introduced during the paper extension roadmap (V110-V113).

## Method

1. Full codebase audit: build, clippy (pedantic+nursery), fmt, doc, test, coverage
2. Cross-reference wateringHole standards (1000 LOC limit, AGPL-3.0, zero unsafe, capability-based)
3. Fix all issues found, verify all checks pass

## Key Results

| Category | Before | After |
|----------|--------|-------|
| `cargo check` | FAIL (2+ errors) | PASS |
| Clippy warnings | 52 | 0 |
| Doc warnings | 2 | 0 |
| Deprecated batch parser usage | 4 binaries | 0 |
| Inline tolerance literals | 4 sites | 0 |
| Hardcoded VRAM | 1 site | capability-based |
| Duplicated `argmax_with_priors` | 2 copies | 1 (taxonomy module) |
| Missing `required-features` | 15 binaries | 0 |

### Fixes Applied

- 15 `[[bin]]` entries received `required-features` gates (10 `gpu`, 6 `json`, 2 `gpu+json`)
- 25 binary files: doc backticks, `mul_add`, `f64::from`, `.clamp()`, iterator patterns, dead fields, bare URLs, long literal separators
- 4 binaries migrated from deprecated `parse_fastq`/`parse_ms2` to `FastqIter`/`Ms2Iter`
- `validate_stable_specials_v1.rs`: inline `1e-6` → `tolerances::ERF_PARITY`, etc.
- `validate_hardware_learning_v1.rs`: `12_u64` → `DeviceCapabilities::max_buffer_size`
- `argmax_with_priors` extracted to `bio::taxonomy::mod.rs` with `#[cfg(feature = "gpu")]`

### Benchmark/Python Baseline Audit

- Python baselines: 28/47 domains covered (23 core + 5 Track 1c + 6 Track 6 - some overlap)
- Kokkos: Not applicable (life science domain; Kokkos lives in hotSpring)
- Galaxy vs GPU: No direct comparison exists; Exp015 is Galaxy vs Rust CPU only
- R industry parity: 53/53 PASS (V107)

## Key Finding

The V113 paper extension sprint introduced 15 binaries without proper feature gates, breaking the default build. The audit pattern — run full build/clippy/test/doc before handoff — must be enforced in CI for every version bump.

## Run

```bash
cargo check
cargo clippy -- -W clippy::pedantic -W clippy::nursery
cargo fmt -- --check
cargo doc --no-deps
cargo test
```
