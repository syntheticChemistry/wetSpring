# Exp311: V97d Deep Audit & Idiomatic Evolution

**Date:** 2026-03-07
**Track:** Infrastructure / Code Quality
**Status:** COMPLETE
**Binary:** N/A (crate-wide evolution, no single validation binary)
**Phase:** 97d

## Purpose

Comprehensive deep audit of the wetSpring codebase targeting documentation
accuracy, API hygiene, crash diagnostics, and modern idiomatic Rust patterns.
Follows wateringHole standards: zero TODO/FIXME, zero unsafe, zero mocks in
production, all files under 1000 LOC, AGPL-3.0 on every .rs file.

## Sections

| ID | Domain | Checks | Status |
|----|--------|:------:|:------:|
| S1 | Rustdoc bracket escaping | 3 | PASS |
| S2 | MSRV doc accuracy (1.85→1.87) | 1 | PASS |
| S3 | wgpu version doc accuracy (v22→v28) | 1 | PASS |
| S4 | I/O buffering API deprecation | 3 | PASS |
| S5 | `.unwrap()` → `.expect()` in 12 validation bins | 104 | PASS |
| S6 | Broken doc reference cleanup | 3 | PASS |
| S7 | Root README stat accuracy | 6 | PASS |
| S8 | Full regression (fmt + clippy + doc + test) | 4 | PASS |

**Total:** 125 evolution items, all PASS.

## S1: Rustdoc Bracket Escaping

`validate_barracuda_cpu_v23.rs:37` had `E[XY] - E[X]E[Y]` interpreted as
intra-doc links. Escaped to `E\[XY\] - E\[X\]E\[Y\]`. Eliminated 3 rustdoc
warnings; `cargo doc --workspace --no-deps` now zero warnings.

## S2–S3: Doc Accuracy

- `EVOLUTION_READINESS.md` MSRV updated from 1.85 to 1.87 (matches Cargo.toml).
- `lib.rs` doc comment updated from "wgpu v22" to "wgpu v28".

## S4: I/O Buffering API Deprecation

Added `#[deprecated]` to `parse_fastq`, `parse_mzml`, `parse_ms2` with
migration guidance pointing to streaming iterators (`FastqIter`, `MzmlIter`,
`Ms2Iter`). Added `#[allow(deprecated)]` to 6 consumer files (4 validation
binaries that intentionally compare batch vs streaming, 2 test modules).

## S5: Crash Diagnostics Evolution

Replaced 104 bare `.unwrap()` calls with contextual `.expect("message")` across
12 barracuda validation binaries and 5 forge validation binaries. Messages
describe the expected-to-succeed operation for CI crash diagnostics.

## S6: Broken Reference Cleanup

- `barracuda/README.md`: removed phantom `DEPRECATION_MIGRATION.md` reference
  (file never existed), replaced with `../CHANGELOG.md`.
- `barracuda/README.md`: fixed stale dependency path
  (`phase1/toadstool/crates/barracuda` → `barraCuda/crates/barracuda`).
- `EVOLUTION_READINESS.md`: removed reference to nonexistent
  `DEPRECATION_MIGRATION` in history table.

## S7: Root README Stat Accuracy

Updated test counts (1,047 lib + 200 forge + 72 integration + 27 doc),
experiment count (286), binary count (290), check count (8,400+), removed
phantom `shaders/` directory from tree diagram.

## S8: Full Regression

| Check | Result |
|-------|--------|
| `cargo fmt --all -- --check` | PASS (0 diffs) |
| `cargo clippy --workspace -- -D warnings -W clippy::pedantic` | PASS (0 warnings) |
| `cargo doc --workspace --no-deps` | PASS (0 warnings) |
| `cargo test --workspace` | PASS (1,347 pass, 0 fail, 1 ignored) |

## Audit Findings (informational — no code changes needed)

- **External deps**: All pure Rust except wgpu transitive `renderdoc-sys`
  (upstream wgpu issue, not locally fixable).
- **Unsafe code**: Zero (crate-level `#![deny(unsafe_code)]`).
- **Mocks**: Zero in production; HTTP mocks only in `#[cfg(test)]`.
- **Hardcoding**: All capability-based (`validation::data_dir()`, XDG cascade,
  Songbird runtime discovery). Only platform-specific paths (`/proc/*`) are
  correctly gated to Linux.
- **Python benchmarks**: `benchmark_python_vs_rust_v3.rs` (15 domains) and
  `benchmark_python_vs_rust_v4.rs` (8 fused domains) prove bit-identical parity
  with scipy/numpy. Python scripts are timing-only.
- **Kokkos benchmarks**: Not applicable to wetSpring (life science domain).
  Kokkos/LAMMPS validation lives in hotSpring (Yukawa MD, lattice QCD).

## Command

```bash
cargo fmt --all && cargo clippy --workspace -- -D warnings -W clippy::pedantic && cargo doc --workspace --no-deps && cargo test --workspace
```
