# Changelog

All notable changes to wetSpring are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed (V38 — Deep Debt Resolution)
- **tolerances.rs**: 3 new named constants (`GPU_LOG_POLYFILL`, `ODE_NEAR_ZERO_RELATIVE`,
  and strengthened `EXACT` usage). Total: 62 named tolerance constants.
- **Tolerance centralization**: ~35 validation binaries migrated from ad-hoc numeric
  literals (`0.0`, `1e-10`, `0.001`) to `tolerances::` module constants. 200+
  individual replacements.
- **MS2 parser**: `Ms2Iter` and `stats_from_file` evolved from `reader.lines()`
  (per-line `String` allocation) to `read_line()` with reusable buffer.
- **Streaming I/O**: `stream_taxonomy_tsv`, `stream_fasta_subsampled`
  (validation_helpers.rs), `spawn_nvidia_smi_poller` (bench/power.rs) all
  migrated from `reader.lines()` to reusable buffer pattern.
- **Provenance**: 25 binaries updated from placeholder commits to `1f9f80e`.
  13 binaries had `| Command |` rows added. 2 binaries reformatted to standard
  `//! # Provenance` table format. All 157 binaries now carry complete provenance.
- **`#[must_use]`**: Added to 7 public API functions (`parse_ms2`, `stats_from_file`
  ×3, `parse_mzml`, `parse_fastq`, `http_get`, `esearch_count`).
- **`Vec::with_capacity`**: Pre-allocation added in 5 library files (xml attributes,
  eic bins, spectral_match indices, mzml spectrum arrays, FASTA refs).
- **Clippy**: `too_many_lines` suppressed on 2 long validation functions. Zero
  warnings with `-D warnings -W pedantic -W nursery`.
- **EVOLUTION_READINESS.md**: Added upstream request #4 for `barracuda::math::{dot, l2_norm}`.

### Changed (V37 — Sovereignty + Safety)
- **ncbi.rs**: Evolved from hardcoded `curl` shell-out to capability-based
  HTTP transport discovery (`WETSPRING_HTTP_CMD` > `curl` > `wget`). Removed
  legacy relative dev paths. `cache_file()` now uses `WETSPRING_DATA_ROOT`
  cascade instead of hardcoded `CARGO_MANIFEST_DIR` paths.
- **I/O parsers**: Deprecated buffering functions (`parse_fastq`, `parse_mzml`,
  `parse_ms2`) evolved from duplicate buffering implementations to thin
  wrappers over streaming iterators (`FastqIter`, `MzmlIter`, `Ms2Iter`).
- **Validation binaries**: All 56 binary files modernized — `partial_cmp().unwrap()`
  replaced with NaN-safe `.unwrap_or(Ordering::Equal)`, bare `.unwrap()`
  replaced with descriptive `.expect()` messages throughout.
- **tolerances.rs**: Added commit-hash provenance to `GC_CONTENT` (`504b0a8`),
  `MEAN_QUALITY` (`cf15167`), `GALAXY_SHANNON_RANGE` / `GALAXY_SIMPSON_RANGE` /
  `GALAXY_BRAY_CURTIS_RANGE` (`21d43a0`).
- **CI**: Added `cargo-llvm-cov` 90% library coverage gate, metalForge clippy
  (pedantic + nursery) and test jobs, `forge-clippy` and `forge-test` pipelines.
- **URL encoding**: `encode_entrez_term` now encodes `&` and `#` characters.

### Removed
- Legacy hardcoded relative dev paths in `ncbi.rs` (`../../../testing-secrets/`).
- Duplicate buffering code in I/O parsers (was behind `#[deprecated]`).

### Revalidated (V37 + V38)
- 759 lib tests pass against ToadStool `02207c4a` (S62+DF64).
- 95.75% library coverage, clippy clean (pedantic + nursery + `-D warnings`).
- cargo doc clean (88 files, 0 warnings).
- All 157 binaries carry complete provenance (commit, command, date).
- ABSORPTION_MANIFEST, EVOLUTION_READINESS, BARRACUDA_REQUIREMENTS all synced.

## [0.1.0] — 2026-02-25

### Added
- Initial release: 47 CPU modules, 42 GPU wrappers, 157 validation/benchmark
  binaries, 4 fuzz targets.
- Three-tier validation (Python baseline → Rust CPU → GPU acceleration).
- 43/43 papers reproduced across 4 tracks.
- 812+ tests, 95.57% library coverage, 3,279+ validation checks.
- AGPL-3.0-or-later license, 100% SPDX header coverage.
