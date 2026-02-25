# Changelog

All notable changes to wetSpring are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## V41 — Deep Audit + Coverage + Idiomatic Evolution (2026-02-25)

### Quality Gates
- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets` — 0 warnings (pedantic + nursery)
- `cargo test` — 918 tests (871 barracuda + 47 forge), 0 failures
- `cargo doc --no-deps` — 0 warnings
- `cargo llvm-cov --lib` — 96.48% line coverage (up from 96.41%)

### Coverage Improvements
- bench/power.rs: 59.5% → 70.6% (6 new tests)
- bench/hardware.rs: 76.9% → 81.4% (6 new tests)
- ncbi.rs: 77.5% → 79.6% (5 new tests)
- io/fastq/mod.rs: 83.7% → 87.2% (7 new tests)
- ncbi_data.rs: 68% → 95% (12 new tests from V40)

### Code Quality
- `#![deny(missing_docs)]` escalated from `warn` — every public item documented
- 33 scattered `clippy::cast_precision_loss` annotations consolidated to function-level
- 9 missing tolerance constants added to `all_tolerances_are_non_negative` test
- Provenance headers added to 20 validation binaries
- Inline dot-product replaced with `special::dot`/`special::l2_norm` in 2 binaries
- SPDX headers verified on all Python scripts

### Dependency Analysis
- 3 direct deps (all pure Rust): barracuda (path), bytemuck, flate2
- 0 C dependencies, 0 HTTP/TLS crate deps
- flate2 uses rust_backend (miniz_oxide)

### Documentation
- specs/README.md: 871 tests, 96.48% coverage
- specs/BARRACUDA_REQUIREMENTS.md: evolution readiness table, blocking items
- Root README.md: Phase 46, updated counts and coverage
- wateringHole handoff V41: deep audit + ToadStool evolution handoff

### Changed (V40 — ToadStool S39-S62+DF64 Catch-Up + Rewiring)
- **ToadStool evolution review**: Reviewed 55+ ToadStool commits (S39-S62+DF64).
  7/9 P0-P3 evolution requests delivered by ToadStool: GemmF64 public API (P0-1),
  PeakDetectF64 f64 fix (P1-2), ComputeDispatch builder (P1-3), GPU dot/l2_norm
  (P1-4), ODE generate_shader (P2-6), TopK GPU (P3-7), quantize_affine_i8 (P3-8).
- **Primitive count**: 44 → 49 ToadStool primitives consumed. New: PeakDetectF64,
  ComputeDispatch, SparseGemmF64, TranseScoreF64, TopK.
- **Passthrough elimination**: 3 → 0. `signal_gpu` leaned on `PeakDetectF64` (S62).
- **Track 3 GPU unblocked**: NMF, SpMM, TransE, cosine, Top-K all upstream.
- **SparseGemmF64 wired**: `validate_gpu_drug_repurposing` now validates GPU sparse
  CSR × dense GEMM (100×80 @ 5% fill) against CPU reference.
- **NMF Top-K wired**: Drug candidate ranking via `barracuda::linalg::nmf::top_k_predictions`.
- **Exp168**: New `validate_cross_spring_s62` binary — comprehensive cross-spring
  evolution validation covering hotSpring precision → wetSpring bio → neuralSpring
  population genetics → Track 3 complete GPU path, with evolution timeline narrative.
- **Pre-existing clippy fix**: `diversity_fusion_gpu.rs` float_cmp → epsilon check.
- **Doc updates**: All handoffs, specs, root docs updated to V40 with 49 primitives,
  0 Passthrough, 7/9 P0-P3 delivered. 158 binaries (was 157).

### Changed (V39 — Comprehensive Audit + Tolerance Completion)
- **tolerances.rs**: 8 additional named constants for full coverage. Total: 70 named
  tolerance constants. New: `RAREFACTION_MONOTONIC`, `PCOA_EIGENVALUE_FLOOR`,
  `KMD_NON_HOMOLOGUE`, `HMM_FORWARD_PARITY`, `GILLESPIE_PYTHON_RANGE_REL`,
  `GILLESPIE_FANO_PHYSICAL`, `ASARI_CROSS_MATCH_PCT`, `ASARI_MZ_RANGE_PCT`.
- **5 validation binaries updated**: `validate_diversity`, `validate_pfas`,
  `validate_barracuda_cpu`, `validate_gillespie`, `validate_features` — all
  remaining ad-hoc tolerance literals replaced with named constants. Zero
  ad-hoc tolerances remain in any validation binary.
- **Comprehensive audit**: Zero TODOs/FIXMEs/mocks, zero unsafe code, zero
  production unwrap/expect, all 145 validation binaries follow hotSpring
  pattern with provenance, all I/O parsers stream (no full-file buffering),
  all external data from public repos with accession numbers.

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
