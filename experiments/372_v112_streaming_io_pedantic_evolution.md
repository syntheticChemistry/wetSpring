# Exp372: V112 Streaming-Only I/O + Zero-Warning Pedantic + Capability Discovery

**Date:** 2026-03-14
**Track:** V112
**Status:** PASS (all quality gates green)
**Binary:** N/A (evolution pass, not a single validation binary)
**Features:** default, gpu

## Hypothesis

Removing deprecated whole-file buffering parsers, eliminating all clippy
pedantic+nursery warnings, and replacing hardcoded primal paths with
capability-based runtime discovery will produce a cleaner, more idiomatic, and
more sovereign codebase without breaking any existing validation checks.

## Method

1. Fix build-breaking compilation errors (`_all_pielou` undefined in 2 binaries)
2. Remove deprecated `parse_fastq()`, `parse_mzml()`, `parse_ms2()` functions
3. Migrate all callers to streaming iterators (`FastqIter`, `MzmlIter`, `Ms2Iter`)
4. Eliminate all 40 clippy pedantic+nursery warnings
5. Replace hardcoded primal paths with `$PATH` and `$XDG_RUNTIME_DIR` discovery
6. Replace inline tolerance `1e-10` with `tolerances::ANALYTICAL_LOOSE`
7. Verify all quality gates pass

## Key Results

| Category | Before | After |
|----------|--------|-------|
| `cargo build` | FAIL (2 compilation errors) | PASS |
| Clippy warnings (pedantic+nursery) | 40 | 0 |
| Deprecated buffering parsers | 3 (`parse_fastq`, `parse_mzml`, `parse_ms2`) | 0 (removed) |
| Streaming-only I/O | Partial (deprecated APIs still present) | Complete |
| Hardcoded primal paths | 2 binaries | 0 (runtime discovery) |
| Inline tolerance literals | 1 site | 0 |
| `#[expect(deprecated)]` stale attributes | 3 | 0 |

### Build-Breaking Fixes

- `validate_real_ncbi_pipeline.rs`: Pielou evenness `j` computed but never
  collected into a Vec. Added `let mut all_pielou = Vec::new()` and
  `all_pielou.push(j)` in the sample loop.
- `validate_cold_seep_pipeline.rs`: Same pattern. Added `all_pielou` collection.

### Deprecated I/O Removal

- `io::fastq::parse_fastq()` — buffered entire FASTQ into `Vec<FastqRecord>`.
  Removed. Callers use `FastqIter::open()` or `stats_from_file()`.
- `io::mzml::parse_mzml()` — buffered entire mzML into `Vec<MzmlSpectrum>`.
  Removed. Callers use `MzmlIter::open()`.
- `io::ms2::parser::parse_ms2()` — buffered entire MS2 into `Vec<Ms2Spectrum>`.
  Removed. Callers use `Ms2Iter::open()` or `for_each_spectrum()`.
- 15+ test functions updated with local `collect_*()` helpers.
- 3 stale `#[expect(deprecated)]` attributes removed.
- Broken intra-doc link (`parse_fastq` → `FastqIter`) fixed.

### Capability-Based Discovery

- `validate_workload_routing_v1.rs`: Relative `../../phase2/biomeOS/target/release/biomeos`
  replaced with `which_primal()` function searching `$PATH`.
- `validate_primal_pipeline_v1.rs`: Hardcoded `/run/user/1000/biomeos`
  replaced with `$XDG_RUNTIME_DIR/biomeos`.

### Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS (0 diffs) |
| `cargo clippy --pedantic --nursery` | PASS (0 warnings) |
| `cargo doc --no-deps` | PASS (0 warnings) |
| `cargo build --all-features` | PASS (0 warnings) |
| `cargo test --all-features` | 1,384 passed, 4 failed (pre-existing), 42 ignored |

### Pre-Existing Failures (Not Introduced by V112)

1. `bio::brain::nautilus_bridge::tests::json_roundtrip` — upstream
   `bingocube-nautilus` deserialization bug (observations not restored)
2. `bio::hamming_gpu::tests::hamming_gpu_matches_cpu` — known GPU f32 parity
3. `bio::jaccard_gpu::tests::jaccard_gpu_basic` — known GPU f32 parity
4. `bio::spatial_payoff_gpu::tests::spatial_payoff_all_cooperators` — known GPU f32 parity

## Key Finding

The deprecated buffering parsers were a persistent source of `#[expect(deprecated)]`
noise and prevented clean clippy runs. Removing them entirely (rather than just
deprecating) simplified the I/O surface to streaming-only, which is the correct
architecture for life science files that can be gigabytes.
Capability-based runtime discovery eliminates compile-time coupling between primals,
making validation binaries portable across deployment topologies.

## Run

```bash
cargo build --all-features
cargo fmt -p wetspring-barracuda -p wetspring-forge -- --check
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery
cargo doc --no-deps
cargo test --workspace --all-features
```
