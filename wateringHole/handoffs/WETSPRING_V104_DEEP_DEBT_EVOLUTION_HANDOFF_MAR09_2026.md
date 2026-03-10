# wetSpring V104 — Deep Debt Evolution & Gap Closure

**Date:** March 9, 2026
**From:** wetSpring V104
**To:** barraCuda team, toadStool team, coralReef team
**License:** AGPL-3.0-or-later
**Builds on:** V103 upstream rewire (`WETSPRING_V103_UPSTREAM_REWIRE_HANDOFF_MAR10_2026`)

---

## Executive Summary

wetSpring V104 is a **deep debt evolution** that closes remaining implementation
gaps, completes the `#[allow] → #[expect]` migration, adds three new sovereign
modules (JCAMP-DX parser, Dorado basecaller delegation, GPU peak integration),
and eliminates all remaining clippy warnings.

**Key numbers:**

| Metric                | V103    | V104    | Delta |
| --------------------- | ------- | ------- | ----- |
| Tests passing         | 1,223   | 1,260   | +37   |
| Clippy warnings       | ~8      | 0       | -8    |
| `#[allow]` attributes | 74      | 0       | -74   |
| I/O parser formats    | 7       | 8       | +1    |
| Industry tool parity  | 18      | 19      | +1    |
| Sovereign modules     | ~80     | ~83     | +3    |

---

## Changes Applied

### 1. Complete `#[allow] → #[expect]` Migration (74 → 0)

All remaining `#[allow(clippy::...)]` attributes across both `wetspring-barracuda`
and `wetspring-forge` were migrated to `#[expect(clippy::...)]`. This immediately
exposed 56 stale suppressions that were removed:

- `clippy::unwrap_used` / `clippy::expect_used` on test modules that no longer use
  `unwrap()` or `expect()`
- `clippy::cast_precision_loss` on functions where casts were already removed
- `clippy::missing_panics_doc` on functions with documented panics
- `clippy::cast_possible_truncation` where truncation was resolved
- `clippy::naive_bytecount` (added proper suppression in test that legitimately counts)

**Zero `#[allow]` attributes remain.** All suppressions now use `#[expect]` with
the guarantee that stale ones produce compile warnings.

### 2. Remaining Clippy Warnings Resolved (8 → 0)

- `barracuda/src/bench/report.rs`: Added `#[expect(clippy::cast_precision_loss)]`
  on justified `usize as f64` cast (count → float for division)
- `barracuda/src/visualization/stream.rs`: Removed unused `std::path::PathBuf` import
  and stale `#[expect(clippy::unwrap_used)]`
- `barracuda/src/io/nanopore/mod.rs`: Replaced stale `clippy::expect_used` with
  `clippy::cast_precision_loss` (actual active lint)
- `barracuda/src/bio/quality/quality_tests.rs`: Added `clippy::naive_bytecount` to
  test module suppression
- `barracuda/tests/io_roundtrip.rs`: Removed unused `std::path::Path` import
- `metalForge/forge/src/nest/tests.rs`: Removed unused `std::path::PathBuf` import

### 3. New Module: JCAMP-DX Streaming Parser (`io::jcamp`)

- **What:** IUPAC JCAMP-DX format parser for spectroscopy data (IR, UV-Vis,
  Raman, NMR, MS)
- **Streaming:** BufReader-based, never loads full file
- **Features:** Compound file support (multiple blocks), `XYDATA` and
  `PEAK TABLE` formats, SQZ (squeezed) digit encoding, metadata preservation
- **API:** `JcampIter::open()`, `parse_jcamp()`, `for_each_block()`
- **Tests:** 10 tests covering IR, peak table, compound files, SQZ encoding,
  empty/missing files, metadata round-trip
- **Error variant:** `Error::Jcamp` added to crate error enum
- **Coverage:** Closes the JCAMP-DX gap identified in `INDUSTRY_TOOL_COVERAGE.md`

### 4. New Module: Dorado Basecaller Delegation (`bio::dorado`)

- **What:** Subprocess delegation for Oxford Nanopore's Dorado neural basecaller
- **Discovery:** Capability-based — `$WETSPRING_DORADO_BIN` → `$PATH` →
  standard install paths (`/opt/ont/dorado/bin/dorado`, `~/.local/bin/dorado`)
- **API:** `discover_dorado()`, `is_dorado_available()`, `dorado_version()`,
  `basecall()`, `parse_basecalled_reads()`
- **Config:** `DoradoConfig` with model selection (Fast/Hac/Sup), device targeting,
  quality filtering, emit-moves
- **Tests:** 9 tests covering discovery, config defaults, FASTQ parsing, error
  cases for missing input/binary
- **Pattern:** Graceful degradation — falls back to built-in `simple_basecall()`
  when Dorado is unavailable

### 5. GPU Peak Integration (`signal_gpu`)

- **New:** `find_peaks_with_area_gpu()` — GPU peak detection + CPU trapezoidal
  integration per peak
- **New:** `find_peaks_with_area_batch_gpu()` — Batch version for multiple
  signals in a pipeline
- **Architecture:** GPU handles the embarrassingly-parallel peak detection
  (N-element local maxima + prominence), CPU handles the sequential per-peak
  integration (dominated by detection cost)

---

## Debt Status After V104

| Category              | Count | Status |
| --------------------- | ----- | ------ |
| Files over 1000 lines | 0     | Clean  |
| `#[allow]` attributes | 0     | Clean  |
| `unsafe` code         | 0     | Clean  |
| Production mocks      | 0     | Clean  |
| Hardcoded paths       | 0     | Clean  |
| TODO/FIXME/HACK       | 0     | Clean  |
| `unwrap()`/`expect()` in lib | 0 | Clean |
| Clippy warnings       | 0     | Clean  |
| External deps needing evolution | 0 | Clean |

---

## I/O Parser Coverage (8 formats)

| Format     | Module          | Streaming | Status |
| ---------- | --------------- | --------- | ------ |
| FASTQ      | `io::fastq`     | Yes       | Stable |
| mzML       | `io::mzml`      | Yes       | Stable |
| mzXML      | `io::mzxml`     | Yes       | V103   |
| MS2        | `io::ms2`       | Yes       | Stable |
| POD5/FAST5 | `io::nanopore`  | Yes       | Stable |
| XML        | `io::xml`       | Yes       | Stable |
| BIOM 1.0   | `io::biom`      | JSON      | V103   |
| JCAMP-DX   | `io::jcamp`     | Yes       | **V104** |

---

## Upstream Requests

### barraCuda

1. **`ops::bio::seed_extend`** — BLAST-like seed-and-extend alignment primitive
   for local homology search. wetSpring needs this for NCBI BLAST parity.
   WGSL shader: parallel seed matching + diagonal extension.

2. **`ops::bio::profile_alignment`** — Position-weight matrix alignment for
   iterative MSA refinement (MAFFT L-INS-i equivalent). Current `bio::msa`
   uses progressive NJ-guided alignment; iterative refinement needs this
   upstream primitive.

3. **`ops::bio::peak_integrate_batch`** — Fused GPU kernel for peak detection +
   trapezoidal integration in a single dispatch. Current `signal_gpu` does
   detection on GPU then integration on CPU; a fused kernel eliminates the
   round-trip.

4. **`GpuView<T>` persistent buffers** — Persistent GPU-resident buffers to
   avoid per-dispatch upload overhead. Critical for closing the 3.7× Kokkos gap.

### toadStool

5. **`PrecisionRoutingAdvice` for bio primitives** — Route `f64` bio workloads
   to native GPU when hardware supports it, emulation when not, CPU fallback
   when emulation is slower than CPU. wetSpring's GPU paths currently check
   `has_f64` manually; ToadStool should handle this routing.

6. **CoralReef routing preference** — When CoralReef can produce native GPU
   binaries, prefer `ComputeDispatch::CoralReef` over `wgpu` generic dispatch.

### coralReef

7. **NVIDIA hardware validation** — CoralReef NVIDIA SASS generation needs
   validation on Titan V and RTX 3090 to establish Kokkos parity baseline.
   Register allocator SSA tracking fix is the known blocker.

8. **Instruction scheduling optimization** — CoralReef codegen quality is the
   primary remaining gap vs PTXAS. Improving instruction scheduling directly
   translates to wetSpring GPU performance.

---

## Test Verification

```
cargo test: 1,260 passed, 0 failed, 1 ignored
cargo clippy --all-targets: 0 warnings, 0 errors
cargo fmt --check: clean
```

---

## Next Steps

1. **Close remaining functional gaps** (BLAST-like search, iterative MSA) pending
   upstream barraCuda primitives
2. **Performance benchmarking** — Re-run Kokkos parity suite after barraCuda
   dispatch optimization lands
3. **CoralReef integration** — Wire `ComputeDispatch::CoralReef` through wetSpring
   GPU paths once NVIDIA validation passes
4. **Cross-spring benchmark suite** — Coordinate with other Springs to establish
   shared GPU benchmark infrastructure
