# wetSpring V112 → barraCuda / toadStool: Streaming-Only I/O + Capability Discovery

**Date:** 2026-03-14
**From:** wetSpring V112 (Streaming-Only I/O + Zero-Warning Pedantic + Capability Discovery)
**To:** barraCuda / toadStool team
**License:** AGPL-3.0-or-later
**Covers:** Deprecated parser removal, streaming-only I/O architecture, capability-based discovery, clippy pedantic zero-warning, absorption status

---

## Executive Summary

V112 removes all deprecated whole-file buffering parsers (`parse_fastq`,
`parse_mzml`, `parse_ms2`), making wetSpring's I/O surface streaming-only.
All 40 clippy pedantic+nursery warnings eliminated. Hardcoded primal paths
replaced with `$PATH` and `$XDG_RUNTIME_DIR` runtime discovery. All quality
gates pass with zero warnings. Build-breaking compilation errors in 2
validation binaries fixed. Inline tolerance replaced with named constant.

---

## Part 1: For barraCuda — I/O Architecture Evolution

### What Changed

wetSpring deleted the three deprecated buffering parsers that loaded entire
files into memory:

| Removed Function | Replaced By | Pattern |
|------------------|-------------|---------|
| `parse_fastq(path) -> Vec<FastqRecord>` | `FastqIter::open(path)` → streaming `Iterator<Item=Result<FastqRecord>>` | Zero-alloc per-record |
| `parse_mzml(path) -> Vec<MzmlSpectrum>` | `MzmlIter::open(path)` → streaming `Iterator<Item=Result<MzmlSpectrum>>` | Zero-alloc per-spectrum |
| `parse_ms2(path) -> Vec<Ms2Spectrum>` | `Ms2Iter::open(path)` → streaming `Iterator<Item=Result<Ms2Spectrum>>` | Zero-alloc per-spectrum |

### Why This Matters for barraCuda

If barraCuda adds I/O modules (e.g., `barracuda::io::biom` per P0.1 request),
they should follow the streaming iterator pattern from the start:

```rust
pub struct BiomIter<R: BufRead> { /* ... */ }
impl<R: BufRead> Iterator for BiomIter<R> {
    type Item = Result<BiomEntry>;
    fn next(&mut self) -> Option<Self::Item> { /* ... */ }
}
impl BiomIter<BufReader<File>> {
    pub fn open(path: &Path) -> Result<Self> { /* ... */ }
}
```

No `parse_biom(path) -> Vec<BiomEntry>` convenience function — the iterator IS
the API. Consumers who need a Vec can `.collect()`.

### Tolerance Constant Usage

Inline `1e-10` in `validate_sovereign_dispatch_v1.rs` replaced with
`tolerances::ANALYTICAL_LOOSE`. All 180 named tolerance constants are consumed;
zero inline tolerance literals remain.

---

## Part 2: For toadStool — Capability-Based Discovery

### What Changed

Two validation binaries had hardcoded primal paths:

| Binary | Before | After |
|--------|--------|-------|
| `validate_workload_routing_v1` | `"../../phase2/biomeOS/target/release/biomeos"` | `which_primal("biomeos")` searching `$PATH` |
| `validate_primal_pipeline_v1` | `"/run/user/1000/biomeos"` | `$XDG_RUNTIME_DIR/biomeos` |

### Why This Matters for toadStool

The sovereignty principle requires primals to discover each other at runtime,
not at compile time. toadStool's dispatch should:

1. Use `$PATH` for binary discovery (standard Unix mechanism)
2. Use `$XDG_RUNTIME_DIR` for socket/IPC discovery (freedesktop standard)
3. Never hardcode UID-specific paths (e.g., `/run/user/1000/`)
4. Never hardcode relative build paths (e.g., `../../phase2/`)

This aligns with the wateringHole principle: primals are sovereign — they only
know themselves and discover others through capability probing.

---

## Part 3: Absorption Status — Still Fully Lean

wetSpring remains **fully lean** on barraCuda v0.3.5 (`0649cd0`):

| Metric | V111 | V112 | Delta |
|--------|------|------|-------|
| barraCuda primitives consumed | 150+ | 150+ | — |
| Local WGSL shaders | 0 | 0 | — |
| Unsafe code | 0 | 0 | — |
| Deprecated buffering parsers | 3 (deprecated) | **0 (removed)** | -3 |
| Clippy warnings (pedantic+nursery) | 0 | 0 | — |
| Hardcoded primal paths | 2 | **0** | -2 |
| Inline tolerance literals | 1 | **0** | -1 |

---

## Part 4: Upstream Requests (Updated from V114)

### P0 — Immediate (Carried Forward)

| # | Target | Justification |
|---|--------|---------------|
| 1 | `barracuda::io::biom` | BIOM format OTU table parser — use streaming iterator pattern |
| 2 | `barracuda::ncbi::entrez::esearch_ids()` | Expose accession list for batch fetch workflows |
| 3 | `barracuda::bio::kinetics::gompertz_fit()` | Nonlinear fitting for Track 6 biogas pipeline |

### P1 — Near-Term (Carried Forward)

| # | Target | Justification |
|---|--------|---------------|
| 4 | `barracuda::bio::anderson::temporal_w()` | Dynamic W(t) model for KBS LTER 30-year data |
| 5 | `barracuda::bio::qs::regulon_map()` | FNR/ArcAB/Rex regulon cross-reference |

### P2 — Unwired Primitives (Carried Forward)

`SparseGemmF64`, `TranseScoreF64`, `TopK`, `ComputeDispatch`, `BandwidthTier`,
`LogsumexpWgsl` — available but not yet consumed by wetSpring.

---

## Part 5: Pre-Existing Test Failures (For Awareness)

4 test failures exist, none introduced by V112:

1. **`nautilus_bridge::json_roundtrip`** — `bingocube-nautilus` crate
   `from_json` does not restore observations count. Root cause: deserialization
   skips the internal observation counter. Fix is in `bingocube-nautilus`.

2. **GPU f32 parity (3 tests)** — `hamming_gpu`, `jaccard_gpu`,
   `spatial_payoff_gpu` fail due to f32 accumulation differences. These are
   documented in `ABSORPTION_MANIFEST.md`. Fix requires either f64 promotion
   or `FusedMapReduceF64` rewire (which would be an absorption evolution).

---

## Part 6: API Observations (New)

1. **Streaming pattern recommendation**: barraCuda's `io::` namespace should
   adopt the `TypeIter::open(path) -> Result<Self>` + `Iterator` pattern rather
   than `parse_type(path) -> Result<Vec<T>>`. wetSpring proves this scales to
   multi-GB files without memory pressure.

2. **`tolerances::` namespace completeness**: wetSpring's 180 named constants
   cover bio, GPU, spectral, instrument, and analytical domains. If barraCuda
   centralizes tolerances (currently spring-local), consider absorbing
   wetSpring's hierarchy as a starting point.

---

## Action Items

- [ ] **barraCuda**: New I/O modules should use streaming iterator pattern (no `parse_*` convenience)
- [ ] **barraCuda**: Consider absorbing wetSpring tolerance hierarchy into `barracuda::tolerances`
- [ ] **toadStool**: Verify all primal discovery uses `$PATH` / `$XDG_RUNTIME_DIR` (no hardcoded paths)
- [ ] **bingocube-nautilus**: Fix `from_json` to restore observation count
- [ ] **wetSpring**: Wire `ComputeDispatch` in GPU modules (next evolution pass)
