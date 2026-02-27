# wetSpring → ToadStool/BarraCuda V66 Handoff

**Date:** February 27, 2026
**From:** wetSpring V66 (Phase 66)
**To:** ToadStool/BarraCuda team
**Status:** 216 experiments, 5,251+ checks, ALL PASS

---

## What Changed in V66

### I/O Evolution (V66 Deep Audit)

1. **Byte-native FASTQ I/O**: `io::fastq` now operates on `&[u8]` throughout,
   eliminating UTF-8 assumptions. Records are `FastqRecord<'a>` with byte
   slices. All downstream consumers (diversity, quality, derep) evolved.

2. **Bytemuck nanopore bulk read**: `io::nanopore::NanoporeRead` uses
   `bytemuck::cast_slice` for zero-copy signal extraction. Bulk iteration
   via `for_each_read` eliminates per-sample I/O overhead.

3. **Streaming APIs**: `io::mzml::for_each_spectrum`, `io::ms2::for_each_record`,
   `io::fastq::for_each_record` — callback-based streaming that never loads
   full files into memory.

4. **Safe env handling**: All `std::env::set_var` replaced with `temp_env::with_var`
   in tests (unsafe in multi-threaded contexts since Rust 1.85).

5. **Tolerance centralization**: 92 named constants in `tolerances.rs` with
   provenance comments. Zero ad-hoc magic numbers in any validation binary.

6. **`partial_cmp` → `total_cmp`**: 10 lib sites migrated from
   `partial_cmp().unwrap()` to `f64::total_cmp()` (panic-free NaN handling).

### Dispatch Evolution (Exp213 — 49/49)

metalForge dispatch routing validated against V66-evolved workloads:
- 29 bio workloads route correctly (GPU preference for F64Compute)
- 0 local WGSL (full absorption — all workloads have `ShaderOrigin::Absorbed`)
- PCIe bypass topology validated (streaming sessions are fully streamable)
- NUCLEUS Tower/Node/Nest model validated at the dispatch level

### NUCLEUS V8 Mixed Hardware (Exp214 — 49/49)

IPC dispatch layer validated through the V66-evolved I/O stack:
- Tower capabilities confirmed (crypto, tls, http, discovery)
- Byte-native FASTQ → diversity → QS pipeline through IPC dispatch
- Bytemuck nanopore → signal stats through IPC dispatch
- MS2 streaming → spectral match through IPC dispatch
- Nest metrics (direct `Metrics` struct, not dispatchable)
- CPU fallback parity maintained
- Full pipeline: FASTQ → quality → diversity → QS ODE (4-stage chain)

### NUCLEUS Local Deployment (Validated on Eastgate)

First real multi-primal NUCLEUS deployment on Eastgate tower:
- BearDog 0.9.0 (crypto, 72 JSON-RPC methods)
- Songbird 0.2.1 (HTTP, discovery, 14 capabilities)
- ToadStool 0.1.0 (compute, JSON-RPC + tarpc dual socket)
- NestGate 0.1.0 (storage, BLAKE3, blob, model cache)
- biomeOS Neural API (121 capability translations, COORDINATED mode)
- End-to-end: Songbird HTTP → BearDog TLS → NCBI → NestGate storage
- Real NCBI data fetched (*Vibrio harveyi* 16S, PX756524.1, 1264bp)

---

## Absorption Candidates for ToadStool

### Priority 0 — Ready Now

| Item | Location | Why |
|------|----------|-----|
| **Byte-native FASTQ** | `io::fastq` | Eliminates UTF-8 assumptions in WGSL dispatch path |
| **Streaming callback APIs** | `io::mzml`, `io::ms2`, `io::fastq` | Pattern for GPU pipeline feeding |
| **`total_cmp` migration** | 10 sites in lib.rs/bio/* | Panic-free NaN handling for GPU dispatch |
| **Tolerance constants** | `tolerances.rs` (92 named) | Cross-spring shared tolerance pattern |

### Priority 1 — Needs Coordination

| Item | Location | Why |
|------|----------|-----|
| **IPC dispatch router** | `ipc::dispatch` | Science method dispatch with GPU-aware threshold |
| **NUCLEUS metrics** | `ipc::metrics` | Neural API performance tracking pattern |
| **Songbird registration** | `ipc::songbird` | Science capability announcement |
| **Validator harness** | `validation.rs` | Structured pass/fail with section headers |

### Priority 2 — Architecture Reference

| Item | Location | Why |
|------|----------|-----|
| **metalForge workload catalog** | `forge::workloads` | 29 bio workloads with absorption status |
| **metalForge streaming** | `forge::streaming` | Pipeline topology for PCIe bypass planning |
| **NUCLEUS deployment** | `baseCamp/NUCLEUS_LOCAL_DEPLOYMENT.md` | Real deployment issues + workarounds |

---

## Deployment Feedback for ToadStool Team

Six issues found during real NUCLEUS deployment (see `NUCLEUS_LOCAL_DEPLOYMENT.md`):

1. **F1**: BearDog `--family-id` CLI doesn't set `FAMILY_ID` env internally
2. **F2**: Songbird starts without IPC unless `--socket` is passed explicitly
3. **F3**: Neural API socket naming mismatch (`biomeos-neural-api-*` vs `neural-api-*`)
4. **F4**: ToadStool → Songbird registration fails (missing `primal_id` in payload)
5. **F5**: NestGate requires `NESTGATE_JWT_SECRET` even for local socket-only mode
6. **F6**: NestGate `storage.list` doesn't find keys stored with `storage.store`

**ToadStool-specific (F4)**: The registration call to Songbird doesn't include
`primal_id` in the JSON payload, causing `-32603 Invalid params`. ToadStool
falls back to standalone mode gracefully, but discovery doesn't work.

---

## Codebase State

| Metric | Value |
|--------|-------|
| Tests | 1,073+ (946 lib + 80 integration/doc + 47 forge) |
| Coverage | 95.77% line / 93.86% fn / 95.33% branch |
| Experiments | 216 (5,251+ checks) |
| Binaries | 200 validation + benchmark |
| `.rs` files | 344 |
| TODO/FIXME/HACK | 0 |
| unsafe code | 0 |
| Local WGSL | 0 |
| Named tolerances | 92 |
| Primitives consumed | 79 (S68+) |
| Papers reproduced | 52/52 |
| Three-tier coverage | 39/39 |

---

## Evolution Pipeline Proven

```
Python/Galaxy baseline (44 scripts)
  → Rust CPU (33.4× faster, V1-V12 math fidelity checks)
    → GPU acceleration (ToadStool/BarraCuda WGSL, 926× peak)
      → Pure GPU streaming (zero CPU round-trips)
        → metalForge cross-substrate (CPU = GPU = NPU output)
          → NUCLEUS deployment (real multi-primal IPC on Eastgate)
            → NCBI data flowing through primal stack
```

## Next Steps for Absorption

1. Absorb byte-native FASTQ pattern into barracuda `io` module
2. Absorb streaming callback pattern (`for_each_*`) for pipeline feeding
3. Consider `total_cmp` migration in existing barracuda comparison code
4. Evaluate tolerance centralization pattern for cross-spring adoption
5. Fix ToadStool Songbird registration (F4) for NUCLEUS discovery
6. Wire NestGate `NCBILiveProvider` through Songbird HTTP (HTTP stub replacement)
