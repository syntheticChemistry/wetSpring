# wetSpring V128 Handoff — Ecosystem Absorption & Evolution Patterns

**Date:** 2026-03-18
**From:** wetSpring V128
**To:** barraCuda team, toadStool team, ecosystem
**Supersedes:** WETSPRING_V127_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

wetSpring V128 completes a full ecosystem absorption sweep, adopting patterns from
airSpring (cast module, FMA), groundSpring (ecoBin deny), healthSpring (PRIMAL_DOMAIN),
and barraCuda (FAMILY_ID sockets). This handoff documents what was absorbed,
what remains for upstream evolution, and learnings relevant to the ecosystem.

**Key metrics:** 1,550+ tests, 7 proptest modules, zero clippy warnings, zero unsafe code,
zero mocks in production, zero TODO/FIXME, 14 C-dep crates banned.

---

## 1. What wetSpring V128 Absorbed

### From airSpring / groundSpring

| Pattern | Source | wetSpring Adoption |
|---------|--------|--------------------|
| `cast` module (9 safe numeric helpers) | airSpring v0.9.0 | `barracuda/src/cast.rs` — `usize_f64`, `f64_usize`, `usize_u32`, `u32_usize`, `i32_f64`, `u32_f64`, `f64_u32`, `u64_usize`, `u64_f64`, `usize_u64`. All `const fn` where possible. Adopted in `flat_tree.rs` (21 casts), `quality_gpu.rs` (10), `dada2_gpu.rs` (9). |
| `mul_add()` FMA | neuralSpring S165 | 10 accumulation loops evolved to `a.mul_add(b, sum)`: felsenstein (4), esn (4), gbm (2), unifrac (2), nanopore (1). |
| ecoBin C-dep ban | groundSpring V115 | 14 crates banned in `deny.toml`: `openssl-sys`, `libz-sys`, `bzip2-sys`, `lzma-sys`, `zstd-sys`, `curl-sys`, `libgit2-sys`, `libssh2-sys`, `cmake`, `cc` (except flate2), `libffi-sys`, `bindgen`, `pkg-config`. |

### From healthSpring

| Pattern | Source | wetSpring Adoption |
|---------|--------|--------------------|
| `PRIMAL_DOMAIN` | healthSpring V34 | `pub const PRIMAL_DOMAIN: &str = "science.ecology"` in `lib.rs`. For biomeOS Neural API registration and Songbird discovery. |

### From barraCuda

| Pattern | Source | wetSpring Adoption |
|---------|--------|--------------------|
| `FAMILY_ID` socket paths | barraCuda v0.3.5 | `discover.rs`: `family_id()` helper reads `FAMILY_ID` / `BIOMEOS_FAMILY_ID` env var (default: `"default"`). All socket names evolved from `{primal}-default.sock` to `{primal}-{family_id}.sock`. Also consolidated in `provenance.rs` (Neural API) and `nestgate/storage.rs` (NestGate RPC). |

---

## 2. What barraCuda Should Absorb from wetSpring

### 2a. `monod()` and `haldane()` (unchanged from V127)

12 validation binaries still define local `fn monod(s, mu_max, ks)`. Recommend
`barracuda::stats::monod` + `barracuda::stats::haldane`.

### 2b. Centralized Tolerance Registry Pattern

wetSpring maintains 214 named tolerance constants in `tolerances/` with zero inline
magic numbers. This pattern would benefit barraCuda's shader test suites.

### 2c. MCP Tool Definitions Pattern

wetSpring defines 8 typed MCP tools for Squirrel AI in `ipc/mcp.rs`. Each tool has
a name, description, and JSON Schema for input parameters. This pattern enables
any primal to be discoverable by AI agents. Recommend barraCuda expose math primitives
as MCP tools for Squirrel.

---

## 3. What toadStool Should Absorb

### 3a. Forge Module Seam (unchanged from V127)

`metalForge/forge/` (252 tests) contains dispatch, probe, inventory, streaming, and
bridge modules. The `bridge` module is the integration point for toadStool absorption.

### 3b. Session-Level Provenance Tracking

wetSpring's `provenance.rs` implements session-scoped DAGs (`provenance.begin` →
`provenance.record` → `provenance.complete`). This pattern enables reproducible
experiment tracking and should be adopted by toadStool for compute provenance.

---

## 4. Ecosystem-Wide Learnings

### 4a. `cast` Module as Ecosystem Standard

airSpring and wetSpring both maintain independent `cast` modules with identical APIs.
**Recommendation:** Promote to a shared crate or document as an ecosystem convention.

### 4b. IPC Property Testing Pattern

wetSpring V128 adds 7 proptest tests to the IPC layer (protocol parsing + dispatch).
Key insight: `parse_request_never_panics` and `dispatch_never_panics` tests with
arbitrary input provide strong guarantees against DoS via malformed JSON-RPC messages.
All springs with IPC should adopt this pattern.

### 4c. `FAMILY_ID` Consolidation

Three independent implementations of `FAMILY_ID` resolution exist:
- `ipc::discover::family_id()` (shared)
- `ipc::provenance::neural_api_socket()` (now uses shared)
- `ncbi::nestgate::storage::active_family_id()` (local copy — ipc feature gate)

The local copy in `nestgate/storage.rs` exists because it's not gated behind `feature = "ipc"`.
If barraCuda promotes `family_id()` to a shared utility crate, this duplication can be eliminated.

---

## 5. Dependency Audit Result

All wetSpring dependencies are pure Rust except `wgpu` (which links `renderdoc-sys`
for GPU debugging — unavoidable for WebGPU). The `deny.toml` C-dep ban list prevents
accidental introduction of C dependencies.

| Category | Count |
|----------|-------|
| Pure Rust deps | All except wgpu |
| Banned C-dep crates | 14 |
| `#![forbid(unsafe_code)]` | Enforced crate-wide |
| Production mocks | 0 |
| Production `assert!` | 0 (only `debug_assert_eq!` in 1 file) |
| TODO/FIXME in code | 0 |

---

## 6. barraCuda Consumption Summary (V128)

| Primitive | Usage |
|-----------|-------|
| `FusedMapReduceF64` | Shannon, Simpson, observed, evenness, Bray-Curtis GPU |
| `GemmF64` | Pairwise spectral cosine similarity GPU |
| `BatchedEighGpu` | PCoA eigendecomposition GPU |
| `KrigingF64` | Spatial interpolation |
| `VarianceF64`, `CorrelationF64`, `CovarianceF64`, `WeightedDotF64` | Statistics GPU |
| `BrayCurtisF64` | Beta diversity GPU |
| `QualityFilterGpu` | Quality filtering shader |
| `Dada2EStepGpu` | DADA2 denoising E-step GPU |
| `kahan_sum` | Compensated summation (delegated, zero local math) |
| `WgpuDevice` | GPU device abstraction |
| `PrecisionRoutingAdvice` | Precision tier routing |

**Unwired but available:** `SparseGemmF64`, `TranseScoreF64`, `TopK`, `LogsumexpWgsl`, `monod()`, `haldane()`.
