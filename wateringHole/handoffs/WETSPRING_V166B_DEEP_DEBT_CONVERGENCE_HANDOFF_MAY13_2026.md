# wetSpring V166b — Deep Debt Resolution + Upstream Convergence Handoff

**From:** wetSpring
**To:** primalSpring (coordination)
**Date:** May 13, 2026
**Audit:** "Niche Atomic Convergence" (May 13, 2026)

---

## Summary

V166b executes the full deep debt directive and upstream convergence audit.
19 clippy pedantic+nursery warnings resolved to zero. 2 primal gaps closed
(PG-03, PG-05), 2 narrowed to deployment-only (PG-02, PG-04). Songbird
`capability.resolve` live RPC wired into discovery. Zero actionable internal
debt across all 7 audit categories.

---

## Deep Debt Audit Results

### 1. TODO/FIXME/HACK Markers

**Result: ZERO** across all `.rs` source files (library + binaries).

### 2. Modern Idiomatic Rust

| Metric | Status |
|--------|--------|
| Edition | **2024** |
| `rust-version` | **1.87** |
| Clippy pedantic+nursery | **0 warnings** (was 19 — all fixed V166b) |
| `#[allow()]` usage | **Zero** in entire codebase |
| `#[expect()]` usage | 3 functions (ESN ridge regression, feature-gated, with `reason`) |
| `#![forbid(unsafe_code)]` | lib.rs + all binaries |

**Warnings fixed (19):**
- 10 `doc_markdown` (backtick primal/method names in doc comments)
- 4 `collection_is_never_read` (ESN train functions — gated `expect` attributes)
- 3 `missing_errors_doc` (added `# Errors` sections to skunkBat + transport)
- 2 `unfulfilled_lint_expectations` (removed stale `unused_variables` expects)

### 3. External Dependencies

**Runtime deps (with `ipc` feature): 8** — all pure Rust.

| Crate | Purpose | C code? |
|-------|---------|---------|
| `blake3` | Content hashing | Build-time ASM only (cc allowed in deny.toml) |
| `bytemuck` | Zero-copy byte casting | No |
| `flate2` | Compression (rust_backend) | No |
| `serde` + `serde_json` | Serialization | No |
| `thiserror` | Error derive | No |
| `tracing` + `tracing-subscriber` | Logging | No |

**deny.toml bans 15 C-linked crates** (ring, openssl-sys, libz-sys, curl-sys, etc.).
Zero violations. ecoBin compliant.

### 4. Large Files (>800 LOC)

**Library code: max 617 LOC** (`facade/provenance.rs`). All under threshold.

**>800 LOC files (5):** All validation binaries (self-contained narrative
harnesses with provenance structure). Intentionally monolithic — splitting
would lose provenance narrative integrity.

### 5. Unsafe Code

`#![forbid(unsafe_code)]` at workspace lib.rs and all binary crate roots.
Zero `unsafe` blocks. Zero `#[allow(unsafe)]`. No transmute, raw pointers,
or FFI in application code.

### 6. Hardcoded Values

All URLs/ports are env-configurable defaults:
- NCBI Efetch: `EFETCH_BASE` env → fallback `eutils.ncbi.nlm.nih.gov`
- ChEMBL/PubChem: `CHEMBL_BASE`/`PUBCHEM_BASE` env → fallback to public APIs
- Facade bind: `FACADE_BIND` env → fallback `127.0.0.1:3100`
- CORS origin: `FACADE_CORS_ORIGIN` env → fallback `primals.eco`
- plasmidBin URL: `PLASMIDBIN_URL` env → fallback GitHub

Zero hardcoded secrets, credentials, or non-configurable paths.

### 7. Production Mocks

All mock functions (`mock_fasta_get`, `mock_genbank_get`, etc.) are in
`#[cfg(test)]` modules. Zero production mocks. Visualization module
explicitly declares "never mock data."

---

## Upstream Gap Convergence (PG-02 → PG-05)

| Gap | Was | Now | Code Change |
|-----|-----|-----|-------------|
| PG-02 | Partial — trio not live | **Narrowed** — deploy only | None needed — method names align with GAP-36 alias targets |
| PG-03 | Waiting for Songbird | **Resolved** | `resolve_via_songbird()` wired in `discover.rs` |
| PG-04 | NestGate not deployed | **Narrowed** — deploy only | None needed — NestGate S60 `content.*` 4-surface shipped |
| PG-05 | Sovereign dispatch | **Resolved** | Tier 2 wiring complete (V165b+V166) |

**Total: 2 gaps open (deployment-only), 20 resolved/closed.**

### Songbird `capability.resolve` Wire (PG-03)

`discover_by_capability(domain)` now:
1. Attempts Songbird `capability.resolve` RPC (Wave 199-201 contract)
2. Parses `{"result":{"socket":"/path/to/primal.sock"}}` response
3. Verifies socket path exists
4. Falls back to static `capability_to_primal` table on any failure

Zero-disruption migration: existing callers unchanged, standalone mode
preserved, timeout via `ipc::timeouts::DISCOVERY` (5s).

---

## Audit Question Answers

### barraCuda CPU Benchmarks (Python Baselines)

**Foundation repo:** 3 Python/NumPy/SciPy baselines:
- `stats_variance.py` → `VarianceF64` (ULP-exact for n<10^6)
- `md_velocity_verlet.py` → velocity Verlet (1e-12, Kokkos/LAMMPS-style)
- `spectral_eigenvalues.py` → Anderson eigensolver (1e-10 vs LAPACK)

**wetSpring:** 380/380 Rust CPU parity checks (Exp101+). 38/38 GPU diversity
domains. 1,783 GPU checks. `validate_barracuda_cpu_*` binaries cover full
surface.

**Operations lacking Python baselines:** Most of the 150+ barraCuda primitives
have Rust-to-Rust parity tests but not separate Python ground truth scripts.
The foundation 3-script set covers the critical numerical paths (variance,
ODE integration, eigenvalues). Expanding to cover all 150+ is a foundation
roadmap item, not a wetSpring blocker.

### barraCuda GPU Parity (Industry Benchmarks)

- **Kokkos:** velocity Verlet parity (1e-12) — foundation baseline
- **LAMMPS:** MD force/energy parity — covered via velocity Verlet tolerance
- **SciPy:** eigenvalue parity (1e-10) — foundation baseline
- **Galaxy/QIIME2:** metagenomic diversity — wetSpring Exp domains, not direct
  benchmark comparisons (Galaxy/QIIME2 are pipeline tools, not numeric libraries)

### Not Yet Implemented / Validated / Tested

**Experiments:**
- 3 PROPOSED: Exp377 (hormesis), Exp378 (trophic cascade), Exp379 (colonization)
- 6 PLANNED: Exp197–202 (NPU field pipelines)
- 1 PARTIAL: Exp196 (Nanopore full POD5 — 196a–c done)

**LTEE papers:** 9/10 queued (B1–B6, B8, E1, E5). B7 Tier 2 COMPLETE.

**GPU engineering:** `GillespieGpu` skip (NVVM driver), DF64 Phase 2 shaders,
`BatchReconcileGpu` promotion, `BandwidthTier` metalForge.

### Unreviewed Papers

- **Main queue:** 63/63 complete
- **LTEE queue:** 9 queued for reproduction
- **Watch:** Liu #19 (fungi-bacteria coevolution)
- **Reference-only:** Waters 2021, Track 5 references (by design)

### Pending Datasets

| Priority | Dataset | Experiment | Status |
|----------|---------|------------|--------|
| Medium | EPA UCMR 5 national PFAS | Exp041 | Download + parse |
| Medium | EPA PFOS surface water | Exp041 | Download + parse |
| Medium | NCBI PRJNA294072 (264 LTEE genomes) | Exp380 | Download for sovereign |
| Low | Jones Lab PFAS library (175 compounds) | Exp018 | Data access pending |
| Queued | 8 LTEE paper-specific datasets | B1–B6,B8,E1,E5 | Pending paper reproduction |
| Hardware | BrainChip AKD1000 NPU | Exp083,188 | Hardware procurement |
| Hardware | Titan V (GV100) | Exp215 | Hardware procurement |

---

## Build Verification

```
cargo build --features ipc --lib          → exit 0
cargo clippy --features ipc --lib         → 0 warnings (pedantic+nursery)
cargo build --release --target x86_64-unknown-linux-musl --bin wetspring_unibin --features guidestone → exit 0
```

---

## Metrics Snapshot (V166b)

| Metric | Value |
|--------|-------|
| Lib tests | 1,962 (0 failures) |
| Integration tests | 97 |
| IPC roundtrip tests | 18 |
| Experiments | 384 indexed (381 complete + 3 proposed) |
| Binaries | 367 (345 barracuda + 22 forge) |
| Coverage | 91.20% (gated at 90%) |
| Papers | 63/63 (+ 9 LTEE queued) |
| Primal gaps | 2 open (deployment-only), 20 resolved/closed |
| guideStone | Level 4 (38/38 pass, 4 skip) |
| Clippy warnings | 0 (pedantic+nursery) |
| `#[allow()]` | 0 |
| `unsafe` | 0 (`forbid` at workspace level) |
| C dependencies | 0 (deny.toml enforced) |

---

## Observation: Holding Pattern per Audit

Per "Niche Atomic Convergence" directive: wetSpring is a cross-atomic
validator. Holding on NUCLEUS composition expansion until Tower (ludoSpring),
Node (hotSpring), and Nest (healthSpring) confirm live atomic validation.

**Deepening niche:**
- PG-02 (trio) and PG-04 (NestGate): Await deployment stack for live roundtrip
- gS L5: 3 of 4 blockers have upstream implementations; gate is live verification
- LTEE: B7 Tier 2 COMPLETE; 9 papers queued for reproduction
- Foundation: 10/10 threads active

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
