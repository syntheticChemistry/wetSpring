# wetSpring V151 — Ecosystem Evolution Handoff

**Date**: April 27, 2026
**From**: wetSpring V151 (Life Science & Analytical Chemistry Validation)
**For**: primalSpring team, primal teams (barraCuda, toadStool, petalTongue,
Songbird, NestGate, rhizoCrypt, loamSpine, sweetGrass, BearDog, biomeOS),
and all spring teams (hot, neural, air, ground, health, ludo)

---

## Executive Summary

wetSpring has reached **V151** — deep debt evolution with interstadial-compliant
Rust (zero `dyn`, zero `async-trait`, Edition 2024, `#[expect]` only,
`forbid(unsafe_code)`). The validation ladder is at **Level 4+**: guideStone
passes 38/38 live NUCLEUS checks with 4 graceful skips. 15 primal gaps are
documented. This handoff captures everything wetSpring has learned about
primal composition patterns, NUCLEUS deployment, and ecosystem gaps — for
the teams that build the primals and the teams that validate through them.

---

## Where wetSpring Stands on the Validation Ladder

```
Level 1: Python baseline          ✅  58 scripts, 71 with SHA-256 provenance
Level 2: Rust validation          ✅  1,594 lib + 18 IPC tests, 364 binaries
Level 3: barraCuda CPU/GPU        ✅  150+ primitives consumed, 44 GPU modules
Level 4: guideStone NUCLEUS       ✅  38/38 pass, 4 skip, exit 0 (v0.9.17)
Level 5: Primal proof (IPC-only)  🔜  PG-09 evaporation surface mapped
Level 6: Clean-machine plasmidBin 🔜  Blocked on Level 5 completion
```

**guideStone Properties Certified:**
1. Deterministic (fixed seeds, bit-reproducible)
2. Reference-traceable (Python provenance, named tolerances)
3. Self-verifying (BLAKE3 checksums, Property 3)
4. Environment-agnostic (env-var discovery, no hardcoded paths)
5. Tolerance-documented (242 named constants, zero inline literals)

---

## Primal Usage — What wetSpring Consumes

### barraCuda (primary dependency — 150+ primitives)

wetSpring is the largest consumer of barraCuda bio-domain primitives.
Library dependency for Level 2 Rust proof. IPC for Level 4 guideStone.

| Domain | Library calls | IPC methods (v0.9.17) | Status |
|--------|:------------:|:---------------------:|--------|
| Stats (mean, std_dev, variance, correlation, weighted_mean) | ✅ Heavy | ✅ 5 methods pass | Green |
| Tensor (matmul, create, add, scale, reduce) | ✅ | ✅ 5 methods pass | Green |
| Linalg (solve, eigenvalues, determinant) | ✅ | ✅ 3 methods pass | Green |
| Spectral (fft, power_spectrum) | ✅ | ✅ 2 methods pass | Green |
| Compute (dispatch) | ✅ | ✅ 1 method pass | Green |
| Special functions (erf, ln_gamma, norm_cdf) | ✅ Delegated | N/A (library-only) | Green |
| ODE bio (5 systems via cpu_derivative) | ✅ Delegated | N/A (library-only) | Green |
| GPU modules (44 via WGSL shaders) | ✅ All lean | N/A (device-local) | Green |

**For primal proof (Level 5):** The 33 v0.9.17 IPC methods are all
reachable and passing. The remaining gap is migrating the library-call
code paths in validation binaries to IPC — the methods already exist on
barraCuda's side. See PG-09 for the evaporation surface.

### BearDog (Tower — crypto)

- `crypto.hash` (BLAKE3): ✅ Passing in guideStone cross-atomic pipeline
- Connection reset without BTSP handshake: Expected (PG-14)
- wetSpring does NOT bundle ring/rustls/ed25519-dalek (Tower delegation)

### Songbird (Tower — discovery)

- Socket discovery: name-based (`discover_primal()`) not capability-based
- PG-03: True `capability.resolve` → socket path not yet available
- PG-22: Songbird timeout (30s default) is aggressive for cold starts

### NestGate (Nest — storage)

- `storage.store` / `storage.retrieve`: ✅ Passing in guideStone
- Cross-atomic pipeline (hash→store→retrieve→verify): ✅ Green
- Full data provenance integration: Not wired (PG-04)

### rhizoCrypt / loamSpine / sweetGrass (Nest — provenance trio)

- IPC wiring exists in `ipc/provenance.rs` and `ipc/sweetgrass.rs`
- All trio calls fall back to local session tracking when unavailable
- PG-02: Trio endpoints not yet live on IPC
- PG-18: Connection reset on UDS contact (composition testing)

### toadStool (Node — compute orchestration)

- Discovery helper exists; no active runtime dispatch calls
- PG-05: Expected — library dep is correct for validation springs
- Becomes relevant when compute routes through toadStool IPC

### petalTongue (Meta — visualization)

- 100-node scene graphs accepted (~41KB payload, <1ms)
- 9 DataChannel types, 33 scenario builders, StreamSession
- Scene format: `"Text":{"x","y","content"}` (PG-19: not yet documented upstream)
- PG-22: Songbird timeout blocks discovery in some environments

### Squirrel (Meta — AI inference)

- `inference.complete`: ✅ Passing in guideStone N0
- Discovery via `discover_squirrel()` — works when neuralSpring is running

### biomeOS (Meta — orchestration)

- Deploy graphs: 7 canonical `[[graph.nodes]]` schema
- Bonding + fragments metadata validated (Exp400 D07)
- 136/136 proto-nucleate alignment checks

---

## Composition Patterns Discovered — For Ecosystem Absorption

### Pattern 1: FastqReader Enum (zero-cost file-format branching)

Instead of `Box<dyn BufRead>`, branch at open time into a concrete enum:

```rust
pub(crate) enum FastqReader {
    Plain(BufReader<File>),
    Gz(BufReader<flate2::read::GzDecoder<File>>),
}
impl Read for FastqReader { /* delegate to variant */ }
impl BufRead for FastqReader { /* delegate to variant */ }
```

Zero allocation, zero vtable, zero overhead. Works for any "open one
of N formats" pattern (mzML/mzXML, compressed/plain, etc.).

**For barraCuda:** If any I/O code uses `Box<dyn Read>`, this pattern
eliminates it. **For all springs:** Replace trait objects in I/O paths.

### Pattern 2: Write-Based Validation Output

All library code uses `writeln!(stdout().lock(), ...)` instead of
`println!`. Benefits: testable (redirect to `Vec<u8>`), tracing-compatible,
no hidden locking overhead.

**For primalSpring composition API:** `validate_parity` and friends
already return results rather than printing — good pattern. Springs
should follow this for their own validation code.

### Pattern 3: gpu_or_skip_sync() — One-Line GPU Bootstrap

```rust
pub fn gpu_or_skip_sync() -> GpuF64 {
    let rt = tokio::runtime::Runtime::new().or_exit("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) { /* ... */ };
    if !gpu.has_f64 { exit_skipped("No SHADER_F64"); }
    gpu
}
```

Eliminates duplicated GPU bootstrap across benchmark binaries.

### Pattern 4: Concrete FacadeError + ExitCode

Binary `main` returns `ExitCode`. Internal `run()` returns
`Result<(), FacadeError>` with concrete enum variants. Eliminates
`Box<dyn Error>` from all binary entry points.

### Pattern 5: check_skip for Absent Primals

From primalSpring `is_skip_error`: when IPC fails because a primal
is absent, classify as SKIP (exit 2), not FAIL (exit 1). CI stays
green when primals are unavailable.

```rust
if primalspring::composition::is_skip_error(&err) {
    validator.skip(&format!("{}: {}", label, err));
} else {
    validator.fail(&format!("{}: {}", label, err));
}
```

### Pattern 6: Environment-Agnostic Socket Discovery

Discovery cascade: `$PRIMAL_SOCKET` env var → `$XDG_RUNTIME_DIR` →
`$BIOMEOS_SOCKET_DIR` → `/tmp/`. Never hardcode `/tmp/primal.sock`.

### Pattern 7: BLAKE3 Self-Verification (Property 3)

Store checksums in `validation/CHECKSUMS`. Binary reads its own source
files and verifies at startup. Drift detection before any science runs.

---

## Open Primal Gaps — Upstream Team Actions

### High Priority (blocks Level 5 primal proof)

| Gap | Owner | Action |
|-----|-------|--------|
| **PG-03**: Capability discovery is name-based | Songbird | Implement `capability.resolve` → socket path. Springs hardcode primal names; true capability routing decouples identity from location. |
| **PG-09**: Library→IPC evaporation surface | wetSpring | Wire remaining library calls through IPC. barraCuda already exposes all methods — gap is in wetSpring's wiring. |
| **PG-10**: `spectral`/`linalg` routing in composition API | primalSpring | Add `"spectral" \| "linalg"` to `method_to_capability_domain()` tensor match arm. Currently falls through to default. |

### Medium Priority (composition quality)

| Gap | Owner | Action |
|-----|-------|--------|
| **PG-02**: Provenance trio IPC not live | rhizoCrypt/loamSpine/sweetGrass | IPC wiring exists in wetSpring; trio endpoints reaching live status unblocks full provenance chain. |
| **PG-04**: NestGate storage not wired for cross-spring data | NestGate | Protocol for cross-spring provenance-wrapped data retrieval. |
| **PG-08**: Binary name inconsistency in manifests | primalSpring | `spring_validate_manifest.toml` says `wetspring`, others say `wetspring_primal`. Canonicalize. |
| **PG-12**: Exp403 legacy method surface | wetSpring | Migrate 15 pre-v0.9.17 method names to canonical surface. |
| **PG-14**: BearDog connection reset without BTSP | BearDog | Expected behavior — document as standard. BTSP handshake required for crypto calls. |
| **PG-18**: Provenance trio UDS connection reset | rhizoCrypt/loamSpine/sweetGrass | Connection accepted then immediately reset. Possibly expects BTSP or different framing. |
| **PG-19**: petalTongue scene format undocumented | petalTongue | `"Text":{"x","y","content"}` works but format is not in petalTongue docs. |

### Low Priority (future evolution)

| Gap | Owner | Action |
|-----|-------|--------|
| **PG-05**: toadStool compute dispatch not active | toadStool | Expected for validation springs. Relevant at NUCLEUS deployment. |
| **PG-06**: Ionic bond negotiation protocol | primalSpring | Metadata declared; protocol not implemented. |
| **PG-20/21**: socat dependency for composition testing | ecosystem | Python `uds_send.py` shim works but adds Python dep. Rust UDS client preferred. |
| **PG-22**: Songbird 30s timeout too aggressive | Songbird | Cold-start NUCLEUS may exceed timeout. Configurable timeout or retry needed. |

---

## For Spring Teams — What wetSpring Learned

### The Validation Ladder Is Real

Python → Rust parity (Level 2) is the foundation. It took wetSpring 58
Python scripts and 1,594 Rust tests to establish that baseline. Don't
skip it. The numbers matter because they are the ground truth for every
subsequent level.

### guideStone Level 4 Is Achievable

The path: bare properties first (deterministic, traceable, self-verifying),
then NUCLEUS liveness, then manifest capabilities, then domain science,
then cross-atomic pipeline. primalSpring's `composition` API handles the
IPC plumbing. Focus on your science, not the wire protocol.

### check_skip Is Essential

Primals will be absent in CI. The Songbird/petalTongue/BearDog HTTP-on-UDS
behavior means some primals are "reachable but incompatible." Use
`is_skip_error` to classify gracefully. Exit code 2 = skipped, not failed.

### NUCLEUS Deployment from plasmidBin Works

Static musl binaries, zero C deps, UDS sockets. The launcher scripts
(`nucleus_launcher.sh`, `composition_nucleus.sh`) handle the lifecycle.
The gap is in wiring your domain science through IPC, not in deployment.

### Data Visualization Through petalTongue Is Fast

100-node scene graphs in <1ms. 9 DataChannel types cover most science
visualization needs. The bottleneck is documentation (PG-19), not
performance. Build scenarios in JSON, push via IPC or export to file.

### Named Tolerances from Day One

wetSpring has 242 named tolerance constants. Every comparison is
`check_relative(&mut v, label, got, expected, tolerances::CONSTANT, source)`.
Zero inline literals. This makes tolerance audits trivial and drift detection
automatic.

---

## For primalSpring — Composition API Feedback

1. **`method_to_capability_domain`** needs `spectral`/`linalg` arms (PG-10).
2. **`is_skip_error`** is excellent — adopted in V149, used everywhere.
3. **`validate_parity`** and `validate_liveness` work well for guideStone.
4. **BLAKE3 checksum module** (`primalspring::checksums`) is clean and reusable.
5. **Family-aware discovery** (`{capability}-{family}.sock`) just works.
6. **Downstream manifest** should canonicalize binary names (PG-08).

---

## For biomeOS — Deployment Observations

1. Deploy graphs with `[[graph.nodes]]` canonical schema work correctly.
2. Bonding metadata and fragments metadata are validated by Exp400 D07.
3. Custom profiles with `resolve = true` fragment inheritance work.
4. UDS is the correct transport — TCP port conflicts are common.
5. Socket discovery cascade (env → XDG → temp) handles all environments.

---

## For neuralSpring — Squirrel Integration

`inference.complete` passes in guideStone N0. Squirrel discovery via
`discover_squirrel()` works when neuralSpring is running. No spring code
changes needed — Squirrel discovers neuralSpring as a provider. If any
spring wants AI capabilities, add Squirrel to composition deps.

---

## Code Quality Snapshot (V151)

| Metric | Value |
|--------|-------|
| Rust edition | 2024 |
| MSRV | 1.87 |
| `dyn` dispatch | 0 (1 justified, documented) |
| `async-trait` crate | 0 |
| `unsafe` code | 0 (`forbid(unsafe_code)`) |
| `#[allow()]` in production | 0 (`#[expect(reason)]` only) |
| `println!` in library code | 0 (`writeln!` only) |
| Hardcoded paths | 0 (env-var discovery) |
| Inline tolerance literals | 0 (242 named constants) |
| Clippy warnings | 0 (pedantic + nursery) |
| C dependencies | 0 (ecoBin compliant) |
| Coverage | 91.20% line / 90.30% function |
| Lib tests | 1,209 pass, 0 fail |
| IPC tests | 18 roundtrip, 0 fail |
| Validation binaries | 364 (342 barracuda + 22 forge) |
| Validation checks | 5,900+ |
| Experiments | 383 (380 complete + 3 proposed) |

---

## Next Steps

1. **Level 5 primal proof**: Migrate PG-09 evaporation surface (library → IPC).
   barraCuda methods already exist — wire wetSpring's validation binaries.
2. **Level 6 clean-machine**: Deploy NUCLEUS from plasmidBin, run guideStone
   externally, validate peer-reviewed science end-to-end through sovereign
   compute stack.
3. **Upstream absorption**: primalSpring audits this handoff, resolves PG-10,
   PG-08. Primal teams resolve their respective gaps.
4. **Composition patterns**: FastqReader enum, Write-based output, gpu_or_skip,
   check_skip — available for ecosystem-wide adoption.

---

*This handoff flows to primalSpring for ecosystem audit. Gaps flow to
owning primal teams via `docs/PRIMAL_GAPS.md`. Patterns flow to
`wateringHole/` for cross-spring reference.*
