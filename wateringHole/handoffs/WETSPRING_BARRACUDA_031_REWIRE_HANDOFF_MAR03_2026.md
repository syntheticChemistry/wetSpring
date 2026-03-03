# SPDX-License-Identifier: AGPL-3.0-or-later

# wetSpring → barraCuda v0.3.1 Rewire Handoff

**Date:** 2026-03-03
**From:** wetSpring team
**To:** barraCuda team, toadStool team, ecosystem
**barraCuda version:** v0.3.1 (standalone primal at `ecoPrimals/barraCuda/`)
**Previous dependency:** v0.2.0 (embedded at `ecoPrimals/phase1/toadstool/crates/barracuda/`)
**MSRV:** 1.85 → 1.87
**License:** AGPL-3.0-or-later

---

## Summary

wetSpring has been rewired from the deprecated ToadStool-embedded barraCuda (v0.2.0)
to the standalone barraCuda primal (v0.3.1). The rewire was a clean path swap with
**zero code changes required** — all 1,044 tests pass, zero clippy errors (pedantic),
zero fmt diffs, and clean doc build.

This mirrors hotSpring's experience (716/716 tests, zero code changes) and confirms
the architectural promise: **barraCuda = "WHAT to compute" (pure math), toadStool = "WHERE and HOW" (hardware dispatch).**

---

## Changes Made

### 1. Cargo.toml Dependency Path Swap

| File | Old Path | New Path |
|------|----------|----------|
| `barracuda/Cargo.toml` | `../../phase1/toadstool/crates/barracuda` | `../../barraCuda/crates/barracuda` |
| `metalForge/forge/Cargo.toml` | `../../../phase1/toadstool/crates/barracuda` | `../../../barraCuda/crates/barracuda` |

### 2. akida-driver (NPU) Dependency Strategy

The `akida-driver` crate remains at its toadStool path
(`../../phase1/toadstool/crates/neuromorphic/akida-driver`). This is correct because:

- akida-driver is independent of barraCuda (no cross-dependency)
- It's gated behind `npu = ["dep:akida-driver"]` feature flag (optional)
- Hardware-only, zero mocks — tests skip when no Akida device is present
- The neuromorphic crate is toadStool's domain (hardware dispatch), not barraCuda's

### 3. MSRV Bump

`rust-version` bumped from `"1.85"` to `"1.87"` in all three `Cargo.toml` files:
- Workspace root (`Cargo.toml`)
- `barracuda/Cargo.toml`
- `metalForge/forge/Cargo.toml`

This matches standalone barraCuda's MSRV requirement.

### 4. Documentation Updates

Updated references in architectural files from "ToadStool barracuda" to "barraCuda" (standalone primal):

- `barracuda/src/gpu.rs` — module-level docs, struct docs, method docs
- `barracuda/src/lib.rs` — crate-level title
- `barracuda/src/npu.rs` — module-level docs
- `barracuda/Cargo.toml` — package description, dependency comments, feature comments

Historical provenance references ("absorbed from ToadStool S54", "ToadStool S79 primitive")
are preserved as-is — they document the evolution lineage accurately.

---

## Verification Results

| Check | Result |
|-------|--------|
| `cargo check --workspace` | clean |
| `cargo test --workspace` | **1,044 passed**, 0 failed, 1 ignored |
| `cargo clippy --pedantic` | 0 errors (nursery suggestions only) |
| `cargo fmt --check` | 0 diffs |
| `cargo doc --no-deps` | clean (1 pre-existing `bench` ambiguity warning) |

---

## Breaking Changes Handled

From barraCuda `BREAKING_CHANGES.md` (v0.2.0 → v0.3.1):

| Change | Impact on wetSpring |
|--------|---------------------|
| `toadstool` feature flag removed | None — wetSpring never used it |
| `npu-akida` feature flag removed | None — akida-driver sourced independently |
| `read_f64_raw`/`read_i32_raw` take `&WgpuDevice` | Not used by wetSpring |
| `WgpuDevice::from_selection()` removed | wetSpring uses `from_existing()` |
| MSRV 1.85 → 1.87 | Bumped (Rust 1.93 installed) |
| Device creation serialized (`DEVICE_CREATION_LOCK`) | No action needed |

**Net result: zero API breakage for wetSpring.**

---

## Architecture After Rewire

```
wetSpring (validation Spring)
  ├── barracuda/     → barraCuda v0.3.1 (standalone math primal)
  │   ├── 144+ GPU primitives consumed
  │   ├── 767+ f64-canonical WGSL shaders
  │   └── Universal precision: f64/f32/f16/Df64 per hardware
  ├── metalForge/    → barraCuda v0.3.1 (substrate routing)
  └── akida-driver   → toadStool neuromorphic (independent, optional)
```

Dependency direction: `wetSpring → barraCuda` (direct). No reverse dependency.
toadStool is NOT a dependency of wetSpring — only barraCuda and akida-driver.

---

## Future Work (Not Blocking)

| Item | Priority | Effort |
|------|----------|--------|
| `ComputeDispatch` adoption (tarpc) | P3 | Medium |
| DF64 GEMM adoption | P3 | Low |
| `BandwidthTier` wiring in metalForge | P3 | Low |
| `domain-genomics` migration to wetSpring | P1 | High |

---

## Ecosystem Signal

wetSpring is the **second Spring** (after hotSpring) to successfully rewire to
standalone barraCuda. Both rewires were zero-code-change path swaps, validating
the clean extraction. Remaining Springs (airSpring, groundSpring) can follow
the same pattern.
