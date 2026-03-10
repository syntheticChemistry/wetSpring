# wetSpring V106 Deep Debt Cleanup — barraCuda/toadStool Evolution Handoff

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

**Date**: 2026-03-10  
**From**: wetSpring (Eastgate)  
**To**: barraCuda / toadStool team  
**License**: AGPL-3.0-or-later  
**Covers**: Deep debt cleanup, #[expect()] validation, zero-unsafe enforcement, BIOM streaming, upstream absorption fidelity

---

## Executive Summary

- **112+ stale #[expect()] annotations removed** — upstream barraCuda evolution means these lints no longer fire in wetSpring consumers. This validates that absorption worked: the code that triggered `cast_precision_loss`, `unwrap_used`, `manual_let_else`, etc. was successfully refactored during upstream absorption.
- **#![forbid(unsafe_code)] on all 320 crate roots** (2 lib + 318 bin) — wetSpring is now zero-unsafe across the entire workspace, not just library code.
- **BIOM parser evolved** from `read_to_string` to `serde_json::from_reader(BufReader)` — streaming I/O for QIIME2 interop.
- **All checks green**: fmt, clippy (pedantic+nursery), doc (0 warnings), test (1,605 pass, 0 fail).

---

## Part 1: What Changed (V106)

| Change | Impact |
|--------|--------|
| 112+ stale #[expect()] removed | Upstream barraCuda evolution; lints no longer fire. Validates absorption fidelity. |
| #![forbid(unsafe_code)] on 320 crate roots | Zero-unsafe across entire workspace (2 lib + 318 bin). |
| BIOM parser: read_to_string → serde_json::from_reader(BufReader) | Streaming I/O for QIIME2 interop. |
| NMF_CONVERGENCE tolerance constant centralized | Was inline 1e-4; now single source of truth. |
| 8 rustdoc broken links fixed | Doc builds clean. |
| All checks green | fmt, clippy (pedantic+nursery), doc (0 warnings), test (1,605 pass, 0 fail). |

---

## Part 2: Primitive Consumption Status

| Metric | Value |
|--------|-------|
| barraCuda primitives consumed | 150+ (unchanged from V105) |
| CPU modules | 47 |
| GPU modules | 47 |
| Local WGSL | 0 |
| Local derivative/regression math | 0 |
| Module style | All lean or compose |

**Key validation**: The stale #[expect()] cleanup **confirms absorption fidelity** — lints that were suppressed during the Write phase are no longer needed because upstream code is clean.

---

## Part 3: What This Means for barraCuda Team

1. **#[expect()] cleanup validates absorption** — Cast precision, unwrap patterns, and type complexity were properly addressed during absorption. The Write → #[expect(lint)] → Absorb → lint goes away → remove #[expect()] cycle is a healthy signal.

2. **Zero-unsafe enforcement (320 crate roots)** — Any future barraCuda API that requires `unsafe` will block wetSpring adoption. Design APIs safe-by-default.

3. **BIOM streaming pattern** — Good candidate for barraCuda I/O primitives if other springs need BIOM/JSON parsing. Consider `barracuda::io::biom` or `barracuda::io::json_streaming`.

---

## Part 4: Upstream Requests (Carried from V105)

| # | Request | Status |
|---|---------|--------|
| 1 | BipartitionEncodeGpu primitive for UniFrac | Pending |
| 2 | CPU Jacobi eigensolver for PCoA (currently local) | Pending |
| 3 | Merge pairs GPU kernel for 16S pipeline | Pending |
| 4 | StreamingSession absorption into barraCuda pipeline module | Pending |
| 5 | PipelineBuilder adoption (available but not yet used in metalForge) | Pending |

---

## Part 5: New Observations for Upstream

1. **clippy pedantic+nursery with -D warnings** — Correct bar. Unfulfilled #[expect()] annotations surface as errors, keeping the codebase honest about what lints actually fire.

2. **Write → #[expect(lint)] → Absorb → lint goes away → remove #[expect()]** — Healthy signal that absorption is working. Stale suppressions indicate either upstream fixed the issue or the consumer no longer hits that code path.

3. **BIOM / JSON streaming** — BIOM (JSON feature tables) is used across life science springs. Consider providing a `barracuda::io::biom` module or generic `barracuda::io::json_streaming` utility.

---

## Part 6: Test & Quality Status

| Metric | Value |
|--------|-------|
| Tests pass | 1,605 (1,288 lib + 218 forge + 72 integration + 27 doc) |
| Failures | 0 |
| Ignored | 2 (GPU hardware, NestGate socket) |
| Coverage (barracuda) | 94.01% |
| Coverage (forge) | 88.78% |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| unsafe code blocks | 0 |
| Named tolerance constants | 180 |
| Validation/benchmark binaries | 318 |
| Experiments | 334 |

---

## Part 7: Evolution Path

```
Python baseline → Rust CPU → GPU (barraCuda primitives) → metalForge cross-substrate → sovereign pipeline
```

All layers validated. wetSpring is fully lean on upstream barraCuda v0.3.3.
