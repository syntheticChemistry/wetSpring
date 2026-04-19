# wetSpring V148 — primalSpring v0.9.16 Alignment Handoff

**Date:** April 20, 2026
**From:** wetSpring V148
**To:** primalSpring, barraCuda, sibling spring teams
**Reference:** primalSpring v0.9.16 (Phase 44) downstream evolution blurb

---

## What Changed

### 1. Manifest Alignment (15 validation_capabilities)

The `wetspring_guidestone` N1 layer now validates all 15 methods from the
v0.9.16 `downstream_manifest.toml` entry:

| Domain | Methods |
|--------|---------|
| tensor | `tensor.matmul`, `tensor.create` |
| stats | `stats.mean`, `stats.std_dev`, `stats.variance`, `stats.correlation` |
| linalg | `linalg.solve`, `linalg.eigenvalues` |
| spectral | `spectral.fft`, `spectral.power_spectrum` |
| compute | `compute.dispatch` |
| storage | `storage.store`, `storage.retrieve` |
| ai | `inference.complete` |
| security | `crypto.hash` |

Every check uses analytically derivable baselines (no file deps, no Python at
runtime). N1 expanded from 7 to 15 manifest parity checks.

### 2. BLAKE3 Checksum Verification (Property 3)

New B2 layer uses `primalspring::checksums::verify_manifest()` to verify 6
tamper-sensitive files at runtime:

- `barracuda/src/bin/wetspring_guidestone.rs`
- `barracuda/src/niche.rs`
- `barracuda/src/tolerances/mod.rs`
- `barracuda/src/lib.rs`
- `barracuda/Cargo.toml`
- `Cargo.toml`

Generated via `cargo run --features guidestone --example gen_checksums`.
Checksums stored in `validation/CHECKSUMS`. When the manifest is absent (e.g.
running from a different directory), B2 gracefully skips.

### 3. Family-Aware Discovery

The guideStone inherits v0.9.16 family-aware socket discovery via
`CompositionContext::from_live_discovery_with_fallback()`. When `FAMILY_ID` is
set, sockets resolve to `{capability}-{family}.sock` before falling back to
plain `{capability}.sock`. This enables multi-NUCLEUS coexistence on the same
machine.

### 4. Protocol Tolerance

Primals that respond with HTTP framing on UDS (Songbird, petalTongue) are now
classified as reachable-but-incompatible (SKIP, not FAIL) by
`validate_liveness`. Inherited from primalSpring v0.9.16 composition API.

---

## Bare Mode Results (V148)

```
B0 — Bare Science Baselines:   7/7 pass
B1 — Tolerance Provenance:     2/2 pass
B2 — Checksum Verification:    7/7 pass (or SKIP if no manifest)
────────────────────────────────────────
Total bare:                    16/16 pass, exit 2
```

## guideStone Readiness

| Field | Value |
|-------|-------|
| `niche::GUIDESTONE_READINESS` | 3 |
| Bare checks | 16/16 |
| NUCLEUS checks | Pending (Tier 2/3 requires live NUCLEUS) |
| Validation layers | B0, B1, B2 (bare) + N0, N1, N2, N3 (NUCLEUS) |
| Exit codes | 0 = certified, 1 = failed, 2 = bare-only |

## Gaps Update

| Gap | Status | Notes |
|-----|--------|-------|
| PG-10 | Open | spectral/linalg routing in `method_to_capability_domain` (primalSpring) |
| PG-11 | **Resolved V148** | Manifest reconciled: 15 capabilities, `guidestone_readiness = 3` |
| PG-12 | Open | Exp403 legacy surface (15 methods pending v0.9.16 migration) |

## Next Steps (wetSpring → Level 4)

1. Deploy live NUCLEUS from plasmidBin with `FAMILY_ID=wetspring-validation`
2. Run `wetspring_guidestone` against live NUCLEUS (N0–N3 activate)
3. Iterate on parity failures (barraCuda response format, tolerance alignment)
4. Promote `GUIDESTONE_READINESS` to 4 when N0–N3 pass
5. Target Level 5 when cross-substrate (Python/CPU/GPU) parity is certified

## Verification

```bash
cd wetSpring
cargo clippy --features guidestone -p wetspring-barracuda --bin wetspring_guidestone -- -D warnings
cargo test --features ipc -p wetspring-barracuda --lib
./target/release/wetspring_guidestone  # bare: 16/16 pass, exit 2
```

---

*Handoff follows naming convention: `WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`.*
*Filed to `wateringHole/handoffs/` per NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
