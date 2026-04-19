# wetSpring V148 guideStone Level 4 — Live NUCLEUS Validation

**Date:** April 19, 2026
**From:** wetSpring V148 (primalSpring v0.9.16)
**For:** primalSpring, barraCuda, ToadStool, Squirrel, BearDog, NestGate teams
**License:** AGPL-3.0-or-later

---

## Summary

wetSpring's `wetspring_guidestone` now passes **31/31 checks (11 skipped)**
against a live NUCLEUS deployed from source-built binaries. Exit code 0.
This promotes wetSpring from guideStone Level 3 (bare) to **Level 4**
(NUCLEUS validated).

## NUCLEUS Stack Deployed

Built and launched from source (no plasmidBin x86_64 binaries available):

| Primal | IPC Socket | Status |
|--------|-----------|--------|
| barraCuda | `math-wetspring-validation.sock` | ALIVE — GPU (RTX 3070) |
| BearDog | `beardog-wetspring-validation.sock` | ALIVE |
| NestGate | `nestgate-wetspring-validation.sock` | ALIVE |
| ToadStool | `toadstool-wetspring-validation.sock` | ALIVE (BTSP) |
| Squirrel | `squirrel-wetspring-validation.sock` | BTSP-only (PG-14) |

biomeOS not deployed (upstream compile error — `biomeos-types` missing
`secret` module). Not required for guideStone validation.

## Validation Results

### PASS (31)

**B0 — Bare Science (7):** Shannon, Hill, mean, std_dev, matmul,
weighted_mean, self-verify tamper detection.

**B1 — Tolerance Provenance (2):** ANALYTICAL_F64, IPC_ROUND_TRIP_TOL.

**B2 — BLAKE3 Checksums (7):** All 6 critical files verified + manifest.

**N0 — Liveness (4):** barraCuda, BearDog, NestGate, ToadStool alive.

**N1 — Manifest IPC (7):** tensor.matmul (handle-based), tensor.create,
stats.mean, stats.std_dev (sample), storage.store, storage.retrieve,
crypto.hash (BearDog BLAKE3).

**N2 — Domain Science (1):** stats.weighted_mean parity (diff=2.22e-16).

**N3 — Cross-Atomic (3):** BearDog hash → NestGate store → retrieve → verify.

### SKIP (11)

| Method | Reason | Gap |
|--------|--------|-----|
| squirrel.liveness | BTSP handshake required | PG-14 |
| stats.variance | Unknown method in ecobin | PG-13 |
| stats.correlation | Unknown method in ecobin | PG-13 |
| linalg.solve | Unknown method in ecobin | PG-13 |
| linalg.eigenvalues | Unknown method in ecobin | PG-13 |
| spectral.fft | Unknown method in ecobin | PG-13 |
| spectral.power_spectrum | Unknown method in ecobin | PG-13 |
| compute.dispatch | Method not found on ToadStool | PG-15 |
| inference.complete | Squirrel BTSP only | PG-14 |
| stats.median | Unknown method in ecobin | PG-13 |
| linalg.determinant | Unknown method in ecobin | PG-13 |

## New Gaps Discovered (PG-13 to PG-17)

### PG-13: barraCuda Missing 6 Manifest Methods
stats.variance, stats.correlation, linalg.solve, linalg.eigenvalues,
spectral.fft, spectral.power_spectrum are in the v0.9.16 downstream
manifest but not in the barraCuda ecobin. Blocks Level 5.

### PG-14: Squirrel BTSP-Only Socket
Squirrel requires BTSP handshake on UDS. Plain JSON-RPC clients get
connection reset. Need either plain JSON-RPC fallback or BTSP client
in primalspring::composition.

### PG-15: ToadStool compute.dispatch Missing
The JSON-RPC socket responds but `compute.dispatch` is not registered.
ToadStool has tarpc and BTSP sockets — may need different routing.

### PG-16: stats.std_dev N-1 vs N Convention
barraCuda uses sample std_dev (N-1, Bessel's). wetSpring's bare B0
uses population std_dev (N). Both correct; guideStone uses sample for
IPC and population for bare. Document as intentional convention gap.

### PG-17: tensor.matmul Handle-Based API
`tensor.matmul` requires pre-created tensor handles (`lhs_id`, `rhs_id`).
`validate_parity` expects scalar `result` field. guideStone works around
this with create→matmul→check-shape. Need either inline-data path in
barraCuda or handle-aware parity helper in primalspring::composition.

## Key Code Changes

- `wetspring_guidestone.rs`: Handle-based tensor.matmul, sample std_dev
  (√250), base64-encoded crypto.hash, `validate_parity_or_skip` for
  unimplemented methods, fixed `stats.weighted_mean` param name (`values`).
- `niche.rs`: `GUIDESTONE_READINESS = 4`.
- `docs/PRIMAL_GAPS.md`: PG-13 through PG-17 added.

## Next Steps → Level 5

1. **barraCuda team** implements PG-13 methods (6 missing)
2. **Squirrel team** adds plain JSON-RPC fallback (PG-14)
3. **ToadStool team** registers compute.dispatch (PG-15)
4. Once 11 skips → 0, wetSpring achieves full N1/N2 parity
5. Cross-substrate validation (Python/CPU/GPU) for Level 5 certification

## Verification

```bash
# Bare mode (no NUCLEUS)
cargo run --features guidestone --bin wetspring_guidestone
# → 16/16 pass, exit 2

# NUCLEUS mode
FAMILY_ID=wetspring-validation ./target/release/wetspring_guidestone
# → 31/31 pass (11 skip), exit 0
```
