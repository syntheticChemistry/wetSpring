# WETSPRING V147 — guideStone Level 3 Handoff

**Date:** April 18, 2026
**From:** wetSpring
**To:** primalSpring, barraCuda, sibling spring teams
**Status:** guideStone Level 3 — bare mode certified, N2 expanded to v0.9.15

---

## What Changed (V146 → V147)

### 1. Bare Mode Certified (Level 3)

The `wetspring_guidestone` binary now passes all bare checks (B0 + B1) and
correctly exits with code 2 when no NUCLEUS is deployed:

- **B0 — Bare Science Baselines** (7 checks, all PASS):
  Shannon H', Hill function, mean, std_dev, matmul, weighted_mean, self-verify
- **B1 — Tolerance Provenance** (2 checks, all PASS):
  `ANALYTICAL_F64 = 1e-12`, `IPC_ROUND_TRIP_TOL ∈ (0, 1e-10]`
- **Exit code 2** enforced for bare-only (was incorrectly 0 in V146)

### 2. N2 Expanded with v0.9.15 Canonical Surface

Six new `validate_parity` checks added to the N2 (domain science IPC) layer,
all using analytically derivable baselines:

| Method | Baseline | Derivation |
|--------|----------|------------|
| `stats.variance` | `[10,20,30,40,50]` → 200.0 | Population variance Σ(x−μ)²/N |
| `stats.median` | `[1,3,5,7,9]` → 5.0 | Odd-length sorted array, middle element |
| `stats.correlation` | `(x, 2x)` → 1.0 | Perfect linear correlation |
| `linalg.determinant` | `[[1,2],[3,4]]` → −2.0 | ad − bc |
| `linalg.eigenvalues` | `[[2,1],[1,2]]` → min = 1.0 | Symmetric 2×2 closed form |
| `spectral.fft` | `[1,0,0,0]` → X[0] = 1.0 | Unit impulse DFT |

### 3. CONSUMED_CAPABILITIES Aligned to v0.9.15

Restructured `niche::CONSUMED_CAPABILITIES` from 22 → 48:
- **33 v0.9.15 canonical**: TENSOR(9), STATS(9), COMPUTE(4), SPECTRAL(3), LINALG(6), HEALTH(2)
- **15 legacy** (Exp403 Tier 2): tensor.scale, noise.perlin2d, activation.fitts, etc.
- Renamed: `tensor.reduce` → `tensor.reduce_sum`

### 4. Exit Code Semantics Fixed

- **Exit 0** = full NUCLEUS certification (all N0–N3 layers pass)
- **Exit 1** = at least one check failed
- **Exit 2** = bare-only mode (no primals discovered, bare properties valid)

Previously, bare mode incorrectly exited 0 when bare checks passed.

### 5. GUIDESTONE_READINESS Promoted: 2 → 3

`niche::GUIDESTONE_READINESS = 3` (bare guideStone works).

---

## New Gaps Discovered (PG-10, PG-11, PG-12)

### PG-10: spectral/linalg Routing (primalSpring)

`method_to_capability_domain()` in `primalspring::composition` does not route
`spectral.*` or `linalg.*` prefixes to `"tensor"` (barraCuda). They fall
through to the default `_ => prefix` branch. Workaround: wetSpring passes
`"tensor"` explicitly. Fix: add `"spectral" | "linalg"` to the tensor match arm.

### PG-11: Manifest Drift (primalSpring)

`downstream_manifest.toml` for wetspring lists 7 `validation_capabilities` and
`guidestone_readiness = 1`. The actual guideStone N2 validates 15 methods, and
readiness is now 3. Manifest should be updated.

### PG-12: Exp403 Legacy Surface (wetSpring)

Exp403 uses 15 methods not on the v0.9.15 canonical surface. These should be
migrated or gated behind a feature flag.

---

## Verification

```bash
# Clippy — zero warnings
cargo clippy --features guidestone -p wetspring-barracuda --bin wetspring_guidestone -- -D warnings

# Bare mode — 9/9 pass, exit 2
./target/release/wetspring_guidestone
echo $?  # → 2

# All lib tests — 1,594 pass
cargo test --features ipc -p wetspring-barracuda --lib
```

---

## guideStone Readiness Table

| Spring | Version | Level | Status |
|--------|---------|-------|--------|
| hotSpring | v0.6.32 | 5 | CERTIFIED (reference) |
| **wetSpring** | **V147** | **3** | **bare certified, N2 v0.9.15** |
| healthSpring | V53 | 1 | exp122 IPC exists |
| neuralSpring | V133 | 1 | IpcMathClient exists |
| ludoSpring | V44 | 1 | validate_primal_proof exists |
| airSpring | v0.10.0 | 0 | no IPC client |
| groundSpring | V124 | 0 | no IPC client |
| primalSpring | v0.9.15 | 1 | 6-layer base certification |

---

## Actionable Feedback

### For primalSpring
- Fix `method_to_capability_domain()` routing for `spectral` and `linalg` (PG-10)
- Update `downstream_manifest.toml` wetspring entry (PG-11)

### For barraCuda
- Confirm `stats.variance`, `stats.median`, `stats.correlation`,
  `linalg.determinant`, `linalg.eigenvalues`, `spectral.fft` are served
  over JSON-RPC IPC (wetSpring's N2 depends on them)
- Clarify `stats.weighted_mean` status in v0.9.15 (still served?)

### For sibling springs
- The bare mode pattern (B0/B1 → N0 liveness → exit 2 if bare) is validated
  and can be adopted as-is. Exit code fix (2 for bare, not 0) is important.
- N2 analytical baselines can be shared: use closed-form values, not Python.

---

*This handoff follows `WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md` convention.*
