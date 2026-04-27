# wetSpring V146 — guideStone Level 2 Handoff

**Date:** 2026-04-18
**From:** wetSpring V146
**To:** primalSpring, barraCuda, toadStool, BearDog, NestGate, Squirrel,
        all spring teams
**Reference:** `GUIDESTONE_COMPOSITION_STANDARD.md`, hotSpring-guideStone-v0.7.0

---

## What Changed

wetSpring now has a **guideStone binary** (`wetspring_guidestone`) that uses
the `primalspring::composition` API for standardized NUCLEUS IPC validation.
This evolves Exp403 (hand-rolled UDS calls) into the ecosystem-standard
guideStone pattern (Level 2: properties documented).

### New Artifacts

| Artifact | Location | What It Does |
|----------|----------|--------------|
| `wetspring_guidestone` binary | `barracuda/src/bin/wetspring_guidestone.rs` | Self-validating NUCLEUS composition node |
| `guidestone` feature | `barracuda/Cargo.toml` | Gates primalspring dep + IPC |
| `GUIDESTONE_BINARY` const | `barracuda/src/niche.rs` | Manifest alignment |
| `GUIDESTONE_READINESS` const | `barracuda/src/niche.rs` | Level 2 (properties documented) |

### Validation Layers

| Layer | What It Validates | Requires NUCLEUS? |
|-------|------------------|------------------|
| **B0 — Bare Science** | Shannon diversity, Hill function, mean, std_dev, matmul, weighted_mean, self-verification | No |
| **B1 — Tolerance Provenance** | `ANALYTICAL_F64 == 0.0`, `IPC_ROUND_TRIP_TOL ∈ (0, 1e-10]` | No |
| **N0 — Liveness** | tensor, security, storage, compute, ai capabilities discoverable | Yes |
| **N1 — Manifest IPC Parity** | 7 `validation_capabilities` from `downstream_manifest.toml` | Yes |
| **N2 — Domain Science** | `stats.std_dev`, `stats.weighted_mean` via IPC | Yes |
| **N3 — Cross-Atomic Pipeline** | BearDog hash → NestGate store → retrieve → verify | Yes |

### The 5 Certified Properties (Status)

| Property | Status | Evidence |
|----------|--------|----------|
| 1. Deterministic Output | **Proven** | Layer B0: 7 analytical baselines, all exact |
| 2. Reference-Traceable | **Proven** | Every baseline labeled with paper/source |
| 3. Self-Verifying | **Proven** | Tampered input → detected in B0 |
| 4. Environment-Agnostic | **Proven** | Pure Rust, no network, no sudo, no GPU |
| 5. Tolerance-Documented | **Proven** | Layer B1: named constants with derivation |

### Exit Codes

- `0` — all checks passed (NUCLEUS certified)
- `1` — at least one check failed
- `2` — bare-only mode (no primals discovered)

---

## Composition API Usage

```rust
use primalspring::composition::{CompositionContext, validate_liveness, validate_parity};
use primalspring::tolerances as ps_tol;
use primalspring::validation::ValidationResult;

let mut v = ValidationResult::new("wetSpring guideStone");
let mut ctx = CompositionContext::from_live_discovery_with_fallback();
let alive = validate_liveness(&mut ctx, &mut v, &["tensor", "security"]);

validate_parity(
    &mut ctx, &mut v,
    "stats.mean([10..50])", "tensor", "stats.mean",
    json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]}),
    "result", 30.0, ps_tol::IPC_ROUND_TRIP_TOL,
);
```

---

## guideStone Readiness

| Level | Requirement | wetSpring Status |
|-------|-------------|-----------------|
| 0 | Not started | ✓ (past) |
| 1 | Validation binary exists (Exp403) | ✓ V145 |
| **2** | **Properties documented, guideStone binary** | **✓ V146 ← CURRENT** |
| 3 | Bare guideStone works (no NUCLEUS) | Next: already works locally |
| 4 | NUCLEUS guideStone works (live primals) | Needs plasmidBin deployment |
| 5 | Certified (cross-substrate parity) | Requires hotSpring-equivalent audit |

---

## Gaps Discovered (Upstream Evolution Needed)

| Gap | Owner | Impact |
|-----|-------|--------|
| `result_key` normalization — barraCuda returns `value`/`mean`/bare, primalspring expects `result` | barraCuda + primalspring | `validate_parity` may skip if key doesn't match |
| `compute.dispatch` schema standardization | toadStool | Params format unclear for generic dispatch |
| `storage.retrieve` response schema | NestGate | Field may be `value` or `data` — not standardized |
| `inference.complete` response schema | Squirrel | Field may be `text`, `completion`, or `content` |
| Cross-atomic pipeline assumes `storage.store` accepts `value` key | NestGate | Not documented in capability spec |

---

## What Each Team Should Do

### primalSpring
- Update `downstream_manifest.toml`: bump `guidestone_readiness = 2` for wetspring
- Add `guidestone_properties = ["deterministic", "reference-traceable", "self-verifying", "environment-agnostic", "tolerance-documented"]`
- Consider standardizing `result_key` across all primal JSON-RPC responses

### barraCuda
- Document canonical response key for scalar results (is it `result`, `value`, or method-specific?)
- This affects all downstream `validate_parity` calls

### NestGate
- Standardize `storage.retrieve` response: always `{"value": ...}` or always `{"data": ...}`?
- Document `storage.store` params schema

### All Spring Teams
- Review the guideStone standard: `primalSpring/wateringHole/GUIDESTONE_COMPOSITION_STANDARD.md`
- If at Level 0-1: evolve toward Level 2 (document 5 properties, build guideStone binary)
- Use the composition API — it handles discovery, fallback, and skip semantics

---

## Verification

```bash
# Build
cargo build --features guidestone --bin wetspring_guidestone

# Run bare (no NUCLEUS — exit 2 expected)
cargo run --features guidestone --bin wetspring_guidestone
echo $?  # expect 2

# Run with NUCLEUS (deploy from plasmidBin first)
biomeos deploy --graph wetspring_science_nucleus.toml
cargo run --features guidestone --bin wetspring_guidestone
echo $?  # expect 0

# Clippy
cargo clippy --features guidestone -p wetspring-barracuda --bin wetspring_guidestone -- -D warnings

# Tests
cargo test --features ipc -p wetspring-barracuda --lib  # 1,592 passed
```

---

*This handoff complements: `WETSPRING_V145_PRIMAL_PROOF_TIER2_HANDOFF_APR17_2026.md`
(Exp403 hand-rolled IPC) and `WETSPRING_V145_ECOSYSTEM_EVOLUTION_HANDOFF_APR17_2026.md`
(full validation ladder). The guideStone supersedes Exp403 for Level 5+.*
