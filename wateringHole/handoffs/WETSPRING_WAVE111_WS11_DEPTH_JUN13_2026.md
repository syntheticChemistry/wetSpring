# wetSpring Wave 111 — Deep Debt + WS-11 Binomial Model Evolution

**Gate:** southGate  
**Date:** 2026-06-13  
**Versions:** V203–V206  
**Tests:** 2,124 → 2,160 (+36)  
**Scenarios:** 345 → 346 (+1: mesh-health)

---

## Shipped (V203–V206)

### V203 — Module Splits + Macro Consolidation
- `gonzales.rs` → `gonzales/` submodule (3 files: pharmacology, tissue, anderson_sweeps)
- `discover.rs` macro: 8 repetitive wrappers → `primal_discover_fn!` (70L→20L)
- `variant_caller/stats.rs`: 267L pure-math extracted from 873L monolith

### V204 — Mesh Health in Certify + Scenario
- `wetspring certify` Layer 3b: 13-primal mesh health audit + version skew detection
- `mesh-health` validation scenario (Tier 2, Composition track)
- Feature-gated (`barracuda-lib`) with graceful skip

### V205 — WS-11 Binomial Model Evolution
- MAPQ-aware quality weighting: P(err) = P_base + P_map - P_base×P_map
- Per-generation LTEE frequency thresholds (`thresholds` module)
- breseq cross-validation engine (`cross_validation` module)
- `PileupColumn.mapq_sums` per-base MAPQ tracking

### V206 — MAPQ Calibration Module
- Simulated read generator (known-position, controlled error, xorshift64 PRNG)
- Training pipeline: simulate → map → compare → build model
- `MapqModel` lookup table (score_gap → Phred MAPQ)
- Wired into `compute_mapq` via `MapperConfig.mapq_model`
- Full chain: calibrate → MapqModel → pileup → binomial_quality

---

## Stream 6 Contributions

- `composition.mesh_health` integrated into certification (Layer 3b)
- Version skew detection for divergence pressure
- Per-primal liveness audit (13/13 NUCLEUS primals)

---

## WS-11 Variant Caller Parity — Status 5/8

| Item | Status |
|------|--------|
| Quality-weighted binomial model | ✅ V201 |
| MAPQ-aware binomial weighting | ✅ V205 |
| Per-generation frequency thresholds | ✅ V205 |
| Cross-validate vs breseq polymorphism | ✅ V205 |
| MAPQ calibration (training set + model) | ✅ V206 |
| Re-run Barrick 2009 with v3 pipeline | ❌ pending |
| GPU binomial shader (SnpCallingF64) | ❌ future |
| Mapper deduplication for repetitive regions | ❌ pending |

---

## Upstream Gaps (for primal team review)

### For primalSpring
- `nucleus_launcher` should prepend riboCipher clear signal `[0xEC, 0x01]` (Stream 7)
- Proto-nucleate manifest: sub-NUCLEUS topology definition (P3)

### For toadStool
- TOADSTOOL-AUTO-REGISTER: PCI/sysfs enumeration → biomeOS (code reportedly done, validation pending)

### For barraCuda
- `SnpCallingF64` GPU shader should incorporate quality-weighted binomial model
- No urgency — CPU path is production-complete

### For songBird
- VPS persistent relay rebuild needed (auto-rebuild should self-heal)

### For cellMembrane
- VPS rebuild to `082d77c` (freshness race fix)
- `uds_jsonrpc_call()` riboCipher signal prepend (Stream 7)

---

## Build Gate

- clippy: 0 warnings (pedantic + nursery, all feature combinations)
- tests: 2,160 (0 failures, 3 ignored)
- scenarios: 346
- unsafe: 0 (workspace `forbid(unsafe_code)`)
- cargo-deny: clean

---

## Wave 111 Exit Assessment

wetSpring is **code-complete** for Wave 111. No overwatch obligations remain.
riboCipher (Stream 7) does not assign wetSpring a task — we're a consumer that
will inherit compliance through barraCuda's transport layer when cellMembrane
and primalSpring ship signal-prepend changes.

Next depth: WS-11 remaining 3 items (Barrick re-run, GPU shader, mapper dedup).
