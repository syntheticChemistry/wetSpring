# wetSpring → ToadStool/BarraCuda V86 Cross-Spring Evolution Handoff

**Date:** March 1, 2026
**From:** wetSpring V86
**To:** ToadStool / BarraCuda team
**Supersedes:** V85 Vault NUCLEUS Evolution Handoff
**ToadStool Pin:** S71+++
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **Validated clean against ToadStool S71+++** (universal precision, DF64 transcendentals, ComputeDispatch migration)
- **Deep debt elimination:** 75 files changed, -4,753 net lines
- **Cross-spring evolution validation:** 23/23 checks across 5 Springs
- **ESN bridge to ToadStool esn_v2:** BioEsn with multi-head bio classifiers
- **4 large module refactors:** node, inventory, tolerances/bio, dada2
- **1,247 tests, 0 failures,** clippy pedantic clean

---

## S71 Evolution Acknowledgment

1. **Universal precision** (op_preamble + naga IR rewrite) — math is universal, precision is silicon
2. **Zero f32-only shaders** — all canonical f64 with downcast
3. **DF64 transcendentals complete** (gamma, erf)
4. **ComputeDispatch builder** for 66+ ops
5. **All wetSpring imports and primitives verified compatible**

---

## What Changed (V85 → V86)

| Category | Files | Description |
|----------|-------|-------------|
| ToadStool rewiring | ~15 | local `bray_curtis`, `variance` → `barracuda::` primitives |
| ESN evolution | 6 | LegacyEsn + BioEsn bridge to `esn_v2` (config, reservoir, training, toadstool_bridge, npu, tests) |
| Module decomposition | 9 | workloads, nanopore, esn, nestgate, ms2, node, inventory, tolerances/bio, dada2 |
| Cross-spring validation | 1 | `validate_cross_spring_evolution_modern` — 23 checks, 5 Springs |
| Cross-spring benchmark | 1 | `benchmark_cross_spring_modern` — 12 primitives by origin |
| Deep debt | 43+ | magic numbers, clone audit, test coverage, path discovery |

---

## Absorption Targets for ToadStool

What ToadStool should absorb or evolve:

1. **`BioEsn` multi-head pattern** → generalize to all Springs
2. **`BioHeadKind` enum** → domain-agnostic head registry (Diversity, Taxonomy, Amr, Bloom, Disorder, Custom)
3. **Cross-spring validation pattern** → shared cross-spring test harness
4. **Tolerance organization** (diversity, alignment, phylogeny, etc.) → model for other Springs

---

## For ToadStool Team

1. **Review BioEsn bridge** — `barracuda/src/bio/esn/toadstool_bridge.rs` wraps `esn_v2::ESN` with bio feature extraction and multi-head classifiers.
2. **Consider head registry** — `BioHeadKind` maps domain heads to indices; generalize for non-bio Springs.
3. **Adopt cross-spring harness** — `validate_cross_spring_evolution_modern` proves primitives from all five Springs work in wetSpring; replicate pattern for other Springs.
4. **Tolerances model** — `tolerances/bio/` (alignment, anderson, diversity, esn, misc, ode, parity, phylogeny) provides domain-specific tolerance constants; consider similar structure upstream.

---

## Reproduction

```bash
cargo test --workspace                                    # 1,247 passed
cargo run --bin validate_cross_spring_evolution_modern    # 23/23 PASS
cargo run --release --bin benchmark_cross_spring_modern    # 12 primitives
cargo clippy --workspace --all-targets --all-features -- -W clippy::pedantic -D warnings
```
