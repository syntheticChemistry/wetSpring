# wetSpring V96 — Deep Debt Audit + Chuna Paper Queue (barraCuda v0.3.1)

**Date**: 2026-03-05
**From**: wetSpring
**To**: toadStool / barraCuda team, hotSpring, neuralSpring
**Predecessor**: V95 Cross-Spring Evolution Complete
**Scope**: Deep debt elimination, silent fallback evolution, capability-based
discovery, platform portability, Chuna paper queue integration

---

## Summary

Comprehensive deep debt audit of wetSpring. Eliminated all silent fallbacks
in library code, evolved hardcoded paths to capability-based discovery,
made benchmarks platform-agnostic, feature-gated the vault module, pinned
the Python baseline environment, and added Thomas Chuna's papers to the
hotSpring (3 papers) and neuralSpring (1 paper) review queues.

**58 files changed, +532/−295 lines.** Zero new dependencies. Zero new
binaries. All changes are quality and correctness improvements.

---

## Deep Debt Changes

### Silent Fallback Evolution

| File | Before | After | Impact |
|------|--------|-------|--------|
| `taxonomy/classifier.rs` | `classify_quantized` returns `0` for empty/unclassifiable | Returns `Option<usize>` — `None` for empty/unclassifiable | 5 callers updated to handle `Option` |
| `dada2_gpu.rs` | `unwrap_or(0)` for `center_slot` | `Option` guard with `continue` on missing center | Prevents silent misclassification |
| `validate_emp_anderson_atlas.rs` | Hardcoded `/run/user/.../biomeos-default.sock` | `ipc::discover::discover_socket()` | Capability-based IPC discovery |

### Platform Portability

| File | Before | After |
|------|--------|-------|
| `bench/mod.rs` | `read_to_string("/proc/self/status")` unconditionally | `#[cfg(target_os = "linux")]` gated, returns `0.0` elsewhere |
| `bench/hardware.rs` | Direct `/proc/cpuinfo` and `/proc/meminfo` reads | `try_read()` helper with graceful fallback |

### Feature Gating

| Feature | Dependencies | Binaries Requiring |
|---------|-------------|-------------------|
| `vault` | `chacha20poly1305`, `ed25519-dalek`, `blake3` (now optional) | `validate_genomic_vault`, `validate_barracuda_cpu_v20`, `validate_metalforge_v12_extended` |
| `gpu` | Now also forwards `barracuda/domain-esn` | Fixes `--all-features` doc/clippy for `toadstool_bridge.rs` |

### Code Quality

| Metric | Before | After |
|--------|--------|-------|
| TODO/FIXME/HACK in .rs files | 0 | 0 (confirmed across 453 files) |
| `todo!()` / `unimplemented!()` | 0 | 0 |
| clippy warnings (pedantic+nursery) | 0 | 0 |
| cargo doc warnings | 0 | 0 (domain-esn forwarding fixed) |
| Silent fallbacks in lib code | 3 | 0 |
| Platform-specific hardcoding | 2 | 0 |

### Validation Helpers

New shared utilities in `validation/mod.rs`:
- `gpu_or_skip()` — standardized GPU availability check with skip message
- `DomainResult` — structured pass/fail/skip per validation domain
- `print_domain_summary()` — consistent summary output across validation binaries

Deduplicated from: `validate_dispatch_overhead_proof.rs`,
`validate_pure_gpu_streaming.rs`, `validate_barrier_disruption_s79.rs`

---

## Chuna Paper Queue (Murillo Group, MSU)

Thomas Chuna — PhD student, MSU Physics & CMSE, Murillo Group. Referred by
Murillo (March 4, 2026). Profile: `whitePaper/attsi/non-anon/contact/murillo/chuna_profile.md`

### hotSpring (3 papers — exact physics)

| # | Paper | Tier | Priority |
|---|-------|------|----------|
| 43 | Bazavov & Chuna — SU(3) gradient flow integrators (arXiv:2101.05320) | Tier 2 (QCD) | P2 — extends Papers 8-12. Compare Lie group RK vs our Omelyan HMC |
| 44 | Chuna & Murillo — Conservative dielectric functions from BGK (Phys Rev E 2024) | Tier 4 (WDM) | P2 — extends Papers 1/5. Completed Mermin susceptibility, f-sum rule |
| 45 | Haack, Murillo, Sagert & Chuna — Multi-species kinetic-fluid coupling (JCP 2024) | Tier 4 (HED) | P3 — new domain. Hybrid kinetic-hydro multi-fidelity |

### neuralSpring (1 paper — ML/learning)

| # | Paper | Priority |
|---|-------|----------|
| 26 | Chuna — T1D blood glucose LSTM prediction (arXiv:2005.09051, 2020) | Queue — same LSTM as Exp 3/9 weather, different domain. Validates isomorphic learning |

### Not queued to wetSpring

None of Chuna's papers are wet lab / life science domain. Correct routing:
hotSpring (exact physics) and neuralSpring (ML).

---

## What toadStool / barraCuda Team Should Know

### 1. `vault` Feature Now Optional

The `vault` module (ChaCha20-Poly1305 + Ed25519 + BLAKE3) is feature-gated.
Three validation binaries require `--features vault`. This reduces default
compile time and dependency surface for non-vault users.

### 2. `domain-esn` Feature Forwarding

The `gpu` feature in wetSpring now forwards `barracuda/domain-esn` to the
upstream barraCuda crate. Without this, `--all-features` builds fail on
`barracuda::esn_v2` imports in `toadstool_bridge.rs`. This is a downstream
workaround; upstream could consider making `domain-esn` default or
documenting the required feature combination.

### 3. `classify_quantized` API Change

If any other Spring or tool consumes `NaiveBayesClassifier::classify_quantized`,
it now returns `Option<usize>` instead of `usize`. Callers must handle `None`.

### 4. Validation Helpers Available for Adoption

The `validation::gpu_or_skip()` + `DomainResult` + `print_domain_summary()`
pattern could be promoted to barraCuda as a shared validation framework.
Currently local to wetSpring. Pattern:

```rust
let gpu = validation::gpu_or_skip();
let mut results = vec![];
results.push(DomainResult { name: "domain", passed: n, failed: 0, skipped: 0 });
validation::print_domain_summary(&results);
```

### 5. Python Baseline Environment Pinned

`scripts/requirements.txt` now pins the Python baseline environment
(numpy>=1.24,<2; scipy>=1.10,<2; etc.) ensuring reproducibility.

---

## Quality Gate Results

| Check | Status |
|-------|--------|
| `cargo test --workspace --lib` | PASS — 1,261 tests, 0 failures |
| `cargo clippy -p wetspring-barracuda -W clippy::pedantic -W clippy::nursery` | PASS — 0 warnings |
| `cargo fmt -p wetspring-barracuda -- --check` | PASS |
| `cargo doc -p wetspring-barracuda --all-features` | PASS — 0 warnings |
| TODO/FIXME/HACK scan (453 .rs files) | PASS — 0 found |
| Silent fallback scan | PASS — 0 remaining in library code |

---

## Files Changed (58 total)

### Library code (correctness)
- `bio/taxonomy/classifier.rs` — `Option<usize>` return type
- `bio/dada2_gpu.rs` — `Option` guard for center_slot
- `bio/ani.rs`, `bistable.rs`, `cooperation.rs`, `diversity_gpu.rs`,
  `eic_gpu.rs`, `kmd.rs`, `kmd_gpu.rs`, `neighbor_joining.rs`, `ode.rs`,
  `pangenome.rs`, `pcoa.rs`, `rarefaction_gpu.rs`, `signal.rs`, `snp.rs`,
  `spectral_match.rs`, `stats_extended_gpu.rs`, `tolerance_search.rs` —
  clippy pedantic fixes, named tolerance adoption
- `bench/mod.rs`, `bench/hardware.rs` — platform-agnostic
- `df64_host.rs`, `special.rs` — clippy fixes
- `validation/mod.rs` — shared helpers
- `lib.rs` — vault feature gate
- `Cargo.toml` — vault feature, domain-esn forwarding

### Validation binaries (callers)
- 15 validate_*.rs binaries — `Option` handling, deduplication, tolerance naming

### Test and data files
- `io/fastq/tests.rs` — tolerance naming
- `tolerances/bio/esn.rs`, `ode.rs`, `misc.rs`, `mod.rs` — new named constants
- `validation/tests.rs` — new test file for validation helpers
- `experiments/results/` — whitespace/format normalization (4 files)

---

## Architectural State After V96

```
wetSpring (V96) ─── depends on ──→ barraCuda v0.3.1 (standalone)
    │                                   │
    │ 0 local WGSL shaders              │ 767+ WGSL shaders
    │ 0 TODO/FIXME/HACK                 │ f64-canonical precision
    │ 0 silent fallbacks                │ Precision::F16/F32/F64/Df64
    │ 0 platform-specific hardcoding    │
    │ 164 named tolerances              │ All springs contribute shaders
    │ 94.69% line coverage              │ All springs consume shaders
    │ 150+ consumed primitives          │
    │ vault feature-gated               │
    └───────────────────────────────────┘
```

## Handoff Prepared By

- wetSpring V96 deep debt audit
- All quality gates green
- hotSpring and neuralSpring paper queues updated and pushed
