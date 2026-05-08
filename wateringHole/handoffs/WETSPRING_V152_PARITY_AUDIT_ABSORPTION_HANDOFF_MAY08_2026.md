# wetSpring V152 — primalSpring Phase 60 Parity Audit Absorption

**From**: wetSpring  
**Date**: May 8, 2026  
**To**: primalSpring, all primal teams, all delta springs  
**Audit Reference**: `primalSpring/docs/CROSS_SPRING_PARITY_SCORECARD.md` (Phase 60, v0.9.25)

---

## Executive Summary

wetSpring absorbed all three **universal evolution targets** and all three **wetSpring-specific targets** from primalSpring's Phase 60 cross-spring parity audit. Score remains **STRONG** with improved security posture, IPC-first architecture, and the first standalone composition experiment crate.

## What We Implemented

### 1. `ring` Crate Ban — Security Posture Alignment

**Target**: Only spring not banning `ring`.  
**Action**: Added `{ crate = "ring", reason = "Non-auditable C/ASM — use BearDog crypto primals" }` to both `deny.toml` (workspace root) and `barracuda/deny.toml`.  
**Verification**: `cargo tree -i ring` confirms ring is not in the dependency graph. Ban is a policy guardrail against future transitive inclusion.

### 2. Registry Cross-Sync Test

**Target**: No spring tests its methods against primalSpring's canonical 389-method registry.  
**Action**: Created `tools/check_registry_sync.sh` — two-phase check:

- **Phase 1 (Cross-sync)**: Compares wetSpring's `capability_registry.toml` (37 methods) against `primalSpring/config/capability_registry.toml` (389 methods). Reports 34 wetSpring domain methods not yet in the canonical registry.
- **Phase 2 (Local advisory)**: Scans Rust source for dotted method strings not in the local TOML. Advisory only — springs naturally consume ecosystem methods from other primals.

**Result**: 34 wetSpring-specific methods need upstream absorption by primalSpring:
- 12 `science.*` methods (diversity, alignment, phylogenetics, etc.)
- 7 `science.anderson.*` methods
- 3 `science.gonzales.*` methods
- 3 `data.fetch.*` methods
- 3 `vault.*` methods
- 3 `brain.*` / `provenance.*` / `ai.*` methods
- 3 infrastructure methods (composition, metrics)

**CI ready**: Script supports `PRIMALSPRING_PATH` override; resolves sibling paths automatically.

### 3. barraCuda `optional = true` (IPC-First)

**Target**: Every spring links barraCuda as a mandatory path dep.  
**Action**: Made `barracuda` dependency `optional = true` in `barracuda/Cargo.toml`. Added `barracuda-lib` feature to `default` features. All existing features (`gpu`, `sovereign-dispatch`) transitively enable `barracuda-lib`.

**Effect**: Default builds are unchanged (barracuda-lib is in `default`). Sovereign NUCLEUS deployment can disable it:
```toml
wetspring-barracuda = { path = "...", default-features = false, features = ["ipc"] }
```

**Verification**: `cargo check` and `cargo test --lib` (252/252 pass) confirm no regressions.

### 4. Composition Experiment Crate (exp400)

**Target**: 0 experiment crates. Extract composition validation from guidestone into standalone crate.  
**Action**: Created `experiments/exp400_nucleus_composition_parity/` following primalSpring's exp094/exp095 pattern:

- **Tower Atomic**: BearDog health, BLAKE3 crypto hash (determinism check)
- **Node Atomic**: compute health, stats.mean IPC parity (expected=3.0)
- **Nest Atomic**: NestGate storage health, provenance trio health
- **Niche**: wetSpring science health, Shannon diversity IPC parity, capability list
- **Cross-Atomic Pipeline**: hash → store → science (end-to-end NUCLEUS)

Pattern: discover → call → extract → compare → report (skip-tolerant when primals are offline).  
Added to workspace `members` in root `Cargo.toml`. Compiles clean.

---

## Updated Scorecard (wetSpring)

| Axis | Before | After |
|------|--------|-------|
| deny.toml `ring` ban | **Y** (missing) | **G** (both root + barracuda) |
| Registry cross-sync | **Y** (TOML, no test) | **G** (script + CI-ready) |
| barraCuda coupling | **Y** (mandatory path dep) | **G** (optional, IPC-first default) |
| Composition experiments | **R** (0 crates) | **Y** (1 crate: exp400) |
| Tests | 1,594 | 1,594+ (252 lib confirmed) |
| Notebooks | 19 | 29 (10 paper notebooks added) |

## What Stays Gap (Upstream Work)

### For primalSpring

1. **Absorb 34 wetSpring domain methods** into `config/capability_registry.toml`. List provided in `tools/check_registry_sync.sh` output. These are wetSpring's registered IPC methods that the canonical registry doesn't yet include.

2. **exp094 template needs `check_skip` semantics**: The exp094 `CompositionContext` doesn't export a `check_skip` equivalent for offline primals. We implemented our own harness; would be cleaner if `primalspring::validation::ValidationResult` supported skip counts.

### For wetSpring (Evolution Roadmap)

1. **More experiment crates**: exp400 is the first. Future crates should isolate:
   - Gonzales provenance chain (currently in `validate_gonzales_provenance_chain`)
   - Cross-spring spectral theory (currently in `validate_spectral_cross_spring`)
   - EMP Anderson atlas (currently in `validate_emp_anderson_atlas`)

2. **Guidestone L4 → L5**: Currently 38/38 NUCLEUS checks pass. L5 requires primalSpring unconditional dep + `ValidationResult` integration (blocked on primalSpring exporting skip semantics).

3. **Notebook count**: 29 now (19 meta + 10 paper). More paper notebooks can be extracted as new baseCamp papers are absorbed.

## Patterns That Work Well

- **Feature-gated optional deps**: `barracuda-lib` as default feature lets us mark the intent (IPC-first) without breaking any existing build path. Other springs should follow this pattern.
- **Two-phase registry sync**: Cross-sync (your methods vs canonical) + local advisory (source strings vs your TOML). The local phase is intentionally advisory because springs consume methods from other primals.
- **Harness-based experiment crates**: Self-contained binaries with skip semantics let experiments run anywhere (CI, laptop, HPC) and report gaps without failing.

## Files Changed

```
deny.toml                                          — ring ban added
barracuda/deny.toml                                — ring ban added
barracuda/Cargo.toml                               — barracuda optional + barracuda-lib feature
Cargo.toml                                         — exp400 in workspace members
tools/check_registry_sync.sh                       — NEW: registry cross-sync test
experiments/exp400_nucleus_composition_parity/      — NEW: NUCLEUS composition parity crate
```
