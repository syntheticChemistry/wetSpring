# Exp331: Local Evolution & Upstream Readiness

## Status: PASS (24/24)

**Binary**: `validate_local_evolution_v1`
**Crate**: `wetspring-forge`

## Scope

Validates all idiomatic improvements made for upstream absorption readiness:

| Domain | What Changed | Checks |
|--------|-------------|--------|
| E1 FitResult | `pangenome.rs` migrated from `.params[0]` to `.slope()` | 7 |
| E2 HmmModel | `#[doc(alias = "HMM")]` and `#[doc(alias = "HiddenMarkovModel")]` added | 5 |
| E3 NMF | Re-exported from `bio::nmf` for domain-level discovery | 4 |
| E4 Quality | 239 LOC tests extracted to `quality_tests.rs` via `#[path]` | 3 |
| E5 Workloads | `data_bytes` wired into 4 bandwidth-sensitive workloads | 5 |

## Key Decisions

- **FitResult**: `.slope()` returns `Option<f64>` so we use `.and_then(|r| r.slope())` instead of `.map(|r| r.params[0])` — type-safe and self-documenting.
- **NMF re-export**: `pub use barracuda::linalg::nmf;` in `bio/mod.rs` — bio callers find NMF alongside diversity, HMM, etc.
- **Test extraction**: Follows V93+ pattern (`#[path = "quality_tests.rs"]`) — `mod.rs` drops from 547 to 308 lines.
- **data_bytes**: Representative sizes for bandwidth-aware dispatch: kmer 10MB, smith_waterman 50MB, pcoa 8MB, dada2 100MB.

## Chain Position

```
Exp327 (viz schema) → Exp328 (CPU/GPU math) → Exp329 (metalForge viz)
    → Exp330 (full chain) → Exp331 (local evolution) → Exp332 (mixed HW)
```
