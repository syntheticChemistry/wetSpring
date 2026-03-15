# Experiment 375: V115 Deep Audit + UniBin + Capability Domains

**Date:** March 15, 2026
**Version:** V115
**Status:** DONE — all 12 audit findings executed and verified

---

## Summary

Comprehensive 12-finding audit executed from V114 deep analysis. All changes
verified with `cargo check --workspace`, `cargo clippy --workspace -- -D warnings`,
and `cargo test --workspace` (1,662 tests pass, 0 failures).

## Changes

### UniBin Compliance
- `wetspring_server` binary → `wetspring` (ecoBin standard)
- Subcommands: `server` (default), `status`, `version`, `help`
- `status` reports socket state, Songbird discovery, all capabilities

### Capability Domain Architecture
- New `ipc/capability_domains.rs`: 19 capabilities across 4 domains
- `capability_registry.toml`: machine-readable TOML manifest
- Runtime Songbird registration uses domain constants

### Tolerance Centralization
- `NMF_CONVERGENCE` (1e-6), `NMF_CONVERGENCE_LOOSE` (1e-4)
- `MATRIX_EPS` replaces inline 1e-12 in NMF epsilon guard
- `STABLE_SPECIAL_TINY` (1e-28) for stable special functions
- All inline tolerance literals in `expanded.rs` → `tolerances::`

### XDG Path Resolution
- `/tmp/` hardcodes → `$XDG_RUNTIME_DIR` with `temp_dir()` fallback
- NestGate socket discovery: removed hardcoded `/run/nestgate/default.sock`

### Python Baseline Provenance
- `python_anaerobic_biogas_baseline.py`: SHA-256 self-hash + git commit in metadata
- New `verify_baseline_outputs.sh`: automated integrity + numeric drift checker
- `validate_barracuda_cpu_v27.rs`: baseline source table for D65-D70

### Code Quality
- `QsType` struct: booleans → u8 bitflags (struct_excessive_bools resolved)
- Cast safety: all `#[expect(clippy::cast_*)]` annotated with rationale
- `doc_markdown` lint fixes across 5 tolerance modules
- metalForge CI coverage: 80% → 90%

### metalForge Coverage
- 6 new tests in `forge/src/inventory/output.rs`
- 6 new tests in `forge/src/data.rs`

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo check --workspace` | Clean |
| `cargo clippy --workspace -- -D warnings` | Zero warnings |
| `cargo test --workspace` | 1,662 pass, 0 fail |
| Experiments | 375 |
| Binaries | 354 (332 barracuda + 22 forge) |
| Validation checks | 5,707+ |
