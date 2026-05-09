# wetSpring V151 — Deep Debt Evolution Handoff

**Date**: April 27, 2026
**From**: wetSpring V151
**For**: primalSpring team, ecosystem audit

## Summary

Systematic technical debt elimination targeting interstadial standards
compliance. Zero `dyn` dispatch in application code (one justified
exception). All library output routed through `Write` trait. Hardcoded
paths replaced with env-var discovery. Shared validation helpers
extracted. Concrete error types in binaries.

## Changes

### Phase 1: `dyn` Dispatch Elimination (7 sites → 1 justified)

- `io/fastq/mod.rs`: `Box<dyn BufRead>` → concrete `FastqReader` enum
  with `BufRead`/`Read` impls. `read_byte_line` uses `impl BufRead`.
- `io/ms2/parser.rs`: `Box<dyn BufRead>` → `BufReader<File>`.
- `io/nanopore/nrs.rs`: `&mut dyn Write` closure → generic `write_ctx`.
- `bio/gillespie.rs`: `Box<dyn Fn>` retained with audit comment
  (heterogeneous `Vec<Reaction>` requires type erasure per SSA design).
- `bin/wetspring_science_facade.rs`: `Box<dyn Error>` → concrete
  `FacadeError` enum + `ExitCode` pattern.

### Phase 2: `println!` → `Write`-Based Output

All `println!`/`eprintln!` in validation library code replaced with
`writeln!(stdout().lock(), ...)` or `writeln!(stderr().lock(), ...)`.
Affected: `validation/{mod,sink,harness,domain,timing,or_exit}.rs`.

### Phase 3: Hardcoded Paths

- `bio/dorado.rs`: Added `$WETSPRING_DORADO_SEARCH_DIRS` env-var
  (colon-separated). Default install dirs remain as fallback.
- `bin/dump_wetspring_scenarios.rs`: `/tmp/petaltongue.sock` →
  `$XDG_RUNTIME_DIR/petaltongue.sock` in doc comment.

### Phase 4: Shared Validation Helpers

New in `validation/timing.rs`:
- `BenchRow { label, origin, ms }` — replaces per-binary `Timing` structs.
- `bench_print(label, f)` — `bench` + stdout reporting.
- `print_bench_table(rows)` — box-drawing three-column timing table.

New in `validation/mod.rs`:
- `gpu_or_skip_sync()` — sync GPU bootstrap via tokio `Runtime::new()`.

Refactored: `benchmark_cross_spring_s65.rs`, `benchmark_cross_spring_s68.rs`
(both use shared helpers). `validate_cross_spring_s57.rs` and
`validate_cpu_vs_gpu_all_domains.rs` use `gpu_or_skip()` async.

### Phase 5: Tolerance Centralization

Remaining `1e-10` literals in `wetspring_guidestone.rs` replaced with
`tolerances::ANALYTICAL_LOOSE`. Codebase was already well-centralized —
audit confirmed all validation binaries use `tolerances::` constants.

## Verification

- `cargo check`: exit 0, only pre-existing warnings (3 unused imports)
- `cargo test --lib`: 1209 pass, 0 fail, 1 ignored
- BLAKE3 checksums regenerated for modified files

## Patterns for Ecosystem

1. **`FastqReader` enum pattern**: Preferred over `Box<dyn BufRead>` for
   branching on file format at open time. Zero overhead, zero allocation.
2. **`bench_print` + `BenchRow`**: Shared timing row + table printer
   eliminates per-binary timing boilerplate.
3. **`gpu_or_skip_sync()`**: One-line GPU bootstrap for sync binaries.
4. **`FacadeError` pattern**: Concrete error enum + `ExitCode` main.
