# wetSpring Scripts

This directory holds automation around wetSpring: Python baselines (below) and
shell helpers for CI-style checks.

## `validate_release.sh`

Runs seven core `validate_*` binaries under `cargo run --release` with
`--features gpu` (package `wetspring-barracuda`) to detect release-mode
FMA/LTO float drift vs debug builds — same idea as groundSpring V120.

From the workspace root:

```bash
./scripts/validate_release.sh
```

The script records pass/fail and elapsed seconds per binary, prints a summary
table, and exits `0` only if every validator exits successfully.

## Coverage (Rust)

HTML coverage and an 80% line floor (ludoSpring V29 pattern) use
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) from the workspace
root. One-time setup:

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
```

Then:

| Command | Purpose |
|--------|---------|
| `cargo coverage` | Workspace HTML report (`target/llvm-cov/html`) |
| `cargo coverage-check` | Enforce **≥ 80%** line coverage (`--fail-under-lines 80`) |
| `cargo coverage-json` | JSON report at `coverage.json` |

---

# wetSpring Python Baseline Scripts

**Purpose:** Python/numpy/scipy/sklearn/vegan baselines for validation against
Rust CPU and GPU implementations.

**Status:** Frozen — 58 baseline scripts + utility/verification scripts. See
`BASELINE_MANIFEST.md` for per-script provenance and SHA-256 hashes.

These scripts establish the gold-standard reference output that wetSpring's
Rust experiments validate against. They are not actively developed — new
baselines are added only when new papers enter the queue.

## Convention

Each script:
1. Has SPDX-License-Identifier header
2. Has `# Commit:` line referencing the commit where its output was validated
3. Produces deterministic output (fixed seeds) for reproducibility
4. Is listed in `BASELINE_MANIFEST.md` with SHA-256 hash

## Running

```bash
# Most scripts require scipy, numpy, sklearn
pip install numpy scipy scikit-learn

# Track-specific:
python scripts/benchmark_python_baseline.py
```

## Relationship to Experiments

Each `scripts/*.py` baseline corresponds to one or more `barracuda/src/bin/validate_*.rs`
experiments. The Rust experiment proves parity with (or improvement over) the
Python baseline.
