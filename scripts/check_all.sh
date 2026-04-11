#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# check_all.sh — full CI-style audit for wetSpring.
#
# Runs: fmt → clippy → test → deny → coverage → Python baselines.
# Exit 0 only if every step passes.
#
# Usage:
#   ./scripts/check_all.sh              # full suite
#   ./scripts/check_all.sh --skip-cov   # skip coverage (faster)
#   ./scripts/check_all.sh --skip-py    # skip Python baselines

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SKIP_COV=false
SKIP_PY=false
for arg in "$@"; do
    case "$arg" in
        --skip-cov) SKIP_COV=true ;;
        --skip-py)  SKIP_PY=true ;;
    esac
done

PASS=0
FAIL=0
step() {
    local label="$1"; shift
    printf "\n══ %s ══\n" "$label"
    if "$@"; then
        printf "  ✓ %s\n" "$label"
        ((PASS++))
    else
        printf "  ✗ %s FAILED\n" "$label"
        ((FAIL++))
    fi
}

step "cargo fmt" cargo fmt --all -- --check
step "cargo clippy" cargo clippy --workspace --all-targets --all-features -- -D warnings
step "cargo test" cargo test --workspace --no-fail-fast
step "cargo deny" cargo deny check

if [ "$SKIP_COV" = false ]; then
    if command -v cargo-llvm-cov &>/dev/null; then
        step "coverage ≥90%" cargo coverage-check
    else
        printf "\n══ coverage ══\n  ○ cargo-llvm-cov not installed, skipping\n"
    fi
fi

if [ "$SKIP_PY" = false ]; then
    if [ -d scripts/baselines ] && command -v python3 &>/dev/null; then
        step "Python baselines" python3 scripts/baselines/run_all.py
    else
        printf "\n══ Python baselines ══\n  ○ baselines dir or python3 not found, skipping\n"
    fi
fi

printf "\n══════════════════════════════════════\n"
printf "  Results: %d passed, %d failed\n" "$PASS" "$FAIL"
printf "══════════════════════════════════════\n"

[ "$FAIL" -eq 0 ]
