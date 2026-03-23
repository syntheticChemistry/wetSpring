#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Release-mode validation: runs core `validate_*` binaries under `--release` with
# `--features gpu` to catch FMA/LTO float drift vs debug builds (groundSpring V120 pattern).
# Invoke from anywhere; uses the wetSpring workspace root as CWD.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

readonly -a VALIDATORS=(
  validate_diversity
  validate_fastq
  validate_alignment
  validate_hmm
  validate_gillespie
  validate_cooperation
  validate_barracuda_cpu
)

declare -a RESULTS=()
declare -a ELAPSED=()

echo "Running ${#VALIDATORS[@]} release validators (package wetspring-barracuda, --features gpu)..."
echo

failed=0
for bin in "${VALIDATORS[@]}"; do
  printf -- '── %s ──\n' "$bin"
  start=$(date +%s.%N)
  set +e
  cargo run --release -p wetspring-barracuda --bin "$bin" --features gpu
  code=$?
  set -e
  end=$(date +%s.%N)
  sec=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
  ELAPSED+=("$sec")
  if [[ "$code" -eq 0 ]]; then
    RESULTS+=("PASS")
  else
    RESULTS+=("FAIL")
    failed=1
  fi
  printf 'exit=%s elapsed=%ss\n\n' "$code" "$sec"
done

echo "═══════════════════════════════════════════════════════════"
printf '%-28s %8s %10s\n' "Binary" "Status" "Time(s)"
echo "───────────────────────────────────────────────────────────"
for i in "${!VALIDATORS[@]}"; do
  printf '%-28s %8s %10s\n' "${VALIDATORS[$i]}" "${RESULTS[$i]}" "${ELAPSED[$i]}"
done
echo "═══════════════════════════════════════════════════════════"

if [[ "$failed" -ne 0 ]]; then
  echo "RESULT: FAIL (one or more validators did not exit 0)"
  exit 1
fi
echo "RESULT: PASS (all ${#VALIDATORS[@]} validators passed)"
exit 0
