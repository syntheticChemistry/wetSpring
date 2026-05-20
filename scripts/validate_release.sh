#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Release-mode validation: runs core scenarios under `--release` with
# `--features guidestone,gpu` to catch FMA/LTO float drift vs debug builds
# (groundSpring V120 pattern). Uses the wetspring UniBin (V182+).
# Invoke from anywhere; uses the wetSpring workspace root as CWD.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

readonly -a SCENARIOS=(
  diversity
  fastq
  alignment
  hmm
  gillespie
  cooperation
  barracuda_cpu
)

readonly FEATURES="guidestone,gpu"
readonly BIN_ARGS="--release -p wetspring-barracuda --features $FEATURES --bin wetspring --"

declare -a RESULTS=()
declare -a ELAPSED=()

echo "Running ${#SCENARIOS[@]} release scenarios (wetspring UniBin, --features $FEATURES)..."
echo

failed=0
for scenario in "${SCENARIOS[@]}"; do
  printf -- '── %s ──\n' "$scenario"
  start=$(date +%s.%N)
  set +e
  cargo run $BIN_ARGS validate --scenario "$scenario"
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
printf '%-28s %8s %10s\n' "Scenario" "Status" "Time(s)"
echo "───────────────────────────────────────────────────────────"
for i in "${!SCENARIOS[@]}"; do
  printf '%-28s %8s %10s\n' "${SCENARIOS[$i]}" "${RESULTS[$i]}" "${ELAPSED[$i]}"
done
echo "═══════════════════════════════════════════════════════════"

if [[ "$failed" -ne 0 ]]; then
  echo "RESULT: FAIL (one or more scenarios did not exit 0)"
  exit 1
fi
echo "RESULT: PASS (all ${#SCENARIOS[@]} scenarios passed)"
exit 0
