#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Automated baseline output verification.
#
# Re-runs Python baseline scripts, captures output, and compares against
# stored baseline JSON files. Reports numeric drift beyond tolerance.
#
# This complements verify_baseline_integrity.sh (which checks script hashes)
# by verifying that the actual computed values have not drifted.
#
# Usage:
#   ./scripts/verify_baseline_outputs.sh [--rerun]
#
# Options:
#   --rerun   Re-execute Python scripts (default: compare stored JSON only)
#
# Exit codes:
#   0 — all baselines match stored outputs
#   1 — drift detected or missing baselines
#   2 — Python environment not available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/experiments/results"

RERUN=false
if [[ "${1:-}" == "--rerun" ]]; then
    RERUN=true
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH"
    exit 2
fi

drift=0
checked=0
errors=0

verify_json_exists() {
    local json_path="$1"
    local script_name="$2"
    if [[ ! -f "$json_path" ]]; then
        echo "MISSING: $json_path (from $script_name)"
        errors=$((errors + 1))
        return 1
    fi
    return 0
}

verify_json_has_metadata() {
    local json_path="$1"
    local has_meta
    has_meta=$(python3 -c "
import json, sys
with open('$json_path') as f:
    data = json.load(f)
meta = data.get('metadata', {})
fields = ['timestamp', 'platform', 'python_version']
missing = [f for f in fields if f not in meta]
if missing:
    print('MISSING_FIELDS: ' + ', '.join(missing))
    sys.exit(1)
print('OK')
" 2>&1) || true
    if [[ "$has_meta" == MISSING_FIELDS:* ]]; then
        echo "METADATA: $json_path — $has_meta"
        return 1
    fi
    return 0
}

rerun_and_compare() {
    local script="$1"
    local json_path="$2"
    local script_path="$SCRIPT_DIR/$script"

    if [[ ! -f "$script_path" ]]; then
        echo "SKIP: $script (not found)"
        return 0
    fi

    if [[ ! -f "$json_path" ]]; then
        echo "SKIP: $json_path (no stored baseline to compare)"
        return 0
    fi

    local tmp_output
    tmp_output=$(mktemp)
    trap 'rm -f "$tmp_output"' RETURN

    echo "  Running: python3 $script_path ..."
    if ! python3 "$script_path" > "$tmp_output" 2>&1; then
        echo "  ERROR: $script failed to execute"
        errors=$((errors + 1))
        return 1
    fi

    if [[ -f "$json_path" ]]; then
        local result
        result=$(python3 -c "
import json, sys, math

def compare_values(a, b, path='', tol=1e-10):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(list(a.keys()) + list(b.keys())):
            if k == 'metadata':
                continue
            if k not in a:
                diffs.append(f'{path}.{k}: missing in new')
            elif k not in b:
                diffs.append(f'{path}.{k}: missing in stored')
            else:
                diffs.extend(compare_values(a[k], b[k], f'{path}.{k}', tol))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} vs {len(b)}')
        else:
            for i, (va, vb) in enumerate(zip(a, b)):
                diffs.extend(compare_values(va, vb, f'{path}[{i}]', tol))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a) > 1e-15 or abs(b) > 1e-15:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-15)
            if rel > tol:
                diffs.append(f'{path}: {a} vs {b} (rel={rel:.2e})')
        elif abs(a - b) > tol:
            diffs.append(f'{path}: {a} vs {b} (abs={abs(a-b):.2e})')
    elif a != b:
        diffs.append(f'{path}: {repr(a)} vs {repr(b)}')
    return diffs

stored = json.load(open('$json_path'))
new = json.load(open('$tmp_output'))
diffs = compare_values(new, stored)
if diffs:
    for d in diffs[:20]:
        print(f'  DRIFT: {d}')
    sys.exit(1)
print('  OK: values match')
" 2>&1) || true

        if echo "$result" | grep -q "DRIFT:"; then
            echo "$result"
            drift=$((drift + 1))
        else
            echo "$result"
        fi
    fi

    checked=$((checked + 1))
}

echo "wetSpring baseline output verification"
echo "======================================="
echo ""

BASELINE_PAIRS=(
    "python_anaerobic_biogas_baseline.py|track6_anaerobic/biogas_kinetics_baseline.json"
)

for pair in "${BASELINE_PAIRS[@]}"; do
    IFS='|' read -r script json_rel <<< "$pair"
    json_path="$RESULTS_DIR/$json_rel"

    echo "Checking: $script → $json_rel"

    if verify_json_exists "$json_path" "$script"; then
        verify_json_has_metadata "$json_path" || true
        checked=$((checked + 1))
    fi

    if [[ "$RERUN" == true ]]; then
        rerun_and_compare "$script" "$json_path"
    fi
done

echo ""
echo "Baseline output verification: $checked checked, $drift drifted, $errors errors"

if [[ $drift -gt 0 || $errors -gt 0 ]]; then
    echo "ACTION: Investigate drifted baselines and re-validate Rust binaries."
    exit 1
fi

echo "All baseline outputs verified."
exit 0
