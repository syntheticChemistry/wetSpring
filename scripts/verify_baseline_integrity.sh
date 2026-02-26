#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Automated baseline drift verification.
#
# Compares current SHA-256 hashes of Python baseline scripts against the
# hashes recorded in BASELINE_MANIFEST.md. Any mismatch indicates a
# script was modified after the last baseline freeze and the corresponding
# Rust validation binary must be re-verified.
#
# Usage:
#   ./scripts/verify_baseline_integrity.sh
#
# Exit codes:
#   0 — all hashes match (no drift)
#   1 — one or more hashes differ (drift detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/BASELINE_MANIFEST.md"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: BASELINE_MANIFEST.md not found at $MANIFEST"
    exit 1
fi

drift=0
checked=0
missing=0

while IFS='|' read -r _ script _ _ _ hash _; do
    script="$(echo "$script" | xargs)"
    hash="$(echo "$hash" | xargs)"

    [[ -z "$script" || "$script" == "Script" || "$script" == "---"* ]] && continue
    [[ -z "$hash" || "$hash" == "SHA-256"* || "$hash" == "---"* ]] && continue

    script="${script#\`}"
    script="${script%\`}"
    hash="${hash#\`}"
    hash="${hash%\`}"

    script_path="$SCRIPT_DIR/$script"
    if [[ ! -f "$script_path" ]]; then
        echo "MISSING: $script (listed in manifest but not on disk)"
        missing=$((missing + 1))
        continue
    fi

    actual_hash="$(sha256sum "$script_path" | cut -c1-16)"

    if [[ "$actual_hash" != "$hash" ]]; then
        echo "DRIFT:   $script  expected=$hash  actual=$actual_hash"
        drift=$((drift + 1))
    fi

    checked=$((checked + 1))
done < "$MANIFEST"

echo ""
echo "Baseline integrity: $checked scripts checked, $drift drifted, $missing missing"

if [[ $drift -gt 0 || $missing -gt 0 ]]; then
    echo "ACTION: Re-verify corresponding Rust validation binaries."
    exit 1
fi

echo "All baseline hashes match. No drift detected."
exit 0
