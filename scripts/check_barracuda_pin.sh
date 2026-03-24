#!/usr/bin/env bash
# Verify that the barraCuda dependency version matches the wetSpring pin.
# Run in CI to catch upstream version drift early.
set -euo pipefail

PINNED="0.3.7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BC_TOML="$SCRIPT_DIR/../../barraCuda/Cargo.toml"

if [[ ! -f "$BC_TOML" ]]; then
    echo "ERROR: barraCuda Cargo.toml not found at $BC_TOML"
    exit 1
fi

# Extract version from [workspace.package] or [package] section
ACTUAL=$(grep '^version' "$BC_TOML" | head -1 | sed 's/.*"\(.*\)".*/\1/')

if [[ -z "$ACTUAL" ]]; then
    echo "ERROR: could not extract barraCuda version from $BC_TOML"
    exit 1
fi

echo "barraCuda pinned: $PINNED"
echo "barraCuda actual: $ACTUAL"

# Compare major.minor.patch
IFS='.' read -r P_MAJ P_MIN P_PAT <<< "$PINNED"
IFS='.' read -r A_MAJ A_MIN A_PAT <<< "${ACTUAL%%-*}"

if (( A_MAJ > P_MAJ )) || \
   (( A_MAJ == P_MAJ && A_MIN > P_MIN )) || \
   (( A_MAJ == P_MAJ && A_MIN == P_MIN && A_PAT >= P_PAT )); then
    echo "OK: barraCuda $ACTUAL >= pin $PINNED"
else
    echo "FAIL: barraCuda $ACTUAL is below pin $PINNED"
    exit 1
fi
