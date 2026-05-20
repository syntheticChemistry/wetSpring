#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Run wetspring UniBin status/validation in live streaming mode against petalTongue.
# Requires petalTongue to be running and accessible via socket.
#
# Usage:
#   ./scripts/live_dashboard.sh
#   PETALTONGUE_SOCKET=/path/to/socket ./scripts/live_dashboard.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

readonly FEATURES="guidestone,gpu"

if [[ -z "${PETALTONGUE_SOCKET:-}" ]]; then
    XDG="${XDG_RUNTIME_DIR:-/tmp}"
    CANDIDATES=("$XDG/petaltongue/"*.sock /tmp/petaltongue*.sock)
    FOUND=""
    for c in "${CANDIDATES[@]}"; do
        if [[ -S "$c" ]]; then
            FOUND="$c"
            break
        fi
    done
    if [[ -z "$FOUND" ]]; then
        echo "ERROR: No petalTongue socket found."
        echo "  Start petalTongue first, or set PETALTONGUE_SOCKET=/path/to/socket"
        exit 1
    fi
    export PETALTONGUE_SOCKET="$FOUND"
    echo "Discovered petalTongue at: $PETALTONGUE_SOCKET"
fi

echo "Building wetspring UniBin (release)..."
cargo build --release -p wetspring-barracuda --features "$FEATURES" --bin wetspring 2>&1

echo ""
echo "Launching wetspring status + validate → petalTongue..."
cargo run --release -p wetspring-barracuda --features "$FEATURES" --bin wetspring -- status
echo ""
cargo run --release -p wetspring-barracuda --features "$FEATURES" --bin wetspring -- validate --format json

echo ""
echo "Dashboard session complete."
echo "Scenario results also available via: wetspring validate --list"
