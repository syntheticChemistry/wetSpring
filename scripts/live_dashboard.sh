#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Run the wetSpring dashboard in live streaming mode against petalTongue.
# Requires petalTongue to be running and accessible via socket.
#
# Usage:
#   ./scripts/live_dashboard.sh
#   PETALTONGUE_SOCKET=/path/to/socket ./scripts/live_dashboard.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

echo "Building wetspring_dashboard (release)..."
cargo build --manifest-path "$PROJECT_ROOT/barracuda/Cargo.toml" \
    --features json \
    --bin wetspring_dashboard \
    --release 2>&1

echo ""
echo "Launching live dashboard → petalTongue..."
cargo run --manifest-path "$PROJECT_ROOT/barracuda/Cargo.toml" \
    --features json \
    --bin wetspring_dashboard \
    --release

echo ""
echo "Dashboard session complete."
echo "Scenarios also saved to: $PROJECT_ROOT/sandbox/scenarios/"
