#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build all wetSpring scenarios and dump to sandbox/scenarios/.
# If petalTongue is running, pushes via IPC automatically.
#
# Usage:
#   ./scripts/visualize.sh          # build + push/dump
#   ./scripts/visualize.sh --open   # also open petalTongue on scenarios dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building wetspring_dashboard..."
cargo build --manifest-path "$PROJECT_ROOT/barracuda/Cargo.toml" \
    --features json \
    --bin wetspring_dashboard \
    --release 2>&1

echo ""
echo "Running dashboard..."
cargo run --manifest-path "$PROJECT_ROOT/barracuda/Cargo.toml" \
    --features json \
    --bin wetspring_dashboard \
    --release

SCENARIO_DIR="$PROJECT_ROOT/sandbox/scenarios"
echo ""
echo "Scenarios written to: $SCENARIO_DIR"
echo "Total files: $(find "$SCENARIO_DIR" -name '*.json' 2>/dev/null | wc -l)"

if [[ "${1:-}" == "--open" ]]; then
    if command -v petaltongue >/dev/null 2>&1; then
        echo "Opening petalTongue..."
        petaltongue --scenario "$SCENARIO_DIR/" &
    else
        echo "petaltongue not found in PATH — install petalTongue or set PETALTONGUE_SOCKET"
    fi
fi
