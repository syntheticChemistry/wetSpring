#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build wetspring UniBin and dump scenario JSON to sandbox/scenarios/.
# If petalTongue is running, pushes via IPC automatically.
#
# Usage:
#   ./scripts/visualize.sh          # build + dump
#   ./scripts/visualize.sh --open   # also open petalTongue on scenarios dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

readonly FEATURES="guidestone,gpu"

echo "Building wetspring UniBin..."
cargo build --release -p wetspring-barracuda --features "$FEATURES" --bin wetspring 2>&1

echo ""
echo "Dumping scenario metadata..."
cargo run --release -p wetspring-barracuda --features "$FEATURES" --bin wetspring -- validate --list --format json > "$PROJECT_ROOT/sandbox/scenarios/scenario_list.json" 2>/dev/null || true

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
