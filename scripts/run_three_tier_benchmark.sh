#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# run_three_tier_benchmark.sh — Unified three-tier benchmark runner (V182+ UniBin)
#
# Executes Python baseline, Rust CPU, and Rust GPU benchmarks in sequence,
# collecting JSON results into benchmarks/results/ and printing a merged
# comparison table.
#
# Usage:
#   ./scripts/run_three_tier_benchmark.sh
#
# Prerequisites:
#   - Python 3 with numpy, scipy (pip install numpy scipy)
#   - Rust toolchain (cargo)
#   - nvidia-smi (optional, for GPU energy monitoring)
#   - RAPL access (optional, for CPU energy: chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj)
#
# See benchmarks/PROTOCOL.md for the full measurement specification.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%S")

readonly FEATURES="guidestone,gpu"
readonly BIN_ARGS="--release -p wetspring-barracuda --features $FEATURES --bin wetspring --"

mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  wetSpring Three-Tier Benchmark Runner (UniBin V182+)              ║"
echo "║  Timestamp: $TIMESTAMP                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

# ── Tier 1: Python baseline ────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  TIER 1: Python Baseline (numpy/scipy)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

PYTHON_OK=0
if command -v python3 &>/dev/null; then
    if python3 -c "import numpy; import scipy" 2>/dev/null; then
        python3 "$SCRIPT_DIR/benchmark_python_baseline.py" && PYTHON_OK=1 || {
            echo "[WARN] Python benchmark failed (non-zero exit)"
            PYTHON_OK=0
        }
    else
        echo "[SKIP] numpy/scipy not installed — run: pip install numpy scipy"
    fi
else
    echo "[SKIP] python3 not found"
fi

# ── Tier 2: Rust CPU (pipeline benchmark via UniBin) ──────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  TIER 2: Rust CPU (wetspring benchmark --scenario pipeline)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

RUST_CPU_OK=0
if command -v cargo &>/dev/null; then
    (cd "$PROJECT_ROOT" && cargo run $BIN_ARGS benchmark --scenario pipeline 2>&1) && RUST_CPU_OK=1 || {
        echo "[WARN] Rust CPU pipeline benchmark failed"
        RUST_CPU_OK=0
    }
else
    echo "[SKIP] cargo not found"
fi

# ── Tier 3: Rust GPU (CPU vs GPU benchmark via UniBin) ────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  TIER 3: Rust GPU (wetspring benchmark --scenario cpu_gpu)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

RUST_GPU_OK=0
if command -v cargo &>/dev/null; then
    if command -v nvidia-smi &>/dev/null; then
        (cd "$PROJECT_ROOT" && cargo run $BIN_ARGS benchmark --scenario cpu_gpu 2>&1) && RUST_GPU_OK=1 || {
            echo "[WARN] Rust GPU benchmark failed (GPU may not be available)"
            RUST_GPU_OK=0
        }
    else
        echo "[SKIP] nvidia-smi not found — GPU tier requires NVIDIA GPU"
    fi
else
    echo "[SKIP] cargo not found"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     THREE-TIER BENCHMARK SUMMARY                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"

if [ "$PYTHON_OK" -eq 1 ]; then
    echo "║  Tier 1 (Python)   — completed                                    ║"
else
    echo "║  Tier 1 (Python)   — skipped/failed                               ║"
fi

if [ "$RUST_CPU_OK" -eq 1 ]; then
    echo "║  Tier 2 (Rust CPU) — completed                                    ║"
else
    echo "║  Tier 2 (Rust CPU) — skipped/failed                               ║"
fi

if [ "$RUST_GPU_OK" -eq 1 ]; then
    echo "║  Tier 3 (Rust GPU) — completed                                    ║"
else
    echo "║  Tier 3 (Rust GPU) — skipped/failed                               ║"
fi

echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Results directory: benchmarks/results/                            ║"

for f in "$RESULTS_DIR"/*.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    printf "║    %-62s ║\n" "$fname"
done

echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ "$PYTHON_OK" -eq 1 ] || [ "$RUST_GPU_OK" -eq 1 ]; then
    echo ""
    echo "To generate a merged comparison table, see:"
    echo "  benchmarks/PROTOCOL.md"
fi
