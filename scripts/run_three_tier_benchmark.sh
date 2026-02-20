#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# run_three_tier_benchmark.sh — Unified three-tier benchmark runner
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
BARRACUDA_DIR="$PROJECT_ROOT/barracuda"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%S")

mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  wetSpring Three-Tier Benchmark Runner                             ║"
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

# ── Tier 2: Rust CPU (pipeline benchmark) ──────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  TIER 2: Rust CPU (BarraCUDA pipeline benchmark)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

RUST_CPU_OK=0
if command -v cargo &>/dev/null; then
    (cd "$BARRACUDA_DIR" && cargo run --release --bin benchmark_pipeline 2>&1) && RUST_CPU_OK=1 || {
        echo "[WARN] Rust CPU pipeline benchmark failed"
        RUST_CPU_OK=0
    }
else
    echo "[SKIP] cargo not found"
fi

# ── Tier 3: Rust GPU (CPU vs GPU benchmark) ────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  TIER 3: Rust GPU (BarraCUDA CPU vs GPU benchmark)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

RUST_GPU_OK=0
if command -v cargo &>/dev/null; then
    if command -v nvidia-smi &>/dev/null; then
        (cd "$BARRACUDA_DIR" && cargo run --release --features gpu --bin benchmark_cpu_gpu 2>&1) && RUST_GPU_OK=1 || {
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
    echo "║  ✓ Tier 1 (Python)   — completed                                  ║"
else
    echo "║  ✗ Tier 1 (Python)   — skipped/failed                             ║"
fi

if [ "$RUST_CPU_OK" -eq 1 ]; then
    echo "║  ✓ Tier 2 (Rust CPU) — completed                                  ║"
else
    echo "║  ✗ Tier 2 (Rust CPU) — skipped/failed                             ║"
fi

if [ "$RUST_GPU_OK" -eq 1 ]; then
    echo "║  ✓ Tier 3 (Rust GPU) — completed                                  ║"
else
    echo "║  ✗ Tier 3 (Rust GPU) — skipped/failed                             ║"
fi

echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Results directory: benchmarks/results/                            ║"

# List generated JSON files
echo "║  JSON files:                                                       ║"
for f in "$RESULTS_DIR"/*.json; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    printf "║    %-62s ║\n" "$fname"
done

echo "╚══════════════════════════════════════════════════════════════════════╝"

# ── Merge into comparison table (if Python available) ──────────────────
if [ "$PYTHON_OK" -eq 1 ] || [ "$RUST_GPU_OK" -eq 1 ]; then
    echo ""
    echo "To generate a merged comparison table, see:"
    echo "  benchmarks/PROTOCOL.md"
    echo ""
    echo "Individual JSON results can be compared manually or via:"
    echo "  python3 -c \"import json, glob; [print(json.dumps(json.load(open(f)), indent=2)) for f in sorted(glob.glob('$RESULTS_DIR/*.json'))]\""
fi
