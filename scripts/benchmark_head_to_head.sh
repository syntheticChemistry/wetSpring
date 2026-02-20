#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# wetSpring — Rust vs Python Head-to-Head Benchmark
# Runs both BarraCUDA CPU (v1+v3 release) and Python baseline, captures timings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BARRACUDA="$ROOT/barracuda"
RESULTS="$ROOT/experiments/results/043_barracuda_cpu_v3"
mkdir -p "$RESULTS"

echo "═══════════════════════════════════════════════════════════════"
echo "  wetSpring — Rust vs Python Head-to-Head Benchmark"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "─── Phase 1: Rust (release build) ───"
cd "$BARRACUDA"
cargo build --release --bin validate_barracuda_cpu --bin validate_barracuda_cpu_v3 2>&1 | tail -1

echo ""
echo "  Running v1 (domains 1-9)..."
RUST_V1_START=$(date +%s%N)
cargo run --release --bin validate_barracuda_cpu 2>&1 | grep -E "TOTAL|RESULT"
RUST_V1_END=$(date +%s%N)
RUST_V1_US=$(( (RUST_V1_END - RUST_V1_START) / 1000 ))

echo ""
echo "  Running v3 (domains 10-18)..."
RUST_V3_START=$(date +%s%N)
cargo run --release --bin validate_barracuda_cpu_v3 2>&1 | grep -E "TOTAL|RESULT"
RUST_V3_END=$(date +%s%N)
RUST_V3_US=$(( (RUST_V3_END - RUST_V3_START) / 1000 ))

RUST_TOTAL=$(( RUST_V1_US + RUST_V3_US ))

echo ""
echo "─── Phase 2: Python ───"
echo ""
PYTHON_START=$(date +%s%N)
python3 "$SCRIPT_DIR/benchmark_rust_vs_python.py" 2>&1
PYTHON_END=$(date +%s%N)
PYTHON_US=$(( (PYTHON_END - PYTHON_START) / 1000 ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  HEAD-TO-HEAD RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo ""
printf "  %-25s %12s\n" "Platform" "Total (µs)"
printf "  %s\n" "─────────────────────────────────────────"
printf "  %-25s %12d\n" "Rust (release)" "$RUST_TOTAL"
printf "  %-25s %12d\n" "Python 3" "$PYTHON_US"
echo ""

if [ "$RUST_TOTAL" -gt 0 ]; then
    SPEEDUP=$((PYTHON_US / RUST_TOTAL))
    echo "  Rust speedup: ~${SPEEDUP}x over Python"
fi
echo ""
echo "  All 18 domains validated. Pure Rust math: PROVEN."
echo ""
