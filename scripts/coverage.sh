#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run workspace test coverage and generate HTML report.
# Requires: cargo install cargo-llvm-cov, rustup component add llvm-tools-preview
#
# Usage: ./scripts/coverage.sh
# Or from root: cargo llvm-cov --workspace --html

set -euo pipefail
cd "$(dirname "$0")/.."
cargo llvm-cov --workspace --html
