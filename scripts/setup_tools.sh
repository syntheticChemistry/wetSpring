#!/usr/bin/env bash
# wetSpring — Install required tools for data acquisition and analysis
#
# Run once after cloning the repo:
#   chmod +x scripts/setup_tools.sh && ./scripts/setup_tools.sh
#
# Prerequisites: Linux (Ubuntu/Pop!_OS), curl, tar, Docker
set -euo pipefail

TOOLS_DIR="${HOME}/.local/bin"
mkdir -p "$TOOLS_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — Tool Setup"
echo "═══════════════════════════════════════════════════════════"
echo

# ── 1. SRA Toolkit (for downloading NCBI sequencing data) ─────────
if command -v fasterq-dump &>/dev/null; then
    echo "  [OK] SRA Toolkit already installed: $(fasterq-dump --version 2>&1 | head -1)"
else
    echo "  [INSTALL] SRA Toolkit..."
    SRA_VERSION="3.1.1"
    SRA_URL="https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/${SRA_VERSION}/sratoolkit.${SRA_VERSION}-ubuntu64.tar.gz"
    TMPDIR=$(mktemp -d)
    curl -fsSL "$SRA_URL" -o "$TMPDIR/sratoolkit.tar.gz"
    tar -xzf "$TMPDIR/sratoolkit.tar.gz" -C "$TMPDIR"
    cp "$TMPDIR"/sratoolkit.*/bin/fasterq-dump "$TOOLS_DIR/"
    cp "$TMPDIR"/sratoolkit.*/bin/prefetch "$TOOLS_DIR/"
    cp "$TMPDIR"/sratoolkit.*/bin/vdb-validate "$TOOLS_DIR/"
    rm -rf "$TMPDIR"
    echo "  [OK] SRA Toolkit installed to $TOOLS_DIR"
fi

# ── 2. Docker check ───────────────────────────────────────────────
if command -v docker &>/dev/null; then
    echo "  [OK] Docker: $(docker --version)"
else
    echo "  [WARN] Docker not found. Install Docker to run Galaxy."
    echo "         https://docs.docker.com/engine/install/ubuntu/"
fi

if docker compose version &>/dev/null 2>&1; then
    echo "  [OK] Docker Compose: $(docker compose version --short)"
else
    echo "  [WARN] Docker Compose v2 not found."
fi

# ── 3. Rust check ────────────────────────────────────────────────
if command -v rustc &>/dev/null; then
    echo "  [OK] Rust: $(rustc --version)"
else
    echo "  [INFO] Rust not installed. Needed for Phase 2+."
    echo "         curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

echo
echo "  Tools directory: $TOOLS_DIR"
echo "  Make sure $TOOLS_DIR is in your PATH:"
echo "    export PATH=\"$TOOLS_DIR:\$PATH\""
echo
echo "  Next: ./scripts/download_data.sh"
echo "═══════════════════════════════════════════════════════════"
