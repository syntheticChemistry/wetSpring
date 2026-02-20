#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# wetSpring — Public Data Acquisition for Paper Queue
#
# Downloads open-access datasets for remaining experiments:
#
#   Tier 1 (immediate, no auth):
#     --sate        SATe-II benchmark alignments from Dryad (Exp019 Phase 4)
#     --phynetpy    PhyNetPy gene trees from GitHub (Exp019 Phase 2)
#     --phylohmm    PhyloNet-HMM empirical data from Rice (Exp019 Phase 3)
#     --epa-pfas    EPA UCMR 5 + PFOS surface water data (Exp041)
#
#   Tier 2 (NCBI SRA, uses API key):
#     --algae-ts    PRJNA382322 full: 128-sample algal pond time series (Exp039)
#     --bloom-ts    PRJNA1224988: 175-sample bloom time series (Exp040)
#     --massbank    MassBank PFAS spectra from GitHub (Exp042)
#
#   --tier1         All Tier 1 downloads
#   --tier2         All Tier 2 downloads
#   --all           Everything
#
# Prerequisites: curl, unzip. SRA Toolkit for Tier 2 NCBI downloads.
# NCBI API key loaded from ../testing-secrets/api-keys.toml if available.
#
# Usage:
#   ./scripts/download_public_data.sh --tier1
#   ./scripts/download_public_data.sh --all
#   ./scripts/download_public_data.sh --sate --epa-pfas
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
SECRETS_DIR="$(cd "$ROOT_DIR/../testing-secrets" 2>/dev/null && pwd)" || SECRETS_DIR=""

# Load NCBI API key if available
NCBI_KEY=""
if [ -n "$SECRETS_DIR" ] && [ -f "$SECRETS_DIR/api-keys.toml" ]; then
    NCBI_KEY=$(grep 'ncbi_api_key' "$SECRETS_DIR/api-keys.toml" 2>/dev/null | head -1 | sed 's/.*= *"\?\([^"]*\)"\?/\1/' || true)
    if [ -n "$NCBI_KEY" ]; then
        echo "[INFO] NCBI API key loaded (10 req/s)"
    fi
fi

ncbi_param() {
    if [ -n "$NCBI_KEY" ]; then
        echo "&api_key=$NCBI_KEY"
    fi
}

# Parse arguments
DO_SATE=false; DO_PHYNETPY=false; DO_PHYLOHMM=false; DO_EPA=false
DO_ALGAE=false; DO_BLOOM=false; DO_MASSBANK=false

for arg in "$@"; do
    case "$arg" in
        --sate)     DO_SATE=true ;;
        --phynetpy) DO_PHYNETPY=true ;;
        --phylohmm) DO_PHYLOHMM=true ;;
        --epa-pfas) DO_EPA=true ;;
        --algae-ts) DO_ALGAE=true ;;
        --bloom-ts) DO_BLOOM=true ;;
        --massbank) DO_MASSBANK=true ;;
        --tier1)    DO_SATE=true; DO_PHYNETPY=true; DO_PHYLOHMM=true; DO_EPA=true ;;
        --tier2)    DO_ALGAE=true; DO_BLOOM=true; DO_MASSBANK=true ;;
        --all)      DO_SATE=true; DO_PHYNETPY=true; DO_PHYLOHMM=true; DO_EPA=true
                    DO_ALGAE=true; DO_BLOOM=true; DO_MASSBANK=true ;;
        *)          echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! $DO_SATE && ! $DO_PHYNETPY && ! $DO_PHYLOHMM && ! $DO_EPA && \
   ! $DO_ALGAE && ! $DO_BLOOM && ! $DO_MASSBANK; then
    echo "Usage: $0 [--tier1 | --tier2 | --all | --sate | --phynetpy | ...]"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — Public Data Acquisition"
echo "  Target: $DATA_DIR"
echo "═══════════════════════════════════════════════════════════"
echo

# ══════════════════════════════════════════════════════════════════
#  Tier 1a: SATe-II Benchmark (Dryad DOI 10.5061/dryad.n9r3h)
# ══════════════════════════════════════════════════════════════════
if $DO_SATE; then
    echo "── Tier 1a: SATe-II Benchmark Alignments ─────────────────"
    SATE_DIR="$DATA_DIR/sate_benchmark"
    mkdir -p "$SATE_DIR"

    if [ -f "$SATE_DIR/COMPLETE" ]; then
        echo "  [SKIP] Already downloaded"
    else
        echo "  Source: Dryad DOI 10.5061/dryad.n9r3h"
        echo "  Content: 16S alignments + Newick trees (~355 MB)"

        # Dryad v2 API for dataset files
        DRYAD_BASE="https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.n9r3h"
        echo "  Fetching file list from Dryad API..."
        curl -fsSL "$DRYAD_BASE/download" -o "$SATE_DIR/sate_data.zip" 2>/dev/null && {
            echo "  Extracting..."
            cd "$SATE_DIR" && unzip -qo sate_data.zip 2>/dev/null || true
            touch COMPLETE
            echo "  [OK] SATe-II data → $SATE_DIR/"
        } || {
            echo "  [INFO] Dryad bulk download failed, trying individual files..."
            # Fallback: download README at minimum to confirm access
            curl -fsSL "https://datadryad.org/stash/dataset/doi:10.5061/dryad.n9r3h" \
                -o "$SATE_DIR/dryad_page.html" 2>/dev/null || true
            echo "  [PARTIAL] Check $SATE_DIR/dryad_page.html for manual download links"
        }
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 1b: PhyNetPy Gene Trees (GitHub NakhlehLab/PhyNetPy)
# ══════════════════════════════════════════════════════════════════
if $DO_PHYNETPY; then
    echo "── Tier 1b: PhyNetPy Gene Trees ──────────────────────────"
    PHYNETPY_DIR="$DATA_DIR/phynetpy_gene_trees"
    mkdir -p "$PHYNETPY_DIR"

    if [ -f "$PHYNETPY_DIR/COMPLETE" ]; then
        echo "  [SKIP] Already downloaded"
    else
        echo "  Source: GitHub NakhlehLab/PhyNetPy"
        echo "  Content: Gene trees in Newick format (DEFJ/ directory)"

        # Download specific directory via GitHub API
        curl -fsSL "https://api.github.com/repos/NakhlehLab/PhyNetPy/contents/src/PhyNetPy/data" \
            -o "$PHYNETPY_DIR/contents.json" 2>/dev/null && {
            echo "  Fetched directory listing"
        } || true

        # Clone just the data we need (shallow)
        if command -v git &>/dev/null; then
            echo "  Cloning PhyNetPy (shallow)..."
            git clone --depth 1 --filter=blob:none --sparse \
                https://github.com/NakhlehLab/PhyNetPy.git \
                "$PHYNETPY_DIR/repo" 2>/dev/null && {
                cd "$PHYNETPY_DIR/repo"
                git sparse-checkout set src/PhyNetPy/data 2>/dev/null || true
                cd "$ROOT_DIR"
                touch "$PHYNETPY_DIR/COMPLETE"
                echo "  [OK] PhyNetPy data → $PHYNETPY_DIR/repo/"
            } || {
                echo "  [INFO] Git sparse clone failed, trying tarball..."
                curl -fsSL "https://github.com/NakhlehLab/PhyNetPy/archive/refs/heads/main.tar.gz" \
                    -o "$PHYNETPY_DIR/phynetpy.tar.gz" 2>/dev/null
                tar xzf "$PHYNETPY_DIR/phynetpy.tar.gz" -C "$PHYNETPY_DIR" \
                    --wildcards "*/src/PhyNetPy/data/*" 2>/dev/null || true
                touch "$PHYNETPY_DIR/COMPLETE"
                echo "  [OK] PhyNetPy data (tarball) → $PHYNETPY_DIR/"
            }
        else
            echo "  [WARN] git not available, downloading tarball..."
            curl -fsSL "https://github.com/NakhlehLab/PhyNetPy/archive/refs/heads/main.tar.gz" \
                -o "$PHYNETPY_DIR/phynetpy.tar.gz" 2>/dev/null
            echo "  [PARTIAL] Extract manually: tar xzf phynetpy.tar.gz"
        fi
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 1c: PhyloNet-HMM Empirical Data (Rice University)
# ══════════════════════════════════════════════════════════════════
if $DO_PHYLOHMM; then
    echo "── Tier 1c: PhyloNet-HMM Empirical Data ──────────────────"
    HMM_DIR="$DATA_DIR/phylonet_hmm"
    mkdir -p "$HMM_DIR"

    if [ -f "$HMM_DIR/empirical-datasets/COMPLETE" ] || [ -d "$HMM_DIR/empirical-datasets" ]; then
        echo "  [SKIP] Empirical datasets already present"
    else
        echo "  Source: Rice University bioinfocs"
        echo "  Content: Mouse chr7 introgression data + simulated datasets"
        echo "  Paper: Liu 2014, DOI 10.1371/journal.pcbi.1003649"

        RICE_URL="https://bioinfocs.rice.edu/sites/g/files/bxs266/f/empirical-datasets.tar.bz2"
        echo "  Downloading empirical datasets..."
        curl -fsSL "$RICE_URL" -o "$HMM_DIR/empirical-datasets.tar.bz2" 2>/dev/null && {
            tar xjf "$HMM_DIR/empirical-datasets.tar.bz2" -C "$HMM_DIR" 2>/dev/null || true
            echo "  [OK] Empirical data → $HMM_DIR/empirical-datasets/"
        } || {
            echo "  [WARN] Rice server may have moved. Try supplementary from paper."
            echo "  DOI: 10.1371/journal.pcbi.1003649 (check Supporting Information)"
        }

        SIM_URL="https://bioinfocs.rice.edu/sites/g/files/bxs266/f/simulated-datasets.tar.bz2"
        echo "  Downloading simulated datasets..."
        curl -fsSL "$SIM_URL" -o "$HMM_DIR/simulated-datasets.tar.bz2" 2>/dev/null && {
            tar xjf "$HMM_DIR/simulated-datasets.tar.bz2" -C "$HMM_DIR" 2>/dev/null || true
            echo "  [OK] Simulated data → $HMM_DIR/simulated-datasets/"
        } || {
            echo "  [INFO] Simulated datasets download failed (non-critical)"
        }
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 1d: EPA PFAS Environmental Data
# ══════════════════════════════════════════════════════════════════
if $DO_EPA; then
    echo "── Tier 1d: EPA PFAS Environmental Data ──────────────────"
    EPA_DIR="$DATA_DIR/epa_pfas"
    mkdir -p "$EPA_DIR/ucmr5" "$EPA_DIR/pfos_surface_water"

    # UCMR 5 occurrence data (29 PFAS, nationwide)
    if [ -f "$EPA_DIR/ucmr5/COMPLETE" ]; then
        echo "  [SKIP] UCMR 5 already downloaded"
    else
        echo "  Downloading EPA UCMR 5 occurrence data..."
        UCMR_URL="https://www.epa.gov/system/files/other-files/2024-10/ucmr5-occurrence-data.zip"
        curl -fsSL "$UCMR_URL" -o "$EPA_DIR/ucmr5/ucmr5-occurrence-data.zip" 2>/dev/null && {
            cd "$EPA_DIR/ucmr5" && unzip -qo ucmr5-occurrence-data.zip 2>/dev/null || true
            cd "$ROOT_DIR"
            touch "$EPA_DIR/ucmr5/COMPLETE"
            echo "  [OK] UCMR 5 → $EPA_DIR/ucmr5/"
        } || {
            echo "  [WARN] UCMR 5 URL may have changed. Check:"
            echo "         https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule"
        }
    fi

    # PFOS surface water concentrations (data.gov)
    if [ -f "$EPA_DIR/pfos_surface_water/COMPLETE" ]; then
        echo "  [SKIP] PFOS surface water already downloaded"
    else
        echo "  Downloading EPA PFOS surface water data..."
        PFOS_URL="https://pasteur.epa.gov/uploads/10.23719/1522855/documents/TableS1_PFOS_Data.xlsx"
        curl -fsSL "$PFOS_URL" -o "$EPA_DIR/pfos_surface_water/pfos_data.xlsx" 2>/dev/null && {
            touch "$EPA_DIR/pfos_surface_water/COMPLETE"
            echo "  [OK] PFOS data → $EPA_DIR/pfos_surface_water/"
        } || {
            echo "  [WARN] PFOS URL may have changed. Check DOI: 10.23719/1522855"
        }
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 2a: PRJNA382322 Full (128-sample algal pond time series)
# ══════════════════════════════════════════════════════════════════
if $DO_ALGAE; then
    echo "── Tier 2a: PRJNA382322 Full Algal Pond Time Series ──────"
    echo "  128 samples, 4-month Nannochloropsis raceway 16S V1-V2"
    echo "  Proxy for: Cahill phage biocontrol (#13)"
    echo
    echo "  Running: scripts/ncbi_bulk_download.sh PRJNA382322"
    bash "$SCRIPT_DIR/ncbi_bulk_download.sh" PRJNA382322 --max-runs 10 \
        --output-dir "$DATA_DIR/prjna382322_full" || {
        echo "  [WARN] SRA download failed. Install SRA Toolkit first."
    }
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 2b: PRJNA1224988 (175-sample bloom time series)
# ══════════════════════════════════════════════════════════════════
if $DO_BLOOM; then
    echo "── Tier 2b: PRJNA1224988 Bloom Time Series ────────────────"
    echo "  175 samples, multi-year cyanobacterial bloom 16S"
    echo "  Proxy for: Smallwood raceway surveillance (#14)"
    echo
    echo "  Running: scripts/ncbi_bulk_download.sh PRJNA1224988"
    bash "$SCRIPT_DIR/ncbi_bulk_download.sh" PRJNA1224988 --max-runs 10 \
        --output-dir "$DATA_DIR/prjna1224988_bloom" || {
        echo "  [WARN] SRA download failed. Install SRA Toolkit first."
    }
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Tier 2c: MassBank PFAS Spectra (GitHub)
# ══════════════════════════════════════════════════════════════════
if $DO_MASSBANK; then
    echo "── Tier 2c: MassBank PFAS Reference Spectra ───────────────"
    MB_DIR="$DATA_DIR/massbank_pfas"
    mkdir -p "$MB_DIR"

    if [ -f "$MB_DIR/COMPLETE" ]; then
        echo "  [SKIP] MassBank data already downloaded"
    else
        echo "  Source: GitHub MassBank/MassBank-data"
        echo "  Content: Reference mass spectra (MSP format)"

        # Get latest release info
        RELEASE_URL=$(curl -fsSL "https://api.github.com/repos/MassBank/MassBank-data/releases/latest" 2>/dev/null \
            | grep '"tarball_url"' | head -1 | sed 's/.*: "\(.*\)".*/\1/' || true)

        if [ -n "$RELEASE_URL" ]; then
            echo "  Downloading latest MassBank release..."
            curl -fsSL -L "$RELEASE_URL" -o "$MB_DIR/massbank-latest.tar.gz" 2>/dev/null && {
                echo "  [OK] MassBank release → $MB_DIR/massbank-latest.tar.gz"
                echo "  [NOTE] Extract PFAS entries with: tar xzf massbank-latest.tar.gz"
                touch "$MB_DIR/COMPLETE"
            } || {
                echo "  [WARN] MassBank download failed"
            }
        else
            echo "  [WARN] Could not determine latest MassBank release"
        fi
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  Download complete. Summary:"
echo
for dir in sate_benchmark phynetpy_gene_trees phylonet_hmm epa_pfas \
           prjna382322_full prjna1224988_bloom massbank_pfas; do
    if [ -d "$DATA_DIR/$dir" ]; then
        SIZE=$(du -sh "$DATA_DIR/$dir" 2>/dev/null | cut -f1)
        COMPLETE=""
        [ -f "$DATA_DIR/$dir/COMPLETE" ] && COMPLETE=" ✓"
        echo "    $dir: $SIZE$COMPLETE"
    fi
done
echo
echo "  Next: Build experiments on downloaded data"
echo "═══════════════════════════════════════════════════════════"
