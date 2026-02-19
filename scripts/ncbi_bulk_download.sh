#!/usr/bin/env bash
# wetSpring — NCBI Bulk Download for GPU-Scale Profiling
#
# Usage:
#   ./scripts/ncbi_bulk_download.sh <BIOPROJECT> [--max-runs N] [--output-dir DIR]
#
# Downloads all SRA runs from a BioProject for BarraCUDA GPU profiling.
# Designed for scaling 16S pipeline validation to all Nannochloropsis/
# Microchloropsis 16S datasets in NCBI SRA.
#
# Examples:
#   # Download all runs from Nannochloropsis outdoor reactors
#   ./scripts/ncbi_bulk_download.sh PRJNA488170
#
#   # Download first 10 runs from a large BioProject
#   ./scripts/ncbi_bulk_download.sh PRJNA382322 --max-runs 10
#
#   # Download to custom directory
#   ./scripts/ncbi_bulk_download.sh PRJNA488170 --output-dir /mnt/fast/sra
#
# For GPU-scale profiling:
#   1. Download datasets with this script
#   2. Run: cargo run --release --features gpu --bin validate_diversity_gpu
#   3. Compare CPU vs GPU throughput on real NCBI data
#
# Prerequisites:
#   - SRA Toolkit (fasterq-dump, prefetch) OR curl (for ENA mirror)
#   - ~4 GB per million paired-end reads
#
# Strategy: GPU-accelerated Rust processes what papers took months, in minutes.
# Same pattern applies to OpenFold replication later.
set -euo pipefail

BIOPROJECT="${1:?Usage: $0 <BIOPROJECT> [--max-runs N] [--output-dir DIR]}"
MAX_RUNS=0  # 0 = all
OUTPUT_DIR=""

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-runs) MAX_RUNS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATA_DIR="${OUTPUT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/ncbi_bulk/$BIOPROJECT}"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — NCBI Bulk Download"
echo "  BioProject: $BIOPROJECT"
echo "  Max runs: ${MAX_RUNS:-all}"
echo "  Target: $DATA_DIR"
echo "═══════════════════════════════════════════════════════════"
echo

# ── Step 1: Fetch run accessions from NCBI ──────────────────────
RUN_LIST="$DATA_DIR/run_accessions.txt"

echo "── Fetching run accessions from NCBI Entrez... ──────────"
curl -fsSL "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&term=${BIOPROJECT}[BioProject]&retmax=10000&usehistory=y" \
    -o "$DATA_DIR/search_result.xml" 2>/dev/null

# Extract count
COUNT=$(grep -oP '<Count>\K[0-9]+' "$DATA_DIR/search_result.xml" | head -1)
echo "  Found $COUNT SRA entries for $BIOPROJECT"

if [[ "$COUNT" == "0" ]]; then
    echo "  [ERROR] No SRA entries found for $BIOPROJECT"
    exit 1
fi

# Fetch run accessions via RunInfo
echo "  Fetching run info..."
curl -fsSL "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&db=sra&rettype=runinfo&term=${BIOPROJECT}" \
    -o "$DATA_DIR/runinfo.csv" 2>/dev/null

# Extract run accessions (SRR* from CSV)
tail -n +2 "$DATA_DIR/runinfo.csv" | cut -d',' -f1 | grep -E '^[SED]RR' > "$RUN_LIST" 2>/dev/null || true

N_RUNS=$(wc -l < "$RUN_LIST")
echo "  Extracted $N_RUNS run accessions"

if [[ "$MAX_RUNS" -gt 0 && "$N_RUNS" -gt "$MAX_RUNS" ]]; then
    echo "  Limiting to first $MAX_RUNS runs (of $N_RUNS)"
    head -n "$MAX_RUNS" "$RUN_LIST" > "$DATA_DIR/run_accessions_limited.txt"
    mv "$DATA_DIR/run_accessions_limited.txt" "$RUN_LIST"
    N_RUNS=$MAX_RUNS
fi

echo

# ── Step 2: Download FASTQ files ────────────────────────────────
echo "── Downloading $N_RUNS runs... ──────────────────────────"

HAS_SRA_TOOLKIT=false
if command -v fasterq-dump &>/dev/null; then
    fasterq-dump --version &>/dev/null 2>&1 && HAS_SRA_TOOLKIT=true
fi

DOWNLOADED=0
FAILED=0

while IFS= read -r RUN; do
    RUN_DIR="$DATA_DIR/$RUN"

    if [ -d "$RUN_DIR" ] && ls "$RUN_DIR"/*.fastq* 1>/dev/null 2>&1; then
        echo "  [SKIP] $RUN — already downloaded"
        DOWNLOADED=$((DOWNLOADED + 1))
        continue
    fi

    mkdir -p "$RUN_DIR"

    if $HAS_SRA_TOOLKIT; then
        # Try SRA Toolkit first
        echo "  [SRA] $RUN..."
        if fasterq-dump "$RUN" --outdir "$RUN_DIR" --split-files --threads 4 2>/dev/null; then
            for f in "$RUN_DIR"/*.fastq; do
                [ -f "$f" ] && gzip "$f"
            done
            DOWNLOADED=$((DOWNLOADED + 1))
            continue
        fi
    fi

    # Fallback: ENA mirror (HTTP range download — first 20MB for validation)
    ACC_PREFIX="${RUN:0:6}"
    ACC_SUFFIX="${RUN: -1}"
    ENA_BASE="https://ftp.sra.ebi.ac.uk/vol1/fastq/${ACC_PREFIX}/00${ACC_SUFFIX}/${RUN}"

    echo "  [ENA] $RUN (20MB subsample)..."
    if curl -fsSL -r 0-20971520 "${ENA_BASE}/${RUN}_1.fastq.gz" -o "$RUN_DIR/${RUN}_1.fastq.gz" 2>/dev/null; then
        curl -fsSL -r 0-20971520 "${ENA_BASE}/${RUN}_2.fastq.gz" -o "$RUN_DIR/${RUN}_2.fastq.gz" 2>/dev/null || true
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "  [FAIL] $RUN — could not download from ENA"
        FAILED=$((FAILED + 1))
        rmdir "$RUN_DIR" 2>/dev/null || true
    fi

done < "$RUN_LIST"

echo

# ── Step 3: Generate manifest ───────────────────────────────────
MANIFEST="$DATA_DIR/MANIFEST.txt"
echo "# NCBI Bulk Download Manifest — $(date -Iseconds)" > "$MANIFEST"
echo "# BioProject: $BIOPROJECT" >> "$MANIFEST"
echo "# Runs: $DOWNLOADED downloaded, $FAILED failed" >> "$MANIFEST"
echo "#" >> "$MANIFEST"
find "$DATA_DIR" -name "*.fastq.gz" -exec sh -c \
    'echo "$(md5sum "$1" | cut -d" " -f1)  $(basename "$1")  $(stat -c%s "$1")"' _ {} \; \
    >> "$MANIFEST" 2>/dev/null || true

echo "═══════════════════════════════════════════════════════════"
echo "  Bulk download complete."
echo "  Downloaded: $DOWNLOADED / $N_RUNS runs"
echo "  Failed: $FAILED"
echo "  Directory: $DATA_DIR"
echo "  Manifest: $MANIFEST"
echo
echo "  Next: Run BarraCUDA GPU pipeline on these datasets:"
echo "    cargo run --release --features gpu --bin validate_diversity_gpu"
echo "═══════════════════════════════════════════════════════════"
