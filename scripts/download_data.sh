#!/usr/bin/env bash
# wetSpring — Download all public datasets for pipeline replication
#
# Usage:
#   ./scripts/download_data.sh [--all | --validation | --algae | --phage]
#
# Datasets are downloaded to data/ with checksums for reproducibility.
# Total download: ~5-10 GB depending on selection.
#
# Prerequisites: SRA Toolkit (fasterq-dump, prefetch) — run setup_tools.sh first
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

MODE="${1:---all}"

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — Data Download"
echo "  Target: $DATA_DIR"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════════════════════"
echo

# ── Helper functions ──────────────────────────────────────────────

download_sra() {
    local accession="$1"
    local label="$2"
    local outdir="$DATA_DIR/$label"

    if [ -d "$outdir" ] && ls "$outdir"/*.fastq* 1>/dev/null 2>&1; then
        echo "  [SKIP] $label ($accession) — already downloaded"
        return
    fi

    echo "  [DOWNLOAD] $label ($accession)..."
    mkdir -p "$outdir"
    prefetch "$accession" --output-directory "$outdir/sra" 2>/dev/null || true
    fasterq-dump "$accession" --outdir "$outdir" --split-files --threads 4 2>/dev/null
    # Compress for storage
    for f in "$outdir"/*.fastq; do
        [ -f "$f" ] && gzip "$f" && echo "    compressed: $(basename "$f").gz"
    done
    # Clean up SRA cache
    rm -rf "$outdir/sra"
    echo "  [OK] $label → $outdir/"
}

download_url() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [ -f "$dest" ]; then
        echo "  [SKIP] $label — already downloaded"
        return
    fi

    echo "  [DOWNLOAD] $label..."
    mkdir -p "$(dirname "$dest")"
    curl -fsSL "$url" -o "$dest"
    echo "  [OK] $label → $dest"
}

# ══════════════════════════════════════════════════════════════════
#  DATASET 1: Galaxy Training — 16S Validation Baseline
# ══════════════════════════════════════════════════════════════════
#
# Source: Galaxy Training Network — mothur MiSeq SOP tutorial
# Data: Schloss lab mouse gut 16S V4 region, paired-end Illumina
# Purpose: Validate our Galaxy installation matches published tutorial output
# Reference: https://training.galaxyproject.org/training-material/topics/microbiome/tutorials/mothur-miseq-sop/tutorial.html

if [[ "$MODE" == "--all" || "$MODE" == "--validation" ]]; then
    echo "── Dataset 1: Galaxy Training (16S validation) ──────────"
    download_url \
        "https://zenodo.org/records/800651/files/MiSeq_SOP.tar.gz" \
        "$DATA_DIR/validation/MiSeq_SOP.tar.gz" \
        "Schloss MiSeq SOP (mouse gut 16S)"

    if [ -f "$DATA_DIR/validation/MiSeq_SOP.tar.gz" ] && [ ! -d "$DATA_DIR/validation/MiSeq_SOP" ]; then
        echo "  [EXTRACT] MiSeq_SOP.tar.gz..."
        tar -xzf "$DATA_DIR/validation/MiSeq_SOP.tar.gz" -C "$DATA_DIR/validation/" 2>/dev/null || true
        echo "  [OK] Extracted"
    fi
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  DATASET 2: Nannochloropsis Microbiome (Algae Pond Studies)
# ══════════════════════════════════════════════════════════════════
#
# BioProject: PRJNA382322
# Source: Wageningen University — bacterial community in outdoor
#         Nannochloropsis sp. pilot-scale photobioreactors
# Data: 16S rRNA amplicon, Illumina paired-end
# Purpose: Closest public analog to Smallwood's Pond Crash Forensics
# Paper: DOI 10.1007/s00253-022-11815-3

if [[ "$MODE" == "--all" || "$MODE" == "--algae" ]]; then
    echo "── Dataset 2: Nannochloropsis Microbiome (PRJNA382322) ──"

    # Main sample accessions from this BioProject
    # These are the 16S amplicon runs from outdoor Nannochloropsis reactors
    ALGAE_ACCESSIONS=(
        "SRR5534045"
    )

    for acc in "${ALGAE_ACCESSIONS[@]}"; do
        download_sra "$acc" "algae_microbiome/$acc"
    done
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  DATASET 3: Nannochloropsis salina Antibiotic Treatment Microbiome
# ══════════════════════════════════════════════════════════════════
#
# Source: Sandia/Texas A&M — "Changes in the Structure of the
#         Microbial Community Associated with N. salina"
# Paper: Frontiers in Microbiology 2016, DOI 10.3389/fmicb.2016.01155
# Data: 16S rRNA amplicon from N. salina cultures treated with
#       antibiotics, signaling compounds, glucose
# Purpose: Direct Sandia N. salina microbiome study with open data
# Note: Check paper supplementary for SRA accessions

if [[ "$MODE" == "--all" || "$MODE" == "--algae" ]]; then
    echo "── Dataset 3: N. salina Antibiotic Study ────────────────"
    echo "  [NOTE] Check paper DOI 10.3389/fmicb.2016.01155 for SRA accessions."
    echo "         Data availability section should list BioProject/SRA IDs."
    echo "         Add accessions to this script once identified."
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  DATASET 4: Phage Reference Genomes
# ══════════════════════════════════════════════════════════════════
#
# Source: NCBI Viral RefSeq — reference phage genomes
# Purpose: Reference database for phage annotation pipeline
# Also useful: INPHARED monthly phage genome database

if [[ "$MODE" == "--all" || "$MODE" == "--phage" ]]; then
    echo "── Dataset 4: Phage Reference Genomes ───────────────────"
    download_url \
        "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.1.genomic.fna.gz" \
        "$DATA_DIR/reference/viral_refseq.1.fna.gz" \
        "NCBI Viral RefSeq genomic (part 1)"
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  DATASET 5: SILVA 16S Reference Database
# ══════════════════════════════════════════════════════════════════
#
# Source: SILVA — curated 16S/18S rRNA reference alignment
# Purpose: Taxonomic classification of amplicon sequences
# Version: 138.1 (standard for QIIME2/mothur)

if [[ "$MODE" == "--all" || "$MODE" == "--validation" || "$MODE" == "--algae" ]]; then
    echo "── Dataset 5: SILVA 138.1 Reference Database ────────────"
    download_url \
        "https://data.qiime2.org/2024.5/common/silva-138-99-seqs.qza" \
        "$DATA_DIR/reference/silva-138-99-seqs.qza" \
        "SILVA 138.1 99% OTU sequences (QIIME2 format)"

    download_url \
        "https://data.qiime2.org/2024.5/common/silva-138-99-tax.qza" \
        "$DATA_DIR/reference/silva-138-99-tax.qza" \
        "SILVA 138.1 99% OTU taxonomy (QIIME2 format)"
    echo
fi

# ── Generate manifest ─────────────────────────────────────────────
echo "── Generating data manifest ─────────────────────────────"
MANIFEST="$DATA_DIR/MANIFEST.txt"
echo "# wetSpring Data Manifest — $(date -Iseconds)" > "$MANIFEST"
echo "# Generated by scripts/download_data.sh" >> "$MANIFEST"
echo "#" >> "$MANIFEST"
find "$DATA_DIR" -type f \( -name "*.fastq.gz" -o -name "*.fna.gz" -o -name "*.qza" -o -name "*.tar.gz" \) \
    -exec sh -c 'echo "$(md5sum "$1" | cut -d" " -f1)  $(basename "$1")  $(stat -c%s "$1")"' _ {} \; \
    >> "$MANIFEST" 2>/dev/null || true
echo "  [OK] Manifest: $MANIFEST"

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Download complete."
echo "  Data directory: $DATA_DIR"
echo "  Manifest: $MANIFEST"
echo
echo "  Next steps:"
echo "    1. Start Galaxy:  cd control/galaxy && docker compose up -d"
echo "    2. Upload data to Galaxy via web UI (localhost:8080)"
echo "    3. Run Experiment 001: Galaxy Bootstrap validation"
echo "═══════════════════════════════════════════════════════════"
