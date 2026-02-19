#!/usr/bin/env bash
# wetSpring — Download paper-specific open data for control validation
#
# Usage:
#   ./scripts/download_paper_data.sh [--all | --algae-16s | --voc]
#
# Datasets:
#   --algae-16s  PRJNA488170: Nannochloropsis sp. outdoor 16S amplicon (Wageningen)
#   --voc        Reese 2019 VOC baselines (already extracted from Table 1)
#   --all        Everything
#
# Data Availability Findings (2026-02-19):
#
#   Paper 1 (Carney 2016, Pond Crash Forensics):
#     Raw 16S reads NOT found in NCBI SRA. DOE/Sandia lab data, likely restricted.
#
#   Paper 2 (Humphrey 2023, Biotic Countermeasures):
#     16S outsourced to Zymo Research. NO NCBI deposit found despite thorough search.
#     OTU-level results available only in paper supplementary figures/tables.
#
#   Paper 3 (Reichardt 2020, Spectroradiometric Detection):
#     Hyperspectral reflectance instrument data. NOT in any public repository.
#     Behind ScienceDirect paywall. Accepted manuscript at OSTI but no data files.
#
#   Paper 4 (Reese 2019, VOC GC-MS):
#     Table 1 has 14 VOC compounds with masses, retention indices, NIST matches.
#     All data included in article body and supplementary.
#     Extracted to: experiments/results/013_voc_baselines/reese2019_table1.tsv
#
#   Best Proxy for Papers 1/2:
#     PRJNA488170 — Nannochloropsis sp. CCAP211/78 outdoor pilot reactors (Wageningen)
#     16S rRNA amplicon, Illumina, 7 Gbases. Published: Appl Microbiol Biotechnol 2022.
#     DOI: 10.1007/s00253-022-11815-3
#
# Prerequisites: SRA Toolkit (fasterq-dump, prefetch) — run setup_tools.sh first
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

MODE="${1:---all}"

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — Paper-Specific Data Download"
echo "  Target: $DATA_DIR"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════════════════════"
echo

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
    for f in "$outdir"/*.fastq; do
        [ -f "$f" ] && gzip "$f" && echo "    compressed: $(basename "$f").gz"
    done
    rm -rf "$outdir/sra"
    echo "  [OK] $label → $outdir/"
}

# ══════════════════════════════════════════════════════════════════
#  PRJNA488170: Nannochloropsis sp. Outdoor 16S Amplicon
# ══════════════════════════════════════════════════════════════════
#
# BioProject: PRJNA488170
# Source: Wageningen University — bacterial community in outdoor
#         Nannochloropsis sp. CCAP211/78 pilot-scale photobioreactors
# Data: 16S rRNA amplicon, Illumina paired-end, 7 Gbases
# Paper: DOI 10.1007/s00253-022-11815-3
# Purpose: Best open proxy for Carney 2016 / Humphrey 2023 16S data
#          Same genus (Nannochloropsis), same setting (outdoor cultivation),
#          same sequencing target (bacterial 16S), open access

if [[ "$MODE" == "--all" || "$MODE" == "--algae-16s" ]]; then
    echo "── PRJNA488170: Nannochloropsis Outdoor 16S ─────────────"

    # SRA Run: SRR7760408 — 11.9M spots, 7.2G bases, paired-end MiSeq
    # Library: AlgaeParc2015, primers 27F/338R (V1-V2), PCR amplicon
    download_sra "SRR7760408" "paper_proxy/nannochloropsis_16s/SRR7760408"

    echo "  [NOTE] After download, run Galaxy/QIIME2 pipeline on this data"
    echo "         to generate baseline for Exp012 (algae pond 16S validation)."
    echo
fi

# ══════════════════════════════════════════════════════════════════
#  PRJNA382322: Nannochloropsis sp. Additional Outdoor Pilots
# ══════════════════════════════════════════════════════════════════
#
# BioProject: PRJNA382322
# Source: Wageningen University — extended outdoor pilot study
# Data: Nannochloropsis sp., 8 Gbases, bacterial community
# Paper: DOI 10.1007/s00253-022-11815-3 (same publication)

if [[ "$MODE" == "--all" || "$MODE" == "--algae-16s" ]]; then
    echo "── PRJNA382322: Nannochloropsis Extended Pilots ──────────"

    download_sra "SRR5534045" "paper_proxy/nannochloropsis_pilots/SRR5534045"

    echo
fi

# ══════════════════════════════════════════════════════════════════
#  Reese 2019 VOC Baselines (already extracted)
# ══════════════════════════════════════════════════════════════════

if [[ "$MODE" == "--all" || "$MODE" == "--voc" ]]; then
    echo "── Reese 2019 VOC Baselines ─────────────────────────────"

    BASELINE="$(cd "$(dirname "$0")/.." && pwd)/experiments/results/013_voc_baselines/reese2019_table1.tsv"
    if [ -f "$BASELINE" ]; then
        echo "  [OK] VOC baseline already extracted: $BASELINE"
        COMPOUNDS=$(tail -n +16 "$BASELINE" | wc -l)
        echo "       $COMPOUNDS compounds from Table 1 (PMC6761164)"
    else
        echo "  [WARN] Baseline file not found: $BASELINE"
        echo "         Re-extract from PMC6761164 Table 1."
    fi
    echo
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Paper data download complete."
echo
echo "  Data Availability Summary:"
echo "    Paper 1 (Carney 2016):     NOT in NCBI SRA (DOE restricted)"
echo "    Paper 2 (Humphrey 2023):   NOT in NCBI SRA (Zymo outsourced)"
echo "    Paper 3 (Reichardt 2020):  NOT publicly available (instrument data)"
echo "    Paper 4 (Reese 2019):      Extracted from Table 1 (14 compounds)"
echo "    Proxy 16S:                 PRJNA488170 + PRJNA382322 (Nannochloropsis outdoor)"
echo
echo "  Next steps:"
echo "    1. Run Galaxy/QIIME2 on proxy 16S data → Exp012 baseline"
echo "    2. Build Rust validation: cargo run --bin validate_algae_16s"
echo "    3. Build Rust validation: cargo run --bin validate_voc_peaks"
echo "═══════════════════════════════════════════════════════════"
