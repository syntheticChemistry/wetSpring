#!/usr/bin/env python3
"""wetSpring — Priority-2 Data Download

Downloads priority-2 datasets:
  1. Algae 16S amplicon (PRJNA488170) — Nannochloropsis outdoor reactors
  2. Phytoplankton 16S (PRJNA1114688) — if accessible
  3. Jones Lab MS data — public mzML from MassBank/GNPS

Usage:
    python3 scripts/download_priority2.py [--dry-run] [--max-runs N]

These datasets are larger and require SRA Toolkit or ENA mirrors.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DRY_RUN = "--dry-run" in sys.argv
MAX_RUNS = 10

for i, arg in enumerate(sys.argv):
    if arg == "--max-runs" and i + 1 < len(sys.argv):
        MAX_RUNS = int(sys.argv[i + 1])

RATE_LIMIT = 0.4


def download_file(url: str, dest: Path, label: str, max_retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {label} ({dest.stat().st_size:,} bytes)")
        return True
    if DRY_RUN:
        print(f"  [DRY-RUN] {label}: {url}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [DOWNLOAD] {label} (attempt {attempt})...")
            req = urllib.request.Request(url, headers={"User-Agent": "wetSpring/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            mb = downloaded / (1024 * 1024)
                            print(f"\r    {mb:.1f} MB ({pct}%)", end="", flush=True)
                print()
                tmp.rename(dest)
            print(f"  [OK] {label} ({dest.stat().st_size:,} bytes)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  [WARN] Attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    print(f"  [FAIL] {label}")
    return False


def fetch_sra_runs(bioproject: str) -> list:
    """Fetch SRA run accessions for a BioProject via Entrez."""
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&db=sra&rettype=runinfo&term={bioproject}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wetSpring/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            csv = resp.read().decode("utf-8", errors="replace")
        runs = []
        for line in csv.strip().split("\n")[1:]:
            parts = line.split(",")
            if parts and parts[0].startswith(("SRR", "ERR", "DRR")):
                runs.append(parts[0])
        return runs
    except Exception as e:
        print(f"  [WARN] Could not fetch run info: {e}")
        return []


def download_sra_via_ena(accession: str, dest_dir: Path, subsample_mb: int = 50) -> bool:
    """Download FASTQ from ENA mirror (first N MB for validation)."""
    dest_r1 = dest_dir / f"{accession}_1.fastq.gz"
    dest_r2 = dest_dir / f"{accession}_2.fastq.gz"

    if dest_r1.exists() and dest_r1.stat().st_size > 0:
        print(f"  [SKIP] {accession}")
        return True

    if DRY_RUN:
        print(f"  [DRY-RUN] {accession}")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = accession[:6]
    suffix = accession[-1]
    ena_base = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/00{suffix}/{accession}"

    max_bytes = subsample_mb * 1024 * 1024
    for mate, dest in [("_1.fastq.gz", dest_r1), ("_2.fastq.gz", dest_r2)]:
        url = f"{ena_base}/{accession}{mate}"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "wetSpring/1.0",
                "Range": f"bytes=0-{max_bytes}",
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.rename(dest)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  [WARN] {accession}{mate}: {e}")
            if mate == "_1.fastq.gz":
                return False

    print(f"  [OK] {accession} ({dest_r1.stat().st_size:,} + {dest_r2.stat().st_size if dest_r2.exists() else 0:,} bytes)")
    return True


def section_algae_16s():
    """Download algae 16S amplicon data from PRJNA488170."""
    print("\n═══ Algae 16S (PRJNA488170) ═══")
    runs = fetch_sra_runs("PRJNA488170")
    if not runs:
        print("  [WARN] No runs found for PRJNA488170")
        return

    print(f"  Found {len(runs)} runs")
    if MAX_RUNS > 0 and len(runs) > MAX_RUNS:
        print(f"  Limiting to first {MAX_RUNS}")
        runs = runs[:MAX_RUNS]

    dest_dir = DATA_DIR / "algae_16s_PRJNA488170"
    downloaded = 0
    for i, run in enumerate(runs, 1):
        print(f"  [{i}/{len(runs)}] {run}")
        if download_sra_via_ena(run, dest_dir / run):
            downloaded += 1
        time.sleep(RATE_LIMIT)

    print(f"  Algae 16S: {downloaded}/{len(runs)} runs downloaded")


def section_phytoplankton():
    """Download phytoplankton 16S amplicon data."""
    print("\n═══ Phytoplankton 16S (PRJNA1114688) ═══")
    runs = fetch_sra_runs("PRJNA1114688")
    if not runs:
        print("  [WARN] No runs found (BioProject may be restricted)")
        print("  Using existing proxy data in data/exp002_phytoplankton/")
        return

    print(f"  Found {len(runs)} runs")
    if MAX_RUNS > 0 and len(runs) > MAX_RUNS:
        runs = runs[:MAX_RUNS]

    dest_dir = DATA_DIR / "phytoplankton_16s_PRJNA1114688"
    downloaded = 0
    for i, run in enumerate(runs, 1):
        print(f"  [{i}/{len(runs)}] {run}")
        if download_sra_via_ena(run, dest_dir / run):
            downloaded += 1
        time.sleep(RATE_LIMIT)

    print(f"  Phytoplankton: {downloaded}/{len(runs)} runs downloaded")


def section_massbank():
    """Download MassBank PFAS reference spectra (MSP format)."""
    print("\n═══ MassBank/GNPS Reference Spectra ═══")
    massbank_dir = DATA_DIR / "massbank"
    massbank_dir.mkdir(parents=True, exist_ok=True)

    download_file(
        "https://github.com/MassBank/MassBank-data/releases/latest/download/MassBank_NIST.msp",
        massbank_dir / "MassBank_NIST.msp",
        "MassBank NIST MSP (all spectra)"
    )


def main():
    print("═══════════════════════════════════════════════════════════")
    print("  wetSpring — Priority-2 Data Download")
    print(f"  Target: {DATA_DIR}")
    print(f"  Max runs per BioProject: {MAX_RUNS}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("═══════════════════════════════════════════════════════════")

    section_algae_16s()
    section_phytoplankton()
    section_massbank()

    print("\n═══════════════════════════════════════════════════════════")
    print("  Priority-2 download complete.")
    print("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
