#!/usr/bin/env python3
"""wetSpring — Priority-1 Data Download

Downloads priority-1 datasets for real-data validation:
  1. PFAS spectral library (Zenodo 14341321, ~50 MB)
  2. Vibrio assemblies (200 genomes from NCBI, ~5 GB)
  3. Campylobacterota assemblies (158 genomes from NCBI, ~3 GB)
  4. SILVA 138.1 raw FASTA (already present — verify only)

Usage:
    python3 scripts/download_priority1.py [--dry-run] [--skip-assemblies]

All downloads go to data/ with provenance tracking.
"""
import json
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NCBI_META = DATA_DIR / "ncbi_phase35"

RATE_LIMIT_DELAY = 0.35  # NCBI requests 3/sec without API key
DRY_RUN = "--dry-run" in sys.argv
SKIP_ASSEMBLIES = "--skip-assemblies" in sys.argv


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, label: str, max_retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {label} — already downloaded ({dest.stat().st_size:,} bytes)")
        return True

    if DRY_RUN:
        print(f"  [DRY-RUN] Would download: {label}")
        print(f"            URL: {url}")
        print(f"            Dest: {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [DOWNLOAD] {label} (attempt {attempt})...")
            req = urllib.request.Request(url, headers={"User-Agent": "wetSpring/1.0 (eastgate@msu.edu)"})
            with urllib.request.urlopen(req, timeout=120) as resp:
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
                            print(f"\r    {downloaded:,} / {total:,} bytes ({pct}%)", end="", flush=True)
                print()
                tmp.rename(dest)
            print(f"  [OK] {label} → {dest} ({dest.stat().st_size:,} bytes)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  [WARN] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    print(f"  [FAIL] {label} — all {max_retries} attempts failed")
    return False


def download_ncbi_assembly(accession: str, dest_dir: Path, label: str) -> bool:
    """Download a genome assembly FASTA from NCBI FTP."""
    dest = dest_dir / f"{accession}.fna.gz"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {label}")
        return True

    prefix = accession.replace("_", "").split(".")[0]
    p1, p2, p3 = prefix[3:6], prefix[6:9], prefix[9:12]
    ftp_dir = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{accession[:3]}/{p1}/{p2}/{p3}"

    if DRY_RUN:
        print(f"  [DRY-RUN] Would download assembly: {accession}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    listing_url = f"{ftp_dir}/"
    try:
        req = urllib.request.Request(listing_url, headers={"User-Agent": "wetSpring/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        import re
        dirs = re.findall(r'href="(' + re.escape(accession.split(".")[0]) + r'[^"]*)"', html)
        if not dirs:
            dirs = re.findall(r'href="(ASM[^"]*)"', html)
        if not dirs:
            print(f"  [WARN] No assembly directory found for {accession}")
            return False

        asm_dir = dirs[0].rstrip("/")
        fasta_url = f"{ftp_dir}/{asm_dir}/{asm_dir}_genomic.fna.gz"
        time.sleep(RATE_LIMIT_DELAY)
        return download_file(fasta_url, dest, label)

    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [WARN] Could not list FTP for {accession}: {e}")
        return False


def section_pfas():
    """Download PFAS spectral library from Zenodo."""
    print("\n═══ PFAS Spectral Library (Zenodo 14341321) ═══")
    dest_dir = DATA_DIR / "pfas_zenodo"
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = "https://zenodo.org/records/14341321/files/PFAS_spectral_library.xlsx?download=1"
    download_file(url, dest_dir / "PFAS_spectral_library.xlsx", "PFAS spectral library (xlsx)")

    url2 = "https://zenodo.org/records/14341321/files/PFAS_spectral_library.csv?download=1"
    download_file(url2, dest_dir / "PFAS_spectral_library.csv", "PFAS spectral library (csv)")

    url3 = "https://zenodo.org/records/14341321/files/PFAS_spectral_library.msp?download=1"
    download_file(url3, dest_dir / "PFAS_spectral_library.msp", "PFAS spectral library (msp)")


def section_vibrio():
    """Download Vibrio assemblies from NCBI."""
    print("\n═══ Vibrio Assemblies (200 genomes, Exp121) ═══")
    meta_file = NCBI_META / "vibrio_assemblies.json"
    if not meta_file.exists():
        print("  [ERROR] Metadata not found — run fetch_ncbi_phase35.py first")
        return

    with open(meta_file) as f:
        data = json.load(f)

    assemblies = data["assemblies"]
    dest_dir = DATA_DIR / "vibrio_assemblies"
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0
    for i, asm in enumerate(assemblies, 1):
        acc = asm["accession"]
        org = asm.get("organism", "unknown")
        label = f"[{i}/{len(assemblies)}] {acc} ({org[:40]})"
        if download_ncbi_assembly(acc, dest_dir, label):
            downloaded += 1
        else:
            failed += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Vibrio: {downloaded} downloaded, {failed} failed (of {len(assemblies)})")


def section_campylobacterota():
    """Download Campylobacterota assemblies from NCBI."""
    print("\n═══ Campylobacterota Assemblies (158 genomes, Exp125) ═══")
    meta_file = NCBI_META / "campylobacterota_assemblies.json"
    if not meta_file.exists():
        print("  [ERROR] Metadata not found — run fetch_ncbi_phase35.py first")
        return

    with open(meta_file) as f:
        data = json.load(f)

    assemblies = data["assemblies"]
    dest_dir = DATA_DIR / "campylobacterota_assemblies"
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0
    for i, asm in enumerate(assemblies, 1):
        acc = asm["accession"]
        org = asm.get("organism", "unknown")
        label = f"[{i}/{len(assemblies)}] {acc} ({org[:40]})"
        if download_ncbi_assembly(acc, dest_dir, label):
            downloaded += 1
        else:
            failed += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Campylobacterota: {downloaded} downloaded, {failed} failed (of {len(assemblies)})")


def section_silva():
    """Verify SILVA reference DB is present."""
    print("\n═══ SILVA 138.1 Reference Database ═══")
    raw_fasta = DATA_DIR / "reference_dbs" / "silva_138" / "silva_138_99_seqs.fasta"
    raw_tax = DATA_DIR / "reference_dbs" / "silva_138" / "silva_138_99_taxonomy.tsv"

    if raw_fasta.exists() and raw_tax.exists():
        print(f"  [OK] SILVA FASTA: {raw_fasta} ({raw_fasta.stat().st_size:,} bytes)")
        print(f"  [OK] SILVA taxonomy: {raw_tax} ({raw_tax.stat().st_size:,} bytes)")
    else:
        print("  [WARN] SILVA raw FASTA not found — downloading QIIME2 format")
        qza_dir = DATA_DIR / "reference"
        qza_dir.mkdir(parents=True, exist_ok=True)
        download_file(
            "https://data.qiime2.org/2024.5/common/silva-138-99-seqs.qza",
            qza_dir / "silva-138-99-seqs.qza",
            "SILVA 138.1 99% sequences"
        )
        download_file(
            "https://data.qiime2.org/2024.5/common/silva-138-99-tax.qza",
            qza_dir / "silva-138-99-tax.qza",
            "SILVA 138.1 99% taxonomy"
        )


def write_manifest():
    """Write provenance manifest for all priority-1 data."""
    print("\n═══ Generating Download Manifest ═══")
    manifest_path = DATA_DIR / "PRIORITY1_MANIFEST.md"

    lines = [
        "# Priority-1 Data Download Manifest",
        f"",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Script:** scripts/download_priority1.py",
        "",
        "## Datasets",
        "",
        "| Dataset | Location | Files | Total Size |",
        "|---------|----------|-------|------------|",
    ]

    for dirname, label in [
        ("pfas_zenodo", "PFAS Zenodo 14341321"),
        ("vibrio_assemblies", "Vibrio assemblies (200)"),
        ("campylobacterota_assemblies", "Campylobacterota (158)"),
        ("reference_dbs/silva_138", "SILVA 138.1 raw"),
    ]:
        d = DATA_DIR / dirname
        if d.exists():
            files = list(d.glob("*"))
            total = sum(f.stat().st_size for f in files if f.is_file())
            lines.append(f"| {label} | `data/{dirname}/` | {len(files)} | {total:,} bytes |")
        else:
            lines.append(f"| {label} | `data/{dirname}/` | — | not downloaded |")

    lines.extend(["", "## File Hashes (first 5 per dataset)", ""])

    for dirname in ["pfas_zenodo", "vibrio_assemblies", "campylobacterota_assemblies"]:
        d = DATA_DIR / dirname
        if d.exists():
            lines.append(f"### {dirname}")
            lines.append("")
            for f in sorted(d.glob("*"))[:5]:
                if f.is_file():
                    h = sha256_file(f)[:16]
                    lines.append(f"- `{f.name}`: `{h}...` ({f.stat().st_size:,} bytes)")
            lines.append("")

    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] Manifest: {manifest_path}")


def main():
    print("═══════════════════════════════════════════════════════════")
    print("  wetSpring — Priority-1 Data Download")
    print(f"  Target: {DATA_DIR}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    if SKIP_ASSEMBLIES:
        print("  Skipping genome assemblies (--skip-assemblies)")
    print("═══════════════════════════════════════════════════════════")

    section_pfas()

    if not SKIP_ASSEMBLIES:
        section_vibrio()
        section_campylobacterota()
    else:
        print("\n  [SKIP] Vibrio assemblies (--skip-assemblies)")
        print("  [SKIP] Campylobacterota assemblies (--skip-assemblies)")

    section_silva()
    write_manifest()

    print("\n═══════════════════════════════════════════════════════════")
    print("  Priority-1 download complete.")
    print("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
