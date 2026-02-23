#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-23
"""Fetch NCBI data for Phase 35 experiments (Exp121-126).

Queries NCBI Entrez for:
  - Vibrio genome assemblies (Exp121: QS parameter landscape)
  - Campylobacterota assemblies (Exp125: cross-ecosystem pangenome)
  - 16S amplicon BioProjects from diverse biomes (Exp126: QS atlas)

Data saved as JSON in data/ncbi_phase35/ for consumption by Rust binaries.
Requires: urllib (stdlib only, no pip deps).

Usage:
    python3 scripts/fetch_ncbi_phase35.py [--api-key KEY] [--dry-run]
"""

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ncbi_phase35"


def esearch(db, term, retmax=200, api_key=""):
    params = {
        "db": db, "term": term, "retmax": retmax,
        "retmode": "json", "usehistory": "y",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def esummary(db, ids, api_key=""):
    if not ids:
        return []
    params = {"db": db, "id": ",".join(str(i) for i in ids[:200]), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS_BASE}/esummary.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    result = data.get("result", {})
    uids = result.get("uids", [])
    return [result[uid] for uid in uids if uid in result]


def load_api_key():
    if len(sys.argv) > 2 and sys.argv[1] == "--api-key":
        return sys.argv[2]
    secrets = Path(__file__).resolve().parent.parent.parent / "testing-secrets" / "api-keys.toml"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            line = line.strip()
            if line.startswith("ncbi_api_key") and "=" in line:
                val = line.split("=", 1)[1].strip().strip('"')
                if val:
                    return val
    return ""


def fetch_datasets_report(taxon, page_size=200, api_key=""):
    """Use NCBI Datasets v2 API for assembly metadata with real genome stats."""
    url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{urllib.parse.quote(taxon)}"
        f"/dataset_report?page_size={page_size}"
        f"&filters.assembly_level=complete_genome&filters.assembly_level=chromosome"
        f"&filters.assembly_source=refseq"
    )
    if api_key:
        url += f"&api-key={api_key}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def fetch_vibrio_assemblies(api_key, delay):
    """Exp121: Vibrio genome assemblies with QS-relevant metadata."""
    print("\n── Exp121: Vibrio Assemblies ────────────────────────────")

    results = []
    try:
        data = fetch_datasets_report("Vibrio", page_size=200, api_key=api_key)
        reports = data.get("reports", [])
        print(f"  NCBI Datasets API returned {len(reports)} reports")

        for r in reports:
            org = r.get("organism", {})
            asm = r.get("assembly_info", {})
            stats = r.get("assembly_stats", {})
            annot = r.get("annotation_info", {})
            gene_counts = annot.get("stats", {}).get("gene_counts", {})
            biosample = asm.get("biosample", {})
            attrs = biosample.get("attributes", []) if isinstance(biosample, dict) else []
            iso_src = ""
            for attr in attrs:
                if isinstance(attr, dict) and attr.get("name", "").lower() in ("isolation_source", "isolation source"):
                    iso_src = attr.get("value", "")
                    break

            acc = r.get("accession", asm.get("assembly_accession", ""))
            entry = {
                "uid": acc,
                "accession": acc,
                "organism": org.get("organism_name", ""),
                "assembly_name": asm.get("assembly_name", ""),
                "assembly_level": asm.get("assembly_level", ""),
                "genome_size_bp": int(stats.get("total_sequence_length", 0) or 0),
                "scaffold_count": int(stats.get("number_of_scaffolds", stats.get("number_of_contigs", 0)) or 0),
                "gene_count": int(gene_counts.get("total", gene_counts.get("protein_coding", 0)) or 0),
                "biosample": biosample.get("accession", "") if isinstance(biosample, dict) else "",
                "isolation_source": iso_src,
                "submitter": asm.get("submitter", ""),
            }
            results.append(entry)
    except Exception as e:
        print(f"  Datasets API error: {e}")
        print("  Falling back to Entrez esummary...")
        term = 'Vibrio[Organism] AND "complete genome"[Assembly Level] AND "latest refseq"[Filter]'
        search = esearch("assembly", term, retmax=200, api_key=api_key)
        ids = search.get("esearchresult", {}).get("idlist", [])
        if ids:
            time.sleep(delay)
            summaries = esummary("assembly", ids, api_key=api_key)
            for s in summaries:
                entry = {
                    "uid": s.get("uid", ""),
                    "accession": s.get("assemblyaccession", s.get("accession", "")),
                    "organism": s.get("organism", s.get("speciesname", "")),
                    "assembly_name": s.get("assemblyname", ""),
                    "assembly_level": s.get("assemblylevel", ""),
                    "genome_size_bp": 0,
                    "scaffold_count": 0,
                    "gene_count": 0,
                    "biosample": "",
                    "isolation_source": s.get("infraspecificname", ""),
                    "submitter": "",
                }
                results.append(entry)

    time.sleep(delay)
    print(f"  Extracted: {len(results)} assembly records")
    has_size = sum(1 for r in results if r["genome_size_bp"] > 0)
    has_genes = sum(1 for r in results if r["gene_count"] > 0)
    print(f"  With genome size: {has_size}, with gene count: {has_genes}")
    return results


def fetch_campylobacterota_assemblies(api_key, delay):
    """Exp125: Campylobacterota assemblies from diverse ecosystems."""
    print("\n── Exp125: Campylobacterota Assemblies ─────────────────")
    genera = ["Campylobacter", "Helicobacter", "Sulfurospirillum", "Arcobacter", "Sulfurimonas", "Nautilia"]

    results = []
    for label in genera:
        print(f"  Searching: {label}")
        try:
            data = fetch_datasets_report(label, page_size=50, api_key=api_key)
            reports = data.get("reports", [])
            print(f"    Found: {len(reports)} assemblies")
            for r in reports:
                org = r.get("organism", {})
                asm = r.get("assembly_info", {})
                stats = r.get("assembly_stats", {})
                annot = r.get("annotation_info", {})
                gene_counts = annot.get("stats", {}).get("gene_counts", {})
                biosample = asm.get("biosample", {})
                attrs = biosample.get("attributes", []) if isinstance(biosample, dict) else []
                iso_src = ""
                for attr in attrs:
                    if isinstance(attr, dict) and attr.get("name", "").lower() in ("isolation_source", "isolation source", "host"):
                        iso_src = attr.get("value", "")
                        break
                # Infer ecosystem from genus when isolation_source is empty
                if not iso_src:
                    if label in ("Sulfurimonas", "Nautilia"):
                        iso_src = "vent"
                    elif label == "Sulfurospirillum":
                        iso_src = "sediment"
                    elif label == "Arcobacter":
                        iso_src = "water"
                    elif label in ("Campylobacter", "Helicobacter"):
                        iso_src = "gut"

                acc = r.get("accession", asm.get("assembly_accession", ""))
                entry = {
                    "uid": acc,
                    "accession": acc,
                    "organism": org.get("organism_name", ""),
                    "genus": label,
                    "assembly_level": asm.get("assembly_level", ""),
                    "genome_size_bp": int(stats.get("total_sequence_length", 0) or 0),
                    "gene_count": int(gene_counts.get("total", gene_counts.get("protein_coding", 0)) or 0),
                    "isolation_source": iso_src,
                }
                results.append(entry)
            time.sleep(delay)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)

    print(f"  Total: {len(results)} Campylobacterota assemblies")
    has_size = sum(1 for r in results if r["genome_size_bp"] > 0)
    print(f"  With genome size: {has_size}")
    return results


def fetch_biome_16s_projects(api_key, delay):
    """Exp126: 16S amplicon BioProjects from diverse biomes for QS atlas."""
    print("\n── Exp126: 16S BioProjects (Diverse Biomes) ────────────")
    queries = [
        ('"human gut"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "gut"),
        ('"oral microbiome"[All Fields] AND "16S"[All Fields]', "oral"),
        ('"soil microbiome"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "soil"),
        ('"marine sediment"[All Fields] AND "16S"[All Fields]', "marine_sediment"),
        ('"hydrothermal vent"[All Fields] AND "16S"[All Fields]', "vent"),
        ('"rhizosphere"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "rhizosphere"),
        ('"freshwater lake"[All Fields] AND "16S"[All Fields]', "freshwater"),
        ('"wastewater"[All Fields] AND "activated sludge"[All Fields] AND "16S"[All Fields]', "wastewater"),
        ('"coral reef"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "coral"),
        ('"deep sea"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "deep_sea"),
        ('"permafrost"[All Fields] AND "16S"[All Fields]', "permafrost"),
        ('"biofilm"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "biofilm"),
        ('"hot spring"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "hot_spring"),
        ('"algal bloom"[All Fields] AND "16S"[All Fields] AND "amplicon"[All Fields]', "algal_bloom"),
    ]

    results = []
    for term, biome in queries:
        print(f"  Searching biome: {biome}")
        try:
            search = esearch("bioproject", term, retmax=10, api_key=api_key)
            ids = search.get("esearchresult", {}).get("idlist", [])
            count = int(search.get("esearchresult", {}).get("count", 0))
            print(f"    Found: {count} total, fetching {len(ids)}")
            if ids:
                time.sleep(delay)
                summaries = esummary("bioproject", ids, api_key=api_key)
                for s in summaries:
                    entry = {
                        "uid": s.get("uid", ""),
                        "accession": s.get("project_acc", ""),
                        "title": s.get("project_title", "")[:200],
                        "description": s.get("project_description", "")[:300],
                        "organism": s.get("organism_name", ""),
                        "biome": biome,
                        "data_type": s.get("project_data_type", ""),
                        "sample_count": int(s.get("registration", {}).get("sample_count", 0)
                                            if isinstance(s.get("registration"), dict) else 0),
                    }
                    results.append(entry)
                time.sleep(delay)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)

    print(f"  Total: {len(results)} BioProjects across {len(set(r['biome'] for r in results))} biomes")
    return results


def main():
    api_key = load_api_key()
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  NCBI Phase 35 Data Fetch — wetSpring Exp121-126")
    print("=" * 60)
    if api_key:
        print(f"  API key: {api_key[:8]}... (10 req/s)")
    else:
        print("  No API key (3 req/s limit)")
    if dry_run:
        print("  DRY RUN — no network requests")

    delay = 0.12 if api_key else 0.35
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("\n  Dry run complete. Would fetch:")
        print("    - Vibrio assemblies → vibrio_assemblies.json")
        print("    - Campylobacterota assemblies → campylobacterota_assemblies.json")
        print("    - 16S BioProjects → biome_16s_projects.json")
        return

    # Fetch all datasets
    vibrio = fetch_vibrio_assemblies(api_key, delay)
    campy = fetch_campylobacterota_assemblies(api_key, delay)
    biome = fetch_biome_16s_projects(api_key, delay)

    # Save
    manifest = {"fetch_date": time.strftime("%Y-%m-%d"), "api_key_used": bool(api_key)}

    vibrio_out = DATA_DIR / "vibrio_assemblies.json"
    with open(vibrio_out, "w") as f:
        json.dump({"metadata": manifest, "assemblies": vibrio}, f, indent=2)
    print(f"\n  Saved: {vibrio_out} ({len(vibrio)} records)")

    campy_out = DATA_DIR / "campylobacterota_assemblies.json"
    with open(campy_out, "w") as f:
        json.dump({"metadata": manifest, "assemblies": campy}, f, indent=2)
    print(f"  Saved: {campy_out} ({len(campy)} records)")

    biome_out = DATA_DIR / "biome_16s_projects.json"
    with open(biome_out, "w") as f:
        json.dump({"metadata": manifest, "projects": biome}, f, indent=2)
    print(f"  Saved: {biome_out} ({len(biome)} records)")

    # Summary
    print("\n" + "=" * 60)
    print("  Phase 35 Data Summary")
    print("=" * 60)
    print(f"  Vibrio assemblies:          {len(vibrio)}")
    print(f"  Campylobacterota assemblies: {len(campy)}")
    print(f"  16S BioProjects:            {len(biome)}")
    biome_counts = {}
    for r in biome:
        b = r.get("biome", "unknown")
        biome_counts[b] = biome_counts.get(b, 0) + 1
    for b, c in sorted(biome_counts.items()):
        print(f"    {b}: {c}")
    print("=" * 60)


if __name__ == "__main__":
    main()
