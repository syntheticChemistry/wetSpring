#!/usr/bin/env python3
"""Search NCBI SRA for public datasets matching paper organisms.

Queries NCBI Entrez for 16S amplicon datasets from:
- Nannochloropsis / Microchloropsis cultivation
- Marine algae pond microbiomes
- Rotifer (Brachionus) associated microbiomes
- Bacillus in aquaculture settings

Outputs a ranked list of BioProjects suitable for pipeline validation.

Usage:
    python3 scripts/search_ncbi_datasets.py [--api-key KEY]

Requires: urllib (stdlib only, no pip deps)
"""

import json
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "ncbi_dataset_search"


def esearch(db: str, term: str, retmax: int = 20, api_key: str = "") -> dict:
    """Search NCBI and return UIDs."""
    params = {
        "db": db,
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def esummary(db: str, ids: list, api_key: str = "") -> list:
    """Fetch summaries for a list of UIDs."""
    if not ids:
        return []
    params = {
        "db": db,
        "id": ",".join(str(i) for i in ids[:20]),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EUTILS_BASE}/esummary.fcgi?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    result = data.get("result", {})
    uids = result.get("uids", [])
    return [result[uid] for uid in uids if uid in result]


def search_sra_datasets(queries: list, api_key: str = "", delay: float = 0.35) -> list:
    """Search SRA for each query and return combined results."""
    all_results = []
    for query_info in queries:
        term = query_info["term"]
        label = query_info["label"]
        print(f"  Searching: {label}")
        print(f"    Term: {term}")

        try:
            search = esearch("sra", term, retmax=15, api_key=api_key)
            count = int(search.get("esearchresult", {}).get("count", 0))
            ids = search.get("esearchresult", {}).get("idlist", [])
            print(f"    Found: {count} total, fetching top {len(ids)}")

            if ids:
                time.sleep(delay)
                summaries = esummary("sra", ids, api_key=api_key)
                for s in summaries:
                    all_results.append({
                        "search_label": label,
                        "uid": s.get("uid", ""),
                        "accession": s.get("accession", ""),
                        "title": s.get("expxml", "")[:200] if "expxml" in s else s.get("title", "")[:200],
                        "total_runs": s.get("runs", ""),
                        "total_bases": s.get("total_bases", ""),
                    })
                time.sleep(delay)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)

    return all_results


def search_bioproject(queries: list, api_key: str = "", delay: float = 0.35) -> list:
    """Search BioProject for each query."""
    all_results = []
    for query_info in queries:
        term = query_info["term"]
        label = query_info["label"]
        print(f"  Searching BioProject: {label}")

        try:
            search = esearch("bioproject", term, retmax=15, api_key=api_key)
            count = int(search.get("esearchresult", {}).get("count", 0))
            ids = search.get("esearchresult", {}).get("idlist", [])
            print(f"    Found: {count} total, fetching top {len(ids)}")

            if ids:
                time.sleep(delay)
                summaries = esummary("bioproject", ids, api_key=api_key)
                for s in summaries:
                    all_results.append({
                        "search_label": label,
                        "uid": s.get("uid", ""),
                        "accession": s.get("project_acc", ""),
                        "title": s.get("project_title", "")[:200],
                        "description": s.get("project_description", "")[:300],
                        "organism": s.get("organism_name", ""),
                        "data_type": s.get("project_data_type", ""),
                    })
                time.sleep(delay)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)

    return all_results


def load_api_key() -> str:
    """Load NCBI API key from testing-secrets or CLI args."""
    if len(sys.argv) > 2 and sys.argv[1] == "--api-key":
        return sys.argv[2]

    secrets_path = Path(__file__).resolve().parent.parent.parent / "testing-secrets" / "api-keys.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ncbi_api_key") and "=" in line:
                val = line.split("=", 1)[1].strip().strip('"')
                if val:
                    return val
    return ""


def main():
    api_key = load_api_key()

    print("=" * 60)
    print("  NCBI Dataset Search for wetSpring Validation")
    print("=" * 60)
    if api_key:
        print(f"  API key: {api_key[:8]}... (10 req/s)")
    else:
        print("  No API key (3 req/s limit)")
    print()

    rate_delay = 0.12 if api_key else 0.35

    # SRA searches targeting 16S amplicon from relevant organisms
    sra_queries = [
        {
            "label": "Nannochloropsis 16S amplicon outdoor",
            "term": '(Nannochloropsis[Organism] OR Microchloropsis[Organism]) AND "16S"[All Fields] AND "amplicon"[All Fields] AND "AMPLICON"[Strategy]',
        },
        {
            "label": "Algae pond microbiome 16S",
            "term": '"algae pond"[All Fields] AND "16S"[All Fields] AND "AMPLICON"[Strategy] AND "PAIRED"[Layout]',
        },
        {
            "label": "Brachionus rotifer microbiome",
            "term": '(Brachionus[Organism]) AND "16S"[All Fields] AND "AMPLICON"[Strategy]',
        },
        {
            "label": "Marine aquaculture microbiome 16S",
            "term": '"aquaculture"[All Fields] AND "microbiome"[All Fields] AND "16S"[All Fields] AND "AMPLICON"[Strategy] AND "PAIRED"[Layout]',
        },
        {
            "label": "Algae biofuel microbiome",
            "term": '"algae"[All Fields] AND "biofuel"[All Fields] AND "microbiome"[All Fields] AND "16S"[All Fields]',
        },
        {
            "label": "Harmful algal bloom microbiome",
            "term": '"harmful algal bloom"[All Fields] AND "16S"[All Fields] AND "AMPLICON"[Strategy] AND "PAIRED"[Layout]',
        },
        {
            "label": "Microalgae toxin 16S",
            "term": '("microalgae"[All Fields] OR "cyanobacteria"[All Fields]) AND ("toxin"[All Fields] OR "microcystin"[All Fields]) AND "16S"[All Fields] AND "AMPLICON"[Strategy]',
        },
        {
            "label": "Algae wastewater treatment microbiome",
            "term": '"algae"[All Fields] AND "wastewater"[All Fields] AND "16S"[All Fields] AND "AMPLICON"[Strategy] AND "PAIRED"[Layout]',
        },
    ]

    # BioProject searches
    bioproject_queries = [
        {
            "label": "Nannochloropsis microbiome BioProject",
            "term": "Nannochloropsis[Organism] AND microbiome[All Fields]",
        },
        {
            "label": "Algae raceway pond BioProject",
            "term": '"raceway pond"[All Fields] AND algae[All Fields] AND 16S[All Fields]',
        },
        {
            "label": "Microalgae grazer interaction",
            "term": '"microalgae"[All Fields] AND ("grazer"[All Fields] OR "rotifer"[All Fields]) AND 16S[All Fields]',
        },
        {
            "label": "Algae pond crash or contamination",
            "term": '"algae"[All Fields] AND ("pond crash"[All Fields] OR "contamination"[All Fields] OR "culture collapse"[All Fields]) AND 16S[All Fields]',
        },
        {
            "label": "Cyanobacteria toxin microbiome",
            "term": '("cyanobacteria"[All Fields] OR "cyanobacterial bloom"[All Fields]) AND ("toxin"[All Fields] OR "microcystin"[All Fields]) AND 16S[All Fields]',
        },
        {
            "label": "Chlorella or Spirulina microbiome",
            "term": '(Chlorella[Organism] OR Spirulina[Organism] OR Arthrospira[Organism]) AND "microbiome"[All Fields] AND 16S[All Fields]',
        },
    ]

    print("── SRA Searches ────────────────────────────────────────")
    sra_results = search_sra_datasets(sra_queries, api_key, rate_delay)
    print(f"\n  Total SRA results: {len(sra_results)}\n")

    print("── BioProject Searches ─────────────────────────────────")
    bp_results = search_bioproject(bioproject_queries, api_key, rate_delay)
    print(f"\n  Total BioProject results: {len(bp_results)}\n")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "search_date": time.strftime("%Y-%m-%d"),
        "api_key_used": bool(api_key),
        "sra_results": sra_results,
        "bioproject_results": bp_results,
    }

    outpath = RESULTS_DIR / "ncbi_search_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved: {outpath}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Summary of Candidate Datasets")
    print("=" * 60)

    seen = set()
    for r in bp_results:
        acc = r.get("accession", "")
        if acc and acc not in seen:
            seen.add(acc)
            print(f"\n  {acc}: {r.get('title', 'N/A')[:80]}")
            print(f"    Organism: {r.get('organism', 'N/A')}")
            print(f"    Search: {r.get('search_label', '')}")

    print(f"\n  Total unique BioProjects: {len(seen)}")
    print(f"  Total SRA experiments: {len(sra_results)}")
    print()
    print("  Next: Download candidates with scripts/ncbi_bulk_download.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
