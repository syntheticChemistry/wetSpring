#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-16
"""
wetSpring Experiment 002 — Phytoplankton Microbiome 16S Pipeline
Runs the full 16S amplicon pipeline on real phytoplankton-associated
bacterial community data from PRJNA1195978.

Usage (inside Galaxy container with qiime2-2026 conda env):
    /tool_deps/_conda/envs/qiime2-2026/bin/python3 /tmp/run_exp002.py
"""

import os, sys, time, csv, json, gzip, shutil
from pathlib import Path

os.environ["PATH"] = "/tool_deps/_conda/envs/qiime2-2026/bin:" + os.environ.get("PATH", "")

WORK = "/tmp/exp002"
DATA = "/tmp/exp002_data"


def log(msg):
    print(f"[EXP002] {msg}", flush=True)


def step_prepare():
    """Build manifest and gzip FASTQs."""
    log("Step 1: Preparing data...")
    os.makedirs(WORK, exist_ok=True)

    samples = set()
    for f in os.listdir(DATA):
        if f.endswith("_1.fastq"):
            samples.add(f.replace("_1.fastq", ""))

    log(f"  Found {len(samples)} samples")

    # Gzip if needed
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".fastq") and not os.path.exists(f"{DATA}/{f}.gz"):
            with open(f"{DATA}/{f}", "rb") as fin:
                with gzip.open(f"{DATA}/{f}.gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)

    # Build manifest
    manifest_path = f"{WORK}/manifest.tsv"
    rows = []
    for sample in sorted(samples):
        rows.append({
            "sample-id": sample,
            "forward-absolute-filepath": f"{DATA}/{sample}_1.fastq.gz",
            "reverse-absolute-filepath": f"{DATA}/{sample}_2.fastq.gz",
        })

    with open(manifest_path, "w") as f:
        w = csv.DictWriter(f,
            fieldnames=["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"],
            delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    log(f"  Manifest: {len(rows)} samples")
    return manifest_path, len(rows)


def step_import(manifest_path):
    """Import paired-end reads into QIIME2."""
    log("Step 2: QIIME2 import...")
    from qiime2 import Artifact

    t0 = time.time()
    demux = Artifact.import_data(
        "SampleData[PairedEndSequencesWithQuality]",
        manifest_path,
        view_type="PairedEndFastqManifestPhred33V2"
    )
    qza_path = f"{WORK}/demux.qza"
    demux.save(qza_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(qza_path) / 1e6
    log(f"  Import: {size_mb:.1f} MB in {elapsed:.1f}s")
    return qza_path


def step_dada2(demux_path):
    """Run DADA2 denoise-paired (151bp reads → trunc at 140/120)."""
    log("Step 3: DADA2 denoise-paired (trunc_f=140, trunc_r=120, 8 threads)...")
    from qiime2 import Artifact
    from qiime2.plugins.dada2.methods import denoise_paired

    demux = Artifact.load(demux_path)
    t0 = time.time()
    results = denoise_paired(
        demultiplexed_seqs=demux,
        trunc_len_f=140,
        trunc_len_r=120,
        n_threads=8,
    )
    elapsed = time.time() - t0

    table = results.table
    rep_seqs = results.representative_sequences
    stats = results.denoising_stats

    table.save(f"{WORK}/table.qza")
    rep_seqs.save(f"{WORK}/rep-seqs.qza")
    stats.save(f"{WORK}/stats.qza")

    import biom
    bt = table.view(biom.Table)
    n_asvs = bt.shape[0]
    n_samples = bt.shape[1]
    total_reads = int(bt.sum())

    log(f"  DADA2: {n_asvs} ASVs, {n_samples} samples, {total_reads:,} reads in {elapsed:.1f}s")

    # Per-sample counts
    log("  Per-sample read counts:")
    for sid in sorted(bt.ids(axis="sample")):
        count = int(bt.data(sid, axis="sample", dense=True).sum())
        log(f"    {sid}: {count:,}")

    return {
        "n_asvs": n_asvs,
        "n_samples": n_samples,
        "total_reads": total_reads,
        "elapsed": elapsed,
    }


def step_taxonomy():
    """Run SILVA 138 NB taxonomy classification."""
    log("Step 4: SILVA 138 taxonomy classification...")
    from qiime2 import Artifact
    from qiime2.plugins.feature_classifier.methods import classify_sklearn

    classifier_path = "/tmp/silva-classifier.qza"
    if not os.path.exists(classifier_path):
        log(f"  ERROR: SILVA classifier not found: {classifier_path}")
        return None

    rep_seqs = Artifact.load(f"{WORK}/rep-seqs.qza")
    classifier = Artifact.load(classifier_path)

    t0 = time.time()
    results = classify_sklearn(reads=rep_seqs, classifier=classifier, n_jobs=8)
    taxonomy = results.classification
    elapsed = time.time() - t0

    taxonomy.save(f"{WORK}/taxonomy.qza")
    log(f"  Classification in {elapsed:.1f}s")

    import pandas as pd
    tax_df = taxonomy.view(pd.DataFrame)

    phyla = {}
    families = {}
    for _, row in tax_df.iterrows():
        lineage = row["Taxon"]
        parts = lineage.split(";")
        phylum, family = "Unassigned", "Unassigned"
        for p in parts:
            ps = p.strip()
            if ps.startswith("p__"):
                phylum = ps.replace("p__", "") or "Unknown"
            if ps.startswith("f__"):
                family = ps.replace("f__", "") or "Unknown"
        phyla[phylum] = phyla.get(phylum, 0) + 1
        families[family] = families.get(family, 0) + 1

    log(f"  Classified: {len(tax_df)} features, {len(phyla)} phyla")
    log(f"  Phylum distribution:")
    for p, count in sorted(phyla.items(), key=lambda x: -x[1])[:10]:
        log(f"    {p}: {count} ASVs")

    log(f"  Top families:")
    for f, count in sorted(families.items(), key=lambda x: -x[1])[:10]:
        log(f"    {f}: {count} ASVs")

    return {
        "n_classified": len(tax_df),
        "n_phyla": len(phyla),
        "phyla": phyla,
        "families": families,
        "elapsed": elapsed,
    }


def step_barplot():
    """Generate taxonomy barplot."""
    log("Step 5: Taxonomy barplot...")
    from qiime2 import Artifact, Metadata
    from qiime2.plugins.taxa.visualizers import barplot
    import pandas as pd
    import biom

    table = Artifact.load(f"{WORK}/table.qza")
    taxonomy = Artifact.load(f"{WORK}/taxonomy.qza")

    bt = table.view(biom.Table)
    meta_data = {}
    for sid in bt.ids(axis="sample"):
        meta_data[sid] = {"SampleType": "phytoplankton-associated"}

    meta_df = pd.DataFrame.from_dict(meta_data, orient="index")
    meta_df.index.name = "sample-id"

    viz = barplot(table=table, taxonomy=taxonomy, metadata=Metadata(meta_df))
    viz.visualization.save(f"{WORK}/barplot.qzv")
    size_kb = os.path.getsize(f"{WORK}/barplot.qzv") / 1024
    log(f"  Barplot: {size_kb:.0f} KB")
    return True


def main():
    t_total = time.time()

    log("=" * 60)
    log("EXPERIMENT 002 — PHYTOPLANKTON MICROBIOME 16S")
    log("BioProject: PRJNA1195978")
    log("=" * 60)

    manifest, n_samples = step_prepare()
    demux_path = step_import(manifest)

    dada2 = step_dada2(demux_path)
    tax = step_taxonomy()
    step_barplot()

    total_time = time.time() - t_total

    log("")
    log("=" * 60)
    log(f"EXPERIMENT 002 COMPLETE")
    log(f"  Samples: {dada2['n_samples']}")
    log(f"  ASVs: {dada2['n_asvs']}")
    log(f"  Total reads: {dada2['total_reads']:,}")
    log(f"  Phyla: {tax['n_phyla'] if tax else '?'}")
    log(f"  Pipeline time: {total_time:.1f}s")
    log("=" * 60)

    # Write report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bioproject": "PRJNA1195978",
        "pipeline_time_s": round(total_time, 1),
        "dada2": {
            "asv_count": dada2["n_asvs"],
            "sample_count": dada2["n_samples"],
            "total_reads": dada2["total_reads"],
            "time_s": round(dada2["elapsed"], 1),
        },
        "taxonomy": {
            "classified": tax["n_classified"] if tax else 0,
            "phyla_count": tax["n_phyla"] if tax else 0,
            "phyla": dict(sorted(tax["phyla"].items(), key=lambda x: -x[1])) if tax else {},
            "top_families": dict(sorted(tax["families"].items(), key=lambda x: -x[1])[:15]) if tax else {},
        } if tax else None,
    }

    report_path = f"{WORK}/exp002_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
