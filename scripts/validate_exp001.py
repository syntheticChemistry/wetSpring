#!/usr/bin/env python3
"""
wetSpring Experiment 001 — Full Pipeline Validation
Runs the complete 16S amplicon pipeline end-to-end and validates outputs.

Usage (inside Galaxy container with qiime2-2026 conda env):
    /tool_deps/_conda/envs/qiime2-2026/bin/python3 /scripts/validate_exp001.py

Expected: deterministic results matching our original run:
  - 232 ASVs, 124,249 non-chimeric reads
  - 9 phyla (Firmicutes dominant at 191 ASVs)
  - Mock community: 89.3% retention, 4,269 reads
"""

import os, sys, time, csv, json, tempfile, zipfile
from pathlib import Path

os.environ["PATH"] = "/tool_deps/_conda/envs/qiime2-2026/bin:" + os.environ.get("PATH", "")

WORK = "/tmp/exp001_validation"
DATA = "/tmp/MiSeq_SOP"

EXPECTED = {
    "asv_count": 232,
    "sample_count": 20,
    "total_nonchimeric": 124249,
    "phyla_count": 9,
    "firmicutes_asvs": 191,
    "bacteroidota_asvs": 20,
    "mock_nonchimeric": 4269,
    "mock_retention_pct": 89.3,
}


def log(msg):
    print(f"[EXP001] {msg}", flush=True)


def fail(msg):
    print(f"[FAIL]   {msg}", flush=True)
    return False


def ok(msg):
    print(f"[OK]     {msg}", flush=True)
    return True


def check(label, actual, expected, tolerance=0):
    if tolerance > 0:
        if abs(actual - expected) <= tolerance:
            return ok(f"{label}: {actual} (expected {expected}, tol={tolerance})")
        return fail(f"{label}: {actual} != {expected} (tol={tolerance})")
    if actual == expected:
        return ok(f"{label}: {actual}")
    return fail(f"{label}: {actual} != {expected}")


def step_manifest():
    """Create QIIME2-format manifest from MiSeq SOP FASTQs."""
    log("Step 1: Building manifest...")
    os.makedirs(WORK, exist_ok=True)

    if not os.path.exists(DATA):
        return fail(f"Data dir missing: {DATA}")

    samples = set()
    for f in os.listdir(DATA):
        if f.endswith("_R1.fastq"):
            samples.add(f.replace("_R1.fastq", ""))

    manifest_path = f"{WORK}/manifest.tsv"
    rows = []
    for sample in sorted(samples):
        r1 = f"{DATA}/{sample}_R1.fastq"
        r2 = f"{DATA}/{sample}_R2.fastq"
        if os.path.exists(r1) and os.path.exists(r2):
            rows.append({
                "sample-id": sample,
                "forward-absolute-filepath": f"{DATA}/{sample}_R1.fastq.gz",
                "reverse-absolute-filepath": f"{DATA}/{sample}_R2.fastq.gz",
            })

    with open(manifest_path, "w") as f:
        w = csv.DictWriter(f,
            fieldnames=["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"],
            delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    check("Samples in manifest", len(rows), 20)
    return manifest_path


def step_gzip():
    """Ensure gzipped copies exist (QIIME2 requires .fastq.gz)."""
    log("Step 2: Ensuring gzipped FASTQs...")
    import gzip, shutil
    count = 0
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".fastq") and not os.path.exists(f"{DATA}/{f}.gz"):
            with open(f"{DATA}/{f}", "rb") as fin:
                with gzip.open(f"{DATA}/{f}.gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            count += 1
    gz_count = len([f for f in os.listdir(DATA) if f.endswith(".fastq.gz")])
    check("Gzipped FASTQs", gz_count, 40)
    return gz_count


def step_import(manifest_path):
    """Import paired-end reads into QIIME2 artifact."""
    log("Step 3: QIIME2 import...")
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
    ok(f"Import: {size_mb:.1f} MB in {elapsed:.1f}s (UUID: {demux.uuid})")
    return qza_path


def step_dada2(demux_path):
    """Run DADA2 denoise-paired."""
    log("Step 4: DADA2 denoise-paired (trunc_f=240, trunc_r=160, 8 threads)...")
    from qiime2 import Artifact
    from qiime2.plugins.dada2.methods import denoise_paired

    demux = Artifact.load(demux_path)
    t0 = time.time()
    results = denoise_paired(
        demultiplexed_seqs=demux,
        trunc_len_f=240,
        trunc_len_r=160,
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
    mock_reads = int(bt.data("Mock", axis="sample", dense=True).sum())

    ok(f"DADA2: {n_asvs} ASVs, {n_samples} samples, {total_reads} reads in {elapsed:.1f}s")

    # Extract stats TSV
    with zipfile.ZipFile(f"{WORK}/stats.qza") as z:
        for name in z.namelist():
            if name.endswith("stats.tsv"):
                with open(f"{WORK}/stats.tsv", "wb") as f:
                    f.write(z.read(name))

    # Parse per-sample stats
    sample_stats = {}
    with open(f"{WORK}/stats.tsv") as f:
        reader = csv.DictReader(
            (row for row in f if not row.startswith("#")),
            delimiter="\t"
        )
        for row in reader:
            sid = row["sample-id"]
            sample_stats[sid] = {
                "input": int(row["input"]),
                "filtered": int(row["filtered"]),
                "non-chimeric": int(row["non-chimeric"]),
            }

    mock_input = sample_stats.get("Mock", {}).get("input", 0)
    mock_nc = sample_stats.get("Mock", {}).get("non-chimeric", 0)
    mock_pct = (mock_nc / mock_input * 100) if mock_input > 0 else 0

    return {
        "n_asvs": n_asvs,
        "n_samples": n_samples,
        "total_reads": total_reads,
        "mock_reads": mock_reads,
        "mock_pct": mock_pct,
        "elapsed": elapsed,
        "sample_stats": sample_stats,
    }


def step_taxonomy():
    """Run SILVA 138 NB taxonomy classification."""
    log("Step 5: SILVA 138 taxonomy classification...")
    from qiime2 import Artifact
    from qiime2.plugins.feature_classifier.methods import classify_sklearn

    classifier_path = "/tmp/silva-classifier.qza"
    if not os.path.exists(classifier_path):
        return fail(f"SILVA classifier not found: {classifier_path}")

    rep_seqs = Artifact.load(f"{WORK}/rep-seqs.qza")
    classifier = Artifact.load(classifier_path)

    t0 = time.time()
    results = classify_sklearn(reads=rep_seqs, classifier=classifier, n_jobs=8)
    taxonomy = results.classification
    elapsed = time.time() - t0

    taxonomy.save(f"{WORK}/taxonomy.qza")
    ok(f"Taxonomy classified in {elapsed:.1f}s")

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

    return {
        "n_classified": len(tax_df),
        "n_phyla": len(phyla),
        "phyla": phyla,
        "families": families,
        "elapsed": elapsed,
    }


def step_barplot():
    """Generate taxonomy barplot."""
    log("Step 6: Taxonomy barplot...")
    from qiime2 import Artifact, Metadata
    from qiime2.plugins.taxa.visualizers import barplot
    import pandas as pd
    import biom

    table = Artifact.load(f"{WORK}/table.qza")
    taxonomy = Artifact.load(f"{WORK}/taxonomy.qza")

    bt = table.view(biom.Table)
    meta_data = {}
    for sid in bt.ids(axis="sample"):
        if sid == "Mock":
            group = "Mock"
        elif int(sid.replace("F3D", "")) < 100:
            group = "Early"
        else:
            group = "Late"
        meta_data[sid] = {"Treatment": group}

    meta_df = pd.DataFrame.from_dict(meta_data, orient="index")
    meta_df.index.name = "sample-id"

    viz = barplot(table=table, taxonomy=taxonomy, metadata=Metadata(meta_df))
    viz.visualization.save(f"{WORK}/barplot.qzv")
    size_kb = os.path.getsize(f"{WORK}/barplot.qzv") / 1024
    ok(f"Barplot: {size_kb:.0f} KB")
    return True


def main():
    t_total = time.time()
    passed = 0
    failed = 0
    results = {}

    log("=" * 60)
    log("EXPERIMENT 001 VALIDATION — CLEAN RERUN")
    log("=" * 60)

    # Step 1: Manifest
    manifest = step_manifest()
    if not manifest:
        return 1

    # Step 2: Gzip
    step_gzip()

    # Step 3: Import
    try:
        demux_path = step_import(manifest)
    except Exception as e:
        fail(f"Import: {e}")
        return 1

    # Step 4: DADA2
    try:
        dada2 = step_dada2(demux_path)
        results["dada2"] = dada2
    except Exception as e:
        fail(f"DADA2: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Step 5: Taxonomy
    try:
        tax = step_taxonomy()
        results["taxonomy"] = tax
    except Exception as e:
        fail(f"Taxonomy: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Step 6: Barplot
    try:
        step_barplot()
    except Exception as e:
        fail(f"Barplot: {e}")
        import traceback; traceback.print_exc()

    # Validation checks
    log("")
    log("=" * 60)
    log("VALIDATION CHECKS")
    log("=" * 60)

    checks = [
        check("ASV count", dada2["n_asvs"], EXPECTED["asv_count"]),
        check("Sample count", dada2["n_samples"], EXPECTED["sample_count"]),
        check("Total non-chimeric", dada2["total_reads"], EXPECTED["total_nonchimeric"]),
        check("Phyla count", tax["n_phyla"], EXPECTED["phyla_count"]),
        check("Firmicutes ASVs", tax["phyla"].get("Firmicutes", 0), EXPECTED["firmicutes_asvs"]),
        check("Bacteroidota ASVs", tax["phyla"].get("Bacteroidota", 0), EXPECTED["bacteroidota_asvs"]),
        check("Mock non-chimeric", dada2["mock_reads"], EXPECTED["mock_nonchimeric"]),
        check("Mock retention %", round(dada2["mock_pct"], 1), EXPECTED["mock_retention_pct"]),
    ]

    passed = sum(1 for c in checks if c)
    failed = sum(1 for c in checks if not c)
    total_time = time.time() - t_total

    log("")
    log("=" * 60)
    log(f"RESULT: {passed}/{len(checks)} checks passed, {failed} failed")
    log(f"Total pipeline time: {total_time:.1f}s")
    log(f"  DADA2: {dada2['elapsed']:.1f}s")
    log(f"  Taxonomy: {tax['elapsed']:.1f}s")
    log("=" * 60)

    if failed == 0:
        log("EXPERIMENT 001 VALIDATION: PASS")
    else:
        log("EXPERIMENT 001 VALIDATION: FAIL")

    # Write JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_time_s": round(total_time, 1),
        "dada2_time_s": round(dada2["elapsed"], 1),
        "taxonomy_time_s": round(tax["elapsed"], 1),
        "checks_passed": passed,
        "checks_failed": failed,
        "dada2": {
            "asv_count": dada2["n_asvs"],
            "sample_count": dada2["n_samples"],
            "total_nonchimeric": dada2["total_reads"],
            "mock_nonchimeric": dada2["mock_reads"],
            "mock_retention_pct": round(dada2["mock_pct"], 1),
        },
        "taxonomy": {
            "classified": tax["n_classified"],
            "phyla_count": tax["n_phyla"],
            "phyla": dict(sorted(tax["phyla"].items(), key=lambda x: -x[1])),
            "top_families": dict(sorted(tax["families"].items(), key=lambda x: -x[1])[:10]),
        },
        "validation": "PASS" if failed == 0 else "FAIL",
    }
    report_path = f"{WORK}/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
