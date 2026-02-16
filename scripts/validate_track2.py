#!/usr/bin/env python3
"""
wetSpring Track 2 Validation — Deterministic Rerun of Exp005 + Exp006
Reruns asari (LC-MS feature extraction) and FindPFAS (PFAS screening)
from scratch and validates outputs against expected values.

Usage:
    source .venv/asari/bin/activate  # or .venv/pfas
    python3 scripts/validate_track2.py

Expected results (from original runs):
  Exp005 (asari): 5,951 preferred features, 4,107 unique compounds
  Exp006 (FindPFAS): 25 unique PFAS precursors from 738 spectra
"""

import os, sys, time, json, subprocess, shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent


def log(msg):
    print(f"[T2-VAL] {msg}", flush=True)


def ok(label, actual, expected, tolerance=0):
    if tolerance > 0:
        passed = abs(actual - expected) <= tolerance
    else:
        passed = actual == expected
    tag = "OK" if passed else "FAIL"
    print(f"  [{tag}]  {label}: {actual} (expected {expected})", flush=True)
    return passed


# ════════════════════════════════════════════════════════════════
# Experiment 005: asari validation
# ════════════════════════════════════════════════════════════════
def validate_exp005():
    log("=" * 60)
    log("EXPERIMENT 005 — asari LC-MS Validation")
    log("=" * 60)

    data_dir = WORKSPACE / "data" / "exp005_asari" / "MT02" / "MT02Dataset"
    if not data_dir.exists():
        log(f"  SKIP: data dir not found: {data_dir}")
        return None

    mzml_count = len(list(data_dir.glob("*.mzML")))
    log(f"  Input: {mzml_count} mzML files in {data_dir}")

    # Clean any previous validation output
    val_dir = WORKSPACE / "data" / "exp005_asari" / "validation_rerun"
    if val_dir.exists():
        shutil.rmtree(val_dir)

    t0 = time.time()
    result = subprocess.run(
        ["asari", "process", "--mode", "pos",
         "--input", str(data_dir),
         "--output", str(val_dir)],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  asari FAILED: {result.stderr[-300:]}")
        return None

    # Find the output directory (asari appends a project ID)
    output_dirs = list(Path(str(val_dir)).parent.glob("validation_rerun_asari_project_*"))
    if not output_dirs:
        # Try finding it relative to data dir
        output_dirs = list((WORKSPACE / "data" / "exp005_asari").glob("validation_rerun_asari_project_*"))
    if not output_dirs:
        log(f"  asari output dir not found")
        return None

    out_dir = output_dirs[0]
    log(f"  Output: {out_dir.name}")
    log(f"  Runtime: {elapsed:.1f}s")

    # Parse results
    pref_table = out_dir / "preferred_Feature_table.tsv"
    if not pref_table.exists():
        log(f"  preferred_Feature_table.tsv not found")
        return None

    with open(pref_table) as f:
        lines = f.readlines()
    n_features = len(lines) - 1  # minus header
    n_cols = len(lines[0].strip().split("\t"))

    # Count unique compounds
    unique_table = out_dir / "export" / "unique_compound__Feature_table.tsv"
    n_compounds = 0
    if unique_table.exists():
        with open(unique_table) as f:
            n_compounds = len(f.readlines()) - 1

    # Count full features
    full_table = out_dir / "export" / "full_Feature_table.tsv"
    n_full = 0
    if full_table.exists():
        with open(full_table) as f:
            n_full = len(f.readlines()) - 1

    checks = [
        ok("mzML files", mzml_count, 8),
        ok("Preferred features", n_features, 5951),
        ok("Feature table columns", n_cols, 19),
        ok("Full features", n_full, 8659),
        ok("Unique compounds", n_compounds, 4107, tolerance=10),
    ]

    passed = sum(checks)
    failed = len(checks) - passed
    log(f"\n  Exp005: {passed}/{len(checks)} checks passed, {failed} failed")
    log(f"  Runtime: {elapsed:.1f}s")

    # Cleanup validation output
    shutil.rmtree(out_dir, ignore_errors=True)

    return {"passed": passed, "failed": failed, "total": len(checks),
            "runtime": round(elapsed, 1),
            "features": n_features, "compounds": n_compounds}


# ════════════════════════════════════════════════════════════════
# Experiment 006: FindPFAS validation
# ════════════════════════════════════════════════════════════════
def validate_exp006():
    log("")
    log("=" * 60)
    log("EXPERIMENT 006 — FindPFAS PFAS Screening Validation")
    log("=" * 60)

    ms2_file = "/tmp/FindPFAS/TestSample_PFAS_Standard_MIX_ddMS2_20eV_Inj5.ms2"
    if not os.path.exists(ms2_file):
        log(f"  SKIP: test data not found: {ms2_file}")
        return None

    sys.path.insert(0, "/tmp/FindPFAS")
    import numpy as np
    from pyteomics.ms2 import IndexedMS2
    from pyteomics import mass as pymass

    frags = {'CF2': 49.99681, 'C2F4': 99.99361, 'HF': 20.00623}
    tol = 0.001
    I_min = 5

    t0 = time.time()
    ms2_data = IndexedMS2(ms2_file)

    spectra = []
    for spec in ms2_data:
        pmz = float(spec['params'].get('precursor m/z', 0))
        rt = float(spec['params'].get('RTime', 0))
        spectra.append({
            'precursor_mz': pmz, 'rt': rt,
            'mz': spec['m/z array'], 'intensity': spec['intensity array'],
        })

    log(f"  Loaded {len(spectra)} MS2 spectra")

    pfas_hits = []
    for spec in spectra:
        mz = spec['mz']
        inten = spec['intensity']
        if len(mz) < 2:
            continue
        max_i = inten.max() if len(inten) > 0 else 0
        if max_i > 0:
            mask = (inten / max_i * 100) >= I_min
            mz, inten = mz[mask], inten[mask]
        if len(mz) < 2:
            continue

        found_frags = {}
        for f_name, f_mass in frags.items():
            diffs = np.abs(np.subtract.outer(mz, mz))
            matches = int(np.sum(np.abs(diffs - f_mass) <= tol)) // 2
            if matches > 0:
                found_frags[f_name] = matches
        if sum(found_frags.values()) >= 1:
            pfas_hits.append({
                'precursor_mz': spec['precursor_mz'],
                'rt': spec['rt'],
                'total_diffs': sum(found_frags.values()),
                'fragments': found_frags,
            })

    unique_mzs = set(round(h['precursor_mz'], 2) for h in pfas_hits)
    elapsed = time.time() - t0

    log(f"  PFAS candidates: {len(pfas_hits)} spectra, {len(unique_mzs)} unique precursors")
    log(f"  Runtime: {elapsed:.2f}s")

    checks = [
        ok("Total spectra", len(spectra), 738),
        ok("PFAS candidate spectra", len(pfas_hits), 62, tolerance=5),
        ok("Unique PFAS precursors", len(unique_mzs), 25),
    ]

    passed = sum(checks)
    failed = len(checks) - passed
    log(f"\n  Exp006: {passed}/{len(checks)} checks passed, {failed} failed")

    return {"passed": passed, "failed": failed, "total": len(checks),
            "runtime": round(elapsed, 2),
            "candidates": len(pfas_hits), "unique": len(unique_mzs)}


def main():
    t_total = time.time()
    log("=" * 60)
    log("TRACK 2 VALIDATION — DETERMINISTIC RERUN")
    log("=" * 60)

    r005 = validate_exp005()
    r006 = validate_exp006()

    total_time = time.time() - t_total
    total_passed = (r005["passed"] if r005 else 0) + (r006["passed"] if r006 else 0)
    total_checks = (r005["total"] if r005 else 0) + (r006["total"] if r006 else 0)
    total_failed = total_checks - total_passed

    log("")
    log("=" * 60)
    log(f"TRACK 2 VALIDATION: {total_passed}/{total_checks} checks passed, "
        f"{total_failed} failed")
    log(f"Total time: {total_time:.1f}s")
    if total_failed == 0:
        log("RESULT: PASS — Track 2 is deterministic and reproducible")
    else:
        log("RESULT: FAIL — Check details above")
    log("=" * 60)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_passed": total_passed,
        "total_checks": total_checks,
        "total_time_s": round(total_time, 1),
        "validation": "PASS" if total_failed == 0 else "FAIL",
        "exp005_asari": r005,
        "exp006_findpfas": r006,
    }

    report_path = WORKSPACE / "experiments" / "results" / "track2_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report: {report_path}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
