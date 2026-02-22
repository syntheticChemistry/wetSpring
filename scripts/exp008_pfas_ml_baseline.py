#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-19
"""Experiment 008: PFAS ML Water Monitoring — Python Baseline.

Trains a Random Forest classifier on Michigan EGLE PFAS surface water data
to predict whether a sample exceeds the EPA PFOA+PFOS advisory threshold
(4 ng/L combined). Produces baseline accuracy metrics for Rust port.

Data: Michigan DEQ/EGLE PFAS surface water sampling (3,719 records)
Source: ArcGIS REST API (gisagoegle.state.mi.us)

Requires: pip install numpy scipy scikit-learn pandas
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_PATH = WORKSPACE / "data/michigan_deq_pfas/pfas_surface_water_all.json"

# EPA health advisory: 4 ng/L (ppt) combined PFOA + PFOS
EPA_ADVISORY_THRESHOLD_NGpL = 4.0

# PFAS analyte columns (CAS number prefix → analyte name)
PFAS_ANALYTES = [
    ("CAS307244_PFHxA", "PFHxA"),
    ("CAS307551_PFDoA", "PFDoA"),
    ("CAS335671_PFOA", "PFOA"),
    ("CAS335762_PFDA", "PFDA"),
    ("CAS335773_PFDS", "PFDS"),
    ("CAS355464_PFHxS", "PFHxS"),
    ("CAS375224_PFBA", "PFBA"),
    ("CAS375735_PFBS", "PFBS"),
    ("CAS375859_PFHpA", "PFHpA"),
    ("CAS375928_PFHpS", "PFHpS"),
    ("CAS375951_PFNA", "PFNA"),
    ("CAS376067_PFTeA", "PFTeA"),
    ("CAS754916_PFOSA", "PFOSA"),
    ("CAS1763231_PFOS", "PFOS"),
    ("CAS2058948_PFUnA", "PFUnA"),
    ("CAS2355319_NMeFOSAA", "NMeFOSAA"),
    ("CAS2706903_PFPeA", "PFPeA"),
    ("CAS2706914_PFPeS", "PFPeS"),
    ("CAS2991506_NEtFOSAA", "NEtFOSAA"),
    ("CAS13252136_GenX", "GenX"),
    ("CAS27619972_62FTS", "6:2 FTS"),
    ("CAS39108344_82FTS", "8:2 FTS"),
]


def load_data():
    """Load Michigan DEQ PFAS data from downloaded JSON."""
    with open(DATA_PATH) as f:
        raw = json.load(f)

    records = [feat["attributes"] for feat in raw["features"]]
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} records, {len(df.columns)} columns")
    return df


def engineer_features(df):
    """Extract ML features from raw PFAS sampling records."""
    features = pd.DataFrame(index=df.index)

    # Geospatial features
    features["latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    features["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # PFAS concentration features (replace non-detect with 0)
    for col, name in PFAS_ANALYTES:
        if col in df.columns:
            features[name] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Derived: total PFAS concentration
    pfas_cols = [name for _, name in PFAS_ANALYTES if name in features.columns]
    features["total_pfas"] = features[pfas_cols].sum(axis=1)

    # Derived: PFAS compound count (non-zero analytes per sample)
    features["pfas_count"] = (features[pfas_cols] > 0).sum(axis=1)

    # Derived: max single analyte
    features["max_single_pfas"] = features[pfas_cols].max(axis=1)

    # Derived: PFCA vs PFSA ratio (carboxylic acids vs sulfonic acids)
    pfca = ["PFHxA", "PFOA", "PFDA", "PFHpA", "PFNA", "PFPeA", "PFBA"]
    pfsa = ["PFHxS", "PFBS", "PFOS", "PFHpS", "PFDS", "PFPeS"]
    pfca_cols = [c for c in pfca if c in features.columns]
    pfsa_cols = [c for c in pfsa if c in features.columns]
    pfca_sum = features[pfca_cols].sum(axis=1) if pfca_cols else 0
    pfsa_sum = features[pfsa_cols].sum(axis=1) if pfsa_cols else 0
    features["pfca_pfsa_ratio"] = np.where(
        pfsa_sum > 0, pfca_sum / pfsa_sum, 0.0
    )

    # Target: PFOA + PFOS exceeds EPA advisory (4 ng/L combined)
    pfoa = features.get("PFOA", pd.Series(0, index=df.index))
    pfos = features.get("PFOS", pd.Series(0, index=df.index))
    target = ((pfoa + pfos) >= EPA_ADVISORY_THRESHOLD_NGpL).astype(int)

    # Drop rows with missing lat/lon
    valid = features["latitude"].notna() & features["longitude"].notna()
    features = features[valid].copy()
    target = target[valid].copy()

    print(f"  Engineered {len(features.columns)} features for {len(features)} samples")
    print(f"  Target: {target.sum()} above advisory ({target.mean() * 100:.1f}%), "
          f"{(~target.astype(bool)).sum()} below")

    return features, target


def run_ml_baseline(features, target):
    """Train RF and GBM with 5-fold stratified CV, return metrics."""
    feature_cols = [c for c in features.columns if features[c].dtype in [np.float64, np.int64, float, int]]
    X = features[feature_cols].fillna(0).values
    y = target.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, clf in [
        ("RandomForest", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )),
        ("GradientBoosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )),
    ]:
        print(f"\n  Training {name} (5-fold stratified CV)...")
        t0 = time.time()

        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict")
        y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

        elapsed = time.time() - t0

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        try:
            auc = roc_auc_score(y, y_prob)
        except ValueError:
            auc = 0.0

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    AUC-ROC:   {auc:.4f}")
        print(f"    Elapsed:   {elapsed:.2f}s")

        # Feature importance (train on full data for this)
        clf.fit(X_scaled, y)
        importances = clf.feature_importances_
        top_features = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1], reverse=True
        )[:10]

        results[name] = {
            "accuracy": round(acc, 6),
            "f1_score": round(f1, 6),
            "auc_roc": round(auc, 6),
            "elapsed_seconds": round(elapsed, 2),
            "top_features": [
                {"name": n, "importance": round(float(imp), 6)}
                for n, imp in top_features
            ],
        }

        print(f"    Top features:")
        for fn, imp in top_features[:5]:
            print(f"      {fn:<20} {imp:.4f}")

    return results, feature_cols


def main():
    print("=" * 70)
    print("  Experiment 008: PFAS ML Water Monitoring — Python Baseline")
    print("  Michigan EGLE PFAS Surface Water Data")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"\n  ERROR: Data not found at {DATA_PATH}")
        print("  Run the download first (scripts/download_michigan_pfas.sh)")
        return 1

    # Load and explore
    print("\n  [1/3] Loading data...")
    df = load_data()

    # Engineer features
    print("\n  [2/3] Engineering features...")
    features, target = engineer_features(df)

    # Train models
    print("\n  [3/3] Training ML models...")
    results, feature_cols = run_ml_baseline(features, target)

    # Save results
    out_dir = WORKSPACE / "experiments/results/008_pfas_ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "008_pfas_ml_water_monitoring",
        "data_source": "Michigan EGLE PFAS Surface Water Sampling",
        "data_url": "https://gisagoegle.state.mi.us/arcgis/rest/services/EGLE/PfasOpenData/MapServer/0",
        "total_records": len(df),
        "valid_samples": len(features),
        "target_threshold_ngl": EPA_ADVISORY_THRESHOLD_NGpL,
        "target_positive_pct": round(float(target.mean()) * 100, 2),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "models": results,
        "python_version": sys.version.split()[0],
        "acceptance_criteria": {
            "f1_target": 0.80,
            "auc_target": 0.85,
        },
    }

    out_path = out_dir / "exp008_python_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Records:    {len(df)}")
    print(f"  Samples:    {len(features)} (with valid coordinates)")
    print(f"  Features:   {len(feature_cols)}")
    print(f"  Threshold:  {EPA_ADVISORY_THRESHOLD_NGpL} ng/L (PFOA+PFOS combined)")
    print(f"  Positive:   {target.sum()} ({target.mean() * 100:.1f}%)")
    for name, m in results.items():
        status = "PASS" if m["f1_score"] >= 0.80 else "BELOW TARGET"
        print(f"  {name}: F1={m['f1_score']:.4f}, AUC={m['auc_roc']:.4f} [{status}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
