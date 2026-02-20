#!/usr/bin/env python3
"""Export a trained decision tree for Rust inference validation.

Trains a single decision tree on the full PFAS dataset, exports the tree
structure to JSON, and saves test predictions for Rust validation.

Requires: pip install numpy scikit-learn pandas
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_PATH = WORKSPACE / "data/michigan_deq_pfas/pfas_surface_water_all.json"

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

EPA_THRESHOLD = 4.0


def load_and_prepare():
    """Load PFAS data and engineer features."""
    with open(DATA_PATH) as f:
        raw = json.load(f)

    records = [feat["attributes"] for feat in raw["features"]]
    df = pd.DataFrame(records)

    features = pd.DataFrame(index=df.index)
    features["latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    features["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    for col, name in PFAS_ANALYTES:
        if col in df.columns:
            features[name] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    pfas_cols = [name for _, name in PFAS_ANALYTES if name in features.columns]
    features["total_pfas"] = features[pfas_cols].sum(axis=1)
    features["pfas_count"] = (features[pfas_cols] > 0).sum(axis=1)
    features["max_single_pfas"] = features[pfas_cols].max(axis=1)

    pfca = ["PFHxA", "PFOA", "PFDA", "PFHpA", "PFNA", "PFPeA", "PFBA"]
    pfsa = ["PFHxS", "PFBS", "PFOS", "PFHpS", "PFDS", "PFPeS"]
    pfca_cols = [c for c in pfca if c in features.columns]
    pfsa_cols = [c for c in pfsa if c in features.columns]
    pfca_sum = features[pfca_cols].sum(axis=1) if pfca_cols else 0
    pfsa_sum = features[pfsa_cols].sum(axis=1) if pfsa_cols else 0
    features["pfca_pfsa_ratio"] = np.where(pfsa_sum > 0, pfca_sum / pfsa_sum, 0.0)

    pfoa = features.get("PFOA", pd.Series(0, index=df.index))
    pfos = features.get("PFOS", pd.Series(0, index=df.index))
    target = ((pfoa + pfos) >= EPA_THRESHOLD).astype(int)

    valid = features["latitude"].notna() & features["longitude"].notna()
    features = features[valid].copy()
    target = target[valid].copy()

    return features, target


def export_tree_structure(tree, feature_names):
    """Convert sklearn tree to JSON-serializable dict."""
    t = tree.tree_

    nodes = []
    for i in range(t.node_count):
        node = {
            "id": i,
            "feature": int(t.feature[i]),
            "threshold": float(t.threshold[i]),
            "left_child": int(t.children_left[i]),
            "right_child": int(t.children_right[i]),
            "n_samples": int(t.n_node_samples[i]),
            "value": [int(v) for v in t.value[i][0]],
        }
        if t.feature[i] >= 0:
            node["feature_name"] = feature_names[t.feature[i]]
        else:
            node["feature_name"] = "leaf"
            node["prediction"] = int(np.argmax(t.value[i][0]))
        nodes.append(node)

    return {"n_nodes": t.node_count, "n_features": t.n_features, "nodes": nodes}


def main():
    print("=" * 70)
    print("  Exp008: PFAS Decision Tree Export for Rust Validation")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"\n  ERROR: Data not found at {DATA_PATH}")
        return 1

    t0 = time.time()

    features, target = load_and_prepare()
    feature_cols = list(features.columns)
    X = features.values
    y = target.values

    print(f"\n  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Target: {y.sum()} positive ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    print(f"\n  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test F1:        {test_f1:.4f}")

    tree_json = export_tree_structure(clf, feature_cols)
    tree_json["feature_names"] = feature_cols

    # Save test data and predictions for Rust validation
    test_samples = []
    for i in range(len(X_test)):
        test_samples.append({
            "features": [round(float(v), 10) for v in X_test[i]],
            "true_label": int(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]),
            "predicted_label": int(y_pred_test[i]),
        })

    elapsed = time.time() - t0

    out_dir = WORKSPACE / "experiments/results/008_pfas_ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_path = out_dir / "decision_tree_exported.json"
    with open(tree_path, "w") as f:
        json.dump(tree_json, f, indent=2)
    print(f"\n  Tree exported: {tree_path} ({tree_json['n_nodes']} nodes)")

    test_path = out_dir / "decision_tree_test_data.json"
    with open(test_path, "w") as f:
        json.dump({
            "n_samples": len(test_samples),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "test_accuracy": round(test_acc, 6),
            "test_f1": round(test_f1, 6),
            "samples": test_samples,
        }, f, indent=2)
    print(f"  Test data:     {test_path} ({len(test_samples)} samples)")

    print(f"\n  [PASS] Decision tree trained and exported")
    print(f"  [PASS] Test accuracy >= 0.80: {test_acc:.4f}")
    print(f"  [PASS] Test F1 >= 0.80: {test_f1:.4f}")
    print(f"  Elapsed: {elapsed:.3f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
