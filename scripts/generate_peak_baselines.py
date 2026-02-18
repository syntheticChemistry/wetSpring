#!/usr/bin/env python3
"""Generate scipy.signal.find_peaks baselines for Exp010.

Produces per-case .dat files and a summary JSON for provenance.
The .dat files use a simple text format readable without a JSON parser:
  Line 1: space-separated data values
  Line 2: space-separated peak indices
  Line 3: space-separated peak heights
  Line 4: space-separated prominences

Requires: pip install scipy numpy
"""
import json
import os
import numpy as np
from scipy.signal import find_peaks

OUTDIR = "experiments/results/010_peak_baselines"


def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def write_case(name, data, params, outdir):
    """Run find_peaks and write results to .dat file."""
    sp_params = {}
    if "min_height" in params:
        sp_params["height"] = params["min_height"]
    if "min_prominence" in params:
        sp_params["prominence"] = params["min_prominence"]
    if "min_width" in params:
        sp_params["width"] = params["min_width"]
    if "distance" in params:
        sp_params["distance"] = params["distance"]

    indices, props = find_peaks(np.array(data), **sp_params)

    heights = [float(data[i]) for i in indices]
    prominences = [float(v) for v in props.get("prominences", [])]

    path = os.path.join(outdir, f"{name}.dat")
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in data) + "\n")
        f.write(" ".join(str(i) for i in indices) + "\n")
        f.write(" ".join(f"{h:.8f}" for h in heights) + "\n")
        f.write(" ".join(f"{p:.8f}" for p in prominences) + "\n")

    return {
        "name": name,
        "n_data": len(data),
        "n_peaks": len(indices),
        "params": params,
        "indices": indices.tolist(),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    summaries = []

    # Case 1: Single Gaussian peak
    rng = np.random.default_rng(42)
    x = np.arange(200, dtype=np.float64)
    y = gaussian(x, 100, 10, 1000.0) + rng.normal(0, 5, 200)
    y = np.clip(y, 0, None)
    summaries.append(write_case(
        "single_gaussian", y.tolist(),
        {"min_height": 50.0, "min_prominence": 50.0, "min_width": 2.0}, OUTDIR))

    # Case 2: Three LC-MS-like peaks
    rng2 = np.random.default_rng(123)
    x = np.arange(500, dtype=np.float64)
    y = (gaussian(x, 80, 8, 5000.0)
         + gaussian(x, 250, 15, 3000.0)
         + gaussian(x, 400, 5, 8000.0)
         + rng2.normal(0, 20, 500))
    y = np.clip(y, 0, None)
    summaries.append(write_case(
        "three_chromatographic", y.tolist(),
        {"min_height": 100.0, "min_prominence": 200.0, "min_width": 2.0}, OUTDIR))

    # Case 3: Noisy baseline with spikes
    rng3 = np.random.default_rng(7)
    y = rng3.normal(100, 30, 1000).astype(np.float64)
    y = np.clip(y, 0, None)
    for pos in [150, 500, 800]:
        y[pos - 3:pos + 4] += gaussian(np.arange(7), 3, 1.5, 500)
    summaries.append(write_case(
        "noisy_with_spikes", y.tolist(),
        {"min_height": 300.0, "min_prominence": 200.0, "distance": 20}, OUTDIR))

    # Case 4: Overlapping peaks
    x = np.arange(200, dtype=np.float64)
    y = gaussian(x, 90, 12, 2000.0) + gaussian(x, 110, 12, 2500.0)
    summaries.append(write_case(
        "overlapping_peaks", y.tolist(),
        {"min_prominence": 100.0}, OUTDIR))

    # Case 5: Monotonic (no peaks)
    y = np.linspace(0, 1000, 100).tolist()
    summaries.append(write_case(
        "monotonic_no_peaks", y, {}, OUTDIR))

    with open(os.path.join(OUTDIR, "scipy_baselines.json"), "w") as f:
        json.dump({"generator": "scipy " + __import__("scipy").__version__,
                    "cases": summaries}, f, indent=2)

    print(f"Wrote {len(summaries)} test cases to {OUTDIR}/")
    for s in summaries:
        print(f"  {s['name']}: {s['n_data']} points, {s['n_peaks']} peaks")


if __name__ == "__main__":
    main()
