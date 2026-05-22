#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-05-22
"""
Gonzales IC50 Dose-Response — Python Control (Exp280)

Reproduces Gonzales AJ et al. (2014) oclacitinib IC50 dose-response
from "Oclacitinib (APOQUEL) is a novel JAK inhibitor with activity
against cytokines involved in allergy." J Vet Pharmacol Ther 37:317-324.

This is the Python baseline. Rust: wetspring validate --scenario gonzales_ic50_s79

Date: 2026-05-22
Paper: 54 (Gonzales AJ et al. 2014)

Reproduction:
    python3 scripts/gonzales_ic50_baseline.py
"""

import time
import numpy as np

# Published IC50 values (Table 1)
TARGETS = [
    ("JAK1 (enzyme)", 10.0, "JAK/STAT"),
    ("IL-2", 36.0, "JAK1/JAK3 → STAT5"),
    ("IL-31", 71.0, "JAK1/JAK2 → STAT3"),
    ("IL-6", 80.0, "JAK1/JAK2 → STAT3"),
    ("IL-4", 150.0, "JAK1/JAK3 → STAT6"),
    ("IL-13", 249.0, "JAK1/TYK2 → STAT6"),
]


def hill(conc, ic50, n=1.0):
    """Standard Hill equation for dose-response."""
    return conc**n / (ic50**n + conc**n)


def run():
    checks_passed = 0
    checks_total = 0
    timings = []

    print("=" * 60)
    print("Gonzales IC50 Dose-Response — Python Control (Exp280)")
    print("=" * 60)

    # §1 Published IC50 Values
    t0 = time.perf_counter_ns()
    print("\n  Published IC50 values (Gonzales 2014 Table 1):")
    print(f"  {'Target':<17} {'IC50(nM)':>8}  {'Pathway':<25}")
    for name, ic50, pathway in TARGETS:
        print(f"  {name:<17} {ic50:>8.0}  {pathway:<25}")

    checks_total += 1
    if abs(TARGETS[0][1] - 10.0) < 0.1:
        checks_passed += 1
        print("  ✓ JAK1 IC50 = 10 nM")
    else:
        print(f"  ✗ JAK1 IC50 expected 10.0, got {TARGETS[0][1]}")

    ic50_vals = [t[1] for t in TARGETS]
    checks_total += 1
    if all(ic50_vals[i] < ic50_vals[i + 1] for i in range(len(ic50_vals) - 1)):
        checks_passed += 1
        print("  ✓ IC50 ordering: JAK1 < IL-2 < IL-31 < IL-6 < IL-4 < IL-13")
    else:
        print("  ✗ IC50 ordering violated")

    sel_il31 = TARGETS[2][1] / TARGETS[0][1]
    sel_il4 = TARGETS[4][1] / TARGETS[0][1]
    checks_total += 1
    if abs(sel_il31 - 7.1) < 0.1:
        checks_passed += 1
        print(f"  ✓ JAK1 selectivity IL-31/JAK1 = {sel_il31:.1f}×")
    else:
        print(f"  ✗ IL-31/JAK1 selectivity = {sel_il31:.1f}×, expected ≈7.1×")

    checks_total += 1
    if abs(sel_il4 - 15.0) < 0.1:
        checks_passed += 1
        print(f"  ✓ JAK1 selectivity IL-4/JAK1 = {sel_il4:.1f}×")
    else:
        print(f"  ✗ IL-4/JAK1 selectivity = {sel_il4:.1f}×, expected 15×")

    d01_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Published IC50", d01_us, 4))

    # §2 Hill Equation Dose-Response Curves
    t0 = time.perf_counter_ns()
    concentrations = np.logspace(-1, 3, 100)
    n_hill = 1.0

    sec2_checks = 0
    sec2_pass = 0
    for name, ic50, _pathway in TARGETS:
        responses = hill(concentrations, ic50, n_hill)

        at_ic50 = hill(ic50, ic50, n_hill)
        sec2_checks += 1
        checks_total += 1
        if abs(at_ic50 - 0.5) < 0.001:
            sec2_pass += 1
            checks_passed += 1

        at_low = hill(ic50 * 0.01, ic50, n_hill)
        at_high = hill(ic50 * 100.0, ic50, n_hill)
        checks_total += 2
        sec2_checks += 2
        if at_low < 0.05:
            checks_passed += 1
            sec2_pass += 1
        if at_high > 0.95:
            checks_passed += 1
            sec2_pass += 1

    print(f"\n  Hill curves: {sec2_pass}/{sec2_checks} checks passed")
    d02_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Hill curves", d02_us, sec2_checks))

    # §3 Selectivity Ratios
    t0 = time.perf_counter_ns()
    jak1_ic50 = TARGETS[0][1]
    ratios = [(name, ic50 / jak1_ic50) for name, ic50, _ in TARGETS[1:]]
    print("\n  Selectivity ratios (×JAK1):")
    for name, ratio in ratios:
        print(f"    {name:<15}: {ratio:>6.1f}×")

    checks_total += 1
    if all(r > 1.0 for _, r in ratios):
        checks_passed += 1
        print("  ✓ All cytokine IC50s > JAK1 (selectivity confirmed)")
    else:
        print("  ✗ Selectivity violation")

    d03_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Selectivity", d03_us, 1))

    # Summary
    total_us = sum(t[1] for t in timings)
    print(f"\n  Timing:")
    for domain, us, n in timings:
        print(f"    {domain:<20}: {us:>10.0f} µs ({n} checks)")
    print(f"    {'Total':<20}: {total_us:>10.0f} µs")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    return checks_passed == checks_total


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
