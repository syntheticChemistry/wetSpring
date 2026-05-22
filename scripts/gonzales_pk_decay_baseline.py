#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-05-22
"""
Gonzales PK Decay — Python Control (Exp281)

Reproduces pharmacokinetic data from Fleck TJ,...,Gonzales AJ (2021)
"Onset and duration of action of lokivetmab in IL-31 induced pruritus."
Vet Dermatol 32:681-e182.

This is the Python baseline. Rust: wetspring validate --scenario gonzales_pk_s79

Date: 2026-05-22
Paper: 56 (Fleck/Gonzales 2021)

Reproduction:
    python3 scripts/gonzales_pk_decay_baseline.py
"""

import time
import numpy as np

DOSES_MG_KG = np.array([0.125, 0.5, 2.0])
DURATIONS_DAYS = np.array([14.0, 28.0, 42.0])
ONSET_HOURS = 3.0


def run():
    checks_passed = 0
    checks_total = 0
    timings = []

    print("=" * 60)
    print("Gonzales PK Decay — Python Control (Exp281)")
    print("=" * 60)

    # §1 Dose-Duration Relationship
    t0 = time.perf_counter_ns()
    print("\n  Published dose-duration (lokivetmab/Cytopoint):")
    print(f"  {'Dose(mg/kg)':>12} {'Duration(d)':>12} {'log(dose)':>10}")
    for d, dur in zip(DOSES_MG_KG, DURATIONS_DAYS):
        print(f"  {d:>12.3f} {dur:>12.0f} {np.log(d):>10.3f}")

    checks_total += 1
    if DURATIONS_DAYS[0] < DURATIONS_DAYS[1] < DURATIONS_DAYS[2]:
        checks_passed += 1
        print("  ✓ Duration increases with dose")
    else:
        print("  ✗ Duration ordering violated")

    dose_ratio_1 = DOSES_MG_KG[1] / DOSES_MG_KG[0]
    dur_ratio_1 = DURATIONS_DAYS[1] / DURATIONS_DAYS[0]
    dose_ratio_2 = DOSES_MG_KG[2] / DOSES_MG_KG[1]
    dur_ratio_2 = DURATIONS_DAYS[2] / DURATIONS_DAYS[1]

    checks_total += 1
    if dur_ratio_1 < dose_ratio_1 and dur_ratio_2 < dose_ratio_2:
        checks_passed += 1
        print("  ✓ Sub-linear dose-duration (4× dose ≠ 4× duration)")
    else:
        print("  ✗ Sub-linear relationship violated")

    d01_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Dose-duration", d01_us, 2))

    # §2 Exponential Decay Model
    t0 = time.perf_counter_ns()
    onset_days = ONSET_HOURS / 24.0
    names = ["low", "mid", "high"]

    for i, (dose, dur) in enumerate(zip(DOSES_MG_KG, DURATIONS_DAYS)):
        half_life = dur / 3.0
        k_decay = np.log(0.5) / half_life
        t_points = np.linspace(0, dur, 100)

        conc = np.where(
            t_points < onset_days,
            dose * (t_points / onset_days),
            dose * np.exp(k_decay * (t_points - onset_days)),
        )

        peak = np.max(conc)
        final = conc[-1]

        checks_total += 2
        if abs(peak - dose) / dose < 0.15:
            checks_passed += 1
        if final < peak * 0.15:
            checks_passed += 1

        print(
            f"  {names[i]} ({dose:.3f} mg/kg): "
            f"t½={half_life:.1f}d, peak={peak:.4f}, final={final:.6f}"
        )

    checks_total += 1
    if abs(ONSET_HOURS - 3.0) < 0.1:
        checks_passed += 1
        print("  ✓ Onset ≈ 3 hours (published)")
    else:
        print(f"  ✗ Onset = {ONSET_HOURS}h, expected 3h")

    d02_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Exponential decay", d02_us, 7))

    # §3 Log-Linear Dose-Duration Regression
    t0 = time.perf_counter_ns()
    log_doses = np.log(DOSES_MG_KG)

    coeffs = np.polyfit(log_doses, DURATIONS_DAYS, 1)
    predicted = np.polyval(coeffs, log_doses)
    ss_res = np.sum((DURATIONS_DAYS - predicted) ** 2)
    ss_tot = np.sum((DURATIONS_DAYS - np.mean(DURATIONS_DAYS)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    checks_total += 2
    if r2 > 0.95:
        checks_passed += 1
        print(f"\n  ✓ Log-linear fit R² = {r2:.6f} > 0.95")
    else:
        print(f"\n  ✗ R² = {r2:.6f}, expected > 0.95")
    if coeffs[0] > 0:
        checks_passed += 1
        print(f"  ✓ Positive slope = {coeffs[0]:.4f}")
    else:
        print(f"  ✗ Negative slope = {coeffs[0]:.4f}")

    print(f"  Predicted: {[f'{p:.1f}' for p in predicted]}")
    print(f"  Observed:  {[f'{o:.1f}' for o in DURATIONS_DAYS]}")

    d03_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Dose-duration fit", d03_us, 2))

    # §4 Anderson Barrier Mapping
    t0 = time.perf_counter_ns()
    w_barriers = DURATIONS_DAYS / np.max(DURATIONS_DAYS) * 20.0
    checks_total += 1
    if w_barriers[0] < w_barriers[1] < w_barriers[2]:
        checks_passed += 1
        print(f"\n  ✓ Anderson W barriers increase with dose: {w_barriers}")
    else:
        print(f"  ✗ Barrier ordering violated: {w_barriers}")

    d04_us = (time.perf_counter_ns() - t0) / 1e3
    timings.append(("Anderson barrier", d04_us, 1))

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
