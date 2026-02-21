// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Hsueh/Severin 2022 phage defense deaminase (Exp030).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline tool | hsueh2022_phage_defense.py |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 scripts/hsueh2022_phage_defense.py |
//! | Baseline date | 2026-02-19 |
//! | Data | phage-bacteria ODE scenarios |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_barracuda::bio::ode::steady_state_mean;
use wetspring_barracuda::bio::phage_defense::{
    scenario_high_cost, scenario_no_phage, scenario_phage_attack, scenario_pure_defended,
    scenario_pure_undefended, PhageDefenseParams,
};
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS: f64 = 0.1;

fn main() {
    let mut v = Validator::new("Exp030: Hsueh 2022 Phage Defense Deaminase");
    let params = PhageDefenseParams::default();

    v.section("── No phage (cost-of-defense) ──");
    let r = scenario_no_phage(&params, DT);
    let bd = steady_state_mean(&r, 0, SS);
    let bu = steady_state_mean(&r, 1, SS);
    v.check(
        "No phage: Bu > Bd (no cost advantage)",
        f64::from(u8::from(bu > bd)),
        1.0,
        0.0,
    );
    v.check("No phage: Bd matches Python", bd, 132_242.0, 1000.0);
    v.check("No phage: Bu matches Python", bu, 138_317.0, 1000.0);

    v.section("── Phage attack (defense advantage) ──");
    let r = scenario_phage_attack(&params, DT);
    let bd = steady_state_mean(&r, 0, SS);
    let bu = steady_state_mean(&r, 1, SS);
    v.check(
        "Attack: Bd > Bu (defense wins)",
        f64::from(u8::from(bd > bu)),
        1.0,
        0.0,
    );
    v.check("Attack: Bu ≈ 0 (crashed)", bu, 0.0, 1.0);
    v.check("Attack: Bd matches Python", bd, 278.0, 10.0);

    v.section("── Pure defended vs pure undefended ──");
    let r_def = scenario_pure_defended(&params, DT);
    let r_undef = scenario_pure_undefended(&params, DT);
    let defended_ss = steady_state_mean(&r_def, 0, SS);
    let undefended_ss = steady_state_mean(&r_undef, 1, SS);
    v.check(
        "Defended survives better",
        f64::from(u8::from(defended_ss > undefended_ss)),
        1.0,
        0.0,
    );
    v.check(
        "Pure defended: Bd matches Python",
        defended_ss,
        119_563.0,
        1000.0,
    );

    v.section("── High cost defense ──");
    let r = scenario_high_cost(&params, DT);
    let bd_hc = steady_state_mean(&r, 0, SS);
    v.check(
        "High cost: Bd still persists",
        f64::from(u8::from(bd_hc > 0.0)),
        1.0,
        0.0,
    );

    v.section("── Non-negativity ──");
    for result in [
        scenario_no_phage(&params, DT),
        scenario_phage_attack(&params, DT),
    ] {
        let min: f64 = result
            .y
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        v.check("All vars ≥ 0", min.max(0.0), min.max(0.0), 0.0);
    }

    v.section("── Determinism ──");
    let r1 = scenario_phage_attack(&params, DT);
    let r2 = scenario_phage_attack(&params, DT);
    let max_diff: f64 = r1
        .y_final
        .iter()
        .zip(&r2.y_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check("Deterministic", max_diff, 0.0, 0.0);

    v.finish();
}
