// SPDX-License-Identifier: AGPL-3.0-or-later
//! `OdeSystem` trait implementations for `ToadStool`'s `BatchedOdeRK4<S>`.
//!
//! **Lean phase complete**: Formally connects `wetSpring`'s 5 ODE systems to the
//! upstream `barracuda::numerical::BatchedOdeRK4` generic framework absorbed in
//! `ToadStool` S51. Each implementation provides both a WGSL derivative function
//! (for GPU shader generation via `generate_shader()`) and a CPU derivative (for
//! `integrate_cpu()`).
//!
//! The 5 local WGSL shader files have been deleted — all GPU modules now use
//! `BatchedOdeRK4::<S>::generate_shader()` at runtime. Per-variable clamping that
//! was previously in the hand-written shaders is handled by derivative-level guards
//! (`fmax_d` in WGSL, `.max(0.0)` in CPU) which prevent negative concentrations.

#![allow(clippy::needless_raw_string_hashes)]

use barracuda::numerical::ode_generic::OdeSystem;

use super::bistable::{self, BistableParams};
use super::capacitor::{self, CapacitorParams};
use super::cooperation::{self, CooperationParams};
use super::multi_signal::{self, MultiSignalParams};
use super::phage_defense::{self, PhageDefenseParams};

// ---------------------------------------------------------------------------
// Capacitor ODE — Mhatre et al. 2020 (6 vars, 16 params)
// ---------------------------------------------------------------------------

/// Phenotypic capacitor ODE for `BatchedOdeRK4<CapacitorOde>`.
pub struct CapacitorOde;

impl OdeSystem for CapacitorOde {
    const N_VARS: usize = capacitor::N_VARS;
    const N_PARAMS: usize = capacitor::N_PARAMS;

    fn system_name() -> &'static str {
        "capacitor"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fpow_d(base: f64, exp_val: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(exp_val * log(base));
}
fn hill_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let xn = fpow_d(x, n);
    let kn = fpow_d(k, n);
    return xn / (kn + xn + 1e-30);
}
fn deriv(state: array<f64, 6>, params: array<f64, 16>, t: f64) -> array<f64, 6> {
    let mu_max   = params[0];  let k_cap   = params[1];  let death    = params[2];
    let k_cdg    = params[3];  let d_cdg   = params[4];  let k_charge = params[5];
    let k_disch  = params[6];  let n_vpsr  = params[7];  let k_vcdg   = params[8];
    let w_bio    = params[9];  let w_mot   = params[10]; let w_rug    = params[11];
    let d_bio    = params[12]; let d_mot   = params[13]; let d_rug    = params[14];
    let stress   = params[15];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let cell = fmax_d(state[0], z); let cdg  = fmax_d(state[1], z);
    let vpsr = fmax_d(state[2], z); let bio  = fmax_d(state[3], z);
    let mot  = fmax_d(state[4], z); let rug  = fmax_d(state[5], z);

    var dy: array<f64, 6>;
    dy[0] = mu_max * cell * (one - cell / (k_cap + 1e-30)) - death * cell;
    dy[1] = stress * k_cdg * cell - d_cdg * cdg;
    let charge   = k_charge * hill_d(cdg, k_vcdg, n_vpsr) * (one - vpsr);
    let discharge = k_disch * vpsr;
    dy[2] = charge - discharge;
    dy[3] = w_bio * vpsr * (one - bio) - d_bio * bio;
    dy[4] = w_mot * (one - vpsr) * (one - mot) - d_mot * mot;
    dy[5] = w_rug * vpsr * vpsr * (one - rug) - d_rug * rug;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = CapacitorParams::from_flat(params);
        let cell = state[0].max(0.0);
        let cdg = state[1].max(0.0);
        let vpsr = state[2].max(0.0);
        let bio = state[3].max(0.0);
        let mot = state[4].max(0.0);
        let rug = state[5].max(0.0);

        let hill_v = {
            if cdg <= 0.0 {
                0.0
            } else {
                let xn = cdg.powf(p.n_vpsr);
                xn / (p.k_vpsr_cdg.powf(p.n_vpsr) + xn)
            }
        };

        let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));
        let d_cdg = (p.stress_factor * p.k_cdg_prod).mul_add(cell, -(p.d_cdg * cdg));
        let charge = p.k_vpsr_charge * hill_v * (1.0 - vpsr);
        let discharge = p.k_vpsr_discharge * vpsr;
        let d_vpsr = charge - discharge;
        let d_bio = (p.w_biofilm * vpsr).mul_add(1.0 - bio, -(p.d_bio * bio));
        let d_mot = (p.w_motility * (1.0 - vpsr)).mul_add(1.0 - mot, -(p.d_mot * mot));
        let d_rug = (p.w_rugose * vpsr * vpsr).mul_add(1.0 - rug, -(p.d_rug * rug));

        vec![d_cell, d_cdg, d_vpsr, d_bio, d_mot, d_rug]
    }
}

// ---------------------------------------------------------------------------
// Cooperation ODE — Bruger & Waters 2018 (4 vars, 13 params)
// ---------------------------------------------------------------------------

/// Cooperative QS game-theory ODE for `BatchedOdeRK4<CooperationOde>`.
pub struct CooperationOde;

impl OdeSystem for CooperationOde {
    const N_VARS: usize = cooperation::N_VARS;
    const N_PARAMS: usize = cooperation::N_PARAMS;

    fn system_name() -> &'static str {
        "cooperation"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn hill2_d(x: f64, k: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let x2 = x * x;
    return x2 / (k * k + x2);
}
fn deriv(state: array<f64, 4>, params: array<f64, 13>, t: f64) -> array<f64, 4> {
    let mu_coop  = params[0];  let mu_cheat = params[1];  let k_cap    = params[2];
    let death    = params[3];  let k_ai     = params[4];  let d_ai     = params[5];
    let benefit  = params[6];  let k_ben    = params[7];  let cost     = params[8];
    let k_bio    = params[9];  let k_bio_ai = params[10]; let disp     = params[11];
    let d_bio    = params[12];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let nc  = fmax_d(state[0], z); let nd  = fmax_d(state[1], z);
    let ai  = fmax_d(state[2], z); let bio = fmax_d(state[3], z);

    let n_total  = nc + nd;
    let crowding = fmax_d(one - n_total / (k_cap + 1e-30), z);
    let sig_benefit = benefit * hill2_d(ai, k_ben);
    let dispersal   = disp * (one - bio);
    let fit_coop  = (mu_coop  - cost + sig_benefit + dispersal) * crowding;
    let fit_cheat = (mu_cheat + sig_benefit + dispersal) * crowding;

    var dy: array<f64, 4>;
    dy[0] = fit_coop  * nc - death * nc;
    dy[1] = fit_cheat * nd - death * nd;
    dy[2] = k_ai * nc - d_ai * ai;
    dy[3] = k_bio * hill2_d(ai, k_bio_ai) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = CooperationParams::from_flat(params);
        let nc = state[0].max(0.0);
        let nd = state[1].max(0.0);
        let ai = state[2].max(0.0);
        let bio = state[3].max(0.0);

        let hill2 = |x: f64, k: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let x2 = x * x;
            x2 / k.mul_add(k, x2)
        };

        let n_total = nc + nd;
        let crowding = (1.0 - n_total / p.k_cap).max(0.0);
        let signal_benefit = p.benefit * hill2(ai, p.k_benefit);
        let dispersal = p.dispersal_bonus * (1.0 - bio);

        let fitness_coop = (p.mu_coop - p.cost + signal_benefit + dispersal) * crowding;
        let fitness_cheat = (p.mu_cheat + signal_benefit + dispersal) * crowding;

        vec![
            fitness_coop.mul_add(nc, -(p.death_rate * nc)),
            fitness_cheat.mul_add(nd, -(p.death_rate * nd)),
            p.k_ai_prod.mul_add(nc, -p.d_ai * ai),
            (p.k_bio * hill2(ai, p.k_bio_ai)).mul_add(1.0 - bio, -(p.d_bio * bio)),
        ]
    }
}

// ---------------------------------------------------------------------------
// Multi-Signal ODE — Srivastava et al. 2011 (7 vars, 24 params)
// ---------------------------------------------------------------------------

/// Dual-signal QS regulatory network for `BatchedOdeRK4<MultiSignalOde>`.
pub struct MultiSignalOde;

impl OdeSystem for MultiSignalOde {
    const N_VARS: usize = multi_signal::N_VARS;
    const N_PARAMS: usize = multi_signal::N_PARAMS;

    fn system_name() -> &'static str {
        "multi_signal"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fpow_d(base: f64, e: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(e * log(base));
}
fn hill_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax_d(x, z);
    let xn = fpow_d(xc, n);
    let kn = fpow_d(fmax_d(k, z + 1e-30), n);
    return xn / (kn + xn);
}
fn hill_repress_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax_d(x, z);
    let kn = fpow_d(fmax_d(k, z + 1e-30), n);
    let xn = fpow_d(xc, n);
    return kn / (kn + xn);
}
fn deriv(state: array<f64, 7>, params: array<f64, 24>, t: f64) -> array<f64, 7> {
    let mu_max     = params[0];  let k_cap      = params[1];  let death_rate = params[2];
    let k_cai1     = params[3];  let d_cai1     = params[4];  let k_cqs      = params[5];
    let k_ai2      = params[6];  let d_ai2      = params[7];  let k_luxpq    = params[8];
    let k_luxo_p   = params[9];  let d_luxo_p   = params[10]; let k_hapr_max = params[11];
    let n_repress  = params[12]; let k_repress  = params[13]; let d_hapr     = params[14];
    let k_dgc_bas  = params[15]; let k_dgc_rep  = params[16]; let k_pde_bas  = params[17];
    let k_pde_act  = params[18]; let d_cdg      = params[19]; let k_bio_max  = params[20];
    let k_bio_cdg  = params[21]; let n_bio      = params[22]; let d_bio      = params[23];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let two = z + 2.0;
    let cell   = fmax_d(state[0], z); let cai1   = fmax_d(state[1], z);
    let ai2    = fmax_d(state[2], z); let luxo_p = fmax_d(state[3], z);
    let hapr   = fmax_d(state[4], z); let cdg    = fmax_d(state[5], z);
    let bio    = fmax_d(state[6], z);

    var dy: array<f64, 7>;
    dy[0] = mu_max * cell * (one - cell / fmax_d(k_cap, z + 1e-30)) - death_rate * cell;
    dy[1] = k_cai1 * cell - d_cai1 * cai1;
    dy[2] = k_ai2 * cell - d_ai2 * ai2;
    let dephos_cai1 = hill_d(cai1, k_cqs, two);
    let dephos_ai2 = hill_d(ai2, k_luxpq, two);
    dy[3] = k_luxo_p - (d_luxo_p + dephos_cai1 + dephos_ai2) * luxo_p;
    dy[4] = k_hapr_max * hill_repress_d(luxo_p, k_repress, n_repress) - d_hapr * hapr;
    let dgc_rate = k_dgc_bas * fmax_d(one - k_dgc_rep * hapr, z);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[5] = dgc_rate - pde_rate * cdg - d_cdg * cdg;
    dy[6] = k_bio_max * hill_d(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = MultiSignalParams::from_flat(params);
        let cell = state[0].max(0.0);
        let cai1 = state[1].max(0.0);
        let ai2 = state[2].max(0.0);
        let luxo_p = state[3].max(0.0);
        let hapr = state[4].max(0.0);
        let cdg = state[5].max(0.0);
        let bio = state[6].max(0.0);

        let hill = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let xn = x.powf(n);
            xn / (k.powf(n) + xn)
        };
        let hill_repress = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 1.0;
            }
            let kn = k.powf(n);
            kn / (kn + x.powf(n))
        };

        let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));
        let d_cai1 = p.k_cai1_prod.mul_add(cell, -p.d_cai1 * cai1);
        let d_ai2 = p.k_ai2_prod.mul_add(cell, -p.d_ai2 * ai2);

        let dephos_cai1 = hill(cai1, p.k_cqs, 2.0);
        let dephos_ai2 = hill(ai2, p.k_luxpq, 2.0);
        let d_luxo_p = (p.d_luxo_p + dephos_cai1 + dephos_ai2).mul_add(-luxo_p, p.k_luxo_phos);

        let d_hapr = p.k_hapr_max.mul_add(
            hill_repress(luxo_p, p.k_repress, p.n_repress),
            -p.d_hapr * hapr,
        );

        let dgc_rate = p.k_dgc_basal * p.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
        let pde_rate = p.k_pde_act.mul_add(hapr, p.k_pde_basal);
        let d_cdg = p.d_cdg.mul_add(-cdg, dgc_rate - pde_rate * cdg);

        let bio_promote = p.k_bio_max * hill(cdg, p.k_bio_cdg, p.n_bio);
        let d_bio = bio_promote.mul_add(1.0 - bio, -(p.d_bio * bio));

        vec![d_cell, d_cai1, d_ai2, d_luxo_p, d_hapr, d_cdg, d_bio]
    }
}

// ---------------------------------------------------------------------------
// Bistable ODE — Fernandez et al. 2020 (5 vars, 21 params)
// ---------------------------------------------------------------------------

/// Bistable phenotypic switching ODE for `BatchedOdeRK4<BistableOde>`.
pub struct BistableOde;

impl OdeSystem for BistableOde {
    const N_VARS: usize = bistable::N_VARS;
    const N_PARAMS: usize = bistable::N_PARAMS;

    fn system_name() -> &'static str {
        "bistable"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fpow_d(base: f64, e: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(e * log(base));
}
fn hill_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax_d(x, z);
    let xn = fpow_d(xc, n);
    let kn = fpow_d(fmax_d(k, z + 1e-30), n);
    return xn / (kn + xn);
}
fn deriv(state: array<f64, 5>, params: array<f64, 21>, t: f64) -> array<f64, 5> {
    let mu_max     = params[0];  let k_cap      = params[1];  let death_rate = params[2];
    let k_ai_prod  = params[3];  let d_ai       = params[4];  let k_hapr_max = params[5];
    let k_hapr_ai  = params[6];  let n_hapr     = params[7];  let d_hapr     = params[8];
    let k_dgc_bas  = params[9];  let k_dgc_rep  = params[10]; let k_pde_bas  = params[11];
    let k_pde_act  = params[12]; let d_cdg      = params[13]; let k_bio_max  = params[14];
    let k_bio_cdg  = params[15]; let n_bio      = params[16]; let d_bio      = params[17];
    let alpha_fb   = params[18]; let n_fb       = params[19]; let k_fb       = params[20];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let cell = fmax_d(state[0], z); let ai   = fmax_d(state[1], z);
    let hapr = fmax_d(state[2], z); let cdg  = fmax_d(state[3], z);
    let bio  = fmax_d(state[4], z);

    var dy: array<f64, 5>;
    dy[0] = mu_max * cell * (one - cell / fmax_d(k_cap, z + 1e-30)) - death_rate * cell;
    dy[1] = k_ai_prod * cell - d_ai * ai;
    dy[2] = k_hapr_max * hill_d(ai, k_hapr_ai, n_hapr) - d_hapr * hapr;
    let basal_dgc = k_dgc_bas * fmax_d(one - k_dgc_rep * hapr, z);
    let feedback_dgc = alpha_fb * hill_d(bio, k_fb, n_fb);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[3] = basal_dgc + feedback_dgc - pde_rate * cdg - d_cdg * cdg;
    dy[4] = k_bio_max * hill_d(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = BistableParams::from_flat(params);
        let cell = state[0].max(0.0);
        let ai = state[1].max(0.0);
        let hapr = state[2].max(0.0);
        let cdg = state[3].max(0.0);
        let bio = state[4].max(0.0);

        let b = &p.base;
        let hill = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let xn = x.powf(n);
            xn / (k.powf(n) + xn)
        };

        let d_cell = (b.mu_max * cell).mul_add(1.0 - cell / b.k_cap, -(b.death_rate * cell));
        let d_ai = b.k_ai_prod.mul_add(cell, -b.d_ai * ai);
        let d_hapr = b
            .k_hapr_max
            .mul_add(hill(ai, b.k_hapr_ai, b.n_hapr), -b.d_hapr * hapr);

        let basal_dgc = b.k_dgc_basal * b.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
        let feedback_dgc = p.alpha_fb * hill(bio, p.k_fb, p.n_fb);
        let pde_rate = b.k_pde_act.mul_add(hapr, b.k_pde_basal);
        let d_cdg = b
            .d_cdg
            .mul_add(-cdg, basal_dgc + feedback_dgc - pde_rate * cdg);

        let bio_promote = b.k_bio_max * hill(cdg, b.k_bio_cdg, b.n_bio);
        let d_bio = bio_promote.mul_add(1.0 - bio, -(b.d_bio * bio));

        vec![d_cell, d_ai, d_hapr, d_cdg, d_bio]
    }
}

// ---------------------------------------------------------------------------
// Phage Defense ODE — Hsueh, Severin et al. 2022 (4 vars, 11 params)
// ---------------------------------------------------------------------------

/// Phage-bacteria defense arms race for `BatchedOdeRK4<PhageDefenseOde>`.
pub struct PhageDefenseOde;

impl OdeSystem for PhageDefenseOde {
    const N_VARS: usize = phage_defense::N_VARS;
    const N_PARAMS: usize = phage_defense::N_PARAMS;

    fn system_name() -> &'static str {
        "phage_defense"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn monod_d(r: f64, k: f64) -> f64 {
    return r / (k + r + 1e-30);
}
fn deriv(state: array<f64, 4>, params: array<f64, 11>, t: f64) -> array<f64, 4> {
    let mu_max = params[0];  let cost   = params[1];  let k_res  = params[2];
    let yld    = params[3];  let ads    = params[4];  let burst  = params[5];
    let eff    = params[6];  let decay  = params[7];  let inflow = params[8];
    let dilut  = params[9];  let death  = params[10];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let bd    = fmax_d(state[0], z); let bu    = fmax_d(state[1], z);
    let phage = fmax_d(state[2], z); let r     = fmax_d(state[3], z);

    let growth_limit = monod_d(r, k_res);
    let mu_d = mu_max * (one - cost) * growth_limit;
    let mu_u = mu_max * growth_limit;
    let inf_d = ads * bd * phage;
    let inf_u = ads * bu * phage;
    let kill_d = inf_d * (one - eff);

    var dy: array<f64, 4>;
    dy[0] = mu_d * bd - death * bd - kill_d;
    dy[1] = mu_u * bu - death * bu - inf_u;
    dy[2] = burst * (inf_u + kill_d) - ads * (bd + bu) * phage - decay * phage;
    dy[3] = inflow - yld * (mu_d * bd + mu_u * bu) - dilut * r;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = PhageDefenseParams::from_flat(params);
        let bd = state[0].max(0.0);
        let bu = state[1].max(0.0);
        let phage = state[2].max(0.0);
        let r = state[3].max(0.0);

        let growth_limit = r / (p.k_resource + r);
        let mu_d = p.mu_max * (1.0 - p.defense_cost) * growth_limit;
        let mu_u = p.mu_max * growth_limit;
        let infection_d = p.adsorption_rate * bd * phage;
        let infection_u = p.adsorption_rate * bu * phage;
        let kill_d = infection_d * (1.0 - p.defense_efficiency);

        let growth_defended = p.death_rate.mul_add(-bd, mu_d * bd - kill_d);
        let growth_undefended = p.death_rate.mul_add(-bu, mu_u * bu - infection_u);
        let burst_from_u = p.burst_size * infection_u;
        let burst_from_d = p.burst_size * (1.0 - p.defense_efficiency) * infection_d;
        let d_phage = (p.adsorption_rate * (bd + bu)).mul_add(
            -phage,
            p.phage_decay.mul_add(-phage, burst_from_u + burst_from_d),
        );
        let consumption = p.yield_coeff * (mu_d * bd + mu_u * bu);
        let d_r = p
            .resource_dilution
            .mul_add(-r, p.resource_inflow - consumption);

        vec![growth_defended, growth_undefended, d_phage, d_r]
    }
}

// ---------------------------------------------------------------------------
// Tests — validate upstream integrate_cpu matches local ODE results
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use barracuda::numerical::ode_generic::BatchedOdeRK4;

    const DT: f64 = 0.01;
    const N_STEPS: usize = 4800;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn capacitor_upstream_cpu_matches_local() {
        let p = CapacitorParams::default();
        let y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0];
        let local = super::super::capacitor::run_capacitor(&y0, 48.0, DT, &p);

        let flat_params = p.to_flat();
        let upstream =
            BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&y0, &flat_params, DT, N_STEPS, 1)
                .expect("upstream integrate_cpu");

        let diff = max_abs_diff(&local.y_final, &upstream);
        assert!(
            diff < 0.5,
            "capacitor CPU parity: max diff {diff} (local={:?} upstream={upstream:?})",
            local.y_final
        );
    }

    #[test]
    fn cooperation_upstream_cpu_matches_local() {
        let p = CooperationParams::default();
        let y0 = [0.01, 0.01, 0.0, 0.0];
        let local = super::super::cooperation::run_cooperation(&y0, 48.0, DT, &p);

        let flat_params = p.to_flat();
        let upstream =
            BatchedOdeRK4::<CooperationOde>::integrate_cpu(&y0, &flat_params, DT, N_STEPS, 1)
                .expect("upstream integrate_cpu");

        let diff = max_abs_diff(&local.y_final, &upstream);
        assert!(
            diff < 0.5,
            "cooperation CPU parity: max diff {diff} (local={:?} upstream={upstream:?})",
            local.y_final
        );
    }

    #[test]
    fn multi_signal_upstream_cpu_matches_local() {
        let p = MultiSignalParams::default();
        let y0 = [0.01, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let local = super::super::multi_signal::run_multi_signal(&y0, 48.0, DT, &p);

        let flat_params = p.to_flat();
        let upstream =
            BatchedOdeRK4::<MultiSignalOde>::integrate_cpu(&y0, &flat_params, DT, N_STEPS, 1)
                .expect("upstream integrate_cpu");

        let diff = max_abs_diff(&local.y_final, &upstream);
        assert!(
            diff < 0.5,
            "multi_signal CPU parity: max diff {diff} (local={:?} upstream={upstream:?})",
            local.y_final
        );
    }

    #[test]
    fn bistable_upstream_cpu_matches_local() {
        let p = BistableParams::default();
        let y0 = [0.01, 0.0, 0.0, 2.0, 0.0];
        let local = super::super::bistable::run_bistable(&y0, 48.0, DT, &p);

        let flat_params = p.to_flat();
        let upstream =
            BatchedOdeRK4::<BistableOde>::integrate_cpu(&y0, &flat_params, DT, N_STEPS, 1)
                .expect("upstream integrate_cpu");

        let diff = max_abs_diff(&local.y_final, &upstream);
        assert!(
            diff < 0.5,
            "bistable CPU parity: max diff {diff} (local={:?} upstream={upstream:?})",
            local.y_final
        );
    }

    #[test]
    fn phage_defense_derivative_matches_local() {
        let p = PhageDefenseParams::default();
        let y0 = [1e6, 1e6, 1e4, 10.0];
        let flat_params = p.to_flat();

        let upstream_dy = PhageDefenseOde::cpu_derivative(0.0, &y0, &flat_params);
        let local_dy = {
            let bd = y0[0];
            let bu = y0[1];
            let phage = y0[2];
            let r = y0[3];
            let gl = r / (p.k_resource + r);
            let mu_d = p.mu_max * (1.0 - p.defense_cost) * gl;
            let mu_u = p.mu_max * gl;
            let inf_d = p.adsorption_rate * bd * phage;
            let inf_u = p.adsorption_rate * bu * phage;
            let kill_d = inf_d * (1.0 - p.defense_efficiency);
            vec![
                p.death_rate.mul_add(-bd, mu_d * bd - kill_d),
                p.death_rate.mul_add(-bu, mu_u * bu - inf_u),
                (p.adsorption_rate * (bd + bu)).mul_add(
                    -phage,
                    p.phage_decay
                        .mul_add(-phage, p.burst_size * (inf_u + kill_d)),
                ),
                p.yield_coeff.mul_add(
                    -(mu_d * bd + mu_u * bu),
                    p.resource_dilution.mul_add(-r, p.resource_inflow),
                ),
            ]
        };

        let diff = max_abs_diff(&local_dy, &upstream_dy);
        assert!(
            diff < 1.0,
            "phage_defense derivative parity: max diff {diff}\n  local={local_dy:?}\n  upstream={upstream_dy:?}"
        );
    }

    #[test]
    fn phage_defense_short_integration_no_clamp_divergence() {
        let p = PhageDefenseParams {
            resource_inflow: 100.0,
            ..PhageDefenseParams::default()
        };
        let y0 = [1e4, 1e4, 0.0, 50.0];
        let flat_params = p.to_flat();
        let steps = 500;
        let upstream =
            BatchedOdeRK4::<PhageDefenseOde>::integrate_cpu(&y0, &flat_params, DT, steps, 1)
                .expect("upstream integrate_cpu");

        assert!(
            upstream.iter().all(|x| x.is_finite()),
            "phage_defense short integration should stay finite: {upstream:?}"
        );
    }

    #[test]
    fn capacitor_batched_integration() {
        let p1 = CapacitorParams::default();
        let p2 = CapacitorParams {
            stress_factor: 3.0,
            ..CapacitorParams::default()
        };
        let y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0];
        let mut states = Vec::with_capacity(12);
        states.extend_from_slice(&y0);
        states.extend_from_slice(&y0);
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&p1.to_flat());
        params.extend_from_slice(&p2.to_flat());

        let result = BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&states, &params, DT, N_STEPS, 2)
            .expect("batched integrate");

        assert_eq!(result.len(), 12);
        assert!(result[0..6].iter().all(|x| x.is_finite()));
        assert!(result[6..12].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn all_systems_generate_valid_wgsl() {
        let cap = BatchedOdeRK4::<CapacitorOde>::generate_shader();
        let coop = BatchedOdeRK4::<CooperationOde>::generate_shader();
        let ms = BatchedOdeRK4::<MultiSignalOde>::generate_shader();
        let bi = BatchedOdeRK4::<BistableOde>::generate_shader();
        let pd = BatchedOdeRK4::<PhageDefenseOde>::generate_shader();

        for (name, shader) in [
            ("capacitor", &cap),
            ("cooperation", &coop),
            ("multi_signal", &ms),
            ("bistable", &bi),
            ("phage_defense", &pd),
        ] {
            assert!(
                shader.contains("fn deriv"),
                "{name}: missing deriv function"
            );
            assert!(shader.contains("fn rk4_step"), "{name}: missing rk4_step");
            assert!(
                shader.contains("@compute"),
                "{name}: missing compute entry point"
            );
        }
    }
}
