// SPDX-License-Identifier: AGPL-3.0-or-later
//! ODE / dynamical system tolerances.

/// ODE convergence epsilon for c-di-GMP concentration checks.
///
/// Used by QS biofilm, bistable, and multi-signal ODE models to
/// determine when a species concentration has effectively reached zero.
pub const ODE_CDG_CONVERGENCE: f64 = 1e-12;

/// ODE steady-state check: ±0.01 for species concentrations at equilibrium.
///
/// Used by Waters 2008 QS, Mhatre 2020 capacitor, Bruger 2018 cooperation,
/// and Fernandez 2020 bistable ODE models. RK4 vs `scipy.integrate` at
/// identical dt accumulates ~1e-3 difference for multi-species systems.
/// Calibrated: Exp020/023/025/027 (all ODE models).
/// Scripts: `waters2008_qs_ode.py`, `fernandez2020_bistable.py`,
/// `bruger2018_cooperation.py`, `mhatre2020_capacitor.py`.
/// Commit `48fb787`.
pub const ODE_STEADY_STATE: f64 = 0.01;

/// ODE total density at carrying capacity (Bruger 2018 cooperation).
///
/// Steady-state sum Nc + Nd should approach `K_cap`. RK4 accumulation over
/// long integration allows ~0.1 drift from the theoretical equilibrium.
/// Validated: `cooperation` `total_density_reaches_capacity`.
pub const ODE_TOTAL_DENSITY_CAPACITY: f64 = 0.1;

/// ODE method parity: RK4 vs LSODA integrator differences.
///
/// Fixed-step RK4 diverges from adaptive LSODA by up to ~1e-3 in
/// concentration for stiff systems.
/// Calibrated: Exp020/023/024/025/027/030 (all 6 ODE models).
/// Scripts: `waters2008_qs_ode.py`, `srivastava2011_multi_signal.py`,
/// `hsueh2022_phage_defense.py`. Commit `48fb787`.
pub const ODE_METHOD_PARITY: f64 = 1e-3;

/// ODE GPU vs CPU parity: same RK4, different instruction ordering.
///
/// GPU WGSL RK4 vs CPU Rust RK4 at identical dt. Both use f64; GPU
/// instruction reordering and FMA behavior yield ~1e-6 drift.
pub const ODE_GPU_PARITY: f64 = 1e-6;

/// Near-zero species concentrations (repressed pathways).
///
/// Biologically "off" species may float slightly above zero due to
/// integrator residual. 0.05 accommodates the numerical floor.
pub const ODE_NEAR_ZERO: f64 = 0.05;

/// GPU ODE sweep absolute parity: max `|CPU - GPU|` over long-horizon
/// integration (1000+ RK4 steps, 128 parameter batches).
///
/// After 1000 steps of RK4 at dt = 0.001, the GPU f64 and CPU f64
/// integrators diverge by up to 0.12 absolute in species concentrations
/// due to instruction reordering and FMA differences. 0.15 covers the
/// worst observed case with 20% margin.
/// Validated: Exp049 (GPU ODE sweep), commit `e4358c5`.
pub const ODE_GPU_SWEEP_ABS: f64 = 0.15;

/// Relative tolerance for near-zero ODE variables (GPU vs CPU).
///
/// When ODE variables are near zero (repressed pathways, depleted species),
/// the *relative* error `|gpu - cpu| / max(|gpu|, |cpu|)` can be large
/// even though the *absolute* difference is negligible. 1.5 (150%)
/// accommodates GPU/CPU integrator divergence at the numerical floor
/// where both values are biologically insignificant (< 0.01).
pub const ODE_NEAR_ZERO_RELATIVE: f64 = 1.5;

/// ODE biofilm dispersed-state `B_ss` tolerance.
///
/// RK4 vs LSODA for biofilm concentration at near-zero steady states.
/// `B_ss` ≈ 0.02–0.10 depending on scenario; 0.03 covers integrator drift
/// between fixed-step RK4 and adaptive LSODA at dt=0.001.
/// Validated: Exp020 (Waters 2008), `scripts/waters2008_qs_ode.py`.
pub const ODE_BIOFILM_SS: f64 = 0.03;

/// Bistable ODE low-biofilm steady-state tolerance.
///
/// For `B_ss` ≈ 0.040 in the zero-feedback scenario, RK4 vs LSODA
/// differ by ~0.005. Tighter than [`ODE_STEADY_STATE`] (0.01) because
/// the baseline value itself is small.
/// Validated: Exp023 (Fernandez 2020), `scripts/fernandez2020_bistable.py`,
/// commit `e4358c5`.
pub const ODE_BISTABLE_LOW_B: f64 = 0.005;

/// ODE c-di-GMP / autoinducer steady-state tolerance.
///
/// For c-di-GMP (`C_ss` ≈ 1.634 in bistable) and autoinducer (`AI_ss` ≈ 1.854
/// in cooperation), RK4 vs LSODA accumulate ~0.02 difference due to
/// stiff feedback loops in the signaling cascade.
/// Validated: Exp023 (Fernandez 2020) and Exp025 (Bruger 2018),
/// `scripts/fernandez2020_bistable.py`, `scripts/bruger2018_cooperation.py`,
/// commit `e4358c5`.
pub const ODE_SIGNAL_SS: f64 = 0.02;

/// Soil recovery W(t) absolute tolerance at 40-year horizon.
///
/// Recovery model W(40yr) approaches ~4.0; RK4 vs analytical differ by
/// up to ~1.0 due to exponential decay accumulation.
/// Validated: Exp216 (`BarraCuda` CPU v13), soil recovery check.
pub const SOIL_RECOVERY_W_TOL: f64 = 1.0;

/// Bistable ODE high-biofilm attractor tolerance.
///
/// For the sessile attractor in the Fernandez 2020 bistable model,
/// `B_ss` ≈ 0.7. RK4 vs LSODA accumulate ~0.10 difference in the
/// high-biofilm state due to stiffness in the feedback loop.
/// 0.15 covers the worst observed case with 50% margin.
/// Validated: Exp079 (`BarraCuda` CPU v6), `scripts/fernandez2020_bistable.py`.
pub const ODE_BISTABLE_HIGH_B: f64 = 0.15;

/// Phage defense population count tolerance (Bd vs Python baseline).
///
/// Phage attack scenario: large population ODE where absolute counts
/// reach ~278 cells. RK4 vs LSODA at dt=0.001 differ by ~10 cells
/// due to step-size sensitivity in rapid population crashes.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_POPULATION_ABSOLUTE: f64 = 10.0;

/// Phage defense large-population tolerance (no-phage / pure-defended).
///
/// Steady-state populations ~100k–140k cells. RK4 vs LSODA at dt=0.001
/// accumulates ~1000 cells of drift over 100+ hours of simulated time.
/// Proportionally ~0.7% — same relative order as [`PHAGE_POPULATION_ABSOLUTE`]
/// for the attack scenario.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_LARGE_POPULATION: f64 = 1000.0;

/// Phage defense near-zero floor (crashed population).
///
/// After phage attack, undefended bacteria crash to ~0. RK4 integrator
/// residual keeps the value slightly above zero. 1.0 cell is the
/// biologically insignificant floor.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_CRASH_FLOOR: f64 = 1.0;

/// Biogas kinetics asymptotic tolerance (Gompertz H(∞) → P).
///
/// Modified Gompertz H(t) approaches P (maximum potential) as t → ∞.
/// At t=50 days with typical parameters, H ≈ P within 1.0 mL/g VS.
/// Validated: `validate_cpu_vs_gpu_v11` D44, `validate_toadstool_dispatch_v4` S12.
pub const BIOGAS_KINETICS_ASYMPTOTIC: f64 = 1.0;

// ═══════════════════════════════════════════════════════════════════
// RK45 adaptive solver defaults
// ═══════════════════════════════════════════════════════════════════

/// Default relative tolerance for `rk45_integrate` adaptive step control.
///
/// Controls step-size adaptation via `|error| / (rel_tol * |y| + abs_tol)`.
/// 1e-8 matches `scipy.integrate.solve_ivp` default `rtol` and provides
/// ~8 digits of accuracy per step for non-stiff systems.
/// Validated: Exp020/023/024/025/027/030 (all 6 ODE models).
pub const RK45_DEFAULT_REL_TOL: f64 = 1e-8;

/// Default absolute tolerance for `rk45_integrate` adaptive step control.
///
/// Floor for error scaling when `|y|` is near zero. 1e-10 prevents
/// overly aggressive step refinement for repressed species at the
/// numerical floor. Matches `scipy.integrate.solve_ivp` default `atol`.
/// Validated: Exp020/023/024/025/027/030 (all 6 ODE models).
pub const RK45_DEFAULT_ABS_TOL: f64 = 1e-10;
