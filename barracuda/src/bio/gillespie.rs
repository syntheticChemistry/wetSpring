// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gillespie Stochastic Simulation Algorithm (SSA).
//!
//! Implements the direct method (Gillespie 1977) for exact stochastic
//! simulation of well-mixed chemical systems. Used to model c-di-GMP
//! signalling specificity (Massie et al. 2012) where molecule counts
//! are low enough that deterministic ODE breaks down.
//!
//! # Algorithm
//!
//! 1. Compute propensities for all reactions
//! 2. Draw exponential waiting time from total propensity
//! 3. Select reaction proportional to its propensity
//! 4. Update state and advance time
//!
//! # PRNG
//!
//! Uses a sovereign Lehmer LCG for reproducibility without external
//! dependencies. Not cryptographic — purely for simulation.
//!
//! # References
//!
//! - Gillespie, D.T. "Exact Stochastic Simulation of Coupled Chemical
//!   Reactions." *J. Phys. Chem.* 81, 2340–2361 (1977).
//! - Massie et al. "Quantification of high-specificity cyclic diguanylate
//!   signaling." *PNAS* 109, 12746–12751 (2012).

/// Sovereign Lehmer LCG — deterministic, no external dependencies.
///
/// Uses Knuth's constants for full-period 64-bit LCG.
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    const MULT: u64 = 6_364_136_223_846_793_005;
    const INC: u64 = 1_442_695_040_888_963_407;

    /// Create a new LCG seeded with the given value.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(Self::MULT).wrapping_add(Self::INC),
        }
    }

    /// Advance state and return raw `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(Self::INC);
        self.state
    }

    /// Uniform `f64` in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    /// Exponential variate with rate `lambda` (mean = `1/lambda`).
    ///
    /// Uses inverse CDF: `-ln(U) / lambda`.
    pub fn exp_variate(&mut self, lambda: f64) -> f64 {
        let u = self.next_f64();
        let u_clamped = if u == 0.0 { f64::MIN_POSITIVE } else { u };
        -u_clamped.ln() / lambda
    }
}

/// Propensity function type: maps molecule state to a reaction rate.
pub type PropensityFn = Box<dyn Fn(&[i64]) -> f64>;

/// A single reaction in the system.
pub struct Reaction {
    /// Propensity function: given current state, returns rate.
    pub propensity: PropensityFn,
    /// State change vector (stoichiometry): added to state when fired.
    pub stoichiometry: Vec<i64>,
}

/// Result of a single SSA trajectory.
///
/// States are stored in a flat `Vec<i64>` of length `n_points * n_species`
/// for contiguous memory access. Use [`state_at`](Self::state_at) or
/// [`states`](Self::states) for ergonomic indexing.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Time points of state transitions.
    pub times: Vec<f64>,
    /// Number of species (state variables) per time point.
    pub n_species: usize,
    /// Flat row-major state history.
    pub states: Vec<i64>,
}

impl Trajectory {
    /// Number of recorded time points.
    #[inline]
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.times.len()
    }

    /// Slice of all species at time point `i`.
    #[inline]
    #[must_use]
    pub fn state_at(&self, i: usize) -> &[i64] {
        let start = i * self.n_species;
        &self.states[start..start + self.n_species]
    }

    /// Final state (last recorded snapshot).
    #[must_use]
    pub fn final_state(&self) -> &[i64] {
        if self.states.is_empty() {
            &[]
        } else {
            &self.states[self.states.len() - self.n_species..]
        }
    }

    /// Final time.
    #[must_use]
    pub fn final_time(&self) -> f64 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Number of events (transitions).
    #[must_use]
    pub fn n_events(&self) -> usize {
        self.times.len().saturating_sub(1)
    }

    /// Iterator over all state snapshots.
    #[inline]
    pub fn states_iter(&self) -> impl Iterator<Item = &[i64]> {
        self.states.chunks_exact(self.n_species)
    }
}

/// Run Gillespie direct-method SSA.
///
/// - `initial`: starting molecule counts per species
/// - `reactions`: list of reactions with propensity functions and stoichiometry
/// - `t_max`: simulation end time
/// - `rng`: seeded PRNG for reproducibility
pub fn gillespie_ssa(
    initial: &[i64],
    reactions: &[Reaction],
    t_max: f64,
    rng: &mut Lcg64,
) -> Trajectory {
    let n_species = initial.len();
    let mut state = initial.to_vec();
    let mut t = 0.0;

    let mut times = vec![t];
    let mut flat_states = Vec::from(initial);

    loop {
        let propensities: Vec<f64> = reactions.iter().map(|r| (r.propensity)(&state)).collect();
        let a0: f64 = propensities.iter().sum();

        if a0 <= 0.0 {
            break;
        }

        let tau = rng.exp_variate(a0);
        t += tau;
        if t > t_max {
            break;
        }

        let r = rng.next_f64() * a0;
        let mut cumulative = 0.0;
        let mut selected = reactions.len() - 1;
        for (j, &aj) in propensities.iter().enumerate() {
            cumulative += aj;
            if r < cumulative {
                selected = j;
                break;
            }
        }

        for (i, &delta) in reactions[selected].stoichiometry.iter().enumerate() {
            state[i] += delta;
            if state[i] < 0 {
                state[i] = 0;
            }
        }

        times.push(t);
        flat_states.extend_from_slice(&state);
    }

    Trajectory {
        times,
        n_species,
        states: flat_states,
    }
}

/// Run a birth-death SSA matching the Massie 2012 simplified c-di-GMP model.
///
/// Species: `[cdGMP]`
/// - Reaction 1: synthesis (rate = `k_dgc`)
/// - Reaction 2: degradation (rate = `k_pde * cdGMP`)
#[must_use]
pub fn birth_death_ssa(k_dgc: f64, k_pde: f64, t_max: f64, seed: u64) -> Trajectory {
    let mut rng = Lcg64::new(seed);
    let reactions = vec![
        Reaction {
            propensity: Box::new(move |_state: &[i64]| k_dgc),
            stoichiometry: vec![1],
        },
        Reaction {
            propensity: Box::new(move |state: &[i64]| {
                #[allow(clippy::cast_precision_loss)]
                let count = state[0].max(0) as f64;
                k_pde * count
            }),
            stoichiometry: vec![-1],
        },
    ];
    gillespie_ssa(&[0], &reactions, t_max, &mut rng)
}

/// Ensemble statistics from multiple SSA runs.
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    /// Mean of final counts across runs.
    pub mean: f64,
    /// Standard deviation of final counts.
    pub std_dev: f64,
    /// Variance of final counts.
    pub variance: f64,
    /// `Var`/mean ratio (`~1.0` for Poisson process).
    pub fano_factor: f64,
    /// Number of runs.
    pub n_runs: usize,
    /// All final counts.
    pub final_counts: Vec<i64>,
}

/// Run an ensemble of birth-death SSA and compute statistics.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn birth_death_ensemble(
    k_dgc: f64,
    k_pde: f64,
    t_max: f64,
    n_runs: usize,
    base_seed: u64,
) -> EnsembleStats {
    let final_counts: Vec<i64> = (0..n_runs)
        .map(|i| {
            #[allow(clippy::cast_possible_truncation)]
            let seed = base_seed + i as u64;
            let traj = birth_death_ssa(k_dgc, k_pde, t_max, seed);
            traj.final_state()[0]
        })
        .collect();

    let n = final_counts.len() as f64;
    let mean = final_counts.iter().sum::<i64>() as f64 / n;
    let variance = final_counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();
    let fano_factor = if mean > 0.0 { variance / mean } else { 0.0 };

    EnsembleStats {
        mean,
        std_dev,
        variance,
        fano_factor,
        n_runs,
        final_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_deterministic() {
        let mut rng1 = Lcg64::new(42);
        let mut rng2 = Lcg64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn lcg_different_seeds() {
        let mut rng1 = Lcg64::new(42);
        let mut rng2 = Lcg64::new(43);
        let same_count = (0..100)
            .filter(|_| rng1.next_u64() == rng2.next_u64())
            .count();
        assert_eq!(same_count, 0);
    }

    #[test]
    fn lcg_f64_range() {
        let mut rng = Lcg64::new(12345);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
        }
    }

    #[test]
    fn exp_variate_positive() {
        let mut rng = Lcg64::new(42);
        for _ in 0..1000 {
            let v = rng.exp_variate(1.0);
            assert!(v > 0.0, "exponential variate must be positive");
        }
    }

    #[test]
    fn birth_death_deterministic_same_seed() {
        let traj1 = birth_death_ssa(10.0, 0.1, 50.0, 42);
        let traj2 = birth_death_ssa(10.0, 0.1, 50.0, 42);
        assert_eq!(traj1.final_state(), traj2.final_state());
        assert_eq!(traj1.n_events(), traj2.n_events());
    }

    #[test]
    fn birth_death_different_seeds() {
        let traj1 = birth_death_ssa(10.0, 0.1, 100.0, 42);
        let traj2 = birth_death_ssa(10.0, 0.1, 100.0, 999);
        assert_ne!(
            traj1.states, traj2.states,
            "different seeds must produce different trajectories"
        );
    }

    #[test]
    fn birth_death_non_negative() {
        let traj = birth_death_ssa(10.0, 0.1, 100.0, 42);
        for state in traj.states_iter() {
            assert!(state[0] >= 0, "molecule count went negative");
        }
    }

    #[test]
    fn birth_death_has_events() {
        let traj = birth_death_ssa(10.0, 0.1, 100.0, 42);
        assert!(traj.n_events() > 10, "too few events: {}", traj.n_events());
    }

    #[test]
    fn ensemble_mean_converges() {
        let stats = birth_death_ensemble(10.0, 0.1, 100.0, 500, 42);
        let analytical = 10.0 / 0.1;
        let error_pct = (stats.mean - analytical).abs() / analytical;
        assert!(
            error_pct < 0.15,
            "ensemble mean {:.1} too far from analytical {analytical:.1} ({:.1}%)",
            stats.mean,
            error_pct * 100.0
        );
    }

    #[test]
    fn ensemble_poisson_like() {
        let stats = birth_death_ensemble(10.0, 0.1, 100.0, 500, 42);
        assert!(
            (0.5..2.0).contains(&stats.fano_factor),
            "Fano factor {:.3} not Poisson-like",
            stats.fano_factor
        );
    }

    #[test]
    fn ensemble_all_non_negative() {
        let stats = birth_death_ensemble(10.0, 0.1, 100.0, 100, 42);
        for &c in &stats.final_counts {
            assert!(c >= 0, "negative final count: {c}");
        }
    }

    #[test]
    fn zero_propensity_halts() {
        let mut rng = Lcg64::new(42);
        let reactions = vec![Reaction {
            propensity: Box::new(|_: &[i64]| 0.0),
            stoichiometry: vec![1],
        }];
        let traj = gillespie_ssa(&[0], &reactions, 100.0, &mut rng);
        assert_eq!(traj.n_events(), 0);
    }

    #[test]
    fn trajectory_time_monotonic() {
        let traj = birth_death_ssa(10.0, 0.1, 50.0, 42);
        for w in traj.times.windows(2) {
            assert!(w[1] >= w[0], "time went backwards: {} -> {}", w[0], w[1]);
        }
    }
}
