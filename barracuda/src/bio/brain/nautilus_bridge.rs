// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nautilus Shell bio bridge — evolutionary reservoir computing for bio sentinel.
//!
//! Maps `BioObservation` data to `bingocube-nautilus` `ReservoirInput`, enabling:
//! - Concept edge detection (community phase boundaries)
//! - Drift monitoring (population health in long-term sentinel)
//! - Evolutionary adaptation (board populations specialize to local community)
//!
//! # Cross-spring provenance
//!
//! - Nautilus Shell core: `primalTools/bingoCube/nautilus` (shared ecosystem tool)
//! - Physics `BetaObservation`: hotSpring v0.6.15 (QCD `β` coupling → bio diversity)
//! - Board evolution + LOO edge detection: `bingocube-nautilus` `NautilusBrain`
//! - Bio observation mapping: wetSpring V89 (diversity/taxonomy/AMR features)

use bingocube_nautilus::{
    BetaObservation, NautilusBrain, NautilusBrainConfig, ReservoirInput, ShellConfig,
};

use super::observation::BioObservation;

/// Default config for the bio Nautilus brain.
///
/// Maps the physics defaults to bio-appropriate values:
/// - 3 prediction targets: diversity regime cost, community phase, AMR load
/// - 24 board population (adequate for bio feature dimensionality)
/// - Concept edge threshold at 0.15 (empirical from hotSpring)
#[must_use]
pub fn default_bio_config() -> NautilusBrainConfig {
    NautilusBrainConfig {
        shell: ShellConfig {
            population_size: 24,
            n_targets: 3,
            ridge_lambda: crate::tolerances::RIDGE_NAUTILUS_DEFAULT,
            ..ShellConfig::default()
        },
        generations_per_cycle: 20,
        min_training_points: 5,
        concept_edge_threshold: 0.15,
        edge_seed_count: 4,
    }
}

/// Bio Nautilus brain adapter wrapping `NautilusBrain`.
///
/// Converts `BioObservation` to `BetaObservation` (physics→bio mapping) and
/// provides bio-specific prediction and edge detection methods.
pub struct BioNautilusBrain {
    inner: NautilusBrain,
}

impl BioNautilusBrain {
    /// Create a new bio Nautilus brain with default bio config.
    #[must_use]
    pub fn new(instance_name: &str) -> Self {
        Self {
            inner: NautilusBrain::new(default_bio_config(), instance_name),
        }
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(config: NautilusBrainConfig, instance_name: &str) -> Self {
        Self {
            inner: NautilusBrain::new(config, instance_name),
        }
    }

    /// Add a bio observation, converting to the Nautilus `BetaObservation` format.
    ///
    /// The mapping:
    /// - `beta` → Shannon diversity (serves as the "coupling constant" axis)
    /// - `plaquette` → evenness (primary observable)
    /// - `cg_iters` → Chao1 (expensive-to-compute estimate)
    /// - `acceptance` → 1 - AMR load (community "health acceptance")
    /// - `delta_h_abs` → Bray-Curtis mean (dissimilarity magnitude)
    /// - `anderson_r` → Anderson W from spectral analysis
    /// - `anderson_lambda_min` → Anderson phase
    pub fn observe(&mut self, obs: &BioObservation) {
        let beta_obs = bio_to_beta(obs);
        self.inner.observe(beta_obs);
    }

    /// Train the Nautilus shell on accumulated observations.
    /// Returns MSE if enough data, `None` otherwise.
    pub fn train(&mut self) -> Option<f64> {
        self.inner.train()
    }

    /// Predict bio observables for a given diversity level.
    ///
    /// Returns `(predicted_chao1, predicted_evenness, predicted_health)`,
    /// or `None` if untrained.
    #[must_use]
    pub fn predict(&self, shannon: f64) -> Option<(f64, f64, f64)> {
        self.inner.predict_dynamical(shannon, None)
    }

    /// Detect concept edges — diversity values where the model fails.
    ///
    /// These represent community phase boundaries where the diversity
    /// regime changes (e.g., healthy → stressed transition).
    pub fn detect_concept_edges(&mut self) -> Vec<(f64, f64)> {
        self.inner.detect_concept_edges()
    }

    /// Whether the board population is drifting (losing genetic diversity).
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.inner.is_drifting()
    }

    /// Concept edges detected so far.
    #[must_use]
    pub fn concept_edges(&self) -> &[f64] {
        &self.inner.concept_edges
    }

    /// Whether the brain has been trained at least once.
    #[must_use]
    pub const fn is_trained(&self) -> bool {
        self.inner.trained
    }

    /// Total observations accumulated.
    #[must_use]
    pub const fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }

    /// Screen candidate diversity values, returning them sorted by information value.
    ///
    /// Concept edges and high-cost regions rank highest — useful for adaptive
    /// sampling strategy (where to sequence next).
    #[must_use]
    pub fn screen_candidates(&self, candidates: &[f64]) -> Vec<(f64, f64)> {
        self.inner.screen_candidates(candidates)
    }

    /// Serialize the brain state to JSON for checkpoint / transfer.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if serialization fails.
    #[cfg(feature = "json")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        self.inner.to_json()
    }

    /// Deserialize brain state from JSON.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if deserialization fails.
    #[cfg(feature = "json")]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let inner = NautilusBrain::from_json(json)?;
        Ok(Self { inner })
    }

    /// Convert a bio observation to a Nautilus reservoir input directly
    /// (for custom prediction pipelines).
    #[must_use]
    pub fn bio_to_reservoir_input(obs: &BioObservation) -> ReservoirInput {
        ReservoirInput::Continuous(vec![
            obs.shannon / 5.0,
            obs.evenness,
            obs.anderson_w.ln_1p().min(1.0),
            obs.anderson_phase,
            obs.bray_curtis_mean,
        ])
    }
}

/// Convert a `BioObservation` to `BetaObservation` (physics→bio mapping).
fn bio_to_beta(obs: &BioObservation) -> BetaObservation {
    BetaObservation {
        beta: obs.shannon,
        quenched_plaq: Some(obs.bray_curtis_mean),
        quenched_plaq_var: None,
        plaquette: obs.evenness,
        cg_iters: obs.chao1,
        acceptance: 1.0 - obs.amr_load,
        delta_h_abs: obs.bray_curtis_mean,
        anderson_r: Some(obs.anderson_w),
        anderson_lambda_min: Some(obs.anderson_phase),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    fn sample_obs() -> BioObservation {
        BioObservation {
            sample_id: "test_001".to_string(),
            shannon: 3.5,
            simpson: 0.85,
            evenness: 0.78,
            chao1: 150.0,
            bray_curtis_mean: 0.42,
            anderson_w: 4.5,
            anderson_phase: 0.5,
            amr_load: 0.12,
        }
    }

    #[test]
    fn bio_to_beta_conversion() {
        let obs = sample_obs();
        let beta = bio_to_beta(&obs);
        assert!((beta.beta - 3.5).abs() < f64::EPSILON);
        assert!((beta.plaquette - 0.78).abs() < f64::EPSILON);
        assert!((beta.cg_iters - 150.0).abs() < f64::EPSILON);
        assert!((beta.acceptance - 0.88).abs() < f64::EPSILON);
    }

    #[test]
    fn brain_creation() {
        let brain = BioNautilusBrain::new("wetspring-test");
        assert!(!brain.is_trained());
        assert_eq!(brain.observation_count(), 0);
    }

    #[test]
    fn brain_observe_and_count() {
        let mut brain = BioNautilusBrain::new("wetspring-test");
        brain.observe(&sample_obs());
        assert_eq!(brain.observation_count(), 1);
    }

    #[test]
    fn brain_train_insufficient_data() {
        let mut brain = BioNautilusBrain::new("wetspring-test");
        brain.observe(&sample_obs());
        assert!(brain.train().is_none(), "should not train with 1 point");
    }

    #[test]
    fn brain_train_with_enough_data() {
        let mut brain = BioNautilusBrain::new("wetspring-test");
        for i in 0..10 {
            let mut obs = sample_obs();
            obs.shannon = f64::from(i).mul_add(0.5, 1.0);
            obs.evenness = f64::from(i).mul_add(0.03, 0.5);
            obs.chao1 = f64::from(i).mul_add(20.0, 50.0);
            brain.observe(&obs);
        }
        let mse = brain.train();
        assert!(mse.is_some(), "should train with 10 points");
    }

    #[test]
    fn predict_after_training() {
        let mut brain = BioNautilusBrain::new("wetspring-test");
        for i in 0..10 {
            let mut obs = sample_obs();
            obs.shannon = f64::from(i).mul_add(0.5, 1.0);
            obs.evenness = f64::from(i).mul_add(0.03, 0.5);
            obs.chao1 = f64::from(i).mul_add(20.0, 50.0);
            brain.observe(&obs);
        }
        brain.train();
        let pred = brain.predict(3.0);
        assert!(pred.is_some(), "should predict after training");
    }

    #[test]
    fn reservoir_input_dimensions() {
        let obs = sample_obs();
        let input = BioNautilusBrain::bio_to_reservoir_input(&obs);
        match input {
            ReservoirInput::Continuous(v) => assert_eq!(v.len(), 5),
            ReservoirInput::Discrete(_) => panic!("expected continuous input"),
        }
    }

    #[test]
    #[cfg(feature = "json")]
    fn json_roundtrip() {
        let mut brain = BioNautilusBrain::new("wetspring-test");
        brain.observe(&sample_obs());
        let json = brain.to_json().expect("serialize");
        let restored = BioNautilusBrain::from_json(&json).expect("deserialize");
        assert_eq!(restored.observation_count(), 1);
    }
}
