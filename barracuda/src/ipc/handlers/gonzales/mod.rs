// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gonzales dermatitis and Anderson immunology IPC handlers.
//!
//! Methods for interactive exploration of the Anderson localization framework
//! applied to cytokine signaling, dose-response, pharmacokinetics, and
//! tissue geometry in atopic dermatitis (Papers 12, 53-58).
//!
//! Split by domain family:
//! - [`pharmacology`]: IC50 dose-response + lokivetmab PK decay
//! - [`tissue`]: Anderson tissue lattice + cross-species morphometry
//! - [`anderson_sweeps`]: biome atlas + disorder sweep + hormesis

mod anderson_sweeps;
mod pharmacology;
mod tissue;

pub use anderson_sweeps::{handle_biome_atlas, handle_disorder_sweep, handle_hormesis};
pub use pharmacology::{handle_dose_response, handle_pk_decay};
pub use tissue::{handle_cross_species, handle_tissue_lattice};

/// Linearly spaced vector from 0.0 to `max`, `n` points.
pub fn linspace(n: usize, max: f64) -> Vec<f64> {
    let denom = n.saturating_sub(1).max(1);
    (0..n)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "n ≤ ~1000 fits in f64 mantissa")]
            let t = (i as f64) / (denom as f64);
            max * t
        })
        .collect()
}
