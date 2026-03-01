// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spectral and analytical chemistry workloads (KMD, PFAS, signal processing).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// KMD (Kendrick mass defect) — element-wise via FMR.
#[must_use]
pub fn kmd() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "kmd",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// GBM batch inference — composes `TreeInferenceGpu`.
#[must_use]
pub fn gbm_inference() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "gbm_inference",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("TreeInferenceGpu")
}

/// Merge pairs — batch overlap scoring via FMR.
#[must_use]
pub fn merge_pairs() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "merge_pairs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Signal processing / peak detection — batch via FMR.
#[must_use]
pub fn signal_processing() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "signal_processing",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Feature table extraction — chains EIC + signal GPU.
#[must_use]
pub fn feature_table() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "feature_table",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + WeightedDotF64")
}

/// PFAS spectral matching against reference libraries.
///
/// Cosine similarity scoring between measured and reference spectra.
#[must_use]
pub fn pfas_spectral_match() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "pfas_spectral_match",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("WeightedDotF64")
}
