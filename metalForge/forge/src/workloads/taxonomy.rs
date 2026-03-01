// SPDX-License-Identifier: AGPL-3.0-or-later

//! Taxonomy classification workloads (NPU candidate via int8 quantization).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::{Capability, SubstrateKind};

/// Taxonomy classification (NPU candidate via int8 quantization).
#[must_use]
pub fn taxonomy() -> BioWorkload {
    let mut w = BioWorkload::new_static(ShaderOrigin::Absorbed).named(
        "taxonomy",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w.with_primitive("TaxonomyFcF64")
}
