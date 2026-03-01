// SPDX-License-Identifier: AGPL-3.0-or-later

//! ODE workloads (`BatchedOdeRK4` trait-generated WGSL).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// QS/c-di-GMP ODE sweep (4 vars, 17 params — absorbed).
#[must_use]
pub fn qs_biofilm_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "qs_biofilm_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4F64")
        .with_ode(4, 17)
}

/// Phage defense ODE (4 vars, 11 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn phage_defense_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "phage_defense_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<PhageDefenseOde>")
        .with_ode(4, 11)
}

/// Bistable QS ODE (5 vars, 21 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn bistable_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "bistable_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<BistableOde>")
        .with_ode(5, 21)
}

/// Multi-signal QS ODE (7 vars, 24 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn multi_signal_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "multi_signal_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<MultiSignalOde>")
        .with_ode(7, 24)
}

/// Cooperation game theory ODE (4 vars, 13 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn cooperation_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cooperation_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CooperationOde>")
        .with_ode(4, 13)
}

/// Capacitor phenotype ODE (6 vars, 16 params — absorbed via `BatchedOdeRK4` trait).
#[must_use]
pub fn capacitor_ode() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "capacitor_ode",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CapacitorOde>")
        .with_ode(6, 16)
}
