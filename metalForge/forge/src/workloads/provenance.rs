// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader provenance tracking for workload dispatch.
//!
//! Tracks where GPU shaders live (`ToadStool` absorbed vs local WGSL) to enable:
//! 1. Dispatch decisions — local shaders need `compile_shader_f64`; absorbed
//!    primitives use `ToadStool`'s pre-built pipelines.
//! 2. Absorption planning — `ToadStool` can see which domains still use local
//!    shaders and prioritize absorption accordingly.
//! 3. Validation routing — local shaders need CPU ↔ GPU parity checks;
//!    absorbed primitives are `ToadStool`-validated upstream.

use crate::dispatch::Workload;
use crate::substrate::Capability;

/// Where the GPU shader for a workload lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderOrigin {
    /// Absorbed by `ToadStool` — uses `barracuda::ops::*` primitives.
    Absorbed,
    /// Local WGSL shader in `barracuda/src/shaders/` — pending absorption.
    Local,
    /// CPU-only domain — no GPU shader exists or is planned.
    CpuOnly,
}

/// ODE system dimensions for dispatch sizing.
#[derive(Debug, Clone, Copy)]
pub struct OdeDims {
    /// Number of state variables.
    pub n_vars: u32,
    /// Number of parameters per batch element.
    pub n_params: u32,
}

/// A bio workload with shader provenance tracking.
#[derive(Debug)]
pub struct BioWorkload {
    /// The dispatch workload (name + capabilities).
    pub workload: Workload,
    /// Where the GPU implementation lives.
    pub origin: ShaderOrigin,
    /// `ToadStool` primitive name (if absorbed).
    pub primitive: Option<&'static str>,
    /// ODE system dimensions (if applicable).
    pub ode_dims: Option<OdeDims>,
}

impl BioWorkload {
    pub(super) const fn new_static(origin: ShaderOrigin) -> Self {
        Self {
            workload: Workload {
                name: String::new(),
                required: Vec::new(),
                preferred_substrate: None,
                data_bytes: None,
            },
            origin,
            primitive: None,
            ode_dims: None,
        }
    }

    pub(super) fn named(mut self, name: &str, required: Vec<Capability>) -> Self {
        self.workload.name = name.to_string();
        self.workload.required = required;
        self
    }

    pub(super) const fn with_primitive(mut self, primitive: &'static str) -> Self {
        self.primitive = Some(primitive);
        self
    }

    pub(super) const fn with_ode(mut self, n_vars: u32, n_params: u32) -> Self {
        self.ode_dims = Some(OdeDims { n_vars, n_params });
        self
    }

    /// Whether this workload uses a local (non-absorbed) WGSL shader.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Local)
    }

    /// Whether this workload has been absorbed by `ToadStool`.
    #[must_use]
    pub const fn is_absorbed(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Absorbed)
    }

    /// Whether this workload is CPU-only (no GPU path).
    #[must_use]
    pub const fn is_cpu_only(&self) -> bool {
        matches!(self.origin, ShaderOrigin::CpuOnly)
    }
}
