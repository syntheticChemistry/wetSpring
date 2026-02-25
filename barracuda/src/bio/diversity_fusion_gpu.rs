// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused diversity metrics GPU extension — Shannon + Simpson + evenness in one dispatch.
//!
//! **Write phase**: Local WGSL extension following hotSpring's absorption pattern.
//! Structured for `ToadStool` absorption as `ops::bio::diversity_fusion`.
//!
//! Computes three diversity indices per sample in a single GPU kernel pass,
//! avoiding three separate [`FusedMapReduceF64`] dispatches. For N samples
//! with S species each, this reduces GPU round-trips from 3N to N.
//!
//! # Binding layout
//!
//! | Binding | Type | Content |
//! |---------|------|---------|
//! | 0 | uniform | `{ n_samples: u32, n_species: u32 }` |
//! | 1 | storage, read | `abundances: array<f64>` (`n_samples` × `n_species`) |
//! | 2 | storage, `read_write` | `results: array<f64>` (`n_samples` × 3) |
//!
//! # Results layout
//!
//! Per sample (3 contiguous f64 values):
//! - `[0]` Shannon entropy H' = −Σ pᵢ ln(pᵢ)
//! - `[1]` Simpson index D = 1 − Σ pᵢ²
//! - `[2]` Pielou evenness J' = H'/`ln(S_obs)`
//!
//! # Dispatch geometry
//!
//! `ceil(n_samples / 64)` workgroups × 1 × 1, `@workgroup_size(64)`.
//! Each thread processes one sample (iterates all species).
//!
//! # CPU reference
//!
//! [`diversity_fusion_cpu`] provides the validated CPU reference for parity checks.
//!
//! # Absorption path
//!
//! 1. WGSL: `barracuda/src/bio/shaders/diversity_fusion_f64.wgsl`
//! 2. Rust: this module (GPU wrapper + CPU reference)
//! 3. Handoff: `wateringHole/handoffs/` with binding layout + validation results
//! 4. `ToadStool` absorbs as `ops::bio::diversity_fusion`
//! 5. wetSpring rewires to upstream, deletes local code

use barracuda::device::{WgpuDevice, storage_bgl_entry, uniform_bgl_entry};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL shader source for fused diversity computation.
pub const WGSL: &str = include_str!("shaders/diversity_fusion_f64.wgsl");

/// Result of fused diversity computation for a single sample.
#[derive(Debug, Clone, Copy)]
pub struct DiversityResult {
    /// Shannon entropy H' = −Σ pᵢ ln(pᵢ).
    pub shannon: f64,
    /// Simpson index D = 1 − Σ pᵢ².
    pub simpson: f64,
    /// Pielou evenness J' = H'/`ln(S_obs)`.
    pub evenness: f64,
}

/// GPU-backed fused diversity computation.
pub struct DiversityFusionGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    n_samples: u32,
    n_species: u32,
}

impl DiversityFusionGpu {
    /// Compile the diversity fusion shader and create the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let d = device.device();
        let module = device.compile_shader_f64(WGSL, Some("DiversityFusion f64"));

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DiversityFusion BGL"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, false),
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DiversityFusion Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DiversityFusion Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bgl,
            device,
        })
    }

    /// Compute fused diversity metrics for multiple samples.
    ///
    /// `abundances` is a flat row-major array of shape `[n_samples × n_species]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or buffer readback fails.
    ///
    /// # Panics
    ///
    /// Panics if `abundances.len() != n_samples * n_species`.
    pub fn compute(
        &self,
        abundances: &[f64],
        n_samples: usize,
        n_species: usize,
    ) -> crate::error::Result<Vec<DiversityResult>> {
        assert_eq!(abundances.len(), n_samples * n_species);

        let d = self.device.device();
        let q = self.device.queue();

        let params = GpuParams {
            n_samples: n_samples as u32,
            n_species: n_species as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DiversityFusion params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let abundances_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DiversityFusion abundances"),
            contents: bytemuck::cast_slice(abundances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let results_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DiversityFusion results"),
            contents: bytemuck::cast_slice(&vec![0.0f64; n_samples * 3]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DiversityFusion BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: abundances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DiversityFusion Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DiversityFusion Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((params.n_samples).div_ceil(64), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let raw = self
            .device
            .read_buffer_f64(&results_buf, n_samples * 3)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(raw
            .chunks_exact(3)
            .map(|c| DiversityResult {
                shannon: c[0],
                simpson: c[1],
                evenness: c[2],
            })
            .collect())
    }
}

/// CPU reference implementation of fused diversity metrics.
///
/// Validated against Python scipy/skbio outputs. Used for CPU ↔ GPU parity checks.
#[must_use]
pub fn diversity_fusion_cpu(abundances: &[f64], n_species: usize) -> Vec<DiversityResult> {
    abundances
        .chunks_exact(n_species)
        .map(|sample| {
            let total: f64 = sample.iter().sum();
            if total <= 0.0 {
                return DiversityResult {
                    shannon: 0.0,
                    simpson: 0.0,
                    evenness: 0.0,
                };
            }

            let mut shannon = 0.0;
            let mut simpson_sum = 0.0;
            let mut s_obs = 0.0_f64;

            for &count in sample {
                if count > 0.0 {
                    let p = count / total;
                    shannon -= p * p.ln();
                    simpson_sum += p * p;
                    s_obs += 1.0;
                }
            }

            let evenness = if s_obs > 1.0 {
                shannon / s_obs.ln()
            } else {
                0.0
            };

            DiversityResult {
                shannon,
                simpson: 1.0 - simpson_sum,
                evenness,
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn cpu_single_sample_known_values() {
        let abundances = [10.0, 20.0, 30.0, 40.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        assert_eq!(results.len(), 1);

        let r = &results[0];
        let total = 100.0;
        let p = [0.1, 0.2, 0.3, 0.4];
        let expected_shannon: f64 = -p.iter().map(|pi: &f64| pi * pi.ln()).sum::<f64>();
        let expected_simpson: f64 = 1.0 - p.iter().map(|pi: &f64| pi * pi).sum::<f64>();
        let expected_evenness = expected_shannon / 4.0_f64.ln();

        assert!((r.shannon - expected_shannon).abs() < TOL, "shannon");
        assert!((r.simpson - expected_simpson).abs() < TOL, "simpson");
        assert!((r.evenness - expected_evenness).abs() < TOL, "evenness");
        let _ = total;
    }

    #[test]
    fn cpu_uniform_distribution() {
        let abundances = [25.0; 4];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        let expected_shannon = 4.0_f64.ln();
        assert!((r.shannon - expected_shannon).abs() < TOL);
        assert!((r.simpson - 0.75).abs() < TOL);
        assert!((r.evenness - 1.0).abs() < TOL);
    }

    #[test]
    fn cpu_single_species_dominance() {
        let abundances = [100.0, 0.0, 0.0, 0.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        assert!(r.shannon.abs() < TOL, "single species → H'=0");
        assert!(r.simpson.abs() < TOL, "single species → D=0");
        assert!(r.evenness.abs() < TOL, "single species → J'=0");
    }

    #[test]
    fn cpu_empty_sample() {
        let abundances = [0.0; 4];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        assert!(r.shannon.abs() < f64::EPSILON, "empty → H'=0");
        assert!(r.simpson.abs() < f64::EPSILON, "empty → D=0");
        assert!(r.evenness.abs() < f64::EPSILON, "empty → J'=0");
    }

    #[test]
    fn cpu_multiple_samples() {
        let abundances = [10.0, 20.0, 30.0, 40.0, 25.0, 25.0, 25.0, 25.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        assert_eq!(results.len(), 2);

        assert!(results[1].shannon > results[0].shannon);
        assert!((results[1].evenness - 1.0).abs() < TOL);
    }

    #[test]
    fn cpu_two_species_even() {
        let abundances = [50.0, 50.0];
        let results = diversity_fusion_cpu(&abundances, 2);
        let r = &results[0];

        assert!((r.shannon - 2.0_f64.ln()).abs() < TOL);
        assert!((r.simpson - 0.5).abs() < TOL);
        assert!((r.evenness - 1.0).abs() < TOL);
    }
}
