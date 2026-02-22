// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated spatial payoff (game theory) via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::spatial_payoff::SpatialPayoffGpu` —
//! evolved by `neuralSpring`, absorbed in `ToadStool` session 31f.
//!
//! Used by `wetSpring` for cooperation dynamics in biofilm models (Exp025)
//! and evolutionary game theory fitness landscapes.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::spatial_payoff::SpatialPayoffGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Spatial payoff result for a cooperation grid.
pub struct SpatialPayoffResult {
    /// Fitness value per cell, row-major `[grid_size × grid_size]`.
    pub fitness: Vec<f32>,
    /// Grid side length.
    pub grid_size: usize,
}

/// GPU-accelerated spatial payoff computation on a toroidal grid.
///
/// Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean).
pub struct SpatialPayoffGpuWrapper {
    device: Arc<WgpuDevice>,
    inner: SpatialPayoffGpu,
}

impl SpatialPayoffGpuWrapper {
    /// Creates a new GPU spatial payoff evaluator.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = SpatialPayoffGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Compute spatial payoff on a cooperation grid.
    ///
    /// `grid`: row-major `[grid_size × grid_size]` where 0 = defector, 1 = cooperator.
    /// Uses prisoner's dilemma payoff with Moore neighborhood on a toroidal grid.
    ///
    /// # Errors
    ///
    /// Returns an error if grid size mismatches or GPU read fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn compute(
        &self,
        grid: &[u32],
        grid_size: usize,
        benefit: f32,
        cost: f32,
    ) -> crate::error::Result<SpatialPayoffResult> {
        if grid.len() != grid_size * grid_size {
            return Err(crate::error::Error::Gpu(format!(
                "Grid size mismatch: {} != {}²",
                grid.len(),
                grid_size
            )));
        }

        let d = self.device.device();
        let n = grid_size * grid_size;

        let grid_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spatial grid"),
            contents: bytemuck::cast_slice(grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial fitness"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inner
            .dispatch(&grid_buf, &fit_buf, grid_size as u32, benefit, cost);
        d.poll(wgpu::Maintain::Wait);

        let fitness = self
            .device
            .read_buffer_f32(&fit_buf, n)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(SpatialPayoffResult { fitness, grid_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[tokio::test]
    async fn spatial_payoff_all_cooperators() {
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(_) => return,
        };
        let device = gpu.to_wgpu_device();
        let sp = SpatialPayoffGpuWrapper::new(&device);

        // 4×4 grid, all cooperators
        let grid = vec![1u32; 16];
        let result = sp.compute(&grid, 4, 3.0, 1.0).expect("spatial dispatch");
        assert_eq!(result.fitness.len(), 16);
        // All cooperators with 8 neighbors: payoff = 8 × (benefit - cost) = 16.0
        for &f in &result.fitness {
            assert!((f - 16.0).abs() < 1e-4);
        }
    }
}
