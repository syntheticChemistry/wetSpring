// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch fitness evaluation via barraCuda.
//!
//! Delegates to `barracuda::ops::bio::batch_fitness::BatchFitnessGpu` —
//! evolved by `neuralSpring`, absorbed in `ToadStool` session 31f.
//!
//! Computes linear fitness: `fitness[i]` = dot(`population[i]`, weights).

use barracuda::BatchFitnessGpu;
use barracuda::device::WgpuDevice;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::cast;

/// GPU-accelerated batch fitness evaluator.
///
/// Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean).
pub struct BatchFitnessGpuWrapper {
    device: Arc<WgpuDevice>,
    inner: BatchFitnessGpu,
}

impl BatchFitnessGpuWrapper {
    /// Creates a new GPU batch fitness evaluator.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = BatchFitnessGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Evaluate fitness for a population.
    ///
    /// `population`: row-major `[pop_size × genome_len]` f32.
    /// `weights`: `[genome_len]` f32.
    /// Returns `[pop_size]` fitness values (dot product per individual).
    ///
    /// # Errors
    ///
    /// Returns an error if size mismatches or GPU read fails.
    pub fn evaluate(
        &self,
        population: &[f32],
        weights: &[f32],
        pop_size: usize,
        genome_len: usize,
    ) -> crate::error::Result<Vec<f32>> {
        if population.len() != pop_size * genome_len {
            return Err(crate::error::Error::Gpu("Population size mismatch".into()));
        }
        if weights.len() != genome_len {
            return Err(crate::error::Error::Gpu("Weights length mismatch".into()));
        }

        let d = self.device.device();

        let pop_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness pop"),
            contents: bytemuck::cast_slice(population),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness output"),
            size: (pop_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inner.dispatch(
            &pop_buf,
            &w_buf,
            &fit_buf,
            cast::usize_u32(pop_size),
            cast::usize_u32(genome_len),
        );
        let _ = d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        self.device
            .read_buffer_f32(&fit_buf, pop_size)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "test module: f64 tolerance → f32 cast is intentional for GPU f32 comparisons"
)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;
    use crate::tolerances;

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn batch_fitness_dot_product() {
        let Ok(gpu) = GpuF64::new().await else { return };
        let device = gpu.to_wgpu_device();
        let bf = BatchFitnessGpuWrapper::new(&device);

        let pop = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 individuals × 2 genes
        let weights = vec![3.0f32, 5.0];
        let result = bf.evaluate(&pop, &weights, 2, 2).expect("fitness");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < tolerances::GPU_F32_PARITY as f32); // 1*3 + 0*5
        assert!((result[1] - 5.0).abs() < tolerances::GPU_F32_PARITY as f32); // 0*3 + 1*5
    }
}
