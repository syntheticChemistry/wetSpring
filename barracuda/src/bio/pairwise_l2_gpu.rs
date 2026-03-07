// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pairwise L2 (Euclidean) distance via barraCuda.
//!
//! Delegates to `barracuda::ops::bio::pairwise_l2::PairwiseL2Gpu` —
//! computes condensed N*(N-1)/2 Euclidean distances for N feature vectors.
//! Uses f32 internally for broad GPU compatibility.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::bio::PairwiseL2Gpu;
use wgpu::util::DeviceExt;

/// Compute condensed pairwise L2 (Euclidean) distances from feature vectors.
///
/// `coords` is row-major `[n × dim]`. Returns `n*(n-1)/2` distances in
/// condensed order: (1,0), (2,0), (2,1), (3,0), ...
///
/// Uses f32 on GPU; results are converted to f64. For validation against
/// CPU, use a slightly relaxed tolerance (e.g. 1e-5) due to f32 precision.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or dimensions are invalid.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn pairwise_l2_condensed_gpu(
    gpu: &GpuF64,
    coords: &[f64],
    n: usize,
    dim: usize,
) -> Result<Vec<f64>> {
    if coords.len() != n * dim {
        return Err(Error::InvalidInput(format!(
            "coords length {} != n×dim {}×{}",
            coords.len(),
            n,
            dim
        )));
    }
    if n < 2 {
        return Ok(vec![]);
    }

    let device = gpu.to_wgpu_device();
    let d = device.device();
    let n_pairs = n * (n - 1) / 2;

    let coords_f32: Vec<f32> = coords.iter().map(|&x| x as f32).collect();

    let input_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PairwiseL2 input"),
        contents: bytemuck::cast_slice(&coords_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PairwiseL2 output"),
        size: (n_pairs * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pl2 = PairwiseL2Gpu::new(device.clone());
    pl2.dispatch(&input_buf, &output_buf, n as u32, dim as u32)
        .map_err(|e| Error::Gpu(format!("PairwiseL2 dispatch: {e}")))?;

    // Poll to wait for dispatch completion (PairwiseL2Gpu submits internally)
    device.submit_and_poll(std::iter::empty::<wgpu::CommandBuffer>());

    let raw = device
        .read_buffer_f32(&output_buf, n_pairs)
        .map_err(|e| Error::Gpu(format!("PairwiseL2 read: {e}")))?;

    Ok(raw.iter().map(|&x| f64::from(x)).collect())
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        let _: fn(&GpuF64, &[f64], usize, usize) -> Result<Vec<f64>> = pairwise_l2_condensed_gpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let coords = vec![0.0, 0.0, 1.0, 1.0];
        let result = pairwise_l2_condensed_gpu(&gpu, &coords, 2, 2);
        assert!(
            result.is_ok(),
            "pairwise_l2_condensed_gpu should succeed with valid input"
        );
    }
}
