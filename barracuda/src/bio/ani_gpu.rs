// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch pairwise ANI (Average Nucleotide Identity).
//!
//! One thread per sequence pair — embarrassingly parallel. Each thread
//! walks the alignment, counts identical non-gap bases, and divides.
//!
//! Uses a local WGSL shader (`ani_batch_f64.wgsl`) — a ToadStool
//! absorption candidate following Write → Absorb → Lean.
//!
//! # GPU Strategy
//!
//! ANI is position-sequential but pair-parallel. For N pairs of length
//! L, dispatch N threads. No transcendentals needed (integer counting
//! + one f64 division), so no polyfill required.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const ANI_WGSL: &str = include_str!("../shaders/ani_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AniGpuParams {
    n_pairs: u32,
    max_seq_len: u32,
}

/// GPU batch ANI result.
pub struct AniGpuResult {
    /// ANI values per pair, in [0.0, 1.0].
    pub ani_values: Vec<f64>,
    /// Aligned positions per pair.
    pub aligned_counts: Vec<u32>,
    /// Identical positions per pair.
    pub identical_counts: Vec<u32>,
}

pub struct AniGpu {
    device: Arc<WgpuDevice>,
}

impl AniGpu {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }

    /// Compute ANI for a batch of sequence pairs on the GPU.
    ///
    /// Sequences are `(seq_a, seq_b)` pairs of equal-length byte slices.
    /// Bases: A/C/G/T as ASCII. Gaps (`-`, `.`) and `N` are excluded.
    #[allow(clippy::cast_possible_truncation)]
    pub fn batch_ani(&self, pairs: &[(&[u8], &[u8])]) -> crate::error::Result<AniGpuResult> {
        let n_pairs = pairs.len();
        if n_pairs == 0 {
            return Ok(AniGpuResult {
                ani_values: Vec::new(),
                aligned_counts: Vec::new(),
                identical_counts: Vec::new(),
            });
        }

        let max_seq_len = pairs
            .iter()
            .map(|(a, b)| a.len().max(b.len()))
            .max()
            .unwrap_or(0);

        let encode_base = |b: u8| -> u32 {
            match b.to_ascii_uppercase() {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                b'-' | b'.' => 4,
                b'N' => 5,
                _ => 5,
            }
        };

        let mut seq_a_flat: Vec<u32> = vec![5; n_pairs * max_seq_len];
        let mut seq_b_flat: Vec<u32> = vec![5; n_pairs * max_seq_len];
        for (i, (a, b)) in pairs.iter().enumerate() {
            let base = i * max_seq_len;
            for (j, &byte) in a.iter().enumerate() {
                seq_a_flat[base + j] = encode_base(byte);
            }
            for (j, &byte) in b.iter().enumerate() {
                seq_b_flat[base + j] = encode_base(byte);
            }
        }

        let dev = &self.device;
        let d = dev.device();

        let params = AniGpuParams {
            n_pairs: n_pairs as u32,
            max_seq_len: max_seq_len as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let seq_a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI seq_a"),
            contents: bytemuck::cast_slice(&seq_a_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let seq_b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI seq_b"),
            contents: bytemuck::cast_slice(&seq_b_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ani_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI out"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let aligned_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI aligned"),
            contents: bytemuck::cast_slice(&vec![0u32; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let identical_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI identical"),
            contents: bytemuck::cast_slice(&vec![0u32; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let patched = ShaderTemplate::for_driver_auto(ANI_WGSL, false);
        let module = dev.compile_shader(&patched, Some("AniBatchF64"));
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AniBatchF64"),
            layout: None,
            module: &module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seq_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seq_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ani_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: aligned_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: identical_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ANI batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n_pairs as u32).div_ceil(256), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let ani_values = dev
            .read_buffer_f64(&ani_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let aligned_counts = dev
            .read_buffer_u32(&aligned_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let identical_counts = dev
            .read_buffer_u32(&identical_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(AniGpuResult {
            ani_values,
            aligned_counts,
            identical_counts,
        })
    }
}
