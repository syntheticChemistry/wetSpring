// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch Random Forest inference.
//!
//! Dispatches one thread per (sample, tree) pair. Each thread traverses
//! its tree for one sample. Results reduced on CPU via majority vote.
//!
//! Uses a SoA (Structure of Arrays) layout with separate buffers for
//! node features (i32), thresholds (f64), and children (i32) to avoid
//! bitcast issues in WGSL.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::random_forest::{RandomForest, RfPrediction};

const RF_WGSL: &str = include_str!("../shaders/rf_batch_inference.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RfGpuParams {
    n_samples: u32,
    n_trees: u32,
    n_nodes_max: u32,
    n_features: u32,
}

pub struct RandomForestGpu {
    device: Arc<WgpuDevice>,
}

impl RandomForestGpu {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }

    /// Run batch RF inference on GPU, returns per-sample predictions.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub fn predict_batch(
        &self,
        forest: &RandomForest,
        samples: &[Vec<f64>],
    ) -> crate::error::Result<Vec<RfPrediction>> {
        let n_samples = samples.len();
        let n_trees = forest.n_trees();
        let n_features = forest.n_features();
        let n_classes = forest.n_classes();

        if n_samples == 0 || n_trees == 0 {
            return Ok(Vec::new());
        }

        let n_nodes_max = (0..n_trees)
            .map(|i| forest.tree_at(i).n_nodes())
            .max()
            .unwrap_or(1);

        // SoA buffers
        let flat_nodes = n_trees * n_nodes_max;
        let mut node_features_buf: Vec<i32> = vec![-1; flat_nodes];
        let mut node_thresh_buf: Vec<f64> = vec![0.0; flat_nodes];
        let mut node_children_buf: Vec<i32> = vec![0; flat_nodes * 2];

        for t in 0..n_trees {
            let tree = forest.tree_at(t);
            let base = t * n_nodes_max;
            for n in 0..tree.n_nodes() {
                let node = tree.node_at(n);
                node_features_buf[base + n] = node.feature;
                node_thresh_buf[base + n] = node.threshold;
                let coff = (base + n) * 2;
                if node.feature < 0 {
                    node_children_buf[coff] = node.prediction.unwrap_or(0) as i32;
                    node_children_buf[coff + 1] = -1;
                } else {
                    node_children_buf[coff] = node.left_child;
                    node_children_buf[coff + 1] = node.right_child;
                }
            }
        }

        // Flatten features: [n_samples × n_features]
        let mut features_flat: Vec<f64> = vec![0.0; n_samples * n_features];
        for (i, sample) in samples.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                if j < n_features {
                    features_flat[i * n_features + j] = val;
                }
            }
        }

        let dev = &self.device;
        let d = dev.device();

        let params = RfGpuParams {
            n_samples: n_samples as u32,
            n_trees: n_trees as u32,
            n_nodes_max: n_nodes_max as u32,
            n_features: n_features as u32,
        };

        let params_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let nf_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node features"),
            contents: bytemuck::cast_slice(&node_features_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nt_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node thresholds"),
            contents: bytemuck::cast_slice(&node_thresh_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nc_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node children"),
            contents: bytemuck::cast_slice(&node_children_buf),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let feat_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF features"),
            contents: bytemuck::cast_slice(&features_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF predictions"),
            contents: bytemuck::cast_slice(&vec![0u32; n_samples * n_trees]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let patched = ShaderTemplate::for_driver_auto(RF_WGSL, false);
        let module = dev.compile_shader(&patched, Some("RfBatchInference"));
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RfBatchInference"),
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
                    resource: params_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nf_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nt_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nc_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: feat_gpu.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_gpu.as_entire_binding(),
                },
            ],
        });

        let total = (n_samples * n_trees) as u32;
        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RF batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let raw = dev
            .read_buffer_u32(&out_gpu, n_samples * n_trees)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        // Majority vote per sample
        let mut results = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let mut votes = vec![0usize; n_classes];
            for t in 0..n_trees {
                let pred = raw[s * n_trees + t] as usize;
                if pred < n_classes {
                    votes[pred] += 1;
                }
            }
            let (class, &max_votes) = votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .unwrap_or((0, &0));
            let confidence = max_votes as f64 / n_trees as f64;
            results.push(RfPrediction {
                class,
                votes,
                confidence,
            });
        }

        Ok(results)
    }
}
