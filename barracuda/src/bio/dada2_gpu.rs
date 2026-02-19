// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated DADA2 denoising — E-step on GPU, control flow on CPU.
//!
//! The DADA2 EM loop has two cost centres:
//!   1. `assign_to_centers` (E-step): O(seqs × centers × seq_length) — **GPU**
//!   2. `find_new_centers` (split): O(seqs × seq_length) log_p_error — **GPU**
//!
//! Everything else (error model update, Poisson test, convergence) stays on CPU.
//! The GPU computes all `log_p_error(seq, center)` pairs in a single batch dispatch.
//!
//! # Key design: no GPU transcendentals
//!
//! The error model `ln(err[from][to][qual])` is precomputed on CPU and uploaded
//! as a flat f64 lookup table. The GPU shader only does f64 addition (sum over
//! positions) — no exp, log, or other transcendentals. This avoids driver-specific
//! f64 transcendental issues entirely.
//!
//! # ToadStool absorption path
//!
//! - `BatchPairReduce<f64>` — per-pair parallel reduction primitive
//! - Pre-compiled pipeline cache (like GemmCached)
//! - BufferPool for sequence data persistence across iterations

use crate::bio::dada2::{self, Asv, Dada2Params, Dada2Stats};
use crate::bio::derep::UniqueSequence;
use crate::error::{Error, Result};
use barracuda::device::{TensorContext, WgpuDevice};
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const E_STEP_WGSL: &str = include_str!("../shaders/dada2_e_step.wgsl");

const NUM_BASES: usize = 4;
const MAX_QUAL: usize = 42;
const MIN_ERR: f64 = 1e-7;
const MAX_ERR: f64 = 0.25;
const MAX_ERR_ITERS: usize = 6;

type ErrorModel = [[[f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EStepParams {
    n_seqs: u32,
    n_centers: u32,
    max_len: u32,
    _pad: u32,
}

/// Pre-compiled DADA2 E-step pipeline for batch log_p_error computation.
pub struct Dada2Gpu {
    device: Arc<WgpuDevice>,
    ctx: Arc<TensorContext>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl Dada2Gpu {
    pub fn new(device: Arc<WgpuDevice>, ctx: Arc<TensorContext>) -> Self {
        let patched = ShaderTemplate::for_driver_auto(
            E_STEP_WGSL,
            device.needs_f64_exp_log_workaround(),
        );
        let shader = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dada2 EStep"),
                source: wgpu::ShaderSource::Wgsl(patched.into()),
            });

        let bgl = device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dada2 BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let pl = device
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dada2 PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            device
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dada2 EStep f64"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: "e_step",
                    cache: None,
                    compilation_options: Default::default(),
                });

        Self {
            device,
            ctx,
            pipeline,
            bgl,
        }
    }

    /// Compute log_p_error for all (seq, center) pairs in a single GPU dispatch.
    ///
    /// Returns an `n_seqs × n_centers` matrix (row-major).
    fn batch_log_p_error(
        &self,
        bases: &[u32],
        quals: &[u32],
        lengths: &[u32],
        center_indices: &[u32],
        log_err_flat: &[f64],
        n_seqs: usize,
        n_centers: usize,
        max_len: usize,
    ) -> Result<Vec<f64>> {
        let total_pairs = n_seqs * n_centers;
        let pool = self.ctx.buffer_pool();

        let params = EStepParams {
            n_seqs: n_seqs as u32,
            n_centers: n_centers as u32,
            max_len: max_len as u32,
            _pad: 0,
        };
        let params_buf = self.device.create_uniform_buffer("Dada2 Params", &params);

        let bases_buf = pool.acquire_pooled(bases.len() * 4);
        self.device
            .queue()
            .write_buffer(&bases_buf, 0, bytemuck::cast_slice(bases));

        let quals_buf = pool.acquire_pooled(quals.len() * 4);
        self.device
            .queue()
            .write_buffer(&quals_buf, 0, bytemuck::cast_slice(quals));

        let lengths_buf = pool.acquire_pooled(lengths.len() * 4);
        self.device
            .queue()
            .write_buffer(&lengths_buf, 0, bytemuck::cast_slice(lengths));

        let centers_buf = pool.acquire_pooled(center_indices.len() * 4);
        self.device
            .queue()
            .write_buffer(&centers_buf, 0, bytemuck::cast_slice(center_indices));

        let log_err_buf = pool.acquire_pooled(log_err_flat.len() * 8);
        self.device
            .queue()
            .write_buffer(&log_err_buf, 0, bytemuck::cast_slice(log_err_flat));

        let scores_buf = pool.acquire_pooled(total_pairs * 8);

        let bg = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Dada2 BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bases_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: quals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lengths_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: centers_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: log_err_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: scores_buf.as_entire_binding(),
                    },
                ],
            });

        let wg_x = (total_pairs as u32).div_ceil(256);
        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dada2 Encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dada2 EStep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.device.queue().submit(Some(encoder.finish()));

        self.device
            .read_buffer_f64(&scores_buf, total_pairs)
            .map_err(|e| Error::Gpu(format!("Dada2 readback: {e}")))
    }
}

/// GPU-accelerated DADA2 denoising.
///
/// Same EM algorithm as CPU, but the E-step (log_p_error for all pairs)
/// runs as a single GPU dispatch per iteration.
#[allow(clippy::cast_precision_loss)]
pub fn denoise_gpu(
    dada2: &Dada2Gpu,
    seqs: &[UniqueSequence],
    params: &Dada2Params,
) -> Result<(Vec<Asv>, Dada2Stats)> {
    let seqs: Vec<&UniqueSequence> = seqs
        .iter()
        .filter(|s| s.abundance >= params.min_abundance && !s.sequence.is_empty())
        .collect();

    let input_uniques = seqs.len();
    let input_reads: usize = seqs.iter().map(|s| s.abundance).sum();

    if seqs.is_empty() {
        return Ok((
            vec![],
            Dada2Stats {
                input_uniques,
                input_reads,
                output_asvs: 0,
                output_reads: 0,
                iterations: 0,
            },
        ));
    }

    let max_len = seqs.iter().map(|s| s.sequence.len()).max().unwrap_or(0);
    let (bases, quals, lengths) = pack_sequences(&seqs, max_len);

    let mut err = init_error_model();
    let mut partition: Vec<usize> = vec![0; seqs.len()];
    let mut centers: Vec<usize> = vec![0];
    let mut last_n_centers = 0;
    let mut iters = 0;

    for _ in 0..params.max_iterations {
        iters += 1;

        let log_err_flat = flatten_log_error_model(&err);
        let center_indices: Vec<u32> = centers.iter().map(|&c| c as u32).collect();

        // E-step: GPU batch dispatch for all (seq, center) log_p_error values
        let scores = dada2.batch_log_p_error(
            &bases,
            &quals,
            &lengths,
            &center_indices,
            &log_err_flat,
            seqs.len(),
            centers.len(),
            max_len,
        )?;

        // Argmax on CPU (trivial: just find best center per sequence)
        for i in 0..seqs.len() {
            let row_start = i * centers.len();
            let mut best_center = centers[0];
            let mut best_lp = f64::NEG_INFINITY;
            for (ci, &c) in centers.iter().enumerate() {
                let lp = scores[row_start + ci];
                if lp > best_lp {
                    best_lp = lp;
                    best_center = c;
                }
            }
            partition[i] = best_center;
        }

        // M-step: error model re-estimation (CPU — cheap)
        for _ in 0..MAX_ERR_ITERS {
            let new_err = estimate_error_model(&seqs, &partition, &centers);
            if err_model_converged(&err, &new_err) {
                err = new_err;
                break;
            }
            err = new_err;
        }

        // Split step: GPU batch log_p_error for Poisson test
        let split_log_err = flatten_log_error_model(&err);
        let split_center_indices: Vec<u32> = centers.iter().map(|&c| c as u32).collect();

        let split_scores = dada2.batch_log_p_error(
            &bases,
            &quals,
            &lengths,
            &split_center_indices,
            &split_log_err,
            seqs.len(),
            centers.len(),
            max_len,
        )?;

        let new_centers =
            find_new_centers_from_matrix(&seqs, &partition, &centers, &split_scores, params.omega_a);

        for &c in &new_centers {
            if !centers.contains(&c) {
                centers.push(c);
            }
        }
        centers.sort_unstable();

        if centers.len() == last_n_centers {
            break;
        }
        last_n_centers = centers.len();
    }

    // Final assignment with GPU E-step
    let log_err_flat = flatten_log_error_model(&err);
    let center_indices: Vec<u32> = centers.iter().map(|&c| c as u32).collect();
    let scores = dada2.batch_log_p_error(
        &bases,
        &quals,
        &lengths,
        &center_indices,
        &log_err_flat,
        seqs.len(),
        centers.len(),
        max_len,
    )?;

    for i in 0..seqs.len() {
        let row_start = i * centers.len();
        let mut best_center = centers[0];
        let mut best_lp = f64::NEG_INFINITY;
        for (ci, &c) in centers.iter().enumerate() {
            let lp = scores[row_start + ci];
            if lp > best_lp {
                best_lp = lp;
                best_center = c;
            }
        }
        partition[i] = best_center;
    }

    let asvs = build_asvs(&seqs, &partition, &centers);
    let output_reads: usize = asvs.iter().map(|a| a.abundance).sum();
    let output_asvs = asvs.len();

    Ok((
        asvs,
        Dada2Stats {
            input_uniques,
            input_reads,
            output_asvs,
            output_reads,
            iterations: iters,
        },
    ))
}

// ── Data packing for GPU ─────────────────────────────────────────────────────

fn base_to_idx(b: u8) -> u32 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

fn pack_sequences(seqs: &[&UniqueSequence], max_len: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = seqs.len();
    let mut bases = vec![0u32; n * max_len];
    let mut quals = vec![0u32; n * max_len];
    let mut lengths = vec![0u32; n];

    for (i, seq) in seqs.iter().enumerate() {
        lengths[i] = seq.sequence.len() as u32;
        let base_offset = i * max_len;
        for (j, &b) in seq.sequence.iter().enumerate() {
            bases[base_offset + j] = base_to_idx(b);
        }
        for (j, &q) in seq.representative_quality.iter().enumerate().take(seq.sequence.len()) {
            quals[base_offset + j] = q.saturating_sub(33).min(41) as u32;
        }
    }

    (bases, quals, lengths)
}

fn init_error_model() -> ErrorModel {
    let mut err = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    for q in 0..MAX_QUAL {
        let p_err = (10.0_f64).powf(-(q as f64) / 10.0).clamp(MIN_ERR, MAX_ERR);
        for from in 0..NUM_BASES {
            for to in 0..NUM_BASES {
                if from == to {
                    err[from][to][q] = 1.0 - p_err;
                } else {
                    err[from][to][q] = p_err / 3.0;
                }
            }
        }
    }
    err
}

fn flatten_log_error_model(err: &ErrorModel) -> Vec<f64> {
    let mut flat = vec![0.0_f64; NUM_BASES * NUM_BASES * MAX_QUAL];
    for from in 0..NUM_BASES {
        for to in 0..NUM_BASES {
            for q in 0..MAX_QUAL {
                let idx = from * NUM_BASES * MAX_QUAL + to * MAX_QUAL + q;
                flat[idx] = err[from][to][q].max(MIN_ERR).ln();
            }
        }
    }
    flat
}

// ── CPU helper functions (replicated from dada2.rs for the GPU EM loop) ──────

#[allow(clippy::cast_precision_loss)]
fn estimate_error_model(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    _centers: &[usize],
) -> ErrorModel {
    let mut counts = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    let mut totals = [[0.0_f64; MAX_QUAL]; NUM_BASES];

    for (i, seq) in seqs.iter().enumerate() {
        let center_idx = partition[i];
        let center = seqs[center_idx];
        let len = seq.sequence.len().min(center.sequence.len());
        let weight = seq.abundance as f64;

        for pos in 0..len {
            let from = base_to_idx(center.sequence[pos]) as usize;
            let to = base_to_idx(seq.sequence[pos]) as usize;
            let q = seq
                .representative_quality
                .get(pos)
                .map_or(0, |&v| v.saturating_sub(33) as usize)
                .min(MAX_QUAL - 1);
            counts[from][to][q] += weight;
            totals[from][q] += weight;
        }
    }

    let mut err = init_error_model();
    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            if totals[from][q] > 0.0 {
                for to in 0..NUM_BASES {
                    let rate = counts[from][to][q] / totals[from][q];
                    err[from][to][q] = rate.clamp(MIN_ERR, 1.0 - MIN_ERR);
                }
            }
        }
    }

    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            let sum: f64 = (0..NUM_BASES).map(|to| err[from][to][q]).sum();
            if sum > 0.0 {
                for to in 0..NUM_BASES {
                    err[from][to][q] /= sum;
                }
            }
        }
    }

    err
}

fn err_model_converged(old: &ErrorModel, new: &ErrorModel) -> bool {
    let mut max_diff = 0.0_f64;
    for from in 0..NUM_BASES {
        for to in 0..NUM_BASES {
            for q in 0..MAX_QUAL {
                let diff = (old[from][to][q] - new[from][to][q]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }
    max_diff < 1e-6
}

/// Poisson upper-tail p-value using GPU E-step scores matrix.
#[allow(clippy::cast_precision_loss)]
fn find_new_centers_from_matrix(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    centers: &[usize],
    scores: &[f64],
    omega_a: f64,
) -> Vec<usize> {
    let n_centers = centers.len();
    let mut new_centers = Vec::new();

    for (i, seq) in seqs.iter().enumerate() {
        if centers.contains(&i) {
            continue;
        }
        let center_idx = partition[i];
        let center_slot = centers.iter().position(|&c| c == center_idx).unwrap_or(0);
        let log_p = scores[i * n_centers + center_slot];
        let lambda = (seqs[center_idx].abundance as f64) * log_p.exp();

        if lambda <= 0.0 {
            new_centers.push(i);
            continue;
        }

        let p_value = dada2::poisson_pvalue(seq.abundance, lambda);
        if p_value < omega_a {
            new_centers.push(i);
        }
    }

    new_centers
}

fn build_asvs(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    centers: &[usize],
) -> Vec<Asv> {
    let mut asvs: Vec<Asv> = centers
        .iter()
        .map(|&c| Asv {
            sequence: seqs[c].sequence.clone(),
            abundance: 0,
            n_members: 0,
        })
        .collect();

    for (i, &center_idx) in partition.iter().enumerate() {
        if let Some(asv_pos) = centers.iter().position(|&c| c == center_idx) {
            asvs[asv_pos].abundance += seqs[i].abundance;
            asvs[asv_pos].n_members += 1;
        }
    }

    asvs.sort_by(|a, b| b.abundance.cmp(&a.abundance));
    asvs
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
