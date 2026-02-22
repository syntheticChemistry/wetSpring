// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated DADA2 denoising via `ToadStool`.
//!
//! The DADA2 EM loop has two cost centres:
//!   1. `assign_to_centers` (E-step): O(seqs × centers × `seq_length`) — **GPU**
//!   2. `find_new_centers` (split): O(seqs × `seq_length`) `log_p_error` — **GPU**
//!
//! Everything else (error model update, Poisson test, convergence) stays on CPU.
//! The GPU computes all `log_p_error(seq, center)` pairs in a single batch dispatch
//! via `barracuda::ops::bio::dada2::Dada2EStepGpu`.
//!
//! # Key design: no GPU transcendentals
//!
//! The error model `ln(err[from][to][qual])` is precomputed on CPU and uploaded
//! as a flat f64 lookup table. The GPU shader only does f64 addition (sum over
//! positions) — no exp, log, or other transcendentals.

use crate::bio::dada2::{
    self, Asv, Dada2Params, Dada2Stats, ErrorModel, MAX_ERR_ITERS, MAX_QUAL, MIN_ERR, NUM_BASES,
};
use crate::bio::derep::UniqueSequence;
use crate::error::{Error, Result};
use barracuda::device::WgpuDevice;
use barracuda::ops::bio::dada2::Dada2EStepGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Pre-compiled DADA2 E-step pipeline via `ToadStool`.
pub struct Dada2Gpu {
    device: Arc<WgpuDevice>,
    inner: Dada2EStepGpu,
}

impl Dada2Gpu {
    /// Create a new DADA2 GPU E-step instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let inner = Dada2EStepGpu::new(Arc::clone(&device))
            .map_err(|e| Error::Gpu(format!("Dada2EStepGpu: {e}")))?;
        Ok(Self { device, inner })
    }

    /// Compute `log_p_error` for all (seq, center) pairs in a single GPU dispatch.
    ///
    /// Returns an `n_seqs × n_centers` matrix (row-major).
    #[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
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
        let d = self.device.device();

        let bases_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 bases"),
            contents: bytemuck::cast_slice(bases),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let quals_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 quals"),
            contents: bytemuck::cast_slice(quals),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lengths_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 lengths"),
            contents: bytemuck::cast_slice(lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let centers_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 centers"),
            contents: bytemuck::cast_slice(center_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let log_err_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 log_err"),
            contents: bytemuck::cast_slice(log_err_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scores_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dada2 scores"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; total_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch(
                n_seqs as u32,
                n_centers as u32,
                max_len as u32,
                &bases_buf,
                &quals_buf,
                &lengths_buf,
                &centers_buf,
                &log_err_buf,
                &scores_buf,
            )
            .map_err(|e| Error::Gpu(format!("Dada2 dispatch: {e}")))?;

        d.poll(wgpu::Maintain::Wait);

        self.device
            .read_buffer_f64(&scores_buf, total_pairs)
            .map_err(|e| Error::Gpu(format!("Dada2 readback: {e}")))
    }
}

/// GPU-accelerated DADA2 denoising.
///
/// Same EM algorithm as CPU, but the E-step (`log_p_error` for all pairs)
/// runs as a single GPU dispatch per iteration.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
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

        for (i, slot) in partition.iter_mut().enumerate() {
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
            *slot = best_center;
        }

        for _ in 0..MAX_ERR_ITERS {
            let new_err = estimate_error_model(&seqs, &partition, &centers);
            if err_model_converged(&err, &new_err) {
                err = new_err;
                break;
            }
            err = new_err;
        }

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

        let new_centers = find_new_centers_from_matrix(
            &seqs,
            &partition,
            &centers,
            &split_scores,
            params.omega_a,
        );

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

    for (i, slot) in partition.iter_mut().enumerate() {
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
        *slot = best_center;
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

#[allow(clippy::cast_possible_truncation)]
const fn base_to_idx(b: u8) -> u32 {
    dada2::base_to_idx(b) as u32
}

#[allow(clippy::cast_possible_truncation)]
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
        for (j, &q) in seq
            .representative_quality
            .iter()
            .enumerate()
            .take(seq.sequence.len())
        {
            quals[base_offset + j] = u32::from(q.saturating_sub(33).min(41));
        }
    }

    (bases, quals, lengths)
}

fn init_error_model() -> ErrorModel {
    dada2::init_error_model()
}

#[allow(clippy::needless_range_loop)]
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

fn estimate_error_model(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    centers: &[usize],
) -> ErrorModel {
    dada2::estimate_error_model(seqs, partition, centers)
}

fn err_model_converged(old: &ErrorModel, new: &ErrorModel) -> bool {
    dada2::err_model_converged(old, new)
}

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

fn build_asvs(seqs: &[&UniqueSequence], partition: &[usize], centers: &[usize]) -> Vec<Asv> {
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
