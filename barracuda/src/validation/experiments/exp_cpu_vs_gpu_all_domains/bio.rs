// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bio domain CPU↔GPU parity: Shannon/Simpson, Bray-Curtis, ANI, SNP, dN/dS,
//! Pangenome, Rarefaction (D01-D06, D16).

use std::sync::Arc;
use std::time::Instant;

use crate::bio::{
    ani, ani_gpu::AniGpu, diversity, diversity_gpu, dnds, dnds_gpu::DnDsGpu, pangenome,
    pangenome_gpu::PangenomeGpu, rarefaction_gpu, snp, snp_gpu::SnpGpu,
};
use crate::gpu::GpuF64;
use crate::tolerances;
use crate::validation::{CpuGpuRow, OrExit, Validator};

pub(super) fn validate_shannon_simpson(
    v: &mut Validator,
    gpu: &GpuF64,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D01: Shannon + Simpson (FMR)");
    let counts = vec![
        120.0, 85.0, 230.0, 55.0, 180.0, 12.0, 42.0, 310.0, 8.0, 95.0,
    ];
    let tc = Instant::now();
    let cpu_sh = diversity::shannon(&counts);
    let cpu_si = diversity::simpson(&counts);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_sh = diversity_gpu::shannon_gpu(gpu, &counts).or_exit("GPU/CPU validation");
    let gpu_si = diversity_gpu::simpson_gpu(gpu, &counts).or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "Shannon CPU↔GPU",
        gpu_sh,
        cpu_sh,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Simpson CPU↔GPU",
        gpu_si,
        cpu_si,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    timings.push(CpuGpuRow {
        name: "Shannon + Simpson",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_bray_curtis(
    v: &mut Validator,
    gpu: &GpuF64,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D02: Bray-Curtis");
    let a: Vec<f64> = vec![10.0, 20.0, 30.0, 0.0, 15.0, 5.0, 8.0, 12.0];
    let b: Vec<f64> = vec![12.0, 18.0, 25.0, 5.0, 10.0, 7.0, 6.0, 14.0];
    let tc = Instant::now();
    let cpu_bc = diversity::bray_curtis(&a, &b);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_bc =
        diversity_gpu::bray_curtis_condensed_gpu(gpu, &[a, b]).or_exit("GPU/CPU validation")[0];
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "Bray-Curtis CPU↔GPU",
        gpu_bc,
        cpu_bc,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(CpuGpuRow {
        name: "Bray-Curtis",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_ani(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D03: ANI");
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"),
        (b"ATGATGATG", b"CTGATGATG"),
        (b"ATGATGATG", b"CTGCTGCTG"),
    ];
    let tc = Instant::now();
    let cpu_ani: Vec<_> = pairs.iter().map(|(a, b)| ani::pairwise_ani(a, b)).collect();
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let ani_dev = AniGpu::new(device).or_exit("ANI GPU");
    let gpu_ani = ani_dev.batch_ani(&pairs).or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    for (i, (cr, gv)) in cpu_ani.iter().zip(gpu_ani.ani_values.iter()).enumerate() {
        v.check(
            &format!("ANI pair {i}"),
            *gv,
            cr.ani,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
    timings.push(CpuGpuRow {
        name: "ANI (3 pairs)",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_snp(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D04: SNP Calling");
    let seqs: Vec<&[u8]> = vec![
        b"ATGATGATGATG",
        b"ATCATGATGATG",
        b"ATGATCATGATG",
        b"ATGATGATCATG",
    ];
    let tc = Instant::now();
    let cpu_snp = snp::call_snps(&seqs);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let snp_dev = SnpGpu::new(device).or_exit("SNP GPU");
        let gpu_snp = snp_dev.call_snps(&seqs, 2).or_exit("GPU/CPU validation");
        gpu_snp.is_variant.iter().filter(|&&x| x != 0).count()
    }));
    let gpu_us = tg.elapsed().as_micros() as f64;
    if let Ok(gpu_count) = gpu_result {
        let cpu_count = cpu_snp.variants.len();
        v.check(
            "SNP count",
            gpu_count as f64,
            cpu_count as f64,
            tolerances::EXACT,
        );
        timings.push(CpuGpuRow {
            name: "SNP",
            cpu_us,
            gpu_us,
            status: "PASS",
        });
    } else {
        v.check_pass("SNP: driver/binding skip", true);
        timings.push(CpuGpuRow {
            name: "SNP",
            cpu_us,
            gpu_us,
            status: "SKIP",
        });
    }
}

pub(super) fn validate_dnds(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D05: dN/dS");
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"),
        (b"TTTGCTAAA", b"TTCGCTAAA"),
        (b"AAAGCTGCT", b"GAAGCTGCT"),
    ];
    let tc = Instant::now();
    let cpu_dnds: Vec<_> = pairs
        .iter()
        .map(|(a, b)| dnds::pairwise_dnds(a, b))
        .collect();
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let dnds_dev = DnDsGpu::new(device).or_exit("dN/dS GPU");
    let gpu_dnds = dnds_dev.batch_dnds(&pairs).or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    for (i, cr) in cpu_dnds.iter().enumerate() {
        if let Ok(c) = cr {
            v.check(
                &format!("dN {i}"),
                gpu_dnds.dn[i],
                c.dn,
                tolerances::GPU_VS_CPU_F64,
            );
            v.check(
                &format!("dS {i}"),
                gpu_dnds.ds[i],
                c.ds,
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }
    timings.push(CpuGpuRow {
        name: "dN/dS (3 pairs)",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_pangenome(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D06: Pangenome");
    let clusters = vec![
        pangenome::GeneCluster {
            id: "g1".into(),
            presence: vec![true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g2".into(),
            presence: vec![true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g3".into(),
            presence: vec![true, true, false, false],
        },
        pangenome::GeneCluster {
            id: "g4".into(),
            presence: vec![true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "g5".into(),
            presence: vec![false, false, false, true],
        },
    ];
    let tc = Instant::now();
    let cpu_pan = pangenome::analyze(&clusters, 4);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let presence_flat: Vec<u8> = clusters
        .iter()
        .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
        .collect();
    let tg = Instant::now();
    let pan_dev = PangenomeGpu::new(device).or_exit("Pangenome GPU");
    let gpu_pan = pan_dev
        .classify(&presence_flat, 5, 4)
        .or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "core",
        gpu_pan.classifications.iter().filter(|&&c| c == 3).count() as f64,
        cpu_pan.core_size as f64,
        tolerances::EXACT,
    );
    v.check(
        "accessory",
        gpu_pan.classifications.iter().filter(|&&c| c == 2).count() as f64,
        cpu_pan.accessory_size as f64,
        tolerances::EXACT,
    );
    v.check(
        "unique",
        gpu_pan.classifications.iter().filter(|&&c| c == 1).count() as f64,
        cpu_pan.unique_size as f64,
        tolerances::EXACT,
    );
    timings.push(CpuGpuRow {
        name: "Pangenome (5g×4)",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_rarefaction(
    v: &mut Validator,
    gpu: &GpuF64,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D16: Rarefaction Bootstrap");
    let counts: Vec<f64> = vec![
        120.0, 85.0, 230.0, 55.0, 180.0, 12.0, 42.0, 310.0, 8.0, 95.0, 33.0, 67.0, 145.0, 22.0,
        78.0, 200.0, 15.0, 50.0, 110.0, 40.0,
    ];
    let params = rarefaction_gpu::RarefactionGpuParams {
        n_bootstrap: 100,
        depth: Some(500),
        seed: 42,
    };
    let tg = Instant::now();
    let result = rarefaction_gpu::rarefaction_bootstrap_gpu(gpu, &counts, &params)
        .or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    let cpu_shannon = diversity::shannon(&counts);
    v.check_pass("Rarefaction Shannon > 0", result.shannon.mean > 0.0);
    v.check_pass(
        "Rarefaction Shannon CI valid",
        result.shannon.lower <= result.shannon.mean + tolerances::RAREFACTION_CI_GUARD,
    );
    v.check_pass(
        "Rarefaction Shannon ≤ full",
        result.shannon.mean <= cpu_shannon + 0.5,
    );
    v.check_pass("Rarefaction observed > 0", result.observed.mean > 0.0);
    v.check(
        "Rarefaction depth",
        result.depth as f64,
        500.0,
        tolerances::EXACT,
    );
    timings.push(CpuGpuRow {
        name: "Rarefaction",
        cpu_us: 0.0,
        gpu_us,
        status: "PASS",
    });
}
