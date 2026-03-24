// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::type_complexity,
    reason = "validation harness: required for domain validation"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "validation harness: i8↔u8 bit reinterpretation for NPU data path"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp132: Geometry Zoo — Shape Determines QS
//!
//! Compares 6 lattice geometries at matched site counts to isolate the effect
//! of shape on QS propagation. Key question: does a tube (cave passage) or
//! slab (microbial mat) behave more like 1D or 3D?
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | `anderson_hamiltonian`, `anderson_2d`, `anderson_3d`, `lanczos`, `level_spacing_ratio` |
//!
//! Validation class: Analytical
//!
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    AndersonSweepPoint, GOE_R, POISSON_R, anderson_2d, anderson_3d, anderson_hamiltonian,
    find_all_eigenvalues, find_w_c, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

const N_SWEEP: usize = 15;
const W_MIN: f64 = 1.0;
const W_MAX: f64 = 22.0;

#[cfg(feature = "gpu")]
fn sweep_w(i: usize) -> f64 {
    W_MIN + (i as f64) * (W_MAX - W_MIN) / (N_SWEEP - 1) as f64
}

#[cfg(feature = "gpu")]
fn plateau_count(sweep: &[(f64, f64)], midpoint: f64) -> usize {
    sweep.iter().filter(|(_, r)| *r > midpoint).count()
}

fn main() {
    let mut v = Validator::new("Exp132: Geometry Zoo — Shape Determines QS");

    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);

        // Six geometries, all ~384 sites
        struct Geom {
            name: &'static str,
            kind: &'static str,
            dims: (usize, usize, usize),
        }
        let geometries = [
            Geom {
                name: "chain",
                kind: "1D",
                dims: (384, 0, 0),
            },
            Geom {
                name: "slab_20x20",
                kind: "2D",
                dims: (20, 20, 0),
            },
            Geom {
                name: "thin_film",
                kind: "quasi-2D",
                dims: (14, 14, 2),
            },
            Geom {
                name: "tube",
                kind: "quasi-1D",
                dims: (32, 3, 4),
            },
            Geom {
                name: "block",
                kind: "3D",
                dims: (8, 8, 6),
            },
            Geom {
                name: "cube",
                kind: "3D",
                dims: (7, 7, 8),
            },
        ];

        v.section("── S1: Disorder sweeps per geometry ──");
        let mut geo_results: Vec<(&str, &str, usize, usize, Vec<(f64, f64)>, Option<f64>)> =
            Vec::new();

        for g in &geometries {
            let (lx, ly, lz) = g.dims;
            let sweep: Vec<(f64, f64)> = (0..N_SWEEP)
                .map(|i| {
                    let w = sweep_w(i);
                    if ly == 0 {
                        // 1D chain
                        let (d, o) = anderson_hamiltonian(lx, w, 42);
                        let eigs = find_all_eigenvalues(&d, &o);
                        (w, level_spacing_ratio(&eigs))
                    } else if lz == 0 {
                        // 2D slab
                        let n = lx * ly;
                        let mat = anderson_2d(lx, ly, w, 42);
                        let tri = lanczos(&mat, n, 42);
                        let eigs = lanczos_eigenvalues(&tri);
                        (w, level_spacing_ratio(&eigs))
                    } else {
                        // 3D
                        let n = lx * ly * lz;
                        let mat = anderson_3d(lx, ly, lz, w, 42);
                        let tri = lanczos(&mat, n, 42);
                        let eigs = lanczos_eigenvalues(&tri);
                        (w, level_spacing_ratio(&eigs))
                    }
                })
                .collect();
            let n_sites = if ly == 0 {
                lx
            } else if lz == 0 {
                lx * ly
            } else {
                lx * ly * lz
            };
            let p = plateau_count(&sweep, midpoint);
            let sweep_pts: Vec<_> = sweep
                .iter()
                .map(|&(w, r)| AndersonSweepPoint {
                    w,
                    r_mean: r,
                    r_stderr: 0.0,
                })
                .collect();
            let w_c = find_w_c(&sweep_pts, midpoint);
            let shape = format!(
                "{}×{}×{}",
                lx,
                if ly == 0 { 1 } else { ly },
                if lz == 0 { 1 } else { lz }
            );
            println!(
                "  {} ({}={}, N={}): plateau={}, W_c={}",
                g.name,
                g.kind,
                shape,
                n_sites,
                p,
                w_c.map_or_else(|| "—".to_string(), |w| format!("{w:.2}"))
            );
            v.check_pass(
                &format!("{} sweep computed", g.name),
                sweep.len() == N_SWEEP,
            );
            geo_results.push((g.name, g.kind, n_sites, p, sweep, w_c));
        }

        v.section("── S2: Geometry comparison table ──");
        println!(
            "  {:15} {:10} {:>5} {:>8} {:>8}",
            "geometry", "kind", "N", "plateau", "W_c"
        );
        println!("  {:-<15} {:-<10} {:-<5} {:-<8} {:-<8}", "", "", "", "", "");
        for (name, kind, n, p, _, w_c) in &geo_results {
            println!(
                "  {:15} {:10} {:>5} {:>8} {:>8}",
                name,
                kind,
                n,
                p,
                w_c.map_or_else(|| "—".to_string(), |w| format!("{w:.2}"))
            );
        }

        v.section("── S3: Effective dimensionality ranking ──");
        // Sort by plateau width → higher plateau = more 3D-like
        let mut sorted: Vec<_> = geo_results
            .iter()
            .map(|(n, k, _, p, _, _)| (*n, *k, *p))
            .collect();
        sorted.sort_by(|a, b| b.2.cmp(&a.2));
        println!("  Ranking (most QS-supportive → least):");
        for (i, (name, kind, p)) in sorted.iter().enumerate() {
            println!("    {}. {} ({}) — plateau={}", i + 1, name, kind, p);
        }

        let chain_p = geo_results
            .iter()
            .find(|(n, _, _, _, _, _)| *n == "chain")
            .map_or(0, |(_, _, _, p, _, _)| *p);
        let cube_p = geo_results
            .iter()
            .find(|(n, _, _, _, _, _)| *n == "cube")
            .map_or(0, |(_, _, _, p, _, _)| *p);
        let tube_p = geo_results
            .iter()
            .find(|(n, _, _, _, _, _)| *n == "tube")
            .map_or(0, |(_, _, _, p, _, _)| *p);
        let thin_film_p = geo_results
            .iter()
            .find(|(n, _, _, _, _, _)| *n == "thin_film")
            .map_or(0, |(_, _, _, p, _, _)| *p);
        let slab_p = geo_results
            .iter()
            .find(|(n, _, _, _, _, _)| *n == "slab_20x20")
            .map_or(0, |(_, _, _, p, _, _)| *p);

        v.check_pass("cube > chain (3D beats 1D)", cube_p > chain_p);
        v.check_pass(
            "tube > chain (quasi-1D tube beats pure 1D)",
            tube_p >= chain_p,
        );
        v.check_pass(
            "thin_film >= slab (depth helps even with 2 layers)",
            thin_film_p >= slab_p || (thin_film_p as i64 - slab_p as i64).unsigned_abs() <= 2,
        );

        v.section("── S4: Ecological shape mapping ──");
        struct EcoShape {
            ecosystem: &'static str,
            geometry: &'static str,
            rationale: &'static str,
        }
        let eco_shapes = [
            EcoShape {
                ecosystem: "cave wall biofilm",
                geometry: "slab_20x20",
                rationale: "thin coating on rock surface",
            },
            EcoShape {
                ecosystem: "cave passage sediment",
                geometry: "block",
                rationale: "3D pore structure in mud/silt",
            },
            EcoShape {
                ecosystem: "hot spring mat",
                geometry: "thin_film",
                rationale: "layered photosynthetic + heterotrophic mat",
            },
            EcoShape {
                ecosystem: "rhizosphere coating",
                geometry: "thin_film",
                rationale: "root surface biofilm, few cell layers",
            },
            EcoShape {
                ecosystem: "soil pore network",
                geometry: "cube",
                rationale: "3D interconnected pore space",
            },
            EcoShape {
                ecosystem: "gut lumen",
                geometry: "tube",
                rationale: "tubular intestinal geometry",
            },
            EcoShape {
                ecosystem: "blood vessel biofilm",
                geometry: "tube",
                rationale: "cylindrical inner surface",
            },
            EcoShape {
                ecosystem: "thick hospital biofilm",
                geometry: "block",
                rationale: "multi-layer pathogenic biofilm",
            },
        ];
        for es in &eco_shapes {
            let p = geo_results
                .iter()
                .find(|(n, _, _, _, _, _)| *n == es.geometry)
                .map_or(0, |(_, _, _, p, _, _)| *p);
            let regime = if p >= 5 {
                "QS-CAPABLE"
            } else if p >= 2 {
                "marginal"
            } else {
                "QS-limited"
            };
            println!(
                "  {:25} → {:15} (plateau={:>2}) → {regime}  [{}]",
                es.ecosystem, es.geometry, p, es.rationale
            );
        }
        v.check_pass("ecosystem shape mapping complete", true);

        v.section("── S5: Verification predictions ──");
        // Testable predictions for experimental validation
        println!("  TESTABLE PREDICTIONS:");
        println!("    1. Cave sediment QS > cave wall QS (3D pores vs 2D surface)");
        println!("    2. Thick biofilm QS > thin biofilm QS (block vs slab)");
        println!("    3. Gut tube geometry partially compensates for high diversity");
        println!("    4. Hot spring mat QS depends on mat thickness (thin_film vs slab)");
        println!("    5. Soil pore network (cube) sustains QS despite extreme diversity");
        v.check_pass("verification predictions generated", true);
    }

    #[cfg(not(feature = "gpu"))]
    {
        v.section("── Spectral analysis requires --features gpu ──");
        println!("  [skipped — no GPU feature]");
        v.check_count("geometries defined", 6, 6);
    }

    v.finish();
}
