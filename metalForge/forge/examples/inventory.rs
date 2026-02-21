// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discover and print all compute substrates on this machine.
//!
//! GPU discovery uses the same wgpu path that toadstool/barracuda uses.
//! NPU and CPU discovery are local probes.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example inventory
//! ```

use wetspring_forge::dispatch::{self, Workload};
use wetspring_forge::substrate::Capability;

fn main() {
    let substrates = wetspring_forge::inventory::discover();
    wetspring_forge::inventory::print_inventory(&substrates);

    println!();
    println!("Dispatch examples (life science workloads):");
    println!();

    let workloads = [
        Workload::new(
            "Felsenstein pruning",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "Diversity map-reduce",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        ),
        Workload::new(
            "HMM batch forward",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "Spectral cosine",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        ),
        Workload::new(
            "Taxonomy classify",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
        Workload::new(
            "PFAS anomaly detect",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
        Workload::new("FASTQ parsing", vec![Capability::F64Compute]),
        Workload::new(
            "ANI batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
    ];

    for work in &workloads {
        match dispatch::route(work, &substrates) {
            Some(d) => println!("  {:25} → {} ({:?})", work.name, d.substrate, d.reason),
            None => println!("  {:25} → NO CAPABLE SUBSTRATE", work.name),
        }
    }
}
