// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! wetSpring interactive dashboard — single binary for scientists.
//!
//! Discovers local GPU via wgpu, runs diversity + ODE + phylogenetics
//! computations on synthetic data, and pushes live results to petalTongue
//! (or dumps scenario JSON if no socket is available).
//!
//! # Usage
//!
//! ```bash
//! cargo run --features json --bin wetspring_dashboard
//! ```
//!
//! Provenance: wetSpring metrics dashboard binary

use std::fs;
use std::path::Path;
use std::process;

use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::scenarios;
use wetspring_barracuda::visualization::{EcologyScenario, ScenarioEdge, scenario_with_edges_json};

type ScenarioEntry<'a> = (&'a str, EcologyScenario, Vec<ScenarioEdge>);

#[expect(clippy::similar_names)]
fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║  wetSpring Scientist Dashboard                           ║");
    eprintln!("║  Sovereign bioinformatics — anyone with a GPU can run it ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0, 10.0, 20.0, 30.0],
        vec![20.0, 10.0, 35.0, 25.0, 15.0, 40.0, 5.0, 30.0],
    ];
    let labels: Vec<String> = vec!["Site_A".into(), "Site_B".into(), "Site_C".into()];

    eprintln!("Building scenarios from live BarraCUDA math...");
    let mut entries: Vec<ScenarioEntry<'_>> = Vec::new();

    let (eco, eco_e) = scenarios::ecology_scenario(&samples, &labels);
    entries.push(("ecology", eco, eco_e));

    let (dyn_s, dyn_e) = scenarios::dynamics_scenario();
    entries.push(("dynamics", dyn_s, dyn_e));

    let (fels, fels_e) = scenarios::felsenstein_scenario();
    entries.push(("felsenstein", fels, fels_e));

    let (place, place_e) = scenarios::placement_scenario();
    entries.push(("placement", place, place_e));

    let (uf, uf_e) = scenarios::unifrac_scenario();
    entries.push(("unifrac", uf, uf_e));

    let (dnds, dnds_e) = scenarios::dnds_scenario();
    entries.push(("dnds", dnds, dnds_e));

    let (mc, mc_e) = scenarios::molecular_clock_scenario();
    entries.push(("molecular_clock", mc, mc_e));

    let (rec, rec_e) = scenarios::reconciliation_scenario();
    entries.push(("reconciliation", rec, rec_e));

    let (pd, pd_e) = scenarios::phage_defense_scenario();
    entries.push(("phage_defense", pd, pd_e));

    let (bi, bi_e) = scenarios::bistable_scenario();
    entries.push(("bistable", bi, bi_e));

    let (co, co_e) = scenarios::cooperation_scenario();
    entries.push(("cooperation", co, co_e));

    let (ms, ms_e) = scenarios::multi_signal_scenario();
    entries.push(("multi_signal", ms, ms_e));

    let (cap, cap_e) = scenarios::capacitor_scenario();
    entries.push(("capacitor", cap, cap_e));

    let (qual, qual_e) = scenarios::quality_scenario();
    entries.push(("quality", qual, qual_e));

    let (dada2, dada2_e) = scenarios::dada2_scenario();
    entries.push(("dada2", dada2, dada2_e));

    let (tax, tax_e) = scenarios::taxonomy_scenario();
    entries.push(("taxonomy", tax, tax_e));

    let (po, po_e) = scenarios::pipeline_overview_scenario();
    entries.push(("pipeline_overview", po, po_e));

    let (snp, snp_e) = scenarios::snp_scenario();
    entries.push(("snp", snp, snp_e));

    let (pg, pg_e) = scenarios::population_genomics_scenario();
    entries.push(("population_genomics", pg, pg_e));

    let (ks, ks_e) = scenarios::kmer_spectrum_scenario();
    entries.push(("kmer_spectrum", ks, ks_e));

    let (sm, sm_e) = scenarios::spectral_match_scenario();
    entries.push(("spectral_match", sm, sm_e));

    let (ts, ts_e) = scenarios::tolerance_search_scenario();
    entries.push(("tolerance_search", ts, ts_e));

    let (pfas, pfas_e) = scenarios::pfas_overview_scenario();
    entries.push(("pfas_overview", pfas, pfas_e));

    let (dt, dt_e) = scenarios::decision_tree_scenario();
    entries.push(("decision_tree", dt, dt_e));

    let (rf, rf_e) = scenarios::random_forest_scenario();
    entries.push(("random_forest", rf, rf_e));

    let (esn, esn_e) = scenarios::esn_scenario();
    entries.push(("esn", esn, esn_e));

    let (full, full_e) = scenarios::full_ecology_scenario(&samples, &labels);
    entries.push(("scientist_dashboard", full, full_e));

    eprintln!("  {} scenarios built.\n", entries.len());

    let out_dir = Path::new("sandbox/scenarios");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("ERROR: cannot create {}: {e}", out_dir.display());
        process::exit(1);
    }

    let mut written = 0u32;
    for (name, scenario, edges) in &entries {
        let path = out_dir.join(format!("{name}.json"));
        match scenario_with_edges_json(scenario, edges) {
            Ok(json) => match fs::write(&path, &json) {
                Ok(()) => {
                    written += 1;
                }
                Err(e) => {
                    eprintln!("ERROR: write {}: {e}", path.display());
                }
            },
            Err(e) => {
                eprintln!("ERROR: serialize {name}: {e}");
            }
        }
    }
    eprintln!(
        "Wrote {written}/{} scenario JSON files to {}",
        entries.len(),
        out_dir.display()
    );

    if let Ok(client) = PetalTonguePushClient::discover() {
        eprintln!("\npetalTongue discovered — pushing scenarios via IPC...");
        let mut pushed = 0u32;
        for (name, scenario, _) in &entries {
            if let Err(e) = client.push_render(name, name, scenario) {
                eprintln!("  push {name}: {e}");
            } else {
                pushed += 1;
            }
        }
        eprintln!("Pushed {pushed}/{} scenarios.", entries.len());
    } else {
        eprintln!("\npetalTongue not running — scenarios saved as JSON.");
        eprintln!("To view: petaltongue --scenario {}/", out_dir.display());
    }

    eprintln!("\nDone. {} domains visualized.", entries.len());
    eprintln!("Tip: run scripts/visualize.sh for the full pipeline.");
}
