// SPDX-License-Identifier: AGPL-3.0-or-later
//! Migrated experiment modules from prokaryotic binaries.
//!
//! 318 validation + 23 benchmark experiments.

pub mod bench_23_domain_timing;
#[cfg(feature = "gpu")]
pub mod bench_all_domains_cpu_gpu;
#[cfg(feature = "gpu")]
pub mod bench_cpu_gpu;
pub mod bench_cross_spring_evolution;
pub mod bench_cross_spring_evolution_s70;
#[cfg(feature = "gpu")]
pub mod bench_cross_spring_evolution_v98;
pub mod bench_cross_spring_modern;
#[cfg(feature = "gpu")]
pub mod bench_cross_spring_modern_s68plus;
#[cfg(feature = "gpu")]
pub mod bench_cross_spring_s65;
#[cfg(feature = "gpu")]
pub mod bench_cross_spring_s68;
#[cfg(feature = "gpu")]
pub mod bench_cross_spring_scaling;
#[cfg(feature = "gpu")]
pub mod bench_dispatch_overhead;
#[cfg(feature = "gpu")]
pub mod bench_modern_systems_df64;
#[cfg(feature = "gpu")]
pub mod bench_ode_lean_crossspring;
#[cfg(feature = "gpu")]
pub mod bench_phylo_hmm_gpu;
pub mod bench_pipeline;
#[cfg(feature = "gpu")]
pub mod bench_progression_cpu_gpu_stream;
pub mod bench_python_vs_rust_v2;
pub mod bench_python_vs_rust_v3;
pub mod bench_python_vs_rust_v4;
pub mod bench_python_vs_rust_v5;
#[cfg(feature = "gpu")]
pub mod bench_streaming_vs_roundtrip;
#[cfg(feature = "gpu")]
pub mod bench_three_tier;
pub mod exp_16s_pipeline;
#[cfg(feature = "gpu")]
pub mod exp_16s_pipeline_gpu;
pub mod exp_adaptive_dispatch_v1;
pub mod exp_algae_16s;
pub mod exp_algae_timeseries;
pub mod exp_alignment;
pub mod exp_anaerobic_afex_stover;
pub mod exp_anaerobic_codigestion;
pub mod exp_anaerobic_coffee_residues;
pub mod exp_anaerobic_culture_response;
#[cfg(feature = "gpu")]
pub mod exp_anderson_2d_qs;
#[cfg(feature = "gpu")]
pub mod exp_anderson_3d_qs;
pub mod exp_anderson_anomalies;
#[cfg(feature = "gpu")]
pub mod exp_anderson_gpu_scaling;
pub mod exp_anderson_qs_environments_v1;
pub mod exp_barracuda_cpu;
pub mod exp_barracuda_cpu_full;
pub mod exp_barracuda_cpu_v10;
pub mod exp_barracuda_cpu_v11;
pub mod exp_barracuda_cpu_v12;
pub mod exp_barracuda_cpu_v13;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_cpu_v14;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_cpu_v15;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_cpu_v16;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_cpu_v17;
pub mod exp_barracuda_cpu_v18;
pub mod exp_barracuda_cpu_v19;
pub mod exp_barracuda_cpu_v2;
#[cfg(feature = "vault")]
pub mod exp_barracuda_cpu_v20;
pub mod exp_barracuda_cpu_v21;
pub mod exp_barracuda_cpu_v22;
pub mod exp_barracuda_cpu_v23;
pub mod exp_barracuda_cpu_v24;
pub mod exp_barracuda_cpu_v25;
pub mod exp_barracuda_cpu_v26;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_cpu_v27;
pub mod exp_barracuda_cpu_v3;
pub mod exp_barracuda_cpu_v4;
pub mod exp_barracuda_cpu_v5;
pub mod exp_barracuda_cpu_v6;
pub mod exp_barracuda_cpu_v7;
pub mod exp_barracuda_cpu_v8;
pub mod exp_barracuda_cpu_v9;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_full;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v1;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v10;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v11;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v12;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v13;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v14;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v3;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v4;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v5;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v6;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v7;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v8;
#[cfg(feature = "gpu")]
pub mod exp_barracuda_gpu_v9;
#[cfg(feature = "gpu")]
pub mod exp_barrier_disruption_s79;
#[cfg(feature = "nautilus")]
pub mod exp_bio_brain_s79;
#[cfg(feature = "gpu")]
pub mod exp_biofilm_3d_qs;
pub mod exp_biomeos_nucleus_v98;
pub mod exp_bistable;
pub mod exp_bloom_surveillance;
pub mod exp_bootstrap;
pub mod exp_breseq_barrick_2009;
pub mod exp_burst_statistics_anderson;
pub mod exp_capacitor;
pub mod exp_cold_seep_pipeline;
pub mod exp_cold_seep_qs_catalog;
pub mod exp_cold_seep_qs_geometry;
pub mod exp_colonization_resistance;
pub mod exp_composition_nucleus_v1;
#[cfg(feature = "facade")]
pub mod exp_composition_parity_v1;
pub mod exp_cooperation;
#[cfg(feature = "gpu")]
pub mod exp_correlated_disorder;
#[cfg(feature = "gpu")]
pub mod exp_cpu_gpu_expanded;
#[cfg(feature = "gpu")]
pub mod exp_cpu_gpu_full_domain_v92g;
#[cfg(feature = "gpu")]
pub mod exp_cpu_gpu_viz_math;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_all_domains;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_pure_math;
pub mod exp_cpu_vs_gpu_v10;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v11;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v5_io_evolution;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v6_extended;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v7;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v8;
#[cfg(feature = "gpu")]
pub mod exp_cpu_vs_gpu_v9;
#[cfg(feature = "gpu")]
pub mod exp_cross_ecosystem_atlas;
pub mod exp_cross_ecosystem_pangenome;
pub mod exp_cross_primal_pipeline_v98;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_evolution;
pub mod exp_cross_spring_evolution_modern;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_evolution_s87;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_evolution_v71;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_evolution_v98;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_modern_s86;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_provenance;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_s57;
#[cfg(feature = "gpu")]
pub mod exp_cross_spring_s62;
pub mod exp_cross_spring_s79;
pub mod exp_cross_spring_s86;
pub mod exp_cross_spring_s93;
#[cfg(feature = "gpu")]
pub mod exp_cross_substrate;
#[cfg(feature = "gpu")]
pub mod exp_cross_substrate_pipeline;
pub mod exp_df64_anderson;
pub mod exp_dictyostelium_relay;
#[cfg(feature = "gpu")]
pub mod exp_dimensional_phase_diagram;
#[cfg(feature = "gpu")]
pub mod exp_dispatch_overhead_proof;
pub mod exp_diversity;
#[cfg(feature = "gpu")]
pub mod exp_diversity_gpu;
pub mod exp_dynamic_anderson;
#[cfg(feature = "gpu")]
pub mod exp_ecosystem_geometry_qs;
pub mod exp_emp_anderson_atlas;
pub mod exp_emp_anderson_v1;
pub mod exp_epa_pfas_ml;
#[cfg(feature = "gpu")]
pub mod exp_eukaryote_scaling;
pub mod exp_extended_algae;
pub mod exp_fajgenbaum_pathway;
pub mod exp_fastq;
pub mod exp_features;
pub mod exp_felsenstein;
#[cfg(feature = "gpu")]
pub mod exp_finite_size_scaling;
#[cfg(feature = "gpu")]
pub mod exp_finite_size_scaling_v2;
pub mod exp_fungal_fermentation_digestate;
#[cfg(feature = "vault")]
pub mod exp_genomic_vault;
#[cfg(feature = "gpu")]
pub mod exp_geometry_zoo;
pub mod exp_gillespie;
pub mod exp_gonzales_cpu_parity;
#[cfg(feature = "gpu")]
pub mod exp_gonzales_gpu;
pub mod exp_gonzales_ic50_s79;
pub mod exp_gonzales_il31_s79;
#[cfg(feature = "gpu")]
pub mod exp_gonzales_metalforge;
pub mod exp_gonzales_pk_s79;
pub mod exp_gonzales_provenance_chain;
#[cfg(feature = "gpu")]
pub mod exp_gonzales_streaming;
#[cfg(feature = "gpu")]
pub mod exp_gpu_diversity_fusion;
#[cfg(feature = "gpu")]
pub mod exp_gpu_drug_repurposing;
#[cfg(feature = "gpu")]
pub mod exp_gpu_extended;
#[cfg(feature = "gpu")]
pub mod exp_gpu_hmm_forward;
#[cfg(feature = "gpu")]
pub mod exp_gpu_ode_sweep;
#[cfg(feature = "gpu")]
pub mod exp_gpu_phylo_compose;
#[cfg(feature = "gpu")]
pub mod exp_gpu_rf;
#[cfg(feature = "gpu")]
pub mod exp_gpu_streaming_pipeline;
#[cfg(feature = "gpu")]
pub mod exp_gpu_track1c;
#[cfg(feature = "gpu")]
pub mod exp_gpu_v59_science;
#[cfg(feature = "gpu")]
pub mod exp_hardware_learning_v1;
#[cfg(feature = "gpu")]
pub mod exp_heterogeneity_sweep_s79;
pub mod exp_hmm;
pub mod exp_hormesis_biphasic;
#[cfg(feature = "gpu")]
pub mod exp_immuno_anderson_cpu_parity;
#[cfg(feature = "gpu")]
pub mod exp_immuno_anderson_gpu;
#[cfg(feature = "gpu")]
pub mod exp_immuno_anderson_metalforge;
#[cfg(feature = "gpu")]
pub mod exp_immuno_anderson_streaming;
pub mod exp_kbs_lter_anderson_v1;
pub mod exp_knowledge_graph_embedding;
#[cfg(feature = "gpu")]
pub mod exp_kriging;
pub mod exp_lan_mesh_plan_v1;
pub mod exp_liao_real_data_v1;
#[cfg(feature = "gpu")]
pub mod exp_local_wgsl_compile;
pub mod exp_ltee_b7_v1;
pub mod exp_luxr_phylogeny_geometry;
#[cfg(feature = "gpu")]
pub mod exp_mapping_sensitivity;
pub mod exp_marine_interkingdom_qs;
#[cfg(feature = "gpu")]
pub mod exp_massbank_gpu_scale;
pub mod exp_massbank_spectral;
#[cfg(feature = "gpu")]
pub mod exp_matrix_pharmacophenomics;
pub mod exp_mechanical_wave_anderson;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_drug_repurposing;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_full;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_full_v2;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_full_v3;
pub mod exp_metalforge_pipeline;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v10_evolution;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v11_extended;
#[cfg(all(feature = "gpu", feature = "vault"))]
pub mod exp_metalforge_v12_extended;
pub mod exp_metalforge_v15;
pub mod exp_metalforge_v16;
pub mod exp_metalforge_v17;
pub mod exp_metalforge_v18;
pub mod exp_metalforge_v19;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v4;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v5;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v59_science;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v6;
pub mod exp_metalforge_v7_mixed;
pub mod exp_metalforge_v8_cross_system;
#[cfg(feature = "gpu")]
pub mod exp_metalforge_v9_nucleus;
pub mod exp_multi_signal;
pub mod exp_myxococcus_critical_density;
pub mod exp_mzml;
pub mod exp_nanopore_int8_quantization;
pub mod exp_nanopore_signal_bridge;
pub mod exp_nanopore_simulated_16s;
pub mod exp_ncbi_pangenome;
pub mod exp_ncbi_qs_atlas;
pub mod exp_ncbi_qs_habitat;
pub mod exp_ncbi_vibrio_qs;
pub mod exp_neighbor_joining;
pub mod exp_newick_parse;
pub mod exp_niche_parity_v1;
pub mod exp_nitrifying_qs;
#[cfg(feature = "gpu")]
pub mod exp_nmf_drug_repurposing;
pub mod exp_notill_brandt_farm;
pub mod exp_notill_longterm_tillage;
pub mod exp_notill_meta_analysis;
#[cfg(feature = "gpu")]
pub mod exp_nouveau_diagnostic_v1;
pub mod exp_npu_bloom_sentinel;
pub mod exp_npu_disorder_classifier;
#[cfg(feature = "npu")]
pub mod exp_npu_funky;
pub mod exp_npu_genome_binning;
#[cfg(feature = "npu")]
pub mod exp_npu_hardware;
#[cfg(feature = "npu")]
pub mod exp_npu_live;
pub mod exp_npu_phylo_placement;
pub mod exp_npu_qs_classifier;
pub mod exp_npu_sentinel_stream;
pub mod exp_npu_spectral_screen;
pub mod exp_npu_spectral_triage;
pub mod exp_nucleus_data_pipeline;
pub mod exp_nucleus_live_gonzales;
pub mod exp_nucleus_tower_node;
pub mod exp_nucleus_v4;
pub mod exp_nucleus_v8_mixed;
pub mod exp_p1_extensions_v1;
pub mod exp_pangenomics;
pub mod exp_paper_math_control_v1;
pub mod exp_paper_math_control_v2;
pub mod exp_paper_math_control_v3;
pub mod exp_paper_math_control_v4;
pub mod exp_paper_math_control_v5;
pub mod exp_paper_math_control_v6;
pub mod exp_pcie_direct;
pub mod exp_peaks;
pub mod exp_petaltongue_anderson_v1;
pub mod exp_petaltongue_biogas_v1;
pub mod exp_petaltongue_live_v1;
pub mod exp_pfas;
pub mod exp_pfas_decision_tree;
pub mod exp_pfas_library;
pub mod exp_phage_defense;
pub mod exp_phosphorus_phylogenomics;
pub mod exp_phylo_placement_scale;
pub mod exp_phylohmm;
pub mod exp_phynetpy_rf;
pub mod exp_physical_comm_anderson;
pub mod exp_placement;
#[cfg(feature = "gpu")]
pub mod exp_planktonic_dilution;
pub mod exp_population_genomics;
#[cfg(feature = "gpu")]
pub mod exp_precision_brain_v1;
pub mod exp_primal_parity_v1;
#[cfg(feature = "gpu")]
pub mod exp_primal_pipeline_v1;
pub mod exp_producer_receiver_qs;
pub mod exp_public_benchmarks;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_complete;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_pipeline;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming;
pub mod exp_pure_gpu_streaming_v10;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v11;
pub mod exp_pure_gpu_streaming_v12;
pub mod exp_pure_gpu_streaming_v13;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v2;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v3;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v4;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v6;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v7;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v8;
#[cfg(feature = "gpu")]
pub mod exp_pure_gpu_streaming_v9;
#[cfg(feature = "gpu")]
pub mod exp_qs_disorder_real;
#[cfg(feature = "gpu")]
pub mod exp_qs_distance_scaling;
pub mod exp_qs_gene_prevalence;
pub mod exp_qs_gene_profiling_v1;
pub mod exp_qs_ode;
pub mod exp_qs_wave_localization;
pub mod exp_r_industry_parity;
pub mod exp_rare_biosphere;
#[cfg(feature = "gpu")]
pub mod exp_real_bloom_gpu;
pub mod exp_real_ncbi_pipeline;
pub mod exp_reconciliation;
#[cfg(feature = "gpu")]
pub mod exp_repodb_nmf;
pub mod exp_rf_distance;
#[cfg(feature = "gpu")]
pub mod exp_s86_streaming_pipeline;
pub mod exp_sate_pipeline;
pub mod exp_science_pipeline;
#[cfg(feature = "gpu")]
pub mod exp_skin_anderson_s79;
pub mod exp_soil_biofilm_aggregate;
pub mod exp_soil_distance_colonization;
pub mod exp_soil_pore_diversity;
pub mod exp_soil_qs_cpu_parity;
#[cfg(feature = "gpu")]
pub mod exp_soil_qs_gpu;
#[cfg(feature = "gpu")]
pub mod exp_soil_qs_metalforge;
pub mod exp_soil_qs_pore_geometry;
#[cfg(feature = "gpu")]
pub mod exp_soil_qs_streaming;
pub mod exp_soil_structure_function;
#[cfg(feature = "gpu")]
pub mod exp_sovereign_dispatch_v1;
pub mod exp_sovereign_resequencing;
#[cfg(feature = "gpu")]
pub mod exp_spectral_cross_spring;
#[cfg(feature = "gpu")]
pub mod exp_square_cubed_scaling;
pub mod exp_stable_specials_v1;
pub mod exp_streaming_dispatch;
pub mod exp_streaming_io_parity;
#[cfg(feature = "gpu")]
pub mod exp_streaming_ode_phylo;
pub mod exp_streaming_pipeline_v5;
#[cfg(feature = "gpu")]
pub mod exp_substrate_router;
pub mod exp_sulfur_phylogenomics;
pub mod exp_temporal_esn_bloom;
pub mod exp_tillage_microbiome_2025;
#[cfg(feature = "gpu")]
pub mod exp_toadstool_bio;
#[cfg(feature = "gpu")]
pub mod exp_toadstool_dispatch_v2;
pub mod exp_toadstool_dispatch_v3;
pub mod exp_toadstool_dispatch_v4;
pub mod exp_toadstool_s70_rewire;
pub mod exp_trophic_cascade;
#[cfg(feature = "gpu")]
pub mod exp_vent_chimney_qs;
#[cfg(feature = "gpu")]
pub mod exp_vibrio_qs_landscape;
pub mod exp_viral_metagenomics;
pub mod exp_visualization_v1;
pub mod exp_visualization_v2;
pub mod exp_voc_peaks;
#[cfg(feature = "gpu")]
pub mod exp_workload_routing_v1;

/// Register all migrated experiment scenarios into the registry.
pub fn register_all(r: &mut super::scenarios::ScenarioRegistry) {
    r.register(bench_23_domain_timing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_all_domains_cpu_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cpu_gpu::SCENARIO);
    r.register(bench_cross_spring_evolution::SCENARIO);
    r.register(bench_cross_spring_evolution_s70::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cross_spring_evolution_v98::SCENARIO);
    r.register(bench_cross_spring_modern::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cross_spring_modern_s68plus::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cross_spring_s65::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cross_spring_s68::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_cross_spring_scaling::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_dispatch_overhead::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_modern_systems_df64::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_ode_lean_crossspring::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_phylo_hmm_gpu::SCENARIO);
    r.register(bench_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_progression_cpu_gpu_stream::SCENARIO);
    r.register(bench_python_vs_rust_v2::SCENARIO);
    r.register(bench_python_vs_rust_v3::SCENARIO);
    r.register(bench_python_vs_rust_v4::SCENARIO);
    r.register(bench_python_vs_rust_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_streaming_vs_roundtrip::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(bench_three_tier::SCENARIO);
    r.register(exp_16s_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_16s_pipeline_gpu::SCENARIO);
    r.register(exp_adaptive_dispatch_v1::SCENARIO);
    r.register(exp_algae_16s::SCENARIO);
    r.register(exp_algae_timeseries::SCENARIO);
    r.register(exp_alignment::SCENARIO);
    r.register(exp_anaerobic_afex_stover::SCENARIO);
    r.register(exp_anaerobic_codigestion::SCENARIO);
    r.register(exp_anaerobic_coffee_residues::SCENARIO);
    r.register(exp_anaerobic_culture_response::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_anderson_2d_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_anderson_3d_qs::SCENARIO);
    r.register(exp_anderson_anomalies::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_anderson_gpu_scaling::SCENARIO);
    r.register(exp_anderson_qs_environments_v1::SCENARIO);
    r.register(exp_barracuda_cpu::SCENARIO);
    r.register(exp_barracuda_cpu_full::SCENARIO);
    r.register(exp_barracuda_cpu_v10::SCENARIO);
    r.register(exp_barracuda_cpu_v11::SCENARIO);
    r.register(exp_barracuda_cpu_v12::SCENARIO);
    r.register(exp_barracuda_cpu_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_cpu_v14::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_cpu_v15::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_cpu_v16::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_cpu_v17::SCENARIO);
    r.register(exp_barracuda_cpu_v18::SCENARIO);
    r.register(exp_barracuda_cpu_v19::SCENARIO);
    r.register(exp_barracuda_cpu_v2::SCENARIO);
    #[cfg(feature = "vault")]
    r.register(exp_barracuda_cpu_v20::SCENARIO);
    r.register(exp_barracuda_cpu_v21::SCENARIO);
    r.register(exp_barracuda_cpu_v22::SCENARIO);
    r.register(exp_barracuda_cpu_v23::SCENARIO);
    r.register(exp_barracuda_cpu_v24::SCENARIO);
    r.register(exp_barracuda_cpu_v25::SCENARIO);
    r.register(exp_barracuda_cpu_v26::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_cpu_v27::SCENARIO);
    r.register(exp_barracuda_cpu_v3::SCENARIO);
    r.register(exp_barracuda_cpu_v4::SCENARIO);
    r.register(exp_barracuda_cpu_v5::SCENARIO);
    r.register(exp_barracuda_cpu_v6::SCENARIO);
    r.register(exp_barracuda_cpu_v7::SCENARIO);
    r.register(exp_barracuda_cpu_v8::SCENARIO);
    r.register(exp_barracuda_cpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_full::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v11::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v12::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v14::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v3::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v6::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barracuda_gpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_barrier_disruption_s79::SCENARIO);
    #[cfg(feature = "nautilus")]
    r.register(exp_bio_brain_s79::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_biofilm_3d_qs::SCENARIO);
    r.register(exp_biomeos_nucleus_v98::SCENARIO);
    r.register(exp_bistable::SCENARIO);
    r.register(exp_bloom_surveillance::SCENARIO);
    r.register(exp_bootstrap::SCENARIO);
    r.register(exp_breseq_barrick_2009::SCENARIO);
    r.register(exp_burst_statistics_anderson::SCENARIO);
    r.register(exp_capacitor::SCENARIO);
    r.register(exp_cold_seep_pipeline::SCENARIO);
    r.register(exp_cold_seep_qs_catalog::SCENARIO);
    r.register(exp_cold_seep_qs_geometry::SCENARIO);
    r.register(exp_colonization_resistance::SCENARIO);
    r.register(exp_composition_nucleus_v1::SCENARIO);
    #[cfg(feature = "facade")]
    r.register(exp_composition_parity_v1::SCENARIO);
    r.register(exp_cooperation::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_correlated_disorder::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_gpu_expanded::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_gpu_full_domain_v92g::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_gpu_viz_math::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_all_domains::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_pure_math::SCENARIO);
    r.register(exp_cpu_vs_gpu_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v11::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v5_io_evolution::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v6_extended::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cpu_vs_gpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_ecosystem_atlas::SCENARIO);
    r.register(exp_cross_ecosystem_pangenome::SCENARIO);
    r.register(exp_cross_primal_pipeline_v98::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_evolution::SCENARIO);
    r.register(exp_cross_spring_evolution_modern::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_evolution_s87::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_evolution_v71::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_evolution_v98::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_modern_s86::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_provenance::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_s57::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_spring_s62::SCENARIO);
    r.register(exp_cross_spring_s79::SCENARIO);
    r.register(exp_cross_spring_s86::SCENARIO);
    r.register(exp_cross_spring_s93::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_substrate::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_cross_substrate_pipeline::SCENARIO);
    r.register(exp_df64_anderson::SCENARIO);
    r.register(exp_dictyostelium_relay::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_dimensional_phase_diagram::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_dispatch_overhead_proof::SCENARIO);
    r.register(exp_diversity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_diversity_gpu::SCENARIO);
    r.register(exp_dynamic_anderson::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_ecosystem_geometry_qs::SCENARIO);
    r.register(exp_emp_anderson_atlas::SCENARIO);
    r.register(exp_emp_anderson_v1::SCENARIO);
    r.register(exp_epa_pfas_ml::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_eukaryote_scaling::SCENARIO);
    r.register(exp_extended_algae::SCENARIO);
    r.register(exp_fajgenbaum_pathway::SCENARIO);
    r.register(exp_fastq::SCENARIO);
    r.register(exp_features::SCENARIO);
    r.register(exp_felsenstein::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_finite_size_scaling::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_finite_size_scaling_v2::SCENARIO);
    r.register(exp_fungal_fermentation_digestate::SCENARIO);
    #[cfg(feature = "vault")]
    r.register(exp_genomic_vault::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_geometry_zoo::SCENARIO);
    r.register(exp_gillespie::SCENARIO);
    r.register(exp_gonzales_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gonzales_gpu::SCENARIO);
    r.register(exp_gonzales_ic50_s79::SCENARIO);
    r.register(exp_gonzales_il31_s79::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gonzales_metalforge::SCENARIO);
    r.register(exp_gonzales_pk_s79::SCENARIO);
    r.register(exp_gonzales_provenance_chain::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gonzales_streaming::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_diversity_fusion::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_drug_repurposing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_extended::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_hmm_forward::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_ode_sweep::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_phylo_compose::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_rf::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_streaming_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_track1c::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_gpu_v59_science::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_hardware_learning_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_heterogeneity_sweep_s79::SCENARIO);
    r.register(exp_hmm::SCENARIO);
    r.register(exp_hormesis_biphasic::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_immuno_anderson_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_immuno_anderson_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_immuno_anderson_metalforge::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_immuno_anderson_streaming::SCENARIO);
    r.register(exp_kbs_lter_anderson_v1::SCENARIO);
    r.register(exp_knowledge_graph_embedding::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_kriging::SCENARIO);
    r.register(exp_lan_mesh_plan_v1::SCENARIO);
    r.register(exp_liao_real_data_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_local_wgsl_compile::SCENARIO);
    r.register(exp_ltee_b7_v1::SCENARIO);
    r.register(exp_luxr_phylogeny_geometry::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_mapping_sensitivity::SCENARIO);
    r.register(exp_marine_interkingdom_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_massbank_gpu_scale::SCENARIO);
    r.register(exp_massbank_spectral::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_matrix_pharmacophenomics::SCENARIO);
    r.register(exp_mechanical_wave_anderson::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_drug_repurposing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_full::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_full_v2::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_full_v3::SCENARIO);
    r.register(exp_metalforge_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v10_evolution::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v11_extended::SCENARIO);
    #[cfg(all(feature = "gpu", feature = "vault"))]
    r.register(exp_metalforge_v12_extended::SCENARIO);
    r.register(exp_metalforge_v15::SCENARIO);
    r.register(exp_metalforge_v16::SCENARIO);
    r.register(exp_metalforge_v17::SCENARIO);
    r.register(exp_metalforge_v18::SCENARIO);
    r.register(exp_metalforge_v19::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v59_science::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v6::SCENARIO);
    r.register(exp_metalforge_v7_mixed::SCENARIO);
    r.register(exp_metalforge_v8_cross_system::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_metalforge_v9_nucleus::SCENARIO);
    r.register(exp_multi_signal::SCENARIO);
    r.register(exp_myxococcus_critical_density::SCENARIO);
    r.register(exp_mzml::SCENARIO);
    r.register(exp_nanopore_int8_quantization::SCENARIO);
    r.register(exp_nanopore_signal_bridge::SCENARIO);
    r.register(exp_nanopore_simulated_16s::SCENARIO);
    r.register(exp_ncbi_pangenome::SCENARIO);
    r.register(exp_ncbi_qs_atlas::SCENARIO);
    r.register(exp_ncbi_qs_habitat::SCENARIO);
    r.register(exp_ncbi_vibrio_qs::SCENARIO);
    r.register(exp_neighbor_joining::SCENARIO);
    r.register(exp_newick_parse::SCENARIO);
    r.register(exp_niche_parity_v1::SCENARIO);
    r.register(exp_nitrifying_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_nmf_drug_repurposing::SCENARIO);
    r.register(exp_notill_brandt_farm::SCENARIO);
    r.register(exp_notill_longterm_tillage::SCENARIO);
    r.register(exp_notill_meta_analysis::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_nouveau_diagnostic_v1::SCENARIO);
    r.register(exp_npu_bloom_sentinel::SCENARIO);
    r.register(exp_npu_disorder_classifier::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(exp_npu_funky::SCENARIO);
    r.register(exp_npu_genome_binning::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(exp_npu_hardware::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(exp_npu_live::SCENARIO);
    r.register(exp_npu_phylo_placement::SCENARIO);
    r.register(exp_npu_qs_classifier::SCENARIO);
    r.register(exp_npu_sentinel_stream::SCENARIO);
    r.register(exp_npu_spectral_screen::SCENARIO);
    r.register(exp_npu_spectral_triage::SCENARIO);
    r.register(exp_nucleus_data_pipeline::SCENARIO);
    r.register(exp_nucleus_live_gonzales::SCENARIO);
    r.register(exp_nucleus_tower_node::SCENARIO);
    r.register(exp_nucleus_v4::SCENARIO);
    r.register(exp_nucleus_v8_mixed::SCENARIO);
    r.register(exp_p1_extensions_v1::SCENARIO);
    r.register(exp_pangenomics::SCENARIO);
    r.register(exp_paper_math_control_v1::SCENARIO);
    r.register(exp_paper_math_control_v2::SCENARIO);
    r.register(exp_paper_math_control_v3::SCENARIO);
    r.register(exp_paper_math_control_v4::SCENARIO);
    r.register(exp_paper_math_control_v5::SCENARIO);
    r.register(exp_paper_math_control_v6::SCENARIO);
    r.register(exp_pcie_direct::SCENARIO);
    r.register(exp_peaks::SCENARIO);
    r.register(exp_petaltongue_anderson_v1::SCENARIO);
    r.register(exp_petaltongue_biogas_v1::SCENARIO);
    r.register(exp_petaltongue_live_v1::SCENARIO);
    r.register(exp_pfas::SCENARIO);
    r.register(exp_pfas_decision_tree::SCENARIO);
    r.register(exp_pfas_library::SCENARIO);
    r.register(exp_phage_defense::SCENARIO);
    r.register(exp_phosphorus_phylogenomics::SCENARIO);
    r.register(exp_phylo_placement_scale::SCENARIO);
    r.register(exp_phylohmm::SCENARIO);
    r.register(exp_phynetpy_rf::SCENARIO);
    r.register(exp_physical_comm_anderson::SCENARIO);
    r.register(exp_placement::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_planktonic_dilution::SCENARIO);
    r.register(exp_population_genomics::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_precision_brain_v1::SCENARIO);
    r.register(exp_primal_parity_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_primal_pipeline_v1::SCENARIO);
    r.register(exp_producer_receiver_qs::SCENARIO);
    r.register(exp_public_benchmarks::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_complete::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming::SCENARIO);
    r.register(exp_pure_gpu_streaming_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v11::SCENARIO);
    r.register(exp_pure_gpu_streaming_v12::SCENARIO);
    r.register(exp_pure_gpu_streaming_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v2::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v3::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v6::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_pure_gpu_streaming_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_qs_disorder_real::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_qs_distance_scaling::SCENARIO);
    r.register(exp_qs_gene_prevalence::SCENARIO);
    r.register(exp_qs_gene_profiling_v1::SCENARIO);
    r.register(exp_qs_ode::SCENARIO);
    r.register(exp_qs_wave_localization::SCENARIO);
    r.register(exp_r_industry_parity::SCENARIO);
    r.register(exp_rare_biosphere::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_real_bloom_gpu::SCENARIO);
    r.register(exp_real_ncbi_pipeline::SCENARIO);
    r.register(exp_reconciliation::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_repodb_nmf::SCENARIO);
    r.register(exp_rf_distance::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_s86_streaming_pipeline::SCENARIO);
    r.register(exp_sate_pipeline::SCENARIO);
    r.register(exp_science_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_skin_anderson_s79::SCENARIO);
    r.register(exp_soil_biofilm_aggregate::SCENARIO);
    r.register(exp_soil_distance_colonization::SCENARIO);
    r.register(exp_soil_pore_diversity::SCENARIO);
    r.register(exp_soil_qs_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_soil_qs_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_soil_qs_metalforge::SCENARIO);
    r.register(exp_soil_qs_pore_geometry::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_soil_qs_streaming::SCENARIO);
    r.register(exp_soil_structure_function::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_sovereign_dispatch_v1::SCENARIO);
    r.register(exp_sovereign_resequencing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_spectral_cross_spring::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_square_cubed_scaling::SCENARIO);
    r.register(exp_stable_specials_v1::SCENARIO);
    r.register(exp_streaming_dispatch::SCENARIO);
    r.register(exp_streaming_io_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_streaming_ode_phylo::SCENARIO);
    r.register(exp_streaming_pipeline_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_substrate_router::SCENARIO);
    r.register(exp_sulfur_phylogenomics::SCENARIO);
    r.register(exp_temporal_esn_bloom::SCENARIO);
    r.register(exp_tillage_microbiome_2025::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_toadstool_bio::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_toadstool_dispatch_v2::SCENARIO);
    r.register(exp_toadstool_dispatch_v3::SCENARIO);
    r.register(exp_toadstool_dispatch_v4::SCENARIO);
    r.register(exp_toadstool_s70_rewire::SCENARIO);
    r.register(exp_trophic_cascade::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_vent_chimney_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_vibrio_qs_landscape::SCENARIO);
    r.register(exp_viral_metagenomics::SCENARIO);
    r.register(exp_visualization_v1::SCENARIO);
    r.register(exp_visualization_v2::SCENARIO);
    r.register(exp_voc_peaks::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(exp_workload_routing_v1::SCENARIO);
}
