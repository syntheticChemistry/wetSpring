// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario registration table extracted from mod.rs.
//!
//! Each line registers one experiment SCENARIO constant into the
//! ScenarioRegistry. Feature-gated experiments are conditionally compiled.

pub(super) fn register_all(r: &mut crate::validation::scenarios::ScenarioRegistry) {
    r.register(super::bench_23_domain_timing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_all_domains_cpu_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cpu_gpu::SCENARIO);
    r.register(super::bench_cross_spring_evolution::SCENARIO);
    r.register(super::bench_cross_spring_evolution_s70::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cross_spring_evolution_v98::SCENARIO);
    r.register(super::bench_cross_spring_modern::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cross_spring_modern_s68plus::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cross_spring_s65::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cross_spring_s68::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_cross_spring_scaling::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_dispatch_overhead::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_modern_systems_df64::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_ode_lean_crossspring::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_phylo_hmm_gpu::SCENARIO);
    r.register(super::bench_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_progression_cpu_gpu_stream::SCENARIO);
    r.register(super::bench_python_vs_rust_v2::SCENARIO);
    r.register(super::bench_python_vs_rust_v3::SCENARIO);
    r.register(super::bench_python_vs_rust_v4::SCENARIO);
    r.register(super::bench_python_vs_rust_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_streaming_vs_roundtrip::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::bench_three_tier::SCENARIO);
    r.register(super::exp_16s_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_16s_pipeline_gpu::SCENARIO);
    r.register(super::exp_adaptive_dispatch_v1::SCENARIO);
    r.register(super::exp_algae_16s::SCENARIO);
    r.register(super::exp_algae_timeseries::SCENARIO);
    r.register(super::exp_alignment::SCENARIO);
    r.register(super::exp_anaerobic_afex_stover::SCENARIO);
    r.register(super::exp_anaerobic_codigestion::SCENARIO);
    r.register(super::exp_anaerobic_coffee_residues::SCENARIO);
    r.register(super::exp_anaerobic_culture_response::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_anderson_2d_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_anderson_3d_qs::SCENARIO);
    r.register(super::exp_anderson_anomalies::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_anderson_gpu_scaling::SCENARIO);
    r.register(super::exp_anderson_qs_environments_v1::SCENARIO);
    r.register(super::exp_barracuda_cpu::SCENARIO);
    r.register(super::exp_barracuda_cpu_full::SCENARIO);
    r.register(super::exp_barracuda_cpu_v10::SCENARIO);
    r.register(super::exp_barracuda_cpu_v11::SCENARIO);
    r.register(super::exp_barracuda_cpu_v12::SCENARIO);
    r.register(super::exp_barracuda_cpu_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_cpu_v14::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_cpu_v15::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_cpu_v16::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_cpu_v17::SCENARIO);
    r.register(super::exp_barracuda_cpu_v18::SCENARIO);
    r.register(super::exp_barracuda_cpu_v19::SCENARIO);
    r.register(super::exp_barracuda_cpu_v2::SCENARIO);
    #[cfg(feature = "vault")]
    r.register(super::exp_barracuda_cpu_v20::SCENARIO);
    r.register(super::exp_barracuda_cpu_v21::SCENARIO);
    r.register(super::exp_barracuda_cpu_v22::SCENARIO);
    r.register(super::exp_barracuda_cpu_v23::SCENARIO);
    r.register(super::exp_barracuda_cpu_v24::SCENARIO);
    r.register(super::exp_barracuda_cpu_v25::SCENARIO);
    r.register(super::exp_barracuda_cpu_v26::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_cpu_v27::SCENARIO);
    r.register(super::exp_barracuda_cpu_v3::SCENARIO);
    r.register(super::exp_barracuda_cpu_v4::SCENARIO);
    r.register(super::exp_barracuda_cpu_v5::SCENARIO);
    r.register(super::exp_barracuda_cpu_v6::SCENARIO);
    r.register(super::exp_barracuda_cpu_v7::SCENARIO);
    r.register(super::exp_barracuda_cpu_v8::SCENARIO);
    r.register(super::exp_barracuda_cpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_full::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v11::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v12::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v14::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v3::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v6::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barracuda_gpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_barrier_disruption_s79::SCENARIO);
    #[cfg(feature = "nautilus")]
    r.register(super::exp_bio_brain_s79::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_biofilm_3d_qs::SCENARIO);
    r.register(super::exp_biomeos_nucleus_v98::SCENARIO);
    r.register(super::exp_bistable::SCENARIO);
    r.register(super::exp_bloom_surveillance::SCENARIO);
    r.register(super::exp_bootstrap::SCENARIO);
    r.register(super::exp_breseq_barrick_2009::SCENARIO);
    r.register(super::exp_burst_statistics_anderson::SCENARIO);
    r.register(super::exp_capacitor::SCENARIO);
    r.register(super::exp_cold_seep_pipeline::SCENARIO);
    r.register(super::exp_cold_seep_qs_catalog::SCENARIO);
    r.register(super::exp_cold_seep_qs_geometry::SCENARIO);
    r.register(super::exp_colonization_resistance::SCENARIO);
    r.register(super::exp_composition_nucleus_v1::SCENARIO);
    #[cfg(feature = "facade")]
    r.register(super::exp_composition_parity_v1::SCENARIO);
    r.register(super::exp_cooperation::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_correlated_disorder::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_gpu_expanded::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_gpu_full_domain_v92g::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_gpu_viz_math::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_all_domains::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_pure_math::SCENARIO);
    r.register(super::exp_cpu_vs_gpu_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v11::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v5_io_evolution::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v6_extended::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cpu_vs_gpu_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_ecosystem_atlas::SCENARIO);
    r.register(super::exp_cross_ecosystem_pangenome::SCENARIO);
    r.register(super::exp_cross_primal_pipeline_v98::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_evolution::SCENARIO);
    r.register(super::exp_cross_spring_evolution_modern::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_evolution_s87::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_evolution_v71::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_evolution_v98::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_modern_s86::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_provenance::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_s57::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_spring_s62::SCENARIO);
    r.register(super::exp_cross_spring_s79::SCENARIO);
    r.register(super::exp_cross_spring_s86::SCENARIO);
    r.register(super::exp_cross_spring_s93::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_substrate::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_cross_substrate_pipeline::SCENARIO);
    r.register(super::exp_df64_anderson::SCENARIO);
    r.register(super::exp_dictyostelium_relay::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_dimensional_phase_diagram::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_dispatch_overhead_proof::SCENARIO);
    r.register(super::exp_diversity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_diversity_gpu::SCENARIO);
    r.register(super::exp_dynamic_anderson::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_ecosystem_geometry_qs::SCENARIO);
    r.register(super::exp_emp_anderson_atlas::SCENARIO);
    r.register(super::exp_emp_anderson_v1::SCENARIO);
    r.register(super::exp_epa_pfas_ml::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_eukaryote_scaling::SCENARIO);
    r.register(super::exp_extended_algae::SCENARIO);
    r.register(super::exp_fajgenbaum_pathway::SCENARIO);
    r.register(super::exp_fastq::SCENARIO);
    r.register(super::exp_features::SCENARIO);
    r.register(super::exp_felsenstein::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_finite_size_scaling::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_finite_size_scaling_v2::SCENARIO);
    r.register(super::exp_fungal_fermentation_digestate::SCENARIO);
    #[cfg(feature = "vault")]
    r.register(super::exp_genomic_vault::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_geometry_zoo::SCENARIO);
    r.register(super::exp_gillespie::SCENARIO);
    r.register(super::exp_gonzales_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gonzales_gpu::SCENARIO);
    r.register(super::exp_gonzales_ic50_s79::SCENARIO);
    r.register(super::exp_gonzales_il31_s79::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gonzales_metalforge::SCENARIO);
    r.register(super::exp_gonzales_pk_s79::SCENARIO);
    r.register(super::exp_gonzales_provenance_chain::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gonzales_streaming::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_diversity_fusion::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_drug_repurposing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_extended::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_hmm_forward::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_ode_sweep::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_phylo_compose::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_rf::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_streaming_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_track1c::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_gpu_v59_science::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_hardware_learning_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_heterogeneity_sweep_s79::SCENARIO);
    r.register(super::exp_hmm::SCENARIO);
    r.register(super::exp_hormesis_biphasic::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_immuno_anderson_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_immuno_anderson_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_immuno_anderson_metalforge::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_immuno_anderson_streaming::SCENARIO);
    r.register(super::exp_kbs_lter_anderson_v1::SCENARIO);
    r.register(super::exp_knowledge_graph_embedding::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_kriging::SCENARIO);
    r.register(super::exp_lan_mesh_plan_v1::SCENARIO);
    r.register(super::exp_liao_real_data_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_local_wgsl_compile::SCENARIO);
    r.register(super::exp_ltee_b7_v1::SCENARIO);
    r.register(super::exp_luxr_phylogeny_geometry::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_mapping_sensitivity::SCENARIO);
    r.register(super::exp_marine_interkingdom_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_massbank_gpu_scale::SCENARIO);
    r.register(super::exp_massbank_spectral::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_matrix_pharmacophenomics::SCENARIO);
    r.register(super::exp_mechanical_wave_anderson::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_drug_repurposing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_full::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_full_v2::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_full_v3::SCENARIO);
    r.register(super::exp_metalforge_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v10_evolution::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v11_extended::SCENARIO);
    #[cfg(all(feature = "gpu", feature = "vault"))]
    r.register(super::exp_metalforge_v12_extended::SCENARIO);
    r.register(super::exp_metalforge_v15::SCENARIO);
    r.register(super::exp_metalforge_v16::SCENARIO);
    r.register(super::exp_metalforge_v17::SCENARIO);
    r.register(super::exp_metalforge_v18::SCENARIO);
    r.register(super::exp_metalforge_v19::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v59_science::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v6::SCENARIO);
    r.register(super::exp_metalforge_v7_mixed::SCENARIO);
    r.register(super::exp_metalforge_v8_cross_system::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_metalforge_v9_nucleus::SCENARIO);
    r.register(super::exp_multi_signal::SCENARIO);
    r.register(super::exp_myxococcus_critical_density::SCENARIO);
    r.register(super::exp_mzml::SCENARIO);
    r.register(super::exp_nanopore_int8_quantization::SCENARIO);
    r.register(super::exp_nanopore_signal_bridge::SCENARIO);
    r.register(super::exp_nanopore_simulated_16s::SCENARIO);
    r.register(super::exp_ncbi_pangenome::SCENARIO);
    r.register(super::exp_ncbi_qs_atlas::SCENARIO);
    r.register(super::exp_ncbi_qs_habitat::SCENARIO);
    r.register(super::exp_ncbi_vibrio_qs::SCENARIO);
    r.register(super::exp_neighbor_joining::SCENARIO);
    r.register(super::exp_newick_parse::SCENARIO);
    r.register(super::exp_niche_parity_v1::SCENARIO);
    r.register(super::exp_nitrifying_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_nmf_drug_repurposing::SCENARIO);
    r.register(super::exp_notill_brandt_farm::SCENARIO);
    r.register(super::exp_notill_longterm_tillage::SCENARIO);
    r.register(super::exp_notill_meta_analysis::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_nouveau_diagnostic_v1::SCENARIO);
    r.register(super::exp_npu_bloom_sentinel::SCENARIO);
    r.register(super::exp_npu_disorder_classifier::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(super::exp_npu_funky::SCENARIO);
    r.register(super::exp_npu_genome_binning::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(super::exp_npu_hardware::SCENARIO);
    #[cfg(feature = "npu")]
    r.register(super::exp_npu_live::SCENARIO);
    r.register(super::exp_npu_phylo_placement::SCENARIO);
    r.register(super::exp_npu_qs_classifier::SCENARIO);
    r.register(super::exp_npu_sentinel_stream::SCENARIO);
    r.register(super::exp_npu_spectral_screen::SCENARIO);
    r.register(super::exp_npu_spectral_triage::SCENARIO);
    r.register(super::exp_nucleus_data_pipeline::SCENARIO);
    r.register(super::exp_nucleus_live_gonzales::SCENARIO);
    r.register(super::exp_nucleus_tower_node::SCENARIO);
    r.register(super::exp_nucleus_v4::SCENARIO);
    r.register(super::exp_nucleus_v8_mixed::SCENARIO);
    r.register(super::exp_p1_extensions_v1::SCENARIO);
    r.register(super::exp_pangenomics::SCENARIO);
    r.register(super::exp_paper_math_control_v1::SCENARIO);
    r.register(super::exp_paper_math_control_v2::SCENARIO);
    r.register(super::exp_paper_math_control_v3::SCENARIO);
    r.register(super::exp_paper_math_control_v4::SCENARIO);
    r.register(super::exp_paper_math_control_v5::SCENARIO);
    r.register(super::exp_paper_math_control_v6::SCENARIO);
    r.register(super::exp_pcie_direct::SCENARIO);
    r.register(super::exp_peaks::SCENARIO);
    r.register(super::exp_petaltongue_anderson_v1::SCENARIO);
    r.register(super::exp_petaltongue_biogas_v1::SCENARIO);
    r.register(super::exp_petaltongue_live_v1::SCENARIO);
    r.register(super::exp_pfas::SCENARIO);
    r.register(super::exp_pfas_decision_tree::SCENARIO);
    r.register(super::exp_pfas_library::SCENARIO);
    r.register(super::exp_phage_defense::SCENARIO);
    r.register(super::exp_phosphorus_phylogenomics::SCENARIO);
    r.register(super::exp_phylo_placement_scale::SCENARIO);
    r.register(super::exp_phylohmm::SCENARIO);
    r.register(super::exp_phynetpy_rf::SCENARIO);
    r.register(super::exp_physical_comm_anderson::SCENARIO);
    r.register(super::exp_placement::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_planktonic_dilution::SCENARIO);
    r.register(super::exp_population_genomics::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_precision_brain_v1::SCENARIO);
    r.register(super::exp_primal_parity_v1::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_primal_pipeline_v1::SCENARIO);
    r.register(super::exp_producer_receiver_qs::SCENARIO);
    r.register(super::exp_public_benchmarks::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_complete::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming::SCENARIO);
    r.register(super::exp_pure_gpu_streaming_v10::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v11::SCENARIO);
    r.register(super::exp_pure_gpu_streaming_v12::SCENARIO);
    r.register(super::exp_pure_gpu_streaming_v13::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v2::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v3::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v4::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v6::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v7::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v8::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_pure_gpu_streaming_v9::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_qs_disorder_real::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_qs_distance_scaling::SCENARIO);
    r.register(super::exp_qs_gene_prevalence::SCENARIO);
    r.register(super::exp_qs_gene_profiling_v1::SCENARIO);
    r.register(super::exp_qs_ode::SCENARIO);
    r.register(super::exp_qs_wave_localization::SCENARIO);
    r.register(super::exp_r_industry_parity::SCENARIO);
    r.register(super::exp_rare_biosphere::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_real_bloom_gpu::SCENARIO);
    r.register(super::exp_real_ncbi_pipeline::SCENARIO);
    r.register(super::exp_reconciliation::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_repodb_nmf::SCENARIO);
    r.register(super::exp_rf_distance::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_s86_streaming_pipeline::SCENARIO);
    r.register(super::exp_sate_pipeline::SCENARIO);
    r.register(super::exp_science_pipeline::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_skin_anderson_s79::SCENARIO);
    r.register(super::exp_soil_biofilm_aggregate::SCENARIO);
    r.register(super::exp_soil_distance_colonization::SCENARIO);
    r.register(super::exp_soil_pore_diversity::SCENARIO);
    r.register(super::exp_soil_qs_cpu_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_soil_qs_gpu::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_soil_qs_metalforge::SCENARIO);
    r.register(super::exp_soil_qs_pore_geometry::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_soil_qs_streaming::SCENARIO);
    r.register(super::exp_soil_structure_function::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_sovereign_dispatch_v1::SCENARIO);
    r.register(super::exp_sovereign_resequencing::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_spectral_cross_spring::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_square_cubed_scaling::SCENARIO);
    r.register(super::exp_stable_specials_v1::SCENARIO);
    r.register(super::exp_streaming_dispatch::SCENARIO);
    r.register(super::exp_streaming_io_parity::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_streaming_ode_phylo::SCENARIO);
    r.register(super::exp_streaming_pipeline_v5::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_substrate_router::SCENARIO);
    r.register(super::exp_sulfur_phylogenomics::SCENARIO);
    r.register(super::exp_temporal_esn_bloom::SCENARIO);
    r.register(super::exp_tillage_microbiome_2025::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_toadstool_bio::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_toadstool_dispatch_v2::SCENARIO);
    r.register(super::exp_toadstool_dispatch_v3::SCENARIO);
    r.register(super::exp_toadstool_dispatch_v4::SCENARIO);
    r.register(super::exp_toadstool_s70_rewire::SCENARIO);
    r.register(super::exp_trophic_cascade::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_vent_chimney_qs::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_vibrio_qs_landscape::SCENARIO);
    r.register(super::exp_viral_metagenomics::SCENARIO);
    r.register(super::exp_visualization_v1::SCENARIO);
    r.register(super::exp_visualization_v2::SCENARIO);
    r.register(super::exp_voc_peaks::SCENARIO);
    #[cfg(feature = "gpu")]
    r.register(super::exp_workload_routing_v1::SCENARIO);
}
