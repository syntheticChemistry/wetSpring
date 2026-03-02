# wetSpring — Paper Review Queue

**Last Updated**: March 2, 2026 (Phase 92H — 1,219 tests, ToadStool S86 (`2fee1969`), 144 primitives (264 ComputeDispatch ops), clippy pedantic CLEAN, 279 experiments, 52 papers complete + 6 reproduced, all 39 three-tier eligible papers validated at CPU + GPU + metalForge. V92H: CPU↔GPU full domain (Exp301, 48/48), NUCLEUS+PCIe+biomeOS (Exp302, 113/113), mixed NUCLEUS orchestration (Exp303, 147/147))
**Purpose**: Track papers for reproduction/review across five tracks

---

## Completed Reproductions

| # | Paper / Pipeline | Phase | Checks | Track |
|---|-----------------|-------|--------|-------|
| 1 | Galaxy/QIIME2 16S pipeline (Exp 001-004) | 1-2 | 92/92 | Track 1 |
| 2 | asari LC-MS feature extraction (Exp 005-006) | 1-2 | 26/26 | Track 2 |
| 3 | FindPFAS screening pipeline (Exp 007-008) | 1-2 | 17/17 | Track 2 |
| 4 | GPU diversity + spectral matching | 3 | 38/38 | All |
| 5 | Sovereign 16S pipeline (DADA2 + chimera + taxonomy + UniFrac) | 4 | 37/37 | Track 1 |
| 6 | Algae pond 16S on real NCBI data (Exp012, PRJNA488170 proxy) | 5 | 29/29 | Track 1 |
| 7 | VOC peak validation vs Reese 2019 Table 1 (Exp013) | 5 | 22/22 | Track 1 |
| 8 | Public data benchmark — 4 BioProjects vs paper ground truth (Exp014) | 6 | 202/202 | Track 1 |
| 9 | Exp019 Phases 2-4: PhyNetPy RF (Exp036), PhyloNet-HMM (Exp037), SATe pipeline (Exp038) | — | 42/42 | Track 1b |
| 10 | Cahill proxy — algal pond time-series (Exp039) | — | 11/11 | Track 1 |
| 11 | Smallwood proxy — bloom surveillance (Exp040) | — | 15/15 | Track 1 |
| 12 | Jones F&T proxy — EPA PFAS ML (Exp041) | — | 14/14 | Track 2 |
| 13 | Jones MS proxy — MassBank spectral (Exp042) | — | 9/9 | Track 2 |

---

## Review Queue

### Track 1 — Microbial Ecology & Signaling (Waters, Cahill, Smallwood)

#### Tier 1 — High priority, fully specified models

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 5 | Waters et al. "Quorum Sensing Controls Biofilm Formation in V. cholerae Through Modulation of Cyclic Di-GMP" | J Bacteriology 190:2527-36 | 2008 | Waters | Foundational QS ↔ c-di-GMP model. ODE system for signal-dependent biofilm phenotype. Fully specified — direct reproduction target | **Exp020 DONE** |
| 6 | Massie et al. "Quantification of High Specificity Cyclic di-GMP Signaling" | PNAS 109:12746-51 | 2012 | Waters | How cells resolve signal from noise with 60+ enzymes controlling one molecule. Stochastic simulation primitives | **Exp022 DONE** |
| 7 | Hsueh, Severin et al. "A Broadly Conserved Deoxycytidine Deaminase Protects Bacteria from Phage Infection" | Nature Microbiology 7:1210-1220 | 2022 | Waters | Phage defense — evolutionary arms race. Connects to Cahill phage biocontrol work | **Exp030 DONE** |

#### Tier 2 — Strong candidates

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 8 | Fernandez et al. "V. cholerae adapts to sessile and motile lifestyles by c-di-GMP regulation of cell shape" | PNAS 117:29046-29054 | 2020 | Waters | Phenotypic switching = bistable dynamical system. Bifurcation analysis | **Exp023 DONE** |
| 9 | Mhatre et al. "One gene, multiple ecological strategies" | PNAS 117:21647-21657 | 2020 | Waters | Capacitor for diversity — single node enabling multiple phenotypes | **Exp027 DONE** |
| 10 | Bruger & Waters "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation" | AEM 84:e00402-18 | 2018 | Waters | Game-theoretic cooperation. Evolutionary strategy landscapes | **Exp025 DONE** |
| 11 | Waters "Au naturale: cyclic di-nucleotides for cancer immunotherapy" | Open Biol 11:210277 | 2021 | Waters | Translational application of fundamental c-di-GMP understanding | Reference |
| 12 | Srivastava et al. "Integration of Cyclic di-GMP and Quorum Sensing" | J Bacteriology 193:6331-41 | 2011 | Waters | Multi-input regulatory network integration | **Exp024 DONE** |

#### Tier 3 — Sandia connections (may require data access)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 13 | Cahill et al. — Phage-mediated biocontrol in algal raceway ponds | — | Cahill | Time-series anomaly detection for pond crash | **Exp039 DONE (Cahill proxy)** |
| 14 | Smallwood et al. — Raceway pond metagenomic surveillance | — | Smallwood | Real-world validation of 16S pipeline on pond data | **Exp040 DONE (Smallwood proxy)** |

### Track 1b — Comparative Genomics & Phylogenetics (Liu)

#### Tier 1

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 15 | Liu et al. "An HMM-based Comparative Genomic Framework for Detecting Introgression" | PLoS Comp Bio 10:e1003649 | 2014 | Liu | PhyloNet-HMM — HMM on genomic data. Validates sequence model primitives | **Exp026 + Exp037 DONE (HMM + PhyloNet-HMM)** |
| 16 | Alamin & Liu "Phylogenetic Placement of Aligned Genomes and Metagenomes with Non-tree-like Histories" | IEEE/ACM TCBB | 2024 | Liu | Metagenomic placement — classifying environmental samples. Direct wetSpring application | **Exp032 DONE** |

#### Tier 2

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 17 | Liu et al. "Rapid and accurate large-scale coestimation of sequence alignments and phylogenetic trees" (SATé) | Science 324:1561-1564 | 2009 | Liu | Divide-and-conquer + iterative refinement at massive scale | **Exp033 + Exp038 DONE (NJ + SATe pipeline)** |
| 18 | Zheng et al. "Impact of Species Tree Estimation Error on Cophylogenetic Reconstruction" | BCB (top 10%) | 2023 | Liu | Host-microbe coevolution methods | **Exp034 DONE** |
| 19 | Liu (working manuscript) — Fungi-bacteria coevolution (Burkholderiaceae) | — | Liu | Direct wetSpring application — fungal-bacterial symbiosis | Watch |
| 20 | Wang et al. "Build a better bootstrap and the RAWR shall beat a random path to your door" | Bioinformatics (ISMB) 37:i111-i119 | 2021 | Liu | Modern resampling for phylogenetic confidence | **Exp031 DONE** |

### Track 2 — Analytical Chemistry & PFAS (Jones)

| # | Paper | Year | Faculty | Why | Status |
|---|-------|------|---------|-----|--------|
| 21 | Jones et al. — PFAS mass spectrometry detection pipelines | — | Jones | Extends current PFAS screening (Exp 007-008) | **Exp042 DONE (Jones MS proxy)** |
| 22 | Jones et al. — Environmental PFAS fate-and-transport modeling | — | Jones | From detection to prediction | **Exp041 DONE (Jones F&T proxy)** |

### Track 1c — Deep-Sea Metagenomics & Microbial Evolution (R. Anderson)

Rika Anderson (Carleton College) studies microbial and viral evolution in
deep-sea hydrothermal vents using metagenomics, pangenomics, and population
genomics — the same computational methods wetSpring validates in its sovereign
pipeline. Her work is the empirical corollary to the Taq polymerase argument
in `gen3/CONSTRAINED_EVOLUTION_FORMAL.md`.

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 24 | Anderson et al. "Genomic variation in microbial populations inhabiting the marine subseafloor at deep-sea hydrothermal vents" | Nature Communications 8:1114 | 2017 | R. Anderson | Population-level genomic variation shaped by geochemistry. SNP/dN/dS analysis on metagenomic data — directly exercises wetSpring's sovereign alignment and diversity pipelines | **Exp055 DONE** |
| 25 | Moulana, Anderson et al. "Selection is a significant driver of gene gain and loss in the pangenome of Sulfurovum" | mSystems 5:e00673-19 | 2020 | R. Anderson | Pangenomics under environmental constraint. Gene gain/loss driven by geochemistry at vents — constrained evolution of microbial functional repertoires. Direct computational reproduction target | **Exp056 DONE** |
| 26 | Mateos, Anderson et al. "The evolution and spread of sulfur-cycling enzymes reflect the redox state of the early Earth" | Science Advances 9:eade4847 | 2023 | R. Anderson | Phylogenomics across 3+ billion years of enzyme evolution. Tree reconciliation, molecular clock analysis, co-evolution of enzymes with geochemistry. Undergraduate co-authors — reproducible pipeline | **Exp053 DONE** |
| 27 | Boden, Anderson et al. "Timing the evolution of phosphorus-cycling enzymes through geological time" | Nature Communications 15:3703 | 2024 | R. Anderson | Same tree reconciliation methodology as Paper 26, applied to phosphorus cycle. Geobiology + bioinformatics | **Exp054 DONE** |
| 28 | Anderson et al. "Evolutionary strategies of viruses and cells in hydrothermal systems revealed through metagenomics" | PLoS ONE 9:e109696 | 2014 | R. Anderson | Viral metagenomics — phage-host interactions in vent ecosystems. Connects to Cahill (algae pond phage) and Waters (phage defense) | **Exp052 DONE** |
| 29 | Anderson, Sogin, Baross "Biogeography and ecology of the rare and abundant microbial lineages in deep-sea hydrothermal vents" | FEMS Microbiol Ecol 91:fiu016 | 2015 | R. Anderson | Rare biosphere — when does a microbial lineage constitute signal vs sampling noise? Directly connects to groundSpring Exp 004 (sequencing depth saturation) | **Exp051 DONE** |

**Why Anderson matters for wetSpring**: Her lab's computational pipelines (MAGs,
pangenomics, SNP analysis, tree reconciliation, viral metagenomics) are exactly
what wetSpring's sovereign 16S/metagenomics pipeline validates. Reproducing her
work would demonstrate that wetSpring's Rust pipeline produces the same biological
conclusions as the Galaxy/QIIME2/Python stack her lab uses. Furthermore, her
co-authors include undergraduate students, suggesting the methods are well-documented
and reproducible — ideal reproduction targets.

### Track 3 — Drug Repurposing via Matrix Mathematics (Fajgenbaum / Every Cure)

Dr. David Fajgenbaum (UPenn) — nearly died 5 times from idiopathic multicentric
Castleman disease. He repurposed sirolimus (a 25-year-old transplant drug) to save
his own life, then built the MATRIX platform (now Every Cure, everycure.org) to
systematically match all ~4,000 FDA-approved drugs to ~18,000 diseases using matrix
factorization, knowledge graph embeddings, and cosine similarity scoring.

The math is linear algebra: NMF, SVD, sparse matrix operations on drug-disease
scoring matrices (~4,000 × 18,000). This is within BarraCUDA's GEMM capability
on a single GPU.

**Connection**: BarraCUDA linear algebra applied to biomedical knowledge graphs.
neuralSpring ML primitives for drug-disease scoring. wetSpring biology pipeline
extended into pharmacology. Links to Jones (analytical chemistry / drug detection),
Waters (c-di-GMP as a drug target), Murillo (surrogate learning for fast scoring).

#### Tier 1 — Fajgenbaum Core Papers

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 39 | Fajgenbaum et al. "Identifying and targeting pathogenic PI3K/AKT/mTOR signaling in IL-6-blockade-refractory iMCD" | J Clin Invest 129(10):4451-4463 | 2019 | Fajgenbaum | The original sirolimus discovery. The paper that saved his life. Computational protocol for drug-pathway matching | **Exp157 DONE** (8/8) |
| 40 | Fajgenbaum et al. "Pioneering a new field of computational pharmacophenomics" | Lancet Haematology 12(2):e94-e96 | 2025 | Fajgenbaum | MATRIX methodology overview. Defines the computational framework for systematic drug repurposing | **Exp158 DONE** (9/9) |

#### Tier 2 — Matrix Factorization for Drug Repurposing (The Math)

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 41 | Yang et al. "Matrix Factorization-based Technique for Drug Repurposing Predictions" | PMC (pubmed 32365039) | 2020 | — | NMF/SVD applied to drug-disease scoring matrices. The core algorithm to reproduce | **Exp159 DONE** (7/7) |
| 42 | Gao et al. "Non-Negative Matrix Factorization for Drug Repositioning: Experiments with the repoDB Dataset" | PMC7153111 | 2020 | — | NMF on the repoDB dataset (1,571 drugs × 1,209 diseases). Benchmark dataset for validation | **Exp160 DONE** (9/9) |
| 43 | ROBOKOP knowledge graph papers | Various | — | — | Infrastructure behind MATRIX. Knowledge graph embedding for drug-disease relationships | **Exp161 DONE** (7/7) |

#### BarraCUDA Requirements for Drug Repurposing

| Primitive | Shader | Status | Notes |
|-----------|--------|--------|-------|
| GEMM (f64) | `gemm_f64.wgsl` | ✅ Upstream | 4,000 × 18,000 well within single-GPU capacity |
| SVD (f64) | `svd_f64.wgsl` | ✅ Upstream | Jacobi SVD, one-sided |
| NMF (f64) | `barracuda::linalg::nmf` | ✅ Upstream (S58) | Absorbed from wetSpring V30. Returns `Result<NmfResult>`. Multiplicative update, Lee & Seung (1999) |
| Ridge regression | `barracuda::linalg::ridge` | ✅ Upstream (S59) | Cholesky-based. Used by ESN readout |
| Sparse GEMM (CSR×Dense) | `barracuda::ops::sparse_gemm_f64` | ✅ Upstream (S60) | Drug-disease matrices are sparse (~5% fill) |
| TransE scoring | `barracuda::ops::transe_score_f64` | ✅ Upstream (S60) | GPU-parallel triple scoring for KG embeddings |
| Peak detection | `barracuda::ops::peak_detect_f64` | ✅ Upstream (S62) | GPU parallel local-maxima, prominence, width |
| Cosine similarity | `barracuda::linalg::nmf::cosine_similarity` | ✅ Upstream | Pairwise scoring on factor matrices |
| Top-K selection | Inlined in `validate_matrix_pharmacophenomics` | ✅ Local | Rank drug-disease pairs by score. Candidate for upstream |

### Track 4 — No-Till Soil QS & Anderson Geometry (baseCamp Sub-thesis 06)

baseCamp Sub-thesis 06 applies the Anderson localization framework to explain
no-till soil health outcomes. These reproduction targets validate the QS-geometry
coupling in soil pore networks and provide the microbial community data for
tilled vs no-till comparisons.

#### Tier 1 — Direct Reproduction Targets

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 44 | Martínez-García et al. "Spatial structure, chemotaxis and quorum sensing shape bacterial biomass accumulation in complex porous media" | Nature Communications 14:8332 | 2023 | — | **Critical**: First paper to show QS + spatial 3D pore structure determines bacterial coordination in porous media. Direct validation of Anderson QS-geometry coupling in soil-like systems | **Exp170 DONE** (26/26) |
| 45 | Feng et al. "Composition and metabolism of microbial communities in soil pores" | Nature Communications 15:3578 | 2024 | — | Microbial diversity differs between large (30-150 µm) and small (4-10 µm) soil pores. Pore-scale Anderson geometry data — different pore sizes = different effective lattice dimensions | **Exp171 DONE** (27/27) |
| 46 | Mukherjee et al. "Manipulating the physical distance between cells during soil colonization reveals the importance of biotic interactions" | Environmental Microbiome 19:14 | 2024 | — | Physical proximity affects community assembly. 41% of dominant groups affected by cell distancing. Validates Anderson's distance/geometry dependence for QS | **Exp172 DONE** (23/23) |

#### Tier 2 — No-Till Microbiome Data

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 47 | Islam et al. "No-till and conservation agriculture in the United States: An example from the David Brandt farm, Carroll, Ohio" | ISWCR 2:97-107 | 2014 | — | Brandt farm soil health data: microbial biomass, aggregate stability, active carbon. The no-till dataset for Anderson analysis | **Exp173 DONE** (14/14) |
| 48 | Zuber & Villamil "Meta-analysis approach to assess effect of tillage on microbial biomass and enzyme activities" | Soil Biology and Biochemistry 97:176-187 | 2016 | — | Meta-analysis: no-till increases microbial biomass C by 16-20%. Systematic evidence for geometry → microbial function link | **Exp174 DONE** (20/20) |
| 49 | Liang et al. "Long term tillage, cover crop, and fertilization effects on microbial community structure, activity" | Soil Biology and Biochemistry 89:37-44 | 2015 | — | 31+ year study: greater mycorrhizal fungi under no-till. Tillage × cover crop × N interaction. Data source for Anderson diversity mapping | **Exp175 DONE** (19/19) |

#### Tier 3 — Soil Structure & QS Dynamics

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 50 | Tecon & Or "Biophysics of bacterial biofilms—insights from soil" | Biochimica et Biophysica Acta 1858:2774-2781 | 2017 | — | Review of soil aggregate geometry → biofilm formation → QS. Bridges Anderson physics to soil microbiology | **Exp176 DONE** (23/23) |
| 51 | Rabot et al. "Soil structure as an indicator of soil functions: A review" | Geoderma 314:122-137 | 2018 | — | Soil structure → microbial functions. Framework for mapping aggregate stability to effective Anderson dimension | **Exp177 DONE** (16/16) |
| 52 | Wang et al. "Effects of tillage practices in stover-return on endosphere and rhizosphere microbiomes" | npj Sustainable Agriculture 3:12 | 2025 | — | 2025 study: different tillage → different endosphere/rhizosphere microbiomes. Geometry-dependent community assembly | **Exp178 DONE** (15/15) |

**Connection to existing wetSpring work**: Track 4 extends the Anderson-QS
framework (Exp107-143, Phase 37-38) from natural biome predictions to
agricultural soil systems. Papers 44-46 are the soil pore-scale equivalents
of the 3D Anderson lattice. Papers 47-49 provide the tilled-vs-no-till
microbial data. Papers 50-52 bridge soil structure science to the Anderson model.
16S data from these studies would be processed through wetSpring's sovereign
Rust pipeline.

### Track 5 — Immunological Anderson & Drug Repurposing (Gonzales / Fajgenbaum Bridge)

Andrea J. Gonzales (Zoetis → MSU Pharmacology & Toxicology, 2025–present)
provides the empirical immunological foundation for extending Anderson
localization from microbial QS (Track 1, Paper 01) to cytokine signal
propagation in skin tissue. The core observation: Th2 cytokines (IL-4,
IL-13, IL-31) are diffusible signals propagating through a disordered
biological medium — the same physics that governs autoinducer propagation
through microbial communities.

**Connection to Fajgenbaum (Track 3)**: The MATRIX drug repurposing framework
scores drug-disease pairs by pathway overlap. Paper 12 adds a spatial
geometry dimension: a drug must both target the right pathway AND physically
reach its target through tissue geometry. Anderson localization quantifies
the "reach" condition.

#### Tier 1 — Gonzales Core Papers

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| 53 | Gonzales AJ et al. "Interleukin-31: its role in canine pruritus and naturally occurring canine atopic dermatitis" | Vet Dermatol 24:48-53 | 2013 | Gonzales | IL-31 elevated in AD dog serum; IV IL-31 induces pruritus in beagles; IL-31 activates peripheral nerves. IL-31 as diffusible signal, W mapping from tissue heterogeneity | **Exp282 DONE** (15/15) |
| 54 | Gonzales AJ et al. "Oclacitinib (APOQUEL) is a novel JAK inhibitor with activity against cytokines involved in allergy" | J Vet Pharmacol Ther 37:317-324 | 2014 | Gonzales | JAK1 IC50 = 10 nM; blocks IL-2, IL-4, IL-6, IL-13, IL-31 (IC50 36-249 nM). Dose-response modeling, IC50 as Anderson barrier height | **Exp280 DONE** (35/35) |
| 55 | Gonzales AJ et al. "IL-31-induced pruritus in dogs: a novel experimental model" | Vet Dermatol 27:34-e10 | 2016 | Gonzales | Standardized IL-31 pruritus model; oclacitinib superior at 1, 6, 11, 16 hr. Time-series pruritus data for LSTM, controlled Anderson perturbation | **Exp281 DONE** (19/19, D04) |
| 56 | Fleck TJ,...,Gonzales AJ "Onset and duration of action of lokivetmab in IL-31 induced pruritus" | Vet Dermatol 32:681-e182 | 2021 | Gonzales | Cytopoint: 3 hr onset, dose-dependent duration (14/28/42 days). PK decay as signal extinction in Anderson model | **Exp281 DONE** (19/19, D01-D03) |
| 57 | Gonzales AJ et al. "Oclacitinib is a selective JAK1 inhibitor with efficacy in canine flea allergic dermatitis" | J Vet Pharmacol Ther 47:447-453 | 2024 | Gonzales | JAK1 selectivity confirmed in different allergic model. Cross-disease validation of same Anderson pathway | **Exp280 DONE** (35/35, D03) |
| 58 | McCandless EE, Rugg CA, Fici GJ et al. "Allergen-induced production of IL-31 by canine Th2 cells and identification of immune, skin, and neuronal target cells" | Vet Immunol Immunopathol 157:42-48 | 2014 | Gonzales | IL-31 produced by Th2 cells; target cells = immune, skin, neuronal. Three-compartment Anderson lattice | **Exp281 DONE** (19/19, D05) |

#### Tier 2 — Companion Literature (Fajgenbaum Bridge + Human AD)

| # | Paper | Journal | Year | Faculty | Why | Status |
|---|-------|---------|------|---------|-----|--------|
| F1 | Fajgenbaum DC et al. "Identifying and targeting pathogenic PI3K/AKT/mTOR signaling in IL-6-blockade-refractory iMCD" | J Clin Invest | 2019 | Fajgenbaum | Proves pathway-based drug repurposing; mTOR cross-talks with JAK/STAT. Already Exp157 | Reference |
| D1 | Simpson et al. "Dupilumab Phase 3 trials" | N Engl J Med | 2020 | — | Human anti-IL-4Rα for AD — blocks IL-4 + IL-13. Cross-species validation of Gonzales's canine work | Reference |
| D2 | Silverberg et al. "JAK inhibitors in AD" | J Am Acad Dermatol | 2023 | — | Upadacitinib, abrocitinib for human AD — human equivalents of Apoquel | Reference |
| N1 | Oetjen et al. "Sensory neurons co-opt immune cells for AD pathogenesis" | Cell | 2023 | — | IL-4/IL-13 directly sensitize sensory neurons — neuro-immune axis | Reference |

#### wetSpring Experiments

| Exp | Description | Validates | Status |
|-----|-------------|-----------|--------|
| Exp273 | Anderson lattice with skin-layer geometry: 2D epidermis + 3D dermis + barrier interface | Core Anderson prediction for immunological signaling | **22/22 PASS** |
| Exp274 | Barrier disruption model: dimensional promotion, P06↔P12 duality, Fajgenbaum scoring | AD scratch cycle as inverse of Paper 06 tillage collapse | **15/15 PASS** |
| Exp275 | Cell-type heterogeneity sweep: W sweep, cross-species, disease profiles | Prediction: inflammation increases W but stays below W_c | **11/11 PASS** |
| Exp276 | CPU parity: immuno-Anderson (alpha diversity, spectral, Pielou→W, Fajgenbaum) | Pure Rust math correctness for Paper 12 framework | **32/32 PASS** |
| Exp277 | GPU validation: immuno-Anderson diversity (Shannon, Simpson, Bray-Curtis on GPU) | GPU portability of Paper 12 math | **21/21 PASS** |
| Exp278 | ToadStool streaming: batched GPU pipeline for immuno-Anderson | Streaming dispatch reduces round-trips | **31/31 PASS** |
| Exp279 | metalForge cross-substrate: CPU↔GPU parity + NUCLEUS atomics | Cross-hardware portability | **25/25 PASS** |
| Exp280 | Gonzales 2014 IC50 dose-response: Hill equation, JAK selectivity, Anderson barrier map | Paper 54+57 reproduction (published data) | **35/35 PASS** |
| Exp281 | Fleck/Gonzales 2021 PK: dose-duration, exponential decay, pruritus model, three-compartment | Paper 53+55+56+58 reproduction (published data) | **19/19 PASS** |
| Exp282 | Gonzales 2013 IL-31 serum: dose-response, receptor→lattice, Anderson spectral, cross-species | Paper 53 reproduction (published data, Anderson mapping) | **15/15 PASS** |
| Exp283 | CPU parity: Gonzales reproductions (Hill, regression, diversity, Anderson, IC50→barrier) | Pure Rust math correctness for Papers 53-56 | **43/43 PASS** |
| Exp284 | GPU validation: Gonzales diversity (Shannon, Simpson, Pielou, Bray-Curtis on GPU) | GPU portability of pharmacological math | **17/17 PASS** |
| Exp285 | ToadStool streaming: Gonzales batched GPU pipeline (Shannon, Simpson, BC matrix) | Streaming dispatch for pharmacological workloads | **37/37 PASS** |
| Exp286 | metalForge cross-substrate: Gonzales CPU↔GPU + Hill + Anderson + NUCLEUS | Cross-hardware portability for drug reproductions | **36/36 PASS** |

### Cross-Spring — Spectral Theory (Kachkovskiy, via groundSpring/hotSpring)

Kachkovskiy's spectral theory has an indirect but real connection to wetSpring
through signal processing in mass spectrometry (Track 2) and through the
mathematical framework for understanding quorum sensing as signal propagation
in a noisy medium (Track 1).

| # | Paper | Faculty | Why | Status |
|---|-------|---------|-----|--------|
| 23 | Bourgain & Kachkovskiy (2018) "Anderson localization for two interacting quasiperiodic particles." GAFA | Kachkovskiy | Quorum sensing = signal propagation through a "disordered" bacterial population. Anderson localization theory describes when signals (autoinducers) reach distant cells vs. when they're absorbed by local noise. Mathematical bridge to Waters' c-di-GMP specificity problem | **DONE** (Exp107: 25/25 checks — Anderson 1D/2D/3D, Almost-Mathieu, Lanczos, QS-disorder analogy) |

---

## Open Data Provenance Audit

All 52 reproductions use publicly accessible data or published model parameters.
No proprietary data dependencies.

| Category | Papers | Data Source | Access |
|----------|:------:|------------|--------|
| **ODE/Stochastic models** | 5-12 | Model parameters from published paper equations | Open (journal) |
| **Real 16S amplicon** | Exp001,012,014,017 | NCBI SRA (PRJNA488170, PRJNA382322, PRJNA1114688, etc.) | Open (NCBI) |
| **VOC biomarkers** | Exp013 | Table 1 from Reese 2019 (PMC6761164) | Open (PMC) |
| **PFAS screening** | Exp005-006 | asari test data (MT02) | Open (asari package) |
| **PFAS library** | Exp018 | Jones Lab (Zenodo 14341321) | Open (Zenodo) |
| **PFAS monitoring** | Exp008 | Michigan EGLE (ArcGIS REST API) | Open (state gov) |
| **PFAS ML** | Exp041 | EPA public PFAS data | Open (EPA) |
| **Spectral library** | Exp042 | MassBank | Open (MassBank) |
| **Phylogenetics** | 15-20 | Algorithms from published papers; PhyNetPy/SATé public datasets | Open (journal + repo) |
| **Deep-sea genomics** | Exp051-056 | NCBI SRA (PRJNA283159, PRJEB5293), MBL darchive, MG-RAST, Figshare, OSF | Open (public repos) |
| **Sandia proxy** | Exp039-040 | Synthetic proxy data (not original Sandia datasets) | Open (generated) |
| **Cross-spring** | Paper 23 (Exp107) | `barracuda::spectral` primitives (Anderson, Lanczos, level statistics) | Open (algorithmic — no external data) |
| **NCBI-scale GPU** | Exp108-113 | Synthetic at NCBI-realistic scale (Vibrio, MassBank, Campylobacterota, HAB, HMP/Tara) | Open (synthetic, mirrors NCBI) |
| **NPU reservoir** | Exp114-119 | ESN trained on GPU output, int8 quantized for Akida AKD1000 | Open (ESN weights from open training data) |
| **Drug repurposing** | Exp157-165 | repoDB (1,571 × 1,209), published equations, ROBOKOP KG | Open (PMC, repoDB, algorithmic) |
| **Diversity fusion** | Exp167 | Synthetic abundance data (CPU ↔ GPU parity) | Open (generated) |
| **Track 4 soil QS** | Exp170-182 | Published soil metrics (Islam 2014, Zuber 2016, Liang 2015), model equations (Martínez-García 2023, Mukherjee 2024), pore geometry data (Feng 2024), review frameworks (Tecon & Or 2017, Rabot 2018), tillage microbiome (Wang 2025) | Open (journal tables, published equations) |

### Validation Tiers by Hardware

| Tier | Description | Experiments | Checks |
|------|-------------|:-----------:|:------:|
| **BarraCuda CPU** | Rust math matches Python baselines | Exp035,043,057,070,079,085,102,163 | 407/407 |
| **BarraCuda GPU** | GPU math matches CPU reference | Exp064,071,087,092,101,164 | 1,783 |
| **metalForge** | Substrate-independent output (CPU/GPU/NPU) | Exp060,065,080,084,086,088,093,103,104,165 | 243+ |
| **Streaming** | Pure GPU pipeline, zero CPU round-trips | Exp072,073,075,089,090,091,105,106 | 252+ |
| **Cross-spring** | neuralSpring + spectral theory primitives | Exp094,095,107 | 71 |
| **NCBI-scale** | Real-scale data extensions | Exp108-113 | 78 |
| **NPU reservoir** | ESN → int8 → Akida deployment | Exp114-119 | 59 |
| **Cross-spring evolution** | 660+ WGSL shaders traced, imports rewired | Exp120 | 9 |
| **Phase 37-38 extensions** | Anderson-QS extension papers + cold seep + phylogeny | Exp144-149,152-156 | 102 |
| **Phase 40 scaling** | Finite-size + correlated disorder + physical comm | Exp150-151 | 22 |
| **Drug repurposing (Track 3)** | NMF, pathway scoring, KG embedding, metalForge | Exp157-165 | 84 |
| **Modern systems (S62+DF64)** | BGL helpers, DF64, modern dispatch | Exp166 | 19 |
| **Write-phase extensions** | Diversity fusion WGSL (CPU ↔ GPU parity) | Exp167 | 18 |
| **Track 4 soil QS (CPU)** | Anderson-QS in soil pores, no-till data, structure→function | Exp170-178 | 183 |
| **Track 4 soil QS (GPU/MF)** | CPU parity, GPU, streaming, metalForge | Exp179-182 | 138 |
| **Track 5 immuno-Anderson (science)** | Skin lattice, barrier disruption, heterogeneity sweep | Exp273-275 | 48 |
| **Track 5 immuno-Anderson (3-tier)** | CPU parity, GPU, streaming, metalForge | Exp276-279 | 109 |
| **Track 5 Gonzales (science)** | IC50 dose-response, PK decay, IL-31 serum, Anderson mapping | Exp280-282 | 69 |
| **Track 5 Gonzales (3-tier)** | CPU parity, GPU, streaming, metalForge | Exp283-286 | 133 |
| **Paper Math Control v4** | All 52 papers' core equations | Exp291 | 45 |
| **CPU v22 Comprehensive** | Full-domain CPU paper parity (8 domains) | Exp292 | 40 |
| **GPU v9 Portability** | 5-track GPU dispatch + Anderson W-map | Exp293 | 35 |
| **Pure GPU Streaming v9** | End-to-end pipeline: diversity→BC→NMF→Anderson→stats | Exp294 | 16 |
| **metalForge v14 Paper Chain** | Cross-system paper math (GPU→CPU transitions) | Exp295 | 28 |

### Phase 37 — Anderson-QS Extension Papers

Papers identified during literature review for the Anderson-QS framework.
Core finding: **no prior work applies Anderson localization to QS signaling**.

#### Tier 1 — Directly Extends Framework

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| 30 | "Physical communication pathways in bacteria: an extra layer to quorum sensing" | Biophys Rev Lett | 2025 | All microbial comm modes beyond QS (mechanical, EM, acoustic). Can Anderson apply to these? | **Exp152 DONE** (9/9) |
| 31 | "Diverse QS systems regulate microbial communication in deep-sea cold seeps" | Microbiome | 2025 | **299,355 QS genes, 170 metagenomes, 34 QS types**. Massive dataset to test Anderson predictions in 3D sediment | **Exp144 DONE** (8/8), **Exp145 DONE** (5/5) |
| 32 | "In silico protein analysis, ecophysiology, and reconstruction of evolutionary history of QS" | BMC Genomics | 2024 | Phylogenetic reconstruction of luxR. Correlate QS gene gain/loss with habitat geometry transitions | **Exp146 DONE** (5/5) |
| 33 | "Spatially propagating activation of QS in V. fischeri" — Meyer et al. | Phys Rev E 101:062421 | 2020 | Closest to our physics approach (traveling waves). Complementary: propagation vs localization | **Exp148 DONE** (6/6) |

#### Tier 2 — Validates or Challenges Predictions

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| 34 | "Burst statistics in biofilm QS: role of spatial colony-growth heterogeneity" | Sci Rep | 2019 | Spatial disorder effects on QS timing. Their "disordered colony" ≈ our Anderson disorder | **Exp149 DONE** (6/6) |
| 35 | "Functional metagenomic analysis of QS in a nitrifying community" | npj Biofilms | 2021 | 13 luxI + 30 luxR from sludge. R:P = 2.3:1. Test eavesdropper prediction | **Exp153 DONE** (12/12) |
| 36 | "A review of QS mediating interkingdom interactions in the ocean" | Commun Biol | 2025 | Marine QS review. Refine our "obligate plankton = no QS" prediction | **Exp154 DONE** (6/6) |

#### Tier 3 — Experimental Validation Targets

| # | Paper | Journal | Year | Why | Status |
|---|-------|---------|------|-----|--------|
| 37 | Rajagopalan et al. "Cell density, alignment, and orientation correlate with C-signal expression" | PNAS | 2021 | Myxococcus C-signal → 3D. Extract critical cell density for Anderson L_min prediction | **Exp155 DONE** (7/7) |
| 38 | "Integrated cross-regulation pathway for cAMP relay in Dictyostelium" | Front Cell Dev Biol | 2023 | Updated relay circuit. Can we model relay as non-Hermitian Anderson? | **Exp156 DONE** (8/8) |

---

## Three-Tier Control Summary

| Track | Papers | CPU | GPU | metalForge | Status |
|-------|:------:|:---:|:---:|:----------:|--------|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 | Full three-tier |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 | Full three-tier |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 | Full three-tier |
| Track 3 (Drug repurposing) | 5 | 5/5 | 5/5 | 5/5 | Full three-tier (Exp163-165) |
| Track 4 (Soil QS/Anderson) | 9 | 9/9 | 9/9 | 9/9 | Full three-tier (Exp170-182) |
| **Subtotal (three-tier)** | **39** | **39/39** | **39/39** | **39/39** | **ALL three-tier** |
| Cross-spring (spectral) | 1 | 1/1 | 1/1 | — | CPU + GPU |
| Extensions (Phase 37-39) | 9 | 9/9 | — | — | CPU only (by design — analytical/catalog) |
| Track 5 immuno-Anderson (science) | 6 | 1/1 | 4/4 | 1/1 | Full three-tier (Exp273-279: 157/157) |
| Track 5 Gonzales reproductions | 6 | 1/1 | 1/1 | 1/1 | Full three-tier (Exp280-286: 202/202) |
| **Grand total** | **52 + 6 reproduced** | **54/54** | **48/48** | **41/41** | |

**All GPU primitives upstream:** NMF (S58), TransE (S60), SpMM (S60), PeakDetect (S62), BGL helpers (S62+DF64).
**All 39 three-tier-eligible papers now have full three-tier validation** (CPU, GPU, metalForge).
Track 3 completed via Exp163 (CPU v9), Exp164 (GPU drug repurposing), Exp165 (metalForge).
Track 4 completed via Exp170-178 (CPU baselines — Anderson-QS soil pore geometry, no-till
meta-analysis, long-term tillage factorial, biofilm aggregate, structure→function, tillage microbiomes).
Modern systems validated via Exp166 (S62+DF64 benchmark, 19 checks).
Diversity fusion WGSL extension validated via Exp167 (CPU ↔ GPU parity, 18 checks).
Extension papers are analytical models — GPU acceleration is not the bottleneck.
Track 4 now has full three-tier: CPU baseline (Exp170-178), CPU parity (Exp179),
GPU validation (Exp180), pure GPU streaming (Exp181), metalForge cross-substrate (Exp182).
Track 5 Gonzales reproductions completed V92: science (Exp280-282: 69/69) +
three-tier (Exp283-286: 133/133) = 202/202 checks. Papers 53-58 now reproduced from
published data (IC50 dose-response, PK decay, pruritus time-series, cell populations).

---

## Hardware Control Chain

Every paper reproduction is validated through a progressive hardware control chain.
Each tier's output is verified against the previous tier to prove substrate independence.

```
Open Data (NCBI SRA, Zenodo, EPA, journal tables, published equations)
  → barracuda CPU:  Pure Rust math matches Python/SciPy baselines (f64 exact or justified tolerance)
    → barracuda GPU: GPU dispatch matches CPU reference (WGSL f64 via ToadStool S79)
      → metalForge:  Mixed-hardware routing produces identical results (CPU ↔ GPU ↔ NPU)
```

**Verification**: 58 papers, 7,384+ checks (incl. Exp291-295: 164 new), 1,309 tests. Zero proprietary data dependencies.
All Python baselines have reproduction headers (script, commit, date, hardware, SHA-256).
All Rust validators have provenance classification headers.
All 103 tolerance constants are scientifically justified and hierarchy-tested.

---

## Notes

- wetSpring has the largest paper queue because it spans four scientific domains + cross-spring
- Track 1 papers (Waters) are the most immediately reproducible — fully specified ODE/stochastic models
- Track 1b papers (Liu) bridge to neuralSpring via HMM ↔ LSTM isomorphism
- Track 2 papers (Jones) depend on accessing PFAS mass spec data
- Track 3 papers (Fajgenbaum) use publicly available repoDB + published algorithms
- Cahill/Smallwood papers may require Sandia data access agreements
- Paper 23 (Kachkovskiy) is now validated via cross-spring spectral primitives (Exp107: 25/25 checks)
- Extension papers (Phase 37-39) are CPU-only by design — they validate analytical models where GPU is not the bottleneck
- V92D: all library code is panic-free, all clippy pedantic warnings resolved under `--all-features`
