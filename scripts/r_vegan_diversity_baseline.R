#!/usr/bin/env Rscript
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-03-10
# Commit: wetSpring V106
#
# R/vegan Industry Baseline — Diversity Metrics
#
# Computes Shannon, Simpson, Bray-Curtis, rarefaction, and PCoA
# using the R vegan package (Oksanen et al.), the *de facto* standard
# in microbial ecology. Emits JSON for Rust validator consumption.
#
# vegan is what QIIME2's `q2-diversity` wraps for alpha/beta diversity.
# This validates wetSpring's sovereign Rust implementations against the
# canonical reference.
#
# Reproduction:
#   Rscript scripts/r_vegan_diversity_baseline.R
#
# Output:
#   experiments/results/r_baselines/vegan_diversity.json
#
# References:
# - Oksanen et al., "vegan: Community Ecology Package" (CRAN)
# - Shannon (1948), Bell System Technical Journal 27:379-423
# - Simpson (1949), Nature 163:688
# - Bray & Curtis (1957), Ecological Monographs 27:325-349

suppressPackageStartupMessages({
  library(vegan)
  library(jsonlite)
})

cat("═══════════════════════════════════════════════════════════════\n")
cat("  R/vegan Industry Baseline — Diversity Metrics\n")
cat("  vegan", as.character(packageVersion("vegan")), "\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

results <- list()

# ── §1 Shannon & Simpson on uniform community ────────────────────
# Analytical: H'(uniform, S=10) = ln(10) ≈ 2.302585...
# Simpson D(uniform, S=10) = 1 - 1/10 = 0.9
uniform_10 <- rep(100, 10)
h_uniform  <- diversity(uniform_10, index = "shannon")
d_uniform  <- diversity(uniform_10, index = "simpson")

cat(sprintf("§1 Shannon(uniform,S=10): %.15f  (analytical: %.15f)\n",
            h_uniform, log(10)))
cat(sprintf("§1 Simpson(uniform,S=10): %.15f  (analytical: %.15f)\n",
            d_uniform, 1 - 1/10))

results$shannon_uniform_10 <- h_uniform
results$simpson_uniform_10 <- d_uniform

# ── §2 Shannon & Simpson on skewed community ─────────────────────
# Dominant species (p=0.9) + 9 rare species (p=0.011... each)
skewed <- c(900, rep(11, 9))
h_skewed <- diversity(skewed, index = "shannon")
d_skewed <- diversity(skewed, index = "simpson")

cat(sprintf("§2 Shannon(skewed): %.15f\n", h_skewed))
cat(sprintf("§2 Simpson(skewed): %.15f\n", d_skewed))

results$shannon_skewed <- h_skewed
results$simpson_skewed <- d_skewed

# ── §3 Bray-Curtis distance ──────────────────────────────────────
comm_a <- c(10, 20, 30, 40, 50)
comm_b <- c(50, 40, 30, 20, 10)
comm_c <- c(10, 20, 30, 40, 50)  # identical to a

mat <- rbind(comm_a, comm_b, comm_c)
bc_dist <- as.matrix(vegdist(mat, method = "bray"))

cat(sprintf("§3 BC(a,b): %.15f\n", bc_dist["comm_a", "comm_b"]))
cat(sprintf("§3 BC(a,c): %.15f  (self-distance, should be 0)\n",
            bc_dist["comm_a", "comm_c"]))
cat(sprintf("§3 BC(a,b)==BC(b,a): %s (symmetry)\n",
            bc_dist["comm_a", "comm_b"] == bc_dist["comm_b", "comm_a"]))

results$bray_curtis_ab <- bc_dist["comm_a", "comm_b"]
results$bray_curtis_ac <- bc_dist["comm_a", "comm_c"]
results$bray_curtis_symmetric <- bc_dist["comm_a", "comm_b"] == bc_dist["comm_b", "comm_a"]

# ── §4 Rarefaction (species richness estimation) ─────────────────
# vegan::rarefy computes expected richness at given sample sizes
# Analytical: for uniform community of S species, E[S_n] = S*(1 - choose(N-n, S) / choose(N, S))
rich_comm <- c(50, 40, 30, 20, 10, 5, 3, 2, 1, 1)
rare_at <- c(10, 50, 100, 150)
rarefied <- rarefy(rich_comm, sample = rare_at)

cat("§4 Rarefaction curve:\n")
for (i in seq_along(rare_at)) {
  cat(sprintf("   n=%d: E[S]=%.10f\n", rare_at[i], rarefied[i]))
}

results$rarefaction_depths <- rare_at
results$rarefaction_expected_richness <- as.numeric(rarefied)

# verify monotonicity (rarefaction must be non-decreasing)
is_monotonic <- all(diff(rarefied) >= 0)
cat(sprintf("§4 Rarefaction monotonic: %s\n", is_monotonic))
results$rarefaction_monotonic <- is_monotonic

# ── §5 Shannon on 5-species known community (wetSpring Exp002 proxy) ──
# This matches the test vectors in bio::diversity unit tests
exp002_proxy <- c(100, 80, 60, 40, 20)
h_exp002 <- diversity(exp002_proxy, index = "shannon")
d_exp002 <- diversity(exp002_proxy, index = "simpson")

cat(sprintf("§5 Shannon(exp002_proxy): %.15f\n", h_exp002))
cat(sprintf("§5 Simpson(exp002_proxy): %.15f\n", d_exp002))

results$shannon_5species <- h_exp002
results$simpson_5species <- d_exp002

# ── §6 Jaccard distance ──────────────────────────────────────────
jac_dist <- as.matrix(vegdist(mat, method = "jaccard", binary = TRUE))
cat(sprintf("§6 Jaccard(a,b): %.15f\n", jac_dist["comm_a", "comm_b"]))
results$jaccard_ab <- jac_dist["comm_a", "comm_b"]

# ── §7 PCoA (principal coordinates analysis) ─────────────────────
# Using the 3-sample Bray-Curtis matrix from §3
pcoa_result <- cmdscale(vegdist(mat, method = "bray"), k = 2, eig = TRUE)
cat("§7 PCoA eigenvalues:", pcoa_result$eig[1:2], "\n")
cat("§7 PCoA coordinates:\n")
print(pcoa_result$points)

results$pcoa_eigenvalues <- pcoa_result$eig[1:min(length(pcoa_result$eig), 3)]
results$pcoa_coordinates <- as.list(as.data.frame(pcoa_result$points))

# ── §8 Chao1 richness estimator ─────────────────────────────────
chao1_comm <- c(50, 40, 30, 20, 10, 1, 1, 1, 1, 1)  # 5 singletons
chao1_est <- estimateR(chao1_comm)
cat(sprintf("§8 Chao1: S.obs=%g, S.chao1=%g\n",
            chao1_est["S.obs"], chao1_est["S.chao1"]))

results$chao1_s_obs <- as.numeric(chao1_est["S.obs"])
results$chao1_estimate <- as.numeric(chao1_est["S.chao1"])

# ── §9 Pielou evenness ──────────────────────────────────────────
# J = H / ln(S)
S <- length(uniform_10)
J_uniform <- h_uniform / log(S)
cat(sprintf("§9 Pielou J(uniform): %.15f (should be 1.0)\n", J_uniform))
results$pielou_uniform <- J_uniform

S_skewed <- length(skewed)
J_skewed <- h_skewed / log(S_skewed)
cat(sprintf("§9 Pielou J(skewed): %.15f\n", J_skewed))
results$pielou_skewed <- J_skewed

# ── Write output ──────────────────────────────────────────────────
results$metadata <- list(
  tool       = "vegan",
  version    = as.character(packageVersion("vegan")),
  r_version  = paste(R.version$major, R.version$minor, sep = "."),
  date       = format(Sys.time(), "%Y-%m-%d"),
  platform   = R.version$platform,
  command    = "Rscript scripts/r_vegan_diversity_baseline.R"
)

outdir <- file.path("experiments", "results", "r_baselines")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
outpath <- file.path(outdir, "vegan_diversity.json")
write_json(results, outpath, pretty = TRUE, auto_unbox = TRUE, digits = 15)
cat(sprintf("\n✓ Results written to %s\n", outpath))
