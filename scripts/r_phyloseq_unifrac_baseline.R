#!/usr/bin/env Rscript
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-03-10
# Commit: wetSpring V106
#
# R/phyloseq Industry Baseline — UniFrac & Ordination
#
# Computes weighted/unweighted UniFrac distances and PCoA ordination
# using phyloseq (McMurdie & Holmes 2013) and ape (Paradis et al.),
# the standard tools for phylogenetic beta diversity in microbial
# ecology. Validates wetSpring's bio::unifrac and bio::pcoa modules.
#
# Reproduction:
#   Rscript scripts/r_phyloseq_unifrac_baseline.R
#
# Output:
#   experiments/results/r_baselines/phyloseq_unifrac.json
#
# References:
# - McMurdie & Holmes (2013), PLoS ONE 8:e61217 (phyloseq)
# - Lozupone & Knight (2005), Appl Environ Microbiol 71:8228 (UniFrac)
# - Paradis et al. (2004), Bioinformatics 20:289-290 (ape)

suppressPackageStartupMessages({
  library(phyloseq)
  library(ape)
  library(jsonlite)
})

cat("═══════════════════════════════════════════════════════════════\n")
cat("  R/phyloseq Industry Baseline — UniFrac & Ordination\n")
cat("  phyloseq", as.character(packageVersion("phyloseq")), "\n")
cat("  ape", as.character(packageVersion("ape")), "\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

results <- list()

# ── Build a small phylogenetic tree and OTU table ─────────────────
# 4 OTUs, 3 samples, with a strictly BIFURCATING tree:
#
#          root
#         /    \
#       (0.3) (0.6)
#       /        \
#     AB          CD
#    / \         / \
# (0.1)(0.2) (0.4)(0.5)
#   A    B     C    D
#
# Using a bifurcating tree avoids phyloseq's node.desc matrix bug
# with trifurcations (phyloseq assumes ncol=2 for internal node
# children, silently dropping the 3rd child via R matrix recycling).
#
# Using Newick: ((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);

tree_str <- "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
tree <- read.tree(text = tree_str)

cat("§1 Tree structure:\n")
cat("   Tips:", paste(tree$tip.label, collapse = ", "), "\n")
cat("   Edge count:", nrow(tree$edge), "\n")
cat("   Is rooted:", is.rooted(tree), "\n\n")

# OTU abundance table (3 samples × 4 OTUs)
otu_mat <- matrix(
  c(100, 50, 30, 10,     # Sample1: dominated by A
    10, 10, 50, 80,      # Sample2: dominated by C,D
    50, 50, 50, 50),     # Sample3: perfectly even
  nrow = 3, byrow = TRUE,
  dimnames = list(
    c("Sample1", "Sample2", "Sample3"),
    c("A", "B", "C", "D")
  )
)

otu_tab <- otu_table(otu_mat, taxa_are_rows = FALSE)
ps <- phyloseq(otu_tab, phy_tree(tree))

# ── §2 Unweighted UniFrac ────────────────────────────────────────
uf_unw <- as.matrix(UniFrac(ps, weighted = FALSE))
cat("§2 Unweighted UniFrac matrix:\n")
print(round(uf_unw, 10))
cat("\n")

results$unifrac_unweighted <- list(
  s1_s2 = uf_unw["Sample1", "Sample2"],
  s1_s3 = uf_unw["Sample1", "Sample3"],
  s2_s3 = uf_unw["Sample2", "Sample3"]
)

# Verify properties
cat(sprintf("§2 UF(S1,S2)=%.15f\n", uf_unw["Sample1", "Sample2"]))
cat(sprintf("§2 UF(S1,S3)=%.15f\n", uf_unw["Sample1", "Sample3"]))
cat(sprintf("§2 Self-distance UF(S1,S1)=%.15f (must be 0)\n",
            uf_unw["Sample1", "Sample1"]))
cat(sprintf("§2 Symmetric: UF(S1,S2)==UF(S2,S1)? %s\n",
            uf_unw["Sample1", "Sample2"] == uf_unw["Sample2", "Sample1"]))

# ── §3 Weighted UniFrac ──────────────────────────────────────────
uf_w <- as.matrix(UniFrac(ps, weighted = TRUE))
cat("\n§3 Weighted UniFrac matrix:\n")
print(round(uf_w, 10))
cat("\n")

results$unifrac_weighted <- list(
  s1_s2 = uf_w["Sample1", "Sample2"],
  s1_s3 = uf_w["Sample1", "Sample3"],
  s2_s3 = uf_w["Sample2", "Sample3"]
)

cat(sprintf("§3 WUF(S1,S2)=%.15f\n", uf_w["Sample1", "Sample2"]))
cat(sprintf("§3 WUF(S1,S3)=%.15f\n", uf_w["Sample1", "Sample3"]))
cat(sprintf("§3 WUF(S2,S3)=%.15f\n", uf_w["Sample2", "Sample3"]))

# ── §4 PCoA on UniFrac distances ─────────────────────────────────
pcoa_uf <- ordinate(ps, method = "PCoA", distance = "unifrac",
                    weighted = TRUE)
cat("\n§4 PCoA on weighted UniFrac:\n")
cat("   Eigenvalues:", round(pcoa_uf$values$Eigenvalues[1:2], 10), "\n")
cat("   Relative eigenvalues:",
    round(pcoa_uf$values$Relative_eig[1:2], 10), "\n")
cat("   Coordinates:\n")
print(round(pcoa_uf$vectors[, 1:2], 10))

results$pcoa_eigenvalues <- pcoa_uf$values$Eigenvalues[1:2]
results$pcoa_relative_eigenvalues <- pcoa_uf$values$Relative_eig[1:2]
results$pcoa_axis1 <- pcoa_uf$vectors[, 1]
results$pcoa_axis2 <- pcoa_uf$vectors[, 2]

# ── §5 Tree distances (cophenetic) ───────────────────────────────
cophen <- cophenetic(tree)
cat("\n§5 Cophenetic (patristic) distances:\n")
print(round(cophen, 10))

results$cophenetic_ab <- cophen["A", "B"]
results$cophenetic_ac <- cophen["A", "C"]
results$cophenetic_cd <- cophen["C", "D"]

cat(sprintf("\n§5 d(A,B)=%.15f (same clade: 0.1+0.2=0.3)\n",
            cophen["A", "B"]))
cat(sprintf("§5 d(A,C)=%.15f (cross-clade: 0.1+0.3+0.6+0.4=1.4)\n",
            cophen["A", "C"]))
cat(sprintf("§5 d(C,D)=%.15f (same clade: 0.4+0.5=0.9)\n",
            cophen["C", "D"]))

# ── §6 Tree input data for Rust validator ─────────────────────────
results$tree_newick <- tree_str
results$otu_table <- as.list(as.data.frame(otu_mat))
results$sample_names <- rownames(otu_mat)
results$otu_names <- colnames(otu_mat)

# ── Write output ──────────────────────────────────────────────────
results$metadata <- list(
  phyloseq_version = as.character(packageVersion("phyloseq")),
  ape_version      = as.character(packageVersion("ape")),
  r_version        = paste(R.version$major, R.version$minor, sep = "."),
  date             = format(Sys.time(), "%Y-%m-%d"),
  platform         = R.version$platform,
  command          = "Rscript scripts/r_phyloseq_unifrac_baseline.R"
)

outdir <- file.path("experiments", "results", "r_baselines")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
outpath <- file.path(outdir, "phyloseq_unifrac.json")
write_json(results, outpath, pretty = TRUE, auto_unbox = TRUE, digits = 15)
cat(sprintf("\n✓ Results written to %s\n", outpath))
