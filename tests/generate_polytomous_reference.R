#!/usr/bin/env Rscript
# Generate polytomous reference values from R mirt for Python validation
# Run: Rscript tests/generate_polytomous_reference.R  (or: make refs)
#
# Uses Science dataset: 4 items, 4 categories (1-4 in R; Python uses 0-3)
# Data: tests/science_data.csv (exported from ltm package)
# Requires: R with mirt installed

set.seed(42)

ltm_dir <- "tests/ltm_reference"
mirt_dir <- "tests/mirt_reference"
dir.create(ltm_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(mirt_dir, showWarnings = FALSE, recursive = TRUE)

# Load Science data (numeric 1-4)
science_csv <- "tests/science_data.csv"
if (!file.exists(science_csv)) {
  stop("Run: Rscript -e \"load('Science.rda'); write.csv(as.matrix(sapply(Science[c(1,3,4,7)], as.numeric)), 'tests/science_data.csv', row.names=F)\" from ltm package")
}
Science_mat <- as.matrix(read.csv(science_csv, check.names = FALSE))

cat("Generating polytomous reference values...\n")
cat("Science: ", nrow(Science_mat), "persons x", ncol(Science_mat), "items\n")
cat("Categories: ", paste(sort(unique(c(Science_mat))), collapse = ", "), "\n")

# Save data to both dirs
write.csv(Science_mat, file.path(ltm_dir, "science_data.csv"), row.names = FALSE)
write.csv(Science_mat, file.path(mirt_dir, "science_data.csv"), row.names = FALSE)

# ============================================================
# mirt: graded (GRM), gpcm, Rasch (PCM), rsm, nominal (NRM)
# ============================================================
library(mirt)

cat("\n--- mirt models ---\n")

# GRM (graded)
cat("Fitting mirt GRM (graded)...\n")
mod_mirt_grm <- mirt(Science_mat, 1, itemtype = "graded", verbose = FALSE)
pars_grm <- coef(mod_mirt_grm, simplify = TRUE, IRTpars = TRUE)$items
grm_df <- data.frame(
  item = seq_len(nrow(pars_grm)),
  a = pars_grm[, "a"],
  b1 = pars_grm[, "b1"],
  b2 = pars_grm[, "b2"],
  b3 = pars_grm[, "b3"]
)
write.csv(grm_df, file.path(ltm_dir, "science_grm_params.csv"), row.names = FALSE)
write.csv(grm_df, file.path(mirt_dir, "science_grm_params.csv"), row.names = FALSE)
cat(extract.mirt(mod_mirt_grm, "logLik"), file = file.path(ltm_dir, "science_grm_loglik.txt"))
cat(extract.mirt(mod_mirt_grm, "logLik"), file = file.path(mirt_dir, "science_grm_loglik.txt"))
cat("  GRM loglik:", extract.mirt(mod_mirt_grm, "logLik"), "\n")

# GPCM
cat("Fitting mirt GPCM...\n")
mod_mirt_gpcm <- mirt(Science_mat, 1, itemtype = "gpcm", verbose = FALSE)
pars_gpcm <- coef(mod_mirt_gpcm, simplify = TRUE, IRTpars = TRUE)$items
gpcm_df <- data.frame(
  item = seq_len(nrow(pars_gpcm)),
  a = pars_gpcm[, "a"],
  b1 = pars_gpcm[, "b1"],
  b2 = pars_gpcm[, "b2"],
  b3 = pars_gpcm[, "b3"]
)
write.csv(gpcm_df, file.path(ltm_dir, "science_gpcm_params.csv"), row.names = FALSE)
write.csv(gpcm_df, file.path(mirt_dir, "science_gpcm_params.csv"), row.names = FALSE)
cat(extract.mirt(mod_mirt_gpcm, "logLik"), file = file.path(ltm_dir, "science_gpcm_loglik.txt"))
cat(extract.mirt(mod_mirt_gpcm, "logLik"), file = file.path(mirt_dir, "science_gpcm_loglik.txt"))
cat("  GPCM loglik:", extract.mirt(mod_mirt_gpcm, "logLik"), "\n")

# PCM (Rasch for polytomous)
cat("Fitting mirt PCM (Rasch)...\n")
mod_mirt_pcm <- mirt(Science_mat, 1, itemtype = "Rasch", verbose = FALSE)
pars_pcm <- coef(mod_mirt_pcm, simplify = TRUE, IRTpars = TRUE)$items
pcm_df <- data.frame(
  item = seq_len(nrow(pars_pcm)),
  b1 = pars_pcm[, "b1"],
  b2 = pars_pcm[, "b2"],
  b3 = pars_pcm[, "b3"]
)
write.csv(pcm_df, file.path(ltm_dir, "science_pcm_params.csv"), row.names = FALSE)
write.csv(pcm_df, file.path(mirt_dir, "science_pcm_params.csv"), row.names = FALSE)
cat(extract.mirt(mod_mirt_pcm, "logLik"), file = file.path(ltm_dir, "science_pcm_loglik.txt"))
cat(extract.mirt(mod_mirt_pcm, "logLik"), file = file.path(mirt_dir, "science_pcm_loglik.txt"))
cat("  PCM loglik:", extract.mirt(mod_mirt_pcm, "logLik"), "\n")

# RSM (mirt uses a1, b1, b2, b3, c)
cat("Fitting mirt RSM...\n")
mod_mirt_rsm <- mirt(Science_mat, 1, itemtype = "rsm", verbose = FALSE)
pars_rsm <- coef(mod_mirt_rsm, simplify = TRUE, IRTpars = TRUE)$items
write.csv(pars_rsm, file.path(mirt_dir, "science_rsm_params.csv"), row.names = TRUE)
cat(extract.mirt(mod_mirt_rsm, "logLik"), file = file.path(mirt_dir, "science_rsm_loglik.txt"))
cat("  RSM loglik:", extract.mirt(mod_mirt_rsm, "logLik"), "\n")

# NRM (nominal)
cat("Fitting mirt NRM (nominal)...\n")
mod_mirt_nrm <- mirt(Science_mat, 1, itemtype = "nominal", verbose = FALSE)
pars_nrm <- coef(mod_mirt_nrm, simplify = TRUE, IRTpars = TRUE)$items
write.csv(pars_nrm, file.path(mirt_dir, "science_nrm_params.csv"), row.names = TRUE)
cat(extract.mirt(mod_mirt_nrm, "logLik"), file = file.path(mirt_dir, "science_nrm_loglik.txt"))
cat("  NRM loglik:", extract.mirt(mod_mirt_nrm, "logLik"), "\n")

# EAP scores (GRM) - use mirt fscores
cat("\nSaving EAP scores (mirt GRM)...\n")
eap_grm <- fscores(mod_mirt_grm, method = "EAP", full.scores.SE = TRUE)
write.csv(
  data.frame(theta = eap_grm[, "F1"], se = eap_grm[, "SE_F1"]),
  file.path(ltm_dir, "science_grm_eap.csv"),
  row.names = FALSE
)

cat("\n============================================================\n")
cat("Polytomous reference values generated.\n")
cat("ltm files:", paste(list.files(ltm_dir, pattern = "science"), collapse = ", "), "\n")
cat("mirt files:", paste(list.files(mirt_dir, pattern = "science"), collapse = ", "), "\n")
cat("============================================================\n")
