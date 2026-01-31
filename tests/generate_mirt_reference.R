#!/usr/bin/env Rscript
# Generate reference values from mirt for Python validation
# Run: Rscript tests/generate_mirt_reference.R

library(mirt)
set.seed(42)

output_dir <- "tests/mirt_reference"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Generating mirt reference values...\n")

# ============================================================
# 1. LSAT7 Dataset (built-in mirt dataset)
# ============================================================
cat("\n1. Processing LSAT7 dataset...\n")

data <- expand.table(LSAT7)
write.csv(data, file.path(output_dir, "lsat7_data.csv"), row.names = FALSE)

# Rasch model
mod_rasch <- mirt(data, 1, itemtype = "Rasch", verbose = FALSE)
rasch_pars <- coef(mod_rasch, simplify = TRUE, IRTpars = TRUE)$items

write.csv(
    data.frame(
        item = rownames(rasch_pars),
        a = rasch_pars[, "a"],
        b = rasch_pars[, "b"]
    ),
    file.path(output_dir, "lsat7_rasch_params.csv"),
    row.names = FALSE
)

cat(extract.mirt(mod_rasch, "logLik"),
    file = file.path(output_dir, "lsat7_rasch_loglik.txt"))

eap_rasch <- fscores(mod_rasch, method = "EAP", full.scores.SE = TRUE)
write.csv(
    data.frame(theta = eap_rasch[, "F1"], se = eap_rasch[, "SE_F1"]),
    file.path(output_dir, "lsat7_rasch_eap.csv"),
    row.names = FALSE
)

map_rasch <- fscores(mod_rasch, method = "MAP", full.scores.SE = TRUE)
write.csv(
    data.frame(theta = map_rasch[, "F1"], se = map_rasch[, "SE_F1"]),
    file.path(output_dir, "lsat7_rasch_map.csv"),
    row.names = FALSE
)

# 2PL model
mod_2pl <- mirt(data, 1, verbose = FALSE)
twopl_pars <- coef(mod_2pl, simplify = TRUE, IRTpars = TRUE)$items

write.csv(
    data.frame(
        item = rownames(twopl_pars),
        a = twopl_pars[, "a"],
        b = twopl_pars[, "b"]
    ),
    file.path(output_dir, "lsat7_2pl_params.csv"),
    row.names = FALSE
)

cat(extract.mirt(mod_2pl, "logLik"),
    file = file.path(output_dir, "lsat7_2pl_loglik.txt"))

eap_2pl <- fscores(mod_2pl, method = "EAP", full.scores.SE = TRUE)
write.csv(
    data.frame(theta = eap_2pl[, "F1"], se = eap_2pl[, "SE_F1"]),
    file.path(output_dir, "lsat7_2pl_eap.csv"),
    row.names = FALSE
)

cat("  Rasch loglik:", extract.mirt(mod_rasch, "logLik"), "\n")
cat("  2PL loglik:", extract.mirt(mod_2pl, "logLik"), "\n")

# ============================================================
# 2. Simulated Data with Known Parameters
# ============================================================
cat("\n2. Generating simulated test data...\n")

N <- 1000
J <- 20

# Rasch data (a=1 for all items)
a_rasch <- matrix(1, J, 1)
b_true <- seq(-2, 2, length.out = J)
d_rasch <- matrix(-b_true, J, 1)  # mirt uses d = -a*b

dat_rasch <- simdata(a_rasch, d_rasch, N, itemtype = "2PL")
write.csv(dat_rasch, file.path(output_dir, "sim_rasch_data.csv"), row.names = FALSE)
write.csv(data.frame(b = b_true), file.path(output_dir, "sim_rasch_true_b.csv"), row.names = FALSE)

# Fit and save estimates
mod_sim_rasch <- mirt(dat_rasch, 1, itemtype = "Rasch", verbose = FALSE)
sim_rasch_pars <- coef(mod_sim_rasch, simplify = TRUE, IRTpars = TRUE)$items
write.csv(
    data.frame(item = rownames(sim_rasch_pars), b = sim_rasch_pars[, "b"]),
    file.path(output_dir, "sim_rasch_mirt_params.csv"),
    row.names = FALSE
)

# 2PL data
a_true_2pl <- exp(rnorm(J, mean = 0.2, sd = 0.3))
d_2pl <- matrix(-a_true_2pl * b_true, J, 1)

dat_2pl <- simdata(matrix(a_true_2pl, J, 1), d_2pl, N, itemtype = "2PL")
write.csv(dat_2pl, file.path(output_dir, "sim_2pl_data.csv"), row.names = FALSE)
write.csv(
    data.frame(a = a_true_2pl, b = b_true),
    file.path(output_dir, "sim_2pl_true_params.csv"),
    row.names = FALSE
)

# Fit and save estimates
mod_sim_2pl <- mirt(dat_2pl, 1, verbose = FALSE)
sim_2pl_pars <- coef(mod_sim_2pl, simplify = TRUE, IRTpars = TRUE)$items
write.csv(
    data.frame(item = rownames(sim_2pl_pars), a = sim_2pl_pars[, "a"], b = sim_2pl_pars[, "b"]),
    file.path(output_dir, "sim_2pl_mirt_params.csv"),
    row.names = FALSE
)

# EAP scores for simulated data
eap_sim <- fscores(mod_sim_rasch, method = "EAP", full.scores.SE = TRUE)
write.csv(
    data.frame(theta = eap_sim[, "F1"], se = eap_sim[, "SE_F1"]),
    file.path(output_dir, "sim_rasch_eap.csv"),
    row.names = FALSE
)

# MAP scores for simulated data
map_sim <- fscores(mod_sim_rasch, method = "MAP", full.scores.SE = TRUE)
write.csv(
    data.frame(theta = map_sim[, "F1"], se = map_sim[, "SE_F1"]),
    file.path(output_dir, "sim_rasch_map.csv"),
    row.names = FALSE
)

cat("  Simulated data saved.\n")

# ============================================================
# 3. Technical Settings Comparison
# ============================================================
cat("\n3. Testing technical settings...\n")

# Compare different quadrature points
mod_q21 <- mirt(data, 1, itemtype = "Rasch", quadpts = 21, verbose = FALSE)
mod_q61 <- mirt(data, 1, itemtype = "Rasch", quadpts = 61, verbose = FALSE)
mod_q121 <- mirt(data, 1, itemtype = "Rasch", quadpts = 121, verbose = FALSE)

quadpts_comparison <- data.frame(
    quadpts = c(21, 61, 121),
    loglik = c(
        extract.mirt(mod_q21, "logLik"),
        extract.mirt(mod_q61, "logLik"),
        extract.mirt(mod_q121, "logLik")
    )
)
write.csv(quadpts_comparison, file.path(output_dir, "quadpts_comparison.csv"), row.names = FALSE)
cat("  Quadpts 21:", extract.mirt(mod_q21, "logLik"), "\n")
cat("  Quadpts 61:", extract.mirt(mod_q61, "logLik"), "\n")
cat("  Quadpts 121:", extract.mirt(mod_q121, "logLik"), "\n")

# ============================================================
# Summary
# ============================================================
cat("\n============================================================\n")
cat("Reference values generated in:", output_dir, "\n")
cat("Files created:\n")
cat(paste("  -", list.files(output_dir), collapse = "\n"), "\n")
cat("============================================================\n")
