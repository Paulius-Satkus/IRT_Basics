#!/usr/bin/env Rscript
# Generate reference values from ltm for Python validation
# Run: Rscript tests/generate_ltm_reference.R

library(ltm)
set.seed(42)

output_dir <- "tests/ltm_reference"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Generating ltm reference values...\n")

# ============================================================
# 1. LSAT Dataset (built-in)
# ============================================================
cat("\n1. Processing LSAT dataset...\n")

data(LSAT)
write.csv(LSAT, file.path(output_dir, "lsat_data.csv"), row.names = FALSE)

cat("  LSAT dimensions:", nrow(LSAT), "x", ncol(LSAT), "\n")

# ============================================================
# 1a. Rasch model with discrimination FIXED at 1
# ============================================================
cat("\n1a. Fitting Rasch model (discrimination = 1)...\n")

mod_rasch_fixed <- rasch(LSAT, constraint = cbind(ncol(LSAT) + 1, 1))
rasch_coef <- coef(mod_rasch_fixed)

write.csv(
    data.frame(
        item = rownames(rasch_coef),
        b = rasch_coef[, "Dffclt"],
        a = rasch_coef[, "Dscrmn"]
    ),
    file.path(output_dir, "lsat_rasch_fixed_params.csv"),
    row.names = FALSE
)

cat(mod_rasch_fixed$log.Lik,
    file = file.path(output_dir, "lsat_rasch_fixed_loglik.txt"))

eap_scores <- factor.scores(mod_rasch_fixed, resp.patterns = LSAT, method = "EAP")
write.csv(
    data.frame(
        theta = eap_scores$score.dat$z1,
        se = eap_scores$score.dat$se.z1,
        obs = eap_scores$score.dat$Obs
    ),
    file.path(output_dir, "lsat_rasch_fixed_eap.csv"),
    row.names = FALSE
)

eb_scores <- factor.scores(mod_rasch_fixed, resp.patterns = LSAT, method = "EB")
write.csv(
    data.frame(
        theta = eb_scores$score.dat$z1,
        se = eb_scores$score.dat$se.z1,
        obs = eb_scores$score.dat$Obs
    ),
    file.path(output_dir, "lsat_rasch_fixed_eb.csv"),
    row.names = FALSE
)

cat("  Rasch (fixed) loglik:", mod_rasch_fixed$log.Lik, "\n")

# ============================================================
# 1b. Rasch model with discrimination ESTIMATED
# ============================================================
cat("\n1b. Fitting Rasch model (discrimination estimated)...\n")

mod_rasch <- rasch(LSAT)
rasch_est_coef <- coef(mod_rasch)

write.csv(
    data.frame(
        item = rownames(rasch_est_coef),
        b = rasch_est_coef[, "Dffclt"],
        a = rasch_est_coef[, "Dscrmn"]
    ),
    file.path(output_dir, "lsat_rasch_estimated_params.csv"),
    row.names = FALSE
)

cat(mod_rasch$log.Lik,
    file = file.path(output_dir, "lsat_rasch_estimated_loglik.txt"))

eap_est <- factor.scores(mod_rasch, resp.patterns = LSAT, method = "EAP")
write.csv(
    data.frame(
        theta = eap_est$score.dat$z1,
        se = eap_est$score.dat$se.z1,
        obs = eap_est$score.dat$Obs
    ),
    file.path(output_dir, "lsat_rasch_estimated_eap.csv"),
    row.names = FALSE
)

cat("  Rasch (estimated) loglik:", mod_rasch$log.Lik, "\n")
cat("  Estimated discrimination:", unique(rasch_est_coef[, "Dscrmn"]), "\n")

# ============================================================
# 1c. 2PL model
# ============================================================
cat("\n1c. Fitting 2PL model...\n")

mod_2pl <- ltm(LSAT ~ z1)
twopl_coef <- coef(mod_2pl)

write.csv(
    data.frame(
        item = rownames(twopl_coef),
        b = twopl_coef[, "Dffclt"],
        a = twopl_coef[, "Dscrmn"]
    ),
    file.path(output_dir, "lsat_2pl_params.csv"),
    row.names = FALSE
)

cat(mod_2pl$log.Lik,
    file = file.path(output_dir, "lsat_2pl_loglik.txt"))

eap_2pl <- factor.scores(mod_2pl, resp.patterns = LSAT, method = "EAP")
write.csv(
    data.frame(
        theta = eap_2pl$score.dat$z1,
        se = eap_2pl$score.dat$se.z1,
        obs = eap_2pl$score.dat$Obs
    ),
    file.path(output_dir, "lsat_2pl_eap.csv"),
    row.names = FALSE
)

eb_2pl <- factor.scores(mod_2pl, resp.patterns = LSAT, method = "EB")
write.csv(
    data.frame(
        theta = eb_2pl$score.dat$z1,
        se = eb_2pl$score.dat$se.z1,
        obs = eb_2pl$score.dat$Obs
    ),
    file.path(output_dir, "lsat_2pl_eb.csv"),
    row.names = FALSE
)

cat("  2PL loglik:", mod_2pl$log.Lik, "\n")

# ============================================================
# 2. Compare different quadrature settings
# ============================================================
cat("\n2. Comparing quadrature settings...\n")

mod_q15 <- rasch(LSAT, constraint = cbind(ncol(LSAT) + 1, 1),
                 control = list(GHk = 15))
mod_q21 <- rasch(LSAT, constraint = cbind(ncol(LSAT) + 1, 1),
                 control = list(GHk = 21))
mod_q41 <- rasch(LSAT, constraint = cbind(ncol(LSAT) + 1, 1),
                 control = list(GHk = 41))

quadpts_comparison <- data.frame(
    quadpts = c(15, 21, 41),
    loglik = c(mod_q15$log.Lik, mod_q21$log.Lik, mod_q41$log.Lik)
)
write.csv(quadpts_comparison,
          file.path(output_dir, "quadpts_comparison.csv"),
          row.names = FALSE)

cat("  Quadpts 15:", mod_q15$log.Lik, "\n")
cat("  Quadpts 21:", mod_q21$log.Lik, "\n")
cat("  Quadpts 41:", mod_q41$log.Lik, "\n")

# ============================================================
# 3. Abortion dataset (built-in)
# ============================================================
cat("\n3. Processing Abortion dataset...\n")

data(Abortion)
write.csv(Abortion, file.path(output_dir, "abortion_data.csv"), row.names = FALSE)

mod_abort_rasch <- rasch(Abortion, constraint = cbind(ncol(Abortion) + 1, 1))
abort_coef <- coef(mod_abort_rasch)

write.csv(
    data.frame(
        item = rownames(abort_coef),
        b = abort_coef[, "Dffclt"],
        a = abort_coef[, "Dscrmn"]
    ),
    file.path(output_dir, "abortion_rasch_params.csv"),
    row.names = FALSE
)

cat(mod_abort_rasch$log.Lik,
    file = file.path(output_dir, "abortion_rasch_loglik.txt"))

mod_abort_2pl <- ltm(Abortion ~ z1)
abort_2pl_coef <- coef(mod_abort_2pl)

write.csv(
    data.frame(
        item = rownames(abort_2pl_coef),
        b = abort_2pl_coef[, "Dffclt"],
        a = abort_2pl_coef[, "Dscrmn"]
    ),
    file.path(output_dir, "abortion_2pl_params.csv"),
    row.names = FALSE
)

cat(mod_abort_2pl$log.Lik,
    file = file.path(output_dir, "abortion_2pl_loglik.txt"))

cat("  Abortion Rasch loglik:", mod_abort_rasch$log.Lik, "\n")
cat("  Abortion 2PL loglik:", mod_abort_2pl$log.Lik, "\n")

# ============================================================
# 4. Item and Person fit statistics
# ============================================================
cat("\n4. Computing fit statistics...\n")

item_fit_rasch <- item.fit(mod_rasch_fixed)
write.csv(
    data.frame(
        item = names(item_fit_rasch$Tobs),
        Tobs = item_fit_rasch$Tobs,
        p_value = item_fit_rasch$p.values
    ),
    file.path(output_dir, "lsat_rasch_item_fit.csv"),
    row.names = FALSE
)

person_fit_rasch <- person.fit(mod_rasch_fixed)
write.csv(
    data.frame(
        L0 = person_fit_rasch$Tobs[, "L0"],
        Lz = person_fit_rasch$Tobs[, "Lz"],
        p_value = person_fit_rasch$p.values[, "Lz"]
    ),
    file.path(output_dir, "lsat_rasch_person_fit.csv"),
    row.names = FALSE
)

cat("  Fit statistics saved.\n")

# ============================================================
# Summary
# ============================================================
cat("\n============================================================\n")
cat("Reference values generated in:", output_dir, "\n")
cat("Files created:\n")
cat(paste("  -", list.files(output_dir), collapse = "\n"), "\n")
cat("============================================================\n")

cat("\nSummary of model log-likelihoods:\n")
cat("  Rasch (a=1):       ", mod_rasch_fixed$log.Lik, "\n")
cat("  Rasch (a estimated):", mod_rasch$log.Lik, "\n")
cat("  2PL:               ", mod_2pl$log.Lik, "\n")
