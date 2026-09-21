#' ---
#' title: "Module 14 - Mixed, multilevel, and repeated-measures models"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 14, equations (14.1)-(14.8)
#'
#' This module is the machinery that fixes Module 01's pseudoreplication.
#'
#' ## What you will learn
#'
#' 1. The marginal covariance V = ZGZ' + R (14.2).
#' 2. ICC (14.3) recovered as a variance-component ratio.
#' 3. **Partial pooling / BLUPs (14.5)-(14.6)**: shrinkage = empirical Bayes.
#' 4. Why the boundary LRT for sigma_b^2 = 0 is a chi2 MIXTURE.
#' 5. Mixed model vs pseudobulk aggregation.
#' 6. Satterthwaite/Kenward-Roger degrees of freedom.

#+ setup, message = FALSE
suppressPackageStartupMessages({library(nlme); library(lme4)})
MODULE_NAME <- "14_mixed_models"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Three ways to analyse clustered data

#+ three-ways
header("1. Wrong, crude, and correct analyses of clustered data")
simulate_clustered <- function(n_donors = 12, m_per = 40, effect = 0.6,
                               sigma_b = 1, sigma_e = 1) {
  donor <- rep(seq_len(n_donors), each = m_per)
  arm <- rep(sample(rep(0:1, n_donors / 2)), each = m_per)
  b <- rep(rnorm(n_donors, 0, sigma_b), each = m_per)
  data.frame(y = 5 + effect * arm + b + rnorm(n_donors * m_per, 0, sigma_e),
             arm = arm, donor = factor(donor))
}
set.seed(1401)
dc <- simulate_clustered()
cat(sprintf("  %d donors x %d cells = %d observations\n",
            nlevels(dc$donor), nrow(dc)/nlevels(dc$donor), nrow(dc)))
m_naive <- lm(y ~ arm, data = dc)                                   # WRONG
agg <- aggregate(y ~ donor + arm, dc, mean)                         # pseudobulk
m_agg <- lm(y ~ arm, data = agg)
m_lmm <- lme(y ~ arm, random = ~ 1 | donor, data = dc, method = "REML")
cat(sprintf("\n  %-34s%10s%10s%11s\n", "analysis", "beta_arm", "SE", "p"))
for (row_ in list(list("OLS on all cells (WRONG)", coef(summary(m_naive))["arm", ]),
                  list("OLS on donor means (pseudobulk)", coef(summary(m_agg))["arm", ]),
                  list("mixed model (random intercept)",
                       summary(m_lmm)$tTable["arm", c(1, 2, 5)])))
  cat(sprintf("  %-34s%10.4f%10.4f%11.4f\n", row_[[1]], row_[[2]][1], row_[[2]][2],
              row_[[2]][length(row_[[2]])]))
cat(sprintf("\n  The naive SE is %.1fx too small.\n",
            coef(summary(m_agg))["arm", 2] / coef(summary(m_naive))["arm", 2]))
cat("  Pseudobulk and the mixed model agree closely - for a BALANCED design\n")
cat("  they are near-equivalent, which is why pseudobulk is the recommended\n")
cat("  default for single-cell differential state (stats.md Topic 21).\n")

#' ## 2. Variance components and ICC, eq. (14.2)-(14.3)

#+ variance-components
header("2. Variance components and the ICC (14.2)-(14.3)")
vc <- as.numeric(VarCorr(m_lmm)[, "Variance"])
sigma_b2 <- vc[1]; sigma_e2 <- vc[2]
icc <- sigma_b2 / (sigma_b2 + sigma_e2)
m_per <- nrow(dc) / nlevels(dc$donor)
cat(sprintf("  estimated sigma_b^2 (between donors) = %.4f  (truth 1.00)\n", sigma_b2))
cat(sprintf("  estimated sigma_e^2 (within donor)   = %.4f  (truth 1.00)\n", sigma_e2))
cat(sprintf("  ICC (eq. 14.3)                       = %.4f\n", icc))
cat(sprintf("\n  design effect 1+(m-1)*ICC with m=%d: %.2f  (eq. 1.8)\n",
            m_per, 1 + (m_per - 1) * icc))
cat(sprintf("  predicted SE inflation: %.2fx     observed: %.2fx\n",
            sqrt(1 + (m_per - 1) * icc),
            coef(summary(m_agg))["arm", 2] / coef(summary(m_naive))["arm", 2]))
cat("  The mixed model has RECOVERED the design effect from the data.\n")

#' ## 3. Partial pooling and BLUPs, eq. (14.5)-(14.6)
#'
#' $$\hat b_j=\frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2/n_j}(\bar y_j-\mathbf x_j'\hat\beta)$$

#+ blups
header("3. Shrinkage: small clusters are trusted less (14.6)")
set.seed(1402)
sizes <- c(2, 3, 5, 10, 20, 50, 100, 200)      # deliberately UNBALANCED
true_b <- rnorm(length(sizes))
du <- do.call(rbind, lapply(seq_along(sizes), function(j)
  data.frame(y = rnorm(sizes[j], 4 + true_b[j], 1),
             donor = factor(sprintf("D%d", j)), nj = sizes[j])))
mu_fit <- lme(y ~ 1, random = ~ 1 | donor, data = du, method = "REML")
vcu <- as.numeric(VarCorr(mu_fit)[, "Variance"])
s_b2 <- vcu[1]; s_e2 <- vcu[2]
blups <- ranef(mu_fit)[, 1]; grand <- fixef(mu_fit)[1]
cat(sprintf("  sigma_b^2 = %.4f,  sigma_e^2 = %.4f\n", s_b2, s_e2))
cat(sprintf("  %-8s%5s%14s%17s%9s%10s\n", "donor", "n_j", "raw mean dev",
            "lambda_j (14.6)", "BLUP", "true b_j"))
for (j in seq_along(sizes)) {
  key <- sprintf("D%d", j)
  raw <- mean(du$y[du$donor == key]) - grand
  lam <- s_b2 / (s_b2 + s_e2 / sizes[j])
  cat(sprintf("  %-8s%5d%14.4f%17.4f%9.4f%10.4f\n", key, sizes[j], raw, lam,
              blups[match(key, rownames(ranef(mu_fit)))], true_b[j]))
}
cat("\n  For n_j=2 the shrinkage factor is small, so the BLUP is pulled far\n")
cat("  toward 0. For n_j=200 lambda is nearly 1 and the BLUP tracks the raw\n")
cat("  mean. This is the empirical-Bayes borrowing limma applies across genes.\n")

#' ## 4. The boundary problem: testing sigma_b^2 = 0
#'
#' H0 sits on the BOUNDARY, so the LRT statistic is a 50:50 mixture
#' 0.5*chi2_0 + 0.5*chi2_1, not chi2_1.

#+ boundary
header("4. LRT on the boundary is a chi2 mixture, not chi2_1")
set.seed(1403)
stats_null <- na.omit(replicate(250, {
  dn <- simulate_clustered(n_donors = 10, m_per = 8, effect = 0, sigma_b = 0)
  f_mm <- try(lme(y ~ arm, random = ~ 1 | donor, data = dn, method = "ML"),
              silent = TRUE)
  if (inherits(f_mm, "try-error")) return(NA)
  f_ols <- gls(y ~ arm, data = dn, method = "ML")
  max(0, 2 * (logLik(f_mm) - logLik(f_ols)))
}))
cat(sprintf("  %d simulations with sigma_b^2 TRULY zero\n", length(stats_null)))
cat(sprintf("  proportion of LRT statistics ~0 : %.3f   (theory: 0.5)\n",
            mean(stats_null < 1e-6)))
cat(sprintf("\n  rejection rate using chi2_1 crit (%.3f) : %.3f   <- conservative\n",
            qchisq(0.95, 1), mean(stats_null > qchisq(0.95, 1))))
cat(sprintf("  rejection rate using mixture crit (%.3f) : %.3f   <- calibrated\n",
            qchisq(0.90, 1), mean(stats_null > qchisq(0.90, 1))))

#' ## 5. Degrees of freedom matter when clusters are few

#+ dof
header("5. Mixed-model p-values with few clusters")
fpr_by_clusters <- function(n_donors, m_per = 20, n_sim = 250) {
  hits <- c(lme = 0, agg = 0); ok <- 0
  for (i in seq_len(n_sim)) {
    dd <- simulate_clustered(n_donors, m_per, effect = 0)
    f <- try(lme(y ~ arm, random = ~ 1 | donor, data = dd), silent = TRUE)
    if (inherits(f, "try-error")) next
    hits["lme"] <- hits["lme"] + (summary(f)$tTable["arm", "p-value"] < 0.05)
    ag <- aggregate(y ~ donor + arm, dd, mean)
    hits["agg"] <- hits["agg"] +
      (t.test(y ~ arm, data = ag)$p.value < 0.05)
    ok <- ok + 1
  }
  hits / ok
}
set.seed(1404)
cat(sprintf("  %8s%22s%20s\n", "donors", "nlme::lme FPR", "pseudobulk t FPR"))
for (nd in c(4, 6, 10, 20)) {
  r <- fpr_by_clusters(nd)
  cat(sprintf("  %8d%21.1f%%%19.1f%%\n", nd, 100*r["lme"], 100*r["agg"]))
}
cat("\n  `nlme::lme` uses a within-group df approximation that behaves well\n")
cat("  here. But GLMM software that reports z-based p-values (and lme4, which\n")
cat("  reports none by default) is ANTICONSERVATIVE with few clusters. Use\n")
cat("  lmerTest's Satterthwaite or Kenward-Roger df, or aggregate.\n")

#' ## 6. Random slopes and repeated measures, eq. (28.1)-(28.2)

#+ random-slopes
header("6. Random intercepts vs random slopes")
set.seed(1405)
n_subj <- 24; n_visit <- 5
dl <- do.call(rbind, lapply(seq_len(n_subj), function(s) {
  arm <- s %% 2
  b0 <- rnorm(1, 0, 1.2); b1 <- rnorm(1, 0, 0.35)    # subject slope heterogeneity
  data.frame(subject = factor(sprintf("S%02d", s)), arm = arm, t = 0:(n_visit-1),
             y = (10 + b0) + (0.20 + 0.30*arm + b1) * (0:(n_visit-1)) + rnorm(n_visit, 0, 0.5))
}))
m_ri <- lme(y ~ t * arm, random = ~ 1 | subject, data = dl)
m_rs <- lme(y ~ t * arm, random = ~ t | subject, data = dl)
cat("  TRUE treatment x time interaction = 0.300\n")
cat(sprintf("  %-28s%13s%9s%11s\n", "model", "beta(t:arm)", "SE", "p"))
for (row_ in list(list("random intercept only", summary(m_ri)$tTable["t:arm", ]),
                  list("random intercept + slope", summary(m_rs)$tTable["t:arm", ])))
  cat(sprintf("  %-28s%13.4f%9.4f%11.4f\n", row_[[1]], row_[[2]]["Value"],
              row_[[2]]["Std.Error"], row_[[2]]["p-value"]))
cat(sprintf("\n  estimated random-slope SD = %.4f  (truth 0.35)\n",
            as.numeric(VarCorr(m_rs)[2, "StdDev"])))
cat(sprintf("  LRT for the random slope: p = %.4f\n",
            anova(m_ri, m_rs)[["p-value"]][2]))
cat("  Ignoring genuine slope heterogeneity understates the SE of the\n")
cat("  treatment x time interaction - the estimand of a longitudinal trial.\n")

#' ## 7. lme4 and subject-specific vs population-average, eq. (14.7)

#+ lme4-marginal
header("7. lme4, and conditional vs marginal effects (14.7)")
m4 <- lmer(y ~ arm + (1 | donor), data = dc, REML = TRUE)
cat(sprintf("  lme4::lmer beta_arm = %.4f (SE %.4f)\n",
            fixef(m4)["arm"], sqrt(diag(vcov(m4)))["arm"]))
cat(sprintf("  nlme::lme  beta_arm = %.4f (SE %.4f)   <- same model, same answer\n",
            fixef(m_lmm)["arm"], summary(m_lmm)$tTable["arm", "Std.Error"]))
cat("\n  For the IDENTITY link, subject-specific and population-average effects\n")
cat("  COINCIDE, which is why this trap is invisible in linear models. In a\n")
cat("  logistic GLMM, beta_marginal ~ beta_cond / sqrt(1 + 0.346 sigma_b^2):\n")
for (s2 in c(0.5, 1, 2, 4))
  cat(sprintf("    sigma_b^2 = %.1f -> attenuation factor %.3f\n",
              s2, 1/sqrt(1 + 0.346*s2)))
cat("\n  GEE estimates the MARGINAL effect; GLMM the CONDITIONAL one. Say which.\n")
cat("  GEE's sandwich needs MANY clusters (J >= 40); with 6 donors it is\n")
cat("  badly biased and a mixed model or aggregation is safer (Module 34).\n")

#' ## 8. Figure

#+ figure
png(file.path(OUT, "mixed_models.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
plot(jitter(dc$arm, 0.4), dc$y, pch = ".", col = as.integer(dc$donor),
     xaxt = "n", xlab = "", ylab = "y", main = "Cells cluster within donors")
axis(1, 0:1, c("control", "treated"))
for (dlev in levels(dc$donor)) {
  sub <- dc[dc$donor == dlev, ]
  points(sub$arm[1], mean(sub$y), pch = "_", cex = 2)
}
lam_v <- s_b2 / (s_b2 + s_e2 / sizes)
raw_v <- sapply(seq_along(sizes), function(j) mean(du$y[du$donor == sprintf("D%d", j)]) - grand)
bl_v <- blups[match(sprintf("D%d", seq_along(sizes)), rownames(ranef(mu_fit)))]
plot(sizes, raw_v, type = "b", pch = 16, log = "x", col = "steelblue", ylim = range(c(raw_v, bl_v, true_b)),
     xlab = "cluster size n_j", ylab = "", main = "Eq. (14.6): shrinkage depends on n_j")
lines(sizes, bl_v, type = "b", pch = 15, col = "darkorange")
lines(sizes, true_b, type = "b", pch = 17, lty = 2, col = "forestgreen")
abline(h = 0)
legend("topright", c("raw deviation", "BLUP (shrunk)", "true b_j"),
       col = c("steelblue", "darkorange", "forestgreen"), pch = c(16, 15, 17),
       bty = "n", cex = 0.7)
plot(NA, xlim = c(0, 4), ylim = range(dl$y), xlab = "visit", ylab = "y",
     main = "Random slopes: subjects diverge")
for (s in levels(dl$subject)) {
  sub <- dl[dl$subject == s, ]
  lines(sub$t, sub$y, col = c("steelblue", "darkorange")[sub$arm[1] + 1], lwd = 0.8)
}
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "mixed_models.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 14)
#'
#' 1. Enumerate every level of clustering. Random effects come from the DESIGN.
#' 2. Fixed blocking effects when clusters are few; random effects when many.
#' 3. Use Satterthwaite/Kenward-Roger df; never trust z-based GLMM p-values
#'    with few clusters.
#' 4. With a simple balanced design, aggregate - transparent and nearly always
#'    valid.
#'
#' **Next:** `15_missing_data_and_censoring.R`
