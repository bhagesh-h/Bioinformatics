#' ---
#' title: "Module 10 - Correlation and dependence"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 10, equations (10.1)-(10.10)
#'
#' ## What you will learn
#'
#' 1. Pearson measures LINEAR association only, and is scale-dependent.
#' 2. Fisher's z (10.3) for correct intervals.
#' 3. Partial correlation (10.5) and the precision matrix (10.6).
#' 4. Simpson's paradox and repeated-measures correlation (10.7).
#' 5. Correlation is NOT agreement -- Bland-Altman (10.8), Lin's CCC (10.9).
#' 6. Correlation matrices under multiplicity, and shrinkage (10.10).

#+ setup, message = FALSE
MODULE_NAME <- "10_correlation_and_dependence"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. What Pearson does and does not detect

#+ what-pearson
header("1. r = 0 does not mean independent (10.1)-(10.2)")
set.seed(1001)
## Distinct names (`x1`, not `x`): later sections rebind `x` and `n`, and the
## figure at the bottom still needs these. Shadowing is the commonest cause of
## "it ran yesterday" breakage in analysis scripts.
n1 <- 2000; x1 <- runif(n1, -3, 3)
cases <- list(linear = 1.5 * x1 + rnorm(n1), quadratic = x1^2 + rnorm(n1),
              `monotone (exp)` = exp(x1) + rnorm(n1), independent = rnorm(n1))
cat(sprintf("  %-18s%11s%10s%10s\n", "relationship", "Pearson r", "Spearman", "Kendall"))
for (nm in names(cases))
  cat(sprintf("  %-18s%11.3f%10.3f%10.3f\n", nm, cor(x1, cases[[nm]]),
              cor(x1, cases[[nm]], method = "spearman"),
              cor(x1, cases[[nm]], method = "kendall")))
cat("\n  'quadratic': r ~ 0 despite a perfect deterministic relationship.\n")
cat("  'monotone' : Spearman/Kendall detect it far better than Pearson.\n")

ca <- rlnorm(500, 3, 1); cb <- ca^1.3 * rlnorm(500, 0, 0.4)
cat(sprintf("\n  Same data, different scale:\n"))
cat(sprintf("    r on raw scale       = %.3f\n", cor(ca, cb)))
cat(sprintf("    r on log scale       = %.3f\n", cor(log(ca), log(cb))))
cat(sprintf("    Spearman (invariant) = %.3f\n", cor(ca, cb, method = "spearman")))
cat("  ALWAYS state the scale on which a correlation was computed.\n")

#' ## 2. Fisher's z-transform, eq. (10.3)

#+ fisher-z
header("2. Correct confidence intervals for a correlation (10.3)")
fisher_ci <- function(r, n, alpha = 0.05) {
  z <- atanh(r); se <- 1 / sqrt(n - 3)
  tanh(z + c(-1, 1) * qnorm(1 - alpha/2) * se)
}
for (cfg in list(c(0.9, 10), c(0.9, 100), c(0.3, 20), c(-0.6, 15))) {
  r <- cfg[1]; nn <- cfg[2]
  ci <- fisher_ci(r, nn); naive <- r + c(-1, 1) * 1.96 / sqrt(nn)
  cat(sprintf("  r=%+.2f, n=%4d: Fisher CI [%+.3f, %+.3f]   naive [%+.3f, %+.3f]\n",
              r, nn, ci[1], ci[2], naive[1], naive[2]))
}
cat("\n  The Fisher interval is asymmetric and stays inside [-1,1]; the naive\n")
cat("  one escapes the parameter space. Coverage check:\n")
set.seed(1002)
RHO <- 0.7; nn <- 15
cov <- replicate(20000, {
  s <- matrix(c(1, RHO, RHO, 1), 2)
  xy <- MASS::mvrnorm(nn, c(0, 0), s)
  r <- cor(xy[, 1], xy[, 2])
  ci <- fisher_ci(r, nn)
  c(ci[1] <= RHO && RHO <= ci[2],
    (r - 1.96/sqrt(nn)) <= RHO && RHO <= (r + 1.96/sqrt(nn)))
})
cat(sprintf("    Fisher coverage = %.1f%%   naive coverage = %.1f%%   (nominal 95%%)\n",
            100 * mean(cov[1, ]), 100 * mean(cov[2, ])))
cat(sprintf("    cor.test() CI for comparison: matches Fisher by construction.\n"))

#' ## 3. Partial correlation and the precision matrix, eq. (10.5)-(10.6)

#+ partial
header("3. Marginal vs conditional independence (10.5)-(10.6)")
set.seed(1003)
n <- 3000
z <- rnorm(n); x <- 1.2 * z + rnorm(n); y <- 1.5 * z + rnorm(n)
partial_corr <- function(a, b, c) {                             # eq. (10.5)
  rab <- cor(a, b); rac <- cor(a, c); rbc <- cor(b, c)
  (rab - rac * rbc) / sqrt((1 - rac^2) * (1 - rbc^2))
}
S <- cov(cbind(x, y, z)); Omega <- solve(S)
pc_precision <- -Omega[1, 2] / sqrt(Omega[1, 1] * Omega[2, 2])   # eq. (10.6)
cat(sprintf("  marginal r(X,Y)       = %+.4f\n", cor(x, y)))
cat(sprintf("  partial  r(X,Y | Z)   = %+.4f  (eq. 10.5)\n", partial_corr(x, y, z)))
cat(sprintf("  from precision matrix = %+.4f  (eq. 10.6) <- same\n", pc_precision))
cat("\n  X and Y look strongly associated but are conditionally independent\n")
cat("  given Z. Co-expression networks on MARGINAL correlations fill with\n")
cat("  such indirect edges (stats.md Topic 30).\n")

#' ## 4. Simpson's paradox and repeated measures, eq. (10.7)

#+ simpsons
header("4. Pooling repeated measures can REVERSE the association (10.7)")
set.seed(1004)
n_subj <- 12; n_visits <- 8
recs <- do.call(rbind, lapply(seq_len(n_subj), function(s) {
  base_x <- rnorm(1, s, 0.3)
  base_y <- rnorm(1, 30 - 2 * s, 0.8)
  xv <- base_x + rnorm(n_visits, 0, 0.5)
  data.frame(subject = sprintf("S%02d", s), x = xv,
             y = base_y + 1.5 * (xv - base_x) + rnorm(n_visits, 0, 0.4))
}))
pooled_r <- cor(recs$x, recs$y)
fit <- lm(y ~ x + subject, data = recs); fit0 <- lm(y ~ subject, data = recs)
ss_model <- sum(residuals(fit0)^2) - sum(residuals(fit)^2)
r_rm <- sign(coef(fit)["x"]) * sqrt(ss_model / (ss_model + sum(residuals(fit)^2)))
cat(sprintf("  naive pooled correlation      r = %+.4f\n", pooled_r))
cat(sprintf("  repeated-measures correlation r = %+.4f   (eq. 10.7)\n", r_rm))
cat(sprintf("  common within-subject slope     = %+.4f\n", coef(fit)["x"]))
cat("\n  The pooled correlation has the OPPOSITE SIGN to the within-subject\n")
cat("  relationship. Handle donor structure before interpreting correlation.\n")

#' ## 5. Correlation is not agreement, eq. (10.8)-(10.9)

#+ agreement
header("5. Bland-Altman and Lin's CCC (10.8)-(10.9)")
set.seed(1005)
truth <- rnorm(120, 100, 15)
assay1 <- truth + rnorm(120, 0, 2)
assay2 <- 1.9 * truth + 8 + rnorm(120, 0, 2)    # perfectly correlated, wrong scale
lins_ccc <- function(a, b) {                                     # eq. (10.9)
  va <- var(a) * (length(a)-1)/length(a); vb <- var(b) * (length(b)-1)/length(b)
  cv <- cov(a, b) * (length(a)-1)/length(a)
  2 * cv / (va + vb + (mean(a) - mean(b))^2)
}
d_ba <- assay1 - assay2
cat(sprintf("  Pearson r(assay1, assay2) = %.5f   <- 'excellent agreement'?\n",
            cor(assay1, assay2)))
cat(sprintf("  Lin's CCC                 = %.5f   <- no, they disagree badly\n",
            lins_ccc(assay1, assay2)))
cat(sprintf("  Bland-Altman bias         = %+.2f\n", mean(d_ba)))
cat(sprintf("  95%% limits of agreement   = [%+.2f, %+.2f]   (eq. 10.8)\n",
            mean(d_ba) - 1.96 * sd(d_ba), mean(d_ba) + 1.96 * sd(d_ba)))
cat("\n  r near 1 with CCC near 0: the assays RANK samples identically but\n")
cat("  report entirely different numbers. For method comparison always use\n")
cat("  (10.8)-(10.9), never r.\n")

#' ## 6. Correlation matrices: multiplicity and shrinkage, eq. (10.10)

#+ multiplicity-shrinkage
header("6. Co-expression under multiplicity, and shrinkage (10.10)")
set.seed(1006)
G <- 200; n_samp <- 20
X <- matrix(rnorm(n_samp * G), nrow = n_samp)      # COMPLETELY independent
R <- cor(X); iu <- upper.tri(R); rs <- R[iu]
tvals <- rs * sqrt((n_samp - 2) / (1 - rs^2))
pvals <- 2 * pt(abs(tvals), n_samp - 2, lower.tail = FALSE)
cat(sprintf("  G = %d features, n = %d samples, ALL truly independent\n", G, n_samp))
cat(sprintf("  number of pairwise tests = G(G-1)/2 = %s\n",
            format(length(rs), big.mark = ",")))
cat(sprintf("  'significant' at p < 0.05 (unadjusted): %s (%.1f%%)\n",
            format(sum(pvals < 0.05), big.mark = ","), 100 * mean(pvals < 0.05)))
cat(sprintf("  |r| > 0.5 purely by chance             : %s\n",
            format(sum(abs(rs) > 0.5), big.mark = ",")))
cat(sprintf("  surviving BH at 0.05                   : %s\n",
            format(sum(p.adjust(pvals, "BH") < 0.05), big.mark = ",")))
cat("\n  Co-expression networks built from unadjusted correlations are mostly\n")
cat("  noise at the small n typical of omics.\n")

ledoit_wolf <- function(X) {                                     # eq. (10.10)
  n <- nrow(X); p <- ncol(X)
  Xc <- scale(X, center = TRUE, scale = FALSE)
  S <- crossprod(Xc) / n
  mu <- sum(diag(S)) / p
  Tm <- mu * diag(p)
  d2 <- sum((S - Tm)^2) / p
  b2bar <- mean(apply(Xc, 1, function(r) sum((tcrossprod(r) - S)^2) / p)) / n
  lam <- min(b2bar, d2) / d2
  list(S = (1 - lam) * S + lam * Tm, lambda = lam)
}
lw <- ledoit_wolf(X)
cat(sprintf("\n  p=%d > n=%d, so the sample covariance is SINGULAR:\n", G, n_samp))
cat(sprintf("    rank(S_sample) = %d (needs %d)\n", qr(cov(X))$rank, G))
cat(sprintf("    Ledoit-Wolf shrinkage intensity lambda = %.4f\n", lw$lambda))
cat(sprintf("    rank(S_shrunk) = %d  -> invertible, which is what makes the\n",
            qr(lw$S)$rank))
cat("    precision matrix of eq. (10.6) computable.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "correlation.png"), width = 1500, height = 400, res = 110)
par(mfrow = c(1, 4), mar = c(4.2, 4.2, 2.5, 1))
plot(x1, cases$quadratic, pch = ".", col = "steelblue", xlab = "x", ylab = "y",
     main = sprintf("r = %.3f, yet deterministic", cor(x1, cases$quadratic)))
plot(recs$x, recs$y, col = factor(recs$subject), pch = 16, cex = 0.6,
     xlab = "x", ylab = "y", main = "Eq. (10.7): Simpson's paradox")
abline(lm(y ~ x, data = recs), lty = 2, lwd = 2)
plot((assay1 + assay2)/2, d_ba, pch = 16, cex = 0.6, col = "steelblue",
     xlab = "mean of the two assays", ylab = "difference",
     main = "Eq. (10.8): Bland-Altman")
abline(h = mean(d_ba), lwd = 2)
abline(h = mean(d_ba) + c(-1, 1) * 1.96 * sd(d_ba), col = "red", lty = 2)
hist(rs, breaks = 60, col = "steelblue", border = "white",
     xlab = "pairwise r among INDEPENDENT features",
     main = sprintf("n=%d: |r|>0.5 in %.1f%% of pairs", n_samp, 100*mean(abs(rs)>0.5)))
abline(v = c(-0.5, 0.5), col = "red", lty = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "correlation.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 10)
#'
#' 1. Pearson for linear association on a justified scale; Spearman/Kendall for
#'    monotone association or ordinal data.
#' 2. Handle donor structure explicitly (10.7).
#' 3. Use Fisher's z (10.3) for intervals and tests.
#' 4. For method comparison report (10.8)-(10.9), not r.
#' 5. Apply log-ratio transformation before correlating compositions.
#'
#' **Next:** `11_linear_models.R`
