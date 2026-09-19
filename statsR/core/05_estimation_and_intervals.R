#' ---
#' title: "Module 05 - Estimation, effect sizes, and confidence intervals"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 5, equations (5.1)-(5.15)
#'
#' Estimation, not dichotomised testing, is the primary scientific task.
#'
#' ## What you will learn
#'
#' 1. Bias-variance (5.1): why accepting bias can REDUCE total error.
#' 2. What a confidence interval guarantees, by simulation (5.3)-(5.4).
#' 3. Cohen's d and the Hedges correction (5.5)-(5.6).
#' 4. Ratio measures on the log scale and the delta method (5.7)-(5.11).
#' 5. Bootstrap intervals with `boot` (5.12)-(5.13), and the unit rule.
#' 6. Robust (sandwich) SEs (5.14) and equivalence testing (5.15).

#+ setup, message = FALSE
suppressPackageStartupMessages(library(boot))
MODULE_NAME <- "05_estimation_and_intervals"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Bias-variance decomposition, eq. (5.1)

#+ bias-variance
header("1. Bias-variance: why a biased estimator can win (5.1)")
set.seed(501)
G <- 60; SIGMA <- 1
theta_true <- rnorm(G, 0, 0.7)
shrink <- function(y, lam) mean(y) + (1 - lam) * (y - mean(y))

tab <- do.call(rbind, lapply(c(0, 0.2, 0.4, 0.6, 0.8, 1.0), function(lam) {
  ests <- replicate(3000, shrink(theta_true + rnorm(G, 0, SIGMA), lam))
  bias2 <- mean((rowMeans(ests) - theta_true)^2)
  vari  <- mean(apply(ests, 1, var))
  data.frame(shrinkage = lam, bias2 = bias2, variance = vari,
             MSE = mean((ests - theta_true)^2), sum = bias2 + vari)
}))
print(format(tab, digits = 4), row.names = FALSE)
cat(sprintf("\nMinimum MSE at shrinkage = %.1f, NOT at 0 (the unbiased estimator).\n",
            tab$shrinkage[which.min(tab$MSE)]))
cat("'sum' reproduces 'MSE' exactly - that is eq. (5.1).\n")

#' ## 2. What a confidence interval guarantees, eq. (5.2)-(5.4)

#+ coverage
header("2. Coverage, and the CI/test duality (5.3)-(5.4)")
set.seed(502)
MU <- 5; SD <- 2; n <- 12
cov_hits <- replicate(20000, {
  x <- rnorm(n, MU, SD)
  ci <- t.test(x)$conf.int
  ci[1] <= MU && MU <= ci[2]
})
cat(sprintf("  nominal coverage 95%%, empirical = %.2f%%\n", 100 * mean(cov_hits)))

set.seed(503)
x <- rnorm(n, MU, SD)
grid <- seq(mean(x) - 4, mean(x) + 4, length.out = 4001)
pvals <- sapply(grid, function(m) t.test(x, mu = m)$p.value)
not_rej <- range(grid[pvals > 0.05])
cat(sprintf("\n  CI from t.test()          : [%.4f, %.4f]\n", t.test(x)$conf.int[1],
            t.test(x)$conf.int[2]))
cat(sprintf("  Set of mu with p > 0.05   : [%.4f, %.4f]   <- eq. (5.4), identical\n",
            not_rej[1], not_rej[2]))

#' ## 3. Standardised effect sizes, eq. (5.5)-(5.6)

#+ effect-size
header("3. Cohen's d and the Hedges correction (5.5)-(5.6)")
cohens_d <- function(x, y) {
  n1 <- length(x); n2 <- length(y)
  sp <- sqrt(((n1 - 1) * var(x) + (n2 - 1) * var(y)) / (n1 + n2 - 2))
  (mean(x) - mean(y)) / sp
}
hedges_J <- function(nu) exp(lgamma(nu/2) - 0.5 * log(nu/2) - lgamma((nu - 1)/2))

cat(sprintf("%12s %5s %10s %13s %17s\n", "n per group", "nu", "exact J",
            "1-3/(4nu-1)", "d overstated by"))
for (n in c(3, 4, 6, 10, 20, 50)) {
  nu <- 2 * n - 2
  cat(sprintf("%12d %5d %10.4f %13.4f %16.1f%%\n",
              n, nu, hedges_J(nu), 1 - 3/(4*nu - 1), 100 * (1/hedges_J(nu) - 1)))
}
set.seed(504)
TRUE_D <- 0.8; n <- 4
ds <- replicate(20000, cohens_d(rnorm(n, TRUE_D), rnorm(n)))
cat(sprintf("\n  true d = %.1f, n = %d per group\n", TRUE_D, n))
cat(sprintf("  mean Cohen's d = %.4f   (biased upward)\n", mean(ds)))
cat(sprintf("  mean Hedges' g = %.4f   (corrected)\n", mean(hedges_J(2*n-2) * ds)))
cat("\n  CAUTION: d is a RATIO. Always report the raw difference and units too.\n")

#' ## 4. Ratio measures and the delta method, eq. (5.7)-(5.10)

#+ ratios
header("4. Odds ratio, risk ratio, delta method (5.7)-(5.10)")
a <- 30; b <- 70; cc <- 15; dd <- 85       # exposed(event,no) / unexposed(event,no)
or_hat <- (a * dd) / (b * cc)
se_log_or <- sqrt(1/a + 1/b + 1/cc + 1/dd)                    # eq. (5.7)
risk1 <- a/(a+b); risk0 <- cc/(cc+dd)
rr_hat <- risk1 / risk0
se_log_rr <- sqrt(1/a - 1/(a+b) + 1/cc - 1/(cc+dd))           # eq. (5.8)
cat(sprintf("  baseline risk (unexposed) = %.3f\n", risk0))
cat(sprintf("  odds ratio  = %.3f   95%% CI [%.3f, %.3f]\n", or_hat,
            exp(log(or_hat) - 1.96*se_log_or), exp(log(or_hat) + 1.96*se_log_or)))
cat(sprintf("  risk ratio  = %.3f   95%% CI [%.3f, %.3f]\n", rr_hat,
            exp(log(rr_hat) - 1.96*se_log_rr), exp(log(rr_hat) + 1.96*se_log_rr)))
cat(sprintf("  risk difference = %+.3f\n", risk1 - risk0))
cat("\n  OR > RR always (when risk > 0). They coincide only for RARE outcomes:\n")
for (p0 in c(0.01, 0.10, 0.40)) {
  OR <- 2
  cat(sprintf("    baseline risk %5.2f:  OR = 2.00  corresponds to RR = %.3f\n",
              p0, OR / (1 - p0 + p0 * OR)))
}
set.seed(505)
sims <- replicate(20000, {
  ea <- rbinom(1, 100, 0.30); ec <- rbinom(1, 100, 0.15)
  if (ea %in% c(0, 100) || ec %in% c(0, 100)) NA
  else log((ea * (100 - ec)) / ((100 - ea) * ec))
})
cat(sprintf("\n  delta-method SE(log OR) = %.4f\n", se_log_or))
cat(sprintf("  simulated  SD(log OR)   = %.4f   <- eq. (5.9) works\n",
            sd(sims, na.rm = TRUE)))

#' ## 5. Bootstrap intervals with `boot`, eq. (5.12)-(5.13)
#'
#' The `boot` package ships with R (a recommended package), so no install.

#+ bootstrap
header("5. Bootstrap: percentile, basic, BCa (5.12)-(5.13)")
set.seed(506)
skewed <- rlnorm(25, meanlog = 1, sdlog = 0.8)     # e.g. cytokine concentrations
bs <- boot(skewed, statistic = function(d, i) mean(d[i]), R = 4000)
bci <- boot.ci(bs, type = c("perc", "basic", "bca"))
cat(sprintf("  sample (n=25, log-normal), mean = %.3f\n", mean(skewed)))
cat(sprintf("    percentile : [%.3f, %.3f]\n", bci$percent[4], bci$percent[5]))
cat(sprintf("    basic      : [%.3f, %.3f]\n", bci$basic[4], bci$basic[5]))
cat(sprintf("    BCa        : [%.3f, %.3f]\n", bci$bca[4], bci$bca[5]))
cat(sprintf("    t-interval : [%.3f, %.3f]\n", t.test(skewed)$conf.int[1],
            t.test(skewed)$conf.int[2]))
cat("\n  BCa is asymmetric here, correctly reflecting the skew.\n")

#+ bootstrap-unit
cat("\n--- Resampling the wrong level reproduces pseudoreplication ---\n")
set.seed(507)
n_donors <- 6; m_cells <- 300
cells <- matrix(rnorm(n_donors * m_cells), nrow = n_donors) + rnorm(n_donors)
flat <- as.vector(cells)
wrong <- replicate(3000, mean(sample(flat, length(flat), replace = TRUE)))
## CLUSTER bootstrap: resample DONORS, carrying all their cells.
right <- replicate(3000, mean(cells[sample(n_donors, n_donors, replace = TRUE), ]))
cat(sprintf("  bootstrap SE resampling cells  : %.4f\n", sd(wrong)))
cat(sprintf("  bootstrap SE resampling donors : %.4f\n", sd(right)))
cat(sprintf("  truth sqrt(sigma_b^2/n + ...)  : %.4f   (eq. 1.5)\n",
            sqrt(1/n_donors + 1/(n_donors * m_cells))))
cat(sprintf("  The cell bootstrap is ~%.0fx too optimistic.\n", sd(right) / sd(wrong)))

#' ## 6. Robust (sandwich) standard errors, eq. (5.14)
#'
#' Base R has no `vcovHC`, so we implement it -- which is instructive anyway.

#+ robust-se
header("6. Heteroscedasticity-consistent SEs (5.14)")
vcov_hc <- function(fit, type = c("HC0", "HC1", "HC2", "HC3")) {
  type <- match.arg(type)
  X <- model.matrix(fit); e <- residuals(fit)
  n <- nrow(X); p <- ncol(X)
  h <- hatvalues(fit)
  omega <- switch(type, HC0 = rep(1, n), HC1 = rep(n/(n - p), n),
                  HC2 = 1/(1 - h), HC3 = 1/(1 - h)^2)
  bread <- solve(crossprod(X))
  meat <- crossprod(X * sqrt(omega * e^2))
  bread %*% meat %*% bread                                  # eq. (5.14)
}
robust_p <- function(fit, term, type = "HC3") {
  b <- coef(fit)[term]
  se <- sqrt(diag(vcov_hc(fit, type))[term])
  2 * pt(abs(b / se), df.residual(fit), lower.tail = FALSE)
}

set.seed(508)
n <- 40
x <- runif(n, 0, 3)
y <- 1 + rnorm(n, 0, 0.3 + 1.2 * x)       # TRUE slope is 0; variance grows with x
fit <- lm(y ~ x)
cat(sprintf("  %-16s%12s%10s\n", "method", "SE(slope)", "p-value"))
cat(sprintf("  %-16s%12.4f%10.4f\n", "classical OLS",
            coef(summary(fit))["x", "Std. Error"], coef(summary(fit))["x", "Pr(>|t|)"]))
for (tp in c("HC0", "HC1", "HC2", "HC3")) {
  cat(sprintf("  %-16s%12.4f%10.4f\n", tp,
              sqrt(diag(vcov_hc(fit, tp))["x"]), robust_p(fit, "x", tp)))
}
set.seed(509)
counts <- c(classical = 0, HC0 = 0, HC3 = 0)
for (i in 1:1500) {
  xx <- runif(n, 0, 3); yy <- 1 + rnorm(n, 0, 0.3 + 1.2 * xx)
  f <- lm(yy ~ xx)
  counts["classical"] <- counts["classical"] + (coef(summary(f))["xx", "Pr(>|t|)"] < 0.05)
  counts["HC0"] <- counts["HC0"] + (robust_p(f, "xx", "HC0") < 0.05)
  counts["HC3"] <- counts["HC3"] + (robust_p(f, "xx", "HC3") < 0.05)
}
cat("\n  False-positive rate under a TRUE null (nominal 5%), n = 40:\n")
for (k in names(counts)) cat(sprintf("    %-11s: %.1f%%\n", k, 100 * counts[k] / 1500))
cat("  HC0 is the most anticonservative and HC3 the least - the ordering\n")
cat("  HC0 > HC1 > HC2 > HC3 in SE size is guaranteed by their weights. At\n")
cat("  n = 40 the gap is modest; it widens sharply as n falls, which is why\n")
cat("  HC3 is the right default at bioinformatics sample sizes (n < 250).\n")

#' ## 7. Equivalence testing (TOST), eq. (5.15)

#+ tost
header("7. TOST: the only valid way to claim 'no meaningful difference' (5.15)")
tost <- function(x, y, delta, alpha = 0.05) {
  tt <- t.test(x, y)                                  # Welch, for df and se
  diff <- diff(rev(c(mean(y), mean(x))))
  se <- sqrt(var(x)/length(x) + var(y)/length(y))
  df <- tt$parameter
  p_lower <- pt((diff + delta) / se, df, lower.tail = FALSE)  # H01: theta <= -delta
  p_upper <- pt((diff - delta) / se, df)                       # H02: theta >= +delta
  p <- max(p_lower, p_upper)
  ci90 <- diff + c(-1, 1) * qt(1 - alpha, df) * se
  list(diff = diff, p = p, ci90 = ci90, equivalent = p < alpha)
}
set.seed(510)
delta <- 0.5
for (n in c(10, 400)) {
  a <- rnorm(n); b <- rnorm(n)
  r <- tost(a, b, delta)
  cat(sprintf("\n  n = %d per group, TRUE difference = 0:\n", n))
  cat(sprintf("    difference = %+.3f, NHST p = %.3f\n", r$diff, t.test(a, b)$p.value))
  cat(sprintf("    TOST p = %.4f, 90%% CI [%+.3f, %+.3f] -> equivalent? %s\n",
              r$p, r$ci90[1], r$ci90[2], r$equivalent))
}
cat("\n  Not significant AND not equivalent = the study was inconclusive.\n")
cat("  Only with enough data can you positively claim |difference| < delta.\n")

#' ## 8. Figure

#+ figure
png(file.path(OUT, "estimation.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
matplot(tab$shrinkage, tab[, c("bias2", "variance", "MSE")], type = "b", pch = 16,
        lty = 1, col = c("steelblue", "darkorange", "firebrick"), lwd = 2,
        xlab = "shrinkage toward the mean", ylab = "",
        main = "Eq. (5.1): bias-variance trade-off")
legend("left", c("bias^2", "variance", "MSE"),
       col = c("steelblue", "darkorange", "firebrick"), lwd = 2, bty = "n", cex = 0.7)

set.seed(511)
plot(NA, xlim = c(2, 8), ylim = c(0, 41), xlab = "estimate", ylab = "",
     yaxt = "n", main = "Eq. (5.3): 95% CIs across repetitions")
for (i in 1:40) {
  xx <- rnorm(12, MU, SD)
  # NOTE the deliberate rename: calling this `ci` would shadow the boot.ci
  # object above. Shadowing is the commonest cause of "it worked yesterday".
  ci_i <- t.test(xx)$conf.int
  segments(ci_i[1], i, ci_i[2], i,
           col = if (ci_i[1] <= MU && MU <= ci_i[2]) "steelblue" else "red", lwd = 1.5)
}
abline(v = MU, lty = 2)

hist(bs$t, breaks = 60, col = "lightsteelblue", border = "white",
     main = "Eq. (5.12)-(5.13): bootstrap distribution", xlab = "bootstrap mean")
abline(v = bci$percent[4:5], col = "darkorange", lty = 2, lwd = 1.5)
abline(v = bci$bca[4:5], col = "forestgreen", lty = 2, lwd = 1.5)
abline(v = mean(skewed), lwd = 2)
legend("topright", c("percentile", "BCa"), col = c("darkorange", "forestgreen"),
       lty = 2, bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "estimation.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 5)
#'
#' 1. Report direction, magnitude, interval and units.
#' 2. Build intervals on the scale where the sampling distribution is symmetric,
#'    then back-transform.
#' 3. To claim equivalence, pre-specify delta and use TOST (5.15).
#' 4. Bootstrap the EXPERIMENTAL UNIT.
#'
#' **Next:** `06_hypothesis_tests_and_pvalues.R`
