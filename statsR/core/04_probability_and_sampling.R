#' ---
#' title: "Module 04 - Probability models and sampling distributions"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 4, equations (4.1)-(4.12)
#'
#' ## What you will learn
#'
#' 1. Variance algebra (4.1)-(4.2) and why **pairing** is so powerful (4.3).
#' 2. The law of total variance (4.4).
#' 3. The CLT (4.5)-(4.6): a statement about the MEAN, not the data.
#' 4. Poisson vs negative binomial (4.7)-(4.9), derived as a gamma-Poisson mixture.
#' 5. Likelihood, score, Fisher information (4.10)-(4.11).
#' 6. Permutation p-values (4.12) and why the `+1` is mandatory.

#+ setup, message = FALSE
MODULE_NAME <- "04_probability_and_sampling"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Variance algebra and the power of pairing, eq. (4.1)-(4.3)
#'
#' $$\operatorname{Var}(D)=2\sigma^2(1-r)\quad\text{for }D=Y_{post}-Y_{pre}$$

#+ pairing
header("1. Why pairing works (4.2)-(4.3)")
set.seed(401)
n_pairs <- 12; sigma <- 1; TRUE_EFFECT <- 0.4
cat(sprintf("%6s %15s %12s %13s %15s\n", "r", "Var(D) theory", "Var(D) sim",
            "paired power", "unpaired power"))
for (r in c(0, 0.3, 0.6, 0.8, 0.95)) {
  theory <- 2 * sigma^2 * (1 - r)                        # eq. (4.3)
  out <- replicate(2000, {
    subj <- rnorm(n_pairs, 0, sqrt(r) * sigma)
    pre  <- subj + rnorm(n_pairs, 0, sqrt(1 - r) * sigma)
    post <- subj + rnorm(n_pairs, 0, sqrt(1 - r) * sigma) + TRUE_EFFECT
    c(var(post - pre),
      t.test(post, pre, paired = TRUE)$p.value,
      t.test(post, pre)$p.value)                         # the WRONG analysis
  })
  cat(sprintf("%6.2f %15.4f %12.4f %12.1f%% %14.1f%%\n", r, theory,
              mean(out[1, ]), 100 * mean(out[2, ] < 0.05),
              100 * mean(out[3, ] < 0.05)))
}
cat("\nSame data, same n. Analysing a paired design as unpaired throws away\n")
cat("the precision you paid for. Pairing comes from the DESIGN (Topic 7).\n")

#' ## 2. Law of total variance, eq. (4.4)

#+ total-variance
header("2. Law of total variance, verified (4.4)")
set.seed(402)
n_donors <- 200; m_cells <- 60; SIGMA_B <- 1.3; SIGMA_E <- 0.8
b <- rnorm(n_donors, 0, SIGMA_B)
y <- matrix(rnorm(n_donors * m_cells, 0, SIGMA_E), nrow = n_donors) + b
total <- var(as.vector(y))
within <- mean(apply(y, 1, var))        # E[Var(Y|Z)]
between <- var(rowMeans(y))             # Var(E[Y|Z])
cat(sprintf("  Var(Y)                     = %.4f\n", total))
cat(sprintf("  E[Var(Y|donor)]  (within)  = %.4f   (truth %.4f)\n", within, SIGMA_E^2))
cat(sprintf("  Var(E[Y|donor])  (between) = %.4f   (truth %.4f)\n", between, SIGMA_B^2))
cat(sprintf("  within + between           = %.4f  <- matches Var(Y)\n", within + between))
cat(sprintf("\n  ICC (eq. 1.7) = %.4f\n", SIGMA_B^2 / (SIGMA_B^2 + SIGMA_E^2)))

#' ## 3. The CLT, eq. (4.5)-(4.6)
#'
#' The CLT is about the SAMPLING DISTRIBUTION OF THE MEAN, not the data.
#' Berry-Esseen (4.6) says the error is O(n^-1/2) and worsens with skewness.

#+ clt
header("3. CLT convergence depends on skewness (4.5)-(4.6)")
ci_coverage <- function(sampler, n, n_sim = 2500, alpha = 0.05) {
  mu_true <- mean(sampler(4e5))
  tcrit <- qt(1 - alpha / 2, n - 1)
  mean(replicate(n_sim, {
    x <- sampler(n)
    abs(mean(x) - mu_true) <= tcrit * sd(x) / sqrt(n)
  }))
}
samplers <- list(
  "Normal (skew 0)"          = function(k) rnorm(k),
  "Exponential (skew 2)"     = function(k) rexp(k),
  "Log-normal (skew ~6)"     = function(k) rlnorm(k),
  "Poisson(0.5) (skew 1.4)"  = function(k) rpois(k, 0.5))
set.seed(403)
cat(sprintf("%-26s%s\n", "distribution",
            paste(sprintf("%9s", paste0("n=", c(5, 10, 30, 100))), collapse = "")))
for (nm in names(samplers)) {
  row <- sapply(c(5, 10, 30, 100), function(n) ci_coverage(samplers[[nm]], n))
  cat(sprintf("%-26s%s\n", nm, paste(sprintf("%8.1f%%", 100 * row), collapse = " ")))
}
cat("\n(nominal coverage is 95.0%)\n")
cat("Symmetric data: fine at n=5. Heavily skewed: still short at n=30.\n")

#' ## 4. Poisson vs negative binomial, eq. (4.7)-(4.9)
#'
#' The NB arises as a **gamma-Poisson mixture**. Using (4.4):
#' `Var(Y) = E[Lambda] + Var(Lambda) = mu + phi*mu^2`.

#+ nb
header("4. NB as a gamma-Poisson mixture (4.8)-(4.9)")
set.seed(404)
MU <- 40; PHI <- 0.16; N <- 4e5
## Build the NB by explicitly compounding, exactly as the derivation says.
lam <- rgamma(N, shape = 1 / PHI, scale = MU * PHI)
y_mix <- rpois(N, lam)
## And from R's own NB parameterisation as a cross-check (size = 1/phi).
y_nb <- rnbinom(N, mu = MU, size = 1 / PHI)
cat(sprintf("  target mean     = %.1f\n", MU))
cat(sprintf("  target variance = mu + phi*mu^2 = %.2f   (eq. 4.9)\n", MU + PHI * MU^2))
cat(sprintf("\n  gamma-Poisson compound : mean %7.3f  var %9.3f\n", mean(y_mix), var(y_mix)))
cat(sprintf("  rnbinom()              : mean %7.3f  var %9.3f\n", mean(y_nb), var(y_nb)))
cat(sprintf("  pure Poisson(mu)       : mean %7.3f  var %9.3f   <- far too small\n",
            MU, MU))
k <- 0:11
cat(sprintf("\n  max |manual eq.(4.8) pmf - dnbinom| = %.3e\n",
            max(abs(exp(lgamma(k + 1/PHI) - lgamma(1/PHI) - lgamma(k + 1) +
                        (1/PHI) * log(1/(1 + MU*PHI)) + k * log(MU*PHI/(1 + MU*PHI))) -
                    dnbinom(k, mu = MU, size = 1/PHI)))))

cat("\nConsequence - testing overdispersed counts with a Poisson model:\n")
set.seed(405)
fp <- mean(replicate(2000, {
  a <- rnbinom(5, mu = MU, size = 1/PHI); b <- rnbinom(5, mu = MU, size = 1/PHI)
  abs((sum(a) - sum(b)) / sqrt(sum(a) + sum(b))) > 1.96   # Poisson SE
}))
cat(sprintf("  Poisson-based false-positive rate on NB data: %.1f%%  (should be 5%%)\n",
            100 * fp))
cat("  This is why RNA-seq needs NB, not Poisson (stats.md Topics 13, 20).\n")

#' ## 5. Likelihood, score, Fisher information, eq. (4.10)-(4.11)

#+ fisher
header("5. Fisher information gives the standard error (4.10)-(4.11)")
set.seed(406)
LAMBDA_TRUE <- 4; n <- 200
y <- rpois(n, LAMBDA_TRUE)
## Poisson: score U = sum(y)/lambda - n  ->  MLE = ybar
##          observed information I = sum(y)/lambda^2
lam_hat <- mean(y)
se_obs <- 1 / sqrt(sum(y) / lam_hat^2)
se_exp <- 1 / sqrt(n / lam_hat)
cat(sprintf("  MLE lambda_hat               = %.4f  (truth %.1f)\n", lam_hat, LAMBDA_TRUE))
cat(sprintf("  SE from observed info (4.11) = %.5f\n", se_obs))
cat(sprintf("  SE from expected info        = %.5f\n", se_exp))
cat(sprintf("  SE = sqrt(lambda_hat/n)      = %.5f   <- all identical\n",
            sqrt(lam_hat / n)))
opt <- optim(log(1), function(p) -sum(dpois(y, exp(p), log = TRUE)),
             method = "BFGS")
cat(sprintf("\n  numerical MLE (optim)        = %.4f\n", exp(opt$par)))
cat(sprintf("  glm() intercept, log link    = %.4f\n",
            exp(coef(glm(y ~ 1, family = poisson())))))
cat("  Every Wald SE printed by glm() is eq. (4.11) at the estimate.\n")

#' ## 6. Permutation tests and the mandatory `+1`, eq. (4.12)

#+ permutation
header("6. Permutation p-values: the +1 is not a fudge (4.12)")
permutation_test <- function(x, y, B = 199, add_one = TRUE) {
  pooled <- c(x, y); n_x <- length(x)
  t_obs <- abs(mean(x) - mean(y))
  cnt <- sum(replicate(B, {
    perm <- sample(pooled)
    abs(mean(perm[seq_len(n_x)]) - mean(perm[-seq_len(n_x)])) >= t_obs
  }))
  if (add_one) (1 + cnt) / (B + 1) else cnt / B
}

set.seed(407)
res <- replicate(1200, {
  a <- rnorm(6); b <- rnorm(6)
  s <- .Random.seed
  p1 <- permutation_test(a, b, 199, TRUE)
  assign(".Random.seed", s, envir = globalenv())
  p2 <- permutation_test(a, b, 199, FALSE)
  c(p1, p2)
})
cat(sprintf("  smallest p WITH the +1    : %.4f  = 1/(B+1)\n", min(res[1, ])))
cat(sprintf("  smallest p WITHOUT the +1 : %.4f  <- 0 is not a p-value\n", min(res[2, ])))
cat("\n  A valid p-value satisfies P(p <= alpha) <= alpha for EVERY alpha.\n")
cat(sprintf("  %8s %12s %12s\n", "alpha", "with +1", "without +1"))
for (a in c(0.05, 0.01, 0.005, 0.001)) {
  cat(sprintf("  %8.3f %12.4f %12.4f\n", a, mean(res[1, ] <= a), mean(res[2, ] <= a)))
}
cat("\n  Below 1/(B+1) the uncorrected version keeps rejecting while the true\n")
cat("  attainable resolution has run out - anticonservative exactly in the\n")
cat("  extreme tail, which is where genome-wide thresholds live (Topic 8).\n")

## Exactness: with n small enough, enumerate ALL splits.
a <- c(5.1, 4.8, 6.2, 5.5); b <- c(6.9, 7.3, 6.5, 7.8)
pooled <- c(a, b); t_obs <- abs(mean(a) - mean(b))
all_stats <- combn(8, 4, function(idx) abs(mean(pooled[idx]) - mean(pooled[-idx])))
cat(sprintf("\n  EXACT permutation p (all C(8,4)=%d splits) = %.4f\n",
            length(all_stats), mean(all_stats >= t_obs)))
cat(sprintf("  Welch t-test p for comparison             = %.4f\n",
            t.test(a, b)$p.value))
cat("  With 4 vs 4 the smallest achievable p is 2/70 = 0.0286 - a hard floor\n")
cat("  imposed by the DESIGN, not by the effect size.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "probability.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))

set.seed(408)
plot(NA, xlim = c(-4, 4), ylim = c(0, 0.55), xlab = "standardised mean",
     ylab = "density", main = "Eq. (4.5): CLT for log-normal data")
cols <- c("steelblue", "darkorange", "forestgreen")
for (i in seq_along(c(1, 5, 30))) {
  nn <- c(1, 5, 30)[i]
  means <- rowMeans(matrix(rlnorm(20000 * nn), ncol = nn))
  z <- (means - mean(means)) / sd(means)
  lines(density(z, from = -4, to = 4), col = cols[i], lwd = 2)
}
curve(dnorm(x), add = TRUE, lty = 2)
legend("topleft", c("n=1", "n=5", "n=30", "N(0,1)"), col = c(cols, "black"),
       lty = c(1, 1, 1, 2), lwd = 2, bty = "n", cex = 0.7)

k <- 0:120
plot(k, dpois(k, 40), type = "l", col = "steelblue", lwd = 2, xlab = "count",
     ylab = "probability", main = "Eq. (4.7)-(4.9): overdispersion")
lines(k, dnbinom(k, mu = 40, size = 1/0.16), col = "firebrick", lwd = 2)
legend("topright", c("Poisson(40)", "NB(mu=40, phi=0.16)"),
       col = c("steelblue", "firebrick"), lwd = 2, bty = "n", cex = 0.7)

set.seed(409)
x0 <- rnorm(10); y0 <- rnorm(10, 1.2)
pooled <- c(x0, y0)
null_stats <- replicate(5000, { p <- sample(pooled); mean(p[1:10]) - mean(p[11:20]) })
hist(null_stats, breaks = 50, col = "lightsteelblue", border = "white",
     main = "Eq. (4.12): permutation null",
     xlab = "difference in means under random labelling")
abline(v = mean(x0) - mean(y0), col = "red", lwd = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "probability.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 4)
#'
#' 1. Name the distribution and the source of randomness before computing a SE.
#' 2. Check overdispersion: if `s^2 >> ybar` for counts, Poisson inference is
#'    anticonservative.
#' 3. Prefer permutation when the design justifies exchangeability and n is
#'    small; prefer likelihood when a parametric mean-variance model is
#'    credible and G is large enough to borrow strength (Module 33).
#'
#' **Next:** `05_estimation_and_intervals.R`
