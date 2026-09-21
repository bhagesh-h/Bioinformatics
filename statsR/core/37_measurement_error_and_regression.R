#' ---
#' title: "Module 37 - Measurement error and regression to the mean"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 37, equations (37.1)-(37.6)
#' **Assumes:** Modules 11, 31.
#'
#' ## What you will learn
#'
#' 1. Noise in a **predictor** biases its coefficient toward zero (37.2)-(37.3).
#'    Noise in the **outcome** does not. The asymmetry decides where to spend
#'    assay budget.
#' 2. How to correct for it when reliability is known (37.4), and why the
#'    correction inflates the standard error too.
#' 3. That with several predictors, error in one biases the coefficients of the
#'    others, in either direction.
#' 4. **Regression to the mean** (37.5)-(37.6): selecting on an extreme value
#'    guarantees an apparent effect with no treatment and no biology.

#+ setup, message = FALSE
MODULE_NAME <- "37_measurement_error_and_regression"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(26)
slope <- function(x, y) unname(coef(lm(y ~ x))[2])

#' ## 1. Noise in the predictor attenuates, eq. (37.1)-(37.3)
#'
#' $$\lambda=\frac{\sigma_X^2}{\sigma_X^2+\sigma_U^2},\qquad
#'   \mathbb{E}[\hat\beta_{obs}]=\lambda\beta \qquad (37.2)-(37.3)$$

#+ attenuation
header("1. Error in X shrinks the slope; error in Y does not")
N <- 4000; BETA <- 2.0; SD_X <- 1.0; SD_E <- 1.0
cat(sprintf("  true slope = %.1f\n\n", BETA))
cat(sprintf("  %-12s%15s%13s%13s%12s\n", "noise SD", "lambda (37.2)",
            "noise in X", "noise in Y", "predicted"))
for (sd_u in c(0.0, 0.5, 1.0, 2.0)) {
  bx <- by <- numeric(300)
  for (i in 1:300) {
    x <- rnorm(N, 0, SD_X)
    y <- BETA * x + rnorm(N, 0, SD_E)
    bx[i] <- slope(x + rnorm(N, 0, sd_u), y)     # error in X
    by[i] <- slope(x, y + rnorm(N, 0, sd_u))     # error in Y
  }
  lam <- SD_X^2 / (SD_X^2 + sd_u^2)              # eq. (37.2)
  cat(sprintf("  %-12.1f%15.3f%13.3f%13.3f%12.3f\n", sd_u, lam, mean(bx),
              mean(by), lam * BETA))
}
cat("\n  The 'noise in X' column matches the prediction column exactly: the
  observed slope is lambda times the true slope, eq. (37.3). The 'noise in Y'
  column does not move at all.

  That asymmetry is worth internalising. If your predictor is an assay readout
  and your outcome is a clean clinical endpoint, your effect size is
  systematically too small and no amount of data fixes it. If it is the other
  way round, you have lost precision but not accuracy.\n")

#' ## 2. What attenuation does to inference

#+ inference
header("2. Attenuation does not just shrink the estimate")
sd_u <- 1.0
lam <- SD_X^2 / (SD_X^2 + sd_u^2)
SMALL <- 0.25; R <- 600
b_clean <- b_noisy <- numeric(R); pw_clean <- pw_noisy <- 0
for (i in 1:R) {
  x <- rnorm(200, 0, SD_X)
  y <- SMALL * x + rnorm(200, 0, SD_E)
  w <- x + rnorm(200, 0, sd_u)
  sc <- summary(lm(y ~ x))$coefficients
  sn <- summary(lm(y ~ w))$coefficients
  b_clean[i] <- sc[2, 1]; b_noisy[i] <- sn[2, 1]
  pw_clean <- pw_clean + (sc[2, 4] < 0.05)
  pw_noisy <- pw_noisy + (sn[2, 4] < 0.05)
}
cat(sprintf("  true effect %.2f, reliability lambda = %.2f, n = 200\n\n",
            SMALL, lam))
cat(sprintf("  %-22s%15s%9s\n", "predictor", "mean estimate", "power"))
cat(sprintf("  %-22s%15.3f%9.2f\n", "measured without error", mean(b_clean),
            pw_clean / R))
cat(sprintf("  %-22s%15.3f%9.2f\n", "measured with error", mean(b_noisy),
            pw_noisy / R))
cat("\n  Attenuation costs you twice. The estimate shrinks toward zero, AND the
  power to detect it falls, because the signal has been diluted while the
  outcome noise stayed put.

  A study that reports \"no significant association\" with an assay-based
  predictor may have had a real association it could not see. Report the
  assay's reliability so a reader can judge.\n")

#' ## 3. Correcting with a known reliability, eq. (37.4)
#'
#' $$\hat\beta_{corr}=\hat\beta_{obs}/\hat\lambda,\qquad
#'   SE_{corr}=SE_{obs}/\hat\lambda \qquad (37.4)$$
#'
#' Replicate measurements give you $\lambda$, and $\lambda$ gives you the
#' correction. The corrected estimate is unbiased; it is also less precise,
#' and the interval must say so.

#+ correction
header("3. Correcting for attenuation (37.4)")
sd_u <- 1.0
lam_true <- SD_X^2 / (SD_X^2 + sd_u^2)
ests <- cors <- numeric(1000); cov_naive <- cov_corr <- 0
for (i in 1:1000) {
  x <- rnorm(400, 0, SD_X)
  y <- BETA * x + rnorm(400, 0, SD_E)
  w1 <- x + rnorm(400, 0, sd_u)          # two replicate measurements of x
  w2 <- x + rnorm(400, 0, sd_u)
  lam_hat <- cor(w1, w2)                 # reliability estimated, not assumed
  sm_ <- summary(lm(y ~ w1))$coefficients
  b <- sm_[2, 1]; se <- sm_[2, 2]
  ests[i] <- b; cors[i] <- b / lam_hat
  cov_naive <- cov_naive + (abs(b - BETA) < 1.96 * se)
  cov_corr <- cov_corr + (abs(b / lam_hat - BETA) < 1.96 * se / lam_hat)
}
cat(sprintf("  true slope %.1f, true lambda %.3f\n\n", BETA, lam_true))
cat(sprintf("  %-34s%9s%9s%18s\n", "estimator", "mean", "bias",
            "95% CI coverage"))
cat(sprintf("  %-34s%9.3f%9.3f%17.1f%%\n", "naive (ignore the error)",
            mean(ests), mean(ests) - BETA, 100 * cov_naive / 1000))
cat(sprintf("  %-34s%9.3f%9.3f%17.1f%%\n", "corrected by lambda-hat (37.4)",
            mean(cors), mean(cors) - BETA, 100 * cov_corr / 1000))
cat("\n  The naive interval has essentially zero coverage: it is a tight
  interval in the wrong place, which is the worst combination. Dividing by
  lambda recovers the truth, and dividing the standard error by lambda as well
  brings coverage most of the way back.

  It does not reach 95%, and the shortfall is informative. The corrected
  interval treats lambda-hat as though it were the known lambda, so it ignores
  the uncertainty in the reliability estimate itself. Problem 3 shows what
  happens when that uncertainty is large.

  Note what supplied lambda: a second measurement of the same samples. Without
  replicates you cannot estimate reliability, and without reliability you
  cannot correct. Build replicates into the design.\n")

#' ## 4. With several predictors, it is not just shrinkage

#+ multi
header("4. Error in one predictor biases the others")
cat("  Two correlated predictors. Only X1 is measured with error.\n\n")
cat(sprintf("  %-14s%11s%11s%11s%11s\n", "corr(X1,X2)", "beta1 true",
            "beta1 est", "beta2 true", "beta2 est"))
for (rho in c(0.0, 0.5, 0.8)) {
  b1s <- b2s <- numeric(400)
  for (i in 1:400) {
    z1 <- rnorm(1500); z2 <- rnorm(1500)
    x1 <- z1
    x2 <- rho * z1 + sqrt(1 - rho^2) * z2
    y <- 1.0 * x1 + 0.0 * x2 + rnorm(1500)
    w1 <- x1 + rnorm(1500)                      # only X1 is noisy
    cf <- coef(lm(y ~ w1 + x2))
    b1s[i] <- cf[2]; b2s[i] <- cf[3]
  }
  cat(sprintf("  %-14.1f%11.2f%11.3f%11.2f%11.3f\n", rho, 1.0, mean(b1s), 0.0,
              mean(b2s)))
}
cat("\n  X2 is measured perfectly and has NO true effect, yet its coefficient
  grows as the correlation rises. The model, unable to see the true X1, uses
  X2 as a partial proxy for it.

  So \"measurement error only attenuates\" is true for simple regression and
  false the moment you adjust for anything. A clean covariate can absorb the
  effect of a noisy one and appear important. This is a real mechanism behind
  spurious 'independent predictors' in clinical models.\n")

#' ## 5. Regression to the mean, eq. (37.5)-(37.6)
#'
#' $$\mathbb{E}[Y_2\mid Y_1=y_1]=\mu+\rho(y_1-\mu) \qquad (37.5)$$
#' $$\mathbb{E}[Y_2-Y_1\mid Y_1=y_1]=(\rho-1)(y_1-\mu) \qquad (37.6)$$
#'
#' No treatment, no biology, no effect. Select on an extreme baseline and the
#' follow-up moves toward the mean by an amount you can predict exactly.

#+ rtm
header("5. Regression to the mean (37.5)-(37.6)")
MU <- 100.0; SD <- 15.0; n <- 20000
cat(sprintf("  A biomarker with mean %.0f and SD %.0f, measured twice.\n", MU, SD))
cat("  NOTHING happens between the measurements.\n\n")
cat(sprintf("  %-18s%19s%18s%19s\n", "reliability rho", "selected top 10%",
            "observed change", "predicted (37.6)"))
for (rho in c(0.9, 0.7, 0.5, 0.3)) {
  true_ <- rnorm(n, MU, SD * sqrt(rho))
  y1 <- true_ + rnorm(n, 0, SD * sqrt(1 - rho))
  y2 <- true_ + rnorm(n, 0, SD * sqrt(1 - rho))
  sel <- y1 >= quantile(y1, 0.90)
  obs <- mean(y2[sel] - y1[sel])
  pred <- (rho - 1) * (mean(y1[sel]) - MU)                # eq. (37.6)
  cat(sprintf("  %-18.1f%19.1f%18.2f%19.2f\n", rho, mean(y1[sel]), obs, pred))
}
cat("\n  The observed change matches eq. (37.6) at every reliability. A study
  that enrolled the top decile and reported an average drop would be reporting
  arithmetic, not pharmacology.

  Note the direction of the dependence: the effect is LARGEST when rho is
  SMALL, that is, when the assay is noisiest. Poor measurement does not merely
  add noise here, it creates a systematic apparent effect.

  The fix is a control group selected the same way. Both arms regress by the
  same amount and the difference between them is clean. A single-arm
  before-and-after study in a selected group cannot be rescued by analysis.\n")

#' ## 6. The same trap inside a model: adjusting for baseline

#+ ancova
header("6. Change scores versus adjusting for baseline")
R <- 500; n <- 300
ch <- an <- numeric(R)
for (i in 1:R) {
  true_ <- rnorm(n, MU, SD * sqrt(0.6))
  y1 <- true_ + rnorm(n, 0, SD * sqrt(0.4))
  arm <- rbinom(n, 1, 0.5)
  # A real treatment effect of -3, plus regression to the mean for everyone.
  y2 <- true_ + rnorm(n, 0, SD * sqrt(0.4)) - 3.0 * arm
  ch[i] <- coef(lm((y2 - y1) ~ arm))[2]
  an[i] <- coef(lm(y2 ~ arm + y1))[2]
}
cat("  true treatment effect = -3.00, randomised arms\n\n")
cat(sprintf("  %-40s%11s%9s\n", "analysis", "estimate", "SD"))
cat(sprintf("  %-40s%11.3f%9.3f\n", "change score (y2 - y1)", mean(ch), sd(ch)))
cat(sprintf("  %-40s%11.3f%9.3f\n", "ANCOVA: y2 ~ arm + y1", mean(an), sd(an)))
cat("\n  Both are unbiased here, because the arms were RANDOMISED, so baseline
  imbalance is only chance. ANCOVA is more precise, and that is the standard
  reason to prefer it.

  The picture changes entirely if allocation depended on the baseline. Then
  the change score is biased by regression to the mean and ANCOVA is not, and
  the gap can be large. Randomise, and adjust for baseline in the model rather
  than selecting on it.\n")

#' ## 7. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "measurement_error.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
sds <- seq(0, 2.5, length.out = 20)
lams <- SD_X^2 / (SD_X^2 + sds^2)
plot(sds, lams * BETA, type = "l", lwd = 2, col = "steelblue", ylim = c(0, 2.2),
     xlab = "measurement error SD in X", ylab = "expected slope",
     main = "Attenuation, eq. (37.3)")
abline(h = BETA, lty = 2, col = "grey40")
obs <- sapply(sds, function(s) {
  x <- rnorm(3000, 0, SD_X); y <- BETA * x + rnorm(3000, 0, SD_E)
  slope(x + rnorm(3000, 0, s), y) })
points(sds, obs, pch = 16, cex = 0.7, col = "firebrick")
legend("topright", c("predicted", "simulated"), bty = "n", lwd = c(2, NA),
       pch = c(NA, 16), col = c("steelblue", "firebrick"))
rho <- 0.5
true_ <- rnorm(4000, MU, SD * sqrt(rho))
y1 <- true_ + rnorm(4000, 0, SD * sqrt(1 - rho))
y2 <- true_ + rnorm(4000, 0, SD * sqrt(1 - rho))
plot(y1, y2, pch = 16, cex = 0.25, col = adjustcolor("grey40", 0.3),
     xlab = "first measurement", ylab = "second measurement",
     main = "Regression to the mean (rho = 0.5)")
abline(0, 1, lty = 2); abline(lm(y2 ~ y1), col = "firebrick", lwd = 2)
sel <- y1 >= quantile(y1, 0.9)
points(y1[sel], y2[sel], pch = 16, cex = 0.3, col = adjustcolor("firebrick", .5))
rhos <- seq(0.1, 0.95, length.out = 18)
appar <- sapply(rhos, function(rr) {
  t_ <- rnorm(20000, MU, SD * sqrt(rr))
  a <- t_ + rnorm(20000, 0, SD * sqrt(1 - rr))
  b <- t_ + rnorm(20000, 0, SD * sqrt(1 - rr))
  s <- a >= quantile(a, 0.9); mean(b[s] - a[s]) })
plot(rhos, appar, type = "b", pch = 16, col = "firebrick", lwd = 2,
     xlab = "assay reliability rho", ylab = "apparent 'improvement'",
     main = "Noisier assay, bigger fake effect")
abline(h = 0, lty = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "measurement_error.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Where should the assay budget go?
#'
#' You can afford to halve the measurement error on either the predictor or
#' the outcome, but not both. Which buys more, and does the answer depend on
#' what you are trying to do?

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# study <- function(sd_ux, sd_uy, beta = 0.4, n = 250, R = 800) {
#   est <- numeric(R); hit <- 0
#   for (i in 1:R) {
#     x <- rnorm(n); y <- beta * x + rnorm(n)
#     s <- summary(lm((y + rnorm(n, 0, sd_uy)) ~ I(x + rnorm(n, 0, sd_ux))))
#     est[i] <- s$coefficients[2, 1]; hit <- hit + (s$coefficients[2, 4] < 0.05)
#   }
#   c(mean(est), hit / R)
# }
# cat("  true beta = 0.40\n\n")
# cat(sprintf("  %-34s%11s%9s%9s\n", "scenario", "estimate", "bias", "power"))
# for (z in list(list("both assays noisy", 1.0, 1.0),
#                list("halve error in the PREDICTOR", 0.5, 1.0),
#                list("halve error in the OUTCOME", 1.0, 0.5))) {
#   r_ <- study(z[[2]], z[[3]])
#   cat(sprintf("  %-34s%11.3f%9.3f%9.2f\n", z[[1]], r_[1], r_[1] - 0.4, r_[2]))
# }
#
# ## Improving the PREDICTOR reduces bias and raises power. Improving the
# ## OUTCOME leaves the estimate where it was and raises power too, sometimes
# ## by more, because outcome noise sits directly in the residual variance.
# ##
# ## So the answer depends on the goal, which is the real lesson:
# ##   if you want an unbiased EFFECT SIZE, spend on the predictor;
# ##   if you only want to DETECT the effect, spend wherever the variance is
# ##   larger, which is often the outcome.
# ##
# ## Most papers want the effect size and most budgets are spent on the outcome.

#' ### Problem 2: Manufacture a treatment effect from nothing
#'
#' Design a single-arm study that reports a large, highly significant
#' improvement for a treatment that does nothing at all. Then show which
#' design change removes it.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# n2 <- 400; rho2 <- 0.5
# t_ <- rnorm(n2, MU, SD * sqrt(rho2))
# base <- t_ + rnorm(n2, 0, SD * sqrt(1 - rho2))
# after <- t_ + rnorm(n2, 0, SD * sqrt(1 - rho2))    # NO treatment effect
# elig <- base >= quantile(base, 0.80)               # "enrol the worst 20%"
# d <- after[elig] - base[elig]
# tt <- t.test(d)
# cat(sprintf("  Single-arm trial, enrolled the worst 20%% at baseline, n = %d\n",
#             sum(elig)))
# cat(sprintf("    mean change    = %+.2f\n", mean(d)))
# cat(sprintf("    paired t-test  p = %.2e\n", tt$p.value))
# cat(sprintf("    Cohen's d      = %.2f\n", mean(d) / sd(d)))
# cat("    TRUE effect    = 0.00\n")
# arm <- rbinom(n2, 1, 0.5)
# d_all <- after - base
# ctl <- d_all[elig & arm == 0]; trt <- d_all[elig & arm == 1]
# tt2 <- t.test(trt, ctl)
# cat("\n  Same data, randomised control selected the same way:\n")
# cat(sprintf("    treated change = %+.2f, control change = %+.2f\n",
#             mean(trt), mean(ctl)))
# cat(sprintf("    difference     = %+.2f, p = %.3f\n", mean(trt) - mean(ctl),
#             tt2$p.value))
#
# ## The single-arm study reports a large, overwhelmingly significant
# ## improvement for a treatment with no effect. Every number in it is correct;
# ## the design is what is wrong.
# ##
# ## The control arm regresses by the same amount, so the difference is clean.
# ## This is why single-arm before-and-after studies in selected patients are
# ## weak evidence however large the p-value, and it is the mechanism behind a
# ## great many "promising pilot study" results that later fail.

#' ### Problem 3: Does correcting for attenuation ever hurt?
#'
#' The correction in (37.4) divides by an estimated $\lambda$. Investigate
#' what happens when $\hat\lambda$ is itself noisy, as it is with few
#' replicates.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  true slope %.1f, true lambda %.3f\n\n", BETA, lam_true))
# cat(sprintf("  %-30s%16s%9s%12s\n", "n used to estimate lambda",
#             "mean corrected", "SD", "worst run"))
# N_STUDY <- 2000                 # the main study, held fixed across rows
# for (n_rel in c(20, 50, 200, 2000)) {
#   cor_ <- numeric(600)
#   for (i in 1:600) {
#     x <- rnorm(N_STUDY, 0, SD_X)
#     y <- BETA * x + rnorm(N_STUDY, 0, SD_E)
#     w1 <- x + rnorm(N_STUDY, 0, 1.0)
#     idx <- sample(N_STUDY, n_rel)      # replicate subset of only n_rel samples
#     w2 <- x[idx] + rnorm(n_rel, 0, 1.0)
#     lam_hat <- min(max(cor(w1[idx], w2), 0.05), 1.0)
#     cor_[i] <- coef(lm(y ~ w1))[2] / lam_hat
#   }
#   cat(sprintf("  %-30d%16.3f%9.3f%12.2f\n", n_rel, mean(cor_), sd(cor_),
#               max(cor_)))
# }
#
# ## With plenty of replicate pairs the correction is well behaved. With few,
# ## it becomes unstable and skewed upward, because lambda-hat appears in a
# ## DENOMINATOR: a lambda that happens to come out small produces an enormous
# ## corrected slope. The mean is dragged by those runs and the worst run can
# ## be far from the truth.
# ##
# ## This is a general hazard of ratio estimators and the reason to prefer
# ## methods that model the error directly, such as SIMEX, structural equation
# ## models, or a Bayesian measurement-error model, over dividing by a noisy
# ## constant. It is also an argument for estimating reliability from a
# ## dedicated, adequately sized reproducibility experiment rather than from a
# ## handful of incidental duplicates.

#' ## What to take away
#'
#' 1. Noise in a **predictor** biases toward zero by $\lambda$ (37.2)-(37.3).
#'    Noise in the **outcome** does not bias at all.
#' 2. Attenuation costs estimate and power together, so "not significant" with
#'    a noisy predictor is weak evidence of no effect.
#' 3. Correct with (37.4) only if reliability came from real replicates, and
#'    inflate the standard error by the same factor.
#' 4. With multiple predictors, error in one distorts the others. A clean
#'    covariate can steal a noisy one's effect.
#' 5. **Selecting on an extreme baseline guarantees an apparent effect**, and
#'    the effect is larger the noisier the assay. Only a control group
#'    selected the same way removes it.
#'
#' **Next:** `38_meta_analysis.R`
