#' ---
#' title: "Foundations 4 - Estimates, standard errors, and the central limit theorem"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Part 0, §0.6, equations (0.11)-(0.12)
#' **Assumes:** `F1`-`F3`.
#'
#' ## The question this module answers
#'
#' You computed an average. **How much should you trust it?**
#'
#' The answer comes from a thought experiment you can never perform but can
#' always simulate: *if I ran this whole experiment again, how different would
#' the answer be?* That spread is the **standard error**, and remarkably it can
#' be worked out from the single sample you actually have.

#+ setup, message = FALSE
MODULE_NAME <- "F4_estimates_and_standard_errors"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(104)
TRUE_MU <- 20; TRUE_SIGMA <- 4

#' ## 1. Estimator vs estimate

#+ estimator
header("1. The recipe and the number it produced")
smp <- rnorm(12, TRUE_MU, TRUE_SIGMA)
cat("  ESTIMATOR: 'take the average of the sample'   <- a recipe\n")
cat(sprintf("  ESTIMATE : %.4f                        <- what it gave, this time\n",
            mean(smp)))
cat(sprintf("  TRUTH    : %.4f                        <- unknown in real life\n", TRUE_MU))
cat("
  The distinction matters because PROPERTIES BELONG TO THE RECIPE, not to the
  number. It is meaningless to ask whether 19.87 is unbiased. It is very
  meaningful to ask whether 'take the average' is unbiased - and it is.

  So when someone says 'the sample mean is unbiased with standard error
  sigma/sqrt(n)', they are describing the recipe's behaviour across all the
  experiments you might have run, not the number on your screen.\n")

#' ## 2. The sampling distribution

#+ sampling
header("2. The distribution of an estimate across repeat experiments")
N <- 12
estimates <- replicate(50000, mean(rnorm(N, TRUE_MU, TRUE_SIGMA)))
true_se <- TRUE_SIGMA/sqrt(N)
cat(sprintf("  50,000 repeats of a %d-sample experiment:\n\n", N))
cat(sprintf("  %-34s%10.4f   (truth = %g)\n", "centre of the estimates",
            mean(estimates), TRUE_MU))
cat(sprintf("  %-34s%10.4f\n", "spread of the estimates (SD)", sd(estimates)))
cat(sprintf("  %-34s%10.4f   <- matches\n", "sigma / sqrt(n), eq. (0.11)", true_se))
cat(sprintf("\n  %-34s%10.3f\n", "how often within 1 SE of truth",
            mean(abs(estimates - TRUE_MU) < true_se)))
cat(sprintf("  %-34s%10.3f\n", "how often within 2 SE of truth",
            mean(abs(estimates - TRUE_MU) < 2*true_se)))
cat("
  THE SPREAD OF THIS DISTRIBUTION IS THE STANDARD ERROR. That is the whole
  definition. Everything else is machinery for estimating it without being
  able to run the experiment 50,000 times.\n")

#' ## 3. SD and SE are different things

#+ sd-se
header("3. Standard DEVIATION vs standard ERROR")
cat(sprintf("  %-8s%22s%18s%10s\n", "n", "SD of the data (s)", "SE of the mean", "ratio"))
for (n in c(5, 20, 100, 500, 2000)) {
  sds <- replicate(4000, sd(rnorm(n, TRUE_MU, TRUE_SIGMA)))
  ses <- sd(replicate(4000, mean(rnorm(n, TRUE_MU, TRUE_SIGMA))))
  cat(sprintf("  %-8d%22.4f%18.4f%10.2f\n", n, mean(sds), ses, mean(sds)/ses))
}
cat(sprintf("
  Read DOWN the two columns. The SD column does not move - it is estimating
  sigma = %g, a fact about how much individual observations vary, and no
  amount of data changes that. The SE column shrinks steadily, because it
  describes how well you know the MEAN.

    s  (standard deviation) : how spread out the OBSERVATIONS are.
                              Does not shrink with n. Use it to DESCRIBE.

    SE (standard error)     : how spread out your ESTIMATE would be across
                              repeat experiments. Shrinks as sqrt(n).
                              Use it to say how PRECISELY you know something.

  An error bar in a figure is uninterpretable unless the caption says which
  one it is. SE bars are always shorter, which is presumably why they are more
  popular.\n", TRUE_SIGMA))

#' ## 4. You only have one sample: and that is enough

#+ onesample
header("4. Estimating the standard error from a single sample")
cat(sprintf("  %-8s%14s%11s%15s%10s\n", "trial", "sample mean", "sample s",
            "estimated SE", "true SE"))
for (trial in 1:6) {
  s_ <- rnorm(N, TRUE_MU, TRUE_SIGMA)
  cat(sprintf("  %-8d%14.3f%11.3f%15.3f%10.3f\n", trial, mean(s_), sd(s_),
              sd(s_)/sqrt(N), true_se))
}
cat(sprintf("
  Each trial's estimated SE is close to the truth (%.3f) without ever
  repeating the experiment. That is what makes statistics practical: the
  variability WITHIN your single sample tells you how much your ESTIMATE would
  vary BETWEEN samples.

  Why does that work? Because of eq. (0.9) - variances of independent things
  add. Nothing else. And that is also its weak point: if your observations are
  not independent, s/sqrt(n) estimates the wrong thing entirely, and no amount
  of care elsewhere repairs it (F2, section 3).\n", true_se))

#' ## 5. The central limit theorem

#+ clt
header("5. Averages become bell-shaped whatever the source")
skewness <- function(v) mean((v - mean(v))^3)/sd(v)^3
sources <- list(
  "uniform (flat)"          = function(k, r) replicate(r, mean(runif(k))),
  "exponential (skewed)"    = function(k, r) replicate(r, mean(rexp(k))),
  "binary coin flips"       = function(k, r) replicate(r, mean(rbinom(k, 1, 0.5))),
  "lognormal (very skewed)" = function(k, r) replicate(r, mean(rlnorm(k, 0, 1.5))))
cat("  Skewness of the AVERAGE (0 = perfectly symmetric):\n\n")
cat(sprintf("  %-26s", "source distribution"))
for (k in c(1, 2, 5, 30, 200)) cat(sprintf("%9s", paste0("n=", k)))
cat("\n")
for (nm in names(sources)) {
  cat(sprintf("  %-26s", nm))
  for (k in c(1, 2, 5, 30, 200)) cat(sprintf("%9.2f", skewness(sources[[nm]](k, 8000))))
  cat("\n")
}
cat("
  Every row marches towards zero. Flat data, skewed data, data that can only
  be 0 or 1 - average enough of it and you get a bell curve.

  This is why methods built on the normal distribution work far more often
  than they have any right to: they are not assuming your DATA is normal, they
  are assuming your ESTIMATE is, and that is a much weaker requirement.

  But read the table again. The lognormal row is still skewed at n = 30. 'n =
  30 is enough' is folklore. The nastier the source, the larger n must be, and
  with small samples of skewed data you should use a permutation test or a
  bootstrap instead of trusting the approximation (Topics 5 and 7).\n")

#' ## 6. The tyranny of the square root

#+ sqrt
header("6. Why precision is expensive")
cat(sprintf("  %-10s%10s%30s\n", "n", "SE", "to halve the SE from here"))
for (n in c(10, 40, 160, 640, 2560))
  cat(sprintf("  %-10d%10.4f%30s\n", n, TRUE_SIGMA/sqrt(n), paste("n =", 4*n)))
cat("
  Each row costs four times the previous one and buys a halving. To improve
  precision tenfold you need a hundred times the data.

  Two practical consequences:

    * Diminishing returns are severe. Going from n = 10 to n = 40 is
      transformative; from n = 1000 to n = 4000 is usually not worth it.

    * BETTER DESIGN BEATS MORE DATA. Removing a source of noise - pairing,
      blocking, a better assay, adjusting for a known covariate - reduces
      sigma directly rather than fighting the square root. That is why
      Topic 9 (power) is about design, not about sample size tables.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "standard_errors.png"), width = 1250, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.4, 4.4, 2.6, 1))
hist(estimates, breaks = 70, col = "steelblue", border = NA, freq = FALSE,
     main = sprintf("Sampling distribution (n = %d)", N), xlab = "sample mean")
abline(v = TRUE_MU, col = "firebrick", lwd = 2)
abline(v = TRUE_MU + c(-1, 1)*true_se, lty = 2)
ns <- c(2, 5, 10, 25, 50, 100, 250, 500, 1000)
sd_line <- sapply(ns, function(n) mean(replicate(1500, sd(rnorm(n, TRUE_MU, TRUE_SIGMA)))))
se_line <- sapply(ns, function(n) sd(replicate(1500, mean(rnorm(n, TRUE_MU, TRUE_SIGMA)))))
plot(ns, sd_line, type = "b", pch = 16, col = "darkorange", log = "x",
     ylim = c(0, max(sd_line)*1.1), xlab = "n", ylab = "",
     main = "SD stays put; SE shrinks")
lines(ns, se_line, type = "b", pch = 16, col = "steelblue")
lines(ns, TRUE_SIGMA/sqrt(ns), lty = 2)
legend("right", c("SD of the data", "SE of the mean", "sigma/sqrt(n)"),
       col = c("darkorange", "steelblue", "black"), lty = c(1, 1, 2), lwd = 2,
       bty = "n", cex = 0.7)
plot(density(scale(replicate(20000, mean(rexp(1))))), col = "firebrick", lwd = 2,
     xlim = c(-4, 4), main = "CLT: averaging skewed data", xlab = "standardised mean")
for (i in seq_along(c(2, 30))) {
  k <- c(2, 30)[i]
  lines(density(scale(replicate(20000, mean(rexp(k))))),
        col = c("darkorange", "steelblue")[i], lwd = 2)
}
curve(dnorm(x), add = TRUE, lty = 2, lwd = 1.5)
legend("topright", c("n=1", "n=2", "n=30", "normal"),
       col = c("firebrick", "darkorange", "steelblue", "black"),
       lty = c(1, 1, 1, 2), lwd = 2, bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "standard_errors.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Why divide by $n-1$?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  true variance = %.2f\n\n", TRUE_SIGMA^2))
# cat(sprintf("  %-8s%16s%18s%20s\n", "n", "divide by n", "divide by n-1",
#             "which is right?"))
# for (n in c(2, 3, 5, 10, 50)) {
#   v_n1 <- mean(replicate(20000, var(rnorm(n, TRUE_MU, TRUE_SIGMA))))
#   ## R's var() already divides by n-1; multiply back to get the n version.
#   v_n <- v_n1*(n - 1)/n
#   cat(sprintf("  %-8d%16.3f%18.3f%20s\n", n, v_n, v_n1, "n-1"))
# }
#
# ## Dividing by n gives a variance that is too SMALL, badly so at small n (at
# ## n = 2 it is half the truth). Dividing by n-1 is right on average.
# ##
# ## The reason: you measure spread around the SAMPLE mean, not the true mean,
# ## and the sample mean sits in the middle of your own data by construction -
# ## closer to it than the true mean would be. So distances from it are
# ## systematically too small, and n-1 corrects exactly for that.
# ##
# ## The n-1 is the "degrees of freedom": you had n numbers, but once the mean
# ## is fixed only n-1 are free to vary. The same idea reappears every time you
# ## fit a model - p parameters leave n-p degrees of freedom, which is why
# ## fitting many parameters to little data leaves you unable to estimate the
# ## noise at all.

#' ### Problem 2: What does non-independence do to the standard error?

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# n_units <- 5; per_unit <- 10
# cat(sprintf("  50 measurements arranged as %d units x %d sub-samples\n\n",
#             n_units, per_unit))
# cat(sprintf("  %-8s%15s%10s%20s%14s\n", "ICC", "design effect", "TRUE SE",
#             "s/sqrt(50) claims", "too small by"))
# for (icc in c(0, 0.1, 0.3, 0.6)) {
#   sd_u <- sqrt(icc); sd_e <- sqrt(1 - icc)
#   means <- numeric(6000); claimed <- numeric(6000)
#   for (i in 1:6000) {
#     u <- rnorm(n_units, 0, sd_u)
#     y <- rep(u, each = per_unit) + rnorm(n_units*per_unit, 0, sd_e)
#     means[i] <- mean(y); claimed[i] <- sd(y)/sqrt(length(y))
#   }
#   de <- 1 + (per_unit - 1)*icc
#   cat(sprintf("  %-8.1f%15.2f%10.4f%20.4f%13.2fx\n", icc, de, sd(means),
#               mean(claimed), sd(means)/mean(claimed)))
# }
#
# ## At ICC = 0 the formula is right. As the correlation grows it understates
# ## the uncertainty by roughly sqrt(design effect), where
# ##
# ##     design effect = 1 + (m - 1) x ICC          (eq. 1.8)
# ##
# ## and m is the number of sub-samples per unit. Note that m matters as much
# ## as the ICC: 20 cells per mouse with ICC 0.1 is as damaging as 3 cells per
# ## mouse with ICC 0.9.
# ##
# ## The number of rows in your table is not n. The number of INDEPENDENT UNITS
# ## is n, and that is the most valuable sentence in this course.

#' ### Problem 3: Bootstrap: a standard error without a formula

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# dat3 <- rlnorm(60, 3, 0.8)          # one skewed sample, as in real life
# bootstrap_se <- function(x, statistic, B = 4000) {
#   ## Resample x WITH REPLACEMENT B times; the spread of the statistic across
#   ## those resamples estimates its standard error.
#   sd(replicate(B, statistic(sample(x, length(x), replace = TRUE))))
# }
# ## The truth, obtained the way we never can in real life: repeat the whole
# ## experiment many times.
# truth_mean <- sd(replicate(4000, mean(rlnorm(60, 3, 0.8))))
# truth_med <- sd(replicate(4000, median(rlnorm(60, 3, 0.8))))
# cat(sprintf("  %-14s%14s%16s%12s\n", "statistic", "formula SE",
#             "bootstrap SE", "TRUE SE"))
# cat(sprintf("  %-14s%14.3f%16.3f%12.3f\n", "mean",
#             sd(dat3)/sqrt(length(dat3)), bootstrap_se(dat3, mean), truth_mean))
# cat(sprintf("  %-14s%14s%16.3f%12.3f\n", "median", "none exists",
#             bootstrap_se(dat3, median), truth_med))
#
# ## For the mean, the bootstrap reproduces what the formula gives - a useful
# ## sanity check that the method is sound.
# ##
# ## For the median there IS no simple formula, and the bootstrap gets it
# ## anyway. That is the point: the bootstrap gives you a standard error for
# ## almost any statistic you can compute, by using your sample as a stand-in
# ## for the population.
# ##
# ## It is not magic. It assumes your sample is representative and that
# ## observations are INDEPENDENT - resampling rows of clustered data
# ## reproduces exactly the error from Problem 2. For grouped data you must
# ## resample whole GROUPS (the "cluster bootstrap"). Topic 5 covers this.

#' ## What to take away
#'
#' 1. **Properties belong to the estimator**, not to the estimate.
#' 2. The **standard error** is the spread of your estimate across repeat
#'    experiments - estimable from one sample.
#' 3. **SD describes the data; SE describes your knowledge.**
#' 4. The **central limit theorem** makes averages normal - but not instantly.
#' 5. Precision costs $n$ **squared**. Better design beats more samples.
#' 6. All of it rests on **independence**.
#'
#' **Next:** `F5_confidence_intervals_and_pvalues.R`
