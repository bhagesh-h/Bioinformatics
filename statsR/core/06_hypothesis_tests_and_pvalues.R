#' ---
#' title: "Module 06 - Hypothesis tests and p-values"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 6, equations (6.1)-(6.7)
#'
#' ## What you will learn
#'
#' 1. Uniformity under the null (6.2) and **the p-value histogram**.
#' 2. False-positive risk (6.4): why "significant" often means "probably wrong".
#' 3. Type S and Type M errors (6.5): the winner's curse.
#' 4. Combining p-values (6.6)-(6.7).

#+ setup, message = FALSE
MODULE_NAME <- "06_hypothesis_tests_and_pvalues"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Uniformity under the null, eq. (6.2)
#'
#' Everything in Topic 8 rests on this. It is also the most useful
#' diagnostic you own.

#+ uniformity
header("1. p-values are uniform under the null (6.2)")
pvalues_under_null <- function(n_tests = 5000, n = 8, kind = "clean") {
  if (kind == "clean") {
    a <- matrix(rnorm(n_tests * n), ncol = n)
    b <- matrix(rnorm(n_tests * n), ncol = n)
  } else if (kind == "unmodelled_batch") {
    ## A batch effect shared within each arm, ignored by the test. The null of
    ## "no condition effect" is still true, but the MODEL is wrong.
    batch <- rnorm(n_tests, 0, 1.2)
    a <- matrix(rnorm(n_tests * n), ncol = n) + batch
    b <- matrix(rnorm(n_tests * n), ncol = n) - batch
  } else {
    ## Small-count Fisher exact tests: discrete and conservative.
    return(vapply(seq_len(n_tests), function(i)
      fisher.test(matrix(rbinom(4, 10, 0.3) + 1, 2))$p.value, numeric(1)))
  }
  vapply(seq_len(n_tests), function(i) t.test(a[i, ], b[i, ])$p.value, numeric(1))
}

set.seed(601)
for (kind in c("clean", "unmodelled_batch", "discrete")) {
  p <- pvalues_under_null(if (kind == "discrete") 2000 else 5000, kind = kind)
  cat(sprintf("  %-18s P(p<0.05) = %.4f   KS vs Uniform p = %.2e\n",
              kind, mean(p < 0.05), suppressWarnings(ks.test(p, "punif")$p.value)))
}
cat("\n  clean            -> ~0.05, uniform. The test is calibrated.\n")
cat("  unmodelled_batch -> inflated. A small p indicts the WHOLE model\n")
cat("                      (eq. 6.1 conditions on EVERY assumption).\n")
cat("  discrete         -> conservative: P(p<0.05) < 0.05.\n")

#' ### How to read a p-value histogram

#+ histograms
header("Diagnosing four p-value histograms")
set.seed(602)
G <- 8000
scenarios <- list(
  "healthy (flat + spike at 0)" = c(runif(0.85 * G),
                                    2 * pnorm(abs(rnorm(0.15 * G, 3.5)), lower.tail = FALSE)),
  "anticonservative (all small)" = rbeta(G, 0.3, 1),
  "conservative (hump near 1)"   = rbeta(G, 1, 0.4),
  "mid-hump (wrong null)"        = pmin(pmax(rnorm(G, 0.5, 0.13), 1e-6), 1 - 1e-6))
interp <- c("proceed to FDR", "fix the model first",
            "discreteness/over-correction", "wrong null or strong dependence")
cat(sprintf("%-32s%11s%10s  interpretation\n", "histogram shape", "P(p<0.05)", "P(p>0.9)"))
for (i in seq_along(scenarios)) {
  p <- scenarios[[i]]
  cat(sprintf("%-32s%11.3f%10.3f  %s\n", names(scenarios)[i],
              mean(p < 0.05), mean(p > 0.9), interp[i]))
}
cat("\n  RULE: adjusting a pathological histogram produces confident nonsense.\n")

#' ## 2. False-positive risk, eq. (6.4)

#+ ppv
header("2. What fraction of 'significant' findings are real? (6.4)")
ppv <- function(power, prior, alpha = 0.05)
  power * prior / (power * prior + alpha * (1 - prior))

powers <- c(0.2, 0.5, 0.8, 0.95)
cat(sprintf("%12s%s\n", "prior P(H1)",
            paste(sprintf("%13s", paste0("power=", powers)), collapse = "")))
for (prior in c(0.01, 0.05, 0.10, 0.30, 0.50)) {
  cat(sprintf("%12.2f%s\n", prior,
              paste(sprintf("%12.1f%% ", 100 * ppv(powers, prior)), collapse = "")))
}
cat(sprintf("\n  Worked example: prior = 0.10, power = 0.5, alpha = 0.05\n"))
cat(sprintf("    PPV = %.1f%%  -> nearly half of 'significant' results are false,\n",
            100 * ppv(0.5, 0.10)))
cat("    with PERFECT statistical practice.\n")

set.seed(603)
G <- 20000; PRIOR <- 0.10; EFFECT <- 1.2; n <- 10
is_alt <- runif(G) < PRIOR
delta <- ifelse(is_alt, EFFECT, 0)
p <- vapply(seq_len(G), function(i)
  t.test(rnorm(n, delta[i]), rnorm(n))$p.value, numeric(1))
sig <- p < 0.05
cat(sprintf("\n  Simulated screen: observed PPV among significant = %.1f%%\n",
            100 * mean(is_alt[sig])))
cat(sprintf("  predicted by eq. (6.4)                          = %.1f%%\n",
            100 * ppv(mean(sig[is_alt]), PRIOR)))

#' ## 3. Type S and Type M errors, eq. (6.5) -- the winner's curse

#+ type-sm
header("3. Conditioning on significance exaggerates effects (6.5)")
set.seed(604)
n <- 10; se <- sqrt(2 / n)
cat(sprintf("%12s%9s%9s%24s\n", "true effect", "power", "Type S", "Type M (exaggeration)"))
for (te in c(0.1, 0.2, 0.5, 1.0, 2.0)) {
  est <- rnorm(60000, te, se)
  sg <- abs(est / se) > 1.96
  cat(sprintf("%12.2f%8.1f%%%8.1f%%%23.2fx\n", te, 100 * mean(sg),
              100 * mean(sign(est[sg]) != sign(te)),
              mean(abs(est[sg])) / abs(te)))
}
cat("\n  At low power the published effect is 2-5x too large and can have the\n")
cat("  WRONG SIGN. This is why underpowered findings 'fail to replicate'.\n")

#' ## 4. One-sided vs two-sided

#+ sidedness
header("4. Sidedness is a design decision, not a data decision")
set.seed(605)
n <- 12; TRUE_EFF <- 0.6
res <- replicate(5000, {
  x <- rnorm(n, TRUE_EFF)
  c(t.test(x)$p.value,
    t.test(x, alternative = "greater")$p.value,
    t.test(x, alternative = "less")$p.value)
})
cat(sprintf("  true effect = +%.1f, n = %d\n", TRUE_EFF, n))
cat(sprintf("    two-sided power              : %.1f%%\n", 100 * mean(res[1, ] < 0.05)))
cat(sprintf("    one-sided, correct direction : %.1f%%\n", 100 * mean(res[2, ] < 0.05)))
cat(sprintf("    one-sided, wrong direction   : %.1f%%\n", 100 * mean(res[3, ] < 0.05)))
cat("\n  A one-sided test has ZERO power against the opposite direction.\n")

#' ## 5. Combining p-values, eq. (6.6)-(6.7)

#+ combining
header("5. Fisher and Stouffer combination (6.6)-(6.7)")
fisher_combine <- function(p) {
  stat <- -2 * sum(log(p)); c(stat = stat, p = pchisq(stat, 2 * length(p), lower.tail = FALSE))
}
stouffer_combine <- function(p, w = NULL, signs = NULL) {
  z <- qnorm(p, lower.tail = FALSE)
  if (!is.null(signs)) z <- z * signs
  if (is.null(w)) w <- rep(1, length(z))
  stat <- sum(w * z) / sqrt(sum(w^2))
  c(stat = stat, p = pnorm(stat, lower.tail = FALSE))
}
studies <- data.frame(study = LETTERS[1:4], n = c(20, 150, 40, 300),
                      p_one_sided = c(0.12, 0.04, 0.20, 0.03),
                      direction = c(1, 1, -1, 1))
print(studies, row.names = FALSE)
f <- fisher_combine(studies$p_one_sided)
s1 <- stouffer_combine(studies$p_one_sided, sqrt(studies$n))
s2 <- stouffer_combine(studies$p_one_sided, sqrt(studies$n), studies$direction)
cat(sprintf("\n  Fisher (6.6)              : X2=%.3f, p=%.5f\n", f["stat"], f["p"]))
cat(sprintf("  Stouffer, weights sqrt(n) : z =%.3f, p=%.5f\n", s1["stat"], s1["p"]))
cat(sprintf("  Stouffer, direction-aware : z =%.3f, p=%.5f\n", s2["stat"], s2["p"]))
cat("\n  Fisher ignores DIRECTION entirely. For meta-analysis prefer the\n")
cat("  direction-aware version - or better, combine EFFECT SIZES (Topic 5).\n")

#' ## 6. Figure

#+ figure
png(file.path(OUT, "pvalues.png"), width = 1400, height = 800, res = 110)
par(mfrow = c(2, 3), mar = c(4.2, 4.2, 3, 1))
for (i in seq_along(scenarios)) {
  hist(scenarios[[i]], breaks = 40, col = "steelblue", border = "white",
       main = names(scenarios)[i], xlab = "p-value", cex.main = 0.9)
  abline(h = length(scenarios[[i]]) / 40, col = "red", lty = 2)
}
priors <- seq(0.005, 0.5, length.out = 200)
plot(NA, xlim = c(0, 0.5), ylim = c(0, 1), xlab = "prior P(H1 true)",
     ylab = "PPV", main = "Eq. (6.4): false-positive risk", cex.main = 0.9)
cols <- c("steelblue", "darkorange", "firebrick")
for (i in seq_along(c(0.2, 0.5, 0.8)))
  lines(priors, ppv(c(0.2, 0.5, 0.8)[i], priors), col = cols[i], lwd = 2)
abline(h = 0.5, lty = 3)
legend("bottomright", paste0("power=", c(0.2, 0.5, 0.8)), col = cols, lwd = 2,
       bty = "n", cex = 0.7)

set.seed(606)
effs <- seq(0.05, 2, length.out = 40)
tm <- sapply(effs, function(te) {
  est <- rnorm(30000, te, sqrt(2/10))
  sg <- abs(est / sqrt(2/10)) > 1.96
  if (sum(sg)) mean(abs(est[sg])) / te else NA
})
plot(effs, tm, type = "l", lwd = 2, col = "firebrick", xlab = "true effect size",
     ylab = "exaggeration ratio", main = "Eq. (6.5): the winner's curse",
     cex.main = 0.9)
abline(h = 1, lty = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "pvalues.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 6)
#'
#' 1. Report the exact p-value with the estimate and interval. Never stars.
#' 2. `p > 0.05` means the data are compatible with H0 AND with many non-zero
#'    effects. Use TOST (Module 05) to claim equivalence.
#' 3. Decide sidedness, stopping rule, covariates and the multiplicity family
#'    BEFORE the analysis.
#' 4. Check the p-value histogram before adjusting anything.
#'
#' **Next:** `07_ttests_ranks_permutation.R`
