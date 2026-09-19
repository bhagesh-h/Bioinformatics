#' ---
#' title: "Module 07 - t-tests, rank tests, and permutation tests"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 7, equations (7.1)-(7.7)
#'
#' ## What you will learn
#'
#' 1. Pooled vs Welch (7.2)-(7.4) and **why R defaults to Welch**.
#' 2. That a t-test IS a linear model (7.5) -- to machine precision.
#' 3. What Mann-Whitney actually tests (7.6)-(7.7): the AUC, not a median.
#' 4. Permutation tests, and when exchangeability fails.

#+ setup, message = FALSE
MODULE_NAME <- "07_ttests_ranks_permutation"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. The three t-tests, from their formulas, eq. (7.1)-(7.4)

#+ ttests
header("1. Implementing the t-tests from their formulas (7.1)-(7.4)")
t_pooled <- function(x, y) {                                   # eq. (7.2)
  n1 <- length(x); n2 <- length(y)
  sp2 <- ((n1-1)*var(x) + (n2-1)*var(y)) / (n1+n2-2)
  t <- (mean(x) - mean(y)) / sqrt(sp2 * (1/n1 + 1/n2))
  df <- n1 + n2 - 2
  c(t = t, df = df, p = 2 * pt(abs(t), df, lower.tail = FALSE))
}
t_welch <- function(x, y) {                                    # eq. (7.3)-(7.4)
  n1 <- length(x); n2 <- length(y)
  v1 <- var(x)/n1; v2 <- var(y)/n2
  t <- (mean(x) - mean(y)) / sqrt(v1 + v2)
  df <- (v1 + v2)^2 / (v1^2/(n1-1) + v2^2/(n2-1))              # Welch-Satterthwaite
  c(t = t, df = df, p = 2 * pt(abs(t), df, lower.tail = FALSE))
}
set.seed(701)
x <- rnorm(8, 0, 1); y <- rnorm(20, 0.8, 2.5)
tp <- t_pooled(x, y); tw <- t_welch(x, y)
cat(sprintf("  pooled (7.2): t=%+.4f  df=%.0f      p=%.4f\n", tp["t"], tp["df"], tp["p"]))
cat(sprintf("  Welch  (7.3): t=%+.4f  df=%.2f   p=%.4f\n", tw["t"], tw["df"], tw["p"]))
tt <- t.test(x, y)
cat(sprintf("  t.test()    : t=%+.4f  df=%.2f   p=%.4f   (var.equal=FALSE default)\n",
            tt$statistic, tt$parameter, tt$p.value))
cat(sprintf("\n  Welch df lies between min(n)-1=%d and n1+n2-2=%d, as required.\n",
            min(length(x), length(y)) - 1, length(x) + length(y) - 2))

#+ welch-calibration
header("Calibration under unequal variances and unequal n")
fpr <- function(n1, n2, sd1, sd2, n_sim = 4000) {
  r <- replicate(n_sim, {
    a <- rnorm(n1, 0, sd1); b <- rnorm(n2, 0, sd2)
    c(t_pooled(a, b)["p"], t_welch(a, b)["p"])
  })
  c(pooled = mean(r[1, ] < 0.05), welch = mean(r[2, ] < 0.05))
}
set.seed(702)
cat(sprintf("%4s%4s%6s%6s%13s%12s\n", "n1", "n2", "sd1", "sd2", "pooled FPR", "Welch FPR"))
for (cfg in list(c(10,10,1,1), c(10,10,1,3), c(5,25,3,1), c(5,25,1,3), c(25,5,1,3))) {
  r <- fpr(cfg[1], cfg[2], cfg[3], cfg[4])
  flag <- if (r["pooled"] > 0.08) "  <-- BROKEN" else ""
  cat(sprintf("%4d%4d%6g%6g%12.1f%%%11.1f%%%s\n", cfg[1], cfg[2], cfg[3], cfg[4],
              100*r["pooled"], 100*r["welch"], flag))
}
cat("\n  Welch holds ~5% everywhere. Pooled fails badly when the SMALL group\n")
cat("  has the LARGE variance. This is why t.test() defaults to Welch.\n")
cat("  DO NOT pre-test variances and then choose: the two-stage procedure has\n")
cat("  its own distorted error rate.\n")

#' ## 2. A t-test IS a linear model, eq. (7.5)

#+ lm-equivalence
header("2. t-test == lm == ANOVA, to machine precision (7.5)")
set.seed(703)
g1 <- rnorm(15, 5); g2 <- rnorm(15, 6.2)
d <- data.frame(y = c(g1, g2), g = rep(c("A", "B"), each = 15))
fit <- lm(y ~ g, data = d)
tt <- t.test(g1, g2, var.equal = TRUE)
av <- anova(fit)
cat(sprintf("  lm coefficient beta1    = %.10f\n", coef(fit)["gB"]))
cat(sprintf("  difference of means     = %.10f\n", mean(g2) - mean(g1)))
cat(sprintf("  lm t-statistic          = %.10f\n", coef(summary(fit))["gB", "t value"]))
cat(sprintf("  pooled t-test statistic = %.10f\n", -tt$statistic))
cat(sprintf("\n  F from anova(lm)        = %.10f\n", av[["F value"]][1]))
cat(sprintf("  t^2                     = %.10f   <- eq. (B.6)\n", tt$statistic^2))
stopifnot(all.equal(unname(coef(summary(fit))["gB", "t value"]), unname(-tt$statistic)))
stopifnot(all.equal(av[["F value"]][1], unname(tt$statistic^2)))
cat("\n  Assertions passed: they are the same computation.\n")

set.seed(704)
subj <- rnorm(14, 0, 2)
pre <- subj + rnorm(14, 0, 0.7); post <- subj + rnorm(14, 0, 0.7) + 0.9
cat(sprintf("\n  paired t-test p             = %.6f\n",
            t.test(post, pre, paired = TRUE)$p.value))
cat(sprintf("  one-sample t on differences = %.6f\n", t.test(post - pre)$p.value))

#' ## 3. What Mann-Whitney actually tests, eq. (7.6)-(7.7)
#'
#' $$\frac{U_1}{n_1n_2}=\widehat{\Pr}(Y_1>Y_2)+\tfrac12\widehat{\Pr}(Y_1=Y_2)
#'   =\widehat{\mathrm{AUC}}$$

#+ mann-whitney
header("3. Mann-Whitney = AUC, not a median test (7.6)-(7.7)")
mann_whitney <- function(x, y) {
  n1 <- length(x); n2 <- length(y)
  R1 <- sum(rank(c(x, y))[seq_len(n1)])
  U1 <- R1 - n1 * (n1 + 1) / 2                                 # eq. (7.6)
  list(U1 = U1, AUC = U1 / (n1 * n2), E_U = n1 * n2 / 2,
       Var_U = n1 * n2 * (n1 + n2 + 1) / 12,
       rank_biserial = 2 * U1 / (n1 * n2) - 1)
}
set.seed(705)
a <- rnorm(40); b <- rnorm(50, 0.7)
r <- mann_whitney(a, b)
direct_auc <- mean(outer(a, b, ">")) + 0.5 * mean(outer(a, b, "=="))
cat(sprintf("  U1 from ranks (7.6)       = %.1f\n", r$U1))
cat(sprintf("  U from wilcox.test()      = %.1f\n",
            wilcox.test(a, b)$statistic))
cat(sprintf("  AUC = U1/(n1*n2)  (7.7)   = %.6f\n", r$AUC))
cat(sprintf("  direct P(a > b)           = %.6f   <- identical\n", direct_auc))
cat(sprintf("  rank-biserial correlation = %+.4f\n", r$rank_biserial))

cat("\n--- Two distributions with the SAME median but different shape ---\n")
set.seed(706)
h1 <- rnorm(300)
h2 <- c(rnorm(150, -0.55, 0.30), rnorm(150, 1.8, 1.2))
h2 <- h2 - median(h2) + median(h1)         # force medians to match EXACTLY
cat(sprintf("  median group 1 = %+.4f\n", median(h1)))
cat(sprintf("  median group 2 = %+.4f   (matched by construction)\n", median(h2)))
cat(sprintf("  mean   group 1 = %+.4f\n", mean(h1)))
cat(sprintf("  mean   group 2 = %+.4f\n", mean(h2)))
cat(sprintf("\n  Mann-Whitney p = %.2e   <- 'significant' despite identical medians\n",
            wilcox.test(h1, h2)$p.value))
cat(sprintf("  AUC            = %.4f\n", mann_whitney(h1, h2)$AUC))
cat(sprintf("  Welch t-test p = %.4f\n", t.test(h1, h2)$p.value))
cat("\n  CONCLUSION: 'use Wilcoxon to compare medians' is FALSE in general.\n")
cat("  Report what it estimates: P(Y1 > Y2), i.e. the AUC.\n")

#' ## 4. Choosing a test: a power comparison

#+ power-comparison
header("4. Relative power under three data-generating processes")
perm_p <- function(a, b, B = 199) {
  pooled <- c(a, b); n <- length(a)
  obs <- abs(mean(a) - mean(b))
  cnt <- sum(replicate(B, {
    p <- sample(pooled); abs(mean(p[seq_len(n)]) - mean(p[-seq_len(n)])) >= obs }))
  (1 + cnt) / (B + 1)
}
compare_power <- function(fa, fb, n = 15, n_sim = 800) {
  r <- replicate(n_sim, {
    a <- fa(n); b <- fb(n)
    c(t.test(a, b)$p.value,
      suppressWarnings(wilcox.test(a, b)$p.value),
      perm_p(a, b))
  })
  rowMeans(r < 0.05)
}
set.seed(707)
cases <- list(
  "Normal, shift 0.8"           = list(function(k) rnorm(k), function(k) rnorm(k, 0.8)),
  "Log-normal, ratio 2"         = list(function(k) rlnorm(k), function(k) 2 * rlnorm(k)),
  "Heavy tails (t3), shift 0.8" = list(function(k) rt(k, 3), function(k) rt(k, 3) + 0.8),
  "TRUE NULL (calibration)"     = list(function(k) rnorm(k), function(k) rnorm(k)))
cat(sprintf("%-30s%10s%11s%13s\n", "scenario", "Welch t", "Wilcoxon", "permutation"))
for (nm in names(cases)) {
  r <- compare_power(cases[[nm]][[1]], cases[[nm]][[2]])
  cat(sprintf("%-30s%9.1f%%%10.1f%%%12.1f%%\n", nm, 100*r[1], 100*r[2], 100*r[3]))
}
cat("\n  All three hold ~5% under the null: all VALID, just differently\n")
cat("  powerful. Choose by ESTIMAND, not by a pre-test.\n")

#' ## 5. Permutation and the limits of exchangeability

#+ exchangeability
header("5. Permutation is exact ONLY under exchangeability")
set.seed(708)
cat(sprintf("  permuting independent observations : FPR = %.1f%%  (valid)\n",
            100 * mean(replicate(600, perm_p(rnorm(8), rnorm(8)) < 0.05))))
clustered_fpr <- mean(replicate(600, {
  b <- rnorm(8)                                   # 8 donors, 4 per arm
  cells <- matrix(rnorm(8 * 30), nrow = 8) + b
  perm_p(as.vector(cells[1:4, ]), as.vector(cells[5:8, ]), B = 99) < 0.05
}))
cat(sprintf("  permuting CELLS from clustered data: FPR = %.1f%%  (broken)\n",
            100 * clustered_fpr))
fixed_fpr <- mean(replicate(600, {
  b <- rnorm(8)
  cells <- matrix(rnorm(8 * 30), nrow = 8) + b
  dm <- rowMeans(cells)
  obs <- abs(mean(dm[1:4]) - mean(dm[5:8]))
  cnt <- sum(replicate(99, { p <- sample(dm); abs(mean(p[1:4]) - mean(p[5:8])) >= obs }))
  (1 + cnt) / 100 < 0.05
}))
cat(sprintf("  permuting DONOR labels             : FPR = %.1f%%  (valid)\n",
            100 * fixed_fpr))
cat("\n  Permutation does NOT rescue pseudoreplication. Permute at the level\n")
cat("  that was randomised.\n")

#' ## 6. Figure

#+ figure
png(file.path(OUT, "two_group_tests.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
plot(density(h1), col = "steelblue", lwd = 2, xlim = c(-4, 5),
     main = "Identical medians, tiny Wilcoxon p", xlab = "value")
lines(density(h2), col = "darkorange", lwd = 2)
abline(v = median(h1), col = "steelblue", lty = 2)
abline(v = median(h2), col = "darkorange", lty = 3)
legend("topright", c("group 1", "group 2"), col = c("steelblue", "darkorange"),
       lwd = 2, bty = "n", cex = 0.7)

scores <- c(a, b); labels <- c(rep(0, length(a)), rep(1, length(b)))
o <- order(-scores)
tpr <- cumsum(labels[o]) / sum(labels)
fpr_ <- cumsum(1 - labels[o]) / sum(1 - labels)
plot(fpr_, tpr, type = "l", lwd = 2, col = "steelblue", xlab = "false positive rate",
     ylab = "true positive rate",
     main = sprintf("Eq. (7.7): AUC = %.3f = U/(n1 n2)", mann_whitney(a, b)$AUC))
abline(0, 1, lty = 2)

ratios <- 10^seq(-1.5, 1.5, length.out = 100)
plot(NA, xlim = range(ratios), ylim = c(0, 30), log = "x", xlab = "sd2 / sd1",
     ylab = "Welch df", main = "Eq. (7.4): Welch-Satterthwaite df")
cols <- c("steelblue", "darkorange", "forestgreen")
cfgs <- list(c(5, 25), c(15, 15), c(25, 5))
for (i in seq_along(cfgs)) {
  n1 <- cfgs[[i]][1]; n2 <- cfgs[[i]][2]
  dfs <- (1/n1 + ratios^2/n2)^2 / ((1/n1)^2/(n1-1) + (ratios^2/n2)^2/(n2-1))
  lines(ratios, dfs, col = cols[i], lwd = 2)
}
legend("topright", sapply(cfgs, function(z) sprintf("n1=%d, n2=%d", z[1], z[2])),
       col = cols, lwd = 2, bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "two_group_tests.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 7)
#'
#' 1. Pairing comes from the design, never from a normality test.
#' 2. Default to Welch for independent two-group comparisons.
#' 3. Use a rank test when the ESTIMAND is rank-based, and report P(Y1>Y2).
#' 4. Normality applies to the sampling distribution, not the raw values.
#'
#' **Next:** `08_multiple_testing.R`
