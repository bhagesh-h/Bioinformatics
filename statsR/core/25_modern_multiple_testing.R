#' ---
#' title: "Module 25 - Modern multiple testing: weighting, e-values, knockoffs"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 36, equations (36.1)-(36.6)
#' **Assumes:** Module 08.
#'
#' ## What you will learn
#'
#' 1. **Weighted BH (36.1)** and why the weights must not come from the p-values.
#' 2. **IHW (36.2)**: independent filtering is a blunt weight, and a smooth one
#'    recovers the power the blunt one throws away.
#' 3. What BH actually promises under dependence, and what it does not.
#' 4. **E-values (36.3)** and **e-BH (36.4)**: FDR control under *any* dependence.
#' 5. **Knockoffs (36.5)-(36.6)**: FDR control when the features are correlated
#'    and no p-value is available.

#+ setup, message = FALSE
MODULE_NAME <- "25_modern_multiple_testing"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
ALPHA <- 0.05

#' Benjamini-Hochberg, returned as a logical vector rather than adjusted
#' p-values, because everything below reasons about the REJECTION SET.
bh <- function(p, alpha = ALPHA) {
  m <- length(p); o <- order(p)
  passed <- p[o] <= alpha * seq_len(m) / m
  k <- if (any(passed)) max(which(passed)) else 0L
  rej <- logical(m); if (k > 0) rej[o[seq_len(k)]] <- TRUE
  rej
}

#' Weighted BH, eq. (36.1). Weights must average to 1 and be independent of
#' the p-values. A weight of zero removes the hypothesis entirely.
weighted_bh <- function(p, w, alpha = ALPHA) {
  w <- w / mean(w)
  pw <- ifelse(w > 0, p / pmax(w, 1e-12), Inf)
  bh(pw, alpha)
}

fdp_tpr <- function(rej, truth)
  c(fdp = sum(rej & !truth) / max(sum(rej), 1),
    tpr = sum(rej & truth) / max(sum(truth), 1))

#' ## 1. A screen where power depends on a covariate
#'
#' In RNA-seq a gene's mean expression predicts how much power it has, and is
#' independent of its p-value when the gene is null. That is exactly the
#' situation weighting is built for.

#+ screen
header("1. A screen in which power varies with an observable covariate")
simulate_screen <- function(m = 8000, pi1 = 0.10, seed = 0) {
  set.seed(seed)
  mu <- exp(rnorm(m, 3.0, 1.8))          # covariate: mean expression
  se <- 1 / sqrt(mu) + 0.05              # precision improves with expression
  is_de <- runif(m) < pi1
  effect <- ifelse(is_de, rnorm(m, 0, 0.55), 0)
  z <- (effect + rnorm(m, 0, se)) / se
  list(p = 2 * pnorm(-abs(z)), truth = is_de, cov = mu)
}
S <- simulate_screen(seed = 1)
p <- S$p; truth <- S$truth; cov_ <- S$cov
cat(sprintf("  %s hypotheses, %s truly non-null (%.1f%%)\n",
            format(length(p), big.mark = ","), format(sum(truth), big.mark = ","),
            100 * mean(truth)))
cat(sprintf("  covariate (mean expression) spans %.1f to %s\n",
            min(cov_), format(round(max(cov_)), big.mark = ",")))
cat("\n  power is not spread evenly across the covariate:\n")
cat(sprintf("  %-22s%10s%10s%10s\n", "expression quartile", "features",
            "true DE", "p<0.05"))
qs <- quantile(cov_, c(0, .25, .5, .75, 1))
for (i in 1:4) {
  m_ <- cov_ >= qs[i] & cov_ <= qs[i + 1]
  cat(sprintf("  %-22s%10s%10s%10.3f\n", paste0("Q", i),
              format(sum(m_), big.mark = ","),
              format(sum(truth[m_]), big.mark = ","), mean(p[m_] < 0.05)))
}
cat("\n  The covariate is informative about POWER but, for a null feature,
  carries no information about the p-value. That is the condition eq. (36.1)
  needs.\n")

#' ## 2. Independent filtering is a weight of zero or one

#+ filtering
header("2. BH, filtering, and what filtering throws away")
cat(sprintf("  %-40s%9s%8s%8s\n", "procedure", "discov.", "FDP", "power"))
rej_bh <- bh(p); ft <- fdp_tpr(rej_bh, truth)
cat(sprintf("  %-40s%9d%8.3f%8.3f\n", "BH on everything", sum(rej_bh),
            ft[1], ft[2]))
for (q in c(0.2, 0.4, 0.6)) {
  keep <- as.numeric(cov_ >= quantile(cov_, q))
  rj <- weighted_bh(p, keep); f_ <- fdp_tpr(rj, truth)
  cat(sprintf("  %-40s%9d%8.3f%8.3f\n",
              sprintf("independent filtering, drop bottom %.0f%%", 100 * q),
              sum(rj), f_[1], f_[2]))
}
cat("\n  Filtering is weighted BH with weights in {0, c}: a hypothesis is either
  in the family or removed from it. It helps, and the FDP stays controlled,
  because the filter statistic is independent of the p-value under the null.

  But a hard cut throws away the gradient. A feature just below the threshold
  gets weight zero; one just above gets full weight. Nothing about the biology
  changes at that line.\n")

#' ## 3. IHW: learn a smooth weight, eq. (36.2)
#'
#' Stratify on the covariate, estimate how much signal each stratum carries,
#' and weight accordingly. The estimate for each fold uses only the *other*
#' folds, which keeps the weights independent of the p-values they act on.

#+ ihw
header("3. Independent hypothesis weighting (36.2)")
storey_pi0 <- function(pv, lam = 0.5) {
  if (!length(pv)) return(1)
  min(1, mean(pv > lam) / (1 - lam))
}
ihw <- function(p, covariate, alpha = ALPHA, n_strata = 8, n_folds = 5,
                seed = 0) {
  set.seed(seed)
  m <- length(p)
  edges <- quantile(covariate, seq(0, 1, length.out = n_strata + 1))
  edges[length(edges)] <- edges[length(edges)] + 1e-9
  stratum <- pmin(pmax(findInterval(covariate, edges), 1), n_strata)
  fold <- sample(rep_len(1:n_folds, m))
  w <- rep(1, m)
  for (k in 1:n_folds) {
    other <- fold != k
    sig <- sapply(1:n_strata,
                  function(g) 1 - storey_pi0(p[other & stratum == g]))
    sig <- pmax(sig, 1e-3); sig <- sig / mean(sig)   # eq. (36.1): mean 1
    w[fold == k] <- sig[stratum[fold == k]]
  }
  list(rej = weighted_bh(p, w, alpha), w = w, stratum = stratum)
}
I <- ihw(p, cov_, seed = 2); fi <- fdp_tpr(I$rej, truth)
best_q <- c(0.2, 0.4, 0.6)[which.max(sapply(c(0.2, 0.4, 0.6), function(q)
  fdp_tpr(weighted_bh(p, as.numeric(cov_ >= quantile(cov_, q))), truth)[2]))]
kb <- weighted_bh(p, as.numeric(cov_ >= quantile(cov_, best_q)))
fb <- fdp_tpr(kb, truth)
cat(sprintf("  %-40s%9s%8s%8s\n", "procedure", "discov.", "FDP", "power"))
cat(sprintf("  %-40s%9d%8.3f%8.3f\n", "BH", sum(rej_bh), ft[1], ft[2]))
cat(sprintf("  %-40s%9d%8.3f%8.3f\n",
            sprintf("best filter (drop bottom %.0f%%)", 100 * best_q),
            sum(kb), fb[1], fb[2]))
cat(sprintf("  %-40s%9d%8.3f%8.3f\n", "IHW (36.2)", sum(I$rej), fi[1], fi[2]))
wm <- sapply(1:8, function(g) mean(I$w[I$stratum == g]))
cat("\n  learned weights by covariate stratum (low to high expression):\n   ")
cat(sprintf("%7.2f", wm), "\n", sep = "")
cat("\n  The weights rise with expression, because that is where the signal is.
  No hypothesis is discarded; the low-expression strata simply need stronger
  evidence. IHW finds more than BH and more than the best hard filter, at the
  same realised FDP.\n")

#' ## 4. The trap: weights that peek at the p-values

#+ trap
header("4. Weights must not come from the p-values they weight")
cheat_w <- 1 / pmax(p, 1e-8)
rej_cheat <- weighted_bh(p, cheat_w); fc <- fdp_tpr(rej_cheat, truth)
cat(sprintf("  %-42s%9d%8.3f\n", "weights from the covariate (IHW)",
            sum(I$rej), fi[1]))
cat(sprintf("  %-42s%9d%8.3f\n", "weights from the p-values themselves",
            sum(rej_cheat), fc[1]))
S0 <- simulate_screen(pi1 = 0, seed = 7)
cat("\n  On a screen with NO non-null features at all:\n")
cat(sprintf("  %-42s%9d discoveries\n", "BH", sum(bh(S0$p))))
cat(sprintf("  %-42s%9d discoveries\n", "IHW",
            sum(ihw(S0$p, S0$cov, seed = 3)$rej)))
cat(sprintf("  %-42s%9d discoveries\n", "weights from the p-values",
            sum(weighted_bh(S0$p, 1 / pmax(S0$p, 1e-8)))))
cat("\n  Weighting by the p-value inflates the FDP, and on a pure null it
  manufactures discoveries from nothing. The rule in eq. (36.1) is not a
  technicality: the weights carry no information about the outcome, or the
  guarantee is void.

  A covariate is admissible if, for a NULL feature, it tells you nothing about
  the p-value. Mean expression, feature variance and prior published evidence
  qualify. The observed effect size does not.\n")

#' ## 5. What BH promises under dependence
#'
#' BH controls the **expected** FDP. Under dependence the expectation still
#' holds but the spread widens, and you only ever run the experiment once.

#+ dependence
header("5. Under dependence, the FDP becomes a lottery")
correlated_screen <- function(m = 2000, pi1 = 0.1, rho = 0, n_blocks = 40,
                              seed = 0) {
  set.seed(seed)
  per <- m %/% n_blocks
  shared <- rnorm(n_blocks)
  z <- as.vector(sqrt(rho) * rep(shared, each = per) +
                   sqrt(1 - rho) * rnorm(m))
  is_de <- runif(m) < pi1
  z <- z + is_de * rnorm(m, 3.0, 0.5)
  list(p = 2 * pnorm(-abs(z)), truth = is_de)
}
cat(sprintf("  %-8s%11s%12s%12s%12s\n", "rho", "mean FDP", "SD of FDP",
            "FDP > 0.10", "FDP > 0.20"))
for (rho in c(0.0, 0.3, 0.7)) {
  fdps <- sapply(1:300, function(i) {
    cs <- correlated_screen(rho = rho, seed = 1000 + i)
    fdp_tpr(bh(cs$p), cs$truth)[1]
  })
  cat(sprintf("  %-8.1f%11.3f%12.3f%11.1f%%%11.1f%%\n", rho, mean(fdps),
              sd(fdps), 100 * mean(fdps > 0.10), 100 * mean(fdps > 0.20)))
}
cat("\n  The mean column stays controlled: BH is valid here, because
  equicorrelated positive dependence satisfies PRDS. Read the other three
  columns.

  As dependence grows, the FDP of a SINGLE experiment becomes wildly variable.
  At rho = 0.7 a substantial fraction of runs exceed twice the nominal rate.
  \"FDR = 5%\" is a statement about the average over experiments you will never
  run. Your gene list is one draw from that distribution.\n")

#' ## 6. E-values and e-BH, eq. (36.3)-(36.4)
#'
#' An e-value is calibrated by its **mean** rather than its tail. That single
#' change buys FDR control under arbitrary dependence.
#'
#' $$\mathbb{E}_{H_0}[e]\le 1 \qquad (36.3)$$

#+ ebh
header("6. e-BH: valid under any dependence (36.3)-(36.4)")
p_to_e <- function(p, kappa = 0.5) kappa * pmax(p, 1e-300)^(kappa - 1)
ebh <- function(e, alpha = ALPHA) {
  m <- length(e); o <- order(e, decreasing = TRUE)
  ok <- which(m / (seq_len(m) * alpha) <= e[o])
  k <- if (length(ok)) max(ok) else 0L
  rej <- logical(m); if (k > 0) rej[o[seq_len(k)]] <- TRUE
  rej
}
set.seed(25)
cat(sprintf("  calibrator check: mean e-value under the null = %.3f",
            mean(p_to_e(runif(200000)))))
cat("  (must be <= 1, eq. 36.3)\n")
cat(sprintf("\n  %-8s%10s%11s%10s%11s%11s%12s\n", "rho", "BH FDP", "BH power",
            "BY FDP", "BY power", "e-BH FDP", "e-BH power"))
for (rho in c(0.0, 0.7)) {
  acc <- numeric(6); R <- 200
  for (i in 1:R) {
    cs <- correlated_screen(rho = rho, seed = 5000 + i)
    m_ <- length(cs$p)
    by <- bh(cs$p, ALPHA / sum(1 / seq_len(m_)))          # eq. (8.8)
    rs <- list(bh(cs$p), by, ebh(p_to_e(cs$p)))
    for (j in 1:3) acc[c(2 * j - 1, 2 * j)] <-
      acc[c(2 * j - 1, 2 * j)] + fdp_tpr(rs[[j]], cs$truth)
  }
  acc <- acc / R
  cat(sprintf("  %-8.1f%10.3f%11.3f%10.3f%11.3f%11.3f%12.3f\n", rho, acc[1],
              acc[2], acc[3], acc[4], acc[5], acc[6]))
}
cat("\n  e-BH's power here is essentially ZERO, not merely reduced. Why:\n")
cat(sprintf("  %-26s%18s%16s\n", "discoveries wanted (k)", "e-value needed",
            "i.e. p below"))
for (k_ in c(1, 100, 200))
  cat(sprintf("  %-26d%18s%16.2e\n", k_,
              format(round(2000 / (k_ * ALPHA)), big.mark = ","),
              (0.5 * k_ * ALPHA / 2000)^2))
cat("
  To reject even one hypothesis out of 2000, e-BH needs a p-value below about
  1e-10. The signal in this screen produces p-values near 0.003. So nothing is
  rejected, and that is the calibrator's fault rather than e-BH's.

  Turning a p-value into an e-value is LOSSY. The calibrator has to work for
  every possible null distribution, so it throws away most of the evidence. An
  e-value that comes directly from the analysis -- a likelihood ratio, a
  betting martingale, a sequential test's wealth process -- is far sharper, and
  that is the setting where e-BH is genuinely competitive.

  BY is the fair comparison for \"the same input, a stronger guarantee\": it
  halves BH's power (0.50 to 0.24) and holds under any dependence.

  The practical reading: BH remains the right default for genomics, where
  positive dependence is the norm. Reach for e-BH when the dependence is
  unknown or negative, or when you need to combine evidence across analyses,
  which e-values do by simple averaging and p-values do not.\n")

#' ## 7. Knockoffs, eq. (36.5)-(36.6)
#'
#' Selecting features in a regression is not a p-value problem. Knockoffs give
#' FDR control directly, with no independence assumption between features.
#'
#' $$W_j=|\hat\beta_j|-|\hat\beta_{\tilde j}| \qquad (36.5)$$

#+ knockoffs
header("7. Knockoff filter for correlated feature selection (36.5)-(36.6)")
#' Equicorrelated model-X knockoffs for Gaussian features. Xt has the same
#' covariance as X but is conditionally independent of y. s is chosen so that
#' 2*Sigma - diag(s) stays positive semi-definite, which is what makes the
#' construction valid.
gaussian_knockoffs <- function(X, Sigma) {
  pdim <- ncol(X)
  s <- rep(min(1, 2 * min(eigen(Sigma, symmetric = TRUE)$values)) * 0.99, pdim)
  Sinv <- solve(Sigma); D <- diag(s)
  mu_t <- X - X %*% Sinv %*% D
  V <- 2 * D - D %*% Sinv %*% D
  V <- (V + t(V)) / 2 + 1e-9 * diag(pdim)
  mu_t + matrix(rnorm(length(X)), nrow(X)) %*% chol(V)
}
#' The feature statistic. Python's module uses a lasso entry order; with no
#' glmnet here we use the least-squares magnitude on the augmented design
#' [X, Xt], which is a valid knockoff statistic whenever n > 2p because it
#' obeys the same flip-sign symmetry.
knockoff_filter <- function(X, y, Sigma, alpha = ALPHA) {
  Xt <- gaussian_knockoffs(X, Sigma)
  Z <- cbind(X, Xt)
  Z <- sweep(Z, 2, sqrt(colSums(Z^2)), "/")
  b <- qr.coef(qr(Z), y); b[is.na(b)] <- 0
  pdim <- ncol(X)
  W <- abs(b[1:pdim]) - abs(b[(pdim + 1):(2 * pdim)])    # eq. (36.5)
  ts <- sort(abs(W[W != 0]))
  sel <- logical(pdim)
  for (t in ts) {
    if ((1 + sum(W <= -t)) / max(sum(W >= t), 1) <= alpha) {  # eq. (36.6)
      sel <- W >= t; break
    }
  }
  list(sel = sel, W = W)
}
n <- 700; pdim <- 200; k_true <- 60; RHO <- 0.3
Sigma <- RHO * matrix(1, pdim, pdim) + (1 - RHO) * diag(pdim)
Lc <- chol(Sigma)
truth_k <- c(rep(TRUE, k_true), rep(FALSE, pdim - k_true))
cat(sprintf("  n = %d, p = %d, %d true signals, feature correlation %.1f\n",
            n, pdim, k_true, RHO))
cat(sprintf("\n  %-34s%10s%8s%8s\n", "method", "selected", "FDP", "power"))
res_k <- res_b <- matrix(NA_real_, 10, 3)
for (i in 1:10) {
  set.seed(600 + i)
  signal <- c(4.0 * sample(c(-1, 1), k_true, TRUE), rep(0, pdim - k_true))
  X <- matrix(rnorm(n * pdim), n) %*% Lc
  y <- as.vector(X %*% signal) + rnorm(n)
  kf <- knockoff_filter(X, y, Sigma)
  res_k[i, ] <- c(fdp_tpr(kf$sel, truth_k), sum(kf$sel))
  pv <- apply(X, 2, function(x) cor.test(x, y)$p.value)
  rb <- bh(pv)
  res_b[i, ] <- c(fdp_tpr(rb, truth_k), sum(rb))
}
for (z in list(list("marginal p-values + BH", res_b),
               list("knockoff filter (36.6)", res_k)))
  cat(sprintf("  %-34s%10.1f%8.3f%8.3f\n", z[[1]], mean(z[[2]][, 3]),
              mean(z[[2]][, 1]), mean(z[[2]][, 2])))
cat("\n  Marginal testing has no chance here: with features correlated at 0.3,
  a null feature correlates with y through its correlated neighbours, so BH on
  marginal p-values selects a large number of them.

  Knockoffs test each feature CONDITIONAL on all the others, and the symmetry
  of W under the null (36.5) supplies the false-discovery estimate in (36.6)
  without any p-value or independence assumption.

  The cost: knockoffs are randomised. Two runs on the same data give different
  answers, which is what derandomised knockoffs fix by aggregating several
  runs through e-values.\n")

#+ knockoff-floor
header("7b. Knockoffs cannot make fewer than 1/alpha discoveries")
cat("  Look again at the threshold rule:

      (1 + #{W <= -t}) / #{W >= t}  <=  alpha

  The numerator is at least 1, always. So the rule can only be satisfied if
  #{W >= t} >= 1/alpha. At alpha = 0.05 that is TWENTY selections. A knockoff
  filter physically cannot report 5 discoveries at the 5% level: it reports at
  least 20, or none at all.\n")
for (z in list(c(0.05, 20), c(0.10, 10), c(0.20, 5)))
  cat(sprintf("    alpha = %-5.2f -> minimum possible selection size = %d\n",
              z[1], as.integer(z[2])))
cat("\n  That is why the run below finds nothing. Same data-generating process,
  only the number of true signals changed:\n")
for (kk in c(8, 25, 60)) {
  hits <- sapply(1:4, function(i) {
    set.seed(770 + i)
    sg <- c(4.0 * sample(c(-1, 1), kk, TRUE), rep(0, pdim - kk))
    X <- matrix(rnorm(n * pdim), n) %*% Lc
    y <- as.vector(X %*% sg) + rnorm(n)
    sum(knockoff_filter(X, y, Sigma)$sel)
  })
  cat(sprintf("    %3d true signals -> mean selections = %.1f\n", kk,
              mean(hits)))
}
cat("\n  With 8 real signals the filter is silent, not because it failed to
  find them but because the guarantee forbids a small answer. Knockoffs are a
  method for screens where you expect many discoveries. For a handful of
  candidates, use a p-value procedure.\n")

#' ## 8. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "modern_multiple_testing.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
plot(log10(cov_), -log10(pmax(p, 1e-30)), pch = 16, cex = 0.25,
     col = adjustcolor("grey40", 0.25), xlab = "log10 mean expression",
     ylab = "-log10 p", main = "Power varies with an observable covariate")
points(log10(cov_[truth]), -log10(pmax(p[truth], 1e-30)), pch = 16,
       cex = 0.3, col = adjustcolor("firebrick", 0.5))
plot(1:8, wm, type = "s", lwd = 2, col = "steelblue", xlab = "covariate stratum",
     ylab = "learned weight", main = "IHW weights (36.2)")
abline(h = 1, lty = 2, col = "grey40")
fd0 <- sapply(1:200, function(i) {
  cs <- correlated_screen(rho = 0, seed = 9000 + i); fdp_tpr(bh(cs$p), cs$truth)[1] })
fd7 <- sapply(1:200, function(i) {
  cs <- correlated_screen(rho = 0.7, seed = 9000 + i); fdp_tpr(bh(cs$p), cs$truth)[1] })
br <- seq(0, max(c(fd0, fd7)) + 0.02, length.out = 25)
hist(fd0, breaks = br, col = adjustcolor("steelblue", 0.6), border = NA,
     xlab = "realised FDP in one experiment", main = "BH controls the mean, not your run")
hist(fd7, breaks = br, col = adjustcolor("firebrick", 0.6), border = NA, add = TRUE)
abline(v = ALPHA, lty = 2)
legend("topright", c("rho = 0", "rho = 0.7"), bty = "n",
       fill = adjustcolor(c("steelblue", "firebrick"), 0.6))
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "modern_multiple_testing.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Is your covariate admissible?
#'
#' IHW needs a covariate independent of the p-value **under the null**. Test
#' three candidates: mean expression, the observed effect size, and a random
#' number. Measure the FDP each produces.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# SS <- simulate_screen(pi1 = 0.10, seed = 99)
# ps <- SS$p; ts <- SS$truth; cs2 <- SS$cov
# eff <- abs(qnorm(ps / 2))                       # observed effect size
# set.seed(4); rnd <- runif(length(ps))
# cat(sprintf("  %-26s%20s%9s%8s%14s\n", "covariate", "corr with p | null",
#             "FDP", "power", "verdict"))
# for (z in list(list("mean expression", cs2), list("observed effect size", eff),
#                list("a random number", rnd))) {
#   rho_null <- cor(z[[2]][!ts], ps[!ts], method = "spearman")
#   ft2 <- fdp_tpr(ihw(ps, z[[2]], seed = 5, n_strata = 40)$rej, ts)
#   cat(sprintf("  %-26s%20.3f%9.3f%8.3f%14s\n", z[[1]], rho_null, ft2[1],
#               ft2[2], if (abs(rho_null) > 0.1) "INADMISSIBLE" else "ok"))
# }
# cat(sprintf("\n  nominal FDR = %.2f\n", ALPHA))
#
# ## Read the correlation column first, because that is the actual condition in
# ## eq. (36.1) and it is checkable before you look at any result. Mean
# ## expression and the random number sit near zero among the nulls. The
# ## observed effect size is perfectly rank-correlated with the p-value by
# ## construction, because it IS the p-value in another coordinate.
# ##
# ## The FDP column shows the consequence: weighting by the effect size
# ## concentrates weight exactly where the small p-values already are.
# ##
# ## Two further points. The random number is ADMISSIBLE but USELESS: it cannot
# ## break FDR control, but it carries no information about power either.
# ## Admissibility and usefulness are separate properties and you need both.
# ##
# ## The damage from an inadmissible covariate also depends on how FINE the
# ## weighting is. With few strata the weights are coarse and the inflation is
# ## mild; with many strata the procedure tracks the p-value closely and the
# ## inflation is severe. Choose the covariate on principle, not by tuning the
# ## stratification until the answer looks good.

#' ### Problem 2: How much does e-BH cost you?
#'
#' Measure the power of BH, BY and e-BH across a range of signal strengths
#' under independence, where BH is valid. What are you paying for the
#' guarantee?

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %-14s%11s%11s%12s%9s%10s\n", "effect size", "BH power",
#             "BY power", "e-BH power", "BH FDP", "e-BH FDP"))
# for (eff2 in c(2, 3, 4, 5)) {
#   acc <- numeric(5); R <- 150
#   for (i in 1:R) {
#     set.seed(7000 + i); m2 <- 2000
#     de <- runif(m2) < 0.1
#     z <- rnorm(m2) + de * eff2
#     pv <- 2 * pnorm(-abs(z))
#     byv <- bh(pv, ALPHA / sum(1 / seq_len(m2)))
#     a <- fdp_tpr(bh(pv), de); b <- fdp_tpr(byv, de)
#     cc <- fdp_tpr(ebh(p_to_e(pv)), de)
#     acc <- acc + c(a[2], b[2], cc[2], a[1], cc[1])
#   }
#   acc <- acc / R
#   cat(sprintf("  %-14.1f%11.3f%11.3f%12.3f%9.3f%10.3f\n", eff2, acc[1],
#               acc[2], acc[3], acc[4], acc[5]))
# }
#
# ## e-BH sits between BH and BY, and the gap closes as the signal strengthens:
# ## when effects are obvious, the price of the stronger guarantee is small.
# ## When effects are marginal, which is when you most want discoveries, the
# ## price is large.
# ##
# ## This is the general shape of the trade. Stronger guarantees cost power
# ## exactly in the regime where power is scarce. Choose the guarantee
# ## deliberately rather than defaulting to the safest one, and report which
# ## you used.

#' ### Problem 3: Derandomising knockoffs
#'
#' Run the knockoff filter several times on the *same* data and count how much
#' the selected set changes. Then aggregate the runs by selection frequency.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(321)
# signal <- c(4.0 * sample(c(-1, 1), k_true, TRUE), rep(0, pdim - k_true))
# Xf <- matrix(rnorm(n * pdim), n) %*% Lc
# yf <- as.vector(Xf %*% signal) + rnorm(n)
# runs <- t(sapply(1:15, function(i) {
#   set.seed(900 + i); knockoff_filter(Xf, yf, Sigma)$sel
# }))
# jac <- c()
# for (i in 1:14) for (j in (i + 1):15)
#   jac <- c(jac, sum(runs[i, ] & runs[j, ]) / max(sum(runs[i, ] | runs[j, ]), 1))
# cat("  15 runs on IDENTICAL data\n")
# cat(sprintf("    selections per run : min %d, max %d\n",
#             min(rowSums(runs)), max(rowSums(runs))))
# cat(sprintf("    pairwise Jaccard   : %.2f\n", mean(jac)))
# freq <- colMeans(runs)
# for (thr in c(0.3, 0.5, 0.8)) {
#   sel <- freq >= thr; ft3 <- fdp_tpr(sel, truth_k)
#   cat(sprintf("    selected in >= %3.0f%% of runs: %3d  FDP %.3f  power %.3f\n",
#               100 * thr, sum(sel), ft3[1], ft3[2]))
# }
#
# ## The data never changed, only the knockoff draw, and the answer moves. That
# ## is uncomfortable to report and impossible to reproduce exactly.
# ##
# ## Aggregating by selection frequency stabilises it, and this is the idea
# ## behind derandomised knockoffs: each run contributes an e-value, and
# ## e-values average legitimately (eq. 36.3), which a p-value cannot do.
# ## Averaging p-values across runs would have no validity at all.
# ##
# ## The general lesson applies well beyond knockoffs: if a method is
# ## randomised, report how much its output moves, and aggregate rather than
# ## reporting one arbitrary draw.

#' ## What to take away
#'
#' 1. Weighted BH (36.1) is valid for any weights **independent of the
#'    p-values**. That is the whole condition, and it is easy to violate.
#' 2. Independent filtering is a binary weight. **IHW** (36.2) is the smooth
#'    version and recovers the power the binary cut discards.
#' 3. BH controls the **expected** FDP. Under dependence a single experiment's
#'    FDP can be far from nominal, and you run one experiment.
#' 4. **E-values** (36.3) are calibrated by their mean, so they combine by
#'    averaging and give FDR control under any dependence, at a power cost.
#' 5. **Knockoffs** (36.5)-(36.6) give conditional feature selection with FDR
#'    control and no independence assumption, at the price of randomness.
#'
#' **Next:** `26_measurement_error_and_regression.R`
