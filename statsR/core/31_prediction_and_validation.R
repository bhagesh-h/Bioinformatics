#' ---
#' title: "Module 31 - Statistical learning and biomarker prediction"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 31, equations (31.1)-(31.12)
#'
#' Prediction is a DIFFERENT objective from association, with different
#' validation standards.
#'
#' ## What you will learn
#'
#' 1. The leaks that destroy biomarker studies -- each on data with NO signal.
#' 2. Group-aware splitting: donors, not cells.
#' 3. Nested cross-validation (31.4).
#' 4. Discrimination (31.8) vs **calibration** (31.9)-(31.11) vs utility (31.12).
#' 5. Why lasso-selected features are unstable (31.6).

#+ setup, message = FALSE
suppressPackageStartupMessages(library(MASS))
MODULE_NAME <- "31_prediction_and_validation"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

## --- Small helpers: base R has no caret/glmnet, so we build what we need. ---
auc <- function(labels, scores) {
  ## AUC = P(score_case > score_control), eq. (31.8) = eq. (7.7).
  r <- rank(scores); n1 <- sum(labels == 1); n0 <- sum(labels == 0)
  (sum(r[labels == 1]) - n1*(n1+1)/2) / (n1*n0)
}
make_folds <- function(n, k, groups = NULL, seed = 1) {
  set.seed(seed)
  if (is.null(groups)) return(sample(rep_len(1:k, n)))
  ## GROUP-AWARE: whole groups go to one fold, never split across folds.
  g <- unique(groups); gf <- sample(rep_len(1:k, length(g)))
  gf[match(groups, g)]
}

#' ## 1. Leak #1 -- feature selection outside the fold

#+ leak-selection
header("1. Feature selection outside the fold (the classic leak)")

fast_tstat <- function(X, y) {
  ## Vectorised two-sample t statistic for every column. Looping t.test() over
  ## 5,000 features x 5 folds x 20 replicates would take minutes.
  a <- X[y == 0,, drop = FALSE]; b <- X[y == 1,, drop = FALSE]
  n1 <- nrow(a); n2 <- nrow(b)
  m1 <- colMeans(a); m2 <- colMeans(b)
  v1 <- matrixStats_colVar(a); v2 <- matrixStats_colVar(b)
  (m1 - m2) / sqrt(v1/n1 + v2/n2)
}
matrixStats_colVar <- function(M) {
  ## Base-R column variances without a dependency.
  mu <- colMeans(M)
  colSums((M - rep(mu, each = nrow(M)))^2) / (nrow(M) - 1)
}

leak_experiment <- function(n = 100, p = 2000, k_sel = 20, n_rep = 20) {
  leaky <- honest <- numeric(n_rep)
  for (r in seq_len(n_rep)) {
    set.seed(5000 + r)
    X <- matrix(rnorm(n*p), n)
    y <- rbinom(n, 1, 0.5)                 # PURE NOISE - no signal whatsoever
    fold <- make_folds(n, 5, seed = r)
    ## WRONG: rank features using ALL the data, then cross-validate.
    top <- order(-abs(fast_tstat(X, y)))[1:k_sel]
    leaky[r] <- mean(sapply(1:5, function(kk) {
      tr <- fold != kk; te <- !tr
      if (length(unique(y[te])) < 2) return(NA)
      fit <- suppressWarnings(glm(y[tr] ~ X[tr, top], family = binomial()))
      auc(y[te], as.vector(cbind(1, X[te, top]) %*% coef(fit)))
    }), na.rm = TRUE)
    ## RIGHT: re-rank features INSIDE each fold.
    honest[r] <- mean(sapply(1:5, function(kk) {
      tr <- fold != kk; te <- !tr
      if (length(unique(y[te])) < 2) return(NA)
      sel <- order(-abs(fast_tstat(X[tr, ], y[tr])))[1:k_sel]
      fit <- suppressWarnings(glm(y[tr] ~ X[tr, sel], family = binomial()))
      auc(y[te], as.vector(cbind(1, X[te, sel]) %*% coef(fit)))
    }), na.rm = TRUE)
  }
  c(leaky = mean(leaky), honest = mean(honest),
    se = sd(honest)/sqrt(n_rep))
}
r1 <- leak_experiment()
cat("  Outcome is a COIN FLIP, so the honest AUC must be 0.5.\n")
cat("  A single 5-fold CV at n=100 has an SE near 0.06, so we average over 20\n")
cat("  independent null datasets - the same discipline as a power study.\n\n")
cat(sprintf("    select on ALL data, then CV  : AUC = %.3f   <- FANTASY\n", r1["leaky"]))
cat(sprintf("    selection INSIDE each fold   : AUC = %.3f   <- honest (+/-%.3f)\n",
            r1["honest"], 1.96*r1["se"]))
cat("\n  The leaky number is what gets published as a '90% accurate biomarker\n")
cat("  signature'. Nothing in the data supports it. The honest number sits at\n")
cat("  chance, exactly as it must.\n")

#' ## 1b. Does OUTCOME-BLIND preprocessing leak too?
#'
#' Section 1 selected features using the outcome, which is catastrophic. A
#' widely repeated claim is that fitting any preprocessing on all the rows
#' leaks as well. That is worth measuring rather than assuming.

#+ preprocessing-leak
header("1b. Unsupervised preprocessing fitted on all rows")
#' A single 5-fold CV at n = 60 has a standard error of roughly 0.06 on the
#' AUC, larger than the effect we are chasing, so replicate over many
#' independent NULL datasets (the discipline of Module 40).
auc_simple <- function(lab, score) {
  r <- rank(score); n1 <- sum(lab == 1); n0 <- sum(lab == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  (sum(r[lab == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
null_auc <- function(p_feat, n = 60, n_rep = 60, k = 20) {
  leaky <- honest <- numeric(n_rep)
  for (rep_ in seq_len(n_rep)) {
    set.seed(10000 + rep_)
    X <- matrix(rnorm(n * p_feat), n)
    y <- rbinom(n, 1, 0.5)
    fold <- sample(rep_len(1:5, n))
    ## LEAKY: PCA sees every row, including the held-out ones.
    pc_all <- prcomp(X, center = TRUE, scale. = TRUE)$x[, 1:k, drop = FALSE]
    pr <- rep(NA_real_, n)
    for (kk in 1:5) {
      tr <- fold != kk
      if (length(unique(y[tr])) < 2) next
      m <- suppressWarnings(glm(y ~ ., binomial(),
                                data.frame(y = y[tr], pc_all[tr, ])))
      pr[!tr] <- predict(m, data.frame(pc_all[!tr, , drop = FALSE]),
                         type = "response")
    }
    leaky[rep_] <- auc_simple(y[!is.na(pr)], pr[!is.na(pr)])
    ## HONEST: PCA refitted inside each fold, test rows projected onto it.
    pr <- rep(NA_real_, n)
    for (kk in 1:5) {
      tr <- fold != kk
      if (length(unique(y[tr])) < 2) next
      p_ <- prcomp(X[tr, , drop = FALSE], center = TRUE, scale. = TRUE)
      Ztr <- p_$x[, 1:k, drop = FALSE]
      Zte <- predict(p_, X[!tr, , drop = FALSE])[, 1:k, drop = FALSE]
      m <- suppressWarnings(glm(y ~ ., binomial(),
                                data.frame(y = y[tr], Ztr)))
      colnames(Zte) <- colnames(Ztr)
      pr[!tr] <- predict(m, data.frame(Zte), type = "response")
    }
    honest[rep_] <- auc_simple(y[!is.na(pr)], pr[!is.na(pr)])
  }
  c(mean(leaky, na.rm = TRUE), mean(honest, na.rm = TRUE),
    sd(leaky, na.rm = TRUE) / sqrt(n_rep))
}
cat("  Pure-noise data (outcome is a coin flip), so the honest AUC is 0.5.\n")
cat("  Averaged over 60 independent datasets to beat the CV noise:\n\n")
cat(sprintf("  %14s%18s%18s%11s%8s\n", "p (features)", "PCA on all rows",
            "PCA in the fold", "optimism", "MC SE"))
for (pf in c(200, 800)) {
  r_ <- null_auc(pf)
  cat(sprintf("  %14s%18.4f%18.4f%+11.4f%8.4f\n",
              format(pf, big.mark = ","), r_[1], r_[2], r_[1] - r_[2], r_[3]))
}
cat("
  (An overfit model can score BELOW 0.5 out of fold, which is why some entries
  are near 0.45: with 20 PCs fitted to 60 noise samples the model has learned
  structure that does not generalise, so it is actively anti-predictive on the
  held-out rows. That is expected, not a bug.)

  READ THE RESULT HONESTLY: the optimism is within Monte-Carlo error of ZERO.
  Fitting an UNSUPERVISED PCA on all rows did not measurably inflate
  performance here. That is the correct conclusion from this experiment, and
  it is worth stating plainly rather than asserting a leak the data do not
  show.

  So distinguish two classes of preprocessing:

  (a) OUTCOME-BLIND steps - scaling, PCA, quantile normalisation,
      unsupervised batch correction. They never see y, so the only thing they
      can leak is the test FEATURE distribution, and as measured above that is
      a small effect. Still put them inside the fold: it costs nothing, and
      steps that borrow across ROWS (kNN or iterative imputation,
      sample-to-sample quantile normalisation) leak considerably more than
      PCA does.

  (b) OUTCOME-AWARE steps - feature selection by association with y,
      normalising cases and controls separately, choosing a cut-point to
      maximise separation. These are CATASTROPHIC, as section 1 showed.

  The distinction matters because treating every preprocessing step as equally
  dangerous leads people to distrust the wrong things and miss the real leak.\n")

#' ## 2. Leak #2 -- splitting a donor's cells across folds

#+ leak-donor
header("2. Donor leakage: the most common bioinformatics leak")
cv_auc <- function(X, y, fold) {
  vals <- sapply(sort(unique(fold)), function(k) {
    tr <- fold != k; te <- !tr
    if (length(unique(y[te])) < 2 || length(unique(y[tr])) < 2) return(NA)
    fit <- suppressWarnings(glm(y[tr] ~ X[tr, ], family = binomial()))
    auc(y[te], as.vector(cbind(1, X[te, ]) %*% coef(fit)))
  })
  mean(vals, na.rm = TRUE)
}
donor_experiment <- function(n_donors = 40, cells_per = 25, n_rep = 15) {
  rand <- grp <- numeric(n_rep)
  for (r in seq_len(n_rep)) {
    set.seed(6000 + r)
    donor <- rep(seq_len(n_donors), each = cells_per)
    sig <- matrix(rnorm(n_donors*30, 0, 2), n_donors)   # donor signature
    Xc <- sig[donor, ] + matrix(rnorm(n_donors*cells_per*30), ncol = 30)
    y_don <- sample(rep(0:1, each = n_donors/2))        # outcome per donor, RANDOM
    yc <- y_don[donor]
    rand[r] <- cv_auc(Xc, yc, make_folds(nrow(Xc), 5, seed = r))
    grp[r]  <- cv_auc(Xc, yc, make_folds(nrow(Xc), 5, groups = donor, seed = r))
  }
  c(random = mean(rand), grouped = mean(grp), se = sd(grp)/sqrt(n_rep))
}
r2 <- donor_experiment()
cat("  40 donors x 25 cells; the outcome is assigned PER DONOR at random, so\n")
cat("  the honest AUC is again 0.5. Averaged over 15 datasets:\n\n")
cat(sprintf("    random 5-fold CV over CELLS  : AUC = %.3f   <- memorises donors\n",
            r2["random"]))
cat(sprintf("    group-aware CV by DONOR      : AUC = %.3f   <- honest (+/-%.3f)\n",
            r2["grouped"], 1.96*r2["se"]))
## Verify no donor straddles a fold boundary.
set.seed(1); dtest <- rep(1:40, each = 25)
ftest <- make_folds(length(dtest), 5, groups = dtest, seed = 1)
cat(sprintf("    any donor in both train and test? %s\n",
            any(sapply(1:5, function(k)
              length(intersect(dtest[ftest == k], dtest[ftest != k])) > 0))))
cat("\n  Random splitting puts near-duplicate cells from the same donor on both\n")
cat("  sides of the fold boundary. The model learns 'which donor is this?',\n")
cat("  which perfectly predicts the label - and generalises to nobody.\n")
cat("  The same applies to technical replicates and repeated visits.\n")

#' ## 3. Leak #3 -- tuning and reporting on the same CV, eq. (31.4)

#+ nested-cv
header("3. Nested cross-validation (31.4)")
set.seed(2103)
n <- 120; p <- 200
X <- matrix(rnorm(n*p), n)
beta <- numeric(p); beta[1:5] <- 0.35                    # a small REAL signal
y <- as.integer(X %*% beta + rnorm(n) > 0)

ridge_fit <- function(X, y, lambda) {
  ## Ridge-penalised logistic via IRLS with an L2 penalty, eq. (31.5).
  Xd <- cbind(1, X); b <- rep(0, ncol(Xd))
  P <- diag(c(0, rep(lambda, ncol(X))))                  # do not penalise intercept
  for (i in 1:40) {
    eta <- as.vector(Xd %*% b); mu <- plogis(eta); W <- pmax(mu*(1-mu), 1e-6)
    z <- eta + (y - mu)/W
    b_new <- solve(crossprod(Xd*sqrt(W)) + P, crossprod(Xd*W, z))
    if (max(abs(b_new - b)) < 1e-8) { b <- b_new; break }
    b <- b_new
  }
  as.vector(b)
}
lambdas <- c(0.1, 1, 10, 100, 1000)
inner_cv <- function(X, y, lambda, seed = 7) {
  f <- make_folds(nrow(X), 5, seed = seed)
  mean(sapply(1:5, function(k) {
    tr <- f != k; te <- !tr
    if (length(unique(y[te])) < 2) return(NA)
    auc(y[te], as.vector(cbind(1, X[te, ]) %*% ridge_fit(X[tr, ], y[tr], lambda)))
  }), na.rm = TRUE)
}
inner_scores <- sapply(lambdas, function(l) inner_cv(X, y, l))
auc_flat <- max(inner_scores)                            # the score we TUNED on
outer <- make_folds(n, 5, seed = 11)
auc_nested <- mean(sapply(1:5, function(k) {
  tr <- outer != k; te <- !tr
  best <- lambdas[which.max(sapply(lambdas, function(l)
    inner_cv(X[tr, ], y[tr], l, seed = 100 + k)))]
  auc(y[te], as.vector(cbind(1, X[te, ]) %*% ridge_fit(X[tr, ], y[tr], best)))
}))
cat(sprintf("  best inner-CV score (used for tuning) : AUC = %.3f   <- optimistic\n",
            auc_flat))
cat(sprintf("  nested CV (outer loop)                : AUC = %.3f   <- honest\n",
            auc_nested))
cat(sprintf("  optimism = %+.3f\n", auc_flat - auc_nested))
cat("\n  Reporting the CV score you SELECTED ON is selection bias. The outer\n")
cat("  loop estimates performance; the inner loop tunes.\n")

#' ## 4. Discrimination vs calibration vs utility, eq. (31.8)-(31.12)

#+ calibration
header("4. AUC is not enough (31.8)-(31.12)")
set.seed(2105)
n_cal <- 4000
zc <- rnorm(n_cal)
p_true <- plogis(-3 + 1.5*zc)                       # ~6% prevalence
y_cal <- rbinom(n_cal, 1, p_true)
models <- list(
  "well calibrated"              = p_true,
  "over-confident (2x log-odds)" = plogis(2*qlogis(p_true)),
  "shifted (+1 on log-odds)"     = plogis(qlogis(p_true) + 1))
calibration_slope <- function(y, p) {                            # eq. (31.11)
  lo <- qlogis(pmin(pmax(p, 1e-9), 1 - 1e-9))
  coef(glm(y ~ lo, family = binomial()))[2]
}
net_benefit <- function(y, p, pt) {                              # eq. (31.12)
  pred <- p >= pt
  sum(pred & y == 1)/length(y) - (sum(pred & y == 0)/length(y)) * (pt/(1-pt))
}
brier <- function(y, p) mean((p - y)^2)                          # eq. (31.9)
cat(sprintf("  simulated cohort: n = %d, prevalence = %.3f\n\n", n_cal, mean(y_cal)))
cat(sprintf("  %-30s%8s%9s%13s%10s\n", "model", "AUC", "Brier", "calib slope", "NB@0.10"))
for (nm in names(models)) {
  pm <- models[[nm]]
  cat(sprintf("  %-30s%8.3f%9.4f%13.3f%10.4f\n", nm, auc(y_cal, pm), brier(y_cal, pm),
              calibration_slope(y_cal, pm), net_benefit(y_cal, pm, 0.10)))
}
cat("\n  All three have IDENTICAL AUC - ranking is unchanged by a monotone\n")
cat("  transformation of the probabilities. Only Brier and the calibration\n")
cat("  slope reveal that two of them produce WRONG PROBABILITIES.\n")
cat("  Calibration slope: 1 = perfect, <1 = predictions too extreme (the usual\n")
cat("  overfitting signature, and the shrinkage factor to apply).\n")
cat(sprintf("\n  Prevalence-dependence of AUC vs PPV at a fixed threshold:\n"))
cat(sprintf("  %12s%8s%17s\n", "prevalence", "AUC", "PPV at p>=0.10"))
for (shift in c(-3, -1.5, 0)) {
  pt_ <- plogis(shift + 1.5*zc); yy <- rbinom(n_cal, 1, pt_)
  sel <- pt_ >= 0.10
  cat(sprintf("  %12.3f%8.3f%17.3f\n", mean(yy), auc(yy, pt_),
              if (any(sel)) mean(yy[sel]) else NA))
}
cat("  AUC is PREVALENCE-INDEPENDENT - a virtue for comparing models and a\n")
cat("  trap for judging usefulness.\n")

#' ## 5. Penalised regression and selection stability, eq. (31.5)-(31.7)

#+ stability
header("5. Correlated features make lasso selection unstable (31.6)-(31.7)")
soft <- function(a, k) sign(a) * pmax(abs(a) - k, 0)
lasso_cd <- function(X, y, lambda, n_iter = 200) {
  ## Coordinate descent for the lasso, eq. (31.6). Standardise first.
  Xs <- scale(X); ys <- y - mean(y)
  n <- nrow(Xs); b <- rep(0, ncol(Xs))
  for (it in 1:n_iter) {
    for (j in seq_along(b)) {
      r <- ys - Xs[, -j, drop = FALSE] %*% b[-j]
      b[j] <- soft(sum(Xs[, j]*r)/n, lambda)
    }
  }
  b
}
enet_cd <- function(X, y, lambda, alpha = 0.3, n_iter = 200) {
  Xs <- scale(X); ys <- y - mean(y); n <- nrow(Xs); b <- rep(0, ncol(Xs))
  for (it in 1:n_iter) for (j in seq_along(b)) {
    r <- ys - Xs[, -j, drop = FALSE] %*% b[-j]
    b[j] <- soft(sum(Xs[, j]*r)/n, lambda*alpha) / (1 + lambda*(1-alpha))
  }
  b
}
set.seed(2106)
n <- 120; p <- 60
latent <- matrix(rnorm(n*10), n)
Xg <- cbind(sapply(1:30, function(k) latent[, ((k-1) %/% 6) + 1] + rnorm(n, 0, 0.35)),
            matrix(rnorm(n*30), n))
bt <- numeric(p); bt[c(1, 7, 13)] <- 1.5              # one per correlated group
yv <- as.vector(Xg %*% bt + rnorm(n))
sel_freq <- function(fn, n_boot = 40, ...) {
  counts <- numeric(p)
  for (b in 1:n_boot) {
    set.seed(b); idx <- sample(n, n, TRUE)
    counts <- counts + (abs(fn(Xg[idx, ], yv[idx], ...)) > 1e-8)
  }
  counts / n_boot
}
f_lasso <- sel_freq(lasso_cd, lambda = 0.15)
f_enet <- sel_freq(enet_cd, lambda = 0.15, alpha = 0.3)
cat("  Features 1-6 are one correlated group; only feature 1 is truly causal.\n")
cat(sprintf("  %9s%18s%24s\n", "feature", "lasso sel. freq", "elastic-net sel. freq"))
for (j in 1:6)
  cat(sprintf("  %9d%18.2f%24.2f%s\n", j, f_lasso[j], f_enet[j],
              if (j == 1) " *causal*" else ""))
cat(sprintf("\n  mean selection frequency of the TRUE features (1, 7, 13):\n"))
cat(sprintf("    lasso %.2f, elastic net %.2f\n",
            mean(f_lasso[c(1,7,13)]), mean(f_enet[c(1,7,13)])))
cat("\n  Lasso picks ONE arbitrary member of each correlated group, and which\n")
cat("  one changes across resamples. Elastic net's ridge component spreads\n")
cat("  weight across the group (the 'grouping effect'). NEVER present\n")
cat("  lasso-selected features as 'the important genes' without a stability\n")
cat("  analysis.\n")

#' ## 6. Figure

#+ figure
png(file.path(OUT, "prediction.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "FPR", ylab = "TPR",
     main = "Eq. (31.8): identical AUCs")
cols <- c("steelblue", "darkorange", "forestgreen")
for (i in seq_along(models)) {
  pm <- models[[i]]; o <- order(-pm)
  lines(cumsum(1 - y_cal[o])/sum(1 - y_cal), cumsum(y_cal[o])/sum(y_cal),
        col = cols[i], lwd = 2)
}
abline(0, 1, lty = 2)
legend("bottomright", sapply(strsplit(names(models), " \\("), `[`, 1),
       col = cols, lwd = 2, bty = "n", cex = 0.6)
plot(NA, xlim = c(0, 0.6), ylim = c(0, 0.6), xlab = "mean predicted probability",
     ylab = "observed", main = "Eq. (31.11): calibration plot")
for (i in seq_along(models)) {
  pm <- models[[i]]
  bins <- cut(pm, quantile(pm, 0:10/10), include.lowest = TRUE)
  lines(tapply(pm, bins, mean), tapply(y_cal, bins, mean), type = "b",
        pch = 16, cex = 0.6, col = cols[i])
}
abline(0, 1, lty = 2)
pts <- seq(0.01, 0.4, length.out = 60)
plot(NA, xlim = range(pts), ylim = c(-0.02, 0.06), xlab = "threshold probability",
     ylab = "net benefit", main = "Eq. (31.12): decision curve")
for (i in seq_along(models))
  lines(pts, sapply(pts, function(t) net_benefit(y_cal, models[[i]], t)),
        col = cols[i], lwd = 2)
lines(pts, sapply(pts, function(t) net_benefit(y_cal, rep(1, n_cal), t)), lty = 3)
abline(h = 0)
legend("topright", c("models", "treat all", "treat none"),
       col = c("steelblue", "black", "black"), lty = c(1, 3, 1), bty = "n", cex = 0.6)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "prediction.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 31)
#'
#' 1. **Split by donor/patient, always.** Group-aware CV is not optional.
#' 2. Put every learned step inside the fold.
#' 3. Use nested CV whenever you tune.
#' 4. Report discrimination AND calibration AND net benefit.
#' 5. Distinguish apparent, internally validated, and externally validated
#'    performance. Only the third predicts the next cohort.
#'
#' **Next:** `32_causal_inference.R`
