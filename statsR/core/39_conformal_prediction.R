#' ---
#' title: "Module 39 - Conformal prediction and distribution-free uncertainty"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 39, equations (39.1)-(39.5)
#' **Assumes:** Module 31.
#'
#' ## What you will learn
#'
#' 1. **Split conformal** (39.1)-(39.3): a finite-sample coverage guarantee for
#'    any model, with no distributional assumption.
#' 2. Why the $n+1$ and the ceiling in (39.2) are load-bearing.
#' 3. Coverage is **marginal**, not conditional, and what that costs you.
#' 4. **CQR** (39.4) gives intervals that widen where the model is uncertain.
#' 5. **Prediction sets** for classification (39.5), including the empty set.
#' 6. Exchangeability is the assumption, and grouped data breaks it exactly as
#'    it breaks cross-validation.

#+ setup, message = FALSE
suppressPackageStartupMessages({ library(splines); library(mgcv); library(nnet) })
MODULE_NAME <- "39_conformal_prediction"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(28)
ALPHA <- 0.10                      # target 90% coverage

#' $$\hat q=\text{the}\ \lceil (n+1)(1-\alpha)\rceil\ \text{th smallest score}
#'   \qquad (39.2)$$
#'
#' The ceiling and the $n+1$ are what make the guarantee exact rather than
#' approximate.
conformal_quantile <- function(scores, alpha = ALPHA) {
  n <- length(scores)
  k <- ceiling((n + 1) * (1 - alpha))
  if (k > n) return(Inf)   # too few calibration points to promise this level
  sort(scores)[k]
}

#' ## 1. A regression problem where the model is wrong

#+ split-conformal
header("1. Split conformal on a deliberately misspecified model")
#' Heteroscedastic and nonlinear, so a linear model is genuinely wrong.
make_data <- function(n, seed = 0) {
  set.seed(seed)
  x <- runif(n, -3, 3)
  noise <- 0.3 + 0.9 * abs(x)             # spread grows with |x|
  data.frame(x = x, y = sin(1.5 * x) * 2 + 0.5 * x + rnorm(n, 0, noise))
}
tr <- make_data(500, 1); cal <- make_data(500, 2); te <- make_data(4000, 3)
model <- lm(y ~ x, data = tr)             # the wrong model, on purpose
s_cal <- abs(cal$y - predict(model, cal))            # eq. (39.1)
q <- conformal_quantile(s_cal)                       # eq. (39.2)
pr_te <- predict(model, te)
cov_ <- mean(abs(te$y - pr_te) <= q)
#' The textbook alternative: a Gaussian interval from the residual SD.
sd_ <- sd(residuals(model))
z <- qnorm(1 - ALPHA / 2)
cov_g <- mean(abs(te$y - pr_te) <= z * sd_)
cat("  model: linear. truth: sinusoidal with heteroscedastic noise.\n")
cat(sprintf("  target coverage = %.0f%%\n\n", 100 * (1 - ALPHA)))
cat(sprintf("  %-34s%11s%13s\n", "interval", "coverage", "mean width"))
cat(sprintf("  %-34s%10.1f%%%13.3f\n", "Gaussian, residual SD", 100 * cov_g,
            2 * z * sd_))
cat(sprintf("  %-34s%10.1f%%%13.3f\n", "split conformal (39.1)-(39.3)",
            100 * cov_, 2 * q))
cat("\n  The model is badly misspecified and the conformal interval still
  covers at the advertised rate. It does not need the model to be right; it
  only needs the calibration and test data to be exchangeable.

  The Gaussian interval happens to be close here, but its validity rests on
  normal, homoscedastic residuals, neither of which holds. It is right by
  luck, and it has no guarantee to fall back on.\n")

#' ## 2. The finite-sample correction is not cosmetic

#+ finite-sample
header("2. Why (39.2) uses ceil((n+1)(1-alpha)) and not a plain quantile")
cat(sprintf("  %-16s%17s%19s\n", "calibration n", "naive quantile",
            "conformal (39.2)"))
for (n_cal in c(10, 20, 50, 200, 1000)) {
  cn <- cc <- numeric(400)
  for (i in 1:400) {
    ca <- make_data(n_cal, 500 + i); tt <- make_data(600, 9000 + i)
    s <- abs(ca$y - predict(model, ca))
    qn <- quantile(s, 1 - ALPHA, type = 7)             # naive
    qc <- conformal_quantile(s)                        # eq. (39.2)
    d <- abs(tt$y - predict(model, tt))
    cn[i] <- mean(d <= qn); cc[i] <- mean(d <= qc)
  }
  cat(sprintf("  %-16d%16.1f%%%18.1f%%\n", n_cal, 100 * mean(cn),
              100 * mean(cc)))
}
cat(sprintf("\n  Target is %.0f%%. The naive empirical quantile undercovers at
  small n, and the shortfall is worst exactly when you have least data to
  spare. The correction in (39.2) fixes it at every sample size.

  Note the top row. With n = 10, ceil(11 x 0.9) = 10, so the rule uses the
  LARGEST of the ten calibration scores and has nothing in reserve. Ask for
  95%% from those same ten points and ceil(11 x 0.95) = 11 > 10: there is no
  score extreme enough, and the function returns infinity rather than
  pretending.\n", 100 * (1 - ALPHA)))
cat("  smallest calibration set that can promise a given level:\n")
for (a_ in c(0.20, 0.10, 0.05, 0.01))
  cat(sprintf("    %.0f%% coverage needs n >= %d\n", 100 * (1 - a_),
              as.integer(ceiling(1 / a_)) - 1))
cat("\n  A method that says 'I cannot do this with the data you gave me' is
  more useful than one that quietly undercovers. Note the practical
  consequence: a 99% conformal interval needs at least 99 calibration points,
  whatever the model.\n")

#' ## 3. Marginal is not conditional

#+ marginal
header("3. The guarantee is marginal, and that matters")
bins <- quantile(te$x, seq(0, 1, length.out = 6))
cat(sprintf("  Overall coverage %.1f%%. By region of the predictor:\n\n",
            100 * cov_))
cat(sprintf("  %-20s%11s%9s\n", "x range", "coverage", "width"))
for (i in 1:5) {
  m <- te$x >= bins[i] & te$x <= bins[i + 1]
  cat(sprintf("  %-20s%10.1f%%%9.3f\n",
              sprintf("[%+.1f, %+.1f]", bins[i], bins[i + 1]),
              100 * mean(abs(te$y[m] - pr_te[m]) <= q), 2 * q))
}
cat("\n  The average is right and the parts are not. Where the noise is small
  the interval is far too wide; where it is large the interval undercovers
  badly.

  This is exactly what eq. (39.3) promises and no more: coverage averaged over
  the population. If your decision is made per patient, an interval that is
  90% correct on average but 60% correct for the patients who matter is not
  fit for purpose. The next section fixes it.\n")

#' ## 4. CQR: let the width follow the data, eq. (39.4)
#'
#' $$s_i=\max\{\hat q_{\alpha/2}(x_i)-y_i,\ y_i-\hat q_{1-\alpha/2}(x_i)\}
#'   \qquad (39.4)$$
#'
#' Python's module fits the conditional quantiles with gradient boosting. Here
#' we fit them directly by minimising the pinball loss over a spline basis,
#' which is quantile regression written out in full.

#+ cqr
header("4. Conformalised quantile regression (39.4)")
KNOTS <- quantile(tr$x, c(.2, .4, .6, .8))
basis <- function(x) cbind(1, ns(x, knots = KNOTS,
                                 Boundary.knots = range(tr$x)))
qreg_fit <- function(x, y, tau) {
  B <- basis(x)
  loss <- function(b) { r <- y - B %*% b; sum(pmax(tau * r, (tau - 1) * r)) }
  optim(qr.coef(qr(B), y), loss, method = "BFGS",
        control = list(maxit = 2000))$par
}
b_lo <- qreg_fit(tr$x, tr$y, ALPHA / 2)
b_hi <- qreg_fit(tr$x, tr$y, 1 - ALPHA / 2)
qlo <- function(x) as.vector(basis(x) %*% b_lo)
qhi <- function(x) as.vector(basis(x) %*% b_hi)
s_cqr <- pmax(qlo(cal$x) - cal$y, cal$y - qhi(cal$x))    # eq. (39.4)
q_cqr <- conformal_quantile(s_cqr)
lo_c <- qlo(te$x) - q_cqr; hi_c <- qhi(te$x) + q_cqr
cov_c <- mean(te$y >= lo_c & te$y <= hi_c)
near <- abs(te$x) < 1; far <- abs(te$x) > 2
cat(sprintf("  %-26s%11s%13s%16s%16s\n", "method", "coverage", "mean width",
            "width at |x|<1", "width at |x|>2"))
cat(sprintf("  %-26s%10.1f%%%13.3f%16.3f%16.3f\n", "split conformal",
            100 * cov_, 2 * q, 2 * q, 2 * q))
cat(sprintf("  %-26s%10.1f%%%13.3f%16.3f%16.3f\n", "CQR (39.4)", 100 * cov_c,
            mean(hi_c - lo_c), mean((hi_c - lo_c)[near]),
            mean((hi_c - lo_c)[far])))
cat("\n  CQR coverage by region:\n")
for (i in 1:5) {
  m <- te$x >= bins[i] & te$x <= bins[i + 1]
  cat(sprintf("    [%+.1f, %+.1f]  %.1f%%\n", bins[i], bins[i + 1],
              100 * mean(te$y[m] >= lo_c[m] & te$y[m] <= hi_c[m])))
}
cat("\n  Same marginal guarantee, far better behaviour within regions, and a
  narrower interval on average. The width now reports where the model is
  genuinely uncertain, which is the information a reader wants.

  Conformal prediction never makes a model better. It makes the model's
  uncertainty honest, and a better model gives tighter honest intervals.\n")

#' ## 5. Classification: prediction sets, eq. (39.5)
#'
#' $$C(x)=\{k:\ 1-\hat p_k(x)\le\hat q\} \qquad (39.5)$$

#+ sets
header("5. Prediction sets, including the empty one (39.5)")
SEP <- 3.5                         # well separated, so the model is confident
make_cls <- function(n, seed, shift = 0) {
  set.seed(seed)
  y <- sample(0:2, n, replace = TRUE)
  X <- matrix(rnorm(n * 4), n) + shift
  X[, 1] <- X[, 1] + y * SEP               # classes separated on x1
  X[, 2] <- X[, 2] + (y == 2) * 1.0
  list(X = as.data.frame(X), y = y)
}
A <- make_cls(1200, 11); B <- make_cls(1200, 12); C <- make_cls(3000, 13)
clf <- multinom(y ~ ., data = cbind(A$X, y = factor(A$y)), trace = FALSE)
prob <- function(d) predict(clf, newdata = d, type = "probs")
pB <- prob(B$X)
s_cls <- 1 - pB[cbind(seq_along(B$y), B$y + 1)]
q_cls <- conformal_quantile(s_cls)
pC <- prob(C$X)
sets <- (1 - pC) <= q_cls                                # eq. (39.5)
sizes <- rowSums(sets)
covered <- sets[cbind(seq_along(C$y), C$y + 1)]
cat(sprintf("  target %.0f%% coverage, 3 classes\n\n", 100 * (1 - ALPHA)))
cat(sprintf("  coverage of the true label : %.1f%%\n", 100 * mean(covered)))
cat(sprintf("  mean set size              : %.2f\n", mean(sizes)))
cat(sprintf("\n  %-12s%10s%26s\n", "set size", "share",
            "coverage in that group"))
for (k in 0:3) {
  m <- sizes == k
  if (any(m))
    cat(sprintf("  %-12d%9.1f%%%25.1f%%\n", k, 100 * mean(m),
                100 * mean(covered[m])))
}
cat(sprintf("\n  the calibrated threshold is: include class k when p_k >= %.3f\n",
            1 - q_cls))
cat("\n  A set of size 1 is a confident call. Size 2 or 3 would mean the model
  cannot separate those classes for this sample, which a single predicted
  label would have concealed.

  Size 0 is the interesting one: no class reaches the threshold, so the sample
  does not look like anything the model was calibrated on. That is an outlier
  flag you get for free, and a point prediction can never produce it.\n")

#' A point deliberately placed midway between two class centres is exactly the
#' case where no label should be asserted.
amb <- as.data.frame(matrix(rnorm(500 * 4), 500))
amb[, 1] <- SEP * 0.5                    # halfway between class 0 and class 1
s_amb <- (1 - prob(amb)) <= q_cls
cat(sprintf("\n  500 samples placed halfway between two class centres:\n"))
cat(sprintf("    mean set size %.2f, empty sets %.0f%%\n", mean(rowSums(s_amb)),
            100 * mean(rowSums(s_amb) == 0)))
cat(sprintf("    a plain classifier would label every one of them, with mean
    confidence %.2f\n", mean(apply(prob(amb), 1, max))))
cat("\n  The conformal rule abstains on nearly all of them. The plain classifier
  cannot: softmax outputs are forced to sum to 1, so SOMETHING always wins,
  however implausible the sample.\n")

#' ## 6. Exchangeability is the assumption that breaks

#+ shift
header("6. Distribution shift and grouped data")
for (sh in c(0.0, 0.5, 1.5)) {
  S <- make_cls(3000, 14, shift = sh)
  st_ <- (1 - prob(S$X)) <= q_cls
  cat(sprintf("  covariate shift %-4.1f -> coverage %.1f%%\n", sh,
              100 * mean(st_[cbind(seq_along(S$y), S$y + 1)])))
}

#+ grouped
header("6b. Calibrating across groups you will not see again")
#' Grouped data: each group (donor, batch, site) has its own offset AND
#' occupies its own region of the predictor, which is what lets a flexible
#' model memorise the offsets of groups it has seen.
grouped <- function(n_groups, per, seed) {
  set.seed(seed)
  g <- rep(seq_len(n_groups), each = per)
  centre <- seq(-3, 3, length.out = n_groups)[sample(n_groups)][g]
  offset <- rnorm(n_groups, 0, 2.0)[g]           # a per-group shift
  x <- rnorm(n_groups * per, centre, 0.06)
  data.frame(x = x, y = sin(1.5 * x) * 2 + offset +
               rnorm(n_groups * per, 0, 0.5), g = g)
}
NG <- 80; PER <- 40; N_FIT <- 30; N_CAL <- 15
G <- grouped(NG, PER, 21)
#' ONE model, fit on groups 1-30. Only the CALIBRATION differs between the two
#' rows below, so nothing else can explain the gap.
fit_rows <- which(G$g <= N_FIT)
set.seed(7); held <- sample(fit_rows, 300)       # rows held out of the fit
mod <- gam(y ~ s(x, k = 100), data = G[setdiff(fit_rows, held), ])
# WRONG: calibrate on held-out ROWS, whose groups the model trained on.
q_row <- conformal_quantile(abs(G$y[held] - predict(mod, G[held, ])))
# RIGHT: calibrate on whole held-out GROUPS, matching how the model is used.
cal_g <- which(G$g > N_FIT & G$g <= N_FIT + N_CAL)
q_grp <- conformal_quantile(abs(G$y[cal_g] - predict(mod, G[cal_g, ])))
new_g <- G$g > N_FIT + N_CAL                     # groups never seen before
err <- abs(G$y[new_g] - predict(mod, G[new_g, ]))
cat(sprintf("  target %.0f%%, evaluated on groups never seen in training\n\n",
            100 * (1 - ALPHA)))
cat(sprintf("  %-42s%11s%9s\n", "calibration set", "coverage", "width"))
cat(sprintf("  %-42s%10.1f%%%9.2f\n", "held-out ROWS from the training groups",
            100 * mean(err <= q_row), 2 * q_row))
cat(sprintf("  %-42s%10.1f%%%9.2f\n", "whole held-out GROUPS",
            100 * mean(err <= q_grp), 2 * q_grp))
cat("\n  Both rows use the SAME model, so the model is not what differs. The
  row-calibrated interval is roughly three times too narrow and covers around
  half the time instead of 90%.

  The reason: those calibration rows came from groups the model had already
  fitted, so their residuals measure how well it INTERPOLATES within a known
  group, not how well it handles a new one. Calibrating on whole held-out
  groups measures the right thing and restores coverage.

  This is the same error as leaking donors across cross-validation folds
  (Module 31), and it has the same fix. Note also what supplies the precision
  of the calibration: the number of GROUPS, not the number of rows. Three
  calibration groups give an unreliable quantile however many cells each one
  contains.

  Conformal prediction is only as valid as the exchangeability you arrange,
  and an interval that is too narrow is worse than no interval, because it
  looks authoritative.\n")

#' ## 7. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "conformal.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
o <- order(te$x)
plot(te$x, te$y, pch = 16, cex = 0.2, col = adjustcolor("grey40", 0.3),
     xlab = "x", ylab = "y", main = "Split conformal: constant width")
lines(te$x[o], pr_te[o], col = "steelblue", lwd = 2)
lines(te$x[o], (pr_te - q)[o], col = "steelblue", lty = 2)
lines(te$x[o], (pr_te + q)[o], col = "steelblue", lty = 2)
plot(te$x, te$y, pch = 16, cex = 0.2, col = adjustcolor("grey40", 0.3),
     xlab = "x", ylab = "y", main = "CQR: width follows the noise (39.4)")
lines(te$x[o], lo_c[o], col = "firebrick", lty = 2)
lines(te$x[o], hi_c[o], col = "firebrick", lty = 2)
cvs <- sapply(1:5, function(i) {
  m <- te$x >= bins[i] & te$x <= bins[i + 1]
  c(mean(abs(te$y[m] - pr_te[m]) <= q),
    mean(te$y[m] >= lo_c[m] & te$y[m] <= hi_c[m])) })
barplot(cvs, beside = TRUE, names.arg = sprintf("%.1f", (bins[-6] + bins[-1]) / 2),
        col = c("steelblue", "firebrick"), ylim = c(0, 1),
        xlab = "region of x", ylab = "coverage",
        main = "Marginal vs conditional coverage")
abline(h = 1 - ALPHA, lty = 2)
legend("bottomright", c("split conformal", "CQR"), bty = "n",
       fill = c("steelblue", "firebrick"))
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "conformal.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Does a better model give better guarantees?
#'
#' Compare conformal intervals from a badly wrong model, a reasonable model
#' and one that overfits. What changes and what does not?

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# knn_fit <- function(k) function(xtr, ytr, xnew)
#   sapply(xnew, function(x0) mean(ytr[order(abs(xtr - x0))[1:k]]))
# cat(sprintf("  target coverage %.0f%%\n\n", 100 * (1 - ALPHA)))
# cat(sprintf("  %-34s%10s%11s%13s\n", "model", "train R2", "coverage",
#             "mean width"))
# preds <- list(
#   "linear (wrong)" = function(d) predict(lm(y ~ x, data = tr), d),
#   "smoothing spline (gam)" = function(d) predict(gam(y ~ s(x), data = tr), d),
#   "3-nearest neighbours (overfits)" =
#     function(d) knn_fit(3)(tr$x, tr$y, d$x))
# for (nm in names(preds)) {
#   f <- preds[[nm]]
#   r2 <- 1 - sum((tr$y - f(tr))^2) / sum((tr$y - mean(tr$y))^2)
#   qq <- conformal_quantile(abs(cal$y - f(cal)))
#   cv <- mean(abs(te$y - f(te)) <= qq)
#   cat(sprintf("  %-34s%10.3f%10.1f%%%13.3f\n", nm, r2, 100 * cv, 2 * qq))
# }
#
# ## The coverage column is essentially constant. That is the guarantee doing
# ## its job: it does not care how good the model is, only that calibration
# ## and test data are exchangeable.
# ##
# ## The width column is where model quality shows up. A better model produces
# ## smaller residuals on the calibration set, so the interval is tighter at
# ## the same coverage.
# ##
# ## This is the right way to read conformal prediction. It converts model
# ## quality into interval WIDTH rather than into a coverage claim you cannot
# ## check. Note the 3-NN row in particular: it overfits the training data, so
# ## its train R2 looks excellent while its calibration residuals, and
# ## therefore its interval, are not.

#' ### Problem 2: How bad does shift have to be?
#'
#' Conformal coverage assumes exchangeability. Quantify how quickly it
#' degrades under covariate shift, and check whether a wider nominal level
#' rescues it.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# shifted <- function(n, seed, shift) {
#   set.seed(seed)
#   x <- runif(n, -3 + shift, 3 + shift)
#   noise <- 0.3 + 0.9 * abs(x)
#   data.frame(x = x, y = sin(1.5 * x) * 2 + 0.5 * x + rnorm(n, 0, noise))
# }
# cat(sprintf("  %-10s%13s%13s%13s\n", "shift", "alpha=0.1", "alpha=0.05",
#             "alpha=0.01"))
# for (sh in c(0.0, 0.25, 0.5, 1.0, 2.0)) {
#   row <- sapply(c(0.10, 0.05, 0.01), function(a_) {
#     qs <- conformal_quantile(abs(cal$y - predict(model, cal)), a_)
#     S <- shifted(3000, 31, sh)
#     mean(abs(S$y - predict(model, S)) <= qs) })
#   cat(sprintf("  %-10.2f%12.1f%%%12.1f%%%12.1f%%\n", sh, 100*row[1],
#               100*row[2], 100*row[3]))
# }
#
# ## Coverage falls steadily as the test distribution moves away from the
# ## calibration distribution, and asking for a stricter nominal level does
# ## not rescue it: every column degrades together. The guarantee was
# ## conditional on exchangeability, and no choice of alpha restores an
# ## assumption that is false.
# ##
# ## This is the honest limitation. Conformal prediction is not robust to
# ## distribution shift, it is exact under exchangeability. In practice that
# ## means recalibrating on data from the new site, batch or platform, which
# ## requires labelled examples from it. Weighted conformal methods exist for
# ## known covariate shift, but they need the shift to be estimable.

#' ### Problem 3: Conformal risk for a decision
#'
#' A prediction set is only useful if it changes what you do. Build a rule
#' that acts when the set contains exactly one class, defers when the set is
#' ambiguous, and measure the error rate among the acted-on cases.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# A2 <- make_cls(1500, 41); B2 <- make_cls(1500, 42); C2 <- make_cls(4000, 43)
# clf2 <- multinom(y ~ ., data = cbind(A2$X, y = factor(A2$y)), trace = FALSE)
# pB2 <- predict(clf2, newdata = B2$X, type = "probs")
# pC2 <- predict(clf2, newdata = C2$X, type = "probs")
# hard <- max.col(pC2) - 1
# cat(sprintf("  %-9s%11s%11s%19s%18s\n", "alpha", "acted on", "deferred",
#             "error | acted on", "error if forced"))
# for (a_ in c(0.20, 0.10, 0.05, 0.01)) {
#   s2 <- 1 - pB2[cbind(seq_along(B2$y), B2$y + 1)]
#   qq <- conformal_quantile(s2, a_)
#   sets2 <- (1 - pC2) <= qq
#   single <- rowSums(sets2) == 1
#   pred1 <- max.col(sets2)[single] - 1
#   cat(sprintf("  %-9.2f%10.1f%%%10.1f%%%18.1f%%%17.1f%%\n", a_,
#               100 * mean(single), 100 * (1 - mean(single)),
#               100 * mean(pred1 != C2$y[single]),
#               100 * mean(hard != C2$y)))
# }
#
# ## The last column is the error rate if you force a single label for every
# ## sample, which is what a normal classifier does. The fourth column is the
# ## error rate among the cases the conformal rule was willing to act on.
# ##
# ## Acting only on singleton sets buys a much lower error rate, paid for by
# ## deferring the hard cases. As alpha shrinks, fewer cases get a singleton
# ## set and the error among those falls further.
# ##
# ## That trade is the practical value of conformal prediction in a clinical
# ## or screening setting: it identifies WHICH predictions are trustworthy
# ## rather than reporting one confidence number for all of them. The deferred
# ## cases are not a failure, they are the cases that should go to a human.

#' ## What to take away
#'
#' 1. Split conformal gives **finite-sample, distribution-free,
#'    model-agnostic** coverage (39.3). The only requirement is
#'    exchangeability.
#' 2. The $n+1$ and the ceiling in (39.2) are what make it exact. A plain
#'    empirical quantile undercovers.
#' 3. Coverage is **marginal**. Use CQR (39.4) when you need the width to
#'    reflect local uncertainty.
#' 4. **Prediction sets** (39.5) express ambiguity, and the empty set flags a
#'    sample unlike anything you calibrated on.
#' 5. A better model does not improve coverage, it narrows the interval.
#' 6. Grouped data and distribution shift break exchangeability, and no choice
#'    of $\alpha$ repairs it.
#'
#' **Next:** `40_simulation_and_benchmarking.R`
