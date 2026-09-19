#' ---
#' title: "Module 13 - Generalised linear models"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 13, equations (13.1)-(13.12)
#'
#' ## What you will learn
#'
#' 1. The exponential family (13.1)-(13.2): mean and variance from one function.
#' 2. **IRLS (13.3)-(13.4) from scratch** -- a GLM is a weighted linear model.
#' 3. Logistic regression (13.6): odds ratios, non-collapsibility, separation
#'    and Firth's fix (13.7).
#' 4. **Offsets (13.8)**: counts vs rates. The most common GLM error.
#' 5. Overdispersion (13.9): quasi-Poisson vs negative binomial.
#' 6. Wald vs LRT (13.11), and why the LRT is safer.

#+ setup, message = FALSE
suppressPackageStartupMessages(library(MASS))
MODULE_NAME <- "13_generalized_linear_models"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. The family table

#+ families
header("1. Family, link, variance function (13.1)-(13.2)")
print(data.frame(
  family = c("Gaussian", "Binomial", "Poisson", "Neg. binomial", "Gamma"),
  V_mu = c("1", "mu(1-mu)", "mu", "mu + phi*mu^2", "mu^2"),
  canonical_link = c("identity", "logit", "log", "log", "inverse"),
  exp_beta_means = c("mean difference", "log odds ratio", "log rate ratio",
                     "log fold change", "log ratio of means")), row.names = FALSE)

#' ## 2. IRLS from scratch, eq. (13.3)-(13.5)

#+ irls
header("2. Implementing Fisher scoring (13.3)-(13.5)")
irls <- function(X, y, family = "binomial", offset = NULL, max_iter = 50, tol = 1e-10) {
  n <- nrow(X); p <- ncol(X)
  if (is.null(offset)) offset <- rep(0, n)
  beta <- rep(0, p)
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta) + offset
    if (family == "binomial") {
      mu <- plogis(eta); V <- mu * (1 - mu); gprime <- 1 / pmax(V, 1e-12)
    } else {
      mu <- exp(eta); V <- mu; gprime <- 1 / pmax(mu, 1e-12)
    }
    z <- (eta - offset) + (y - mu) * gprime                  # eq. (13.3)
    w <- 1 / (gprime^2 * pmax(V, 1e-12))                     # eq. (13.3)
    ## Weighted least squares step (13.4), solved stably via QR.
    beta_new <- qr.coef(qr(X * sqrt(w)), z * sqrt(w))
    if (max(abs(beta_new - beta)) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  cov <- chol2inv(chol(crossprod(X * sqrt(w))))               # eq. (13.5)
  list(beta = beta, se = sqrt(diag(cov)), iterations = it)
}
set.seed(1301)
n <- 400
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
yb <- rbinom(n, 1, plogis(-0.5 + 1.1*x1 + 0.8*x2))
Xd <- cbind(1, x1, x2)
mine <- irls(Xd, yb, "binomial")
theirs <- glm(yb ~ x1 + x2, family = binomial())
cat(sprintf("  logistic - converged in %d IRLS iterations\n", mine$iterations))
cat(sprintf("  %-12s%12s%11s%19s%17s\n", "term", "mine beta", "mine SE",
            "glm() beta", "glm() SE"))
for (i in 1:3)
  cat(sprintf("  %-12s%12.6f%11.6f%19.6f%17.6f\n", c("intercept","x1","x2")[i],
              mine$beta[i], mine$se[i], coef(theirs)[i],
              coef(summary(theirs))[i, "Std. Error"]))
cat(sprintf("  max |difference| = %.2e\n", max(abs(mine$beta - coef(theirs)))))
expo <- runif(n, 0.5, 2)
yp <- rpois(n, expo * exp(0.3 + 0.7*x1))
mp <- irls(Xd, yp, "poisson", offset = log(expo))
tp <- glm(yp ~ x1 + x2, family = poisson(), offset = log(expo))
cat(sprintf("\n  Poisson+offset - max |mine - glm()| = %.2e\n",
            max(abs(mp$beta - coef(tp)))))

#' ## 3. Logistic regression: odds ratios and their traps, eq. (13.6)

#+ logistic
header("3. Odds ratios, risk ratios, non-collapsibility (13.6)")
set.seed(1302)
N <- 30000
z <- rnorm(N)                       # a PROGNOSTIC factor, NOT a confounder
a <- rbinom(N, 1, 0.5)              # RANDOMISED exposure
yy <- rbinom(N, 1, plogis(-0.5 + 0.9*a + 1.5*z))
m_crude <- glm(yy ~ a, family = binomial())
m_adj <- glm(yy ~ a + z, family = binomial())
cat(sprintf("  TRUE conditional log-OR for a (by construction) = 0.900\n"))
cat(sprintf("  crude    log-OR = %.4f  (OR = %.3f)\n", coef(m_crude)["a"],
            exp(coef(m_crude)["a"])))
cat(sprintf("  adjusted log-OR = %.4f  (OR = %.3f)\n", coef(m_adj)["a"],
            exp(coef(m_adj)["a"])))
cat("\n  The exposure was RANDOMISED, so z cannot be a confounder - yet the\n")
cat("  estimate changed. This is NON-COLLAPSIBILITY: logistic coefficients\n")
cat("  shift when covariates are added even WITHOUT confounding. Do not read\n")
cat("  a shift in the OR as evidence of confounding.\n")
cat("\n  OR is not RR unless the outcome is rare:\n")
cat(sprintf("  %15s%7s%13s\n", "baseline risk", "OR", "implied RR"))
for (p0 in c(0.01, 0.05, 0.20, 0.40, 0.60))
  cat(sprintf("  %15.2f%7.1f%13.3f\n", p0, 2, 2 / (1 - p0 + p0*2)))

#+ separation
cat("\n--- Complete separation: the MLE diverges ---\n")
xs <- c(-3, -2, -1.5, -1, 1, 1.5, 2, 3); ys <- c(0,0,0,0,1,1,1,1)
f_sep <- suppressWarnings(glm(ys ~ xs, family = binomial()))
cat(sprintf("  unpenalised MLE slope = %10.3f  SE = %10.3f   <- both exploding\n",
            coef(f_sep)[2], coef(summary(f_sep))[2, "Std. Error"]))
firth_logit <- function(X, y, max_iter = 200, tol = 1e-8) {
  ## stats.md eq. (13.7): Firth's penalised likelihood (Jeffreys prior).
  ## Modified score: U*(b)_j = U(b)_j + sum_i h_ii (0.5 - mu_i) x_ij.
  beta <- rep(0, ncol(X))
  for (i in seq_len(max_iter)) {
    mu <- plogis(as.vector(X %*% beta)); W <- mu * (1 - mu)
    XtWX <- crossprod(X * sqrt(W)); inv <- MASS::ginv(XtWX)
    h <- diag((X * W) %*% inv %*% t(X))
    U <- crossprod(X, y - mu + h * (0.5 - mu))
    step <- inv %*% U; beta <- beta + step
    if (max(abs(step)) < tol) break
  }
  list(beta = as.vector(beta), se = sqrt(diag(MASS::ginv(crossprod(X * sqrt(W))))))
}
fb <- firth_logit(cbind(1, xs), ys)
cat(sprintf("  Firth penalised slope = %10.3f  SE = %10.3f   <- finite\n",
            fb$beta[2], fb$se[2]))
cat("  Separation is common with small n and rare outcomes. Firth is the fix.\n")

#' ## 4. Offsets: counts vs rates, eq. (13.8)

#+ offsets
header("4. The offset: the most common GLM error (13.8)")
set.seed(1303)
n <- 200
group <- rbinom(n, 1, 0.5)
libsize <- exp(rnorm(n, ifelse(group == 1, 11.5, 10.8), 0.25))  # UNBALANCED depth
TRUE_LOG_RR <- 0                                                 # NO real difference
counts <- rpois(n, libsize * exp(-6 + TRUE_LOG_RR * group))
dd <- data.frame(y = counts, g = group, lib = libsize)
m_no <- glm(y ~ g, data = dd, family = poisson())
m_off <- glm(y ~ g, data = dd, family = poisson(), offset = log(lib))
m_cov <- glm(y ~ g + log(lib), data = dd, family = poisson())
cat(sprintf("  TRUE log rate ratio = %.3f\n", TRUE_LOG_RR))
cat(sprintf("  %-34s%10s%11s\n", "model", "log RR", "p"))
cat(sprintf("  %-34s%10.4f%11.3e   <- spurious!\n", "y ~ g  (NO offset)",
            coef(m_no)["g"], coef(summary(m_no))["g", "Pr(>|z|)"]))
cat(sprintf("  %-34s%10.4f%11.4f   <- correct\n", "y ~ g + offset(log lib)",
            coef(m_off)["g"], coef(summary(m_off))["g", "Pr(>|z|)"]))
cat(sprintf("  %-34s%10.4f%11.4f\n", "y ~ g + log(lib) as covariate",
            coef(m_cov)["g"], coef(summary(m_cov))["g", "Pr(>|z|)"]))
cat(sprintf("       (its log(lib) coefficient = %.4f, freely estimated rather\n",
            coef(m_cov)["log(lib)"]))
cat("        than FIXED at 1, which is what makes an offset an offset)\n")
cat("\n  Without the offset the model detects a difference in DEPTH and reports\n")
cat("  it as a difference in BIOLOGY. This is why RNA-seq size factors enter\n")
cat("  as offsets (stats.md eq. 20.1).\n")

#' ## 5. Overdispersion, eq. (13.9)

#+ overdispersion
header("5. Detecting and handling overdispersion (13.9)")
set.seed(1304)
n <- 150; PHI <- 0.4
xg <- rbinom(n, 1, 0.5)
y_od <- rnbinom(n, mu = exp(2.5 + 0.35*xg), size = 1/PHI)
do_ <- data.frame(y = y_od, g = xg)
m_pois <- glm(y ~ g, data = do_, family = poisson())
phi_hat <- sum(residuals(m_pois, "pearson")^2) / df.residual(m_pois)   # eq. (13.9)
cat(sprintf("  Pearson dispersion phi_hat = %.3f  (1.0 = no overdispersion)\n", phi_hat))
cat(sprintf("  -> Poisson SEs are too small by a factor sqrt(phi) = %.2f\n",
            sqrt(phi_hat)))
m_qp <- glm(y ~ g, data = do_, family = quasipoisson())
m_nb <- glm.nb(y ~ g, data = do_)
cat(sprintf("\n  %-22s%10s%10s%11s  variance model\n", "model", "beta_g", "SE", "p"))
## Loop variable deliberately NOT called `z`: the data vector `z` from
## section 3 is still needed in section 6. R has no block scope, so a loop
## variable silently overwrites anything of the same name.
for (row_ in list(list("Poisson", m_pois, "Var = mu"),
                  list("quasi-Poisson", m_qp, "Var = phi*mu   (linear)"),
                  list("negative binomial", m_nb, "Var = mu+phi*mu^2 (quadratic)"))) {
  cs <- coef(summary(row_[[2]]))
  cat(sprintf("  %-22s%10.4f%10.4f%11.4f  %s\n", row_[[1]], cs["g", 1], cs["g", 2],
              cs["g", 4], row_[[3]]))
}
cat("\n  All three give the same POINT estimate; they differ entirely in the SE.\n")
cat("  Calibration check under a TRUE null:\n")
set.seed(1305)
hits <- c(Poisson = 0, `quasi-Poisson` = 0, NB = 0)
N_SIM <- 500
for (i in seq_len(N_SIM)) {
  ## Distinct names: `yy` and `a`/`z` from section 3 are still needed below.
  xx <- rbinom(n, 1, 0.5)
  y_sim <- rnbinom(n, mu = exp(2.5), size = 1/PHI)
  d2 <- data.frame(y = y_sim, g = xx)
  hits["Poisson"] <- hits["Poisson"] +
    (coef(summary(glm(y ~ g, d2, family = poisson())))["g", 4] < 0.05)
  hits["quasi-Poisson"] <- hits["quasi-Poisson"] +
    (coef(summary(glm(y ~ g, d2, family = quasipoisson())))["g", 4] < 0.05)
  nbf <- try(glm.nb(y ~ g, data = d2), silent = TRUE)
  if (!inherits(nbf, "try-error"))
    hits["NB"] <- hits["NB"] + (coef(summary(nbf))["g", 4] < 0.05)
}
for (k in names(hits))
  cat(sprintf("    %-16s false-positive rate = %.1f%%\n", k, 100*hits[k]/N_SIM))
cat("  Poisson on overdispersed counts is badly anticonservative.\n")

#' ## 6. Wald vs LRT, eq. (13.11)

#+ wald-lrt
header("6. Three tests, one hypothesis (13.11)")
## m_adj / m_crude were fitted in section 3 on the 30,000-row randomised data.
wald <- (coef(summary(m_adj))["a", 1] / coef(summary(m_adj))["a", 2])^2
lr <- anova(glm(yy ~ z, family = binomial()), m_adj, test = "LRT")
cat(sprintf("  Wald chi2 = %.4f, p = %.3e\n", wald, pchisq(wald, 1, lower.tail = FALSE)))
cat(sprintf("  LRT  chi2 = %.4f, p = %.3e\n", lr[["Deviance"]][2], lr[["Pr(>Chi)"]][2]))
cat("  They agree asymptotically. Now near separation, where they do not:\n")
xs2 <- c(-3,-2,-1.5,-1,-0.2,1,1.5,2,3,0.1); ys2 <- c(0,0,0,0,0,1,1,1,1,1)
f_full <- suppressWarnings(glm(ys2 ~ xs2, family = binomial()))
f_red <- glm(ys2 ~ 1, family = binomial())
w2 <- (coef(summary(f_full))[2, 1] / coef(summary(f_full))[2, 2])^2
l2 <- deviance(f_red) - deviance(f_full)
cat(sprintf("    Wald chi2 = %.4f, p = %.4f   <- no power\n", w2,
            pchisq(w2, 1, lower.tail = FALSE)))
cat(sprintf("    LRT  chi2 = %.4f, p = %.4f   <- correct\n", l2,
            pchisq(l2, 1, lower.tail = FALSE)))
cat("  The Wald statistic COLLAPSES as |beta| grows because its SE grows\n")
cat("  faster (the Hauck-Donner effect). PREFER THE LRT.\n")

#' ## 7. Deviance residuals, eq. (13.12)

#+ deviance
header("7. Use deviance residuals, not raw ones (13.12)")
m <- glm.nb(y ~ g, data = do_)
cat(sprintf("  deviance = %.4f, df = %d, deviance/df = %.3f\n",
            deviance(m), df.residual(m), deviance(m)/df.residual(m)))
## A binary predictor gives only TWO distinct fitted values, so splitting on
## `< median(fitted)` can select an empty set. Split on the predictor instead.
lowm <- do_$g == 0
raw <- do_$y - fitted(m); dev <- residuals(m, "deviance")
cat(sprintf("  raw residual SD:      low-mean half %.3f, high-mean half %.3f\n",
            sd(raw[lowm]), sd(raw[!lowm])))
cat(sprintf("  deviance residual SD: low-mean half %.3f, high-mean half %.3f\n",
            sd(dev[lowm]), sd(dev[!lowm])))
cat("  Deviance residuals are homoscedastic by construction.\n")

#' ## 8. Figure

#+ figure
png(file.path(OUT, "glm.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
xg_ <- seq(-3.5, 3.5, length.out = 200)
plot(xg_, plogis(-0.5 + 1.1*xg_), type = "l", lwd = 2, col = "steelblue",
     xlab = "x1", ylab = "P(y=1)", main = "Eq. (13.6): logistic mean function")
## x1/yb come from section 2 (n = 400); both still have their original length.
points(x1, jitter(yb, 0.1), pch = ".", col = rgb(0, 0, 0, 0.3))
plot(dd$lib, dd$y, col = c("steelblue", "firebrick")[dd$g + 1], pch = 16, cex = 0.6,
     xlab = "library size (exposure)", ylab = "count",
     main = "Eq. (13.8): counts scale with exposure")
plot(fitted(m), residuals(m, "deviance"), pch = 16, cex = 0.6, col = "steelblue",
     xlab = "fitted mean", ylab = "residual", main = "Eq. (13.12): deviance residuals")
points(fitted(m), raw / sd(raw), pch = 16, cex = 0.6, col = rgb(0.8, 0.4, 0, 0.5))
abline(h = 0)
legend("topright", c("deviance", "raw (scaled)"), col = c("steelblue", "darkorange"),
       pch = 16, bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "glm.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 13)
#'
#' 1. Choose the family from the outcome's support and mean-variance
#'    relationship; the link from the estimand you want to report.
#' 2. Counts with varying exposure require an offset (13.8), always.
#' 3. Check the dispersion (13.9).
#' 4. Report both scales: the odds/rate ratio AND a predicted probability.
#'
#' **Next:** `14_mixed_models.R`
