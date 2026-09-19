#' ---
#' title: "Module 23 - Bayesian reasoning and hierarchical shrinkage"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 33, equations (33.1)-(33.7)
#'
#' With 3 replicates and 20,000 genes, each gene alone is uninformative -- but
#' the ENSEMBLE is not.
#'
#' ## What you will learn
#'
#' 1. Conjugate updating (33.2)-(33.3); the posterior mean is a
#'    precision-weighted average.
#' 2. **The James-Stein theorem (33.5)**: shrinkage is a FREQUENTIST result.
#' 3. Empirical Bayes: a from-scratch **moderated t-test** (limma, eq. 20.6).
#' 4. Posterior predictive checks (33.6).
#' 5. A Metropolis sampler with convergence diagnostics.
#' 6. Why Bayes factors (33.7) are prior-sensitive.

#+ setup, message = FALSE
MODULE_NAME <- "23_bayesian_and_shrinkage"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Conjugate updating, eq. (33.2)-(33.3)

#+ conjugate
header("1. Beta-binomial and normal-normal conjugacy (33.2)-(33.3)")
a0 <- 2; b0 <- 8; y_obs <- 14; n_obs <- 40
a1 <- a0 + y_obs; b1 <- b0 + n_obs - y_obs                       # eq. (33.2)
cat(sprintf("  prior Beta(%g, %g): mean %.3f, worth %g pseudo-observations\n",
            a0, b0, a0/(a0+b0), a0+b0))
cat(sprintf("  data: %d/%d = %.3f\n", y_obs, n_obs, y_obs/n_obs))
cat(sprintf("  posterior Beta(%g, %g): mean %.3f\n", a1, b1, a1/(a1+b1)))
cat(sprintf("  95%% CREDIBLE interval [%.3f, %.3f]\n",
            qbeta(0.025, a1, b1), qbeta(0.975, a1, b1)))
cat(sprintf("  P(theta > 0.30 | data) = %.4f\n", pbeta(0.30, a1, b1, lower.tail = FALSE)))
cat("  A credible interval IS a probability statement about theta given the\n")
cat("  data - which is what people wrongly believe a confidence interval to be.\n")
cat("\n  Normal-normal (33.3): posterior mean = lambda*y + (1-lambda)*mu0\n")
cat(sprintf("  %22s%10s%17s\n", "sigma^2 (data noise)", "lambda", "weight on data"))
for (s2 in c(0.1, 0.5, 1, 4, 20)) {
  lam <- 1/(1 + s2)
  cat(sprintf("  %22.1f%10.3f%16.1f%%\n", s2, lam, 100*lam))
}
cat("  Noisier data -> more shrinkage toward the prior. Compare with the\n")
cat("  mixed-model BLUP, eq. (14.6): it is the SAME formula.\n")

#' ## 2. The James-Stein theorem, eq. (33.5)

#+ james-stein
header("2. James-Stein: shrinkage wins, provably (33.5)")
james_stein <- function(y, sigma2 = 1) {                         # eq. (33.5)
  max(1 - (length(y) - 2)*sigma2/sum(y^2), 0) * y                # positive part
}
set.seed(2301)
cat(sprintf("  %-34s%11s%10s%12s\n", "scenario", "MLE risk", "JS risk", "JS better?"))
scenarios <- list("all means zero" = rep(0, 20),
                  "means near zero (typical omics)" = rnorm(20, 0, 0.4),
                  "means well spread" = rnorm(20, 0, 2),
                  "one huge outlier" = c(rep(0, 19), 10))
for (nm in names(scenarios)) {
  theta <- scenarios[[nm]]
  r <- replicate(3000, {
    y <- theta + rnorm(length(theta))
    c(sum((y - theta)^2), sum((james_stein(y) - theta)^2))
  })
  cat(sprintf("  %-34s%11.2f%10.2f%12s\n", nm, mean(r[1, ]), mean(r[2, ]),
              mean(r[2, ]) < mean(r[1, ])))
}
cat("\n  JS wins in EVERY case (that is the theorem), and wins MOST when the\n")
cat("  true effects are clustered near zero - exactly the omics situation.\n")
cat("  This is NOT a Bayesian assumption; it is a frequentist theorem, and it\n")
cat("  is the licence for every shrinkage method in this course.\n")

#' ## 3. Empirical Bayes: a moderated t-test, eq. (20.6)
#'
#' $$\tilde s_g^2=\frac{d_0s_0^2+d_gs_g^2}{d_0+d_g},\qquad
#'   \tilde t_g \sim t_{d_g+d_0}$$

#+ moderated-t
header("3. From-scratch moderated t-test (20.6)")
fit_prior_variance <- function(s2, df) {
  ## Match moments of log(s2), as limma does. Var(log s2) exceeds
  ## trigamma(df/2) by exactly trigamma(d0/2); solve for d0. A ROBUST (MAD)
  ## estimate of the spread is essential - a few near-zero variances would
  ## otherwise dominate and collapse the shrinkage.
  e <- log(s2) - digamma(df/2) + log(df/2)
  target <- (mad(e))^2 - trigamma(df/2)
  if (target <= 0) return(list(d0 = Inf, s0_2 = exp(mean(e))))
  f <- function(d0) trigamma(d0/2) - target
  if (f(1e-4) * f(1e4) > 0) return(list(d0 = Inf, s0_2 = exp(mean(e))))
  d0 <- uniroot(f, c(1e-4, 1e4))$root
  list(d0 = d0, s0_2 = exp(mean(e) + digamma(d0/2) - log(d0/2)))
}
moderated_t <- function(A, B) {
  n1 <- ncol(A); n2 <- ncol(B); df <- n1 + n2 - 2
  diff <- rowMeans(A) - rowMeans(B)
  s2 <- ((n1-1)*apply(A, 1, var) + (n2-1)*apply(B, 1, var)) / df
  s2 <- pmax(s2, 1e-12)
  pr <- fit_prior_variance(s2, df)
  if (is.infinite(pr$d0)) { s2t <- rep(pr$s0_2, length(s2)); total_df <- Inf }
  else { s2t <- (pr$d0*pr$s0_2 + df*s2)/(pr$d0 + df); total_df <- df + pr$d0 }
  se <- sqrt(s2t * (1/n1 + 1/n2))
  t <- diff/se
  list(t = t, p = 2*pt(abs(t), total_df, lower.tail = FALSE), s2 = s2, s2t = s2t,
       d0 = pr$d0, s0_2 = pr$s0_2, df_total = total_df, diff = diff)
}
set.seed(2302)
G <- 6000; n_rep <- 3; n_de <- 400
true_sd <- sqrt(1/rgamma(G, 5, 5))
is_de <- c(rep(TRUE, n_de), rep(FALSE, G - n_de))
delta <- ifelse(is_de, 2.6 * sample(c(-1, 1), G, TRUE), 0)
A <- matrix(rnorm(G*n_rep, delta, true_sd), nrow = G)
B <- matrix(rnorm(G*n_rep, 0, true_sd), nrow = G)
mod <- moderated_t(A, B)
ord_p <- sapply(1:G, function(i) t.test(A[i, ], B[i, ], var.equal = TRUE)$p.value)
cat(sprintf("  G = %d, n = %d per group, %d truly differential\n", G, n_rep, n_de))
cat(sprintf("  estimated prior df d0 = %.2f   (the ordinary test has only %d df)\n",
            mod$d0, 2*n_rep - 2))
cat(sprintf("  moderated tests use %.2f df - the equivalent of several extra\n",
            mod$df_total))
cat("  samples, for free.\n")
cat(sprintf("\n  %-22s%10s%10s%11s%8s%8s\n", "method", "rejected", "true pos",
            "false pos", "sens", "FDP"))
for (row_ in list(list("ordinary t", ord_p), list("moderated t (20.6)", mod$p))) {
  q <- p.adjust(row_[[2]], "BH"); rej <- q < 0.05
  cat(sprintf("  %-22s%10d%10d%11d%8.2f%8.3f\n", row_[[1]], sum(rej),
              sum(rej & is_de), sum(rej & !is_de), sum(rej & is_de)/n_de,
              sum(rej & !is_de)/max(sum(rej), 1)))
}
worst <- order(mod$s2)[1:5]
cat("\n  The pathology this removes - genes with an accidentally TINY s_g:\n")
cat(sprintf("  %8s%10s%14s%15s%11s\n", "gene", "s_g^2", "ordinary |t|",
            "moderated |t|", "truly DE?"))
for (g in worst)
  cat(sprintf("  %8d%10.5f%14.2f%15.2f%11s\n", g, mod$s2[g],
              abs(t.test(A[g, ], B[g, ], var.equal = TRUE)$statistic),
              abs(mod$t[g]), is_de[g]))
cat("  Without moderation these dominate the top of the gene list purely\n")
cat("  because their variance estimate was unluckily small.\n")

#' ## 4. Posterior predictive checks, eq. (33.6)

#+ ppc
header("4. Posterior predictive check: can the model reproduce the data? (33.6)")
set.seed(2303)
y_real <- rnbinom(300, mu = 15, size = 2)          # overdispersed TRUTH
lam_hat <- mean(y_real)
ppc <- function(y, simulate, stat, n_sim = 4000) {
  obs <- stat(y)
  reps <- replicate(n_sim, stat(simulate(length(y))))
  list(obs = obs, reps = reps, p = mean(reps >= obs))
}
stats_to_check <- list(variance = var,
                       `proportion of zeros` = function(v) mean(v == 0),
                       maximum = max)
cat(sprintf("  Fitting a POISSON model to negative-binomial data (lambda_hat = %.2f)\n",
            lam_hat))
cat(sprintf("  %-24s%11s%17s%13s\n", "statistic", "observed", "mean replicate",
            "ppc p-value"))
for (nm in names(stats_to_check)) {
  r <- ppc(y_real, function(k) rpois(k, lam_hat), stats_to_check[[nm]])
  cat(sprintf("  %-24s%11.3f%17.3f%13.4f\n", nm, r$obs, mean(r$reps), r$p))
}
phi_hat <- max((var(y_real) - lam_hat)/lam_hat^2, 1e-6)
cat("\n  Now the CORRECT negative-binomial model:\n")
for (nm in names(stats_to_check)) {
  r <- ppc(y_real, function(k) rnbinom(k, mu = lam_hat, size = 1/phi_hat),
           stats_to_check[[nm]])
  cat(sprintf("  %-24s%11.3f%17.3f%13.4f\n", nm, r$obs, mean(r$reps), r$p))
}
cat("\n  ppc p-values near 0 or 1 mean the model CANNOT reproduce that feature\n")
cat("  of the data. This is the Bayesian analogue of residual diagnostics.\n")

#' ## 5. A Metropolis sampler with convergence diagnostics

#+ mcmc
header("5. MCMC and the diagnostics you must report")
metropolis <- function(log_post, init, n_iter = 8000, step = 0.4, seed = 1) {
  set.seed(seed)
  x <- init; chain <- matrix(NA, n_iter, length(x)); lp <- log_post(x); acc <- 0
  for (i in seq_len(n_iter)) {
    prop <- x + rnorm(length(x), 0, step)
    lp_prop <- log_post(prop)
    if (log(runif(1)) < lp_prop - lp) { x <- prop; lp <- lp_prop; acc <- acc + 1 }
    chain[i, ] <- x
  }
  list(chain = chain, acc = acc/n_iter)
}
set.seed(2304)
n <- 200; xv <- rnorm(n)
yv <- rbinom(n, 1, plogis(-0.4 + 1.2*xv))
log_posterior <- function(beta) {
  eta <- beta[1] + beta[2]*xv
  sum(yv*eta - log1p(exp(eta))) + sum(dnorm(beta, 0, 2.5, log = TRUE))
}
chains <- lapply(1:4, function(s) metropolis(log_posterior, c(0, 0), seed = s)$chain)
acc <- metropolis(log_posterior, c(0, 0), seed = 1)$acc
burn <- 2000
post <- do.call(rbind, lapply(chains, function(ch) ch[-(1:burn), ]))
r_hat <- function(chains, burn) {
  x <- lapply(chains, function(ch) ch[-(1:burn),, drop = FALSE])
  m <- length(x); nn <- nrow(x[[1]])
  W <- rowMeans(sapply(x, function(ch) apply(ch, 2, var)))
  B <- nn * apply(sapply(x, colMeans), 1, var)
  sqrt(((nn-1)/nn * W + B/nn) / W)
}
ess <- function(v) {
  v <- v - mean(v); nn <- length(v)
  ac <- acf(v, lag.max = min(nn - 1, 500), plot = FALSE)$acf[-1]
  k <- which(ac < 0.05)[1]; if (is.na(k)) k <- length(ac)
  nn / (1 + 2*sum(ac[seq_len(max(k - 1, 1))]))
}
rh <- r_hat(chains, burn)
cat(sprintf("  acceptance rate = %.3f   (0.2-0.4 is healthy for random walk)\n", acc))
cat(sprintf("  %-14s%12s%24s%8s%9s\n", "parameter", "post. mean", "95% credible",
            "R-hat", "ESS"))
for (j in 1:2) {
  ci <- quantile(post[, j], c(0.025, 0.975))
  cat(sprintf("  %-14s%12.4f   [%7.4f, %7.4f]%8.4f%9.0f\n",
              c("intercept", "slope")[j], mean(post[, j]), ci[1], ci[2], rh[j],
              ess(chains[[1]][-(1:burn), j])))
}
cat("  truth: intercept -0.400, slope 1.200\n")
cat("\n  NEVER report a posterior without R-hat < 1.01 and ESS > 400.\n")
mle <- glm(yv ~ xv, family = binomial())
cat(sprintf("\n  frequentist MLE: intercept %+.4f, slope %+.4f\n",
            coef(mle)[1], coef(mle)[2]))
cat("  With a weakly informative prior and n=200 the two coincide. The prior\n")
cat("  matters at SMALL n - precisely when regularisation is most valuable.\n")

#' ## 6. Bayes factors are prior-sensitive, eq. (33.7)

#+ bayes-factor
header("6. Lindley's paradox (33.7)")
z_obs <- 2.5; n_bf <- 100; sigma <- 1
se <- sigma/sqrt(n_bf); y_bar <- z_obs*se
cat(sprintf("  data: y_bar = %.4f, SE = %.4f, z = %.1f, p = %.4f\n",
            y_bar, se, z_obs, 2*pnorm(z_obs, lower.tail = FALSE)))
cat(sprintf("\n  %24s%10s  interpretation\n", "prior SD tau under H1", "BF_10"))
for (tau in c(0.05, 0.1, 0.5, 1, 5, 50)) {
  bf <- dnorm(y_bar, 0, sqrt(tau^2 + se^2)) / dnorm(y_bar, 0, se)
  note <- if (bf > 3) "favours H1" else if (bf < 1/3) "favours H0" else "ambiguous"
  cat(sprintf("  %24.2f%10.3f  %s\n", tau, bf, note))
}
cat("\n  The SAME data give BF from strongly favouring H1 to favouring H0,\n")
cat("  purely by changing the prior width. The p-value did not move. If you\n")
cat("  report a Bayes factor, report its sensitivity across prior scales.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "bayes.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
gx <- seq(0, 1, length.out = 400)
plot(gx, dbeta(gx, a0, b0), type = "l", lwd = 2, col = "steelblue", ylim = c(0, 6),
     xlab = "theta", ylab = "density", main = "Eq. (33.2): conjugate updating")
lines(gx, dbeta(gx, a1, b1), lwd = 2, col = "firebrick")
abline(v = y_obs/n_obs, lty = 2)
legend("topright", c("prior", "posterior", "data"), col = c("steelblue","firebrick","black"),
       lty = c(1,1,2), lwd = 2, bty = "n", cex = 0.7)
lim <- c(0, quantile(sqrt(mod$s2), 0.995))
plot(sqrt(mod$s2), sqrt(mod$s2t), pch = ".", col = adjustcolor("steelblue", 0.3),
     xlim = lim, ylim = lim, xlab = "ordinary s_g", ylab = "moderated s_g",
     main = "Eq. (20.6): variance shrinkage")
abline(0, 1, lty = 2); abline(h = sqrt(mod$s0_2), col = "red", lty = 3)
matplot(sapply(chains, function(ch) ch[, 2]), type = "l", lty = 1, lwd = 0.5,
        col = c("steelblue","darkorange","forestgreen","firebrick"),
        xlab = "iteration", ylab = "slope",
        main = sprintf("MCMC trace (R-hat = %.4f)", rh[2]))
abline(v = burn, col = "red", lty = 2)
r <- ppc(y_real, function(k) rpois(k, lam_hat), var)
hist(r$reps, breaks = 50, col = "lightsteelblue", border = "white",
     xlim = range(c(r$reps, r$obs)), main = "Eq. (33.6): posterior predictive check",
     xlab = "replicated variance")
abline(v = r$obs, col = "red", lwd = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "bayes.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 33)
#'
#' 1. Use empirical Bayes shrinkage whenever you have many parallel features
#'    and few replicates.
#' 2. Report credible intervals AS probability statements, and say so.
#' 3. Make priors explicit, weakly informative, and checked.
#' 4. Check convergence (R-hat, ESS) before interpreting anything.
#'
#' **Next:** `24_diagnostics_and_reproducibility.R`
