#' ---
#' title: "Applied 33 - DNA methylation and epigenomics"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 23, equations (23.1)-(23.3)
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | Illumina EPIC whole-blood case/control cohort |
#' | **Key confounder** | **Blood cell composition** - usually the dominant signal |
#'
#' ## The three decisions that dominate a methylation analysis
#'
#' 1. **Scale**: model on M, report on beta (23.1).
#' 2. **Cell composition**: confounder, mediator, or outcome? (23.3)
#' 3. **Region vs position**: neighbouring CpGs are correlated.

#+ setup, message = FALSE
suppressPackageStartupMessages(library(MASS))
MODULE_NAME <- "33_methylation_epigenomics"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
beta_to_m <- function(b, eps = 1e-6) { b <- pmin(pmax(b, eps), 1-eps); log2(b/(1-b)) }
m_to_beta <- function(m) 2^m/(2^m + 1)
CELL_TYPES <- c("CD4T","CD8T","NK","Bcell","Mono","Gran")

#' ## 1. Beta and M-values, eq. (23.1)

#+ beta-m
header("1. Beta values are heteroscedastic; M-values are not (23.1)")
set.seed(3301)
cat(sprintf("  %11s%11s%9s\n", "mean beta", "SD(beta)", "SD(M)"))
for (target in c(0.02, 0.10, 0.30, 0.50, 0.75, 0.95)) {
  m_draws <- rnorm(40000, log2(target/(1-target)), 0.5)
  cat(sprintf("  %11.2f%11.4f%9.4f\n", target, sd(m_to_beta(m_draws)), sd(m_draws)))
}
cat("\n  Probes near 0 or 1 are STRUCTURALLY less variable. A linear model on\n")
cat("  beta gives them standard errors that are too small, so they dominate a\n")
cat("  hit list for a purely mathematical reason. MODEL ON M, REPORT BETA.\n")

#' ## 2. Simulate an array cohort with cell-composition confounding

#+ simulate
header("2. A whole-blood EPIC-like cohort")
simulate_methylation <- function(n = 120, n_probes = 3000, n_dmp = 150) {
  ref_M <- matrix(rnorm(n_probes*length(CELL_TYPES), 0, 2), n_probes) +
    rnorm(n_probes, 0, 1.5)
  case <- rbinom(n, 1, 0.5); age <- rnorm(n, 55, 12); sex <- rbinom(n, 1, 0.5)
  ## Cases have MORE granulocytes and fewer CD4T -> composition is a CONFOUNDER.
  base_logit <- c(1.1, 0.5, 0.2, 0.3, 0.8, 1.9)
  shift <- c(-0.35, 0, 0, 0, 0.10, 0.30)
  W <- t(sapply(seq_len(n), function(i) {
    lg <- base_logit + case[i]*shift + rnorm(length(CELL_TYPES), 0, 0.22)
    exp(lg)/sum(exp(lg))
  }))
  colnames(W) <- CELL_TYPES
  true_eff <- numeric(n_probes)
  dmp <- sample(n_probes, n_dmp); true_eff[dmp] <- rnorm(n_dmp, 0, 0.85)
  M <- ref_M %*% t(W) + outer(true_eff, case) +
    outer(rnorm(n_probes, 0, 0.012), age - mean(age)) +
    matrix(rnorm(n_probes*n, 0, 0.45), n_probes)
  probes <- sprintf("cg%07d", seq_len(n_probes))
  samples <- sprintf("S%03d", seq_len(n))
  dimnames(M) <- list(probes, samples)
  meta <- data.frame(case = case, age = age, sex = sex, W, row.names = samples)
  list(beta = m_to_beta(M), M = M, meta = meta,
       truth = data.frame(true_eff = true_eff, is_dmp = true_eff != 0, row.names = probes),
       ref_M = `dimnames<-`(ref_M, list(probes, CELL_TYPES)))
}
set.seed(3302)
sim <- simulate_methylation()
beta_df <- sim$beta; M_df <- sim$M; meta <- sim$meta; truth <- sim$truth
stopifnot(identical(colnames(beta_df), rownames(meta)))
cat(sprintf("  %d probes x %d samples\n", nrow(beta_df), ncol(beta_df)))
cat(sprintf("  %d probes have a TRUE direct effect of case status\n", sum(truth$is_dmp)))
cat("\n  mean cell proportions by group:\n")
print(round(aggregate(meta[, CELL_TYPES], by = list(case = meta$case), mean), 3))
cat(sprintf("\n  Granulocytes differ by %+.3f between groups -> cell composition is\n",
            diff(tapply(meta$Gran, meta$case, mean))))
cat("  associated with the outcome, i.e. a CONFOUNDER for any probe whose\n")
cat("  methylation differs between cell types.\n")

#' ## 3. Probe-wise testing: what the M scale actually buys you

#+ probewise
header("3. Probe-wise linear models on M vs beta")
probewise <- function(Y, meta, rhs_cols, coef_index = 2) {
  X <- cbind(1, as.matrix(sapply(rhs_cols, function(c_) as.numeric(meta[[c_]]))))
  beta_hat <- qr.coef(qr(X), t(Y))
  resid <- t(Y) - X %*% beta_hat
  dof <- nrow(X) - ncol(X)
  s2 <- pmax(colSums(resid^2)/dof, 1e-14)
  se <- sqrt(s2 * chol2inv(qr.R(qr(X)))[coef_index, coef_index])
  p <- 2*pt(abs(beta_hat[coef_index, ]/se), dof, lower.tail = FALSE)
  data.frame(effect = beta_hat[coef_index, ], s2 = s2, pvalue = p,
             padj = p.adjust(p, "BH"), row.names = rownames(Y))
}
moderated_probewise <- function(Y, meta, rhs_cols, coef_index = 2) {
  base <- probewise(Y, meta, rhs_cols, coef_index)
  X <- cbind(1, as.matrix(sapply(rhs_cols, function(c_) as.numeric(meta[[c_]]))))
  dof <- nrow(X) - ncol(X)
  e <- log(base$s2) - digamma(dof/2) + log(dof/2)
  target <- mad(e)^2 - trigamma(dof/2)
  d0 <- if (target > 0) tryCatch(uniroot(function(d) trigamma(d/2) - target,
                                         c(1e-4, 1e4))$root, error = function(e) 50)
        else 50
  s0_2 <- exp(mean(e) + digamma(d0/2) - log(d0/2))
  s2t <- (d0*s0_2 + dof*base$s2)/(d0 + dof)
  se <- sqrt(s2t * chol2inv(qr.R(qr(X)))[coef_index, coef_index])
  base$pvalue <- 2*pt(abs(base$effect/se), dof + d0, lower.tail = FALSE)
  base$padj <- p.adjust(base$pvalue, "BH")
  list(res = base, d0 = d0)
}
mean_beta <- rowMeans(beta_df)
dist_edge <- pmin(mean_beta, 1 - mean_beta)
res_beta <- probewise(beta_df, meta, c("case","age","sex"))
res_M <- probewise(M_df, meta, c("case","age","sex"))
cat(sprintf("  Spearman(distance from the 0/1 boundary, residual variance):\n"))
cat(sprintf("    beta scale : %+.3f   <- strong dependence\n",
            cor(dist_edge, res_beta$s2, method = "spearman")))
cat(sprintf("    M scale    : %+.3f   <- essentially none\n",
            cor(dist_edge, res_M$s2, method = "spearman")))
cat("\n  That is the heteroscedasticity of eq. (23.1), measured directly.\n")
mb <- moderated_probewise(beta_df, meta, c("case","age","sex"))
mM <- moderated_probewise(M_df, meta, c("case","age","sex"))
cat("\n  WHY IT MATTERS - and it is NOT what most people assume. An ordinary\n")
cat("  probe-wise t-test on beta is largely self-correcting: an extreme probe\n")
cat("  has both a smaller effect AND a smaller SE, and the two mostly cancel.\n")
cat("  The damage appears once you MODERATE the variances, which every standard\n")
cat("  pipeline does (limma, eq. 20.6). Empirical Bayes assumes the residual\n")
cat("  variances are EXCHANGEABLE across probes. On the beta scale they are\n")
cat("  not, so the prior degrees of freedom collapse:\n")
cat(sprintf("\n    prior df d0, beta scale : %8.1f\n", mb$d0))
cat(sprintf("    prior df d0, M scale    : %8.1f\n", mM$d0))
cat("    With n = 4 or 6 per group - the usual EWAS pilot - that is decisive.\n")
score <- function(res, label) {
  rej <- res$padj < 0.05
  tp <- sum(rej & truth$is_dmp); fp <- sum(rej & !truth$is_dmp)
  cat(sprintf("  %-44s%7d%7d%7d%9.2f%8.3f\n", label, sum(rej), tp, fp,
              tp/sum(truth$is_dmp), fp/max(sum(rej), 1)))
}
cat(sprintf("\n  %-44s%7s%7s%7s%9s%8s\n", "analysis", "rej", "TP", "FP", "sens", "FDP"))
score(res_beta, "beta, ordinary t, adj. age+sex")
score(res_M, "M, ordinary t, adj. age+sex")
score(mM$res, "M, MODERATED t, adj. age+sex")
cat("\n  Note the FDP in all rows is far above 0.05. None of these adjusts for\n")
cat("  CELL COMPOSITION, which is the real driver - that is section 4, and it\n")
cat("  matters far more than the scale choice.\n")

#' ## 4. Cell composition: confounder, mediator, or outcome? eq. (23.3)

#+ deconvolution
header("4. Reference-based deconvolution and the adjustment decision (23.3)")
houseman <- function(Y, reference) {
  ## eq. (23.3): non-negative, sum-to-one constrained regression. Implemented
  ## as NNLS via a simple projected-gradient loop, then renormalised.
  R <- as.matrix(reference)
  t(apply(Y, 2, function(y) {
    w <- rep(1/ncol(R), ncol(R))
    for (it in 1:500) {
      g <- crossprod(R, R %*% w - y)
      w <- pmax(w - 0.5*g/max(sum(R^2), 1e-9), 0)
      if (sum(w) > 0) w <- w/sum(w)
    }
    w
  }))
}
disc <- names(sort(apply(sim$ref_M, 1, sd), decreasing = TRUE))[1:400]
W_hat <- houseman(M_df[disc, ], sim$ref_M[disc, ])
colnames(W_hat) <- CELL_TYPES
cat("  Estimated vs true cell proportions (correlation across samples):\n")
for (ct in CELL_TYPES)
  cat(sprintf("    %-8s r = %+.3f   (true mean %.3f, est mean %.3f)\n", ct,
              cor(W_hat[, ct], meta[[ct]]), mean(meta[[ct]]), mean(W_hat[, ct])))
meta_w <- meta
for (ct in CELL_TYPES[-length(CELL_TYPES)]) meta_w[[paste0("est_", ct)]] <- W_hat[, ct]
res_adj <- probewise(M_df, meta_w,
                     c("case","age","sex", paste0("est_", CELL_TYPES[-length(CELL_TYPES)])))
cat(sprintf("\n  %-44s%7s%7s%7s%9s%8s\n", "analysis", "rej", "TP", "FP", "sens", "FDP"))
score(res_M, "M scale, NOT adjusted for composition")
score(res_adj, "M scale, adjusted for estimated composition")
cat("\n  Adjusting removes composition-driven false positives. But BEFORE\n")
cat("  adjusting, decide the CAUSAL ROLE (Module 22):\n")
cat("    CONFOUNDER - composition differs for reasons unrelated to disease\n")
cat("                 (age, smoking, sampling) -> ADJUST.\n")
cat("    MEDIATOR   - the disease CAUSES the composition change, which then\n")
cat("                 changes methylation -> adjusting removes the effect you\n")
cat("                 wanted. Do a mediation analysis instead (eq. 32.6).\n")
cat("    OUTCOME    - the composition change IS the finding -> model it\n")
cat("                 directly and do not adjust methylation for it.\n")
med_frac <- sapply(rownames(truth)[truth$is_dmp][1:60], function(p_) {
  d <- meta; d$y <- M_df[p_, ]
  tot <- coef(lm(y ~ case + age + sex, data = d))["case"]
  dir_ <- coef(lm(as.formula(paste("y ~ case + age + sex +",
                                   paste(CELL_TYPES[-length(CELL_TYPES)], collapse = "+"))),
                  data = d))["case"]
  if (abs(tot) > 1e-6) 1 - dir_/tot else NA
})
cat(sprintf("\n  Median proportion of the case effect 'explained' by composition\n"))
cat(sprintf("  across 60 true DMPs: %.1f%%\n", 100*median(med_frac, na.rm = TRUE)))
cat("  If composition is a MEDIATOR, that fraction is a real biological\n")
cat("  pathway and adjusting it away is a mistake.\n")

#' ## 5. Bisulfite sequencing: beta-binomial counts, eq. (23.2)

#+ bisulfite
header("5. Bisulfite counts need a beta-binomial (23.2)")
set.seed(3303)
n_s <- 24; n_sites <- 400
group <- rbinom(n_s, 1, 0.5)
true_pi <- rbeta(n_sites, 2, 2)
delta <- c(rep(0.12, 40), rep(0, n_sites - 40))
RHO <- 0.03
coverage <- matrix(rnbinom(n_sites*n_s, mu = 30, size = 2), n_sites) + 1
y <- matrix(0L, n_sites, n_s)
for (i in seq_len(n_sites)) for (j in seq_len(n_s)) {
  pij <- min(max(true_pi[i] + delta[i]*group[j], 0.01), 0.99)
  a <- pij*(1/RHO - 1); b <- (1 - pij)*(1/RHO - 1)
  y[i, j] <- rbinom(1, coverage[i, j], rbeta(1, a, b))
}
is_de <- delta != 0
test_sites <- function(quasi) {
  vapply(seq_len(n_sites), function(i) {
    d <- data.frame(m = y[i, ], u = coverage[i, ] - y[i, ], g = group)
    f <- try(glm(cbind(m, u) ~ g, data = d, family = binomial()), silent = TRUE)
    if (inherits(f, "try-error")) return(1)
    if (!quasi) return(coef(summary(f))["g", 4])
    phi <- max(sum(residuals(f, "pearson")^2)/df.residual(f), 1)
    2*pt(abs(coef(f)["g"]/(coef(summary(f))["g",2]*sqrt(phi))),
         df.residual(f), lower.tail = FALSE)
  }, numeric(1))
}
cat(sprintf("  %-34s%7s%7s%7s%8s\n", "model", "rej", "TP", "FP", "FDP"))
for (row_ in list(list("binomial (ignores overdispersion)", FALSE),
                  list("quasi-binomial (eq. 23.2)", TRUE))) {
  p <- test_sites(row_[[2]]); rej <- p.adjust(p, "BH") < 0.05
  cat(sprintf("  %-34s%7d%7d%7d%8.3f\n", row_[[1]], sum(rej), sum(rej & is_de),
              sum(rej & !is_de), sum(rej & !is_de)/max(sum(rej), 1)))
}
cat(sprintf("\n  Coverage ranges %d-%dx. A plain binomial treats a high-coverage\n",
            min(coverage), max(coverage)))
cat("  site as far more certain than a low-coverage one, ignoring the\n")
cat("  BIOLOGICAL variability between samples entirely (eq. 23.2).\n")

#' ## 6. Regions beat positions

#+ regions
header("6. Region-level inference")
set.seed(3304)
probe_pos <- sort(sample(1e7, nrow(truth)))
names(probe_pos) <- rownames(truth)
z_scores <- qnorm(res_adj$pvalue/2, lower.tail = FALSE) * sign(res_adj$effect)
region_scan <- function(z, positions, window = 2000, min_probes = 3) {
  o <- order(positions); zs <- z[o]; ps <- positions[o]
  out <- list(); i <- 1
  while (i <= length(ps)) {
    j <- i
    while (j + 1 <= length(ps) && ps[j+1] - ps[i] <= window) j <- j + 1
    k <- j - i + 1
    ## Stouffer (eq. 6.7). NOTE: assumes INDEPENDENT probes, which adjacent
    ## CpGs violate - so this p-value needs permutation calibration.
    if (k >= min_probes) out[[length(out)+1]] <-
      data.frame(start = ps[i], end = ps[j], n_probes = k, z = sum(zs[i:j])/sqrt(k))
    i <- j + 1
  }
  if (length(out)) do.call(rbind, out) else data.frame()
}
regions <- region_scan(z_scores, probe_pos)
cat(sprintf("  %d candidate regions with >= 3 probes\n", nrow(regions)))
cat(sprintf("  largest |region z|: %.2f\n", max(abs(regions$z))))
cat("\n  CRITICAL: the Stouffer combination assumes INDEPENDENT probes. Adjacent\n")
cat("  CpGs are spatially correlated, so this p-value is anticonservative.\n")
cat("  Calibrate by permuting SAMPLE LABELS and rerunning the whole scan -\n")
cat("  which preserves the spatial correlation structure:\n")
null_max <- replicate(50, {
  mp <- meta; mp$case <- sample(mp$case)
  rp <- probewise(M_df, mp, c("case","age","sex"))
  zp <- qnorm(rp$pvalue/2, lower.tail = FALSE) * sign(rp$effect)
  rg <- region_scan(zp, probe_pos)
  if (nrow(rg)) max(abs(rg$z)) else 0
})
thresh <- quantile(null_max, 0.95)
cat(sprintf("    95th percentile of the max |region z| under permutation: %.2f\n", thresh))
cat(sprintf("    regions exceeding it: %d\n", sum(abs(regions$z) > thresh)))
cat(sprintf("    naive |z| > 1.96 would have called: %d\n", sum(abs(regions$z) > 1.96)))

#' ## 7. Figure

#+ figure
png(file.path(OUT, "methylation.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
bb <- seq(0.02, 0.98, length.out = 60)
sds <- sapply(bb, function(b) sd(m_to_beta(rnorm(3000, log2(b/(1-b)), 0.5))))
plot(bb, sds, type = "l", lwd = 2, col = "steelblue", xlab = "mean methylation beta",
     ylab = "SD", main = "Eq. (23.1): beta is heteroscedastic")
abline(h = 0.5, col = "darkorange", lty = 2)
plot(as.vector(as.matrix(meta[, CELL_TYPES])), as.vector(W_hat), pch = 16, cex = 0.4,
     col = adjustcolor("steelblue", 0.5), xlab = "true proportion",
     ylab = "deconvolved (eq. 23.3)", main = "Reference-based deconvolution")
abline(0, 1, lty = 2)
plot(mean_beta, -log10(res_adj$pvalue), pch = 16, cex = 0.3,
     col = ifelse(truth$is_dmp, "firebrick", adjustcolor("grey50", 0.3)),
     xlab = "mean beta", ylab = "-log10 p (M scale, adjusted)",
     main = "Hits are not concentrated at the boundaries")
hist(null_max, breaks = 25, col = "lightsteelblue", border = "white",
     xlim = range(c(null_max, max(abs(regions$z)))),
     main = "Region significance by label permutation", xlab = "max |region z|")
abline(v = thresh, col = "red", lty = 2); abline(v = max(abs(regions$z)), lwd = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "methylation.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Report an M-scale effect as a beta difference
#'
#' Take the top 5 hits and convert their effects into "percentage points of
#' methylation". Why can you not just exponentiate the M coefficient?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# top <- rownames(res_adj)[order(res_adj$pvalue)][1:5]
# cat(sprintf("  %12s%10s%11s%14s%14s%12s\n", "probe","M effect","mean beta",
#             "beta at ctrl","beta at case","difference"))
# for (pr in top) {
#   eff <- res_adj[pr, "effect"]
#   m_ctrl <- mean(M_df[pr, meta$case == 0])
#   b_ctrl <- m_to_beta(m_ctrl); b_case <- m_to_beta(m_ctrl + eff)
#   cat(sprintf("  %12s%10.3f%11.3f%14.3f%14.3f%+12.3f\n", pr, eff,
#               mean(beta_df[pr, ]), b_ctrl, b_case, b_case - b_ctrl))
# }
#
# ## The map from M to beta is NONLINEAR (a logistic), so the SAME M-scale
# ## effect corresponds to a different beta difference depending on where you
# ## start. Near beta = 0.5 the map is steepest; near 0 or 1 a large M change
# ## barely moves beta. Evaluate m_to_beta() at the ACTUAL baseline, which is
# ## why you report "beta went from 0.42 to 0.55".

#' ### Problem 2: Confounder or mediator?
#'
#' Simulate a variant where cell composition is a pure MEDIATOR. Show that
#' adjusting destroys the total effect, and compute the mediated proportion.

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(77)
# n <- 200
# case <- rbinom(n, 1, 0.5)
# gran <- 0.30 + 0.10*case + rnorm(n, 0, 0.04)     # disease -> composition
# meth <- 2.0*gran + rnorm(n, 0, 0.15)             # composition -> methylation
# d <- data.frame(case = case, gran = gran, meth = meth)
# tot <- lm(meth ~ case, data = d); dir_ <- lm(meth ~ case + gran, data = d)
# a_path <- coef(lm(gran ~ case, data = d))["case"]
# cat(sprintf("  total effect (TE)             = %+.4f  p = %.2e\n",
#             coef(tot)["case"], coef(summary(tot))["case",4]))
# cat(sprintf("  direct effect after adjusting = %+.4f  p = %.3f\n",
#             coef(dir_)["case"], coef(summary(dir_))["case",4]))
# cat(sprintf("  indirect (NIE) = a*b           = %+.4f\n",
#             a_path*coef(dir_)["gran"]))
# cat(sprintf("  proportion mediated            = %.1f%%\n",
#             100*a_path*coef(dir_)["gran"]/coef(tot)["case"]))
#
# ## There IS a real total effect of disease on methylation - it just runs
# ## THROUGH cell composition. Adjusting for the mediator makes it vanish and
# ## you would wrongly conclude "no methylation difference". The CAUSAL question
# ## decides the model; the data cannot tell you which role composition plays.

#' ### Problem 3: Probe-number bias in gene-set testing
#'
#' Assign probes to genes with unequal probes-per-gene. Show that a naive
#' hypergeometric test on "genes with >= 1 significant probe" is biased toward
#' probe-rich genes even when significance is assigned AT RANDOM.

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(88)
# ppg <- rnbinom(400, mu = 7, size = 1.2) + 1
# gene_of_probe <- rep(seq_along(ppg), ppg)[1:nrow(truth)]
# sig_probe <- runif(length(gene_of_probe)) < 0.05    # COMPLETELY AT RANDOM
# gene_hit <- tapply(sig_probe, gene_of_probe, any)
# n_probes <- tapply(rep(1, length(gene_of_probe)), gene_of_probe, sum)
# cat(sprintf("  %17s%17s\n", "probes per gene", "P(gene called)"))
# for (rg in list(c(1,1), c(2,4), c(5,9), c(10,100))) {
#   m <- n_probes >= rg[1] & n_probes <= rg[2]
#   cat(sprintf("  %17s%16.1f%%\n", sprintf("%d-%d", rg[1], rg[2]),
#               100*mean(gene_hit[m])))
# }
#
# ## Probe-rich genes are called far more often, purely because they have more
# ## chances to contain a random hit. Any pathway enriched for probe-rich genes
# ## (promoter CpG islands, long genes) looks "significant" in every study.
# ## missMethyl::gometh corrects exactly this bias; a plain hypergeometric test
# ## on the gene list does not.

#' ## What to take away
#'
#' 1. **Model M, report beta.** Model bisulfite counts as beta-binomial.
#' 2. Decide the CAUSAL ROLE of cell composition before adjusting.
#' 3. Filter probes by outcome-independent quality criteria first.
#' 4. Use region-level inference, calibrated by LABEL PERMUTATION.
#' 5. Correct gene-set tests for probe-number bias.
#'
#' **Next:** `34_gwas_association.R`
