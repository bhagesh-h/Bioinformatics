#' ---
#' title: "Applied 28b - Survival analysis with molecular biomarkers"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 28, equations (28.4)-(28.13)
#'
#' ## The question
#'
#' Is a gene **prognostic**, and would a risk score built from expression
#' generalise to the next cohort?

#+ setup, message = FALSE
suppressPackageStartupMessages({library(survival); library(splines)})
MODULE_NAME <- "28b_survival_biomarkers"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Simulate a molecular survival cohort

#+ simulate
header("1. A TCGA-like cohort with a few truly prognostic genes")
simulate_cohort <- function(n = 400, G = 200, n_prog = 5, beta = 0.55,
                            censor_scale = 9) {
  expr <- matrix(rnorm(n*G), n)
  ## Genes are correlated in MODULES, as real expression is.
  for (b in seq(1, G, by = 25)) {
    cols <- b:min(b+24, G); latent <- rnorm(n)
    expr[, cols] <- 0.6*latent + 0.8*expr[, cols]
  }
  expr <- scale(expr)
  age <- rnorm(n, 62, 11); stage <- sample(1:3, n, TRUE, c(0.4, 0.35, 0.25))
  prog <- seq_len(n_prog)
  lp <- as.vector(expr[, prog] %*% rep(beta, n_prog)) + 0.02*(age - 62) + 0.45*(stage - 1)
  Tm <- rexp(n, 0.06*exp(lp))
  ## size = n: without it rexp() returns a SCALAR and every patient would share
  ## one censoring time (a bug worth remembering).
  Cm <- rexp(n, 1/censor_scale)
  d <- data.frame(expr); names(d) <- sprintf("G%03d", seq_len(G))
  d$time <- pmin(Tm, Cm); d$event <- as.integer(Tm <= Cm)
  d$age <- age; d$stage <- stage
  list(d = d, is_prog = setNames(seq_len(G) %in% prog, sprintf("G%03d", seq_len(G))))
}
set.seed(3801)
sim <- simulate_cohort()
d <- sim$d; is_prog <- sim$is_prog
genes <- names(is_prog)
cat(sprintf("  n = %d, events = %d (%.0f%%), censored = %d\n", nrow(d), sum(d$event),
            100*mean(d$event), sum(!d$event)))
cat(sprintf("  %d genes, %d truly prognostic: %s\n", length(genes), sum(is_prog),
            paste(names(is_prog)[is_prog], collapse = " ")))
cat(sprintf("\n  Effective sample size for a Cox model is the NUMBER OF EVENTS (%d),\n",
            sum(d$event)))
cat(sprintf("  not the number of patients. Guidance: >= 10 events per covariate\n"))
cat(sprintf("  -> at most %d covariates.\n", sum(d$event) %/% 10))

#' ## 2. Kaplan-Meier and the at-risk table, eq. (28.6)-(28.7)

#+ km
header("2. Kaplan-Meier with the at-risk table (28.6)-(28.7)")
km <- survfit(Surv(time, event) ~ 1, data = d)
cat(sprintf("  median survival = %.2f\n", summary(km)$table["median"]))
cat(sprintf("\n  %8s%10s%9s%22s\n", "time", "at risk", "S(t)", "95% CI"))
for (t in c(1, 2, 4, 8, 12, 16)) {
  s <- summary(km, times = t)
  if (!length(s$surv)) next
  cat(sprintf("  %8d%10d%9.3f   [%.3f, %.3f]\n", t, s$n.risk, s$surv, s$lower, s$upper))
}
cat("\n  Watch the at-risk column collapse. The right tail of a KM curve is\n")
cat("  estimated from a handful of patients and looks far more precise than it\n")
cat("  is. ALWAYS publish the number-at-risk table under the curve.\n")

#' ## 3. Gene-wise Cox screening, eq. (28.9)-(28.10)

#+ cox-screen
header("3. Screening genes with Cox models (28.9)-(28.10)")
cox_screen <- function(d, genes, adjust = c("age","stage")) {
  res <- t(sapply(genes, function(g) {
    f <- try(coxph(as.formula(paste("Surv(time, event) ~", g, "+",
                                    paste(adjust, collapse = "+"))), data = d),
             silent = TRUE)
    if (inherits(f, "try-error")) return(c(NA, NA, 1))
    cs <- summary(f)$coefficients
    c(cs[g, "coef"], cs[g, "se(coef)"], cs[g, "Pr(>|z|)"])
  }))
  out <- data.frame(logHR = res[, 1], se = res[, 2], p = res[, 3], row.names = genes)
  out$HR <- exp(out$logHR); out$padj <- p.adjust(out$p, "BH"); out
}
res <- cox_screen(d, genes)
rej <- res$padj < 0.05
cat(sprintf("  %d genes at FDR 5%%; %d of them truly prognostic\n",
            sum(rej), sum(rej & is_prog[rownames(res)])))
cat("\n  Top 8 by p-value:\n")
cat(sprintf("  %8s%8s%9s%12s%11s%9s\n", "gene","HR","logHR","p","padj","truth"))
for (g in rownames(res)[order(res$p)][1:8])
  cat(sprintf("  %8s%8.3f%9.3f%12.2e%11.2e%9s\n", g, res[g,"HR"], res[g,"logHR"],
              res[g,"p"], res[g,"padj"], if (is_prog[g]) "PROG" else "-"))
cat("\n  exp(beta) is a HAZARD RATIO: a ratio of instantaneous event rates among\n")
cat("  those still at risk. NOT a risk ratio, and NOT a ratio of survival times.\n")

#' ## 4. The optimal cut-point trap

#+ cutpoint
header("4. Optimal cut-point bias, measured")
best_cutpoint_p <- function(d, gene, qs = seq(0.2, 0.8, 0.05)) {
  ps <- sapply(qs, function(q) {
    hi <- as.integer(d[[gene]] > quantile(d[[gene]], q))
    if (sum(hi) < 10 || sum(1-hi) < 10) return(NA)
    pchisq(survdiff(Surv(time, event) ~ hi, data = d)$chisq, 1, lower.tail = FALSE)
  })
  min(ps, na.rm = TRUE)
}
## Null genes from a DIFFERENT correlation module: genes 1-25 form the module
## containing the prognostic genes, so their module-mates are indirectly
## associated and are not true nulls.
null_genes <- genes[51:length(genes)][1:80]
p_best <- sapply(null_genes, function(g) best_cutpoint_p(d, g))
p_cont <- res[null_genes, "p"]
cat(sprintf("  Tested %d genes from modules containing NO prognostic gene:\n",
            length(null_genes)))
cat(sprintf("    'best cut-point' p < 0.05  : %.1f%%   <- should be 5%%\n",
            100*mean(p_best < 0.05)))
cat(sprintf("    continuous Cox p < 0.05    : %.1f%%\n", 100*mean(p_cont < 0.05)))
cat(sprintf("    'best cut-point' p < 0.01  : %.1f%%\n", 100*mean(p_best < 0.01)))
cat("\n  Searching over cut-points and reporting the best inflates the error\n")
cat("  rate several-fold. Keep the biomarker CONTINUOUS (with a spline if the\n")
cat("  effect may be nonlinear), or pre-specify the cut-point and validate it\n")
cat("  in an independent cohort.\n")

#' ## 5. Proportional hazards and RMST, eq. (28.11)-(28.12)

#+ ph-rmst
header("5. Checking PH, and an estimand that does not need it (28.11)-(28.12)")
top_gene <- rownames(res)[which.min(res$p)]
cph <- coxph(as.formula(paste("Surv(time, event) ~", top_gene, "+ age + stage")), data = d)
cat("  PH test (scaled Schoenfeld residuals, eq. 28.11):\n")
print(round(cox.zph(cph)$table, 4))
rmst <- function(time, event, tau) {
  k <- survfit(Surv(time, event) ~ 1)
  ts <- c(0, k$time[k$time <= tau], tau)
  ss <- c(1, k$surv[k$time <= tau])
  sum(ss * diff(ts))                                             # eq. (28.12)
}
TAU <- 10
hi <- d[[top_gene]] > median(d[[top_gene]])
r_hi <- rmst(d$time[hi], d$event[hi], TAU); r_lo <- rmst(d$time[!hi], d$event[!hi], TAU)
cat(sprintf("\n  RMST up to tau = %d (eq. 28.12):\n", TAU))
cat(sprintf("    low  %s: %.3f\n    high %s: %.3f\n    difference: %+.3f time units\n",
            top_gene, r_lo, top_gene, r_hi, r_lo - r_hi))
cat("  RMST differences are directly interpretable as 'time gained' and need\n")
cat("  NO proportional-hazards assumption - increasingly preferred when the PH\n")
cat("  test fails or the curves cross.\n")

#' ## 6. Building and validating a risk score, eq. (31.3)-(31.4)

#+ risk-score
header("6. A multi-gene risk score, honestly validated")
make_folds <- function(n, k, seed = 1) { set.seed(seed); sample(rep_len(1:k, n)) }
## concordance(): larger x must mean LONGER survival, so a hazard-scale linear
## predictor (larger = worse) needs reverse = TRUE. Get this backwards and every
## C-index comes out as 1 - C, which looks like a uselessly bad model.
cidx <- function(test, risk)
  concordance(Surv(time, event) ~ risk,
              data = cbind(test["time"], test["event"], risk = risk),
              reverse = TRUE)$concordance
fit_score <- function(train, sel)
  coxph(as.formula(paste("Surv(time, event) ~",
                         paste(c(sel, "age", "stage"), collapse = "+"))), data = train)
select_genes <- function(train, genes, p_thresh = 0.01) {
  r <- cox_screen(train, genes)
  sel <- rownames(r)[r$p < p_thresh]
  if (!length(sel)) sel <- rownames(r)[order(r$p)][1:3]
  sel
}
fold <- make_folds(nrow(d), 5, seed = 2)
sel_all <- select_genes(d, genes)
## (a) HONEST: selection AND fitting inside each fold.
c_honest <- mean(sapply(1:5, function(k) {
  tr <- d[fold != k, ]; te <- d[fold == k, ]
  cidx(te, predict(fit_score(tr, select_genes(tr, genes)), newdata = te, type = "lp"))
}))
## (b) Clinical-only baseline - the comparison that is usually missing.
c_clin <- mean(sapply(1:5, function(k) {
  tr <- d[fold != k, ]; te <- d[fold == k, ]
  f <- coxph(Surv(time, event) ~ age + stage, data = tr)
  cidx(te, predict(f, newdata = te, type = "lp"))
}))
f_all <- fit_score(d, sel_all)
c_apparent <- cidx(d, predict(f_all, type = "lp"))
cat(sprintf("  %-46s%10s\n", "evaluation", "C-index"))
cat(sprintf("  %-46s%10.3f\n", "apparent (fit and test on all data)", c_apparent))
cat(sprintf("  %-46s%10.3f\n", "CV, selection INSIDE each fold (honest)", c_honest))
cat(sprintf("  %-46s%10.3f\n", "CV, age + stage only (clinical baseline)", c_clin))
cat(sprintf("  %-46s%10.3f\n", "chance", 0.5))
cat(sprintf("\n  apparent optimism (eq. 31.4)   : %+.3f\n", c_apparent - c_honest))
cat(sprintf("  gain over clinical baseline    : %+.3f\n", c_honest - c_clin))
cat(sprintf("  genes selected on the full data: %d (%d truly prognostic)\n",
            length(sel_all), sum(is_prog[sel_all])))
cat("\n  ALWAYS report the clinical-only baseline. A molecular score that does\n")
cat("  not beat age and stage is not a useful biomarker, however small its\n")
cat("  p-value. Here the 5 simulated genes are strong, so the score clears the\n")
cat("  baseline by a wide margin. Real prognostic signatures rarely do.\n")

#' ### 6b. Why the selection leak needs a null cohort to be seen

#+ leak-null
header("6b. Measuring the selection leak on cohorts with NO prognostic gene")
## In section 6 the leaky and honest C-indices are nearly identical. That is not
## evidence the leak is harmless - it is because the 5 true genes have a large
## effect and get selected in EVERY fold, so 'selecting on all the data' picks
## the same genes anyway. The leak is invisible when the signal is strong.
##
## Turn the signal OFF (beta = 0) and it becomes glaring: the honest C-index
## must sit at 0.5, and anything above that is pure leakage.
set.seed(3803)
REP <- 12
cl <- ch <- cc <- numeric(REP)
for (i in 1:REP) {
  s0 <- simulate_cohort(n = 150, G = 200, n_prog = 0, beta = 0)
  d0 <- s0$d; g0 <- names(s0$is_prog)
  fo <- make_folds(nrow(d0), 5, seed = 100 + i)
  sel0 <- select_genes(d0, g0)                        # selected on ALL the data
  ch[i] <- mean(sapply(1:5, function(k) {
    tr <- d0[fo != k, ]; te <- d0[fo == k, ]
    cidx(te, predict(fit_score(tr, select_genes(tr, g0)), newdata = te, type = "lp"))
  }))
  cl[i] <- mean(sapply(1:5, function(k) {
    tr <- d0[fo != k, ]; te <- d0[fo == k, ]
    cidx(te, predict(fit_score(tr, sel0), newdata = te, type = "lp"))
  }))
  ## The reference is NOT 0.5: age and stage stay genuinely prognostic even when
  ## no gene is. The honest gene score should match this, adding nothing.
  cc[i] <- mean(sapply(1:5, function(k) {
    tr <- d0[fo != k, ]; te <- d0[fo == k, ]
    f <- coxph(Surv(time, event) ~ age + stage, data = tr)
    cidx(te, predict(f, newdata = te, type = "lp"))
  }))
}
cat(sprintf("  %d null cohorts (n = 150, 200 genes, NO true prognostic gene):\n", REP))
cat(sprintf("  %-42s%10s%16s\n", "evaluation", "mean C", "beats clinical"))
cat(sprintf("  %-42s%10.3f%15.0f%%\n", "CV, genes selected on ALL data (LEAKY)",
            mean(cl), 100*mean(cl > cc)))
cat(sprintf("  %-42s%10.3f%15.0f%%\n", "CV, selection INSIDE each fold (honest)",
            mean(ch), 100*mean(ch > cc)))
cat(sprintf("  %-42s%10.3f%16s\n", "CV, age + stage only (the true ceiling)",
            mean(cc), "-"))
cat(sprintf("\n  leaky  - clinical baseline: %+.3f   <- entirely spurious\n", mean(cl - cc)))
cat(sprintf("  honest - clinical baseline: %+.3f\n", mean(ch - cc)))
cat("\n  Note the reference is the CLINICAL baseline, not 0.5: age and stage stay\n")
cat("  prognostic here, so even a gene score built from pure noise inherits\n")
cat("  their skill. Read the two gaps against it:\n")
cat("\n    honest CV lands BELOW the clinical baseline. Correct - the noise genes\n")
cat("    add parameters and no signal, so they cost precision and degrade the\n")
cat("    model. An honest evaluation is able to say 'these genes make it worse'.\n")
cat("\n    leaky CV lands ABOVE it, beating the baseline in every single cohort.\n")
cat("    The genes were chosen while looking at the test fold, so the score is\n")
cat("    scored on the very data that defined it. Select features INSIDE the\n")
cat("    fold - the same leak as Module 31.\n")

#' ## 7. Immortal time bias

#+ immortal
header("7. Immortal time bias and the landmark fix")
set.seed(3802)
LANDMARK <- 2
d2 <- d
## "Responders" can only be assessed at the landmark - so to BE a responder you
## must first SURVIVE to it. The label has NO real effect on survival.
d2$responder <- as.integer(d2$time > LANDMARK & runif(nrow(d2)) < 0.5)
cph_bad <- coxph(Surv(time, event) ~ responder, data = d2)
land <- subset(d2, time > LANDMARK); land$time <- land$time - LANDMARK
cph_land <- coxph(Surv(time, event) ~ responder, data = land)
cat("  TRUE effect of 'responder' = NONE (assigned at random)\n")
cat(sprintf("  naive Cox HR            = %.3f  (p = %.2e)   <- spurious\n",
            exp(coef(cph_bad)), summary(cph_bad)$coefficients[1, "Pr(>|z|)"]))
cat(sprintf("  landmark analysis at t=%d: HR = %.3f  (p = %.3f)   <- corrected\n",
            LANDMARK, exp(coef(cph_land)), summary(cph_land)$coefficients[1, "Pr(>|z|)"]))
cat("\n  The bias is STRUCTURAL: to be classified you must survive long enough\n")
cat("  to be classified. Fix with a landmark analysis or a time-dependent\n")
cat("  covariate. Same 'time zero' problem target-trial emulation prevents.\n")

#' ## 8. Figure

#+ figure
png(file.path(OUT, "survival_biomarkers.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
plot(survfit(Surv(time, event) ~ hi, data = d), col = c("steelblue","darkorange"),
     lwd = 2, xlab = "time", ylab = "S(t)",
     main = sprintf("KM by median split (log-rank p = %.1e)",
                    pchisq(survdiff(Surv(time, event) ~ hi, data = d)$chisq, 1,
                           lower.tail = FALSE)))
legend("topright", c("low","high"), col = c("steelblue","darkorange"), lwd = 2,
       bty = "n", cex = 0.7)
hist(p_best, breaks = 25, col = adjustcolor("steelblue", 0.6), border = "white",
     main = "Optimal cut-point bias", xlab = "p-value (genes with NO true effect)")
hist(p_cont, breaks = 25, col = adjustcolor("darkorange", 0.6), border = "white", add = TRUE)
abline(v = 0.05, col = "red", lty = 2)
legend("topright", c("'best' cut-point","continuous Cox"),
       fill = c(adjustcolor("steelblue",0.6), adjustcolor("darkorange",0.6)),
       bty = "n", cex = 0.6)
barplot(c(c_apparent, c_honest, c_clin, mean(cl), mean(ch), mean(cc)), ylim = c(0.4, 0.95),
        names.arg = c("appar.","honest","clin.","null\nleaky","null\nhonest","null\nclin."),
        xpd = FALSE, col = c("firebrick","steelblue","grey60","darkorange","steelblue","grey60"),
        ylab = "C-index", main = "Eq. (31.4): apparent vs validated", cex.names = 0.6)
abline(h = 0.5, lty = 2)
plot(res$logHR, -log10(res$p), pch = 16, cex = 0.6,
     col = ifelse(is_prog[rownames(res)], "firebrick", adjustcolor("grey50", 0.5)),
     xlab = "log hazard ratio", ylab = "-log10 p", main = "Cox screen (eq. 28.10)")
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "survival_biomarkers.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Events, not patients
#'
#' Re-run the Cox screen on subsets with different PATIENT counts, then on
#' cohorts with different CENSORING (so events vary). Which predicts power?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(9)
# cat(sprintf("  %-28s%10s%9s%19s\n","subset","patients","events","prognostic found"))
# for (n_sub in c(400, 200, 100)) {
#   idx <- sample(nrow(d), n_sub); sub <- d[idx, ]
#   r <- cox_screen(sub, names(is_prog)[is_prog])
#   cat(sprintf("  %-28s%10d%9d%19d\n", sprintf("%d random patients", n_sub),
#               n_sub, sum(sub$event), sum(r$p < 0.05)))
# }
# for (cs in c(20, 6, 2.5)) {
#   s2 <- simulate_cohort(n = 300, censor_scale = cs)
#   r <- cox_screen(s2$d, names(s2$is_prog)[s2$is_prog])
#   cat(sprintf("  %-28s%10d%9d%19d\n", sprintf("n=300, censor scale %g", cs),
#               300, sum(s2$d$event), sum(r$p < 0.05)))
# }
#
# ## Power tracks EVENTS, not patients. A large cohort with few events is a
# ## SMALL study. This is why survival trials are powered on the number of
# ## events and why 'events per variable >= 10' is the rule of thumb.

#' ### Problem 2: Competing risks
#'
#' Add a competing cause of death and compare 1 - KM against the CIF (28.13).

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(13)
# n <- 900
# T1 <- rexp(n, 1/14); T2 <- rexp(n, 1/9); Cc <- rexp(n, 1/30)
# obs <- pmin(T1, T2, Cc)
# cause <- ifelse(T1 <= T2 & T1 <= Cc, 1, ifelse(T2 < T1 & T2 <= Cc, 2, 0))
# t_eval <- 10
# km_n <- survfit(Surv(obs, as.integer(cause == 1)) ~ 1)
# naive <- 1 - summary(km_n, times = t_eval)$surv
# cif <- function(time, cause, k, t_eval) {
#   o <- order(time); time <- time[o]; cause <- cause[o]; S <- 1; out <- 0
#   for (ti in sort(unique(time[cause > 0]))) {
#     n_i <- sum(time >= ti); d_k <- sum(time == ti & cause == k)
#     d_all <- sum(time == ti & cause > 0)
#     if (ti <= t_eval) out <- out + S*d_k/n_i
#     S <- S*(1 - d_all/n_i)
#   }
#   out
# }
# cat(sprintf("  at t = %d:\n", t_eval))
# cat(sprintf("    TRUE probability of relapse  = %.4f\n", mean(T1 <= t_eval & T1 <= T2)))
# cat(sprintf("    1 - Kaplan-Meier (naive)     = %.4f   <- overestimates\n", naive))
# cat(sprintf("    cumulative incidence (28.13) = %.4f\n", cif(obs, cause, 1, t_eval)))
#
# ## Treating the competing death as CENSORING asks 'what would the relapse risk
# ## be in a world where nobody died of anything else?' - rarely the clinical
# ## question. Use CIF for absolute risk, cause-specific hazards for aetiology,
# ## and Fine-Gray for subdistribution modelling.

#' ### Problem 3: Nonlinear biomarker effects
#'
#' Fit the top gene with a natural spline in the Cox model and compare to the
#' linear fit with a likelihood-ratio test. Is linearity adequate?

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# f_lin <- coxph(as.formula(paste("Surv(time, event) ~", top_gene, "+ age + stage")), data = d)
# f_spl <- coxph(as.formula(paste0("Surv(time, event) ~ ns(", top_gene,
#                                  ", df = 4) + age + stage")), data = d)
# cat(sprintf("  linear partial log-lik  = %.3f\n", f_lin$loglik[2]))
# cat(sprintf("  spline partial log-lik  = %.3f\n", f_spl$loglik[2]))
# lr <- 2*(f_spl$loglik[2] - f_lin$loglik[2]); ddf <- 3
# cat(sprintf("  LRT chi2 = %.3f on %d df, p = %.4f\n", lr, ddf,
#             pchisq(max(lr,0), ddf, lower.tail = FALSE)))
#
# ## Here the truth IS linear, so the spline should NOT improve the fit
# ## significantly - a good negative control for the method. On real data a
# ## significant LRT means the linear hazard assumption is inadequate, and the
# ## spline fit is both more honest and more informative than dichotomising.

#' ## What to take away
#'
#' 1. Count **events**, not patients.
#' 2. Publish the at-risk table with every KM curve.
#' 3. Never dichotomise a biomarker at a data-chosen optimal cut-point.
#' 4. Check PH; switch estimand to RMST when it fails.
#' 5. Select features INSIDE the fold, and always report the clinical baseline.
#' 6. Competing risks need the CIF, never 1 - KM.
#'
#' **Next:** `29_gene_set_enrichment.R`
