#' ---
#' title: "Exercise 3 - Audit a biomarker classifier"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topics 18, 21, 31
#' **Core modules used:** 18, 21, 31, 38
#'
#' ## The brief
#'
#' A group reports a gene signature that predicts treatment response with a
#' large **AUC** in 200 patients. Their pipeline, paraphrased:
#'
#' > 1. Normalise and scale all 5,000 genes across the full cohort.
#' > 2. Impute the few missing clinical values using column means.
#' > 3. Keep the genes most associated with response (t-test on all patients).
#' > 4. Fit logistic regression on those genes.
#' > 5. Report cross-validated AUC.
#' >
#' > *"The cross-validation proves it generalises."*
#'
#' There are **four** distinct leaks in that list, plus a missing comparison.
#' Find them, quantify each, and produce the number you would believe.

#+ setup, message = FALSE
MODULE_NAME <- "E3_prediction_audit"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
N_GENES <- 25          # size of the signature the group reports

#' ## The data

#+ data
make_cohort <- function(seed = 41, n = 200, p = 2000, n_signal = 25,
                        effect = 0.70, n_sites = 8) {
  set.seed(seed)
  X <- matrix(rnorm(n*p), n)
  ## Genes come in correlated modules, as always.
  for (b in seq(1, p, by = 50)) {
    cols <- b:min(b + 49, p)
    X[, cols] <- 0.55*rnorm(n) + 0.83*X[, cols]
  }
  ## Patients come from 8 recruiting sites; site shifts a block of genes and
  ## also shifts the response rate. The site is the natural grouping unit.
  site <- sample(seq_len(n_sites), n, replace = TRUE)
  site_shift <- matrix(rnorm(n_sites*p, 0, 0.8), n_sites)
  X <- X + site_shift[site, ]
  sig <- seq_len(n_signal)
  age <- rnorm(n, 63, 10)
  stage <- sample(1:3, n, replace = TRUE)
  lp <- effect*rowSums(X[, sig, drop = FALSE])/sqrt(n_signal) +
        0.06*(age - 63) + 0.9*(stage - 2) + 0.30*(site - n_sites/2)/n_sites
  y <- rbinom(n, 1, plogis(lp))
  clin <- data.frame(age = age, stage = as.numeric(stage))
  clin$age[sample(n, 18)] <- NA          # a few missing values, as in real data
  list(X = X, y = y, site = site, clin = clin, signal = sig)
}
header("The cohort")
C <- make_cohort()
X <- C$X; y <- C$y; site <- C$site; clin <- C$clin
cat(sprintf("  %d patients x %d genes\n", nrow(X), ncol(X)))
cat(sprintf("  response rate: %.1f%% (%d responders)\n", 100*mean(y), sum(y)))
cat(sprintf("  %d recruiting sites, sizes %s\n", length(unique(site)),
            paste(as.vector(table(site)), collapse = " ")))
cat(sprintf("  missing ages: %d\n", sum(is.na(clin$age))))
cat(sprintf("  events per candidate predictor if all %d genes used: %.4f (want >= 10)\n",
            ncol(X), sum(y)/ncol(X)))

#' ## Helpers

#+ helpers
auc <- function(y, p) {
  ## Mann-Whitney form of the AUC: P(score of a positive > score of a
  ## negative), with ties counted as a half.
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1*(n1 + 1)/2)/(n1*n0)
}
scale_with <- function(M, ctr, scl) sweep(sweep(M, 2, ctr), 2, scl, "/")
colt <- function(M, grp) {
  ## Vectorised two-sample t statistic per column - fast enough to run inside
  ## every fold, which is the whole point.
  a <- M[grp == 1,, drop = FALSE]; b <- M[grp == 0,, drop = FALSE]
  (colMeans(a) - colMeans(b)) /
    sqrt(apply(a, 2, var)/nrow(a) + apply(b, 2, var)/nrow(b))
}
make_folds <- function(y, k, groups = NULL, seed = 1) {
  set.seed(seed)
  if (is.null(groups)) {
    ## Stratified by outcome so every fold contains both classes.
    f <- integer(length(y))
    for (cl in unique(y)) {
      idx <- which(y == cl)
      f[idx] <- sample(rep_len(seq_len(k), length(idx)))
    }
    f
  } else {
    ## Whole GROUPS are assigned to folds, never split across them.
    g <- unique(groups)
    ga <- sample(rep_len(seq_len(k), length(g)))
    ga[match(groups, g)]
  }
}

#' ### Q1: Reproduce their number

#+ q1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# their_pipeline <- function(X, y, clin, n_genes = N_GENES, seed = 0) {
#   ## Steps 1-2: scale and impute using ALL patients   <- LEAK 1 and LEAK 2
#   Xs <- scale(X)
#   cl <- clin
#   cl$age[is.na(cl$age)] <- mean(cl$age, na.rm = TRUE)
#   ## Step 3: select genes using ALL patients                    <- LEAK 3
#   sel <- order(-abs(colt(Xs, y)))[seq_len(n_genes)]
#   Z <- data.frame(Xs[, sel, drop = FALSE], cl)
#   ## Step 5: cross-validate only the final fit    (and ignore site <- LEAK 4)
#   fold <- make_folds(y, 5, seed = seed)
#   pred <- numeric(length(y))
#   for (k in unique(fold)) {
#     tr <- fold != k
#     m <- suppressWarnings(glm(y[tr] ~ ., binomial, data = Z[tr,, drop = FALSE]))
#     pred[!tr] <- predict(m, newdata = Z[!tr,, drop = FALSE], type = "response")
#   }
#   auc(y, pred)
# }
# a_theirs <- mean(sapply(0:4, function(s) their_pipeline(X, y, clin, seed = s)))
# cat(sprintf("  Their reported CV AUC: %.3f\n", a_theirs))
#
# ## The four leaks, in the order they occur:
# ##   LEAK 1  scaling on the full cohort - the test fold's mean and SD leak in.
# ##   LEAK 2  mean-imputation computed on the full cohort - same problem.
# ##   LEAK 3  GENE SELECTION on the full cohort. This is the big one: the genes
# ##           were chosen because they looked good in the very patients used to
# ##           test them.
# ##   LEAK 4  patients from the same SITE appear in both training and test
# ##           folds, and site shifts both genes and response - so the model can
# ##           recognise the site instead of the biology.
# ## And the missing comparison: no age+stage-only baseline, so we cannot tell
# ## whether the genes add anything to what a clinician already knows.

#' ### Q2: Quantify each leak separately
#'
#' Fix the leaks **one at a time**. Which accounts for most of the inflation?

#+ q2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cv_auc <- function(seed = 0, fold_scale = FALSE, fold_impute = FALSE,
#                    fold_select = FALSE, group_cv = FALSE,
#                    use_genes = TRUE, use_clin = TRUE, n_genes = N_GENES) {
#   fold <- if (group_cv) make_folds(y, 5, groups = site, seed = seed)
#           else make_folds(y, 5, seed = seed)
#   ## Whatever is NOT done inside the fold is done once, up front - which is
#   ## precisely the leak.
#   Xg <- if (fold_scale) X else scale(X)
#   cg <- clin
#   if (!fold_impute) cg$age[is.na(cg$age)] <- mean(cg$age, na.rm = TRUE)
#   sel_global <- if (!fold_select) order(-abs(colt(Xg, y)))[seq_len(n_genes)]
#   pred <- rep(NA_real_, length(y))
#   for (k in unique(fold)) {
#     tr <- fold != k
#     if (length(unique(y[tr])) < 2 || sum(!tr) == 0) next
#     Xtr <- Xg[tr,, drop = FALSE]; Xte <- Xg[!tr,, drop = FALSE]
#     if (fold_scale) {
#       ctr <- colMeans(Xtr); s <- apply(Xtr, 2, sd); s[s == 0] <- 1
#       Xtr <- scale_with(Xtr, ctr, s); Xte <- scale_with(Xte, ctr, s)
#     }
#     ctr_df <- cg[tr,, drop = FALSE]; cte_df <- cg[!tr,, drop = FALSE]
#     if (fold_impute) {
#       mu <- mean(ctr_df$age, na.rm = TRUE)
#       ctr_df$age[is.na(ctr_df$age)] <- mu
#       cte_df$age[is.na(cte_df$age)] <- mu
#     }
#     sel <- if (fold_select) order(-abs(colt(Xtr, y[tr])))[seq_len(n_genes)]
#            else sel_global
#     parts_tr <- list(); parts_te <- list()
#     if (use_genes) {
#       parts_tr$g <- Xtr[, sel, drop = FALSE]; parts_te$g <- Xte[, sel, drop = FALSE]
#     }
#     if (use_clin) { parts_tr$c <- as.matrix(ctr_df); parts_te$c <- as.matrix(cte_df) }
#     Ztr <- as.data.frame(do.call(cbind, parts_tr))
#     Zte <- as.data.frame(do.call(cbind, parts_te))
#     names(Zte) <- names(Ztr) <- paste0("v", seq_len(ncol(Ztr)))
#     m <- suppressWarnings(glm(y[tr] ~ ., binomial, data = Ztr))
#     pred[!tr] <- predict(m, newdata = Zte, type = "response")
#   }
#   ok <- !is.na(pred)
#   auc(y[ok], pred[ok])
# }
# ## Average over several fold assignments: a single CV on 200 patients is noisy.
# avg <- function(...) mean(sapply(0:7, function(s) cv_auc(seed = s, ...)))
# cat(sprintf("  %-50s%8s%9s\n", "pipeline", "AUC", "change"))
# a0 <- avg()
# cat(sprintf("  %-50s%8.3f\n", "as reported (all four leaks)", a0))
# a1 <- avg(fold_scale = TRUE)
# cat(sprintf("  %-50s%8.3f%+9.3f\n", "+ scale inside the fold", a1, a1 - a0))
# a2 <- avg(fold_scale = TRUE, fold_impute = TRUE)
# cat(sprintf("  %-50s%8.3f%+9.3f\n", "+ impute inside the fold", a2, a2 - a1))
# a3 <- avg(fold_scale = TRUE, fold_impute = TRUE, fold_select = TRUE)
# cat(sprintf("  %-50s%8.3f%+9.3f\n", "+ SELECT GENES inside the fold", a3, a3 - a2))
# a4 <- avg(fold_scale = TRUE, fold_impute = TRUE, fold_select = TRUE,
#           group_cv = TRUE)
# cat(sprintf("  %-50s%8.3f%+9.3f\n", "+ group CV by recruiting site", a4, a4 - a3))
# cat(sprintf("\n  total inflation removed: %+.3f\n", a0 - a4))
#
# ## Look carefully at that table before drawing the obvious conclusion. The
# ## total inflation is SMALL - around 0.01 AUC - and almost all of the little
# ## there is comes from gene selection and grouped patients, as expected.
# ##
# ## It would be a serious mistake to read this as "the leaks do not matter".
# ## What it shows is that THE SIZE OF A LEAK DEPENDS ON THE SIGNAL. Here 25
# ## genes carry a strong, coherent effect, so the t-test picks essentially the
# ## same genes whether it sees the test fold or not. Selecting on all the data
# ## therefore changes almost nothing, and the honest and leaky pipelines agree.
# ##
# ## Now recall what that means. The leak is invisible precisely when the
# ## finding is real and you did not need the check - and it is devastating
# ## when the finding is not real and the check is the only thing that would
# ## have told you. Q5 measures exactly that case, and the same asymmetry
# ## appears in Module 38 (section 6b), where a selection leak worth +0.09
# ## C-index on null data is worth ~0.00 on data with real signal.
# ##
# ## So do not calibrate your concern from a table like this one. The
# ## distinction that survives is about MECHANISM, not magnitude: leaks that
# ## touch y (feature selection, grouped patients) can be catastrophic, while
# ## leaks that touch only X (scaling, imputation) are usually mild - cf.
# ## Module 40, where unsupervised factors barely leaked at all.

#' ### Q3: The missing baseline

#+ q3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# hon <- list(fold_scale = TRUE, fold_impute = TRUE, fold_select = TRUE,
#             group_cv = TRUE)
# a_clin <- do.call(avg, c(hon, list(use_genes = FALSE)))
# a_gene <- do.call(avg, c(hon, list(use_clin = FALSE)))
# a_both <- do.call(avg, hon)
# cat(sprintf("  %-42s%8s\n", "model (all leaks fixed)", "AUC"))
# cat(sprintf("  %-42s%8.3f\n", "age + stage only", a_clin))
# cat(sprintf("  %-42s%8.3f\n", "genes only", a_gene))
# cat(sprintf("  %-42s%8.3f\n", "genes + age + stage", a_both))
# cat(sprintf("  %-42s%8.3f\n", "chance", 0.5))
# cat(sprintf("\n  genes add over the clinical baseline: %+.3f AUC\n", a_both - a_clin))
#
# ## This is the number the original report never gave, and the only one a
# ## clinician cares about. A large AUC that does not beat age and stage is
# ## worth nothing; a modest AUC that does beat them may be worth a great deal.
# ## Always report the baseline, and compare on the SAME folds so the
# ## difference is paired.

#' ### Q4: Is it calibrated?
#'
#' Discrimination is not calibration. Compute the Brier score and the
#' calibration slope of the honest model.

#+ q4
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# honest_pred <- function(seed = 0, n_genes = N_GENES) {
#   fold <- make_folds(y, 5, groups = site, seed = seed)
#   pred <- rep(NA_real_, length(y))
#   for (k in unique(fold)) {
#     tr <- fold != k
#     if (length(unique(y[tr])) < 2 || sum(!tr) == 0) next
#     ctr <- colMeans(X[tr, ]); s <- apply(X[tr, ], 2, sd); s[s == 0] <- 1
#     Xtr <- scale_with(X[tr, ], ctr, s); Xte <- scale_with(X[!tr, ], ctr, s)
#     cl_tr <- clin[tr, ]; cl_te <- clin[!tr, ]
#     mu <- mean(cl_tr$age, na.rm = TRUE)
#     cl_tr$age[is.na(cl_tr$age)] <- mu; cl_te$age[is.na(cl_te$age)] <- mu
#     sel <- order(-abs(colt(Xtr, y[tr])))[seq_len(n_genes)]
#     Ztr <- as.data.frame(cbind(Xtr[, sel], as.matrix(cl_tr)))
#     Zte <- as.data.frame(cbind(Xte[, sel], as.matrix(cl_te)))
#     names(Zte) <- names(Ztr) <- paste0("v", seq_len(ncol(Ztr)))
#     m <- suppressWarnings(glm(y[tr] ~ ., binomial, data = Ztr))
#     pred[!tr] <- predict(m, newdata = Zte, type = "response")
#   }
#   pred
# }
# p <- honest_pred(); ok <- !is.na(p)
# pc <- pmin(pmax(p[ok], 1e-6), 1 - 1e-6)
# ## Calibration slope: regress the outcome on the predicted LOG-ODDS.
# slope <- coef(glm(y[ok] ~ qlogis(pc), binomial))[2]
# cat(sprintf("  AUC (discrimination)  : %.3f\n", auc(y[ok], pc)))
# cat(sprintf("  Brier score           : %.3f  (a constant predictor of %.2f scores %.3f)\n",
#             mean((pc - y[ok])^2), mean(y), mean((y - mean(y))^2)))
# cat(sprintf("  calibration slope     : %.3f   (1.0 = perfect)\n", slope))
# cat(sprintf("  mean predicted risk   : %.3f\n  observed event rate   : %.3f\n",
#             mean(pc), mean(y[ok])))
# qs <- quantile(pc, seq(0, 1, length.out = 6))
# cat(sprintf("\n  %-24s%6s%16s%11s\n", "predicted risk bin", "n",
#             "mean predicted", "observed"))
# for (i in 1:5) {
#   m_ <- pc >= qs[i] & pc <= qs[i + 1]
#   if (sum(m_))
#     cat(sprintf("  %-24s%6d%16.3f%11.3f\n",
#                 sprintf("[%.2f, %.2f]", qs[i], qs[i + 1]), sum(m_),
#                 mean(pc[m_]), mean(y[ok][m_])))
# }
#
# ## A calibration slope BELOW 1 means the predictions are too extreme - the
# ## model is overconfident, the normal consequence of fitting many parameters
# ## to few events with an UNPENALISED glm. Penalisation (ridge/lasso) or
# ## recalibration on held-out data fixes it, and it matters enormously for any
# ## model used to make a treatment decision: a model can rank patients
# ## perfectly (AUC 1.0) while every probability it reports is wrong.

#' ### Q5: The null cohort
#'
#' Rerun the ORIGINAL (leaky) pipeline where the response is pure noise.

#+ q5
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# leaky_null <- honest_null <- numeric(0)
# for (rep in 1:8) {
#   Cn <- make_cohort(seed = 500 + rep, effect = 0)   # NO signal at all
#   set.seed(900 + rep)
#   yn <- rbinom(length(Cn$y), 1, 0.5)                # response is a coin flip
#   ## Temporarily rebind the globals the helpers close over.
#   X <<- Cn$X; y <<- yn; site <<- Cn$site; clin <<- Cn$clin
#   leaky_null <- c(leaky_null, their_pipeline(X, y, clin, seed = rep))
#   honest_null <- c(honest_null, cv_auc(seed = rep, fold_scale = TRUE,
#                                        fold_impute = TRUE, fold_select = TRUE,
#                                        group_cv = TRUE))
# }
# X <<- C$X; y <<- C$y; site <<- C$site; clin <<- C$clin   # restore
# cat("  8 cohorts in which response is a COIN FLIP (true AUC = 0.500):\n\n")
# cat(sprintf("  %-44s%10s%8s\n", "pipeline", "mean AUC", "max"))
# cat(sprintf("  %-44s%10.3f%8.3f\n", "as reported (leaky)",
#             mean(leaky_null), max(leaky_null)))
# cat(sprintf("  %-44s%10.3f%8.3f\n", "all leaks fixed",
#             mean(honest_null), max(honest_null)))
#
# ## The leaky pipeline reports a strong AUC on data with NO signal whatsoever.
# ## That is the definitive test: a pipeline that cannot return 0.5 on noise
# ## cannot be trusted to return the truth on real data, and no amount of
# ## cross-validation language changes it. Run this check on your own
# ## pipelines - it takes minutes, and it cannot be argued with.

#' ## Debrief

#+ debrief
header("The generative truth")
cat("  25 of the genes carry a real but modest effect, shared across a
  correlated module. 8 recruiting sites shift both gene expression and
  response rate. Age and stage genuinely predict response.

  So the signature is NOT nothing. The honest model does beat the clinical
  baseline - but by far less than the headline number implied, and you only
  learn that by asking the question the original report never asked.

  The four leaks, ranked by how much damage they CAN do:

    1. FEATURE SELECTION outside the fold. Uses the test patients' OUTCOMES.
       Catastrophic, and the most common error in the literature.
    2. GROUPED PATIENTS split across folds. The model learns the site.
       Catastrophic when the group drives both X and y.
    3. IMPUTATION outside the fold. Uses test patients' predictors only. Mild.
    4. SCALING outside the fold. Same. Mild.

  Note the words 'CAN do'. On THIS cohort, with a strong real signal,
  removing all four changed the AUC by about 0.01 (Q2). On the null cohort of Q5 the same leaky
  pipeline reported ~0.72 where the truth was 0.50. A leak is a conditional
  hazard, not a fixed penalty: it is smallest exactly when your result is
  real, and largest exactly when it is not. That is why the null-outcome run
  is the check that matters, and why you cannot judge a pipeline by how
  little the leaks appear to cost on the data you happen to have.

  The rule that covers all four: EVERY step that looks at the data must
  happen inside the fold, and the unit of splitting must be the unit of
  independence (Topic 1 again - it is always Topic 1).\n")

#' ## What to take away
#'
#' 1. Cross-validation validates **the pipeline you cross-validate**.
#' 2. Leaks that touch $y$ are catastrophic; leaks that touch only $X$ are
#'    usually mild. Fix both anyway.
#' 3. Split on the **unit of independence** - patient, site, subject.
#' 4. Always report the clinical baseline and the increment over it.
#' 5. AUC says nothing about calibration, and decisions need calibration.
#' 6. Run the pipeline on a null outcome. It must return 0.5. This is the
#'    check that works even when the leak is invisible on your real data.
#'
#' **Next:** `E4_causal_question.R`
