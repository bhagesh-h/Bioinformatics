#' ---
#' title: "Applied 35 - Proteomics: missing values, batch structure, inference"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 25, equations (25.1)-(25.4)
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | A TMT experiment: 3 plexes x 2 conditions, peptide level |
#' | **Key threats** | 20-50% missing of MIXED mechanism, plex effects, run-order drift |
#'
#' Missingness in proteomics is a **mixture**: abundance-dependent censoring
#' (MNAR) plus stochastic identification failure (MAR). Applying one imputer to
#' both is the standard mistake - and it fails in opposite directions.

#+ setup, message = FALSE
MODULE_NAME <- "35_proteomics_missing_values"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Simulate a peptide-level TMT experiment

#+ simulate
header("1. Peptide intensities with plex effects and run-order drift")
simulate_proteomics <- function(n_proteins = 600, n_per_group = 9,
                                peptides_per_protein = 4, n_de = 100, effect = 0.9) {
  n <- 2*n_per_group
  cond <- factor(rep(c("ctrl", "case"), each = n_per_group), levels = c("ctrl","case"))
  plex <- factor(sprintf("plex%d", (seq_len(n) - 1) %% 3 + 1))   # BALANCED design
  run_order <- sample(n)
  prot_ab <- rnorm(n_proteins, 22, 2.3)
  true_eff <- numeric(n_proteins)
  de <- sample(n_proteins, n_de); true_eff[de] <- rnorm(n_de, 0, effect)
  plex_shift <- c(plex1 = 0, plex2 = 0.45, plex3 = -0.30)
  drift <- 0.02*(run_order - mean(run_order))
  rows <- list(); pep_meta <- list()
  for (p in seq_len(n_proteins)) {
    pep_off <- rnorm(peptides_per_protein, 0, 1.6)   # ionisation efficiency
    for (k in seq_len(peptides_per_protein)) {
      rows[[length(rows)+1]] <- prot_ab[p] + pep_off[k] +
        true_eff[p]*(cond == "case") + plex_shift[as.character(plex)] + drift +
        rnorm(n, 0, 0.35)
      pep_meta[[length(pep_meta)+1]] <-
        data.frame(protein = sprintf("P%04d", p), peptide = sprintf("P%04d_%d", p, k))
    }
  }
  Y <- do.call(rbind, rows)
  pm <- do.call(rbind, pep_meta)
  dimnames(Y) <- list(pm$peptide, sprintf("S%02d", seq_len(n)))
  ## MISSINGNESS: an explicit MIXTURE.
  lod_center <- quantile(Y, 0.22)
  p_mnar <- 1/(1 + exp((Y - lod_center)/0.55))      # abundance-dependent (MNAR)
  missing <- (matrix(runif(length(Y)), nrow(Y)) < p_mnar) |
    (matrix(runif(length(Y)), nrow(Y)) < 0.08)      # flat MAR component
  Yobs <- Y; Yobs[missing] <- NA
  list(Yobs = Yobs, Ytrue = Y, pep_meta = pm, lod = lod_center,
       smeta = data.frame(condition = cond, plex = plex, run_order = run_order,
                          row.names = colnames(Y)),
       truth = data.frame(true_eff = true_eff, is_de = true_eff != 0,
                          row.names = sprintf("P%04d", seq_len(n_proteins))))
}
set.seed(3501)
sim <- simulate_proteomics()
Yobs <- sim$Yobs; smeta <- sim$smeta; truth <- sim$truth; pep_meta <- sim$pep_meta
cat(sprintf("  %d peptides (%d proteins) x %d samples\n", nrow(Yobs), nrow(truth),
            ncol(Yobs)))
cat(sprintf("  overall missingness: %.1f%%\n", 100*mean(is.na(Yobs))))
cat(sprintf("  missingness per sample: %.1f%% - %.1f%%\n",
            100*min(colMeans(is.na(Yobs))), 100*max(colMeans(is.na(Yobs)))))
print(table(smeta$plex, smeta$condition))
cat("\n  Plexes are BALANCED across conditions - the design of Module 19.\n")

#' ## 2. Diagnose the missingness mechanism

#+ diagnose
header("2. Missingness vs abundance: the MNAR signature (Topic 15)")
miss_rate <- rowMeans(is.na(Yobs))
mean_obs <- rowMeans(Yobs, na.rm = TRUE)
ok <- is.finite(mean_obs)
sp <- cor(mean_obs[ok], miss_rate[ok], method = "spearman")
cat(sprintf("  Spearman(mean observed intensity, missingness rate) = %+.3f\n", sp))
qs <- cut(mean_obs[ok], quantile(mean_obs[ok], 0:5/5), include.lowest = TRUE,
          labels = paste0("Q", 1:5))
cat(sprintf("\n  %-24s%18s%21s\n", "abundance quintile", "mean missingness",
            "mean observed log2"))
for (lv in levels(qs))
  cat(sprintf("  %-24s%18.3f%21.2f\n", lv, mean(miss_rate[ok][qs == lv]),
              mean(mean_obs[ok][qs == lv])))
cat("\n  A strong NEGATIVE relationship is the signature of abundance-dependent\n")
cat("  (MNAR / left-censored) missingness. A flat cloud would indicate MAR.\n")
ctrl <- smeta$condition == "ctrl"; case <- smeta$condition == "case"
miss_c <- rowMeans(is.na(Yobs[, ctrl])); miss_s <- rowMeans(is.na(Yobs[, case]))
mnar_like <- (miss_c > 0.7 & miss_s < 0.3) | (miss_s > 0.7 & miss_c < 0.3)
cat(sprintf("\n  peptides missing systematically in ONE condition (MNAR-like): %d\n",
            sum(mnar_like)))
cat(sprintf("  peptides with sporadic gaps (MAR-like)                     : %d\n",
            sum(!mnar_like & miss_rate > 0 & miss_rate < 0.6)))
cat("  These two classes need DIFFERENT treatment (Topic 25).\n")

#' ## 3. Normalisation and run-order drift, eq. (25.2)

#+ normalise
header("3. Median normalisation and drift correction (25.2)")
median_normalise <- function(Y) {                                # eq. (25.2)
  cm <- apply(Y, 2, median, na.rm = TRUE)
  sweep(Y, 2, cm) + median(cm)
}
Ynorm <- median_normalise(Yobs)
cat(sprintf("  %-8s%15s%14s%8s%11s\n", "sample", "median before", "median after",
            "plex", "run order"))
for (s_ in colnames(Yobs)[1:6])
  cat(sprintf("  %-8s%15.3f%14.3f%8s%11d\n", s_,
              median(Yobs[, s_], na.rm = TRUE),
              median(Ynorm[, s_], na.rm = TRUE),
              smeta[s_, "plex"], smeta[s_, "run_order"]))
before <- cor(smeta$run_order, apply(Yobs, 2, median, na.rm = TRUE),
              method = "spearman")
cat(sprintf("\n  Spearman(run order, sample median) BEFORE: %+.3f\n", before))
cat("  Spearman(run order, sample median) AFTER : undefined\n")
cat("
  That second line is not a bug, and it is worth understanding. Median
  normalisation subtracts each column's median and adds one common constant,
  so after it every sample median is IDENTICAL by construction. A correlation
  against a constant has zero variance in one argument and is undefined.

  The trap is to read that as \"the drift is gone\". It is not gone; this
  particular diagnostic has simply been blinded to it. Median normalisation
  removes a GLOBAL shift per sample. Drift that pushes different features by
  different amounts survives untouched, and a per-feature check still sees
  it:\n")
#' Fraction of features whose intensity still tracks run order.
drift_fraction <- function(Y) {
  ro <- smeta$run_order
  hits <- 0
  for (i in seq_len(nrow(Y))) {
    v <- as.numeric(Y[i, ]); ok <- is.finite(v)
    if (sum(ok) < 8) next
    if (suppressWarnings(cor.test(ro[ok], v[ok],
                                  method = "spearman"))$p.value < 0.05)
      hits <- hits + 1
  }
  hits / nrow(Y)
}
cat(sprintf("    features correlated with run order, before: %.1f%%\n",
            100 * drift_fraction(Yobs)))
cat(sprintf("    features correlated with run order, after : %.1f%%\n",
            100 * drift_fraction(Ynorm)))
cat("
  Always plot intensity against INJECTION ORDER before and after, per feature
  and not just per sample. With pooled QC samples injected throughout the run,
  fit a LOESS curve per feature against run order and subtract it, which is
  the metabolomics standard.\n")

#' ## 4. Peptide -> protein summarisation, eq. (25.3)

#+ summarise
header("4. Median polish vs summing (25.3)")
median_polish <- function(mat, n_iter = 12) {
  ## Tukey's median polish: additive peptide (row) + sample (column) effects.
  m <- mat; overall <- 0
  row_eff <- numeric(nrow(m)); col_eff <- numeric(ncol(m))
  for (it in seq_len(n_iter)) {
    rmed <- apply(m, 1, median, na.rm = TRUE); rmed[!is.finite(rmed)] <- 0
    m <- m - rmed; row_eff <- row_eff + rmed
    d <- median(row_eff); row_eff <- row_eff - d; overall <- overall + d
    cmed <- apply(m, 2, median, na.rm = TRUE); cmed[!is.finite(cmed)] <- 0
    m <- sweep(m, 2, cmed); col_eff <- col_eff + cmed
    d <- median(col_eff); col_eff <- col_eff - d; overall <- overall + d
  }
  overall + col_eff
}
summarise_proteins <- function(Y, pep_meta, method = "median_polish") {
  prots <- unique(pep_meta$protein)
  out <- t(sapply(prots, function(pr) {
    ## Intersect: some peptides may have been dropped upstream.
    present <- intersect(pep_meta$peptide[pep_meta$protein == pr], rownames(Y))
    if (!length(present)) return(rep(NA_real_, ncol(Y)))
    sub <- Y[present,, drop = FALSE]
    if (all(is.na(sub))) return(rep(NA_real_, ncol(Y)))
    if (method == "median_polish") median_polish(sub)
    else { lin <- colSums(2^sub, na.rm = TRUE); log2(ifelse(lin > 0, lin, NA)) }
  }))
  colnames(out) <- colnames(Y); out
}
prot_mp <- summarise_proteins(Ynorm, pep_meta, "median_polish")
prot_sum <- summarise_proteins(Ynorm, pep_meta, "sum")
n_obs_pep <- tapply(rowSums(!is.na(Ynorm)), pep_meta$protein, sum)
cat("  Correlation between the summarised value and the NUMBER OF OBSERVED\n")
cat("  PEPTIDES (a pure artefact - it should be near zero):\n")
for (row_ in list(list("median polish (25.3)", prot_mp), list("sum of intensities", prot_sum))) {
  v <- rowMeans(row_[[2]], na.rm = TRUE)
  common <- intersect(names(v), names(n_obs_pep))
  m <- is.finite(v[common]) & is.finite(n_obs_pep[common])
  cat(sprintf("    %-24s Spearman = %+.3f\n", row_[[1]],
              cor(n_obs_pep[common][m], v[common][m], method = "spearman")))
}
cat("\n  Summing intensities confounds 'how much protein' with 'how many\n")
cat("  peptides happened to be identified'. Median polish estimates an additive\n")
cat("  peptide effect and is robust to individual peptides dropping out.\n")

#' ## 5. Imputation: one imputer for two mechanisms is the standard mistake

#+ imputation
header("5. MNAR vs MAR imputation (Topic 15, eq. 15.8)")
impute_mnar <- function(Y, shift = 1.8, width = 0.3) {
  ## A column with fewer than two observed values has no usable sd, and a
  ## subset of rows (as impute_mixed passes) can easily produce one. Falling
  ## back to the matrix-wide spread keeps every cell imputed; leaving it NA
  ## would silently drop the protein from the comparison below.
  g_sd <- sd(Y, na.rm = TRUE)
  g_mu <- mean(Y, na.rm = TRUE)
  if (!is.finite(g_sd) || g_sd == 0) g_sd <- 1
  if (!is.finite(g_mu)) g_mu <- 0
  apply(Y, 2, function(col) {
    s <- sd(col, na.rm = TRUE); if (!is.finite(s) || s == 0) s <- g_sd
    m <- mean(col, na.rm = TRUE); if (!is.finite(m)) m <- g_mu
    n_miss <- sum(is.na(col))
    if (n_miss) col[is.na(col)] <- rnorm(n_miss, m - shift*s, width*s)
    col
  })
}
impute_mar_knn <- function(Y, k = 6) {
  ## Simple kNN over PEPTIDES: borrow from the k most similar complete rows.
  Yc <- Y; rm_ <- rowMeans(Y, na.rm = TRUE)
  complete <- which(rowSums(is.na(Y)) == 0)
  if (!length(complete)) { Yc[is.na(Yc)] <- rep(rm_, ncol(Y))[is.na(Yc)]; return(Yc) }
  for (i in which(rowSums(is.na(Y)) > 0)) {
    obs <- !is.na(Y[i, ])
    if (!any(obs)) { Yc[i, ] <- rm_[i]; next }
    d <- colSums((t(Y[complete, obs, drop = FALSE]) - Y[i, obs])^2)
    nb <- complete[order(d)[1:min(k, length(complete))]]
    Yc[i, !obs] <- colMeans(Y[nb, !obs, drop = FALSE], na.rm = TRUE)
  }
  Yc[is.na(Yc)] <- mean(Y, na.rm = TRUE)
  Yc
}
impute_mixed <- function(Y, mnar_mask) {
  out <- Y
  if (any(mnar_mask)) out[mnar_mask, ] <- impute_mnar(Y[mnar_mask,, drop = FALSE])
  if (any(!mnar_mask)) out[!mnar_mask, ] <- impute_mar_knn(Y[!mnar_mask,, drop = FALSE])
  out
}
test_proteins <- function(P, label, min_obs = 6) {
  keep <- rowSums(!is.na(P)) >= min_obs
  P <- P[keep,, drop = FALSE]
  res <- t(sapply(rownames(P), function(pr) {
    d <- smeta; d$y <- P[pr, ]
    d <- d[!is.na(d$y), ]
    if (nlevels(droplevels(d$condition)) < 2 || nrow(d) < min_obs) return(c(NA, 1))
    f <- try(suppressWarnings(lm(y ~ condition + plex, data = d)), silent = TRUE)
    if (inherits(f, "try-error")) return(c(NA, 1))
    ## summary() warns on a perfect fit; the NaN p-value it produces is
    ## handled explicitly below, so the warning adds nothing.
    cs <- suppressWarnings(coef(summary(f)))
    key <- grep("^condition", rownames(cs), value = TRUE)[1]
    if (is.na(key) || !(key %in% rownames(cs))) return(c(NA, 1))
    est <- cs[key, 1]; pv <- cs[key, 4]
    ## A perfectly-fitting model (zero residual df after dropping NAs) yields
    ## NaN p-values. Treat those as uninformative rather than letting NA
    ## propagate silently into the counts.
    c(est, if (is.finite(pv)) pv else 1)
  }))
  q <- p.adjust(res[, 2], "BH")
  t0 <- truth[rownames(P), ]
  rej <- q < 0.05
  cat(sprintf("  %-40s%8d%7d%6d%6d%9.2f%8.3f\n", label, nrow(P), sum(rej),
              sum(rej & t0$is_de), sum(rej & !t0$is_de),
              sum(rej & t0$is_de)/sum(truth$is_de), sum(rej & !t0$is_de)/max(sum(rej),1)))
}
set.seed(3502)
pep_mnar <- mnar_like[rownames(Ynorm)]
pep_mnar[is.na(pep_mnar)] <- FALSE
cat(sprintf("  peptides routed to MNAR imputation: %d\n", sum(pep_mnar)))
cat(sprintf("  peptides routed to MAR imputation : %d\n",
            sum(rowSums(is.na(Ynorm)) > 0 & !pep_mnar)))
cat(sprintf("\n  %-40s%8s%7s%6s%6s%9s%8s\n", "strategy (applied to PEPTIDES)",
            "tested", "rej", "TP", "FP", "sens", "FDP"))
test_proteins(summarise_proteins(Ynorm, pep_meta), "no imputation (median polish handles NA)")
test_proteins(summarise_proteins(impute_mnar(Ynorm), pep_meta),
              "ALL peptides MNAR-imputed (down-shift)")
test_proteins(summarise_proteins(impute_mar_knn(Ynorm), pep_meta),
              "ALL peptides MAR-imputed (kNN)")
test_proteins(summarise_proteins(impute_mixed(Ynorm, pep_mnar), pep_meta),
              "MIXED: route each peptide by mechanism")
cat("\n  READ THIS TABLE CAREFULLY - it does not say what most tutorials say.\n")
cat("\n  1. NOT IMPUTING does well. Median polish estimates an additive peptide\n")
cat("     effect from whatever peptides ARE observed, and the linear model\n")
cat("     drops missing samples. Neither step needs a filled-in value, so\n")
cat("     nothing has to be invented. Modern guidance increasingly agrees:\n")
cat("     prefer methods that TOLERATE missingness over imputing.\n")
cat("  2. Down-shifting EVERYTHING gives every sporadically-missing peptide a\n")
cat("     value ~1.8 SD below the mean regardless of WHY it was missing.\n")
cat("  3. Check how many features each branch of a MIXED strategy received -\n")
cat("     when almost nothing is classified MNAR it reduces to the MAR branch.\n")
cat("  4. The 'tested' column differs between rows: imputation changes WHICH\n")
cat("     proteins are analysable, so the FDPs are not on equal footing.\n")

#' ## 6. Censoring, not missingness, eq. (15.8)

#+ censoring
header("6. Treating the LOD as censoring (15.8)")
tobit_group <- function(y_obs, censored, lod, group) {
  n_cen <- sum(censored)
  nll <- function(par) {
    mu <- par[1] + par[2]*group; s <- exp(par[3])
    ll <- 0
    if (any(!censored)) ll <- ll + sum(dnorm(y_obs[!censored], mu[!censored], s, log = TRUE))
    if (n_cen) ll <- ll + sum(pnorm((lod - mu[censored])/s, log.p = TRUE))
    -ll
  }
  st <- c(mean(y_obs[!censored]), 0, log(max(sd(y_obs[!censored]), 0.1)))
  optim(st, nll, method = "Nelder-Mead", control = list(maxit = 3000))$par[2]
}
set.seed(3503)
TRUE_DELTA <- 0.8; N <- 40; lod <- 21
res <- t(replicate(250, {
  g <- rep(0:1, each = N/2)
  z <- rnorm(N, 21.5 + TRUE_DELTA*g, 1)
  cen <- z < lod
  yh <- ifelse(cen, lod/2, z); yl <- ifelse(cen, lod, z)
  zd <- z; zd[cen] <- NA
  c(drop = mean(zd[g==1], na.rm=TRUE) - mean(zd[g==0], na.rm=TRUE),
    half = mean(yh[g==1]) - mean(yh[g==0]),
    lod = mean(yl[g==1]) - mean(yl[g==0]),
    tobit = tobit_group(ifelse(cen, lod, z), cen, lod, g))
}))
cat(sprintf("  TRUE group difference = %.1f; ~%.0f%% censored\n\n", TRUE_DELTA,
            100*mean(rnorm(1e4, 21.5, 1) < lod)))
cat(sprintf("  %-32s%15s%9s%9s\n", "strategy", "mean estimate", "bias", "SD"))
for (nm in c("drop","half","lod","tobit"))
  cat(sprintf("  %-32s%15.4f%9.4f%9.4f\n",
              c(drop="drop censored values", half="substitute LOD/2",
                lod="substitute LOD", tobit="censored MLE (15.8)")[nm],
              mean(res[, nm]), mean(res[, nm]) - TRUE_DELTA, sd(res[, nm])))
cat("\n  Every substitution rule is biased. The censored likelihood USES the\n")
cat("  information that the value was BELOW the limit, which is real data.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "proteomics.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
plot(mean_obs[ok], miss_rate[ok], pch = ".", col = "steelblue",
     xlab = "mean observed log2 intensity", ylab = "missingness rate",
     main = sprintf("MNAR signature (Spearman %+.2f)", sp))
hist(sim$Ytrue, breaks = 80, col = adjustcolor("steelblue", 0.5), border = "white",
     main = "Eq. (15.8): the left tail is censored", xlab = "log2 intensity")
hist(Yobs[!is.na(Yobs)], breaks = 80, col = adjustcolor("darkorange", 0.6),
     border = "white", add = TRUE)
abline(v = sim$lod, col = "red", lwd = 2)
plot(smeta$run_order, apply(Yobs, 2, median, na.rm = TRUE), pch = 16, cex = 1.2,
     col = c("steelblue","darkorange","forestgreen")[smeta$plex],
     xlab = "injection order", ylab = "sample median log2",
     main = "Run-order drift, coloured by plex")
boxplot(res, col = c("grey70","grey70","grey70","steelblue"), las = 2,
        main = "Eq. (15.8) beats every substitution rule",
        ylab = "estimated group difference", cex.axis = 0.7)
abline(h = TRUE_DELTA, col = "red", lwd = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "proteomics.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Ignoring the plex
#'
#' Re-run the protein-level test WITHOUT `plex` in the design. What happens to
#' power? Then build a CONFOUNDED design and check the design rank.

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# P <- summarise_proteins(impute_mixed(Ynorm, pep_mnar), pep_meta)
# for (form in c("y ~ condition", "y ~ condition + plex")) {
#   res_ <- sapply(rownames(P), function(pr) {
#     d <- smeta; d$y <- P[pr, ]; d <- d[!is.na(d$y), ]
#     f <- try(lm(as.formula(form), data = d), silent = TRUE)
#     if (inherits(f, "try-error")) return(1)
#     cs <- coef(summary(f)); cs[grep("^condition", rownames(cs))[1], 4]
#   })
#   q <- p.adjust(res_, "BH"); t0 <- truth[rownames(P), ]
#   cat(sprintf("  %-26s rej=%4d TP=%4d FDP=%.3f\n", form, sum(q < 0.05),
#               sum(q < 0.05 & t0$is_de),
#               sum(q < 0.05 & !t0$is_de)/max(sum(q < 0.05), 1)))
# }
# ## Blocking on plex removes the plex offsets from the residual - free power
# ## from a term the design already provides.
#
# smeta_bad <- smeta
# smeta_bad$plex <- factor(ifelse(smeta_bad$condition == "ctrl", "plex1", "plex2"))
# Xd <- model.matrix(~ condition + plex, data = smeta_bad)
# cat(sprintf("  confounded design: %d columns, rank %d -> condition and plex ALIASED\n",
#             ncol(Xd), qr(Xd)$rank))
# ## No software can separate them (eq. 19.3). Design plexes BALANCED, and
# ## include a common reference channel in every plex.

#' ### Problem 2: How much does the imputation shift matter?
#'
#' Vary the MNAR down-shift from 0 to 3 SDs and record hits and FDP. This is the
#' sensitivity analysis Topic 15 requires.

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %18s%11s%6s%6s%8s\n", "down-shift (SDs)", "rejected", "TP", "FP", "FDP"))
# for (sh in c(0, 0.8, 1.8, 3.0)) {
#   set.seed(2)
#   P <- summarise_proteins(impute_mnar(Ynorm, shift = sh), pep_meta)
#   res_ <- sapply(rownames(P), function(pr) {
#     d <- smeta; d$y <- P[pr, ]; d <- d[!is.na(d$y), ]
#     f <- try(lm(y ~ condition + plex, data = d), silent = TRUE)
#     if (inherits(f, "try-error")) return(1)
#     cs <- coef(summary(f)); cs[grep("^condition", rownames(cs))[1], 4]
#   })
#   q <- p.adjust(res_, "BH"); t0 <- truth[rownames(P), ]
#   cat(sprintf("  %18.1f%11d%6d%6d%8.3f\n", sh, sum(q < 0.05),
#               sum(q < 0.05 & t0$is_de), sum(q < 0.05 & !t0$is_de),
#               sum(q < 0.05 & !t0$is_de)/max(sum(q < 0.05), 1)))
# }
#
# ## The down-shift is an ASSUMPTION about how far below the limit the missing
# ## values sit, and it is UNTESTABLE from the observed data. If your conclusion
# ## changes across this range, say so - that IS the result (Module 24).

#' ### Problem 3: A protein seen in 3 of 18 samples
#'
#' Find proteins observed in fewer than 5 samples and inspect their results.
#' Should they be reported as findings?

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# n_obs <- rowSums(!is.na(prot_mp))
# set.seed(3)
# P <- summarise_proteins(impute_mnar(Ynorm), pep_meta)
# res_ <- sapply(rownames(P), function(pr) {
#   d <- smeta; d$y <- P[pr, ]; d <- d[!is.na(d$y), ]
#   f <- try(lm(y ~ condition + plex, data = d), silent = TRUE)
#   if (inherits(f, "try-error")) return(1)
#   cs <- coef(summary(f)); cs[grep("^condition", rownames(cs))[1], 4]
# })
# q <- p.adjust(res_, "BH")
# cat(sprintf("  %13s%10s%9s%16s\n", "observed in", "proteins", "called", "of which TRUE"))
# for (rg in list(c(0,4), c(5,9), c(10,14), c(15,18))) {
#   m <- n_obs[names(q)] >= rg[1] & n_obs[names(q)] <= rg[2]
#   called <- m & q < 0.05
#   cat(sprintf("  %13s%10d%9d%16d\n", sprintf("%d-%d samples", rg[1], rg[2]),
#               sum(m), sum(called), sum(called & truth[names(q), "is_de"])))
# }
#
# ## Proteins seen in a handful of samples generate "significant" calls whose
# ## effect is entirely determined by the IMPUTATION, not by measurement. Set a
# ## minimum-observation rule BEFORE testing (e.g. "observed in >= 70% of one
# ## condition") and report how many proteins it removed. A protein quantified
# ## in 3 of 18 samples is a QC finding, not a biological one.

#' ## What to take away
#'
#' 1. Log-transform, normalise, then summarise to protein level - in that
#'    order, inspecting after each step.
#' 2. **Ask whether you need to impute at all.** If you must, diagnose the
#'    mechanism first and route MNAR and MAR features differently.
#' 3. Model detection limits as CENSORING, not missingness.
#' 4. Model plex/batch/run-order explicitly; design them balanced.
#' 5. Report how many proteins were quantified in how many samples.
#'
#' **Next:** `36_microbiome_compositional.R`
