#' ---
#' title: "Applied 20 - Bulk RNA-seq differential expression"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 20, equations (20.1)-(20.7)
#' **Core modules used:** 04 (NB), 08 (FDR), 13 (GLM), 19 (batch), 33 (shrinkage)
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | `airway` (Himes et al. 2014, GSE52778): 4 donors x (untreated, dexamethasone) |
#' | **Design** | Paired: each donor contributes both conditions |
#' | **Here** | Simulated with the same structure and realistic NB parameters |
#'
#' Every number below is reproducible offline and we KNOW THE TRUTH, so we can
#' measure sensitivity and FDP rather than assert them. Section 8 shows how to
#' swap in the real `airway` counts.
#'
#' ## The biological question
#'
#' Which genes respond to dexamethasone, given that the four donors differ
#' enormously from each other?

#+ setup, message = FALSE
suppressPackageStartupMessages(library(MASS))
MODULE_NAME <- "20_bulk_rnaseq_differential_expression"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. Simulate a realistic paired RNA-seq experiment
#'
#' $$K_{gj}\sim\mathrm{NB}(\mu_{gj},\phi_g),\qquad
#'   \log\mu_{gj}=\log s_j+\mathbf x_j'\boldsymbol\beta_g$$

#+ simulate
header("1. Simulating airway-like paired RNA-seq counts (20.1)")
simulate_rnaseq <- function(n_donors = 4, G = 8000, n_de = 800, lfc = 1.2) {
  base_mu <- exp(rnorm(G, 3.2, 2.2))
  ## Mean-dispersion TREND: dispersion falls with expression, as edgeR models.
  phi <- 0.12 + 3.0/(base_mu + 5)
  true_lfc <- numeric(G)
  de_idx <- sample(G, n_de)
  true_lfc[de_idx] <- rnorm(n_de, 0, lfc)
  donor_eff <- matrix(rnorm(G*n_donors, 0, 0.55), G)   # SHARED by the pair
  size_factors <- exp(rnorm(2*n_donors, 0, 0.28))
  mu <- matrix(0, G, 2*n_donors)
  cols <- character(2*n_donors); cond <- character(2*n_donors); don <- character(2*n_donors)
  for (d in seq_len(n_donors)) for (t in 0:1) {
    j <- 2*(d-1) + t + 1
    mu[, j] <- exp(log(base_mu) + donor_eff[, d] + log(2)*true_lfc*t +
                     log(size_factors[j]))
    cols[j] <- sprintf("D%d_%s", d-1, c("untrt","trt")[t+1])
    cond[j] <- c("untrt","trt")[t+1]; don[j] <- sprintf("D%d", d-1)
  }
  counts <- matrix(rnbinom(length(mu), mu = as.vector(mu), size = 1/rep(phi, 2*n_donors)),
                   nrow = G)
  dimnames(counts) <- list(sprintf("ENSG%07d", seq_len(G)), cols)
  list(counts = counts,
       ## EXPLICIT factor levels. R orders levels alphabetically, so leaving
       ## `condition` as a character vector would make "trt" the reference
       ## ("trt" < "untrt") and name the coefficient `conditionuntrt` - with
       ## the sign flipped. Always set the reference level deliberately.
       meta = data.frame(donor = factor(don),
                         condition = factor(cond, levels = c("untrt", "trt")),
                         row.names = cols),
       truth = data.frame(true_lfc = true_lfc, is_de = true_lfc != 0,
                          base_mu = base_mu, phi = phi,
                          row.names = rownames(counts)))
}
set.seed(3001)
sim <- simulate_rnaseq()
counts <- sim$counts; meta <- sim$meta; truth <- sim$truth
print(meta)
cat(sprintf("\n  matrix: %d genes x %d libraries\n", nrow(counts), ncol(counts)))
cat(sprintf("  library sizes (millions): %s\n",
            paste(round(colSums(counts)/1e6, 2), collapse = " ")))
cat(sprintf("  truly DE genes: %d\n", sum(truth$is_de)))
stopifnot(identical(colnames(counts), rownames(meta)))     # eq. (2.1)
cat("  eq. (2.1) alignment contract: OK\n")

#' ## 2. Filtering -- outcome-independent, before multiplicity

#+ filtering
header("2. Independent filtering (edgeR filterByExpr logic)")
filter_by_expr <- function(counts, group, min_count = 10, min_total = 15) {
  lib <- colSums(counts); median_lib <- median(lib)
  cpm <- t(t(counts)/lib) * 1e6
  cpm_cutoff <- min_count / (median_lib/1e6)
  min_n <- min(table(group))
  rowSums(cpm >= cpm_cutoff) >= min_n & rowSums(counts) >= min_total
}
keep <- filter_by_expr(counts, meta$condition)
cat(sprintf("  kept %d of %d genes (%.1f%%)\n", sum(keep), length(keep), 100*mean(keep)))
cat(sprintf("  of the truly DE genes, kept %d of %d (%.1f%%)\n",
            sum(keep & truth$is_de), sum(truth$is_de),
            100*sum(keep & truth$is_de)/sum(truth$is_de)))
cat(sprintf("  median expression of DROPPED genes: %.2f counts\n",
            median(truth$base_mu[!keep])))
cat("\n  The filter uses ONLY expression level, never the condition labels.\n")
cat("  Filtering on a group-wise fold change would invalidate every\n")
cat("  downstream p-value (Module 08, section 6).\n")
cts <- counts[keep, ]; tr <- truth[keep, ]

#' ## 3. Normalisation: median-of-ratios size factors, eq. (20.2)

#+ normalisation
header("3. Size factors (20.2) and why composition bias matters")
median_of_ratios <- function(counts) {                          # eq. (20.2)
  logc <- log(counts); logc[!is.finite(logc)] <- NA
  log_geo <- rowMeans(logc)
  usable <- is.finite(log_geo)
  exp(apply(logc[usable,, drop = FALSE] - log_geo[usable], 2, median, na.rm = TRUE))
}
sf <- median_of_ratios(cts)
total_scaling <- colSums(cts)/mean(colSums(cts))
cat(sprintf("  %-14s%20s%19s\n", "library", "total-count factor", "median-of-ratios"))
for (cc in colnames(cts))
  cat(sprintf("  %-14s%20.4f%19.4f\n", cc, total_scaling[cc], sf[cc]))
## Composition bias: force ONE gene to ~30% of each TREATED library.
cts_biased <- cts
trt_cols <- rownames(meta)[meta$condition == "trt"]
cts_biased[1, trt_cols] <- round(colSums(cts_biased[, trt_cols])*0.43)
cat(sprintf("\n  After forcing one gene to ~30%% of each TREATED library:\n"))
cat(sprintf("    total-count factors shift by      %.4f on average\n",
            mean(abs(colSums(cts_biased)/mean(colSums(cts_biased)) - total_scaling))))
cat(sprintf("    median-of-ratios factors shift by %.4f on average\n",
            mean(abs(median_of_ratios(cts_biased) - sf))))
cat("  The robust estimator barely moves. With total-count scaling every OTHER\n")
cat("  gene would appear down-regulated in the treated samples - a pure\n")
cat("  artefact (composition bias).\n")

#' ## 4. Dispersion estimation with empirical-Bayes shrinkage, eq. (20.3)

#+ dispersion
header("4. Mean-dispersion trend and shrinkage (20.3)")
estimate_dispersions <- function(counts, size_factors, design) {
  ## CRITICAL: estimate dispersion from the RESIDUAL variability AFTER removing
  ## the design, not from the raw variance across libraries. Raw variance also
  ## contains the donor and treatment effects - real biology - and using it
  ## inflates phi and destroys power.
  ##
  ## Delta method (eq. 2.3 / 9.8): Var(log2 y) ~ (1/mu + phi)/(ln 2)^2, so
  ## phi ~ s^2 (ln 2)^2 - 1/mu. This is the relationship voom exploits.
  norm <- t(t(counts)/size_factors)
  mean_norm <- rowMeans(norm)
  logy <- log2(norm + 0.5)
  n <- ncol(counts); p <- ncol(design)
  beta <- qr.coef(qr(design), t(logy))
  resid <- t(logy) - design %*% beta
  s2_log <- colSums(resid^2)/(n - p)
  phi_gw <- pmax(s2_log*log(2)^2 - 1/pmax(mean_norm, 1e-8), 1e-4)
  ok <- mean_norm > 1 & is.finite(phi_gw)
  cf <- coef(lm(phi_gw[ok] ~ I(1/mean_norm[ok])))                # phi = a/mu + b
  phi_trend <- pmax(cf[2]/mean_norm + cf[1], 1e-4)
  ## Empirical-Bayes shrinkage toward the trend. TWO details decide whether
  ## this works: the sampling variance of log(s^2) with d df is trigamma(d/2),
  ## NOT 2/d; and the prior variance must be estimated ROBUSTLY (MAD), because
  ## a handful of genes at the dispersion floor would otherwise dominate.
  log_resid <- log(phi_gw) - log(phi_trend)
  obs_var <- trigamma(max(n - p, 1)/2)
  prior_var <- max(mad(log_resid[ok])^2 - obs_var, 1e-4)
  w <- prior_var/(prior_var + obs_var)
  list(tab = data.frame(mean = mean_norm, phi_gw = phi_gw, phi_trend = phi_trend,
                        phi_map = exp(log(phi_trend) + w*log_resid)), w = w)
}
design <- model.matrix(~ donor + condition, data = meta)
dsp <- estimate_dispersions(cts, sf, design)
disp <- dsp$tab
cat(sprintf("  shrinkage weight toward the gene-wise estimate: %.3f\n", dsp$w))
cat(sprintf("  With only n-p = %d residual df the gene-wise estimate is almost\n",
            ncol(cts) - ncol(design)))
cat("  pure noise, so the weight is small and most genes are pulled onto the\n")
cat("  trend. That is CORRECT behaviour, not a failure.\n")
bins <- cut(disp$mean, quantile(disp$mean, 0:5/5), include.lowest = TRUE,
            labels = c("Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"))
cat(sprintf("\n  %-20s%14s%16s%14s%11s\n", "expression bin", "mean phi_gw",
            "mean phi_trend", "mean phi_MAP", "true phi"))
for (b in levels(bins)) {
  m <- bins == b
  cat(sprintf("  %-20s%14.4f%16.4f%14.4f%11.4f\n", b, mean(disp$phi_gw[m]),
              mean(disp$phi_trend[m]), mean(disp$phi_map[m]), mean(tr$phi[m])))
}
cat("\n  RMSE of the dispersion estimate against the truth (log scale):\n")
for (nm in c("phi_gw", "phi_trend", "phi_map"))
  cat(sprintf("    %-24s%.4f\n",
              c(phi_gw = "gene-wise (raw)", phi_trend = "trend only",
                phi_map = "shrunken MAP (20.3)")[nm],
              sqrt(mean((log(disp[[nm]]) - log(tr$phi))^2))))
cat("\n  READ THIS HONESTLY. The gene-wise estimate is hopeless (RMSE ~1.9)\n")
cat("  with only 3 residual df, and the fitted TREND is already excellent, so\n")
cat("  shrinking part-way back toward the noisy gene-wise value costs a little\n")
cat("  accuracy here. That is the correct conclusion from these numbers.\n")
cat("\n  Shrinkage earns its keep as residual df GROW - the gene-wise estimate\n")
cat("  becomes informative and the trend becomes too rigid. It also matters\n")
cat("  for the TEST rather than the point estimate: a gene whose dispersion is\n")
cat("  underestimated by chance produces a spuriously large statistic, and\n")
cat("  pulling it toward the trend is what stops that (section 5). Report the\n")
cat("  weight, and check the dispersion plot rather than assuming.\n")

#' ## 5. Fitting the NB GLM and testing the contrast, eq. (20.1)

#+ glm-fit
header("5. NB GLM with the paired design (20.1)")
nb_glm_de <- function(counts, size_factors, design, coef_name, dispersions) {
  offset <- log(size_factors)
  j <- which(colnames(design) == coef_name)
  if (length(j) != 1L)
    stop("coefficient '", coef_name, "' not in the design. Columns are: ",
         paste(colnames(design), collapse = ", "),
         "\n  (R names dummies from the FACTOR LEVELS - check your reference level.)")
  res <- t(vapply(seq_len(nrow(counts)), function(i) {
    fit <- try(glm(counts[i, ] ~ design - 1, family = negative.binomial(1/dispersions[i]),
                   offset = offset), silent = TRUE)
    if (inherits(fit, "try-error")) return(c(NA, NA, NA))
    ## dispersion = 1 is ESSENTIAL. MASS's negative.binomial() family is not on
    ## summary.glm's "known dispersion" list, so by default summary() estimates
    ## an EXTRA scale parameter from the Pearson residuals on 3 df and switches
    ## to a t reference. That double-counts the overdispersion already carried
    ## by theta and wipes out the power. Fixing dispersion = 1 gives the Wald
    ## z-test that DESeq2/edgeR use.
    cs <- coef(summary(fit, dispersion = 1))
    c(cs[j, 1], cs[j, 2], cs[j, 4])
  }, numeric(3)))
  out <- data.frame(log_fc = res[, 1], se = res[, 2], pvalue = res[, 3],
                    row.names = rownames(counts))
  out$lfc <- out$log_fc/log(2)                                  # eq. (5.11)
  out$base_mean <- rowMeans(t(t(counts)/size_factors))
  out$padj <- p.adjust(out$pvalue, "BH")
  out
}
res_paired <- nb_glm_de(cts, sf, design, "conditiontrt", disp$phi_map)
## The unpaired model must estimate its OWN dispersion - handing it the paired
## one would secretly give it the benefit of the blocking it is missing.
design_unp <- model.matrix(~ condition, data = meta)
dsp_unp <- estimate_dispersions(cts, sf, design_unp)
res_unpaired <- nb_glm_de(cts, sf, design_unp, "conditiontrt", dsp_unp$tab$phi_map)
cat(sprintf("  median fitted dispersion, PAIRED design   : %.4f\n", median(disp$phi_map)))
cat(sprintf("  median fitted dispersion, UNPAIRED design : %.4f\n",
            median(dsp_unp$tab$phi_map)))
cat("  Dropping the donor term pushes the donor-to-donor variance into the\n")
cat("  dispersion, which is what costs the power.\n\n")
score <- function(res, tr, label) {
  rej <- !is.na(res$padj) & res$padj < 0.05
  tp <- sum(rej & tr$is_de); fp <- sum(rej & !tr$is_de)
  r <- suppressWarnings(cor(res$lfc[tr$is_de], tr$true_lfc[tr$is_de], use = "complete.obs"))
  cat(sprintf("  %-38s%7d%7d%7d%9.2f%8.3f%9.3f\n", label, sum(rej), tp, fp,
              tp/sum(tr$is_de), fp/max(sum(rej), 1), r))
}
cat(sprintf("  %-38s%7s%7s%7s%9s%8s%9s\n", "design", "rej", "TP", "FP", "sens",
            "FDP", "r(LFC)"))
score(res_paired, tr, "~ donor + condition  (PAIRED)")
score(res_unpaired, tr, "~ condition  (pairing ignored)")
cat(sprintf("\n  The paired design finds %.2fx as many genes at the same FDR.\n",
            sum(res_paired$padj < 0.05, na.rm = TRUE) /
              max(sum(res_unpaired$padj < 0.05, na.rm = TRUE), 1)))
cat("  The DESIGN already removed the donor variance; the MODEL must be told\n")
cat("  about it to collect the benefit (eq. 4.3).\n")

#' ## 6. LFC shrinkage, eq. (20.7), and how to rank a gene list

#+ shrinkage
header("6. Shrink the fold change for RANKING, not for testing (20.7)")
shrink_lfc <- function(res) {
  s2 <- (res$se/log(2))^2
  prior_var <- max(var(res$lfc, na.rm = TRUE) - mean(s2, na.rm = TRUE), 1e-4)
  list(lfc = res$lfc * prior_var/(prior_var + s2), prior_sd = sqrt(prior_var))
}
sh <- shrink_lfc(res_paired)
res_paired$lfc_shrunk <- sh$lfc
cat(sprintf("  estimated prior SD of true LFCs: %.3f\n", sh$prior_sd))
de <- res_paired[!is.na(res_paired$padj) & res_paired$padj < 0.05, ]
de$true_lfc <- tr[rownames(de), "true_lfc"]
cat("\n  Accuracy of the LFC estimate among significant genes:\n")
for (nm in c("lfc", "lfc_shrunk"))
  cat(sprintf("    RMSE vs truth, %-22s%.4f\n",
              c(lfc = "raw MLE LFC", lfc_shrunk = "shrunken LFC (20.7)")[nm],
              sqrt(mean((de[[nm]] - de$true_lfc)^2))))
raw_top <- de[order(-abs(de$lfc)), ]
shr_top <- de[order(-abs(de$lfc_shrunk)), ]
cat(sprintf("\n  median base mean of the raw-LFC top 20      : %.1f\n",
            median(head(raw_top$base_mean, 20))))
cat(sprintf("  median base mean of the shrunken-LFC top 20 : %.1f\n",
            median(head(shr_top$base_mean, 20))))
cat("\n  Ranking by RAW LFC puts low-count, high-variance genes at the top -\n")
cat("  they have the noisiest estimates, so they win an |LFC| contest by\n")
cat("  accident. Rank by the SHRUNKEN value; TEST with the unshrunken one.\n")

#' ## 7. Diagnostics

#+ diagnostics
header("7. The diagnostic battery")
p_hist <- hist(res_paired$pvalue, breaks = seq(0, 1, length.out = 21), plot = FALSE)$counts
cat(sprintf("  p-value histogram (20 bins, expect flat + a spike at 0):\n    %s\n",
            paste(p_hist, collapse = " ")))
cat(sprintf("    first bin / last bin ratio = %.2f   (>> 1 means real signal)\n",
            p_hist[1]/max(p_hist[20], 1)))
pi0 <- min(1, mean(res_paired$pvalue > 0.5, na.rm = TRUE)/0.5)
cat(sprintf("\n  Storey pi0 estimate (8.9)  : %.3f\n", pi0))
cat(sprintf("  true null proportion        : %.3f\n", 1 - mean(tr$is_de)))
lib <- colSums(cts)
cat(sprintf("\n  library sizes span %.1f-%.1f M reads (%.2fx)\n",
            min(lib)/1e6, max(lib)/1e6, max(lib)/min(lib)))
logcpm <- log2(t(t(cts)/lib)*1e6 + 1)
cm <- cor(logcpm, method = "spearman"); diag(cm) <- NA
cat(sprintf("  sample-sample Spearman: min %.3f, median %.3f\n",
            min(cm, na.rm = TRUE), median(cm, na.rm = TRUE)))
pca <- prcomp(t(logcpm), center = TRUE)
pve <- pca$sdev^2/sum(pca$sdev^2)
cat("\n  PC x metadata R^2 (eq. 17.6):\n")
cat(sprintf("  %4s%8s%9s%12s\n", "PC", "PVE", "donor", "condition"))
for (k in 1:3)
  cat(sprintf("  %4d%7.1f%%%9.3f%12.3f\n", k, 100*pve[k],
              summary(lm(pca$x[, k] ~ factor(meta$donor)))$r.squared,
              summary(lm(pca$x[, k] ~ factor(meta$condition)))$r.squared))
cat("  DONOR dominates PC1 - which is exactly why the paired design and the\n")
cat("  `~ donor + condition` model matter so much here.\n")

#' ## 8. Using the REAL airway dataset (optional, needs Bioconductor)

#+ real-data
header("8. Swapping in real data")
cat('
  # The airway RangedSummarizedExperiment is distributed via Bioconductor:
  #   if (!requireNamespace("BiocManager")) install.packages("BiocManager")
  #   BiocManager::install("airway")
  #   library(airway); data(airway)
  #   counts <- assay(airway)
  #   meta   <- as.data.frame(colData(airway))
  #   design <- model.matrix(~ cell + dex, data = meta)   # cell = donor
  #
  # Everything above then runs unchanged. In production you would use DESeq2
  # or edgeR rather than these from-scratch versions:
  #   dds <- DESeqDataSetFromMatrix(counts, meta, ~ cell + dex)
  #   dds <- DESeq(dds); res <- lfcShrink(dds, coef = "dex_trt_vs_untrt")
  #
  # NOTE the trap: the GEO supplementary file for GSE52778 contains FPKM, NOT
  # counts. Feeding FPKM into an NB model violates eq. (20.1) - the count
  # information that determines the variance has already been divided out.
  # Use the `airway` package or recount3, both of which ship RAW COUNTS.
')

#' ## 9. Figure

#+ figure
png(file.path(OUT, "rnaseq.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
plot(disp$mean, disp$phi_gw, log = "xy", pch = ".", col = adjustcolor("grey40", 0.4),
     xlab = "mean of normalised counts", ylab = "dispersion phi",
     main = "Eq. (20.3): dispersion shrinkage")
o <- order(disp$mean)
lines(disp$mean[o], disp$phi_trend[o], col = "red", lwd = 2)
points(disp$mean, disp$phi_map, pch = ".", col = adjustcolor("forestgreen", 0.4))
legend("topright", c("gene-wise", "fitted trend", "shrunken MAP"),
       col = c("grey40", "red", "forestgreen"), pch = c(16, NA, 16),
       lty = c(NA, 1, NA), bty = "n", cex = 0.6)
sig <- !is.na(res_paired$padj) & res_paired$padj < 0.05
plot(res_paired$base_mean, res_paired$lfc, log = "x", pch = ".",
     col = ifelse(sig, "firebrick", adjustcolor("grey50", 0.4)),
     xlab = "base mean", ylab = "log2 fold change",
     main = sprintf("MA plot (%d genes at FDR 5%%)", sum(sig)))
abline(h = 0)
hist(res_paired$pvalue, breaks = 40, col = "steelblue", border = "white",
     main = "Eq. (6.2): healthy histogram", xlab = "p-value")
abline(h = nrow(res_paired)*pi0/40, col = "red", lty = 2)
plot(de$true_lfc, de$lfc, pch = 16, cex = 0.3, col = adjustcolor("steelblue", 0.4),
     xlab = "true log2 FC", ylab = "estimated log2 FC",
     main = "Eq. (20.7): shrinkage improves accuracy")
points(de$true_lfc, de$lfc_shrunk, pch = 16, cex = 0.3,
       col = adjustcolor("darkorange", 0.4))
abline(0, 1, lty = 2)
legend("topleft", c("raw MLE", "shrunken"), col = c("steelblue", "darkorange"),
       pch = 16, bty = "n", cex = 0.65)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "rnaseq.png"), "\n")

#' # PROBLEMS
#'
#' Work through these before reading the solutions. Uncomment the solution
#' block only after attempting it.
#'
#' ### Problem 1: TPM into a count model
#'
#' A collaborator sends TPM instead of counts and asks you to "just run DESeq2
#' on these". Convert the counts to CPM, round to integers, and run the same NB
#' analysis. **Predict before running:** better, worse, or unchanged?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------
# cpm_rounded <- ...
# res_cpm <- nb_glm_de(...)
# score(res_cpm, tr, "CPM-as-counts")

## ---- SOLUTION (uncomment to check) -------------------------------------
# cpm_rounded <- round(t(t(cts)/colSums(cts)) * 1e6)
# ## Size factors are ~1 by construction: CPM already divided them out.
# sf_cpm <- setNames(rep(1, ncol(cts)), colnames(cts))
# res_cpm <- nb_glm_de(cpm_rounded, sf_cpm, design, "conditiontrt", disp$phi_map)
# cat(sprintf("  %-38s%7s%7s%7s%9s%8s%9s\n", "design", "rej","TP","FP","sens","FDP","r(LFC)"))
# score(res_paired, tr, "raw counts + size factors")
# score(res_cpm,    tr, "CPM rounded to integers")
#
# ## WHY it degrades: CPM rescaling makes every library appear to have the same
# ## depth, so a gene with 5 counts in a shallow library and one with 50 counts
# ## in a deep library are treated as equally precise. The NB variance function
# ## (eq. 4.9) is then wrong for both. Normalisation belongs in the OFFSET
# ## (eq. 20.1), NEVER in the response.

#' ### Problem 2: How many donors do you actually need?
#'
#' Using eq. (9.9), compute the donors per group needed for 80% power to detect
#' a 2-fold change at mu = 50 and the MEDIAN fitted dispersion. Verify by
#' simulation. Then re-run with mu = 5000 and explain what changes.

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# phi_med <- median(disp$phi_map)
# lam <- log(2)                       # a 2-fold change on the NATURAL log
# z <- qnorm(0.975) + qnorm(0.80)
# n_needed <- 2 * z^2 * (1/50 + phi_med) / lam^2
# cat(sprintf("  median fitted dispersion = %.4f (BCV %.2f)\n", phi_med, sqrt(phi_med)))
# cat(sprintf("  eq. (9.9) requires n = %.1f per group for 80%% power\n", n_needed))
# n_int <- ceiling(n_needed)
# set.seed(99)
# pw <- mean(replicate(400, {
#   a <- rnbinom(n_int, mu = 100, size = 1/phi_med)
#   b <- rnbinom(n_int, mu = 50,  size = 1/phi_med)
#   d <- data.frame(y = c(a, b), g = rep(0:1, each = n_int))
#   f <- try(glm.nb(y ~ g, data = d), silent = TRUE)
#   if (inherits(f, "try-error")) NA else coef(summary(f))["g", 4] < 0.05
# }), na.rm = TRUE)
# cat(sprintf("  simulated power at n=%d: %.2f  (target 0.80)\n", n_int, pw))
# cat(sprintf("  same calculation at mu=5000: n = %.1f\n",
#             2*z^2*(1/5000 + phi_med)/lam^2))
#
# ## n barely changes: eq. (9.9) contains 1/mu + phi, and phi does not depend on
# ## depth. 100x more sequencing buys almost nothing; more donors is the only
# ## lever once mu >> 1/phi.

#' ### Problem 3: An `lfcThreshold` test
#'
#' Your biologists say a change below 1.5-fold is not worth following up.
#' Compare (a) testing H0: beta = 0 then filtering the list to |LFC| > log2(1.5),
#' with (b) testing H0: |beta| <= log2(1.5) directly. Which has error control
#' for the claim "this gene changes by more than 1.5-fold"?

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# tau <- log2(1.5)
# big_truth <- abs(tr$true_lfc) > tau            # the claim we want to make
# a_rej <- !is.na(res_paired$padj) & res_paired$padj < 0.05 & abs(res_paired$lfc) > tau
# ## (b) threshold test: a two-sided test against the MARGIN, not against zero.
# se_l2 <- res_paired$se/log(2)
# p_thr <- pmin(1, 2*pmin(pnorm((res_paired$lfc - tau)/se_l2, lower.tail = FALSE),
#                         pnorm((-res_paired$lfc - tau)/se_l2, lower.tail = FALSE)))
# b_rej <- p.adjust(p_thr, "BH") < 0.05
# for (z in list(list("(a) test vs 0, then filter |LFC|", a_rej),
#                list("(b) lfcThreshold test", b_rej))) {
#   r <- z[[2]]; r[is.na(r)] <- FALSE
#   cat(sprintf("  %-36s rejected=%5d  correct=%5d  wrong=%5d  FDP=%.3f\n",
#               z[[1]], sum(r), sum(r & big_truth), sum(r & !big_truth),
#               sum(r & !big_truth)/max(sum(r), 1)))
# }
#
# ## Strategy (a) has NO error control for the THRESHOLD claim: the p-value it
# ## adjusted was for beta = 0, and the |LFC| filter is applied afterwards using
# ## the SAME noisy estimate. Genes whose true LFC sits just below tau get
# ## selected whenever their estimate happens to be high. Strategy (b) builds
# ## the margin INTO the null, so the FDR statement covers the claim you make.
# ## In practice: DESeq2's `lfcThreshold=` or limma's `treat()`.

#' ### Problem 4: Diagnose a broken analysis
#'
#' Run the paired analysis on condition labels permuted WITHIN donor (which
#' preserves the pairing). What must the histogram and discovery count look
#' like? Then permute ACROSS donors and explain the difference.

#+ problem4
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(4)
# meta_w <- meta
# for (d in unique(meta$donor)) {
#   idx <- which(meta$donor == d)
#   if (runif(1) < 0.5) meta_w$condition[idx] <- rev(meta_w$condition[idx])
# }
# des_w <- model.matrix(~ donor + condition, data = meta_w)
# r_w <- nb_glm_de(cts, sf, des_w, "conditiontrt", disp$phi_map)
# cat(sprintf("  within-donor permutation: %d discoveries, P(p<0.05) = %.3f\n",
#             sum(r_w$padj < 0.05, na.rm = TRUE), mean(r_w$pvalue < 0.05, na.rm = TRUE)))
#
# ## This is the VALID null: the design's exchangeable unit is the label WITHIN
# ## a donor pair, so this permutation preserves the dependence structure.
# ## Expect ~0 discoveries and a flat histogram (Module 34, section 4).
#
# meta_a <- meta; meta_a$condition <- sample(meta_a$condition)
# des_a <- model.matrix(~ donor + condition, data = meta_a)
# cat(sprintf("  across-donor permutation: design rank %d of %d columns\n",
#             qr(des_a)$rank, ncol(des_a)))
#
# ## Across-donor permutation can put BOTH libraries of a donor in the same
# ## condition, which ALIASES donor with condition (eq. 19.3): the design loses
# ## rank and the contrast becomes unestimable. The permutation scheme must
# ## respect the design (Module 07, section 5).

#' ## What to take away
#'
#' 1. Raw integer counts into the count model; normalisation goes in the offset.
#' 2. Filter by an outcome-independent rule, BEFORE FDR.
#' 3. Put the blocking/pairing structure in the design - it is free power.
#' 4. Shrink dispersions and fold changes; rank by the shrunken LFC, TEST with
#'    the unshrunken statistic.
#' 5. Test the threshold you care about; do not filter a zero-null result list.
#' 6. Run the permuted-label negative control before believing anything.
#'
#' **Next:** `21_single_cell_pseudobulk.R`
