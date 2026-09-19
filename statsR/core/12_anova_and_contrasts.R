#' ---
#' title: "Module 12 - ANOVA, factorial designs, and contrasts"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 12, equations (12.1)-(12.7)
#'
#' ## What you will learn
#'
#' 1. The SS decomposition (12.1) and F = t^2 for two groups (12.2).
#' 2. **Contrasts (12.3) are where the science lives** -- with `emmeans`.
#' 3. Coding schemes change parameter MEANING, not fit.
#' 4. Interaction as a difference of differences (12.4)-(12.5).
#' 5. Type I/II/III sums of squares, and why the argument is a distraction.
#' 6. Effect sizes (12.6) and post-hoc multiplicity (12.7).

#+ setup, message = FALSE
suppressPackageStartupMessages(library(emmeans))
MODULE_NAME <- "12_anova_and_contrasts"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. The decomposition and the omnibus F, eq. (12.1)-(12.2)

#+ decomposition
header("1. SS decomposition and F (12.1)-(12.2)")
set.seed(1201)
true_means <- c(10, 10.6, 11.6, 13.0)     # a dose-response, roughly linear
n_per <- 8
d <- data.frame(y = unlist(lapply(true_means, function(m) rnorm(n_per, m, 1.2))),
                dose = factor(rep(paste0("D", 0:3), each = n_per)),
                dose_num = rep(0:3, each = n_per))
grand <- mean(d$y); gm <- tapply(d$y, d$dose, mean); cnt <- table(d$dose)
ss_total <- sum((d$y - grand)^2)
ss_between <- sum(cnt * (gm - grand)^2)
ss_within <- sum((d$y - gm[d$dose])^2)
k <- nlevels(d$dose); N <- nrow(d)
Fstat <- (ss_between/(k-1)) / (ss_within/(N-k))
cat(sprintf("  SS_total   = %9.4f\n  SS_between = %9.4f\n  SS_within  = %9.4f\n",
            ss_total, ss_between, ss_within))
cat(sprintf("  sum        = %9.4f   <- eq. (12.1) holds exactly\n",
            ss_between + ss_within))
cat(sprintf("\n  F(%d, %d) = %.4f, p = %.3e\n", k-1, N-k, Fstat,
            pf(Fstat, k-1, N-k, lower.tail = FALSE)))
cat(sprintf("  anova(lm): F = %.4f\n", anova(lm(y ~ dose, d))[["F value"]][1]))
two <- subset(d, dose %in% c("D0", "D3"))
tt <- t.test(y ~ dose, data = two, var.equal = TRUE)
cat(sprintf("\n  k=2 check: t^2 = %.6f   F = %.6f   <- eq. (B.6)\n",
            tt$statistic^2, anova(lm(y ~ dose, two))[["F value"]][1]))

#' ## 2. Contrasts: where the science lives, eq. (12.3)

#+ contrasts
header("2. Building and testing contrasts (12.3)")
test_contrast <- function(d, weights, label = "") {
  levs <- levels(d$dose)
  means <- tapply(d$y, d$dose, mean)[levs]
  ns <- table(d$dose)[levs]
  ms_w <- sum((d$y - means[d$dose])^2) / (nrow(d) - length(levs))
  cc <- as.numeric(weights)
  stopifnot(abs(sum(cc)) < 1e-10)                 # weights MUST sum to zero
  est <- sum(cc * means)
  se <- sqrt(ms_w * sum(cc^2 / ns))               # eq. (12.3)
  df <- nrow(d) - length(levs)
  t <- est / se
  list(label = label, estimate = est, se = se, t = t, df = df,
       p = 2*pt(abs(t), df, lower.tail = FALSE),
       ci = est + c(-1, 1) * qt(0.975, df) * se)
}
contrasts_list <- list(
  "any treated vs control" = c(-1, 1/3, 1/3, 1/3),
  "highest vs lowest"      = c(-1, 0, 0, 1),
  "linear trend"           = c(-3, -1, 1, 3),
  "quadratic trend"        = c(1, -1, -1, 1),
  "cubic trend"            = c(-1, 3, -3, 1))
cat(sprintf("  %-26s%10s%8s%8s%11s%22s\n", "contrast", "estimate", "SE", "t", "p", "95% CI"))
results <- lapply(names(contrasts_list), function(nm) {
  r <- test_contrast(d, contrasts_list[[nm]], nm)
  cat(sprintf("  %-26s%10.4f%8.4f%8.3f%11.3e   [%+.3f, %+.3f]\n",
              nm, r$estimate, r$se, r$t, r$p, r$ci[1], r$ci[2]))
  r })
cat("\n  The LINEAR TREND contrast is by far the most significant - it is the\n")
cat("  question the design was built to answer.\n")

cat("\n  The same thing with emmeans (what you would use in practice):\n")
em <- emmeans(lm(y ~ dose, d), ~ dose)
print(contrast(em, list(linear = c(-3, -1, 1, 3),
                        `treated vs control` = c(-1, 1/3, 1/3, 1/3))))

cat("\n  Orthogonality of the trend contrasts (balanced design):\n")
trend <- contrasts_list[grepl("trend", names(contrasts_list))]
for (i in 1:(length(trend)-1)) for (j in (i+1):length(trend))
  cat(sprintf("    <%-16s, %-16s> = %+.1f\n", names(trend)[i], names(trend)[j],
              sum(trend[[i]] * trend[[j]])))
cat("  All zero -> they partition SS_between into independent pieces.\n")

#' ## 3. Coding schemes change meaning, not fit

#+ coding
header("3. Treatment vs sum-to-zero coding")
fit_treat <- lm(y ~ dose, data = d)                             # R default
d2 <- d; contrasts(d2$dose) <- contr.sum(4)
fit_sum <- lm(y ~ dose, data = d2)
cat(sprintf("  Treatment coding: intercept = %.4f  (= mean of reference D0 = %.4f)\n",
            coef(fit_treat)[1], gm["D0"]))
cat(sprintf("  Sum-to-zero     : intercept = %.4f  (= unweighted grand mean = %.4f)\n",
            coef(fit_sum)[1], mean(gm)))
cat(sprintf("\n  RSS identical?  %s  (%.6f vs %.6f)\n",
            isTRUE(all.equal(sum(residuals(fit_treat)^2), sum(residuals(fit_sum)^2))),
            sum(residuals(fit_treat)^2), sum(residuals(fit_sum)^2)))
cat(sprintf("  Fitted values identical? %s\n",
            isTRUE(all.equal(fitted(fit_treat), fitted(fit_sum)))))
cat("\n  Same model, different parameterisation. Reporting 'the effect of\n")
cat("  treatment' without stating the coding is ambiguous.\n")

#' ## 4. Factorial designs and interaction, eq. (12.4)-(12.5)

#+ factorial
header("4. Interaction is a difference of differences (12.4)-(12.5)")
set.seed(1202)
cells <- list(c("WT","veh",10), c("WT","drug",12), c("KO","veh",10.5), c("KO","drug",10.7))
fac <- do.call(rbind, lapply(cells, function(z)
  data.frame(y = rnorm(10, as.numeric(z[3])), genotype = z[1], treatment = z[2])))
fac$genotype <- factor(fac$genotype); fac$treatment <- factor(fac$treatment)
cm <- tapply(fac$y, list(fac$genotype, fac$treatment), mean)
print(round(cm, 3))
simple_wt <- cm["WT","drug"] - cm["WT","veh"]
simple_ko <- cm["KO","drug"] - cm["KO","veh"]
cat(sprintf("\n  simple effect of drug in WT = %+.4f\n", simple_wt))
cat(sprintf("  simple effect of drug in KO = %+.4f\n", simple_ko))
cat(sprintf("  interaction (difference of differences) = %+.4f\n", simple_wt - simple_ko))
m_add <- lm(y ~ genotype + treatment, fac); m_int <- lm(y ~ genotype * treatment, fac)
av <- anova(m_add, m_int)
cat(sprintf("\n  interaction test: F = %.3f, p = %.4f\n", av[["F"]][2], av[["Pr(>F)"]][2]))
main_drug <- mean(cm[, "drug"]) - mean(cm[, "veh"])
cat(sprintf("\n  'main effect' of drug (averaged over genotype) = %+.4f\n", main_drug))
cat("  But the drug does nothing in KO. The main effect describes a\n")
cat("  hypothetical average genotype nobody studied. When an interaction is\n")
cat("  present, REPORT SIMPLE EFFECTS (eq. 12.5):\n")
print(contrast(emmeans(m_int, ~ treatment | genotype), "pairwise"))

#' ## 5. Type I / II / III sums of squares

#+ ss-types
header("5. Type I/II/III SS: only matters when unbalanced")
cat("  BALANCED design - Type I SS with both term orders:\n")
for (form in c("y ~ genotype + treatment", "y ~ treatment + genotype")) {
  a1 <- anova(lm(as.formula(form), fac))
  cat(sprintf("    %-34s %s\n", form,
              paste(sprintf("%s=%.3f", rownames(a1)[1:2], a1[["Sum Sq"]][1:2]),
                    collapse = "  ")))
}
set.seed(1203)
drop_idx <- sample(which(fac$genotype == "KO" & fac$treatment == "drug"), 6)
unbal <- fac[-drop_idx, ]
cat(sprintf("\n  UNBALANCED design (cell sizes %s):\n",
            paste(table(unbal$genotype, unbal$treatment), collapse = " ")))
for (form in c("y ~ genotype + treatment", "y ~ treatment + genotype")) {
  a1 <- anova(lm(as.formula(form), unbal))
  cat(sprintf("    Type I, %-34s %s\n", form,
              paste(sprintf("%s=%.3f", rownames(a1)[1:2], a1[["Sum Sq"]][1:2]),
                    collapse = "  ")))
}
## Type II by hand: each main effect adjusted for the OTHER main effect only.
ss2_geno <- anova(lm(y ~ treatment, unbal), lm(y ~ treatment + genotype, unbal))[["Sum of Sq"]][2]
ss2_trt  <- anova(lm(y ~ genotype, unbal), lm(y ~ genotype + treatment, unbal))[["Sum of Sq"]][2]
cat(sprintf("    Type II (order-independent)            genotype=%.3f  treatment=%.3f\n",
            ss2_geno, ss2_trt))
cat("\n  Type I depends on term ORDER; Type II does not. Type III requires\n")
cat("  SUM-TO-ZERO contrasts to be meaningful - with R's DEFAULT treatment\n")
cat("  coding it tests hypotheses nobody wants. The honest framing: write\n")
cat("  down the two models you want to compare and use anova().\n")

#' ## 6. Effect sizes and post-hoc multiplicity, eq. (12.6)-(12.7)

#+ effect-multiplicity
header("6. eta^2 vs omega^2 (12.6); Tukey HSD (12.7)")
ms_within <- ss_within / (N - k)
eta2 <- ss_between / ss_total
omega2 <- (ss_between - (k-1)*ms_within) / (ss_total + ms_within)   # eq. (12.6)
cat(sprintf("  eta^2   = %.4f   (biased upward)\n", eta2))
cat(sprintf("  omega^2 = %.4f   (eq. 12.6, less biased)\n", omega2))
cat("\n  All-pairs comparisons via TukeyHSD (12.7):\n")
print(round(TukeyHSD(aov(y ~ dose, d))$dose, 4))
cat(sprintf("\n  Tukey controls FWER across all %d pairs. But the single\n", choose(k, 2)))
cat(sprintf("  pre-specified LINEAR TREND contrast had p = %.2e, far stronger\n",
            results[[3]]$p))
cat("  than any adjusted pairwise test. Pre-specify!\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "anova.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
stripchart(y ~ dose, d, vertical = TRUE, method = "jitter", pch = 16,
           col = "steelblue", xlab = "dose", ylab = "response",
           main = "Eq. (12.1): between vs within variation")
for (i in seq_along(gm)) segments(i - 0.25, gm[i], i + 0.25, gm[i], lwd = 2)
abline(h = grand, col = "red", lty = 2)

est <- sapply(results, `[[`, "estimate")
lo <- sapply(results, function(r) r$ci[1]); hi <- sapply(results, function(r) r$ci[2])
plot(est, seq_along(est), pch = 16, xlim = range(c(lo, hi)), yaxt = "n",
     xlab = "contrast estimate", ylab = "",
     main = "Eq. (12.3): contrasts with 95% CIs")
segments(lo, seq_along(est), hi, seq_along(est))
axis(2, seq_along(est), sapply(results, `[[`, "label"), las = 1, cex.axis = 0.55)
abline(v = 0, lty = 2)

matplot(t(cm), type = "b", pch = 16, lty = 1, lwd = 2, xaxt = "n",
        col = c("steelblue", "darkorange"), xlab = "", ylab = "response",
        main = "Eq. (12.5): non-parallel lines = interaction")
axis(1, 1:2, colnames(cm))
legend("topleft", rownames(cm), col = c("steelblue", "darkorange"), lwd = 2,
       bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "anova.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 12)
#'
#' 1. Pre-specify contrasts; omnibus F then all-pairs is rarely best.
#' 2. Check the interaction before interpreting main effects.
#' 3. For ordered doses a trend contrast has far more power than all-pairs.
#' 4. Balance the design when you can.
#'
#' **Next:** `13_generalized_linear_models.R`
