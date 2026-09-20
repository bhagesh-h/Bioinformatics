#' ---
#' title: "Module 03 - Exploratory data analysis and quality diagnostics"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 3, equations (3.1)-(3.7)
#'
#' ## What you will learn
#'
#' 1. The ECDF and the DKW bound (3.1)-(3.2).
#' 2. Robust spread: MAD (3.3)-(3.4) and the MAD outlier rule (3.5).
#' 3. **The mean-variance plot (3.6) is how you choose a likelihood.**
#' 4. Why "test for normality, then pick a test" fails (3.7).
#' 5. The standard omics QC battery.

#+ setup, message = FALSE
MODULE_NAME <- "03_exploratory_data_analysis"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. The ECDF and how much it can be trusted, eq. (3.1)-(3.2)

#+ ecdf
header("1. ECDF and the DKW uniform confidence band (3.1)-(3.2)")
dkw_halfwidth <- function(n, alpha = 0.05) sqrt(log(2 / alpha) / (2 * n))
for (n in c(6, 12, 30, 100, 1000)) {
  cat(sprintf("  n = %5d  95%% DKW band half-width = +/-%.3f\n", n, dkw_halfwidth(n)))
}
cat("\nWith n=6 the true CDF could be anywhere within +/-0.55 of the observed\n")
cat("one. That is why 'the data look non-normal' is a weak basis for\n")
cat("switching tests at small n.\n")

#' ## 2. Robust location and spread, eq. (3.3)-(3.5)
#'
#' R's `mad()` already applies the 1.4826 constant by default (`constant=`).

#+ mad
header("2. MAD: the 1.4826 constant, and breakdown (3.3)-(3.4)")
cat(sprintf("  1 / qnorm(0.75) = 1 / %.6f = %.6f\n", qnorm(0.75), 1 / qnorm(0.75)))
cat(sprintf("  R's mad() default constant = %.4f\n\n", formals(mad)$constant))

set.seed(303)
clean <- rnorm(1000)
cat(sprintf("  clean N(0,1): sd = %.4f   mad = %.4f   (truth 1.0)\n",
            sd(clean), mad(clean)))
cat(sprintf("\n%12s %10s %11s\n", "% corrupted", "sd", "mad"))
for (frac in c(0, 0.01, 0.05, 0.20, 0.45)) {
  dirty <- clean
  k <- floor(frac * length(dirty))
  if (k > 0) dirty[seq_len(k)] <- 1000
  cat(sprintf("%11.0f%% %10.2f %11.3f\n", 100 * frac, sd(dirty), mad(dirty)))
}
cat("\nsd has breakdown point 0; mad has 50%.\n")

#+ mad-outliers
## The MAD-based rule used in single-cell QC (scverse default), eq. (3.5).
mad_outliers <- function(x, k = 3) abs(x - median(x)) > k * mad(x)

set.seed(304)
qc_metric <- c(rnorm(1940, 3.5, 0.25),   # good cells
               rnorm(40, 2.2, 0.20),     # empty droplets
               rnorm(20, 4.4, 0.15))     # doublets
for (k in c(3, 5)) {
  fl <- mad_outliers(qc_metric, k)
  cat(sprintf("  k = %d:  %4d / %d cells flagged (%.1f%%)\n",
              k, sum(fl), length(qc_metric), 100 * mean(fl)))
}

#' ## 3. The mean-variance relationship IS the assay's fingerprint, eq. (3.6)

#+ mean-var
header("3. Reading the likelihood off the mean-variance plot (3.6)")
set.seed(305)
G <- 3000; n_samp <- 8
mus <- exp(rnorm(G, 3, 1.6))

nb_matrix <- function(mu_vec, phi, n) {
  if (phi <= 0) matrix(rpois(length(mu_vec) * n, rep(mu_vec, n)), ncol = n)
  else matrix(rnbinom(length(mu_vec) * n, mu = rep(mu_vec, n), size = 1 / phi), ncol = n)
}

assays <- list(
  "Technical replicates (Poisson)"        = nb_matrix(mus, 0, n_samp),
  "Biological replicates (NB, phi=0.16)"  = nb_matrix(mus, 0.16, n_samp),
  "Log-intensities (Gaussian)"            = matrix(rnorm(G * n_samp, log2(mus), 0.35),
                                                   ncol = n_samp))

cat(sprintf("%-40s %28s\n", "assay", "slope of log(var)~log(mean)"))
for (nm in names(assays)) {
  M <- assays[[nm]]
  m <- rowMeans(M); v <- apply(M, 1, var)
  keep <- m > 1 & v > 0
  slope <- coef(lm(log(v[keep]) ~ log(m[keep])))[2]
  cat(sprintf("%-40s %28.3f\n", nm, slope))
}
cat("\n  slope ~ 1 -> Var = mean       -> Poisson  (technical only)\n")
cat("  slope ~ 2 -> Var = phi*mean^2 -> NB       (biological variability)\n")
cat("  slope ~ 0 -> Var independent  -> Gaussian (already stabilised)\n")

nb <- assays[["Biological replicates (NB, phi=0.16)"]]
m <- rowMeans(nb); v <- apply(nb, 1, var); keep <- m > 20
phi_hat <- median((v[keep] - m[keep]) / m[keep]^2)
cat(sprintf("\nMethod-of-moments dispersion from Var = mu + phi*mu^2 (eq. 4.9):\n"))
cat(sprintf("  phi_hat = %.4f  (true 0.16)\n", phi_hat))
cat(sprintf("  BCV = sqrt(phi) = %.3f   <- edgeR reports this\n", sqrt(phi_hat)))

#' ## 4. Why the normality pre-test fails, eq. (3.7)

#+ normality-pretest
header("4. The normality-pretest trap (3.7)")
set.seed(306)
cat(sprintf("%6s %20s %34s\n", "n", "SE(skew)=sqrt(6/n)",
            "P(Shapiro rejects | TRUE normal)"))
for (n in c(8, 20, 50, 200, 2000)) {
  rej <- mean(replicate(500, shapiro.test(rnorm(n))$p.value < 0.05))
  cat(sprintf("%6d %20.3f %33.1f%%\n", n, sqrt(6 / n), 100 * rej))
}
cat("\nNow a genuinely skewed distribution (log-normal), same test:\n")
cat(sprintf("%6s %34s\n", "n", "P(Shapiro rejects | NOT normal)"))
for (n in c(8, 20, 50, 200)) {
  pw <- mean(replicate(500, shapiro.test(rlnorm(n, 0, 0.75))$p.value < 0.05))
  cat(sprintf("%6d %33.1f%%\n", n, 100 * pw))
}
cat("\nAt n=8 the test misses clear log-normality most of the time. Choose the\n")
cat("model from the MEASUREMENT (Topic 2) and the mean-variance plot (3.6).\n")

#' ## 5. The standard omics QC battery

#+ qc-battery
header("5. QC battery on a simulated multi-batch RNA-seq experiment")
set.seed(307)
G <- 4000; n <- 12
meta <- data.frame(
  sample = sprintf("S%02d", 1:n),
  condition = rep(c("ctrl", "trt"), each = 6),
  ## Batches BALANCED across conditions (the good design of Module 01).
  batch = rep(c("b1", "b1", "b2", "b2", "b3", "b3"), 2),
  stringsAsFactors = FALSE)
rownames(meta) <- meta$sample

base <- exp(rnorm(G, 3, 1.5))
depth <- runif(n, 0.6, 1.6)
batch_shift <- c(b1 = 0, b2 = 0.35, b3 = -0.30)
mu <- outer(base, depth) * rep(exp(batch_shift[meta$batch]), each = G)
mu[1:200, 7:12] <- mu[1:200, 7:12] * 2                # a real effect
counts <- matrix(rnbinom(G * n, mu = as.vector(mu), size = 1 / 0.16), nrow = G)
dimnames(counts) <- list(sprintf("G%04d", 1:G), meta$sample)
stopifnot(identical(colnames(counts), rownames(meta)))   # eq. (2.1)

qc <- data.frame(
  library_size = colSums(counts),
  genes_detected = colSums(counts > 0),
  pct_top50 = apply(counts, 2, function(x) 100 * sum(sort(x, TRUE)[1:50]) / sum(x)),
  condition = meta$condition, batch = meta$batch)
print(round(qc[, 1:3], 1))

cat("\nPer-batch library-size medians (drift check):\n")
print(tapply(qc$library_size, qc$batch, median))

logcpm <- log2(t(t(counts) / colSums(counts)) * 1e6 + 1)
cm <- cor(logcpm, method = "spearman"); diag(cm) <- NA
cat(sprintf("\nSample-sample Spearman: min=%.3f, median=%.3f\n",
            min(cm, na.rm = TRUE), median(cm, na.rm = TRUE)))
cat("A sample correlating much worse with everything else is an outlier/swap.\n")

## PC x metadata association (eq. 17.6) - the key batch diagnostic.
pca <- prcomp(t(logcpm), center = TRUE, scale. = FALSE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
cat("\nPC x metadata association (eq. 17.6) - R^2 of each PC on each factor:\n")
cat(sprintf("%4s %7s %11s %8s %13s\n", "PC", "PVE", "condition", "batch", "log lib size"))
for (k in 1:3) {
  z <- pca$x[, k]
  r2 <- function(f) summary(lm(z ~ f))$r.squared
  cat(sprintf("%4d %6.1f%% %11.3f %8.3f %13.3f\n", k, 100 * pve[k],
              r2(factor(meta$condition)), r2(factor(meta$batch)),
              r2(log(qc$library_size))))
}
cat("\nIf PC1 is better explained by batch or library size than by condition,\n")
cat("you have a technical problem, not a discovery (stats.md Topic 19).\n")

#' ## 6. Figure

#+ figure
png(file.path(OUT, "qc_panel.png"), width = 1400, height = 800, res = 110)
par(mfrow = c(2, 3), mar = c(4.2, 4.2, 2.5, 1))

set.seed(308)
small <- rnorm(12)
plot(ecdf(small), main = "Eq. (3.1)-(3.2): ECDF uncertainty", xlab = "value",
     verticals = TRUE, do.points = FALSE, col = "steelblue", lwd = 2)
eps <- dkw_halfwidth(12); xs <- sort(small); ys <- seq_along(xs) / 12
lines(xs, pmin(ys + eps, 1), col = "grey60", lty = 2)
lines(xs, pmax(ys - eps, 0), col = "grey60", lty = 2)
curve(pnorm(x), add = TRUE, lty = 3)

hist(qc_metric, breaks = 60, col = "steelblue", border = "white",
     main = "Eq. (3.5): MAD outlier rule", xlab = "log10 counts per cell")
for (k in c(3, 5)) abline(v = median(qc_metric) + c(-1, 1) * k * mad(qc_metric),
                          col = if (k == 3) "orange" else "red", lty = 2)

M1 <- assays[[1]]; M2 <- assays[[2]]
## A log-log plot silently drops non-positive values, so select them
## explicitly: a gene with zero mean or zero variance is a real observation
## about the data, not something to let a plotting default hide.
.mv_m <- rowMeans(M1); .mv_v <- apply(M1, 1, var)
.mv_ok <- .mv_m > 0 & .mv_v > 0
if (any(!.mv_ok))
  cat(sprintf("  (%d of %d genes have zero mean or zero variance and cannot\n   appear on log axes)\n",
              sum(!.mv_ok), length(.mv_ok)))
plot(.mv_m[.mv_ok], .mv_v[.mv_ok], log = "xy", pch = ".", col = "steelblue",
     xlab = "mean", ylab = "variance", main = "Eq. (3.6): the assay fingerprint")
points(rowMeans(M2), apply(M2, 1, var), pch = ".", col = "firebrick")
lim <- 10^seq(0, 3.5, length.out = 50)
lines(lim, lim, lty = 2); lines(lim, lim + 0.16 * lim^2, lty = 3)
legend("topleft", c("Poisson", "NB", "Var=mean", "Var=mu+0.16mu^2"),
       col = c("steelblue", "firebrick", "black", "black"),
       pch = c(16, 16, NA, NA), lty = c(NA, NA, 2, 3), bty = "n", cex = 0.6)

boxplot(library_size ~ batch, data = qc, col = "lightsteelblue",
        main = "Library size by batch", xlab = "batch", ylab = "library size")
plot(qc$library_size, qc$genes_detected, pch = 16, col = "steelblue",
     xlab = "library size", ylab = "genes detected", main = "Detection is depth-driven")
plot(pca$x[, 1], pca$x[, 2], pch = c(16, 17, 15)[factor(meta$batch)],
     col = "steelblue", cex = 1.4,
     xlab = sprintf("PC1 (%.0f%%)", 100 * pve[1]),
     ylab = sprintf("PC2 (%.0f%%)", 100 * pve[2]),
     main = "PCA coloured by BATCH")
legend("topright", levels(factor(meta$batch)), pch = c(16, 17, 15), bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "qc_panel.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 3)
#'
#' 1. Explore to diagnose data quality and choose a LIKELIHOOD, not a
#'    HYPOTHESIS. Choosing the hypothesis after seeing the outcome is HARKing.
#' 2. Never delete an outlier without a documented, outcome-independent reason.
#' 3. Any preprocessing that uses the outcome must be abandoned or moved inside
#'    the resampling loop (Module 21).
#'
#' **Next:** `04_probability_and_sampling.R`
