#' ---
#' title: "Applied 24 - Genotypes, GWAS, and statistical genetics"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 24, equations (24.1)-(24.9)
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | A two-ancestry case/control GWAS with LD structure |
#' | **Key threats** | Population structure, LD, extreme multiplicity |
#'
#' Ancestry creates allele-frequency differences that correlate with phenotype.
#' Uncorrected, this produces genome-wide false positives that look exactly
#' like real associations.

#+ setup, message = FALSE
MODULE_NAME <- "24_gwas_association"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
ALPHA_GW <- 5e-8

#' ## 1. Simulate a structured cohort with LD blocks

#+ simulate
header("1. Genotypes with ancestry structure and LD")
simulate_gwas <- function(n = 1500, n_snps = 3000, block_size = 25, n_causal = 3,
                          beta_causal = 0.45, ancestry_effect = 0.9) {
  pop <- rbinom(n, 1, 0.45)
  anc_freq <- runif(n_snps, 0.1, 0.9); fst <- 0.12
  f0 <- pmin(pmax(rbeta(n_snps, anc_freq*(1-fst)/fst, (1-anc_freq)*(1-fst)/fst), 0.02), 0.98)
  f1 <- pmin(pmax(rbeta(n_snps, anc_freq*(1-fst)/fst, (1-anc_freq)*(1-fst)/fst), 0.02), 0.98)
  G <- matrix(0L, n, n_snps)
  for (b in seq_len(n_snps %/% block_size)) {
    cols <- ((b-1)*block_size + 1):(b*block_size)
    for (a in 1:2) {
      latent <- rnorm(n)
      for (k in seq_along(cols)) {
        j <- cols[k]
        f <- ifelse(pop == 1, f1[j], f0[j])
        rho <- 0.85^(k-1)
        u <- rho*latent + sqrt(1 - rho^2)*rnorm(n)
        G[, j] <- G[, j] + as.integer(u > qnorm(1 - f))
      }
    }
  }
  causal <- sample(n_snps, n_causal)
  lin <- ancestry_effect*pop + as.vector(G[, causal] %*% rep(beta_causal, n_causal)) - 1
  list(G = G, y = rbinom(n, 1, plogis(lin)), pop = pop, causal = sort(causal))
}
set.seed(3401)
sim <- simulate_gwas()
G <- sim$G; y <- sim$y; pop <- sim$pop; causal <- sim$causal
n <- nrow(G); n_snps <- ncol(G)
cat(sprintf("  %d individuals x %d SNPs\n", n, n_snps))
cat(sprintf("  case prevalence: overall %.3f; pop0 %.3f, pop1 %.3f\n",
            mean(y), mean(y[pop == 0]), mean(y[pop == 1])))
cat(sprintf("  causal SNPs: %s\n", paste(causal, collapse = " ")))
cat("\n  Ancestry raises BOTH allele frequencies and disease risk - a textbook\n")
cat("  confounder (Module 32, fork structure).\n")

#' ## 2. Quality control, eq. (24.2)

#+ qc
header("2. QC: MAF, Hardy-Weinberg (24.2)")
hwe_p <- function(g) {
  n_aa <- sum(g == 0); n_ab <- sum(g == 1); n_bb <- sum(g == 2); N <- length(g)
  if (N == 0) return(1)
  p <- (2*n_bb + n_ab)/(2*N)
  exp_ <- c(N*(1-p)^2, 2*N*p*(1-p), N*p^2)
  obs <- c(n_aa, n_ab, n_bb)
  k <- exp_ > 0
  if (sum(k) < 2) return(1)
  pchisq(sum((obs[k] - exp_[k])^2/exp_[k]), 1, lower.tail = FALSE)
}
maf <- pmin(colMeans(G)/2, 1 - colMeans(G)/2)
## CRITICAL: HWE is tested in CONTROLS only. A real disease association can
## cause HWE deviation in cases, so testing everyone discards true signals.
hwe_ctrl <- apply(G[y == 0, ], 2, hwe_p)
hwe_all <- apply(G, 2, hwe_p)
keep <- maf > 0.01 & hwe_ctrl > 1e-6
cat(sprintf("  MAF > 0.01              : %d SNPs pass\n", sum(maf > 0.01)))
cat(sprintf("  HWE in CONTROLS > 1e-6  : %d pass\n", sum(hwe_ctrl > 1e-6)))
cat(sprintf("  HWE in EVERYONE > 1e-6  : %d pass\n", sum(hwe_all > 1e-6)))
cat(sprintf("  combined                : %d of %d SNPs retained\n", sum(keep), n_snps))
cat(sprintf("\n  causal SNPs surviving QC: %s\n",
            paste(causal[keep[causal]], collapse = " ")))
cat("  Apply HWE filtering to CONTROLS only.\n")

#' ## 3. The naive scan, and genomic inflation, eq. (24.1), (24.4)

#+ scan
header("3. Uncorrected scan -> genome-wide false positives (24.1), (24.4)")
scan_gwas <- function(G, y, covars = NULL, keep) {
  idx <- which(keep)
  B <- if (is.null(covars)) matrix(1, length(y), 1) else cbind(1, covars)
  p <- rep(NA_real_, ncol(G)); b <- rep(NA_real_, ncol(G))
  for (j in idx) {
    X <- cbind(B, G[, j])
    f <- try(suppressWarnings(glm.fit(X, y, family = binomial())), silent = TRUE)
    if (inherits(f, "try-error")) next
    cf <- f$coefficients
    if (any(is.na(cf))) next
    ## Wald z from the IRLS weights (eq. 13.5).
    W <- f$fitted.values*(1 - f$fitted.values)
    V <- try(chol2inv(chol(crossprod(X*sqrt(W)))), silent = TRUE)
    if (inherits(V, "try-error")) next
    se <- sqrt(V[ncol(X), ncol(X)])
    b[j] <- cf[ncol(X)]; p[j] <- 2*pnorm(abs(cf[ncol(X)]/se), lower.tail = FALSE)
  }
  list(p = p, beta = b)
}
lambda_gc <- function(p) {                                       # eq. (24.4)
  p <- p[is.finite(p) & p > 0]
  median(qchisq(p, 1, lower.tail = FALSE))/0.4549   # median of chi2_1 = 0.4549
}
r_naive <- scan_gwas(G, y, NULL, keep)
cat(sprintf("  lambda_GC (uncorrected) = %.3f   (1.00 = calibrated)\n",
            lambda_gc(r_naive$p)))
cat(sprintf("  SNPs below 5e-8         : %d\n", sum(r_naive$p < ALPHA_GW, na.rm = TRUE)))
cat(sprintf("  of which are causal     : %d\n",
            sum(r_naive$p[causal] < ALPHA_GW, na.rm = TRUE)))
cat(sprintf("\n  Bonferroni threshold for %d tested SNPs: %.2e\n", sum(keep), 0.05/sum(keep)))
cat("  The conventional 5e-8 (eq. 24.5) is 0.05/10^6, a correction for the\n")
cat("  ~1 million effectively independent common variants in European LD.\n")

#' ## 4. Correcting for structure: ancestry PCs, eq. (24.3)

#+ pcs
header("4. Principal components of the genotype matrix (24.3)")
genotype_pcs <- function(G, keep, k = 10, ld_thin = 5) {
  idx <- which(keep)[seq(1, sum(keep), by = ld_thin)]
  Z <- scale(G[, idx])
  Z[is.na(Z)] <- 0
  sv <- svd(Z, nu = k, nv = 0)
  list(x = sv$u %*% diag(sv$d[1:k]), pve = sv$d^2/sum(sv$d^2))
}
pcs <- genotype_pcs(G, keep)
cat(sprintf("  %4s%9s%24s\n", "PC", "PVE", "|corr with ancestry|"))
for (j in 1:5)
  cat(sprintf("  %4d%8.2f%%%24.3f\n", j, 100*pcs$pve[j], abs(cor(pcs$x[, j], pop))))
cat("\n  PC1 captures ancestry almost perfectly. In a real study you would also\n")
cat("  plot PCs against a reference panel (1000 Genomes) to LABEL the ancestry\n")
cat("  groups rather than just adjust for them.\n\n")
for (n_pc in c(0, 1, 2, 5, 10)) {
  cov_ <- if (n_pc == 0) NULL else pcs$x[, 1:n_pc, drop = FALSE]
  r <- scan_gwas(G, y, cov_, keep)
  cat(sprintf("  %2d PCs: lambda_GC = %.3f, %3d genome-wide hits, %d of them causal\n",
              n_pc, lambda_gc(r$p), sum(r$p < ALPHA_GW, na.rm = TRUE),
              sum(r$p[causal] < ALPHA_GW, na.rm = TRUE)))
  if (n_pc == 10) r_final <- r
}
cat("\n  Adjusting for ancestry PCs restores calibration (lambda -> 1) and\n")
cat("  leaves the true signals. NOTE lambda > 1 is NOT automatically\n")
cat("  confounding: a highly polygenic trait genuinely inflates it at large n.\n")
cat("  LD score regression separates polygenicity (slope) from confounding\n")
cat("  (intercept); the QQ plot is the visual version.\n")

#' ## 5. LD and clumping, eq. (24.6)

#+ clumping
header("5. LD means hits come in blocks (24.6)")
sig_idx <- which(is.finite(r_final$p) & r_final$p < ALPHA_GW)
cat(sprintf("  %d SNPs pass 5e-8 after PC adjustment\n", length(sig_idx)))
clump <- function(p, G, sig_idx, r2_thresh = 0.1, window = 60) {
  ord <- sig_idx[order(p[sig_idx])]
  lead <- integer(0); taken <- integer(0)
  for (j in ord) {
    if (j %in% taken) next
    lead <- c(lead, j)
    for (k in max(1, j-window):min(ncol(G), j+window)) {
      if (k %in% sig_idx && !(k %in% taken)) {
        if (sd(G[, j]) > 0 && sd(G[, k]) > 0 && cor(G[, j], G[, k])^2 > r2_thresh)
          taken <- c(taken, k)
      }
    }
    taken <- c(taken, j)
  }
  lead
}
lead_snps <- clump(r_final$p, G, sig_idx)
cat(sprintf("  after clumping (r^2 < 0.1): %d independent loci\n", length(lead_snps)))
cat(sprintf("  lead SNPs: %s\n", paste(sort(lead_snps), collapse = " ")))
cat(sprintf("  true causal SNPs: %s\n", paste(causal, collapse = " ")))
for (L in head(lead_snps, 3)) {
  near <- causal[abs(causal - L) < 30]
  if (length(near)) {
    c0 <- near[1]
    cat(sprintf("    lead SNP %d: causal variant %d is %d SNPs away, r^2 = %.3f,\n",
                L, c0, abs(L - c0), cor(G[, L], G[, c0])^2))
    cat(sprintf("      p_lead = %.2e, p_causal = %.2e\n", r_final$p[L], r_final$p[c0]))
  }
}
cat("\n  THE LEAD SNP IS OFTEN NOT THE CAUSAL VARIANT. It is the one that\n")
cat("  happened to be typed and happened to tag the causal haplotype best in\n")
cat("  THIS sample. Fine-mapping computes posterior inclusion probabilities\n")
cat("  and a 95% credible set rather than naming the top SNP.\n")

#' ## 6. Polygenic scores and leakage, eq. (24.8)

#+ pgs
header("6. Polygenic scores: the leakage trap (24.8)")
auc <- function(labels, scores) {
  r <- rank(scores); n1 <- sum(labels == 1); n0 <- sum(labels == 0)
  (sum(r[labels == 1]) - n1*(n1+1)/2)/(n1*n0)
}
set.seed(3402)
te <- sample(n, round(0.35*n)); tr <- setdiff(seq_len(n), te)
build_pgs <- function(idx_w, idx_score, p_thresh = 1e-3) {
  r <- scan_gwas(G[idx_w, ], y[idx_w], pcs$x[idx_w, 1:5], keep)
  sel <- which(is.finite(r$p) & r$p < p_thresh)
  if (!length(sel)) sel <- order(r$p)[1:5]
  list(score = as.vector(G[idx_score, sel, drop = FALSE] %*% r$beta[sel]), n_snp = length(sel))
}
h <- build_pgs(tr, te)                       # HONEST: weights from TRAINING only
l <- build_pgs(seq_len(n), te)               # LEAKY: weights from ALL samples
cat(sprintf("  weights from TRAINING only : %4d SNPs, test AUC = %.3f\n",
            h$n_snp, auc(y[te], h$score)))
cat(sprintf("  weights from ALL samples   : %4d SNPs, test AUC = %.3f   <- inflated\n",
            l$n_snp, auc(y[te], l$score)))
cat(sprintf("  optimism from the leak     : %+.3f\n",
            auc(y[te], l$score) - auc(y[te], h$score)))
p0 <- which(pop == 0); p1 <- which(pop == 1)
tr0 <- p0[-(1:floor(length(p0)/3))]; te0 <- p0[1:floor(length(p0)/3)]
s0 <- build_pgs(tr0, te0); s1 <- build_pgs(tr0, p1)
a_same <- auc(y[te0], s0$score); a_other <- auc(y[p1], s1$score)
cat("\n  Portability (trained entirely in population 0):\n")
cat(sprintf("    tested in population 0 : AUC = %.3f\n", a_same))
cat(sprintf("    tested in population 1 : AUC = %.3f\n", a_other))
cat(sprintf("    relative loss          : %.0f%%\n",
            100*(1 - (a_other - 0.5)/max(a_same - 0.5, 1e-9))))
cat("  Real PGS lose 50-80% of their predictive R^2 across ancestries, because\n")
cat("  LD patterns and allele frequencies differ. ALWAYS report the ancestry of\n")
cat("  both the training and the test data.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "gwas.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
plot(NA, xlim = c(0, 4), ylim = c(0, max(-log10(r_naive$p), na.rm = TRUE)),
     xlab = "expected -log10 p", ylab = "observed", main = "Eq. (24.4): QQ plot")
for (row_ in list(list(r_naive$p, "firebrick"), list(r_final$p, "steelblue"))) {
  q <- sort(row_[[1]][is.finite(row_[[1]]) & row_[[1]] > 0])
  points(-log10(ppoints(length(q))), -log10(q), pch = ".", col = row_[[2]])
}
abline(0, 1, lty = 2)
legend("topleft", sprintf(c("uncorrected (lambda=%.2f)", "10 PCs (lambda=%.2f)"),
                          c(lambda_gc(r_naive$p), lambda_gc(r_final$p))),
       col = c("firebrick", "steelblue"), pch = 16, bty = "n", cex = 0.6)
ok <- is.finite(r_final$p)
plot(which(ok), -log10(r_final$p[ok]), pch = ".",
     col = ifelse(which(ok) %in% causal, "firebrick", "grey50"),
     xlab = "SNP index", ylab = "-log10 p", main = "Manhattan plot (PC-adjusted)")
abline(h = -log10(ALPHA_GW), col = "red", lty = 2)
plot(pcs$x[, 1], pcs$x[, 2], pch = 16, cex = 0.4,
     col = c("steelblue", "darkorange")[pop + 1],
     xlab = sprintf("PC1 (%.1f%%)", 100*pcs$pve[1]),
     ylab = sprintf("PC2 (%.1f%%)", 100*pcs$pve[2]), main = "Eq. (24.3): ancestry PCs")
if (length(lead_snps)) {
  L <- lead_snps[1]; lo <- max(1, L-60); hi <- min(n_snps, L+60)
  r2s <- sapply(lo:hi, function(k) if (sd(G[, k]) > 0) cor(G[, L], G[, k])^2 else 0)
  plot(lo:hi, -log10(pmax(r_final$p[lo:hi], 1e-300)), pch = 16, cex = 0.7,
       col = hcl.colors(10, "viridis")[cut(r2s, 10)],
       xlab = "SNP index", ylab = "-log10 p", main = "Eq. (24.6): LD around the lead SNP")
  abline(v = L, col = "red", lty = 2)
  for (c0 in causal) if (c0 >= lo && c0 <= hi) abline(v = c0, lty = 3)
}
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "gwas.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Why 5e-8?
#'
#' Estimate the number of EFFECTIVELY INDEPENDENT tests (eigenvalues of the SNP
#' correlation matrix explaining 99.5% of variance) and derive the appropriate
#' Bonferroni threshold. Compare to 0.05/n_snps.

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# idx <- which(keep)[1:1200]
# Z <- scale(G[, idx]); Z[is.na(Z)] <- 0
# ev <- pmax(eigen(cor(Z), only.values = TRUE)$values, 0)
# m_eff <- which(cumsum(ev)/sum(ev) >= 0.995)[1]
# cat(sprintf("  SNPs tested                  : %d\n", length(idx)))
# cat(sprintf("  effectively independent tests: %d\n", m_eff))
# cat(sprintf("  naive Bonferroni 0.05/M      : %.2e\n", 0.05/length(idx)))
# cat(sprintf("  LD-aware  0.05/M_eff         : %.2e\n", 0.05/m_eff))
# cat(sprintf("  ratio                        : %.2fx too strict\n", length(idx)/m_eff))
#
# ## LD makes neighbouring tests redundant, so a naive Bonferroni over every SNP
# ## is CONSERVATIVE. The genome-wide 5e-8 is calibrated to ~1e6 effectively
# ## independent COMMON variants, not to the number of SNPs on the array - which
# ## is why denser arrays do not need a stricter threshold, while whole-genome
# ## sequencing (which adds genuinely independent RARE variants) does.

#' ### Problem 2: Case/control ancestry imbalance
#'
#' Re-simulate with `ancestry_effect = 0` but sample cases preferentially from
#' population 1. Show that lambda_GC still inflates, and PCs still fix it.

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(777)
# s2 <- simulate_gwas(ancestry_effect = 0)
# ## Deliberately over-sample population 1 among cases (a recruitment artefact).
# drop <- sample(which(s2$y == 1 & s2$pop == 0),
#                size = floor(0.7*sum(s2$y == 1 & s2$pop == 0)))
# sel <- setdiff(seq_along(s2$y), drop)
# G2 <- s2$G[sel, ]; y2 <- s2$y[sel]; pop2 <- s2$pop[sel]
# maf2 <- pmin(colMeans(G2)/2, 1 - colMeans(G2)/2)
# keep2 <- maf2 > 0.01
# r2a <- scan_gwas(G2, y2, NULL, keep2)
# pcs2 <- genotype_pcs(G2, keep2)
# r2b <- scan_gwas(G2, y2, pcs2$x[, 1:10], keep2)
# cat(sprintf("  case fraction from pop1: %.2f vs control %.2f\n",
#             mean(pop2[y2 == 1]), mean(pop2[y2 == 0])))
# cat(sprintf("  lambda_GC uncorrected : %.3f\n", lambda_gc(r2a$p)))
# cat(sprintf("  lambda_GC with 10 PCs : %.3f\n", lambda_gc(r2b$p)))
# cat(sprintf("  genome-wide hits: %d -> %d\n",
#             sum(r2a$p < 5e-8, na.rm = TRUE), sum(r2b$p < 5e-8, na.rm = TRUE)))
#
# ## Ancestry does not have to CAUSE the disease. It only has to be associated
# ## with case/control STATUS - here through recruitment - to confound every SNP
# ## whose frequency differs between populations. This is the selection/collider
# ## structure of Module 32.

#' ### Problem 3: Mendelian randomisation and the F-statistic
#'
#' Use the causal SNPs as instruments for a simulated exposure and compute the
#' IVW estimate (eq. 24.9). Add WEAK instruments and watch what happens.

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(31)
# TRUE_CAUSAL <- 0.6
# Xexp <- as.vector(G[, causal] %*% rep(0.5, length(causal))) + rnorm(n)
# U <- rnorm(n)                                        # confounder
# Yout <- TRUE_CAUSAL*Xexp + 1.2*U + rnorm(n)
# Xexp <- Xexp + 1.2*U                                 # confound the exposure too
# ivw <- function(instr) {
#   st <- t(sapply(instr, function(j) {
#     fx <- lm(Xexp ~ G[, j]); fy <- lm(Yout ~ G[, j])
#     c(coef(fx)[2], coef(summary(fx))[2,2], coef(fy)[2], coef(summary(fy))[2,2])
#   }))
#   bx <- st[,1]; sx <- st[,2]; by <- st[,3]; sy <- st[,4]
#   c(est = sum(bx*by/sy^2)/sum(bx^2/sy^2), F = mean((bx/sx)^2))   # eq. (24.9)
# }
# strong <- ivw(causal)
# weak <- ivw(c(causal, setdiff(seq_len(n_snps), causal)[1:8]))
# cat(sprintf("  TRUE causal effect            = %.3f\n", TRUE_CAUSAL))
# cat(sprintf("  naive OLS (confounded)        = %.3f\n", coef(lm(Yout ~ Xexp))[2]))
# cat(sprintf("  IVW, strong instruments only  = %.3f  (mean F = %.1f)\n",
#             strong["est"], strong["F"]))
# cat(sprintf("  IVW, plus 8 weak instruments  = %.3f  (mean F = %.1f)\n",
#             weak["est"], weak["F"]))
#
# ## Relevance is CHECKABLE: mean F > 10 is the usual rule. Weak instruments
# ## bias IVW TOWARD the confounded OLS value. The other two MR assumptions -
# ## independence and the EXCLUSION RESTRICTION - are NOT testable. MR-Egger and
# ## weighted-median estimators are sensitivity analyses for horizontal
# ## pleiotropy, not proofs that it is absent.

#' ## What to take away
#'
#' 1. Do QC first, and test HWE in **controls only**.
#' 2. Control structure with PCs or an LMM; verify with lambda_GC and a QQ plot.
#' 3. Clump before counting "independent loci"; fine-map before naming a gene.
#' 4. Never evaluate a PGS in samples that contributed to its weights, and
#'    report ancestry transferability.
#'
#' **Next:** `25_proteomics_missing_values.R`
