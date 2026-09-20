#' ---
#' title: "Applied 43 - Fine-mapping, colocalisation, and heritability"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 43, equations (43.1)-(43.8)
#' **Core modules used:** 08, 22, 23, 34
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | A GWAS locus taken forward: fine-mapping, eQTL colocalisation, LDSC, MR |
#' | **Key threats** | LD, the single-causal assumption, confounding vs polygenicity, invalid instruments, winner's curse |
#' | **Here** | Simulated genotypes with realistic LD and a known causal variant |
#'
#' ## What module 34 left unfinished
#'
#' Module 34 found associated loci and stopped. This module does the four
#' things that come next, and each one rests on an assumption worth stating
#' out loud.

#+ setup, message = FALSE
MODULE_NAME <- "43_advanced_statistical_genetics"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(43)

#' ## 1. A locus with linkage disequilibrium

#+ locus
header("1. Simulating a locus with realistic linkage disequilibrium")
#' An AR(1) LD structure: correlation decays with distance along the locus.
simulate_locus <- function(n = 6000, n_snp = 60, causal = 26, beta = 0.16,
                           rho = 0.99, seed = 0) {
  set.seed(seed)
  idx <- seq_len(n_snp)
  Sigma <- rho^abs(outer(idx, idx, "-"))
  L <- chol(Sigma + 1e-8 * diag(n_snp))
  liab <- matrix(rnorm(n * n_snp), n) %*% L
  maf <- runif(n_snp, 0.15, 0.45)
  G <- sweep(liab, 2, qnorm(1 - maf), ">") +
    sweep(liab, 2, qnorm(1 - maf / 3), ">")
  G <- matrix(as.numeric(G), n)
  list(G = G, y = beta * G[, causal] + rnorm(n), causal = causal,
       Sigma = Sigma)
}
Lo <- simulate_locus(seed = 1)
G <- Lo$G; y <- Lo$y; CAUSAL <- Lo$causal; Sigma <- Lo$Sigma
n <- nrow(G); n_snp <- ncol(G)
#' Marginal scan: one simple regression per SNP, as a GWAS does.
scan_locus <- function(G, y) {
  res <- t(apply(G, 2, function(g) summary(lm(y ~ g))$coefficients[2, c(1, 2, 4)]))
  list(b = res[, 1], s = res[, 2], p = res[, 3])
}
S1 <- scan_locus(G, y)
beta_hat <- S1$b; se_hat <- S1$s; p_hat <- S1$p
lead <- which.min(p_hat)
cat(sprintf("  %d individuals, %d SNPs, causal variant at index %d\n", n,
            n_snp, CAUSAL))
cat(sprintf("  lead SNP (smallest p) is index %d, p = %.2e\n", lead,
            p_hat[lead]))
cat(sprintf("  correlation between lead and causal: r = %.3f\n",
            Sigma[lead, CAUSAL]))
cat(sprintf("  SNPs with p < 5e-8: %d\n", sum(p_hat < 5e-8)))
cat("\n  The whole LD block clears genome-wide significance, so the locus is
  unambiguous but the VARIANT is not. Whether the lead SNP happens to be the
  causal one is partly luck, which a single dataset cannot show you. Section 2
  repeats the experiment to measure it.\n")

#' ## 2. Fine-mapping, eq. (43.1)-(43.3)
#'
#' $$\text{ABF}_j=\sqrt{1-r}\,\exp\!\left(\frac{z_j^2 r}{2}\right),\quad
#'   r=\frac{W}{se_j^2+W} \qquad (43.1)$$
#' $$\text{PIP}_j=\text{ABF}_j/\textstyle\sum_k \text{ABF}_k \qquad (43.2)$$

#+ finemap
header("2. Approximate Bayes factors and credible sets (43.1)-(43.3)")
#' Wakefield's ABF. W is the prior variance of the effect.
abf <- function(beta, se, W = 0.04) {
  z2 <- (beta / se)^2
  r_ <- W / (se^2 + W)
  sqrt(1 - r_) * exp(z2 * r_ / 2)
}
#' Eq. (43.3): smallest set whose PIPs sum to `level`.
credible_set <- function(pip, level = 0.95) {
  o <- order(pip, decreasing = TRUE)
  k <- which(cumsum(pip[o]) >= level)[1]
  o[seq_len(k)]
}
bf <- abf(beta_hat, se_hat)
pip <- bf / sum(bf)                                   # eq. (43.2)
cs <- credible_set(pip)
cat(sprintf("  %-7s%11s%13s%13s\n", "SNP", "p", "PIP (43.2)", "in 95% set"))
for (j in order(pip, decreasing = TRUE)[1:6])
  cat(sprintf("  %-7d%11.2e%13.3f%13s\n", j, p_hat[j], pip[j],
              if (j %in% cs) "yes" else "no"))
cat(sprintf("\n  95%% credible set: %d variants, %s the causal variant\n",
            length(cs), if (CAUSAL %in% cs) "contains" else "MISSES"))
cat(sprintf("  PIP of the true causal variant: %.3f\n", pip[CAUSAL]))
cat(sprintf("  PIP of the lead SNP: %.3f\n", pip[lead]))
cat("\n  Now repeat the whole experiment 20 times, which is the only way to
  see how often the lead SNP is the right one:\n\n")
sizes <- pips <- numeric(20); hits <- leadok <- 0
for (i in 1:20) {
  Li <- simulate_locus(seed = 400 + i)
  Si <- scan_locus(Li$G, Li$y)
  pp <- abf(Si$b, Si$s); pp <- pp / sum(pp)
  cs_ <- credible_set(pp)
  sizes[i] <- length(cs_); pips[i] <- pp[Li$causal]
  hits <- hits + (Li$causal %in% cs_)
  leadok <- leadok + (which.min(Si$p) == Li$causal)
}
cat(sprintf("  %-40s%12.1f\n", "mean 95% credible set size", mean(sizes)))
cat(sprintf("  %-40s%11.0f%%\n", "set contains the causal variant",
            100 * hits / 20))
cat(sprintf("  %-40s%11.0f%%\n", "LEAD SNP is the causal variant",
            100 * leadok / 20))
cat(sprintf("  %-40s%12.3f\n", "mean PIP of the causal variant", mean(pips)))
cat("\n  The set nearly always contains the causal variant, and the lead SNP
  is it only some of the time. Reporting a short set is therefore honest in a
  way that naming the top hit is not.\n")

#' ### 2b. The harder, and more common, case: the causal variant is untyped
#'
#' Arrays genotype a subset of variants, so the true causal one is frequently
#' absent and represented only by proxies in LD with it.

#+ untyped
header("2b. When the causal variant is not on the array")
sizes2 <- numeric(20)
for (i in 1:20) {
  Li <- simulate_locus(seed = 400 + i)
  keep <- setdiff(seq_len(ncol(Li$G)), Li$causal)     # drop the causal SNP
  Si <- scan_locus(Li$G[, keep], Li$y)
  pp <- abf(Si$b, Si$s); pp <- pp / sum(pp)
  sizes2[i] <- length(credible_set(pp))
}
cat(sprintf("  %-40s%12.1f\n", "mean set size, causal variant TYPED",
            mean(sizes)))
cat(sprintf("  %-40s%12.1f\n", "mean set size, causal variant UNTYPED",
            mean(sizes2)))
cat("\n  With the causal variant absent, the credible set roughly doubles and
  its posterior probability is spread over proxies, none of which is the
  answer. The method cannot tell you this has happened: it reports a set with
  the usual 95% label, and every variant in it is innocent.

  That is the honest reading of a credible set. It is a statement about which
  TYPED variants are compatible with the data, not a guarantee that the causal
  variant is among them.

  Note the assumption behind all of it. Eqs. (43.1)-(43.3) assume EXACTLY ONE
  causal variant in the locus. If two variants act independently, the
  single-causal PIPs are wrong, and methods such as SuSiE that allow several
  are needed. Problem 2 breaks exactly this assumption.\n")

#' ## 3. Colocalisation, eq. (43.4)

#+ coloc
header("3. Do two traits share a causal variant? (43.4)")
#' Posterior over the five hypotheses, eq. (43.4).
coloc <- function(b1, s1, b2, s2, p1 = 1e-4, p2 = 1e-4, p12 = 1e-5) {
  a1 <- abf(b1, s1); a2 <- abf(b2, s2)
  H1 <- p1 * sum(a1)
  H2 <- p2 * sum(a2)
  H3 <- p1 * p2 * (sum(a1) * sum(a2) - sum(a1 * a2))
  H4 <- p12 * sum(a1 * a2)
  v <- c(1, H1, H2, H3, H4)
  v / sum(v)
}
set.seed(4301)
y2_shared <- 0.14 * G[, CAUSAL] + rnorm(n)      # shares the causal variant
other <- CAUSAL + 12
y3_distinct <- 0.14 * G[, other] + rnorm(n)     # different variant, same block
for (z in list(list("shares the causal variant", y2_shared),
               list("different variant, same block", y3_distinct))) {
  S2 <- scan_locus(G, z[[2]])
  post <- coloc(beta_hat, se_hat, S2$b, S2$s)
  cat(sprintf("  %s\n", z[[1]]))
  cat(sprintf("    P(H0 none)=%.3f  P(H1 trait1 only)=%.3f  P(H2 trait2 only)=%.3f\n",
              post[1], post[2], post[3]))
  cat(sprintf("    P(H3 distinct variants)=%.3f  P(H4 SHARED variant)=%.3f\n\n",
              post[4], post[5]))
}
cat("  H4 is high when the traits genuinely share a variant and H3 takes over
  when they do not, which is the intended behaviour.

  The practical hazard is that strong LD makes H3 and H4 hard to separate, and
  a modest posterior for H3 is frequently reported as if it supported sharing.
  Read all five numbers, and treat colocalisation as evidence about a LOCUS,
  not proof of a mechanism.\n")

#' ## 4. Polygenicity or confounding? LDSC, eq. (43.5)-(43.6)
#'
#' $$\mathbb{E}[\chi^2_j]=\frac{N h^2}{M}\ell_j+1+Na \qquad (43.5)$$
#' $$\hat h^2=\hat b_{slope}\,M/N \qquad (43.6)$$

#+ ldsc
header("4. LD score regression separates the two causes of inflation (43.5)")
#' M is the number of causal variants GENOME-WIDE, not the number we simulate.
#' Using the simulated count here is a common slip and inflates chi-squares by
#' orders of magnitude.
M_GENOME <- 1e6
N <- 200000
ldsc_sim <- function(n = N, n_snp = 4000, h2 = 0.35, confound = 0, seed = 0) {
  set.seed(seed)
  ld <- rgamma(n_snp, 3.0, scale = 1.5) + 1.0   # how many variants each tags
  expected <- n * h2 / M_GENOME * ld + 1.0 + confound
  list(chi2 = expected * rchisq(n_snp, 1), ld = ld)
}
cat("  averaged over 30 replicate GWAS, N = 200,000\n\n")
cat(sprintf("  %-34s%11s%12s%11s%11s\n", "scenario", "lambda_GC",
            "LDSC slope", "h2 (43.6)", "intercept"))
for (z in list(list("no signal, no confounding", 0.0, 0.0),
               list("polygenic, no confounding", 0.35, 0.0),
               list("no signal, stratification", 0.0, 0.35),
               list("polygenic AND stratified", 0.35, 0.35))) {
  acc <- t(sapply(1:30, function(i) {
    L2 <- ldsc_sim(h2 = z[[2]], confound = z[[3]], seed = 3000 + i)
    cf <- coef(lm(L2$chi2 ~ L2$ld))
    c(median(L2$chi2) / qchisq(0.5, 1), cf[2], cf[2] * M_GENOME / N, cf[1])
  }))
  m <- colMeans(acc)
  cat(sprintf("  %-34s%11.3f%12.5f%11.3f%11.3f\n", z[[1]], m[1], m[2], m[3],
              m[4]))                                     # eq. (43.6)
}
cat(sprintf("\n  true h2 is 0.35 in the polygenic rows; the true intercept is
  1.00 without stratification and 1.35 with it.\n"))
cat("\n  Read the lambda column first: it is elevated in BOTH the polygenic
  row and the stratified row, and cannot tell them apart. That is the
  limitation Module 34 flagged and could not resolve.

  The LDSC columns resolve it. The SLOPE responds to heritability, because
  real polygenic signal accumulates in proportion to how many variants a SNP
  tags. The INTERCEPT responds to confounding, which inflates every SNP
  equally regardless of its LD score.

  So an intercept near 1 with a large slope is a genuinely polygenic trait; an
  intercept well above 1 is a warning about population structure or cryptic
  relatedness, whatever lambda says.\n")

#' ## 5. Mendelian randomisation with invalid instruments, eq. (43.7)

#+ mr
header("5. IVW, Egger, weighted median and mode (43.7)")
mr_estimators <- function(bx, sx, by, sy) {
  w <- 1 / sy^2
  ivw <- sum(w * bx * by) / sum(w * bx^2)                    # eq. (24.9)
  eg <- coef(lm(by ~ bx, weights = w))      # MR-Egger: WITH an intercept
  ratio <- by / bx                          # per-instrument ratio estimates
  rw <- bx^2 / sy^2
  o <- order(ratio)
  cw <- cumsum(rw[o]) / sum(rw)
  med <- ratio[o][which(cw >= 0.5)[1]]
  grid <- seq(min(ratio), max(ratio), length.out = 512)
  dens <- rowSums(exp(-0.5 * (outer(grid, ratio, "-") / 0.08)^2))
  c(ivw = ivw, egger = eg[2], egger_int = eg[1], median = med,
    mode = grid[which.max(dens)])
}
TRUE_CAUSAL <- 0.30
cat(sprintf("  true causal effect = %.2f\n\n", TRUE_CAUSAL))
cat(sprintf("  %-32s%9s%9s%11s%10s%9s\n", "instruments", "IVW", "Egger",
            "Egger int", "W median", "mode"))
for (z in list(list("all valid", 0, 0.0),
               list("30% invalid, balanced", 9, 0.0),
               list("30% invalid, directional", 9, 0.18))) {
  set.seed(64)
  K <- 30
  bx <- runif(K, 0.10, 0.28)                # instrument-exposure effects
  sx <- rep(0.012, K)
  pleio <- rep(0, K)
  if (z[[2]] > 0) {
    who <- sample(K, z[[2]])
    pleio[who] <- rnorm(z[[2]], z[[3]], 0.12)
  }
  by <- TRUE_CAUSAL * bx + pleio + rnorm(K, 0, 0.012)
  m <- mr_estimators(bx, sx, by, rep(0.012, K))
  cat(sprintf("  %-32s%9.3f%9.3f%11.3f%10.3f%9.3f\n", z[[1]], m[1], m[2],
              m[3], m[4], m[5]))
}
cat("\n  With valid instruments everything agrees, which is the case where the
  choice does not matter.

  With BALANCED pleiotropy, IVW stays roughly unbiased because the invalid
  effects cancel. With DIRECTIONAL pleiotropy it is badly biased, and the
  Egger intercept moves away from zero, which is precisely what it is for.

  The estimators are ordered by the strength of what they assume: IVW needs
  every instrument valid, the weighted median needs over half the weight
  valid, and the mode needs only that the valid ones form the largest cluster.
  Weaker assumptions cost precision, so report all of them. Agreement is the
  evidence; a single estimator is a choice you have not justified.\n")

#' ## 6. Winner's curse, eq. (43.8)
#'
#' $$\mathbb{E}[\hat\beta\mid |\hat\beta/se|>c]>\beta \qquad (43.8)$$

#+ winners-curse
header("6. Effects estimated in the data that selected them (43.8)")
set.seed(71)
M_SNP <- 20000; N_DISC <- 8000
true_b <- rep(0, M_SNP)
causal_idx <- sample(M_SNP, 400)
true_b[causal_idx] <- rnorm(400, 0, 0.035)
se_d <- 1 / sqrt(N_DISC)
b_disc <- true_b + rnorm(M_SNP, 0, se_d)
b_rep <- true_b + rnorm(M_SNP, 0, se_d)        # independent replication
for (tz in c(4.0, 5.0, 5.45)) {
  sel <- abs(b_disc / se_d) > tz
  if (sum(sel) < 5) next
  cat(sprintf("  |z| > %.2f: %4d selected   discovery |b| = %.4f   replication |b| = %.4f   true |b| = %.4f\n",
              tz, sum(sel), mean(abs(b_disc[sel])), mean(abs(b_rep[sel])),
              mean(abs(true_b[sel]))))
}
cat("\n  The replication estimate is consistently smaller than the discovery
  estimate, and the truth is smaller still. Nothing failed to replicate: the
  discovery estimate was inflated by the act of selection (43.8).

  Two consequences worth carrying. Replication studies powered on the
  discovery effect size are systematically underpowered. And polygenic scores
  built with discovery weights over-fit, which is why PRS weights must be
  shrunk or re-estimated in independent data.\n")

#' ## 7. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "statistical_genetics.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
plot(seq_len(n_snp), -log10(p_hat), pch = 16, col = "grey50",
     xlab = "SNP index", ylab = "-log10 p",
     main = "Fine-mapping (43.2)-(43.3)")
points(cs, -log10(p_hat[cs]), pch = 16, col = "steelblue", cex = 1.2)
points(CAUSAL, -log10(p_hat[CAUSAL]), pch = 8, col = "firebrick", cex = 2)
legend("topleft", c("95% set", "causal"), pch = c(16, 8), bty = "n",
       col = c("steelblue", "firebrick"))
L3 <- ldsc_sim(h2 = 0.35, confound = 0.35, seed = 9)
plot(L3$ld, L3$chi2, pch = 16, cex = 0.3, col = adjustcolor("grey40", 0.25),
     xlab = "LD score", ylab = "chi-square",
     main = "LDSC: slope vs intercept (43.5)")
abline(lm(L3$chi2 ~ L3$ld), col = "firebrick", lwd = 2)
abline(h = 1, lty = 2)
sel5 <- abs(b_disc / se_d) > 5.0
plot(abs(b_disc[sel5]), abs(b_rep[sel5]), pch = 16,
     col = adjustcolor("steelblue", 0.6), xlab = "discovery |beta|",
     ylab = "replication |beta|", main = "Winner's curse (43.8)")
abline(0, 1, lty = 2)
abline(h = mean(abs(b_rep[sel5])), col = "firebrick", lty = 3)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "statistical_genetics.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: What makes a credible set small?
#'
#' Vary the LD strength and the sample size, and see which one shrinks the
#' credible set.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %-9s%8s%11s%18s%16s%17s\n", "LD rho", "n", "set size",
#             "contains causal", "PIP of causal", "lead is causal"))
# for (rho in c(0.70, 0.92, 0.98)) {
#   for (n_ in c(3000, 12000)) {
#     sizes <- pips <- numeric(12); hits <- leadok <- 0
#     for (i in 1:12) {
#       Li <- simulate_locus(n = n_, rho = rho, seed = 300 + i)
#       Si <- scan_locus(Li$G, Li$y)
#       pp <- abf(Si$b, Si$s); pp <- pp / sum(pp)
#       cs_ <- credible_set(pp)
#       sizes[i] <- length(cs_); pips[i] <- pp[Li$causal]
#       hits <- hits + (Li$causal %in% cs_)
#       leadok <- leadok + (which.min(Si$p) == Li$causal)
#     }
#     cat(sprintf("  %-9.2f%8d%11.1f%17.0f%%%16.3f%16.0f%%\n", rho, n_,
#                 mean(sizes), 100 * hits / 12, mean(pips), 100 * leadok / 12))
#   }
# }
#
# ## Two levers, and they do different things. More SAMPLES sharpen the PIPs
# ## and shrink the set, because the association signal is estimated more
# ## precisely. Stronger LD widens the set, because more variants are
# ## statistically indistinguishable from the causal one no matter how much
# ## data you have.
# ##
# ## The last column is the point of the exercise: the lead SNP is frequently
# ## not the causal variant, and its being lead is partly luck. A credible set
# ## reports that honestly. This is why fine-mapping results are given as sets
# ## and why functional follow-up should target the set, not the top hit.

#' ### Problem 2: Break the single-causal-variant assumption
#'
#' Put TWO independent causal variants in the locus and see what the
#' single-causal fine-mapping does.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(88)
# idx <- 1:60
# Sg <- 0.92^abs(outer(idx, idx, "-"))
# L2 <- chol(Sg + 1e-8 * diag(60))
# for (z in list(list(30, "two causals, far apart (low LD)"),
#                list(6, "two causals, close (high LD)"))) {
#   c1 <- 15; c2 <- 15 + z[[1]]
#   hits1 <- hits2 <- 0; sizes <- numeric(12)
#   for (i in 1:12) {
#     liab <- matrix(rnorm(6000 * 60), 6000) %*% L2
#     Gx <- matrix(as.numeric(liab > qnorm(0.7)), 6000)
#     yx <- 0.16 * Gx[, c1] + 0.16 * Gx[, c2] + rnorm(6000)
#     Sx <- scan_locus(Gx, yx)
#     pp <- abf(Sx$b, Sx$s); pp <- pp / sum(pp)
#     cs_ <- credible_set(pp)
#     sizes[i] <- length(cs_)
#     hits1 <- hits1 + (c1 %in% cs_); hits2 <- hits2 + (c2 %in% cs_)
#   }
#   cat(sprintf("  %s\n", z[[2]]))
#   cat(sprintf("    mean set size %.1f, contains causal 1 %.0f%%, contains causal 2 %.0f%%\n",
#               mean(sizes), 100 * hits1 / 12, 100 * hits2 / 12))
# }
#
# ## When the two causal variants are far apart in LD, the single-causal model
# ## concentrates on one of them and the credible set frequently MISSES the
# ## other entirely. The 95% claim is about a model with one causal variant,
# ## and that model is false here, so the coverage claim does not apply.
# ##
# ## When they are close, the set tends to cover the region containing both,
# ## so the failure is less visible but the PIPs are still not interpretable
# ## as per-variant probabilities.
# ##
# ## This is why SuSiE and FINEMAP allow multiple credible sets. If you use a
# ## single-causal method, say so, and check whether conditioning on the lead
# ## SNP leaves a second independent signal.

#' ### Problem 3: Winner's curse and replication power
#'
#' Design a replication study powered on the discovery effect size, then
#' measure how often it actually replicates.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# sel <- abs(b_disc / se_d) > 5.0
# b_sel <- abs(b_disc[sel]); b_true_sel <- abs(true_b[sel])
# cat(sprintf("  %d variants selected at |z| > 5\n\n", sum(sel)))
# cat(sprintf("  %-26s%13s%11s%15s\n", "powered on", "assumed |b|",
#             "n needed", "actual power"))
# for (z in list(list("discovery estimate", mean(b_sel)),
#                list("truth (unknowable)", mean(b_true_sel)))) {
#   n_need <- floor(((1.96 + 0.84) / z[[2]])^2)   # 80% power, alpha = 0.05
#   se_r <- 1 / sqrt(n_need)
#   pw <- mean(b_true_sel / se_r > 1.96)   # power against the TRUE effects
#   cat(sprintf("  %-26s%13.4f%11s%14.1f%%\n", z[[1]], z[[2]],
#               format(n_need, big.mark = ","), 100 * pw))
# }
#
# ## Powering the replication on the discovery estimate produces a study that
# ## is far short of 80% power, because the effect it was designed to detect
# ## is larger than the effect that exists. Roughly half the "failures to
# ## replicate" in such a design are failures of the design.
# ##
# ## Two practical responses. Shrink the discovery estimate before powering,
# ## using a winner's-curse correction or simply the lower bound of its
# ## confidence interval. And when a replication fails, compare the two
# ## CONFIDENCE INTERVALS rather than the two p-values: overlapping intervals
# ## with a smaller replication point estimate is the signature of selection,
# ## not of a false original finding.

#' ## What to take away
#'
#' 1. The lead SNP is often not the causal one, and if the causal variant is
#'    untyped it is not in the set at all. Report a **credible set** (43.3), and
#'    state the single-causal assumption it rests on.
#' 2. **Colocalisation** (43.4) has five hypotheses. Read all of them; LD
#'    makes H3 and H4 hard to separate.
#' 3. $\lambda_{GC}$ cannot distinguish polygenicity from confounding. The
#'    LDSC **slope** gives heritability, the **intercept** gives confounding
#'    (43.5).
#' 4. In MR, report IVW, Egger, weighted median and mode together. They assume
#'    progressively less and cost progressively more precision (43.7).
#' 5. **Winner's curse** (43.8) inflates discovery effects, underpowers
#'    replications, and over-fits polygenic scores.
#'
#' **Next:** `44_power_and_design_for_omics.R`
