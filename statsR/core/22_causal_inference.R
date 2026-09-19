#' ---
#' title: "Module 22 - Causal inference for observational bioinformatics"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 32, equations (32.1)-(32.8)
#'
#' Adjusting for more covariates feels safer. **It is not.**
#'
#' ## What you will learn
#'
#' 1. Confounder vs mediator vs collider -- each with known truth.
#' 2. **Collider bias** and how sample selection creates it.
#' 3. The g-formula (32.2), propensity scores and IPTW (32.3)-(32.4).
#' 4. Overlap and balance diagnostics (NOT p-values).
#' 5. Doubly robust AIPW (32.5), mediation (32.6)-(32.7), the E-value (32.8).

#+ setup, message = FALSE
MODULE_NAME <- "22_causal_inference"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

#' ## 1. The three elementary structures
#'
#' | Structure | Path | Adjusting for M... |
#' |---|---|---|
#' | Chain (mediator) | A -> M -> Y | **blocks** the indirect effect |
#' | Fork (confounder) | A <- C -> Y | **removes** confounding |
#' | Collider | A -> K <- Y | **creates** spurious association |

#+ structures
header("1. Confounder, mediator, collider: adjust or not?")
set.seed(2201)
N <- 40000
## FORK: C confounds A -> Y. TRUE direct effect of A is 0.8.
Cc <- rnorm(N); A_f <- rbinom(N, 1, plogis(0.9*Cc))
Y_f <- 0.8*A_f + 1.2*Cc + rnorm(N)
## CHAIN: A -> M -> Y. TOTAL effect 0.3 + 0.5*1.0 = 0.8, DIRECT 0.3.
A_c <- rbinom(N, 1, 0.5); M <- 1.0*A_c + rnorm(N); Y_c <- 0.3*A_c + 0.5*M + rnorm(N)
## COLLIDER: A -> K <- Y. TRUE effect of A on Y is EXACTLY 0.
A_k <- rbinom(N, 1, 0.5); Y_k <- rnorm(N); K <- 1.2*A_k + 1.2*Y_k + rnorm(N)

cat(sprintf("  %-26s%8s%10s%11s  verdict\n", "structure", "truth", "crude", "adjusted"))
cat(sprintf("  %-26s%8.2f%10.3f%11.3f  ADJUST\n", "FORK (C = confounder)", 0.80,
            coef(lm(Y_f ~ A_f))[2], coef(lm(Y_f ~ A_f + Cc))[2]))
cat(sprintf("  %-26s%8.2f%10.3f%11.3f  DO NOT adjust for a total effect\n",
            "CHAIN (M = mediator)", 0.80, coef(lm(Y_c ~ A_c))[2],
            coef(lm(Y_c ~ A_c + M))[2]))
cat("     (total = 0.3 + 0.5*1.0 = 0.8; adjusting returns the DIRECT 0.3)\n")
cat(sprintf("  %-26s%8.2f%10.3f%11.3f  NEVER adjust\n", "COLLIDER (K)", 0.00,
            coef(lm(Y_k ~ A_k))[2], coef(lm(Y_k ~ A_k + K))[2]))
cat("\n  Adjusting for the collider MANUFACTURES a strong association where the\n")
cat("  truth is exactly zero. 'Adjust for everything you measured' is not\n")
cat("  caution - it is a way to generate false findings.\n")

#' ## 2. Collider bias via sample selection

#+ selection
header("2. Selection on a collider (the bioinformatics version)")
set.seed(2202)
N <- 30000
expr <- rnorm(N); pheno <- rnorm(N)         # INDEPENDENT by construction
## Sample quality depends on BOTH (both raise RNA yield, so both raise the
## chance the sample passes QC). QC is therefore a COLLIDER.
quality <- 0.9*expr + 0.9*pheno + rnorm(N)
cat(sprintf("  TRUE correlation(expression, phenotype) = %+.4f  (independent)\n",
            cor(expr, pheno)))
cat(sprintf("\n  %22s%24s\n", "QC stringency (kept)", "observed correlation"))
for (keep in c(1, 0.8, 0.6, 0.4, 0.2)) {
  sel <- quality > quantile(quality, 1 - keep)
  cat(sprintf("  %21.0f%%%24.4f\n", 100*keep, cor(expr[sel], pheno[sel])))
}
cat("\n  The stricter the filter, the stronger the FAKE association. Draw your\n")
cat("  QC and selection steps INTO the causal graph - they are causal\n")
cat("  structures, not neutral preprocessing.\n")

#' ## 3. The g-formula, propensity scores and IPTW, eq. (32.2)-(32.5)

#+ gformula
header("3. Standardisation, propensity scores, IPTW, AIPW (32.2)-(32.5)")
make_observational <- function(n = 6000, true_ate = 1.5, overlap = 1) {
  Z1 <- rnorm(n); Z2 <- rbinom(n, 1, 0.4)
  ps <- plogis(overlap * (1.3*Z1 + 1.1*Z2 - 0.6))
  A <- rbinom(n, 1, ps)
  Y <- 2 + true_ate*A + 1.8*Z1 + 1.4*Z2 + rnorm(n)
  data.frame(Y = Y, A = A, Z1 = Z1, Z2 = Z2)
}
set.seed(2203)
TRUE_ATE <- 1.5
d <- make_observational(true_ate = TRUE_ATE)
crude <- mean(d$Y[d$A == 1]) - mean(d$Y[d$A == 0])
reg <- coef(lm(Y ~ A + Z1 + Z2, data = d))["A"]
## g-formula (32.2): average the CONDITIONAL contrast over the covariate
## distribution of the WHOLE sample - not the same as reading a coefficient.
m_out <- lm(Y ~ A * (Z1 + Z2), data = d)
d1 <- transform(d, A = 1); d0 <- transform(d, A = 0)
mu1 <- predict(m_out, d1); mu0 <- predict(m_out, d0)
gform <- mean(mu1 - mu0)
## IPTW (32.3)-(32.4)
ps_model <- glm(A ~ Z1 + Z2, data = d, family = binomial())
e <- fitted(ps_model)
w <- d$A/e + (1 - d$A)/(1 - e)
iptw <- sum(w*d$A*d$Y)/sum(w*d$A) - sum(w*(1-d$A)*d$Y)/sum(w*(1-d$A))
## AIPW (32.5)
aipw <- mean(mu1 - mu0 + d$A*(d$Y - mu1)/e - (1 - d$A)*(d$Y - mu0)/(1 - e))
cat(sprintf("  TRUE ATE = %.1f\n", TRUE_ATE))
cat(sprintf("  %-40s%10s%9s\n", "estimator", "estimate", "bias"))
for (row_ in list(list("crude difference of means", crude),
                  list("regression coefficient", reg),
                  list("g-formula / standardisation (32.2)", gform),
                  list("IPTW (32.4)", iptw),
                  list("AIPW, doubly robust (32.5)", aipw)))
  cat(sprintf("  %-40s%10.4f%9.4f\n", row_[[1]], row_[[2]], row_[[2]] - TRUE_ATE))

cat("\n  'Doubly robust' demonstrated - deliberately MISSPECIFY one model:\n")
m_bad <- lm(Y ~ A, data = d)
mu1b <- predict(m_bad, d1); mu0b <- predict(m_bad, d0)
aipw_bad_out <- mean(mu1b - mu0b + d$A*(d$Y - mu1b)/e - (1-d$A)*(d$Y - mu0b)/(1-e))
ps_bad <- fitted(glm(A ~ 1, data = d, family = binomial()))
aipw_bad_ps <- mean(mu1 - mu0 + d$A*(d$Y - mu1)/ps_bad - (1-d$A)*(d$Y - mu0)/(1-ps_bad))
cat(sprintf("    AIPW with WRONG outcome model, right PS : %.4f\n", aipw_bad_out))
cat(sprintf("    AIPW with right outcome model, WRONG PS : %.4f\n", aipw_bad_ps))
cat(sprintf("    (both close to %.1f; you need only ONE of the two correct)\n", TRUE_ATE))

#' ## 4. Diagnose overlap and balance -- not with p-values

#+ overlap-balance
header("4. Positivity (overlap) and covariate balance")
smd <- function(x, a, weights = NULL) {                 # standardised mean diff
  if (is.null(weights)) weights <- rep(1, length(x))
  m1 <- weighted.mean(x[a == 1], weights[a == 1]); m0 <- weighted.mean(x[a == 0], weights[a == 0])
  v1 <- weighted.mean((x[a==1] - m1)^2, weights[a==1])
  v0 <- weighted.mean((x[a==0] - m0)^2, weights[a==0])
  (m1 - m0)/sqrt((v1 + v0)/2)
}
cat(sprintf("  %-12s%13s%17s  target |SMD| < 0.1\n", "covariate", "SMD before",
            "SMD after IPTW"))
for (col in c("Z1", "Z2"))
  cat(sprintf("  %-12s%13.3f%17.3f\n", col, smd(d[[col]], d$A), smd(d[[col]], d$A, w)))
cat("\n  Overlap (positivity). Propensity range by arm:\n")
for (a in 0:1) {
  pe <- e[d$A == a]
  cat(sprintf("    arm %d: min %.4f, 1st pct %.4f, 99th pct %.4f, max %.4f\n",
              a, min(pe), quantile(pe, 0.01), quantile(pe, 0.99), max(pe)))
}
cat(sprintf("    largest IPTW weight = %.1f  (weights > ~10 are a warning)\n", max(w)))
cat("\n  Now the SAME analysis with POOR overlap (strong confounding):\n")
for (ov in c(1, 2.5, 4)) {
  set.seed(2204)
  dd <- make_observational(overlap = ov)
  ee <- fitted(glm(A ~ Z1 + Z2, data = dd, family = binomial()))
  ww <- dd$A/ee + (1 - dd$A)/(1 - ee)
  est <- sum(ww*dd$A*dd$Y)/sum(ww*dd$A) - sum(ww*(1-dd$A)*dd$Y)/sum(ww*(1-dd$A))
  cat(sprintf("    overlap scale %.1f: min(e)=%.5f, max weight=%8.1f, ATE=%.3f (bias %+.3f)\n",
              ov, min(ee), max(ww), est, est - TRUE_ATE))
}
cat("  Propensities near 0 or 1 produce enormous weights and unstable\n")
cat("  estimates. That is a POSITIVITY VIOLATION - trimming or restricting to\n")
cat("  the region of common support is more honest than extrapolating.\n")
cat("\n  NOTE: balance is judged by SMD, NOT by p-values. A p-value for balance\n")
cat("  just measures your sample size, not the size of the imbalance.\n")

#' ## 5. Mediation, eq. (32.6)-(32.7)

#+ mediation
header("5. Decomposing a total effect (32.6)-(32.7)")
set.seed(2205)
n <- 20000
A <- rbinom(n, 1, 0.5)
Mcell <- 0.9*A + rnorm(n)                  # exposure -> cell composition
Ym <- 0.4*A + 0.7*Mcell + rnorm(n)         # composition -> methylation
TOTAL <- 0.4 + 0.7*0.9
m_tot <- lm(Ym ~ A); m_med <- lm(Mcell ~ A); m_out2 <- lm(Ym ~ A + Mcell)
nde <- coef(m_out2)["A"]
nie <- coef(m_med)["A"] * coef(m_out2)["Mcell"]
cat(sprintf("  TRUE total effect  = %.3f\n", TOTAL))
cat(sprintf("  estimated total    = %.4f\n", coef(m_tot)["A"]))
cat(sprintf("  direct effect NDE  = %.4f   (truth 0.400)\n", nde))
cat(sprintf("  indirect NIE       = %.4f   (truth %.3f)\n", nie, 0.7*0.9))
cat(sprintf("  NDE + NIE          = %.4f  <- eq. (32.6)\n", nde + nie))
cat(sprintf("  proportion mediated= %.1f%%\n", 100*nie/(nde + nie)))
cat("\n  This is the correct framing for 'is the methylation effect mediated by\n")
cat("  cell composition?' (Module 33). But the product-of-coefficients\n")
cat("  approach assumes NO exposure-mediator interaction AND no unmeasured\n")
cat("  mediator-outcome confounding - which randomising A does NOT deliver.\n")

#' ## 6. Sensitivity: the E-value, eq. (32.8)

#+ evalue
header("6. How strong would an unmeasured confounder have to be? (32.8)")
e_value <- function(rr) { rr <- max(rr, 1/rr); rr + sqrt(rr*(rr - 1)) }
cat(sprintf("  %13s%10s  interpretation\n", "observed RR", "E-value"))
for (rr in c(1.1, 1.25, 1.5, 2, 3, 5)) {
  ev <- e_value(rr)
  note <- if (ev < 1.7) "trivially explained away" else
    if (ev < 2.2) "easily explained away" else
      if (ev < 4) "would need a strong unmeasured confounder" else
        "robust to plausible confounding"
  cat(sprintf("  %13.2f%10.2f  %s\n", rr, ev, note))
}
cat("\n  The E-value is the minimum association (on the RR scale) an unmeasured\n")
cat("  confounder would need with BOTH exposure and outcome to fully explain\n")
cat("  the observed effect. Reporting it converts 'residual confounding is\n")
cat("  possible' into a quantitative claim readers can judge. Compare it to\n")
cat("  the strength of your MEASURED confounders for context.\n")

#' ## 7. Figure

#+ figure
png(file.path(OUT, "causal.png"), width = 1300, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.5, 1))
sel <- quality > quantile(quality, 0.6)
plot(expr[!sel][1:3000], pheno[!sel][1:3000], pch = ".", col = "grey60",
     xlab = "expression", ylab = "phenotype",
     main = "Selection on a collider creates correlation")
points(expr[sel][1:3000], pheno[sel][1:3000], pch = ".", col = "firebrick")
legend("topleft", c("failed QC", "passed QC"), col = c("grey60", "firebrick"),
       pch = 16, bty = "n", cex = 0.7)
hist(e[d$A == 0], breaks = 40, freq = FALSE, col = adjustcolor("steelblue", 0.6),
     border = "white", xlab = "propensity score e(Z)",
     main = "Eq. (32.3): overlap check")
hist(e[d$A == 1], breaks = 40, freq = FALSE, col = adjustcolor("darkorange", 0.6),
     border = "white", add = TRUE)
legend("topright", c("untreated", "treated"),
       fill = c(adjustcolor("steelblue",0.6), adjustcolor("darkorange",0.6)),
       bty = "n", cex = 0.7)
rrs <- seq(1.01, 6, length.out = 200)
plot(rrs, sapply(rrs, e_value), type = "l", lwd = 2, col = "steelblue",
     xlab = "observed risk ratio", ylab = "E-value",
     main = "Eq. (32.8): sensitivity to confounding")
abline(0, 1, lty = 2)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "causal.png"), "\n")

#' ## Decision rules (from `stats.md` Topic 32)
#'
#' 1. **Draw the DAG first.** Adjustment sets come from the graph, not from
#'    "everything available".
#' 2. Never adjust for a mediator when estimating a total effect; never adjust
#'    for or select on a collider.
#' 3. Check positivity and covariate balance (SMD), not model fit.
#' 4. State the untestable assumptions and quantify sensitivity (32.8).
#' 5. Batch, selection and technical filtering are causal structures too.
#'
#' **Next:** `23_bayesian_and_shrinkage.R`
