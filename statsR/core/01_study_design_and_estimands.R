#' ---
#' title: "Module 01 - Statistical thinking: units, estimands, pseudoreplication"
#' output:
#'   html_document:
#'     toc: true
#'     toc_depth: 2
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 1, equations (1.1)-(1.9)
#'
#' ```bash
#' docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 \
#'     Rscript statsR/core/01_study_design_and_estimands.R
#' ```
#'
#' ## Why this module exists
#'
#' Every other module assumes you know what `n` is. This one proves, by
#' simulation, that getting it wrong turns a 5% false-positive rate into a 60%
#' one. If you internalise only one module, make it this one.
#'
#' ## What you will learn
#'
#' 1. The variance decomposition `Var(Ybar) = sigma_b^2/n + sigma_e^2/(nm)` -- eq. (1.5).
#' 2. Why more cells per donor cannot rescue 3 donors -- eq. (1.6).
#' 3. Intraclass correlation and the design effect -- eq. (1.7)-(1.8).
#' 4. Effective sample size -- eq. (1.9).
#' 5. How randomisation turns an unobservable causal contrast into an
#'    observable one -- eq. (1.2)-(1.3).
#' 6. How to detect a design that CANNOT answer the question.

#+ setup, message = FALSE
MODULE_NAME <- "01_study_design_and_estimands"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

header <- function(txt) {
  cat("\n", strrep("=", 72), "\n", txt, "\n", strrep("=", 72), "\n", sep = "")
}

#' ## 1. The hierarchical data-generating process
#'
#' `stats.md` eq. (1.4) describes the two-level structure that essentially
#' every biological experiment has:
#'
#' $$Y_{ij} = \mu + b_i + e_{ij}, \qquad b_i \sim (0,\sigma_b^2),\ e_{ij}\sim(0,\sigma_e^2)$$
#'
#' * `b_i` is the DONOR effect -- biological variation.
#' * `e_ij` is the MEASUREMENT effect -- cells, reads, technical replicates.
#'
#' The distinction determines which term in eq. (1.5) your sample size divides.

#+ simulate-hierarchy
simulate_hierarchy <- function(n_donors, m_per_donor, sigma_b, sigma_e, mu = 0) {
  ## Returns long-form data: one row per measurement.
  donor_effects <- rnorm(n_donors, 0, sigma_b)                  # b_i
  noise <- matrix(rnorm(n_donors * m_per_donor, 0, sigma_e),
                  nrow = n_donors)                              # e_ij
  values <- mu + donor_effects + noise
  data.frame(donor = rep(seq_len(n_donors), each = m_per_donor),
             value = as.vector(t(values)))
}

header("1. One simulated hierarchical dataset")
set.seed(101)
demo <- simulate_hierarchy(5, 200, sigma_b = 1, sigma_e = 1)
print(aggregate(value ~ donor, demo,
                function(v) c(n = length(v), mean = mean(v), sd = sd(v))))
cat("\nThe donor MEANS differ far more than sampling error alone would allow:\n")
cat("that spread is sigma_b, and no number of cells removes it.\n")

#' ## 2. Equation (1.5) verified numerically

#+ verify-eq15
header("2. Verifying the pseudoreplication equation (1.5)")

empirical_var_grand_mean <- function(n, m, sigma_b, sigma_e, n_sim = 4000) {
  means <- replicate(n_sim, {
    b <- rnorm(n, 0, sigma_b)
    e <- matrix(rnorm(n * m, 0, sigma_e), nrow = n)
    mean(b + e)
  })
  var(means)
}

grid <- expand.grid(n = c(3, 30), m = c(1, 10, 100, 1000))
grid <- grid[order(grid$n, grid$m), ]
sb <- 1; se <- 1
grid$theory_eq1.5 <- sb^2 / grid$n + se^2 / (grid$n * grid$m)
grid$naive_wrong  <- (sb^2 + se^2) / (grid$n * grid$m)
grid$simulated <- mapply(function(n, m) {
  set.seed(n * 1000 + m); empirical_var_grand_mean(n, m, sb, se)
}, grid$n, grid$m)
grid$understated_by <- grid$theory_eq1.5 / grid$naive_wrong
print(format(grid, digits = 4), row.names = FALSE)

cat("\n'simulated' tracks 'theory_eq1.5', never the naive formula.\n")
cat("The last column is how many times too SMALL the naive variance is.\n")

#' ### Equation (1.6): the hard floor
#'
#' $$\lim_{m\to\infty}\operatorname{Var}(\bar Y) = \frac{\sigma_b^2}{n}$$

#+ eq16
header("Equation (1.6): the floor you cannot sequence past")
for (m in c(1, 10, 100, 1000, 1e5)) {
  cat(sprintf("  m = %9s  Var(mean) = %.6f   (floor = %.6f)\n",
              format(m, big.mark = ",", scientific = FALSE),
              1/3 + 1/(3 * m), 1/3))
}

#' ## 3. ICC, design effect, and effective sample size
#'
#' $$\rho = \frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2}\ (1.7),\quad
#'   \mathrm{DE} = 1+(m-1)\rho\ (1.8),\quad
#'   n_{\text{eff}} = \frac{nm}{\mathrm{DE}}\ (1.9)$$

#+ icc
header("3. ICC, design effect, effective sample size")

icc <- function(sigma_b, sigma_e) sigma_b^2 / (sigma_b^2 + sigma_e^2)
design_effect <- function(m, rho) 1 + (m - 1) * rho
effective_n <- function(n, m, rho) n * m / design_effect(m, rho)

cat(sprintf("%6s %7s %14s %13s %12s %12s\n",
            "rho", "m", "design effect", "SE inflation", "n_eff (n=3)", "nominal n*m"))
for (rho in c(0.01, 0.02, 0.05, 0.20)) {
  for (m in c(100, 500, 2000)) {
    de <- design_effect(m, rho)
    cat(sprintf("%6.2f %7d %14.2f %13.2f %12.1f %12d\n",
                rho, m, de, sqrt(de), effective_n(3, m, rho), 3 * m))
  }
}
cat(sprintf("\nWorked example: n=3 donors, m=500 cells, rho=0.05\n"))
cat(sprintf("  design effect   = %.2f\n", design_effect(500, 0.05)))
cat(sprintf("  SE too small by = %.2fx\n", sqrt(design_effect(500, 0.05))))
cat(sprintf("  n_eff           = %.1f  (not 1500)\n", effective_n(3, 500, 0.05)))

#' ## 4. The consequence: false-positive rate of a naive cell-level test
#'
#' We simulate a study with NO real effect and test it two ways.

#+ fpr
header("4. False-positive rate: naive cell-level vs donor-level testing")

false_positive_rate <- function(n_per_group, m_per_donor, sigma_b, sigma_e,
                                n_sim = 1500, alpha = 0.05) {
  naive <- donor <- 0L
  for (i in seq_len(n_sim)) {
    ## Both groups come from the SAME distribution: the null is TRUE.
    b <- rnorm(2 * n_per_group, 0, sigma_b)
    e <- matrix(rnorm(2 * n_per_group * m_per_donor, 0, sigma_e),
                nrow = 2 * n_per_group)
    cells <- b + e                                   # donor x cell matrix
    g1 <- as.vector(cells[seq_len(n_per_group), ])
    g2 <- as.vector(cells[-seq_len(n_per_group), ])
    ## (a) NAIVE: every cell treated as an independent observation.
    naive <- naive + (t.test(g1, g2)$p.value < alpha)
    ## (b) CORRECT: aggregate to one value per donor, then test donors.
    dm <- rowMeans(cells)
    donor <- donor + (t.test(dm[seq_len(n_per_group)],
                             dm[-seq_len(n_per_group)])$p.value < alpha)
  }
  c(naive = naive / n_sim, donor = donor / n_sim)
}

set.seed(42)
for (cfg in list(c(sb = 0.5, se = 1, m = 200),
                 c(sb = 1.0, se = 1, m = 200),
                 c(sb = 1.0, se = 1, m = 1000))) {
  rho <- icc(cfg["sb"], cfg["se"])
  r <- false_positive_rate(4, cfg["m"], cfg["sb"], cfg["se"], n_sim = 800)
  cat(sprintf("  sigma_b=%.1f sigma_e=%.1f m=%4d  (rho=%.2f, DE=%6.1f)\n",
              cfg["sb"], cfg["se"], cfg["m"], rho, design_effect(cfg["m"], rho)))
  cat(sprintf("      naive cell-level FPR : %5.1f%%   <-- should be 5%%\n",
              100 * r["naive"]))
  cat(sprintf("      donor-level      FPR : %5.1f%%   <-- is 5%%\n\n",
              100 * r["donor"]))
}
cat("The naive test is not 'slightly liberal'. It is broken.\n")
cat("This is the mechanism behind false-discovery inflation in naive\n")
cat("per-cell single-cell differential expression (stats.md Topic 21).\n")

#' ## 5. Estimands: what randomisation actually buys you
#'
#' Eq. (1.1): we only ever see one potential outcome per unit.
#' Eq. (1.2): the causal estimand is `tau = E[Y(1) - Y(0)]`.
#' Eq. (1.3): randomisation makes the observable group contrast equal it.

#+ estimand
header("5. ATE, randomisation, and confounding")
set.seed(2026)
N <- 20000
severity <- rnorm(N)                 # a confounder
TRUE_ATE <- 2.0
y0 <- 10 + 3 * severity + rnorm(N)   # both potential outcomes, known here
y1 <- y0 + TRUE_ATE

## Design A: randomised -- treatment independent of severity.
a_rand <- rbinom(N, 1, 0.5)
y_rand <- ifelse(a_rand == 1, y1, y0)                       # eq. (1.1)
est_rand <- mean(y_rand[a_rand == 1]) - mean(y_rand[a_rand == 0])

## Design B: observational -- sicker patients more likely to be treated.
p_treat <- plogis(1.5 * severity)
a_obs <- rbinom(N, 1, p_treat)
y_obs <- ifelse(a_obs == 1, y1, y0)
est_obs_crude <- mean(y_obs[a_obs == 1]) - mean(y_obs[a_obs == 0])
est_obs_adj <- coef(lm(y_obs ~ a_obs + severity))["a_obs"]

cat(sprintf("  TRUE ATE (known by construction)     : %.3f\n", TRUE_ATE))
cat(sprintf("  Randomised design, crude difference  : %.3f   <-- unbiased\n", est_rand))
cat(sprintf("  Observational, crude difference      : %.3f   <-- biased\n", est_obs_crude))
cat(sprintf("  Observational, adjusted for severity : %.3f   <-- recovered\n", est_obs_adj))
cat("\nEq. (1.3) holds ONLY under randomisation. Adjustment worked here only\n")
cat("because we happened to MEASURE the confounder - see Module 32.\n")

#' ## 6. Designs that cannot answer the question
#'
#' Cross-tabulate the factor of interest against batch. If the table is
#' block-diagonal the effects are ALIASED: the design matrix is rank-deficient
#' (eq. 16.5 / 19.3) and no software can separate them.

#+ identifiability
header("6. Detecting a confounded (rank-deficient) design")

design_is_identifiable <- function(condition, batch, label = "") {
  condition <- factor(condition); batch <- factor(batch)
  X <- model.matrix(~ condition + batch)
  r <- qr(X)$rank
  cat(label, "\n")
  print(table(condition, batch))
  cat(sprintf("    design matrix: %d columns, rank %d  ->  %s\n\n",
              ncol(X), r,
              if (r == ncol(X)) "IDENTIFIABLE" else "CONFOUNDED (aliased)"))
  ## R silently returns NA coefficients for aliased terms - a warning sign
  ## that is very easy to miss in a long summary() printout.
  invisible(r == ncol(X))
}

design_is_identifiable(rep(c("ctrl", "trt"), 6), rep(c("b1", "b2", "b3"), each = 4),
                       "Design A - conditions balanced across batches (good):")
design_is_identifiable(rep(c("ctrl", "trt"), each = 6), rep(c("b1", "b2"), each = 6),
                       "Design B - every control in batch 1, treated in batch 2 (fatal):")

cat("Watch what lm() does with the confounded design:\n")
set.seed(7)
bad <- data.frame(condition = rep(c("ctrl", "trt"), each = 6),
                  batch = rep(c("b1", "b2"), each = 6),
                  y = rnorm(12))
print(coef(lm(y ~ condition + batch, data = bad)))
cat("\nThe batch coefficient is NA: R is telling you the effect is not\n")
cat("estimable. Never ignore an NA coefficient.\n")

#' ## 7. Figure

#+ figure, fig.width = 11, fig.height = 4.5
png(file.path(OUT, "pseudoreplication.png"), width = 1100, height = 450, res = 110)
par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.5, 1))

ms <- 10^seq(0, 4, length.out = 60)
plot(ms, 1/3 + 1/(3 * ms), log = "xy", type = "l", col = "steelblue", lwd = 2,
     xlab = "measurements per donor, m", ylab = "Var(grand mean)",
     main = "Eq. (1.5)-(1.6): the floor is sigma_b^2/n")
abline(h = 1/3, col = "steelblue", lty = 3)
lines(ms, 1/10 + 1/(10 * ms), col = "darkorange", lwd = 2)
abline(h = 1/10, col = "darkorange", lty = 3)
lines(ms, 2/(3 * ms), col = "black", lty = 2)
legend("bottomleft", c("eq.(1.5), n=3", "eq.(1.5), n=10", "naive, n=3 (wrong)"),
       col = c("steelblue", "darkorange", "black"), lty = c(1, 1, 2), bty = "n", cex = 0.7)

plot(NA, xlim = range(ms), ylim = c(1, 1000), log = "xy",
     xlab = "measurements per donor, m", ylab = "design effect  1+(m-1)rho",
     main = "Eq. (1.8): variance inflation from clustering")
cols <- c("steelblue", "darkorange", "forestgreen", "firebrick", "purple")
rhos <- c(0.01, 0.02, 0.05, 0.10, 0.20)
for (i in seq_along(rhos)) lines(ms, 1 + (ms - 1) * rhos[i], col = cols[i], lwd = 2)
legend("topleft", paste0("rho=", rhos), col = cols, lty = 1, bty = "n", cex = 0.7)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "pseudoreplication.png"), "\n")

#' ## Checklist to run before every analysis
#'
#' 1. **Unit.** What was independently randomised or sampled? That is `n`.
#' 2. **Estimand.** Write it as an equation before opening the data.
#' 3. **Structure.** Paired? Blocked? Nested? If designed in, model it in.
#' 4. **Identifiability.** `table(condition, batch)` and check `qr(X)$rank`.
#' 5. **ICC.** Estimate rho and compute the design effect before trusting any
#'    cell-level p-value.
#'
#' ## Self-check
#'
#' * 50,000 cells from 3 patients per arm. What is `n`? *(3 per arm.)*
#' * ICC = 0.03 with 800 cells/donor: how wrong are naive SEs?
#'   *(`sqrt(1 + 799*0.03)` = 5.0x too small.)*
#'
#' **Next:** `02_data_structures_and_scales.R`
