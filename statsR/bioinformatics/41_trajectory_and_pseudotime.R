#' ---
#' title: "Applied 41 - Trajectory inference and pseudotime"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 41, equations (41.1)-(41.3)
#' **Core modules used:** 17, 18, 11, 31
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | Differentiation time course profiled by scRNA-seq, analysed with Slingshot or Monocle |
#' | **Input** | Cells x genes, reduced to a few dimensions |
#' | **Key threats** | Pseudotime is an ordering, not time; **double dipping**; unstable branches |
#' | **Here** | Simulated cells along a known trajectory, plus a null with no trajectory at all |
#'
#' ## The single most important fact
#'
#' The pseudotime you test genes against was **computed from those genes**.
#' That is the clustering problem of Module 18 in a continuous disguise, and
#' it produces significant genes on data with no trajectory whatsoever.

#+ setup, message = FALSE
suppressPackageStartupMessages(library(splines))
MODULE_NAME <- "41_trajectory_and_pseudotime"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(41)

#' ## 1. Cells along a known trajectory

#+ simulate
header("1. A differentiation trajectory with a known ordering")
#' Cells sampled along a latent progression. When has_trajectory is FALSE the
#' latent variable is removed and only noise remains.
simulate_trajectory <- function(n_cells = 1200, n_genes = 400, n_dynamic = 80,
                                has_trajectory = TRUE, seed = 0) {
  set.seed(seed)
  t_true <- runif(n_cells)
  base <- exp(rnorm(n_genes, 2.5, 0.9))
  log_mu <- matrix(log(base), n_cells, n_genes, byrow = TRUE)
  dyn <- rep(FALSE, n_genes)
  if (has_trajectory) {
    dyn[seq_len(n_dynamic)] <- TRUE
    peak <- runif(n_dynamic)                  # each gene peaks somewhere
    width <- runif(n_dynamic, 0.12, 0.35)
    amp <- runif(n_dynamic, 1.0, 2.2)
    shape <- outer(t_true, peak, function(a, b) a - b)
    shape <- sweep(shape, 2, width, "/")
    log_mu[, seq_len(n_dynamic)] <- log_mu[, seq_len(n_dynamic)] +
      sweep(exp(-0.5 * shape^2), 2, amp, "*")
  }
  size <- rlnorm(n_cells, 0, 0.35)
  mu <- exp(log_mu) * size
  list(counts = matrix(rpois(length(mu), mu), n_cells), t_true = t_true,
       dyn = dyn, size = size)
}
S <- simulate_trajectory(seed = 1)
counts <- S$counts; t_true <- S$t_true; is_dynamic <- S$dyn; size <- S$size
n_cells <- nrow(counts); n_genes <- ncol(counts)
cat(sprintf("  %d cells x %d genes, %d truly dynamic\n", n_cells, n_genes,
            sum(is_dynamic)))
cat(sprintf("  library size varies %.2f to %.2f fold\n", min(size), max(size)))
cat("\n  The latent t is a POSITION along a progression. It has no units and
  no direction that the data can identify: reversing it fits equally well.\n")

#' ## 2. Inferring pseudotime, eq. (41.1)
#'
#' $$\min_{f}\sum_i\|z_i-f(\lambda_i)\|^2,\qquad
#'   \lambda_i=\arg\min_\lambda\|z_i-f(\lambda)\| \qquad (41.1)$$
#'
#' Reduce, fit a curve, project. This is what Slingshot and Monocle do, with
#' more care about branches.

#+ principal-curve
header("2. A principal curve and the ordering it induces (41.1)")
normalise <- function(counts) log1p(counts / rowSums(counts) * 1e4)
#' Iterate: project points onto the curve, then re-smooth the curve through
#' the ordered points. This is eq. (41.1) solved by alternation.
principal_curve <- function(Z, n_iter = 12, n_knots = 24) {
  lam <- Z[, 1]
  for (it in seq_len(n_iter)) {
    grid <- seq(min(lam), max(lam), length.out = n_knots)
    bw <- 0.12 * diff(range(lam))
    W <- exp(-0.5 * outer(lam, grid, "-")^2 / bw^2)
    curve <- (t(W) %*% Z) / colSums(W)
    d2 <- outer(rowSums(Z^2), rowSums(curve^2), "+") - 2 * Z %*% t(curve)
    lam <- grid[max.col(-d2, ties.method = "first")]
  }
  list(lam = (lam - min(lam)) / diff(range(lam)), curve = curve)
}
embed <- function(counts, k = 3) {
  Y <- normalise(counts)
  Yc <- sweep(Y, 2, colMeans(Y))
  sv <- svd(Yc, nu = k, nv = 0)
  sv$u %*% diag(sv$d[seq_len(k)])
}
Z <- embed(counts)
pc <- principal_curve(Z); pt <- pc$lam; curve <- pc$curve
r_sp <- cor(pt, t_true, method = "spearman")
if (r_sp < 0) { pt <- 1 - pt; r_sp <- -r_sp }
cat(sprintf("  Spearman(pseudotime, true t) = %.3f\n", r_sp))
cat(sprintf("  Spearman after flipping direction would be %.3f\n", -r_sp))
cat("\n  The ordering is recovered well. The DIRECTION is not: the data cannot
  say which end is the start, and every trajectory tool asks you to supply a
  root cell or a marker. That choice is biology, not inference.

  Equal steps in pseudotime are also not equal steps in real time. Cells
  accumulate where transitions are slow, so density along the curve reflects
  dwell time, not sampling time.\n")

#' ## 3. Testing genes along the trajectory, eq. (41.2)
#'
#' $$\log \mathbb{E}[Y_{ig}]=\log s_i+\sum_k\beta_{gk}B_k(\lambda_i)
#'   \qquad (41.2)$$

#+ gene-test
header("3. A spline GLM along pseudotime (41.2)")
#' Poisson with a spline in pseudotime. A real analysis would fit the negative
#' binomial dispersion as well; the double-dipping lesson below is unaffected.
trajectory_test <- function(counts, pseudotime, size, df = 5, genes) {
  B <- bs(pseudotime, df = df)
  off <- log(size)
  sapply(genes, function(g) {
    y <- counts[, g]
    out <- try({
      f1 <- glm(y ~ B, family = poisson(), offset = off)
      f0 <- glm(y ~ 1, family = poisson(), offset = off)
      pchisq(max(f0$deviance - f1$deviance, 0), df, lower.tail = FALSE)
    }, silent = TRUE)
    if (inherits(out, "try-error")) 1 else out
  })
}
SUB <- seq(1, n_genes, by = 2)          # half the genes, for speed
p_real <- trajectory_test(counts, pt, size, genes = SUB)
rej <- p.adjust(p_real, "BH") < 0.05
tr <- is_dynamic[SUB]
cat(sprintf("  tested %d genes against the INFERRED pseudotime\n", length(SUB)))
cat(sprintf("    discoveries : %d\n", sum(rej)))
cat(sprintf("    truly dynamic among them : %d\n", sum(rej & tr)))
cat(sprintf("    FDP : %.3f\n", sum(rej & !tr) / max(sum(rej), 1)))
cat(sprintf("    power : %.3f\n", sum(rej & tr) / max(sum(tr), 1)))
cat("\n  Good sensitivity, and the FDP looks acceptable. Now run exactly the
  same pipeline on data with no trajectory in it.\n")

#' ## 4. The double dipping problem, eq. (41.3)

#+ double-dipping
header("4. The same analysis on data with NO trajectory")
S0 <- simulate_trajectory(has_trajectory = FALSE, seed = 7)
counts0 <- S0$counts; size0 <- S0$size
pt0 <- principal_curve(embed(counts0))$lam
p_null <- trajectory_test(counts0, pt0, size0, genes = SUB)
rej0 <- p.adjust(p_null, "BH") < 0.05
cat("  There is NO latent trajectory. Every gene is independent noise.\n\n")
cat(sprintf("  %-44s%10s\n", "quantity", "value"))
cat(sprintf("  %-44s%10d\n", "genes called dynamic at FDR 5%", sum(rej0)))
cat(sprintf("  %-44s%10s\n", sprintf("of %d tested", length(SUB)), ""))
cat(sprintf("  %-44s%10.3f\n", "raw p < 0.05", mean(p_null < 0.05)))
cat(sprintf("  %-44s%10s\n", "(should be 0.05)", ""))
# Compare with a pseudotime that was NOT derived from these genes.
pt_indep <- sample(pt0)
p_indep <- trajectory_test(counts0, pt_indep, size0, genes = SUB)
cat("\n  Same genes, but ordered by a pseudotime unrelated to them:\n")
cat(sprintf("  %-44s%10.3f\n", "raw p < 0.05", mean(p_indep < 0.05)))
cat(sprintf("  %-44s%10d\n", "genes called dynamic",
            sum(p.adjust(p_indep, "BH") < 0.05)))
cat("\n  The curve was fitted to whatever structure the noise happened to
  contain, and then the genes were tested for agreement with the structure
  they themselves created. The p-values are not uniform and the discoveries
  are entirely fictitious.

  Break the circularity, as in the last block, and the test behaves.

  This is eq. (41.3), and it is the same error as testing the genes that
  defined a cluster (Module 18). It is easier to miss here because pseudotime
  feels like an external covariate, like age or dose. It is not: it is a
  function of the expression matrix.\n")

#' ## 5. A defence that works: split the genes

#+ split
header("5. Building the trajectory on one half, testing the other")
split_gene_test <- function(counts, size, seed = 0) {
  set.seed(seed)
  g <- sample(ncol(counts))
  half <- ncol(counts) %/% 2
  build <- g[seq_len(half)]; test <- g[(half + 1):ncol(counts)]
  ptb <- principal_curve(embed(counts[, build]))$lam
  sub <- test[seq_len(length(test) %/% 2)]
  list(p = trajectory_test(counts, ptb, size, genes = sub), sub = sub)
}
cat(sprintf("  %-26s%-34s%13s\n", "data", "analysis", "discoveries"))
s0 <- split_gene_test(counts0, size0, seed = 3)
cat(sprintf("  %-26s%-34s%13d\n", "NO trajectory",
            "naive (same genes build and test)", sum(rej0)))
cat(sprintf("  %-26s%-34s%13d\n", "NO trajectory",
            "split: build on other genes", sum(p.adjust(s0$p, "BH") < 0.05)))
s1 <- split_gene_test(counts, size, seed = 3)
rj1 <- p.adjust(s1$p, "BH") < 0.05
cat(sprintf("  %-26s%-34s%13d\n", "real trajectory",
            "split: build on other genes", sum(rj1)))
cat(sprintf("\n  On real data the split analysis still recovers %d of %d dynamic
  genes (FDP %.3f).\n", sum(rj1 & is_dynamic[s1$sub]), sum(is_dynamic[s1$sub]),
            sum(rj1 & !is_dynamic[s1$sub]) / max(sum(rj1), 1)))
cat("\n  Splitting genes costs power, because the trajectory is built from
  half the information. It buys a test that says nothing when there is
  nothing.

  Other defences: build the trajectory on one set of CELLS and project the
  rest; or use marker genes chosen a priori to define the ordering. The
  principle is always the same, the ordering must not be a function of the
  quantities you test.\n")

#' ## 6. The ordering is uncertain, and the uncertainty is rarely propagated

#+ bootstrap
header("6. Bootstrap the cells and watch the ordering move")
NB <- 12
boots <- numeric(NB)
ranks <- matrix(NA_real_, NB, n_cells)
for (b in seq_len(NB)) {
  set.seed(500 + b)
  idx <- sample(n_cells, n_cells, replace = TRUE)
  ptb <- principal_curve(embed(counts[idx, ]))$lam
  if (cor(ptb, t_true[idx], method = "spearman") < 0) ptb <- 1 - ptb
  boots[b] <- cor(ptb, t_true[idx], method = "spearman")
  for (k in seq_along(idx)) ranks[b, idx[k]] <- ptb[k]
}
sd_cell <- apply(ranks, 2, sd, na.rm = TRUE)
early <- t_true < 0.2
mid <- t_true > 0.4 & t_true < 0.6
late <- t_true > 0.8
cat(sprintf("  Spearman with truth across %d bootstraps: %.3f (SD %.3f)\n\n",
            NB, mean(boots), sd(boots)))
cat(sprintf("  %-22s%28s\n", "cells", "SD of assigned pseudotime"))
cat(sprintf("  %-22s%28.4f\n", "early (t < 0.2)", mean(sd_cell[early], na.rm = TRUE)))
cat(sprintf("  %-22s%28.4f\n", "middle (0.4-0.6)", mean(sd_cell[mid], na.rm = TRUE)))
cat(sprintf("  %-22s%28.4f\n", "late (t > 0.8)", mean(sd_cell[late], na.rm = TRUE)))
cat("\n  The global ordering is essentially fixed (Spearman SD 0.000), yet individual
  cells still move between runs, and they do not all move by the same amount.

  Note that the MIDDLE is the least stable band here, not the ends. That is
  worth pausing on, because the usual intuition says the opposite. It is an
  artefact of the estimator: cells are projected onto a fixed grid of knots
  along the curve, and a cell at either extreme is pinned against the end of
  that grid, so its position cannot vary much in one direction. A cell in the
  middle is free to shift either way.

  The lesson is not \"middles are unstable\". It is that the uncertainty
  structure belongs to the ALGORITHM as much as to the data, so measure it on
  your own pipeline rather than assuming the textbook picture.

  Almost no published analysis propagates any of this into the gene-level
  p-values, which are computed as though pseudotime were measured without
  error. That is a measurement-error problem (Module 26), and it biases the
  estimated shapes toward flatness.\n")

#' ## 7. RNA velocity, eq. (41.3)
#'
#' $$\frac{du}{dt}=\alpha-\beta u,\qquad
#'   \frac{ds}{dt}=\beta u-\gamma s \qquad (41.3)$$

#+ velocity
header("7. Velocity assumes constant kinetics across cells")
cat("  The steady-state estimator reads gamma off the slope of spliced on
  unspliced among the extreme cells, then calls a cell 'increasing' when
  s - gamma*u > 0.\n")
for (z in list(list("same kinetics in both states", 1.0),
               list("gamma differs 3-fold", 3.0))) {
  set.seed(99)
  u_a <- rgamma(600, 3, 1); s_a <- u_a / 1.0 + rnorm(600, 0, 0.25)
  u_b <- rgamma(600, 3, 1); s_b <- u_b / z[[2]] + rnorm(600, 0, 0.25)
  u <- c(u_a, u_b); s <- c(s_a, s_b)
  hi <- u >= quantile(u, 0.95)
  gamma_hat <- sum(u[hi] * s[hi]) / sum(u[hi]^2)
  vel <- s - gamma_hat * u
  cat(sprintf("\n  %s\n", z[[1]]))
  cat(sprintf("    fitted single gamma = %.2f\n", gamma_hat))
  cat(sprintf("    cells called 'increasing': group A %.0f%%, group B %.0f%%\n",
              100 * mean(vel[1:600] > 0), 100 * mean(vel[601:1200] > 0)))
}
cat("\n  With one kinetic rate the arrows are meaningless in aggregate but
  unbiased. When the two populations genuinely differ, a single fitted gamma
  declares one population to be increasing and the other decreasing, with
  confidence, purely because the model forced them to share a parameter.

  Treat velocity as a hypothesis generator. It is a strong assumption stated
  as a picture, and pictures are persuasive out of proportion to their
  evidence.\n")

#' ## 8. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "trajectory.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
cols <- hcl.colors(50, "viridis")[cut(t_true, 50, labels = FALSE)]
plot(Z[, 1], Z[, 2], col = cols, pch = 16, cex = 0.4, xlab = "PC1",
     ylab = "PC2", main = sprintf("Principal curve (41.1), rho = %.2f", r_sp))
lines(curve[, 1], curve[, 2], col = "firebrick", lwd = 3)
h1 <- hist(p_null, breaks = 25, plot = FALSE)
h2 <- hist(p_indep, breaks = 25, plot = FALSE)
plot(h1, freq = FALSE, col = adjustcolor("firebrick", 0.7), border = NA,
     xlab = "p-value", main = "NO trajectory in the data",
     ylim = c(0, max(h1$density, h2$density)))
plot(h2, freq = FALSE, col = adjustcolor("steelblue", 0.6), border = NA,
     add = TRUE)
abline(h = 1, lty = 2)
legend("topright", c("pseudotime from these genes", "independent ordering"),
       bty = "n", cex = 0.8,
       fill = adjustcolor(c("firebrick", "steelblue"), 0.7))
plot(t_true, sd_cell, pch = 16, cex = 0.3, col = adjustcolor("steelblue", 0.35),
     xlab = "true position", ylab = "SD of pseudotime (bootstrap)",
     main = "Uncertainty is not uniform")
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "trajectory.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: How much signal does it take to invent a trajectory?
#'
#' Vary the strength of the real trajectory from zero upward, and record both
#' the correlation with truth and the number of "dynamic" genes found naively.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %-26s%16s%20s\n", "dynamic genes in truth", "rho with truth",
#             "naive discoveries"))
# for (nd in c(0, 10, 40, 120)) {
#   Sl <- simulate_trajectory(n_dynamic = max(nd, 1), has_trajectory = nd > 0,
#                             seed = 200 + nd)
#   ptl <- principal_curve(embed(Sl$counts))$lam
#   rho <- abs(cor(ptl, Sl$t_true, method = "spearman"))
#   pl <- trajectory_test(Sl$counts, ptl, Sl$size, genes = SUB)
#   cat(sprintf("  %-26d%16.3f%20d\n", nd, rho,
#               sum(p.adjust(pl, "BH") < 0.05)))
# }
#
# ## With zero dynamic genes the correlation with "truth" is meaningless, yet
# ## the naive test still returns discoveries. The pipeline never refuses: it
# ## always produces a curve, an ordering and a gene list.
# ##
# ## That is the practical danger. There is no step in a standard trajectory
# ## workflow that says "there is no trajectory here". You have to build that
# ## check yourself, by running the same pipeline on permuted or simulated
# ## null data and comparing.

#' ### Problem 2: Does the trajectory even exist?
#'
#' Build a null for the trajectory itself. Permute each gene independently
#' across cells, which destroys any coordinated structure but keeps every
#' gene's marginal distribution, then compare a measure of curve quality.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# curve_quality <- function(c_) {
#   Yl <- normalise(c_); Yl <- sweep(Yl, 2, colMeans(Yl))
#   d <- svd(Yl, nu = 0, nv = 0)$d
#   d[1]^2 / sum(d^2)
# }
# gene_permute <- function(c_, seed) {
#   set.seed(seed); apply(c_, 2, sample)
# }
# for (z in list(list("real trajectory", counts), list("no trajectory", counts0))) {
#   obs <- curve_quality(z[[2]])
#   null <- sapply(1:20, function(i) curve_quality(gene_permute(z[[2]], 900 + i)))
#   cat(sprintf("  %-20s observed %.4f  null %.4f +/- %.4f   z = %7.1f\n",
#               z[[1]], obs, mean(null), sd(null), (obs - mean(null)) / sd(null)))
# }
#
# ## The real dataset sits far above its permutation null; the null dataset
# ## does not. This is the check the standard workflow omits, and it costs one
# ## line of code.
# ##
# ## Note what the permutation preserves: every gene's own distribution, its
# ## mean, its dispersion, its zeros. What it destroys is the COORDINATION
# ## between genes, which is the only thing a trajectory can be made of. That
# ## is what makes it the right null here, and it is the same reasoning used
# ## for gene-set tests in Module 39.

#' ### Problem 3: Pseudotime is not time
#'
#' Sample cells non-uniformly in real time, so that some stages are
#' over-represented, and check whether pseudotime recovers real time or
#' something else.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# set.seed(55)
# ## Real time is uniform, but cells DWELL in the middle stage, so we capture
# ## far more of them there.
# real_time <- c(runif(200, 0, 0.35), runif(800, 0.35, 0.65),
#                runif(200, 0.65, 1.0))
# n_c <- length(real_time)
# base <- exp(rnorm(300, 2.5, 0.9))
# peak <- runif(60); width <- runif(60, 0.12, 0.3)
# lm_ <- matrix(log(base), n_c, 300, byrow = TRUE)
# sh <- sweep(outer(real_time, peak, "-"), 2, width, "/")
# lm_[, 1:60] <- lm_[, 1:60] + 1.8 * exp(-0.5 * sh^2)
# sz <- rlnorm(n_c, 0, 0.3)
# cc <- matrix(rpois(n_c * 300, exp(lm_) * sz), n_c)
# ptl <- principal_curve(embed(cc))$lam
# if (cor(ptl, real_time, method = "spearman") < 0) ptl <- 1 - ptl
# cat(sprintf("  Spearman(pseudotime, real time) = %.3f\n",
#             cor(ptl, real_time, method = "spearman")))
# cat(sprintf("\n  %-22s%8s%26s\n", "real-time window", "cells",
#             "pseudotime span covered"))
# for (w in list(c(0, 0.35), c(0.35, 0.65), c(0.65, 1.0))) {
#   m <- real_time >= w[1] & real_time < w[2]
#   cat(sprintf("  %-22s%8d%26.3f\n", sprintf("[%.2f, %.2f)", w[1], w[2]),
#               sum(m), diff(range(ptl[m]))))
# }
#
# ## The rank correlation is high, so pseudotime "works". But look at the span
# ## column: the crowded middle stage, which occupies 30% of real time, is
# ## stretched across most of the pseudotime axis, because pseudotime is
# ## arclength through the data cloud and the cloud is dense where cells dwell.
# ##
# ## So a gene that changes steeply in pseudotime may be changing slowly in
# ## real time, and vice versa. Any statement of the form "gene X switches on
# ## halfway through differentiation" is a statement about the sampling, not
# ## about the biology, unless you have external time points to anchor it.

#' ## What to take away
#'
#' 1. Pseudotime is an **ordering**, with no units and no identifiable
#'    direction. Density along it reflects dwell time, not elapsed time.
#' 2. Testing genes against a pseudotime built from those genes is **double
#'    dipping** and produces discoveries on data with no trajectory (41.3).
#' 3. Split genes or cells so the ordering is not a function of what you test.
#' 4. Bootstrap the ordering. Cell positions move between runs, the pattern
#'    of that movement depends on the algorithm, and it is almost never
#'    propagated into the gene-level p-values.
#' 5. **Velocity** (41.3) assumes shared kinetic rates. When they differ it
#'    reports confident arrows that are artefacts of the shared parameter.
#'
#' **Next:** `42_deconvolution_and_integration.R`
