#' ---
#' title: "Module 29 - Designing a simulation study"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 40, equations (40.1)-(40.4)
#' **Assumes:** Modules 05, 09, 24.
#'
#' ## What you will learn
#'
#' Every module in this course evaluates a method by simulation. This one
#' makes that method explicit, because a simulation is an experiment and
#' deserves the same design discipline you would demand of a wet-lab one.
#'
#' 1. **ADEMP**: aims, data-generating mechanism, estimand, methods,
#'    performance measures. State them before writing code.
#' 2. Performance measures (40.1)-(40.2), and the identity linking them.
#' 3. **Monte Carlo standard error** (40.3)-(40.4). A simulation result
#'    without an MCSE is a number without an error bar.
#' 4. **Common random numbers**: compare methods on shared datasets.
#' 5. Why a benchmark every method passes discriminates nothing.

#+ setup, message = FALSE
MODULE_NAME <- "29_simulation_and_benchmarking"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(29)

#' ## 1. ADEMP, stated before any code is written
#'
#' The example throughout: estimating the centre of a skewed distribution.

#+ ademp
header("1. ADEMP for the study in this module")
cat("  AIMS        Which estimator of central tendency should be used for a
              right-skewed biomarker, and does the answer depend on n?

  DATA        Y ~ LogNormal(mu = 0, sigma = 0.9), n in {10, 30, 100}.
              Chosen because biomarkers are right-skewed and this has
              closed-form truth.

  ESTIMAND    The POPULATION MEAN, exp(mu + sigma^2 / 2).
              Stated explicitly, because the median estimates something
              else and comparing them without saying so is meaningless.

  METHODS     sample mean; sample median; 20% trimmed mean;
              exp(mean of logs), the naive back-transform.

  PERFORMANCE Bias, empirical SE, RMSE, and CI coverage. Each with a
              Monte Carlo standard error.\n")
SIGMA <- 0.9
TRUTH <- exp(0 + SIGMA^2 / 2)
cat(sprintf("\n  estimand (population mean) = %.4f\n", TRUTH))
cat("\n  Write this down first. A simulation with no stated estimand cannot be
  wrong, which is another way of saying it cannot be informative. The most
  common failure in published simulation studies is comparing methods that
  target different quantities.\n")

#' ## 2. Performance measures and their Monte Carlo error, eq. (40.1)-(40.3)
#'
#' $$\text{bias}=\bar{\hat\theta}-\theta \qquad (40.1)$$
#' $$\text{MSE}=\text{bias}^2+\text{EmpSE}^2 \qquad (40.2)$$
#' $$\text{MCSE}(\text{bias})=\text{EmpSE}/\sqrt{n_{sim}} \qquad (40.3)$$

#+ performance
header("2. Every performance number needs an MCSE (40.1)-(40.3)")
trim_mean <- function(y, p) mean(y, trim = p)
estimators <- function(y)
  c(mean = mean(y), median = median(y), `trimmed 20%` = trim_mean(y, 0.2),
    `exp(mean log)` = exp(mean(log(y))))
run_study <- function(n, n_sim, seed = 0) {
  set.seed(seed)
  t(replicate(n_sim, estimators(rlnorm(n, 0, SIGMA))))
}
N_SIM <- 2000
res <- run_study(n = 30, n_sim = N_SIM, seed = 1)
cat(sprintf("  n = 30, n_sim = %d, estimand = %.4f\n\n", N_SIM, TRUTH))
cat(sprintf("  %-16s%9s%8s%9s%9s%9s\n", "method", "bias", "MCSE", "EmpSE",
            "RMSE", "% bias"))
for (k in colnames(res)) {
  v <- res[, k]
  bias <- mean(v) - TRUTH                                 # eq. (40.1)
  emp <- sd(v)
  rmse <- sqrt(mean((v - TRUTH)^2))                       # eq. (40.2)
  mcse <- emp / sqrt(N_SIM)                               # MCSE of bias
  cat(sprintf("  %-16s%9.4f%8.4f%9.4f%9.4f%9.1f\n", k, bias, mcse, emp, rmse,
              100 * bias / TRUTH))
}
cat("\n  The MCSE column tells you which differences are real. The sample
  mean's bias is SMALLER than its own MCSE, so it is indistinguishable from
  zero; reporting it as \"slightly biased\" would be reporting noise. The other
  three biases are a hundred times their MCSE and are unambiguously real.

  Check the identity in eq. (40.2), MSE = bias^2 + EmpSE^2:\n")
for (k in colnames(res)[1:2]) {
  v <- res[, k]
  cat(sprintf("    %-16s MSE = %.6f,  bias^2 + EmpSE^2 = %.6f\n", k,
              mean((v - TRUTH)^2),
              (mean(v) - TRUTH)^2 + (N_SIM - 1) / N_SIM * sd(v)^2))
}
cat("\n  The median and the log-based estimator are heavily biased FOR THIS
  ESTIMAND, and that is not a criticism of them. They estimate the median,
  which for this distribution is 1.000 rather than 1.499. Naming the estimand
  is what turns an unfair comparison into an informative one.\n")

#' ## 3. How many repetitions? eq. (40.4)
#'
#' $$n_{sim}\ge \frac{p(1-p)}{\text{MCSE}_{target}^2} \qquad (40.4)$$

#+ nsim
header("3. Choosing n_sim from the precision you need (40.4)")
cat("  For a coverage or rejection RATE near p, eq. (40.3) gives\n")
cat("  MCSE = sqrt(p(1-p)/n_sim), so eq. (40.4) inverts it:\n\n")
cat(sprintf("  %-16s%12s%12s%12s\n", "target MCSE", "p = 0.05", "p = 0.50",
            "p = 0.80"))
for (target in c(0.01, 0.005, 0.002, 0.001)) {
  row <- sapply(c(0.05, 0.5, 0.8),
                function(p) format(ceiling(p * (1 - p) / target^2),
                                   big.mark = ",", scientific = FALSE))
  cat(sprintf("  %-16.3f%12s%12s%12s\n", target, row[1], row[2], row[3]))
}
cat("\n  To claim a test holds its 5% level to within half a percentage point
  you need about 1,900 repetitions. To resolve it to a tenth of a point you
  need about 47,500.

  This is why \"the false positive rate was 4% in 100 runs\" is not evidence of
  anything: the MCSE there is 2%, so 4% and 8% are not distinguishable.\n")
cat("\n  The same quantity estimated with different n_sim:\n")
cat(sprintf("  %-10s%52s\n", "n_sim",
            "5 independent estimates of the type I error rate"))
for (n_sim in c(100, 1000, 10000)) {
  vals <- sapply(1:5, function(rep_) {
    set.seed(3000 + rep_ * 97 + n_sim)
    mean(replicate(n_sim,
                   t.test(rnorm(15), rnorm(15))$p.value < 0.05)) })
  cat(sprintf("  %-10d%s\n", n_sim, paste(sprintf("%10.3f", vals),
                                          collapse = "")))
}

#' ## 4. Common random numbers
#'
#' Comparing methods on the *same* simulated datasets removes between-dataset
#' variation from the comparison, exactly as pairing does in an experiment.

#+ crn
header("4. Compare methods paired, not independently")
n_sim <- 1000
# Paired: both estimators see identical data.
set.seed(11)
pa <- pb <- numeric(n_sim)
for (i in 1:n_sim) {
  y <- rlnorm(30, 0, SIGMA)
  pa[i] <- mean(y); pb[i] <- trim_mean(y, 0.2)
}
# Unpaired: each estimator gets its own datasets.
set.seed(12); ua <- replicate(n_sim, mean(rlnorm(30, 0, SIGMA)))
set.seed(13); ub <- replicate(n_sim, trim_mean(rlnorm(30, 0, SIGMA), 0.2))
d_p <- pa - pb; d_u <- ua - ub
cat(sprintf("  difference in RMSE between two estimators, n_sim = %d\n\n",
            n_sim))
cat(sprintf("  %-26s%18s%21s\n", "design", "mean difference",
            "MCSE of difference"))
cat(sprintf("  %-26s%18.4f%21.5f\n", "shared datasets", mean(d_p),
            sd(d_p) / sqrt(n_sim)))
cat(sprintf("  %-26s%18.4f%21.5f\n", "independent datasets", mean(d_u),
            sd(d_u) / sqrt(n_sim)))
cat(sprintf("\n  correlation between the two estimators on shared data: %.3f\n",
            cor(pa, pb)))
cat(sprintf("  precision gained by pairing: %.1fx fewer repetitions for the same MCSE\n",
            (sd(d_u) / sd(d_p))^2))
cat("\n  The two estimators are strongly correlated when computed on the same
  data, so most of the variation cancels in the difference. Pairing therefore
  buys a large reduction in the repetitions needed, for free.

  This is the simulation equivalent of a paired design (Topic 12), and the
  same reasoning applies: remove the variation you do not care about.\n")

#' ## 5. A benchmark that everything passes is not a benchmark

#+ benchmark
header("5. Include a case where the method should fail")
benchmark <- function(scenario, n_sim = 1200, seed = 0) {
  set.seed(seed)
  out <- c(`t pooled` = 0, `t Welch` = 0, Wilcoxon = 0, permutation = 0)
  for (i in 1:n_sim) {
    ab <- switch(scenario,
      "normal, null" = list(rnorm(25), rnorm(25)),
      "lognormal, null" = list(rlnorm(25, 0, 1.2), rlnorm(25, 0, 1.2)),
      "unequal var, EQUAL n" = list(rnorm(25), rnorm(25, 0, 4)),
      "unequal var, UNEQUAL n" = list(rnorm(10, 0, 4), rnorm(40)),
      "normal, real effect" = list(rnorm(25), rnorm(25, 0.8)))
    a <- ab[[1]]; b <- ab[[2]]
    out[1] <- out[1] + (t.test(a, b, var.equal = TRUE)$p.value < 0.05)
    out[2] <- out[2] + (t.test(a, b)$p.value < 0.05)     # Welch
    out[3] <- out[3] + (suppressWarnings(wilcox.test(a, b)$p.value) < 0.05)
    pool <- c(a, b); na <- length(a); obs <- abs(mean(b) - mean(a))
    nd <- replicate(99, { pp <- sample(pool)
      abs(mean(pp[(na + 1):length(pp)]) - mean(pp[1:na])) })
    out[4] <- out[4] + ((1 + sum(nd >= obs)) / 100 < 0.05)
  }
  out / n_sim
}
cat(sprintf("  %-26s%10s%10s%11s%13s%9s\n", "scenario", "t pooled", "t Welch",
            "Wilcoxon", "permutation", "target"))
for (sc in c("normal, null", "lognormal, null", "unequal var, EQUAL n",
             "unequal var, UNEQUAL n", "normal, real effect")) {
  r_ <- benchmark(sc, seed = nchar(sc) * 37)
  cat(sprintf("  %-26s%10.3f%10.3f%11.3f%13.3f%9s\n", sc, r_[1], r_[2], r_[3],
              r_[4], if (grepl("null|var", sc)) "0.05" else "high"))
}
cat("\n  The first row separates nothing: every method is correct on clean
  normal data, which is why a benchmark consisting only of that row would be
  uninformative.

  The rows that discriminate are the awkward ones, and they do not all break
  the same method.

  With unequal variance but EQUAL group sizes, the pooled t-test is fine: the
  variance misspecification cancels. Wilcoxon is the one that inflates,
  because it tests stochastic equality rather than equality of means, and two
  distributions with the same mean and different spreads are not stochastically
  equal.

  Make the group sizes UNEQUAL and the pooled t-test fails badly. It pools the
  two variances by sample size, so the larger group's variance dominates the
  estimate; when the SMALL group is the variable one the test is far too
  liberal. Welch holds throughout, which is why it is R's default. The
  permutation test does not rescue this either: permuting assumes the two
  groups are exchangeable under the null, and different variances mean they
  are not.

  A benchmark with only the first two rows would have reported all four
  methods as equivalent.

  Design benchmarks around the cases where methods SHOULD differ, always
  include a null, and always include a case the method under test is expected
  to fail. A comparison in which your preferred method wins every scenario is
  usually a sign that the scenarios were chosen after the fact.\n")

#' ## 6. Figure

#+ figure, fig.width = 13, fig.height = 4.5
png(file.path(OUT, "simulation_design.png"), width = 1300, height = 450)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
boxplot(as.data.frame(res), outline = FALSE, col = "grey85",
        ylab = "estimate", main = "Four estimators, one estimand",
        cex.axis = 0.8)
abline(h = TRUTH, col = "firebrick", lwd = 2, lty = 2)
legend("topright", "estimand", lty = 2, lwd = 2, col = "firebrick", bty = "n")
ns <- c(50, 100, 200, 500, 1000, 5000, 10000)
plot(ns, sqrt(0.05 * 0.95 / ns), type = "b", pch = 16, log = "x",
     col = "steelblue", lwd = 2, ylim = c(0, 0.04),
     xlab = "n_sim (log scale)", ylab = "MCSE of a rate",
     main = "MCSE of a rate, eq. (40.3)")
lines(ns, sqrt(0.5 * 0.5 / ns), type = "b", pch = 17, col = "firebrick", lwd = 2)
legend("topright", c("p = 0.05", "p = 0.50"), pch = c(16, 17), lwd = 2,
       bty = "n", col = c("steelblue", "firebrick"))
hist(d_u, breaks = 30, col = adjustcolor("firebrick", 0.6), border = NA,
     xlab = "difference between the two estimators",
     main = "Pairing removes shared variation")
hist(d_p, breaks = 30, col = adjustcolor("steelblue", 0.7), border = NA,
     add = TRUE)
legend("topright", c("independent data", "shared data"), bty = "n",
       fill = adjustcolor(c("firebrick", "steelblue"), 0.7))
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "simulation_design.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: How many repetitions did this course need?
#'
#' Several modules report a false positive rate from a few hundred runs. Work
#' out the MCSE of those claims and decide which are safe.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %-42s%8s%8s%8s%18s\n", "claim", "n_sim", "rate", "MCSE",
#             "95% CI"))
# claims <- list(list("module 08: BH holds its FDR", 500, 0.048),
#                list("E1: confounded design false positives", 500, 0.238),
#                list("F5: p-hacking inflates alpha", 5000, 0.098),
#                list("module 25: e-BH under dependence", 200, 0.031))
# for (cl in claims) {
#   mcse <- sqrt(cl[[3]] * (1 - cl[[3]]) / cl[[2]])
#   cat(sprintf("  %-42s%8d%8.3f%8.4f%18s\n", cl[[1]], cl[[2]], cl[[3]], mcse,
#               sprintf("(%.3f, %.3f)", cl[[3]] - 1.96 * mcse,
#                       cl[[3]] + 1.96 * mcse)))
# }
#
# ## The claims differ in how much they can bear.
# ##
# ## "The confounded design gives 24% false positives" is safe: the interval is
# ## nowhere near 5%, so the qualitative conclusion is unambiguous.
# ##
# ## "BH holds its FDR at 4.8%" from 500 runs has an interval of roughly 3% to
# ## 7%. That supports "BH is approximately calibrated" and does NOT support
# ## "BH is slightly conservative here", which is the kind of over-reading an
# ## MCSE prevents.
# ##
# ## The general rule: use few repetitions for claims about large qualitative
# ## differences, and many for claims about whether a rate equals its nominal
# ## value. Report n_sim so a reader can do this arithmetic themselves.

#' ### Problem 2: Build a simulation study end to end
#'
#' Compare three ways of handling a skewed outcome: t-test on the raw values,
#' t-test on logs, and a Wilcoxon test. Write the ADEMP first, then run it.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# ## AIMS: which test for a right-skewed outcome with a multiplicative effect?
# ## DATA: y = lognormal, treated group multiplied by a fold change.
# ## ESTIMAND: the null is "no difference in distribution"; under the
# ##           alternative the effect is MULTIPLICATIVE.
# ## METHODS: t on raw, t on log, Wilcoxon.
# ## PERFORMANCE: type I error and power, each with an MCSE.
# one <- function(n, fold, seed) {
#   set.seed(seed)
#   a <- rlnorm(n, 0, 1.0); b <- rlnorm(n, log(fold), 1.0)
#   c(t.test(a, b)$p.value, t.test(log(a), log(b))$p.value,
#     suppressWarnings(wilcox.test(a, b)$p.value))
# }
# NS <- 3000
# cat(sprintf("  %-14s%18s%18s%18s\n", "fold change", "t raw", "t on log",
#             "Wilcoxon"))
# for (fold in c(1.0, 1.3, 1.8)) {
#   hits <- rowSums(sapply(1:NS, function(i) one(30, fold, 200000 + i) < 0.05))
#   rates <- hits / NS
#   cells <- paste(sprintf("%11.3f +/-%5.3f", rates,
#                          sqrt(rates * (1 - rates) / NS)), collapse = "")
#   cat(sprintf("  %-14.1f%s\n", fold, cells))
# }
#
# ## At fold = 1.0 all three hold the nominal level, so all three are VALID.
# ## Validity was never the question.
# ##
# ## At fold > 1 the log t-test and Wilcoxon are clearly more powerful than the
# ## t-test on raw values, because the effect is multiplicative and the raw
# ## scale spreads it across a long tail. The MCSEs are small enough that the
# ## ordering is real rather than noise, which is exactly what reporting them
# ## lets you assert.
# ##
# ## Notice the estimand did the work again. On the log scale the t-test
# ## targets a ratio of geometric means, which is the quantity the data
# ## generating mechanism actually contains.

#' ### Problem 3: A benchmark designed to be won
#'
#' Show how a scenario set can be chosen so that a preferred method wins, then
#' show what an honest scenario set reveals.

## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# power_of <- function(method, n, shift, tail, seed, NS = 1200) {
#   set.seed(seed)
#   mean(replicate(NS, {
#     if (tail == "normal") { a <- rnorm(n); b <- rnorm(n, shift) }
#     else { a <- rt(n, 2); b <- rt(n, 2) + shift }
#     p_ <- if (method == "t") t.test(a, b)$p.value
#           else suppressWarnings(wilcox.test(a, b)$p.value)
#     p_ < 0.05 }))
# }
# cat("  A 'benchmark' using only normal data:\n\n")
# cat(sprintf("  %-26s%10s%11s%10s\n", "scenario", "t-test", "Wilcoxon",
#             "winner"))
# for (sh in c(0.4, 0.6, 0.8)) {
#   t_ <- power_of("t", 30, sh, "normal", 1)
#   w_ <- power_of("w", 30, sh, "normal", 1)
#   cat(sprintf("  %-26s%10.3f%11.3f%10s\n", sprintf("normal, shift %.1f", sh),
#               t_, w_, if (t_ > w_) "t-test" else "Wilcoxon"))
# }
# cat("\n  The same comparison with a heavy-tailed scenario added:\n\n")
# cat(sprintf("  %-26s%10s%11s%10s\n", "scenario", "t-test", "Wilcoxon",
#             "winner"))
# for (sh in c(0.4, 0.8)) {
#   t_ <- power_of("t", 30, sh, "t2", 2); w_ <- power_of("w", 30, sh, "t2", 2)
#   cat(sprintf("  %-26s%10.3f%11.3f%10s\n",
#               sprintf("heavy-tailed, shift %.1f", sh), t_, w_,
#               if (t_ > w_) "t-test" else "Wilcoxon"))
# }
#
# ## Restricted to normal data, the t-test wins every row, and a paper could
# ## report that table truthfully and conclude the t-test is superior.
# ##
# ## Add one heavy-tailed scenario and the ordering reverses. Nothing in the
# ## first table was false; it was incomplete, and incompleteness is how
# ## benchmarks mislead without anyone lying.
# ##
# ## Two defences when reading a benchmark: ask which scenarios are ABSENT, and
# ## ask whether the authors' own method was the one the scenarios were
# ## designed around. Two defences when writing one: pre-specify the scenarios,
# ## and include the case where your method should lose.

#' ## What to take away
#'
#' 1. Write **ADEMP** before code. Naming the **estimand** is what makes a
#'    comparison meaningful.
#' 2. Report **bias, EmpSE, RMSE and coverage** (40.1)-(40.2), never a single
#'    summary.
#' 3. Every simulated number needs an **MCSE** (40.3). Use (40.4) to choose
#'    $n_{sim}$ from the precision you need, before running anything.
#' 4. Use **common random numbers**: comparing methods on shared datasets can
#'    cut the required repetitions by an order of magnitude.
#' 5. A benchmark without a scenario your method should fail is marketing.
#'
#' **Next:** `../bioinformatics/41_trajectory_and_pseudotime.R`
