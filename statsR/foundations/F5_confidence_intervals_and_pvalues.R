#' ---
#' title: "Foundations 5 - Confidence intervals and p-values"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Part 0, §0.7-§0.8, equations (0.13)-(0.15)
#' **Assumes:** `F1`-`F4`.
#'
#' ## Why this module matters most
#'
#' Confidence intervals and p-values are the two things every paper reports and
#' the two things most often misread. Both are statements about a **procedure**
#' repeated many times, not about your particular result - and almost every
#' misinterpretation comes from forgetting that.
#'
#' This module builds both from scratch and **checks them by simulation**, so
#' you see the procedure succeed at exactly the advertised rate rather than
#' taking anyone's word for it.

#+ setup, message = FALSE
MODULE_NAME <- "F5_confidence_intervals_and_pvalues"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")
set.seed(105)
TRUE_MU <- 50; TRUE_SIGMA <- 10

#' ## 1. Building a confidence interval, eq. (0.13)-(0.14)

#+ build
header("1. One interval, built by hand")
n <- 16
s_ <- rnorm(n, TRUE_MU, TRUE_SIGMA)
xbar <- mean(s_); s <- sd(s_); se <- s/sqrt(n)
## With sigma unknown we use the t critical value, a little larger than 1.96,
## which makes the interval appropriately wider.
crit <- qt(0.975, df = n - 1)
lo <- xbar - crit*se; hi <- xbar + crit*se
cat(sprintf("  sample mean  x-bar = %.3f\n", xbar))
cat(sprintf("  sample SD    s     = %.3f\n", s))
cat(sprintf("  standard error     = s/sqrt(n) = %.3f\n", se))
cat(sprintf("  critical value     = %.3f   (t with %d df; normal would use 1.960)\n",
            crit, n - 1))
cat(sprintf("\n  95%% CI = %.3f +/- %.3f x %.3f = (%.3f, %.3f)\n",
            xbar, crit, se, lo, hi))
cat(sprintf("  true value = %g  ->  %s\n", TRUE_MU,
            if (lo <= TRUE_MU && TRUE_MU <= hi) "INSIDE" else "OUTSIDE"))

#' ## 2. What "95% confident" actually means

#+ coverage
header("2. The coverage check: does the procedure really work 95% of the time?")
ci <- function(v, level = 0.95) {
  k <- length(v); m <- mean(v); e <- sd(v)/sqrt(k)
  c(m - qt(0.5 + level/2, k - 1)*e, m + qt(0.5 + level/2, k - 1)*e)
}
cat(sprintf("  %-10s%-8s%18s%22s\n", "level", "n", "intervals built",
            "contained the truth"))
for (level in c(0.80, 0.90, 0.95, 0.99)) {
  hits <- mean(replicate(10000, {
    b <- ci(rnorm(16, TRUE_MU, TRUE_SIGMA), level); b[1] <= TRUE_MU && TRUE_MU <= b[2]
  }))
  cat(sprintf("  %-10s%-8d%18s%21.1f%%\n", paste0(100*level, "%"), 16,
              "10,000", 100*hits))
}
cat("
  The procedure delivers what it promises. That is the ONLY guarantee a
  confidence interval carries, and it is a guarantee about the METHOD.

  CORRECT   : 'if I repeated this experiment many times, 95% of the intervals
               I built would contain the true value.'
  INCORRECT : 'there is a 95% probability the true value is in MY interval.'

  Why is the second wrong? Your interval is now a fixed pair of numbers and
  the truth is a fixed number. Either it is in there or it is not - there is
  no randomness left to have a probability about. The randomness was in the
  sampling, which has already happened.

  (If you genuinely want '95% probability the truth is in here', that is a
  CREDIBLE interval and requires Bayesian machinery and a prior - Topic 33.
  In practice the two often nearly coincide, which is why the sloppy reading
  usually survives contact with reality.)\n")

#' ## 3. Read the width, not whether it excludes zero

#+ width
header("3. Two 'non-significant' results that mean opposite things")
cat(sprintf("  %-26s%10s%22s%9s  verdict\n", "study", "estimate", "95% CI", "p"))
for (st_ in list(c("large, precise study", 0.1, 0.15),
                 c("small, noisy study", 2.5, 1.55))) {
  est <- as.numeric(st_[2]); se_ <- as.numeric(st_[3])
  p <- 2*pnorm(abs(est/se_), lower.tail = FALSE)
  cat(sprintf("  %-26s%10.2f%22s%9.3f  %s\n", st_[1], est,
              sprintf("(%+.2f, %+.2f)", est - 1.96*se_, est + 1.96*se_), p,
              if (p > 0.05) "n.s." else "sig"))
}
cat("
  Both are 'not significant'. They say completely different things.

    The first RULES OUT anything larger than about 0.4. That is a real,
    informative, publishable finding: the effect, if any, is small.

    The second rules out nothing. Effects of 0, of 2, of 5 are all consistent
    with it. It is not evidence of absence - it is absence of evidence.

  Reporting both as 'no significant difference' throws away the only
  information that distinguishes them. THIS is the strongest practical
  argument for reporting intervals, and the reason 'no significant difference'
  must never be written as 'no difference'.\n")

#' ## 4. What a p-value is, eq. (0.15)

#+ pvalue
header("4. Building a p-value from scratch, by simulation")
## NPG and the shift are named so the permutation below cannot drift out of
## step with the data, and the effect is big enough that the test should find it.
## A local seed, so this demonstration is the same every run and does not
## depend on how many draws the sections above happened to consume.
set.seed(14)
NPG <- 14; SHIFT <- 2.4
group_a <- rnorm(NPG, 50.0, 2)
group_b <- rnorm(NPG, 50.0 + SHIFT, 2)
observed_diff <- mean(group_b) - mean(group_a)
## The null hypothesis says the labels are meaningless. So SHUFFLE them and
## see how big a difference appears by chance alone. This is a permutation
## test: no formula, no distributional assumption, just the definition.
pooled <- c(group_a, group_b)
null_diffs <- replicate(20000, {
  perm <- sample(pooled)
  mean(perm[(NPG + 1):(2*NPG)]) - mean(perm[1:NPG])
})
p_perm <- mean(abs(null_diffs) >= abs(observed_diff))
cat(sprintf("  observed difference between the groups : %+.4f\n", observed_diff))
cat("  if the labels were meaningless, differences this big or bigger\n")
cat(sprintf("  happened in %s of 20,000 shuffles\n",
            format(sum(abs(null_diffs) >= abs(observed_diff)), big.mark = ",")))
cat(sprintf("\n  p (permutation) = %.4f\n", p_perm))
cat(sprintf("  p (t-test)      = %.4f   <- agrees\n",
            t.test(group_b, group_a, var.equal = TRUE)$p.value))
cat("
  Read the definition off the simulation:

      p = P( data at least this extreme | nothing is going on )

  Note what is on each side of the bar. The p-value is the probability of THE
  DATA given the HYPOTHESIS. It is not the probability of the hypothesis given
  the data - that is the reversal from F2, section 4, and it is the single
  most common error in interpreting statistics.

  A p-value of 0.03 does NOT mean 'a 3% chance there is no real effect'.\n")

#' ## 5. The kinds of p-value

#+ kinds
header("5. Five ways to get a p-value for the same comparison")
a <- rnorm(14, 0); b <- rnorm(14, 0.9)
p_t <- t.test(a, b, var.equal = TRUE)$p.value
p_welch <- t.test(a, b)$p.value
pool <- c(a, b); obs <- mean(b) - mean(a)
nd <- replicate(20000, { q <- sample(pool); mean(q[15:28]) - mean(q[1:14]) })
p_permut <- mean(abs(nd) >= abs(obs))
p_rank <- suppressWarnings(wilcox.test(a, b)$p.value)
boot <- replicate(20000, mean(sample(b, 14, TRUE)) - mean(sample(a, 14, TRUE)))
p_boot <- 2*min(mean(boot <= 0), mean(boot >= 0))
cat(sprintf("  %-16s%-26s%-30s%9s\n", "kind", "method", "assumes", "p"))
rows <- list(
  c("parametric", "Student t-test", "normality, equal variance", p_t),
  c("parametric", "Welch t-test", "normality only", p_welch),
  c("permutation", "shuffle the labels", "exchangeability", p_permut),
  c("rank-based", "Wilcoxon rank-sum", "nothing about shape", p_rank),
  c("bootstrap", "resample each group", "sample is representative", p_boot))
for (r in rows)
  cat(sprintf("  %-16s%-26s%-30s%9.4f\n", r[1], r[2], r[3], as.numeric(r[4])))
cat("
  On clean, well-behaved data they agree closely - as they should. They
  diverge when their assumptions diverge: on skewed data with outliers the
  t-test and the rank test can disagree sharply, and then you must decide
  WHICH QUESTION you are asking (a difference in means, or a tendency for one
  group to exceed the other - they are not the same question).

  The one that needs extra care: PERMUTATION assumes only that the labels are
  exchangeable under the null. That makes it robust - but you must shuffle
  WITHIN the structure of your design. With batches or paired samples, free
  shuffling builds a null that is simply wrong (exercise E2 measures this).\n")

#' ## 6. One-sided, two-sided, and p-hacking

#+ onesided
header("6. One-sided tests, and why they are abused")
cat(sprintf("  two-sided p = %.4f\n", t.test(a, b, var.equal = TRUE)$p.value))
cat(sprintf("  one-sided p = %.4f   <- exactly half, when the effect is in\n",
            t.test(a, b, var.equal = TRUE, alternative = "less")$p.value))
cat("                              the predicted direction\n")
cat("
  A one-sided test is legitimate only when a difference in the OTHER direction
  would be acted on identically to no difference at all, and only when you
  declared it before seeing the data. That is rare. Switching to one-sided
  because two-sided gave 0.06 is cheating, and it has a name.\n")

header("6b. What 'trying a few things' does to your error rate")
one_experiment <- function() {
  ## Two groups with NO real difference, measured 4 ways.
  x <- rnorm(15); y <- rnorm(15)
  c(t.test(x, y, var.equal = TRUE)$p.value,
    t.test(x, y, var.equal = TRUE, alternative = "less")$p.value,
    t.test(x, y, var.equal = TRUE, alternative = "greater")$p.value,
    suppressWarnings(wilcox.test(x, y)$p.value))
}
R <- 5000
res <- t(replicate(R, one_experiment()))
cat(sprintf("  %s experiments in which NOTHING is going on:\n\n",
            format(R, big.mark = ",")))
cat(sprintf("  %-46s%20s\n", "strategy", "false positive rate"))
cat(sprintf("  %-46s%19.1f%%\n", "pre-specified two-sided t-test",
            100*mean(res[, 1] < 0.05)))
cat(sprintf("  %-46s%19.1f%%\n", "try 4 analyses, report the smallest p",
            100*mean(apply(res, 1, min) < 0.05)))
cat("
  Four analyses is a modest amount of flexibility - far less than a real
  analyst has - and it roughly doubles the false positive rate.

  The reported p-value is then meaningless, because a p-value describes the
  PROCEDURE that produced it, and the procedure was 'try things until one
  works'. That procedure does not have a 5% error rate.

  The same mechanism drives:
    p-hacking         - trying analyses until one is significant
    HARKing           - inventing the hypothesis after seeing the result
    optional stopping - collecting more data until p drops below 0.05
                        (this reaches significance eventually with certainty)
    subgroup fishing  - 'it worked in the older female patients'

  The defence is always the same: DECIDE THE ANALYSIS BEFORE SEEING THE
  OUTCOME, and report everything you tried.\n")

#' ## 7. Where does 0.05 come from, and what are alpha and beta?

#+ errors
header("7. Two kinds of error, and the threshold that trades them off")
DIFF <- 1.0
error_rates <- function(alpha, n = 15, reps = 4000) {
  fp <- mean(replicate(reps, t.test(rnorm(n), rnorm(n))$p.value < alpha))
  tp <- mean(replicate(reps, t.test(rnorm(n), rnorm(n, DIFF))$p.value < alpha))
  c(fp, tp)
}
cat(sprintf("  %-12s%22s%22s\n", "alpha", "Type I (false pos)", "power (1 - Type II)"))
for (alpha in c(0.20, 0.10, 0.05, 0.01, 0.001)) {
  er <- error_rates(alpha)
  cat(sprintf("  %-12g%22.3f%22.3f\n", alpha, er[1], er[2]))
}
cat("
  TYPE I error  - you claim an effect that is not there. Its rate IS alpha.
  TYPE II error - you miss an effect that is there. Its rate is beta, and
                  POWER = 1 - beta is the second column.

  Lowering alpha buys fewer false positives and pays in missed real effects.
  There is no setting that avoids both; the threshold is a statement about
  which error is more costly IN YOUR SITUATION.

  0.05 is not a law of nature. Ronald Fisher suggested one-in-twenty in the
  1920s as a rough working line, explicitly as a rule of thumb. It survived
  because a shared default stops people choosing a threshold after seeing the
  answer - a real benefit, but not a mathematical one.

  Sensible thresholds vary enormously with the consequences:
    screening assay, cheap follow-up      alpha = 0.10
    ordinary single hypothesis            alpha = 0.05
    genome-wide association study         alpha = 5e-8   (Topic 24)
    particle physics discovery            alpha = 3e-7

  The modern position - the American Statistical Association's 2016 and 2019
  statements - is that 'statistically significant' should not be treated as a
  conclusion at all. Report the estimate, the interval, and the p-value.\n")

#' ## 8. Many tests at once: FWER and FDR

#+ multiple
header("8. What happens when you test 20,000 things")
G <- 20000
null_p <- runif(G)
cat(sprintf("  %s tests, NOT ONE of which has a real effect:\n\n",
            format(G, big.mark = ",")))
cat(sprintf("  %-42s%14s%12s\n", "rule", "discoveries", "all false?"))
cat(sprintf("  %-42s%14s%12s\n", "raw p < 0.05",
            format(sum(null_p < 0.05), big.mark = ","), "yes"))
for (m in list(c("bonferroni", "Bonferroni (controls FWER)"),
               c("holm", "Holm (controls FWER, more powerful)"),
               c("BH", "Benjamini-Hochberg (controls FDR)")))
  cat(sprintf("  %-42s%14s%12s\n", m[2],
              format(sum(p.adjust(null_p, m[1]) < 0.05), big.mark = ","), "yes"))
## Now with some real signal mixed in.
mixed <- c(runif(19000), rbeta(1000, 0.2, 12))
truth <- c(rep(FALSE, 19000), rep(TRUE, 1000))
cat("\n  Now 20,000 tests of which 1,000 ARE real:\n\n")
cat(sprintf("  %-42s%13s%8s%8s%8s\n", "rule", "discoveries", "false", "FDP", "power"))
for (m in list(c("bonferroni", "Bonferroni (FWER)"), c("holm", "Holm (FWER)"),
               c("BH", "Benjamini-Hochberg (FDR)"))) {
  rej <- p.adjust(mixed, m[1]) < 0.05
  cat(sprintf("  %-42s%13s%8d%8.3f%8.3f\n", m[2],
              format(sum(rej), big.mark = ","), sum(rej & !truth),
              sum(rej & !truth)/max(sum(rej), 1), mean(rej[truth])))
}
cat("
  Two different promises, and you must know which one you want:

    FWER (Bonferroni, Holm) - controls the probability of making EVEN ONE
      false claim. Appropriate when a single false positive is expensive.
      Very conservative when there are many tests, so power collapses.

    FDR (Benjamini-Hochberg) - controls the EXPECTED PROPORTION of your
      discoveries that are false. You accept that ~5% of your 200-gene list is
      wrong in exchange for getting a list at all. The right default in
      genomics; its adjusted values are called q-values.

  Look at the power column. That difference is why genomics runs on FDR.

  Topic 8 covers this properly, including what happens when the tests are
  correlated - which, for genes in a pathway, they always are.\n")

#' ## 9. Figure

#+ figure
png(file.path(OUT, "intervals_and_pvalues.png"), width = 1250, height = 440, res = 110)
par(mfrow = c(1, 3), mar = c(4.4, 4.4, 2.6, 1))
plot(NA, xlim = c(TRUE_MU - 15, TRUE_MU + 15), ylim = c(0, 41), yaxt = "n",
     xlab = "value", ylab = "", main = "40 intervals; red ones miss")
for (i in 1:40) {
  smp <- rnorm(16, TRUE_MU, TRUE_SIGMA); b <- ci(smp)
  covers <- b[1] <= TRUE_MU && TRUE_MU <= b[2]
  lines(b, c(i, i), col = if (covers) "steelblue" else "firebrick", lwd = 1.6)
}
abline(v = TRUE_MU, lwd = 1.5)
hist(null_diffs, breaks = 70, col = "grey70", border = NA, freq = FALSE,
     main = sprintf("Permutation null (p = %.3f)", p_perm),
     xlab = "difference under shuffled labels")
abline(v = c(-1, 1)*observed_diff, col = "firebrick", lwd = 2)
alphas <- c(0.5, 0.2, 0.1, 0.05, 0.01, 0.001)
er <- t(sapply(alphas, error_rates, reps = 1500))
plot(alphas, er[, 1], type = "b", pch = 16, col = "firebrick", log = "x",
     ylim = c(0, 1), xlab = "threshold alpha", ylab = "",
     main = "The trade-off you are choosing")
lines(alphas, er[, 2], type = "b", pch = 16, col = "steelblue")
legend("right", c("Type I rate", "power"), col = c("firebrick", "steelblue"),
       lwd = 2, bty = "n", cex = 0.75)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "intervals_and_pvalues.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Does the interval still work when the data is ugly?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# true_mean_ln <- exp(0 + 1.4^2/2)          # mean of lognormal(0, 1.4)
# cat("  A 95% interval SHOULD contain the truth 95% of the time.\n\n")
# cat(sprintf("  %-8s%16s%16s\n", "n", "normal data", "skewed data"))
# for (n_ in c(5, 10, 30, 100, 500)) {
#   cov_n <- mean(replicate(4000, { b <- ci(rnorm(n_)); b[1] <= 0 && 0 <= b[2] }))
#   cov_s <- mean(replicate(4000, { b <- ci(rlnorm(n_, 0, 1.4))
#                                   b[1] <= true_mean_ln && true_mean_ln <= b[2] }))
#   cat(sprintf("  %-8d%15.1f%%%15.1f%%\n", n_, 100*cov_n, 100*cov_s))
# }
#
# ## The normal column sits at 95% at every n - as it must, since the method
# ## was derived for exactly that case.
# ##
# ## The skewed column is too LOW at small n: the interval misses more often
# ## than advertised, so you are over-confident. It improves as n grows,
# ## because the central limit theorem (F4) gradually makes the sample mean
# ## normal.
# ##
# ## This is what an assumption violation actually looks like. Not a crash, not
# ## an error message - just a procedure quietly delivering 88% coverage while
# ## claiming 95%. The fixes are a bootstrap interval, a transformation (log),
# ## or more data. Topic 5 covers all three.

#' ### Problem 2: The p-value is not the probability you are wrong

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# run <- function(n_tests, frac_real, effect = 0.8, n = 20) {
#   real <- runif(n_tests) < frac_real
#   ps <- sapply(real, function(r)
#     t.test(rnorm(n), rnorm(n, if (r) effect else 0))$p.value)
#   sig <- ps < 0.05
#   c(sum(sig), if (sum(sig)) mean(!real[sig]) else NA)
# }
# cat(sprintf("  %-34s%13s%20s\n", "% of hypotheses that are REAL",
#             "significant", "of those, % FALSE"))
# for (frac in c(0.90, 0.50, 0.10, 0.01)) {
#   r <- run(4000, frac)
#   cat(sprintf("  %-33.0f%%%13s%19.1f%%\n", 100*frac,
#               format(r[1], big.mark = ","), 100*r[2]))
# }
#
# ## The threshold is 0.05 in every row. The chance that a significant result
# ## is wrong ranges from a few percent to most of them.
# ##
# ## The p-value cannot know this, because it is computed ASSUMING the null is
# ## true - it never considers how plausible the null was to begin with. So
# ## 'p < 0.05' carries completely different weight for a pre-registered
# ## hypothesis with good prior support than for the 4,000th exploratory test.
# ##
# ## This is the same base-rate effect as the screening test in F2, and it is
# ## the honest answer to 'why do so many published findings fail to
# ## replicate?' It is also why FDR methods (section 8) are framed around the
# ## PROPORTION of discoveries that are false - exactly the last column.

#' ### Problem 3: Optional stopping

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# peek_until_significant <- function(max_n, alpha = 0.05) {
#   ## Add one observation per group at a time; stop at the first p < alpha.
#   x <- rnorm(5); y <- rnorm(5)
#   for (i in seq_len(max_n - 5)) {
#     x <- c(x, rnorm(1)); y <- c(y, rnorm(1))
#     if (t.test(x, y)$p.value < alpha) return(TRUE)
#   }
#   FALSE
# }
# cat("  No real effect. Nominal false positive rate: 5%.\n\n")
# cat(sprintf("  %-18s%20s\n", "max n allowed", "reached p < 0.05"))
# for (max_n in c(5, 10, 20, 50, 100, 200))
#   cat(sprintf("  %-18d%19.1f%%\n", max_n,
#               100*mean(replicate(1500, peek_until_significant(max_n)))))
#
# ## Testing once at n = 5 gives the advertised 5%. Allowing yourself to keep
# ## looking pushes it far higher, and it keeps climbing with the budget - with
# ## unlimited data it reaches 100%. The p-value WILL dip below 0.05 eventually
# ## just by wandering, and if you stop the moment it does, you always "win".
# ##
# ## This is why 'we added a few more samples because the trend looked
# ## promising' invalidates a p-value, and why clinical trials must pre-specify
# ## their sample size or use a formal sequential design with corrected
# ## boundaries (O'Brien-Fleming, alpha-spending).
# ##
# ## Note the pattern across all three problems and section 6b: the p-value is
# ## a property of the PROCEDURE. Change the procedure - by trying variants, by
# ## peeking, by choosing which hypotheses to test after looking - and the
# ## number on the screen no longer means what its definition says.

#' ## What to take away
#'
#' 1. A confidence interval is a guarantee about the **procedure**.
#' 2. **Read the width.** Two non-significant results can mean opposite things.
#' 3. A p-value is $P(\text{data} \mid \text{no effect})$ - never the reverse.
#' 4. There are many kinds of p-value; one- vs two-sided is a separate axis.
#' 5. $\alpha$ **is** your false positive rate; power is $1-\beta$.
#' 6. With many tests, choose **FWER** or **FDR** deliberately.
#' 7. Every p-value describes a **procedure**. Changing it mid-analysis breaks it.
#'
#' **Next:** `F6_connecting_variables.R`
