#' ---
#' title: "Foundations 1 - Variables, populations, and samples"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Part 0, §0.1-§0.3
#' **Assumes:** nothing. If you can read a table, you can read this.
#'
#' ## The one idea
#'
#' You measure a few things. You want to say something about all the things.
#'
#' That gap - between the handful you measured and the many you care about -
#' is what statistics is for. This module makes the gap visible by doing
#' something you can never do in real life: we *invent* an entire population,
#' so we know the true answer, and then take samples from it and see how close
#' we get.
#'
#' Everything else in this course is a more sophisticated version of this.

#+ setup, message = FALSE
MODULE_NAME <- "F1_variables_populations_samples"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

## set.seed() fixes the starting point of R's random number generator, so you
## get the same "random" numbers every run. That is what makes these results
## reproducible. Change the number and every value below changes - but every
## CONCLUSION stays the same. That is the point.
set.seed(101)

#' ## 1. A variable is something that varies

#+ variables
header("1. Variables: one row per unit, one column per measurement")
n_mice <- 8
dat <- data.frame(
  mouse     = paste0("m", 1:n_mice),
  treatment = rep(c("control", "drug"), each = 4),
  weight_g  = round(rnorm(n_mice, 24, 2), 1),
  marker    = round(rnorm(n_mice, 12, 1.5), 1),
  responded = rbinom(n_mice, 1, 0.5),
  grade     = sample(c("I", "II", "III"), n_mice, replace = TRUE)
)
print(dat, row.names = FALSE)
cat(sprintf("
  The UNIT here is the mouse. There are %d of them, so n = %d.

  Deciding what the unit is turns out to be the hardest and most important
  question in experimental biology. If we had measured 20 cells from each
  mouse we would have 160 rows - but still only %d units. Counting those 160
  rows as 160 independent observations is the most common fatal error in
  published biology. That is Topic 1, and exercise E1.\n", n_mice, n_mice, n_mice))

#' ## 2. The type of a variable decides what arithmetic is allowed

#+ types
header("2. Variable types, and what you may do with each")
cat(sprintf("  %-12s%-16s%-14s%s\n", "variable", "type", "average it?", "why"))
rows <- list(
  c("weight_g", "continuous", "yes", "any value in a range; differences mean something"),
  c("marker", "continuous", "yes", "same"),
  c("responded", "binary", "yes (!)", "0/1, so the average IS the proportion"),
  c("grade", "ordinal", "careful", "ordered, but is III - II the same as II - I?"),
  c("treatment", "categorical", "no", "'control' and 'drug' are labels, not numbers"))
for (r in rows) cat(sprintf("  %-12s%-16s%-14s%s\n", r[1], r[2], r[3], r[4]))
cat(sprintf("
  The binary case is worth pausing on:
    responded = %s
    average   = %.3f  <- this is the PROPORTION that responded

  Coding yes/no as 1/0 makes the mean and the proportion the same number. That
  small trick is why binary outcomes slot into the same machinery as
  continuous ones (Topic 13).

  The categorical case is where software betrays you. In R, `treatment` is a
  character column and most functions will refuse to average it - which is
  good. But code it as 1 and 2 and R will happily return 1.5 and warn you
  about nothing. ALWAYS store categories as factors:\n",
  paste(dat$responded, collapse = " "), mean(dat$responded)))
dat$treatment <- factor(dat$treatment, levels = c("control", "drug"))
cat(sprintf("    levels(dat$treatment) = %s\n",
            paste(levels(dat$treatment), collapse = ", ")))
cat("    The FIRST level is the reference that everything is compared against.\n")
cat("    R orders levels alphabetically unless you say otherwise, which is a\n")
cat("    frequent source of sign errors (see Module 30).\n")

#' ## 3. Population vs sample: the whole problem, made visible

#+ population
header("3. Inventing a population so we can see how well sampling works")
POP_SIZE <- 100000
TRUE_MEAN <- 24
TRUE_SD <- 2
population <- rnorm(POP_SIZE, TRUE_MEAN, TRUE_SD)
cat(sprintf("  The population: %s mice\n", format(POP_SIZE, big.mark = ",")))
cat(sprintf("    true mean   mu    = %.4f   <- Greek letter = TRUTH\n", mean(population)))
cat(sprintf("    true SD     sigma = %.4f\n", sd(population)))
cat("\n  Now take samples from it, as a real experiment would:\n\n")
cat(sprintf("  %-16s%20s%12s\n", "sample size n", "sample mean x-bar", "error"))
for (n in c(3, 10, 30, 100, 1000)) {
  s <- sample(population, n)
  cat(sprintf("  %-16d%20.3f%+12.3f\n", n, mean(s), mean(s) - TRUE_MEAN))
}
cat(sprintf("
  Two things to notice, and they are the whole of §0.3:

    1. Every sample gives a DIFFERENT answer. None equals %.1f. The sample
       mean is not the truth - it is a noisy glimpse of the truth.

    2. Bigger samples are closer ON AVERAGE - but any single small sample can
       land close by luck. That is exactly why you cannot judge a method from
       one run. Section 4 repeats each experiment 1,000 times, which is the
       only way to tell lucky from reliable.

  Notation, which you will see everywhere:
    mu, sigma  (Greek)      = population values. Fixed. Unknown. What you want.
    x-bar, s   (Latin/hats) = sample values.     Known. They vary.

  Greek is truth; Latin is your guess.\n", TRUE_MEAN))

#' ## 4. The same experiment, run 1,000 times

#+ repeats
header("4. Repeating the experiment 1,000 times")
N_REPEATS <- 1000
cat(sprintf("  %-8s%28s%22s\n", "n", "mean of the 1000 estimates", "spread of them (SD)"))
spreads <- list()
for (n in c(3, 10, 30, 100)) {
  means <- replicate(N_REPEATS, mean(sample(population, n)))
  spreads[[as.character(n)]] <- means
  cat(sprintf("  %-8d%28.3f%22.3f\n", n, mean(means), sd(means)))
}
cat(sprintf("
  Read the two columns separately - they say different things.

  COLUMN 1 is about BIAS: every row lands on %.1f. The sample mean is RIGHT ON
  AVERAGE, even with n = 3. A single 3-mouse experiment can be far off, but it
  is not systematically too high or too low.

  COLUMN 2 is about PRECISION: the spread shrinks as n grows. This spread has
  a name - the STANDARD ERROR - and it is what 'how much should I trust this
  number?' actually means.

  Bias and precision are independent. An estimate can be unbiased and useless
  (huge spread), or precise and wrong (small spread, systematically off). A
  confounded experiment (E1, E4) is precise and wrong, which is the dangerous
  combination, because precision looks like credibility.\n", TRUE_MEAN))

#' ## 5. The surprising part: the fraction sampled does not matter

#+ fraction
header("5. Sample SIZE matters; sample FRACTION does not")
small_pop <- rnorm(2000, TRUE_MEAN, TRUE_SD)
big_pop <- rnorm(2000000, TRUE_MEAN, TRUE_SD)
cat(sprintf("  %-16s%14s%8s%12s%18s\n", "population", "pop size", "n",
            "fraction", "SD of estimate"))
for (nm in c("small", "big")) {
  pop <- if (nm == "small") small_pop else big_pop
  n <- 100
  sdm <- sd(replicate(600, mean(sample(pop, n))))
  cat(sprintf("  %-16s%14s%8d%11.4f%%%18.4f\n", nm,
              format(length(pop), big.mark = ","), n, 100*n/length(pop), sdm))
}
cat("
  One population is a THOUSAND times bigger than the other. A sample of 100
  covers 5% of the first and 0.005% of the second. The precision is the same.

  This is why a poll of 1,000 people works about equally well for a small
  country and a huge one, and why 'we sequenced 90% of the cells' is not the
  reassurance it sounds like. What buys precision is n - the number of
  INDEPENDENT units - and nothing else.\n")

#' ## 6. Figure

#+ figure
png(file.path(OUT, "populations_and_samples.png"), width = 1250, height = 420, res = 110)
par(mfrow = c(1, 3), mar = c(4.4, 4.2, 2.6, 1))
hist(population, breaks = 60, col = "grey80", border = NA,
     main = "The population (100,000 mice)", xlab = "weight (g)")
abline(v = TRUE_MEAN, col = "firebrick", lwd = 2)
one <- sample(population, 10)
hist(population, breaks = 60, col = "grey90", border = NA, freq = FALSE,
     main = "One sample is a noisy glimpse", xlab = "weight (g)")
points(one, rep(0.02, 10), pch = 16, col = "steelblue", cex = 1.1)
abline(v = TRUE_MEAN, col = "firebrick", lwd = 2)
abline(v = mean(one), col = "steelblue", lwd = 2, lty = 2)
cols <- c("firebrick", "darkorange", "seagreen", "steelblue")
plot(density(spreads[["3"]]), col = cols[1], lwd = 2, xlim = c(21, 27),
     main = "Estimates from 1,000 repeats", xlab = "sample mean")
for (i in 2:4) lines(density(spreads[[as.character(c(3,10,30,100)[i])]]),
                     col = cols[i], lwd = 2)
abline(v = TRUE_MEAN, lwd = 1.5)
legend("topright", paste0("n = ", c(3,10,30,100)), col = cols, lwd = 2,
       bty = "n", cex = 0.75)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "populations_and_samples.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Does a bigger sample fix a biased one?

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cat(sprintf("  %-8s%16s%24s\n", "n", "random sample", "heaviest-60% sample"))
# heavy <- population[population > quantile(population, 0.40)]
# for (n in c(10, 100, 1000, 10000))
#   cat(sprintf("  %-8d%16.3f%24.3f\n", n,
#               mean(sample(population, n)), mean(sample(heavy, n))))
# cat(sprintf("  %-8s%16.3f%24.3f\n", "TRUTH", TRUE_MEAN, TRUE_MEAN))
#
# ## The random column converges on the truth. The biased column converges just
# ## as tightly - on the WRONG NUMBER. More data makes a biased estimate more
# ## precisely wrong, and gives you more confidence in a false answer.
# ##
# ## This is the most important thing to understand about sample size. n fixes
# ## NOISE. It does nothing about BIAS. No amount of data repairs a sample
# ## collected in a way related to what you are measuring - which is why HOW
# ## you select units matters more than how many, and why randomisation is the
# ## most valuable single step in an experiment.

#' ### Problem 2: The median and the outlier

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# clean <- sample(population, 20)
# dirty <- c(clean, 240)                  # a decimal point in the wrong place
# cat(sprintf("  %-22s%10s%10s%10s\n", "", "mean", "median", "SD"))
# cat(sprintf("  %-22s%10.2f%10.2f%10.2f\n", "clean (n=20)",
#             mean(clean), median(clean), sd(clean)))
# cat(sprintf("  %-22s%10.2f%10.2f%10.2f\n", "with one typo (n=21)",
#             mean(dirty), median(dirty), sd(dirty)))
# cat(sprintf("  %-22s%+10.2f%+10.2f%+10.2f\n", "shift",
#             mean(dirty)-mean(clean), median(dirty)-median(clean),
#             sd(dirty)-sd(clean)))
#
# ## One bad value moves the mean by about 10 g and the median by almost
# ## nothing. The median is ROBUST: it depends on the ORDER of the values, not
# ## their magnitudes, so a single extreme point cannot drag it.
# ##
# ## The SD is hit even harder than the mean, because it squares distances - so
# ## an outlier damages your UNCERTAINTY estimate more than your estimate.
# ##
# ## This is why you always PLOT your data before analysing it (Topic 3). It is
# ## not that the median is "better" - it answers a different question. Use the
# ## mean when you want a total or an average effect; use the median when you
# ## want a typical value and the tail is not the point.

#' ### Problem 3: What is the unit?
#'
#' For each study, decide what one row of the analysis table should be, and
#' therefore what $n$ is.
#'
#' 1. 6 mice per group, one blood sample each, one measurement per sample.
#' 2. 6 mice per group, 20 cells imaged per mouse.
#' 3. 3 flasks of cells per condition, each split into 3 wells.
#' 4. 200 patients, one biopsy each, 20,000 genes measured per biopsy.
#' 5. 4 patients, 5,000 single cells sequenced per patient.

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# cases <- list(
#   c("6 mice/group, 1 measurement each", "mouse", "6",
#     "nothing to collapse; rows already independent"),
#   c("6 mice/group, 20 cells each", "mouse", "6",
#     "120 cells, but cells share a mouse -> average to 6 values"),
#   c("3 flasks/condition, 3 wells each", "flask", "3",
#     "wells are technical replicates OF the flask"),
#   c("200 patients, 20,000 genes each", "patient", "200",
#     "genes are VARIABLES, not units - 20,000 tests of n=200"),
#   c("4 patients, 5,000 cells each", "patient", "4",
#     "20,000 cells, n = 4. The single-cell trap (Module 31)"))
# cat(sprintf("  %-38s%-10s%4s   %s\n", "study", "unit", "n", "why"))
# for (cs in cases)
#   cat(sprintf("  %-38s%-10s%4s   %s\n", cs[1], cs[2], cs[3], cs[4]))
#
# ## The rule: THE UNIT IS WHAT YOU RANDOMISED, or what varies independently.
# ## Everything measured inside a unit is a sub-sample of it, and averaging
# ## those sub-samples loses nothing about the treatment effect.
# ##
# ## Case 4 is the one people get right by accident and case 5 is the one they
# ## get wrong. In both, the big number (20,000) is tempting, and in both it is
# ## not n. In case 4 the 20,000 genes create a MULTIPLE TESTING problem
# ## (Topic 8); in case 5 the 20,000 cells create a PSEUDOREPLICATION problem
# ## (Topic 1). Different problems, opposite fixes, same tempting big number.

#' ## What to take away
#'
#' 1. A **unit** is the thing that varies independently. Counting sub-samples
#'    as units is the most common fatal error in biology.
#' 2. **Greek letters are the truth**; Latin letters are your estimates.
#' 3. A sample mean is **unbiased but noisy**.
#' 4. Larger $n$ reduces **noise**. It does nothing about **bias**.
#' 5. What matters is the **size** of the sample, not the fraction covered.
#'
#' **Next:** `F2_probability_basics.R`
