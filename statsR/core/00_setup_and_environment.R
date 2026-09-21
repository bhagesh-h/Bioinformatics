#' ---
#' title: "Module 00 - Setup, environment check, and house rules"
#' output:
#'   html_document:
#'     toc: true
#'     toc_depth: 2
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 35 (Reproducible statistical workflows)
#'
#' **Run this first.** It verifies that every package the course uses is present
#' and that the random-number machinery behaves deterministically. If this module
#' prints `ALL CHECKS PASSED`, every other module in `statsR/` will run.
#'
#' ```bash
#' docker run --rm -v "$PWD":/work -w /work learn-stats-r:1.0 \
#'     Rscript statsR/core/00_setup_and_environment.R
#' ```
#'
#' ## What you will learn
#'
#' 1. How to capture an environment so a result can be regenerated (Topic 35).
#' 2. Why `set.seed()` placement matters and how to make parallel work reproducible.
#' 3. The output convention used by every later module.
#'
#' ## 1. Environment capture
#'
#' `stats.md` Topic 35 notes that statistical results genuinely change across
#' package versions. So the first thing any analysis should emit is *what
#' produced it*.

#+ environment-report
## Packages the course relies on. Everything in `base_recommended` ships with
## any standard R installation, which is why ~90% of this course runs without a
## single CRAN download.
base_recommended <- c("stats", "MASS", "nlme", "mgcv", "survival",
                      "cluster", "boot", "Matrix", "splines")
extra_packages <- c("knitr", "rmarkdown", "lme4", "emmeans", "pwr", "ggplot2")

environment_report <- function() {
  cat(strrep("=", 70), "\n")
  cat("ENVIRONMENT RECORD\n")
  cat(strrep("=", 70), "\n")
  cat(sprintf("R version   : %s\n", getRversion()))
  cat(sprintf("Platform    : %s\n", R.version$platform))
  cat(strrep("-", 70), "\n")

  missing <- character(0)
  for (pkg in c(base_recommended, extra_packages)) {
    if (requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("%-14s: %s\n", pkg, as.character(packageVersion(pkg))))
    } else {
      cat(sprintf("%-14s: *** MISSING ***\n", pkg))
      missing <- c(missing, pkg)
    }
  }
  cat(strrep("=", 70), "\n")
  invisible(missing)
}

missing_packages <- environment_report()

#' ## 2. Random number generation, done properly
#'
#' Two rules from `stats.md` Topic 35:
#'
#' * **Seed at the point of use.** Seeding once at the top of a script is
#'   fragile: inserting a new line above shifts every downstream draw. Anything
#'   whose value you will *report* should be immediately preceded by its seed.
#' * **Record the RNG kind.** R's default RNG algorithm has changed between
#'   versions (notably the `sample()` algorithm in R 3.6.0). `RNGkind()` tells
#'   you which one produced a result.

#+ rng-basics
cat("\nRNG kind in use:", paste(RNGkind(), collapse = " / "), "\n")

set.seed(20260919)
sample_a <- rnorm(5)
cat("draw after set.seed(20260919):", paste(round(sample_a, 4), collapse = " "), "\n")

set.seed(20260919)
sample_b <- rnorm(5)
cat("same seed again             :", paste(round(sample_b, 4), collapse = " "), "\n")
cat("identical?                  :", identical(sample_a, sample_b), "\n")

#' ### Parallel-safe streams
#'
#' If you parallelise a simulation (Topic 9 power analysis, Topic 34 calibration
#' checks), workers must **not** share a stream. `"L'Ecuyer-CMRG"` is R's
#' parallel-safe generator; `parallel::mclapply` and `clusterSetRNGStream` use it
#' to create provably independent substreams.

#+ rng-parallel
old_kind <- RNGkind("L'Ecuyer-CMRG")
set.seed(12345)
## .Random.seed now holds a 7-element L'Ecuyer state that can be advanced into
## independent substreams with parallel::nextRNGStream().
worker_first_draws <- numeric(4)
state <- .Random.seed
for (i in seq_len(4)) {
  assign(".Random.seed", state, envir = globalenv())
  worker_first_draws[i] <- rnorm(1)
  state <- parallel::nextRNGStream(state)
}
cat("four independent worker streams:",
    paste(round(worker_first_draws, 4), collapse = " "), "\n")
cat("all distinct?                  :",
    length(unique(worker_first_draws)) == 4, "\n")
do.call(RNGkind, as.list(old_kind))  # restore the default generator

#' ## 3. Output convention
#'
#' Every module writes figures to `results/<module_name>/` **relative to the
#' directory you launch from**, which is why the documented way to run these
#' scripts is from the repository root.

#+ output-dir
MODULE_NAME <- "00_setup_and_environment"

results_dir <- function(module_name = MODULE_NAME) {
  base <- Sys.getenv("STATS_OUT", unset = "results")
  path <- file.path(base, module_name)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

OUT <- results_dir()
cat("figures for this module would be written to:", OUT, "\n")

#' ## 4. A smoke test of every library the course uses
#'
#' Each check exercises the *specific* function that later modules depend on, so
#' a failure points at a real incompatibility rather than a missing import.

#+ smoke-tests
checks <- list()

check <- function(name, condition, detail = "") {
  ok <- isTRUE(condition)
  checks[[length(checks) + 1L]] <<- list(name = name, ok = ok)
  cat(sprintf("[%s] %s%s\n", if (ok) "PASS" else "FAIL", name,
              if (nzchar(detail)) paste0(" - ", detail) else ""))
  invisible(ok)
}

set.seed(7)
x <- rnorm(30, 0, 1)
y <- rnorm(30, 0.5, 1.5)

## --- stats: Welch t-test (stats.md eq. 7.3-7.4) -----------------------------
welch <- t.test(x, y)                      # var.equal = FALSE is the default
check("stats Welch t-test", is.finite(welch$p.value),
      sprintf("p = %.4f", welch$p.value))

## --- stats: a t-test IS a linear model (stats.md eq. 7.5) -------------------
d <- data.frame(y = c(x, y), g = factor(rep(c("a", "b"), each = 30)))
lm_fit <- lm(y ~ g, data = d)
pooled <- t.test(x, y, var.equal = TRUE)
check("lm t-statistic == pooled t-test",
      isTRUE(all.equal(unname(abs(coef(summary(lm_fit))["gb", "t value"])),
                       unname(abs(pooled$statistic)))),
      sprintf("%.6f vs %.6f",
              coef(summary(lm_fit))["gb", "t value"], pooled$statistic))

## --- stats: Benjamini-Hochberg (stats.md eq. 8.5) ---------------------------
set.seed(11)
pvals <- runif(100)
padj <- p.adjust(pvals, method = "BH")
check("BH adjustment monotone", all(diff(sort(padj)) >= -1e-12))

## --- stats: Poisson GLM with an offset (stats.md eq. 13.8) ------------------
set.seed(3)
counts <- rpois(40, 20)
exposure <- runif(40, 0.8, 1.2)
glm_fit <- glm(counts ~ 1, family = poisson(), offset = log(exposure))
check("glm Poisson with offset", glm_fit$converged)

## --- MASS: negative binomial GLM (stats.md eq. 4.9) -------------------------
set.seed(4)
nb_counts <- MASS::rnegbin(60, mu = 30, theta = 5)
nb_fit <- MASS::glm.nb(nb_counts ~ 1)
check("MASS::glm.nb fits", is.finite(nb_fit$theta),
      sprintf("theta = %.2f", nb_fit$theta))

## --- nlme: linear mixed model (stats.md eq. 14.1) ---------------------------
set.seed(5)
donor <- factor(rep(1:8, each = 5))
b <- rnorm(8, 0, 1.5)[donor]
yy <- 2 + b + rnorm(40, 0, 1)
lmm <- nlme::lme(yy ~ 1, random = ~ 1 | donor, data = data.frame(yy, donor))
check("nlme::lme fits random intercept", inherits(lmm, "lme"))

## --- lme4: crossed random effects -------------------------------------------
lmer_fit <- lme4::lmer(yy ~ 1 + (1 | donor), data = data.frame(yy, donor))
check("lme4::lmer fits", inherits(lmer_fit, "lmerMod"))

## --- survival: Kaplan-Meier and Cox (stats.md eq. 28.6, 28.10) --------------
set.seed(6)
tt <- rexp(60, 0.1)
ev <- rbinom(60, 1, 0.7)
grp <- rep(0:1, each = 30)
km <- survival::survfit(survival::Surv(tt, ev) ~ grp)
cox <- survival::coxph(survival::Surv(tt, ev) ~ grp)
check("survival::survfit + coxph",
      length(km$time) > 1 && is.finite(coef(cox)[1]))

## --- emmeans: estimated marginal means and contrasts (eq. 12.3) -------------
em <- emmeans::emmeans(lm_fit, ~ g)
check("emmeans contrasts", nrow(as.data.frame(emmeans::contrast(em, "pairwise"))) == 1)

## --- pwr: closed-form power (stats.md eq. 9.2-9.4) --------------------------
pw <- pwr::pwr.t.test(d = 0.8, sig.level = 0.05, power = 0.8)
check("pwr::pwr.t.test", is.finite(pw$n), sprintf("n = %.1f per group", pw$n))

## --- graphics: a figure can be written head-less ----------------------------
png(file.path(OUT, "smoke_test.png"), width = 400, height = 300)
plot(0:1, 0:1, type = "l", main = "smoke test")
invisible(dev.off())
check("head-less PNG device", file.exists(file.path(OUT, "smoke_test.png")))

#' ## 5. Verdict

#+ verdict
n_failed <- sum(!vapply(checks, `[[`, logical(1), "ok"))
cat("\n", strrep("=", 70), "\n", sep = "")
if (length(missing_packages)) {
  cat("MISSING PACKAGES:", paste(missing_packages, collapse = ", "), "\n")
}
if (n_failed == 0 && length(missing_packages) == 0) {
  cat("ALL CHECKS PASSED - the environment is ready for the whole course.\n")
} else {
  cat(sprintf("%d check(s) failed. Rebuild the image:\n", n_failed))
  cat("  docker build -f docker/Dockerfile.r -t learn-stats-r:1.0 .\n")
}
cat(strrep("=", 70), "\n", sep = "")

#' ## Where to go next
#'
#' | Next | Topic |
#' |---|---|
#' | `01_study_design_and_estimands.R` | What is `n`? The pseudoreplication equation |
#' | `02_data_structures_and_scales.R` | Counts vs intensities vs proportions |
#'
#' **House rules used throughout the course**
#'
#' 1. `set.seed()` immediately precedes anything stochastic you will report.
#' 2. Every claim in a comment cites the equation number in `stats.md`.
#' 3. Nothing is installed at runtime; the container *is* the environment.
#' 
#' **Next:** `01_study_design_and_estimands.R`
