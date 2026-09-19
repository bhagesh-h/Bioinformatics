# ============================================================================
# R environment for "Statistics for Bioinformatics"
# ----------------------------------------------------------------------------
# This file is BOTH documentation and an installer. It is executed during the
# Docker build; you never need to run it on the host.
#
#     docker build -f docker/Dockerfile.r -t learn-stats-r:1.0 .
#
# Design decision: the course deliberately leans on base R + the *recommended*
# packages that ship with every R installation (MASS, nlme, mgcv, survival,
# cluster, boot, Matrix, splines, lattice). That keeps ~90% of the material
# runnable in a bare `R` with no network access at all. Only the packages
# below are genuinely extra.
# ============================================================================

# --- Already present in any standard R (listed here for reference only) -----
base_recommended <- c(
  "stats",     # distributions, lm/glm/aov, p.adjust, prcomp, optim, ...
  "MASS",      # glm.nb (negative binomial), rlm (robust), lda, mvrnorm
  "nlme",      # lme(): linear mixed models, correlation structures
  "mgcv",      # gam(): splines / generalised additive models
  "survival",  # Surv, survfit (Kaplan-Meier), survdiff (log-rank), coxph
  "cluster",   # pam, silhouette, agnes
  "boot",      # bootstrap resampling and bootstrap CIs
  "Matrix",    # sparse matrix classes
  "splines",   # ns(), bs() basis functions
  "lattice"    # trellis graphics (used by nlme plots)
)

# --- Extra packages installed into the image --------------------------------
extra_packages <- c(
  "knitr",     # spin(): .R -> .Rmd, and chunk evaluation
  "rmarkdown", # render(): .Rmd -> HTML notebook
  "lme4",      # lmer/glmer: crossed random effects, GLMMs
  "emmeans",   # estimated marginal means and custom contrasts
  "pwr",       # closed-form power/sample-size calculations
  "ggplot2"    # optional grammar-of-graphics layer (base graphics also used)
)

# Only run the installation when this file is sourced non-interactively as an
# installer (i.e. during the Docker build), never by accident from a session.
if (!interactive()) {
  options(
    repos = c(CRAN = Sys.getenv("CRAN", "https://cloud.r-project.org")),
    Ncpus = max(1L, parallel::detectCores())
  )
  missing <- extra_packages[!vapply(extra_packages, requireNamespace,
                                    logical(1), quietly = TRUE)]
  if (length(missing)) install.packages(missing)

  # Fail the build loudly if anything did not install.
  still_missing <- extra_packages[!vapply(extra_packages, requireNamespace,
                                          logical(1), quietly = TRUE)]
  if (length(still_missing)) {
    stop("Failed to install: ", paste(still_missing, collapse = ", "))
  }
  cat("All R packages available.\n")
}
