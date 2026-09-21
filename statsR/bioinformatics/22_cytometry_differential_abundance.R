#' ---
#' title: "Applied 22 - Flow / mass cytometry: differential abundance and state"
#' output: html_document
#' ---
#'
#' **Curriculum link:** `stats.md` -> Topic 22, equations (22.1)-(22.4)
#'
#' ## Dataset card
#'
#' | | |
#' |---|---|
#' | **Real analogue** | Bodenmiller / `diffcyt` BCR-XL: 8 paired samples, CyTOF |
#' | **Unit of inference** | **Specimen/donor** (8), not event (~150,000) |
#'
#' `diffcyt` separates two questions: differential ABUNDANCE (did a cluster
#' change size? -> count model with a total-events offset) and differential
#' STATE (did a marker change within a cluster? -> moderated linear model).

#+ setup, message = FALSE
suppressPackageStartupMessages(library(MASS))
MODULE_NAME <- "22_cytometry_differential_abundance"
OUT <- file.path(Sys.getenv("STATS_OUT", unset = "results"), MODULE_NAME)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
header <- function(txt) cat("\n", strrep("=", 72), "\n", txt, "\n",
                            strrep("=", 72), "\n", sep = "")

TYPE_MARKERS <- c("CD3","CD4","CD8","CD19","CD14","CD56")
STATE_MARKERS <- c("pS6","pERK","pSTAT1","pNFkB")
POPULATIONS <- list(`CD4 T` = c(5,4.5,0.2,0.2,0.2,0.2),
                    `CD8 T` = c(5,0.2,4.5,0.2,0.2,0.2),
                    `B cells` = c(0.2,0.2,0.2,5,0.2,0.2),
                    Monocytes = c(0.2,1,0.2,0.2,5,0.2),
                    `NK cells` = c(0.2,0.2,1,0.2,0.2,4.5))
BASE_FREQ <- c(0.33, 0.18, 0.12, 0.28, 0.09)
DA_TRUTH <- c(`B cells` = 0.55)                # B cells expand under stimulation
DS_TRUTH <- list(c("B cells", "pS6", 1.1))     # pS6 rises in B cells

#' ## 1. Simulate event-level CyTOF data

#+ simulate
header("1. Simulating a paired CyTOF experiment")
simulate_cytometry <- function(n_donors = 8, events_per_sample = 5000, cofactor = 5) {
  pops <- names(POPULATIONS)
  frames <- list()
  for (d in seq_len(n_donors)) {
    donor_freq_eff <- rnorm(length(pops), 0, 0.18)
    donor_state_eff <- setNames(rnorm(length(STATE_MARKERS), 0, 0.25), STATE_MARKERS)
    acq_day <- sprintf("day%d", (d - 1) %% 3)
    day_shift <- c(day0 = 0, day1 = 0.22, day2 = -0.15)[acq_day]
    for (cond in c("Ref", "BCRXL")) {
      shift <- sapply(pops, function(p)
        if (cond == "BCRXL" && p %in% names(DA_TRUTH)) DA_TRUTH[[p]] else 0)
      logits <- log(BASE_FREQ) + donor_freq_eff + shift
      probs <- exp(logits)/sum(exp(logits))
      n_ev <- sample(round(0.7*events_per_sample):round(1.3*events_per_sample), 1)
      assign_ <- sample(seq_along(pops), n_ev, TRUE, prob = probs)
      mat <- matrix(0, n_ev, length(TYPE_MARKERS) + length(STATE_MARKERS))
      for (k in seq_along(pops)) {
        sel <- assign_ == k
        if (!any(sel)) next
        mat[sel, seq_along(TYPE_MARKERS)] <-
          matrix(rnorm(sum(sel)*length(TYPE_MARKERS),
                       rep(POPULATIONS[[k]], each = sum(sel)), 0.55), nrow = sum(sel))
        for (jm in seq_along(STATE_MARKERS)) {
          sm <- STATE_MARKERS[jm]
          eff <- 0
          for (ds in DS_TRUTH)
            if (pops[k] == ds[1] && sm == ds[2] && cond == "BCRXL") eff <- as.numeric(ds[3])
          mat[sel, length(TYPE_MARKERS) + jm] <-
            rnorm(sum(sel), 1 + donor_state_eff[sm] + day_shift + eff, 0.6)
        }
      }
      ## Convert the arcsinh-scale values back to RAW intensities, so the
      ## script has to transform them like real FCS data (eq. 2.5).
      raw <- sinh(pmax(pmin(mat, 12), -5)) * cofactor
      df <- as.data.frame(raw)
      names(df) <- c(TYPE_MARKERS, STATE_MARKERS)
      df$sample_id <- sprintf("D%02d_%s", d - 1, cond)
      df$donor <- sprintf("D%02d", d - 1); df$condition <- cond
      df$acq_day <- acq_day; df$true_pop <- pops[assign_]
      frames[[length(frames) + 1]] <- df
    }
  }
  do.call(rbind, frames)
}
set.seed(3201)
ev <- simulate_cytometry()
cat(sprintf("  %s events, %d samples, %d donors\n", format(nrow(ev), big.mark = ","),
            length(unique(ev$sample_id)), length(unique(ev$donor))))
cat(sprintf("  events per sample: %s - %s\n",
            format(min(table(ev$sample_id)), big.mark = ","),
            format(max(table(ev$sample_id)), big.mark = ",")))
cat(sprintf("\n  n for CONDITION-level inference = %d donors, NOT %s events\n",
            length(unique(ev$donor)), format(nrow(ev), big.mark = ",")))
cat(sprintf("  raw intensity range: %.1f to %s (and some are negative)\n",
            min(ev[, TYPE_MARKERS]), format(round(max(ev[, TYPE_MARKERS])), big.mark=",")))

#' ## 2. Transformation, eq. (2.5)

#+ arcsinh
header("2. arcsinh transformation with a modality-appropriate cofactor (2.5)")
cat(sprintf("  %10s%13s%27s\n", "cofactor", "SD of CD19", "bimodality (CD19)"))
for (cf in c(1, 5, 150)) {
  t_ <- asinh(ev$CD19/cf)
  km <- kmeans(matrix(t_), 2, nstart = 10)
  within <- mean(sapply(1:2, function(k) var(t_[km$cluster == k])))
  cat(sprintf("  %10.0f%13.3f%27.3f\n", cf, sd(t_), 1 - within/var(t_)))
}
cat("\n  c=5 gives the cleanest separation for these mass-cytometry-scale\n")
cat("  intensities. c=150 is typical for fluorescence flow. Report the\n")
cat("  cofactor: it changes every downstream clustering result. Note asinh()\n")
cat("  handles the NEGATIVE values compensation produces, where log() cannot.\n")
MARKERS <- c(TYPE_MARKERS, STATE_MARKERS)
Xt <- as.data.frame(asinh(as.matrix(ev[, MARKERS])/5))
Xt[c("sample_id","donor","condition","acq_day","true_pop")] <-
  ev[c("sample_id","donor","condition","acq_day","true_pop")]

#' ## 3. Clustering on TYPE markers only

#+ clustering
header("3. High-resolution clustering on lineage markers")
set.seed(3202)
km <- kmeans(Xt[, TYPE_MARKERS], centers = 12, nstart = 25)
Xt$cluster <- km$cluster
ann <- sapply(sort(unique(Xt$cluster)), function(c_)
  names(which.max(table(Xt$true_pop[Xt$cluster == c_]))))
names(ann) <- sort(unique(Xt$cluster))
cat(sprintf("  %8s%11s%13s%9s\n", "cluster", "n events", "annotation", "purity"))
for (c_ in sort(unique(Xt$cluster))) {
  sel <- Xt$cluster == c_
  cat(sprintf("  %8d%11s%13s%9.3f\n", c_, format(sum(sel), big.mark = ","),
              ann[as.character(c_)], mean(Xt$true_pop[sel] == ann[as.character(c_)])))
}
Xt$population <- ann[as.character(Xt$cluster)]
cat("\n  Over-clustering (12 clusters for 5 populations) is deliberate - the\n")
cat("  `diffcyt` strategy. Rare subsets get their own cluster, and merging is\n")
cat("  a reversible annotation decision made afterwards.\n")

#' ## 4. Event level -> sample level, eq. (22.1)

#+ sample-level
header("4. Reducing to sample-level tables (22.1)")
counts <- table(Xt$sample_id, Xt$cluster)
sample_meta <- data.frame(
  donor = sapply(rownames(counts), function(s) Xt$donor[Xt$sample_id == s][1]),
  condition = sapply(rownames(counts), function(s) Xt$condition[Xt$sample_id == s][1]),
  acq_day = sapply(rownames(counts), function(s) Xt$acq_day[Xt$sample_id == s][1]),
  n_events = as.integer(rowSums(counts)), row.names = rownames(counts))
sample_meta$condition <- factor(sample_meta$condition, levels = c("Ref", "BCRXL"))
cat(sprintf("  counts table: %d samples x %d clusters\n", nrow(counts), ncol(counts)))
print(counts[1:4, 1:6])
medians <- aggregate(Xt[, STATE_MARKERS],
                     by = list(sample_id = Xt$sample_id, cluster = Xt$cluster), median)
cat(sprintf("\n  medians table: %d (sample x cluster) rows x %d state markers\n",
            nrow(medians), length(STATE_MARKERS)))
cat("\n  Eq. (22.4): RSE ~ 1/sqrt(Y). Minimum-event guidance:\n")
for (y in c(10, 25, 100, 400, 2500))
  cat(sprintf("    %6d events -> RSE %5.1f%%\n", y, 100/sqrt(y)))
cat(sprintf("  %d of %d sample x cluster cells have < 25 events\n",
            sum(counts < 25), length(counts)))

#' ## 5. Differential abundance, eq. (22.2)

#+ da
header("5. Differential abundance: counts with a total-events offset (22.2)")
da_test <- function(counts, meta, extra = "") {
  ## Poisson GLM with a log(total events) offset, then QUASI-LIKELIHOOD
  ## inference: the dispersion is estimated per cluster (eq. 13.9) and every
  ## SE multiplied by sqrt(phi_hat). Estimating (not fixing) the dispersion is
  ## what lets a blocking term pay off.
  res <- t(vapply(colnames(counts), function(c_) {
    d <- meta; d$y <- as.numeric(counts[, c_])
    d$stim <- as.numeric(d$condition == "BCRXL")
    f <- try(glm(as.formula(paste("y ~ stim", extra)), data = d,
                 family = poisson(), offset = log(d$n_events)), silent = TRUE)
    if (inherits(f, "try-error")) return(c(NA, NA, 1))
    phi <- max(sum(residuals(f, "pearson")^2)/df.residual(f), 1)
    se <- coef(summary(f))["stim", 2]*sqrt(phi)
    tstat <- coef(f)["stim"]/se
    c(coef(f)["stim"], phi, 2*pt(abs(tstat), df.residual(f), lower.tail = FALSE))
  }, numeric(3)))
  out <- data.frame(logFC = res[, 1], phi = res[, 2], pvalue = res[, 3],
                    row.names = colnames(counts))
  out$padj <- p.adjust(out$pvalue, "BH")
  out$population <- ann[rownames(out)]
  out
}
res_da <- da_test(counts, sample_meta)
res_da_p <- da_test(counts, sample_meta, "+ factor(donor)")
prop <- prop.table(counts, 1)
cat(sprintf("  %8s%12s%12s%13s%9s%10s%15s\n", "cluster", "population", "mean % Ref",
            "mean % Stim", "logFC", "padj", "padj (paired)"))
for (c_ in rownames(res_da)) {
  pr <- mean(prop[sample_meta$condition == "Ref", c_])
  ps <- mean(prop[sample_meta$condition == "BCRXL", c_])
  star <- if (res_da$population[rownames(res_da) == c_] %in% names(DA_TRUTH)) "  *TRUE*" else ""
  cat(sprintf("  %8s%12s%11.3f%%%12.3f%%%9.3f%10.4f%15.4f%s\n", c_,
              res_da[c_, "population"], 100*pr, 100*ps, res_da[c_, "logFC"],
              res_da[c_, "padj"], res_da_p[c_, "padj"], star))
}
cat(sprintf("\n  median estimated dispersion, unpaired : %.2f\n", median(res_da$phi)))
cat(sprintf("  median estimated dispersion, paired   : %.2f\n", median(res_da_p$phi)))
cat("  The paired model pulls the donor-to-donor composition variance out of\n")
cat("  the residual, the quasi-dispersion falls, and every SE shrinks (eq. 4.3).\n")
extra_hits <- rownames(res_da_p)[res_da_p$padj < 0.05 &
                                   !(res_da_p$population %in% names(DA_TRUTH))]
cat(sprintf("\n  NOW LOOK CAREFULLY at what the more powerful model found. Besides\n"))
cat(sprintf("  the true B-cell clusters it also flagged %d others, all with a\n",
            length(extra_hits)))
cat("  NEGATIVE logFC. Those are not shrinking populations - they are being\n")
cat("  DILUTED, because when B cells expand every other cluster's share of a\n")
cat("  fixed event total must fall (eq. 2.7). Extra power does not rescue you\n")
cat("  from the constraint; it just lets you detect the artefact confidently.\n")
cat("\n  A claim about ABSOLUTE cell numbers needs an absolute measurement\n")
cat("  (counting beads, or cells per mL), not a frequency table.\n")

#' ## 6. Differential state, eq. (22.3)

#+ ds
header("6. Differential state: linear models on per-sample medians (22.3)")
ds_test <- function(medians, meta, markers, paired = TRUE) {
  rows <- list()
  for (c_ in sort(unique(medians$cluster))) {
    sub <- medians[medians$cluster == c_, ]
    d <- merge(sub, cbind(sample_id = rownames(meta), meta), by = "sample_id")
    if (length(unique(d$condition)) < 2 || nrow(d) < 6) next
    d$stim <- as.numeric(d$condition == "BCRXL")
    for (m in markers) {
      f <- try(lm(as.formula(paste(m, "~ stim", if (paired) "+ factor(donor)" else "")),
                  data = d), silent = TRUE)
      if (inherits(f, "try-error")) next
      cs <- coef(summary(f))
      rows[[length(rows)+1]] <- data.frame(cluster = c_, marker = m,
                                           diff = cs["stim",1], pvalue = cs["stim",4])
    }
  }
  out <- do.call(rbind, rows)
  out$padj <- p.adjust(out$pvalue, "BH")
  out$population <- ann[as.character(out$cluster)]
  out
}
res_ds <- ds_test(medians, sample_meta, STATE_MARKERS, TRUE)
hits <- res_ds[res_ds$padj < 0.05, ]
hits <- hits[order(hits$padj), ]
cat(sprintf("  multiplicity family = %d clusters x %d markers = %d hypotheses\n",
            length(unique(res_ds$cluster)), length(STATE_MARKERS), nrow(res_ds)))
cat(sprintf("\n  Significant (cluster, marker) pairs at FDR 5%%:\n"))
cat(sprintf("  %8s%12s%9s%12s%11s\n", "cluster", "population", "marker",
            "difference", "padj"))
is_true_ds <- function(pop, mk) any(sapply(DS_TRUTH, function(z) z[1]==pop && z[2]==mk))
tp <- 0
for (i in seq_len(min(nrow(hits), 12))) {
  r <- hits[i, ]
  st <- if (is_true_ds(r$population, r$marker)) { tp <- tp + 1; "  *TRUE*" } else ""
  cat(sprintf("  %8s%12s%9s%12.3f%11.2e%s\n", r$cluster, r$population, r$marker,
              r$diff, r$padj, st))
}
tp_all <- sum(mapply(is_true_ds, hits$population, hits$marker))
cat(sprintf("\n  %d of %d hits are the injected pS6-in-B-cells effect (%d false).\n",
            tp_all, nrow(hits), nrow(hits) - tp_all))
cat("  Note the contrast with differential ABUNDANCE above: DS is a\n")
cat("  WITHIN-cluster question, so it is NOT subject to the compositional\n")
cat("  constraint. A marker can rise in B cells without forcing anything to\n")
cat("  fall elsewhere. State which of the two questions you are answering.\n")

#' ## 7. Acquisition day is a batch effect

#+ batch
header("7. Modelling acquisition day (Topic 19)")
summarise_ds <- function(res, label) {
  h <- res[res$padj < 0.05, ]
  t_ <- sum(mapply(is_true_ds, h$population, h$marker))
  cat(sprintf("  %-44s%7d%7d%7d\n", label, nrow(h), t_, nrow(h) - t_))
}
cat(sprintf("  %-44s%7s%7s%7s\n", "model", "hits", "TP", "FP"))
summarise_ds(ds_test(medians, sample_meta, STATE_MARKERS, FALSE), "~ stim (no blocking)")
summarise_ds(res_ds, "~ stim + donor (paired)")
cat("\n  Because each donor was acquired on ONE day, the donor term already\n")
cat("  absorbs the acquisition-day effect. When donors span days, include BOTH\n")
cat("  terms and check the design rank (Module 01, section 6) before fitting.\n")

#' ## 8. Figure

#+ figure
png(file.path(OUT, "cytometry.png"), width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4.2, 4.2, 2.5, 1))
plot(density(asinh(ev$CD19[1:20000]/5)), col = "steelblue", lwd = 2,
     main = "Eq. (2.5): the cofactor sets the linear region", xlab = "arcsinh(CD19/c)")
lines(density(asinh(ev$CD19[1:20000]/1)), col = "darkorange", lwd = 2)
lines(density(asinh(ev$CD19[1:20000]/150)), col = "forestgreen", lwd = 2)
legend("topright", c("c=5","c=1","c=150"),
       col = c("steelblue","darkorange","forestgreen"), lwd = 2, bty = "n", cex = 0.65)
sm <- Xt[sample(nrow(Xt), 8000), ]
plot(sm$CD19, sm$CD3, pch = ".", col = factor(sm$population),
     xlab = "arcsinh CD19", ylab = "arcsinh CD3",
     main = "Clusters annotated by lineage markers")
matplot(t(prop), type = "p", pch = ifelse(sample_meta$condition == "Ref", 1, 16),
        col = ifelse(ann[colnames(prop)] %in% names(DA_TRUTH), "firebrick", "grey50"),
        xlab = "cluster", ylab = "proportion of events",
        main = "Eq. (22.2): DA (red = truly expanded)")
piv <- reshape(res_ds[, c("cluster","marker","diff")], idvar = "cluster",
               timevar = "marker", direction = "wide")
mat <- as.matrix(piv[, -1]); rownames(mat) <- piv$cluster
image(seq_len(ncol(mat)), seq_len(nrow(mat)), t(mat), col = hcl.colors(30, "RdBu", rev = TRUE),
      xaxt = "n", yaxt = "n", xlab = "", ylab = "",
      main = "Eq. (22.3): DS effect per cluster x marker")
axis(1, seq_len(ncol(mat)), sub("diff\\.", "", colnames(mat)), cex.axis = 0.6)
axis(2, seq_len(nrow(mat)), rownames(mat), las = 1, cex.axis = 0.5)
invisible(dev.off())
cat("\nFigure written to", file.path(OUT, "cytometry.png"), "\n")

#' # PROBLEMS
#'
#' ### Problem 1: Event-level testing
#'
#' Test pS6 in the B-cell clusters using EVERY EVENT as an observation, and
#' compare to the sample-level median analysis. Do it for a marker with a TRUE
#' effect and one WITHOUT. Predict the direction before running.

#+ problem1
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# bcl <- names(ann)[ann == "B cells"]
# sub <- Xt[Xt$cluster %in% as.integer(bcl), ]
# for (marker in c("pS6", "pERK", "pSTAT1")) {
#   aa <- sub[[marker]][sub$condition == "Ref"]
#   bb <- sub[[marker]][sub$condition == "BCRXL"]
#   pe <- t.test(aa, bb)$p.value
#   ss <- aggregate(sub[[marker]], by = list(sid = sub$sample_id), median)
#   ss$cond <- sample_meta[ss$sid, "condition"]
#   ps <- t.test(x ~ cond, data = ss)$p.value
#   truth_ <- if (any(sapply(DS_TRUTH, function(z) z[1]=="B cells" && z[2]==marker)))
#     "TRUE effect" else "no effect"
#   cat(sprintf("  %-8s (%-11s): event p=%.2e  sample p=%.4f\n",
#               marker, truth_, pe, ps))
# }
#
# ## The event-level p-values are astronomically small for EVERY marker,
# ## including those with no true effect, because ~50,000 events are treated as
# ## independent when there are only 8 donors (eq. 1.8). The sample-level
# ## analysis separates the real effect from the rest.

#' ### Problem 2: Minimum-event threshold
#'
#' Re-run the DS analysis keeping only (sample x cluster) combinations with at
#' least 5, 25 and 200 events. Use eq. (22.4) to justify a threshold.

#+ problem2
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# ev_counts <- as.data.frame(table(sample_id = Xt$sample_id, cluster = Xt$cluster))
# names(ev_counts)[3] <- "n_ev"
# ev_counts$cluster <- as.integer(as.character(ev_counts$cluster))
# med2 <- merge(medians, ev_counts, by = c("sample_id", "cluster"))
# cat(sprintf("  %12s%12s%7s%6s%6s\n", "min events", "rows kept", "hits", "TP", "FP"))
# for (thr in c(1, 5, 25, 200)) {
#   keep <- med2[med2$n_ev >= thr, setdiff(names(med2), "n_ev")]
#   r <- ds_test(keep, sample_meta, STATE_MARKERS, TRUE)
#   h <- r[r$padj < 0.05, ]
#   t_ <- sum(mapply(is_true_ds, h$population, h$marker))
#   cat(sprintf("  %12d%12d%7d%6d%6d\n", thr, nrow(keep), nrow(h), t_, nrow(h)-t_))
# }
#
# ## A median from 3 events is almost pure noise - eq. (22.4) gives an RSE of
# ## ~58%. Raising the threshold removes those unstable rows. Set it BEFORE
# ## looking at the outcome, and report how many rows it dropped.

#' ### Problem 3: Cluster-resolution sensitivity
#'
#' Repeat the DA analysis at k = 6, 12 and 25 clusters. Does the B-cell
#' expansion survive? What happens to the multiplicity burden?

#+ problem3
## ---- YOUR CODE HERE ----------------------------------------------------

## ---- SOLUTION (uncomment to check) -------------------------------------
# for (k in c(6, 12, 25)) {
#   set.seed(1)
#   kmk <- kmeans(Xt[, TYPE_MARKERS], k, nstart = 25)
#   annk <- sapply(sort(unique(kmk$cluster)), function(c_)
#     names(which.max(table(Xt$true_pop[kmk$cluster == c_]))))
#   names(annk) <- sort(unique(kmk$cluster))
#   ck <- table(Xt$sample_id, kmk$cluster)[rownames(sample_meta), ]
#   pk <- sapply(colnames(ck), function(c_) {
#     d <- sample_meta; d$y <- as.numeric(ck[, c_])
#     d$stim <- as.numeric(d$condition == "BCRXL")
#     f <- glm(y ~ stim + factor(donor), data = d, family = poisson(),
#              offset = log(d$n_events))
#     phi <- max(sum(residuals(f, "pearson")^2)/df.residual(f), 1)
#     2*pt(abs(coef(f)["stim"]/(coef(summary(f))["stim",2]*sqrt(phi))),
#          df.residual(f), lower.tail = FALSE)
#   })
#   sig <- names(pk)[p.adjust(pk, "BH") < 0.05]
#   cat(sprintf("  k=%3d: %d significant clusters, %d of them B cells, %d hypotheses\n",
#               k, length(sig), sum(annk[sig] == "B cells"), k))
# }
#
# ## The B-cell expansion is robust to resolution - that is the point of a
# ## sensitivity analysis (Module 34). Higher k splits populations, raising the
# ## multiplicity burden and potentially diluting an effect across siblings.
# ## Report the range, not one k.

#' ## What to take away
#'
#' 1. **Specimen/donor is n**; events are not replicates.
#' 2. arcsinh with a stated cofactor; cluster on lineage markers only.
#' 3. Counts + total-events offset for abundance; linear models on per-sample
#'    medians for state.
#' 4. Pre-specify the minimum-event rule and cluster resolution; report
#'    sensitivity to both.
#' 5. Multiplicity family = clusters x markers x contrasts.
#'
#' **Next:** `23_methylation_epigenomics.R`
