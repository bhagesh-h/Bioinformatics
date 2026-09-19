# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 2: A differential expression analysis, end to end
#
# **Curriculum link:** `stats.md` -> Topics 8, 13, 20, 29
# **Core modules used:** 08, 13, 20, 30, 39
#
# ## The brief
#
# You are handed a raw count matrix: 12 samples (6 control, 6 treated),
# 4,000 genes. Nobody tells you how it was made. Build the analysis yourself
# and, at every step, **check the step rather than trusting it**.
#
# The five questions follow the order of a real pipeline:
#
# | Q | Step | The thing that goes wrong |
# |---|---|---|
# | 1 | Normalisation | composition bias: a few huge genes distort every library |
# | 2 | Dispersion | per-gene estimates are hopeless at n = 6 |
# | 3 | Testing + FDR | the test is fine; the filtering is where power is lost |
# | 4 | Enrichment | inter-gene correlation inflates set-level p-values |
# | 5 | Negative control | does the whole pipeline return nothing on a null? |
#
# Question 5 is the one that matters most and the one almost nobody runs.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import polygamma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

MODULE_NAME = "E2_rnaseq_end_to_end"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## The data

# %%
def make_counts(seed=7, n_per_group=6, n_genes=4000, n_de=300,
                composition_bias=True):
    """A negative-binomial count matrix with realistic pathologies.

    Built in deliberately: very different library sizes, a mean-dependent
    dispersion trend, correlated gene modules, and - if `composition_bias` -
    a handful of genes that become enormous in the treated group and so eat
    a large share of every treated library's reads.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    group = np.repeat([0, 1], n_per_group)
    # True expression level per gene, log-normal as real data are.
    base = np.exp(rng.normal(4.0, 1.6, n_genes))
    # Mean-dispersion trend: low-count genes are much noisier (eq. 20.3).
    disp = 0.10 + 18.0 / (base + 12.0)
    lfc = np.zeros(n_genes)
    de_idx = rng.choice(n_genes, n_de, replace=False)
    lfc[de_idx] = rng.normal(0, 1.1, n_de)
    # Correlated modules, so gene-set tests face realistic dependence. Each
    # module responds to the PROCESSING BATCH, plus a smaller unmeasured
    # latent factor. Both move all of a module's genes together, which is what
    # makes genes in a pathway non-independent.
    #
    # Batch is balanced within each group (3 of each batch per arm), so it is
    # NOT confounded with the treatment - but leaving it out of the design
    # still inflates the error rate, because it makes whole modules move.
    batch = np.concatenate([np.tile([0, 1], n_per_group // 2 + 1)[:n_per_group]] * 2)
    n_mod, mod_size = 12, 60
    modules = {}
    mod_factor = np.zeros((n_genes, n))
    for k in range(n_mod):
        members = np.arange(k * mod_size, (k + 1) * mod_size)
        modules[f"module_{k:02d}"] = members
        mod_factor[members] = (rng.normal(0, 0.55) * batch[None, :]
                               + rng.normal(0, 0.18, n)[None, :])
    # Modules 0 and 1 are genuinely up-regulated; the rest are not.
    for k in (0, 1):
        lfc[modules[f"module_{k:02d}"]] = rng.normal(0.9, 0.25, mod_size)
        de_idx = np.union1d(de_idx, modules[f"module_{k:02d}"])
    # Library size varies 4-fold, as it does in practice.
    lib = rng.uniform(0.5, 2.0, n)
    mu = (base[:, None] * np.exp(np.log(2) * lfc[:, None] * group[None, :])
          * np.exp(mod_factor) * lib[None, :])
    if composition_bias:
        # 5 genes explode in the treated group: not an artefact, real biology,
        # but it steals sequencing depth from every other treated gene.
        hogs = np.argsort(base)[-5:]
        mu[np.ix_(hogs, group == 1)] *= 60
    # Renormalise each column to a fixed total: sequencing measures PROPORTIONS.
    mu = mu / mu.sum(0, keepdims=True) * (8e6 * lib)
    counts = rng.negative_binomial(1.0 / disp[:, None],
                                   1.0 / (1.0 + disp[:, None] * mu),
                                   size=(n_genes, n))
    genes = np.array([f"g{i:04d}" for i in range(n_genes)])
    return dict(counts=counts, group=group, batch=batch, genes=genes, lfc_true=lfc,
                is_de=np.isin(np.arange(n_genes), de_idx), modules=modules,
                disp_true=disp, base=base)


header("The count matrix as received")
D = make_counts()
counts, group, genes, batch = D["counts"], D["group"], D["genes"], D["batch"]
print(f"  {counts.shape[0]} genes x {counts.shape[1]} samples "
      f"({(group==0).sum()} control, {(group==1).sum()} treated)")
print(f"  library sizes (millions): "
      f"{np.array2string(counts.sum(0)/1e6, precision=2)}")
print(f"  ratio largest/smallest library = {counts.sum(0).max()/counts.sum(0).min():.2f}")
print(f"  genes with zero counts in all samples: {(counts.sum(1)==0).sum()}")
print(f"  median count: {np.median(counts):.0f}, max count: {counts.max():,}")
print(f"  group : {group}")
print(f"  batch : {batch}   <- recorded in the sample sheet, balanced within group")

# %% [markdown]
# ### Q1: Normalisation
#
# Compare three size factors: total-count (CPM), upper-quartile, and a
# median-of-ratios factor (DESeq2's, eq. 20.1). Which recovers the truth, and
# how would you *detect* the problem without knowing the truth?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def sf_total(c):
#     s = c.sum(0); return s / np.exp(np.mean(np.log(s)))
#
# def sf_upperquartile(c):
#     q = np.array([np.quantile(c[c[:, j] > 0, j], 0.75) for j in range(c.shape[1])])
#     return q / np.exp(np.mean(np.log(q)))
#
# def sf_median_ratio(c):
#     """DESeq2 median-of-ratios (eq. 20.1).
#
#     The geometric mean across samples is a per-gene REFERENCE; a sample's
#     size factor is the median of its ratios to that reference. Because it is
#     a MEDIAN, a handful of exploding genes cannot move it.
#     """
#     keep = (c > 0).all(1)                      # geometric mean needs no zeros
#     logc = np.log(c[keep])
#     ref = logc.mean(1, keepdims=True)
#     return np.exp(np.median(logc - ref, axis=0))
#
# sfs = {"total count (CPM)": sf_total(counts),
#        "upper quartile": sf_upperquartile(counts),
#        "median of ratios": sf_median_ratio(counts)}
# # The TRUE size factor is the library-size multiplier used in the simulation,
# # which we can recover because we know the design.
# print(f"  {'method':<22}{'ctrl mean':>11}{'trt mean':>11}{'trt/ctrl':>11}")
# for name, sf in sfs.items():
#     print(f"  {name:<22}{sf[group==0].mean():>11.3f}{sf[group==1].mean():>11.3f}"
#           f"{sf[group==1].mean()/sf[group==0].mean():>11.3f}")
#
# # How well does each recover the truth? Compare the median log-ratio of
# # NON-DE genes after normalisation: it should be 0.
# nde = ~D["is_de"]
# print(f"\n  {'method':<22}{'median logFC of NON-DE genes':>32}")
# for name, sf in sfs.items():
#     norm = counts / sf
#     a = np.log2(norm[nde][:, group == 1].mean(1) + 1)
#     b = np.log2(norm[nde][:, group == 0].mean(1) + 1)
#     print(f"  {name:<22}{np.median(a - b):>32.3f}")
#
# # DIAGNOSIS WITHOUT THE TRUTH: plot an MA of raw data, or simply check what
# # fraction of each library the top genes consume.
# frac = np.sort(counts, axis=0)[-5:].sum(0) / counts.sum(0)
# print(f"\n  share of library taken by the top 5 genes:")
# print(f"    control : {frac[group==0].mean():.1%}")
# print(f"    treated : {frac[group==1].mean():.1%}   <- composition bias")
#
# # Total-count normalisation divides by a number that the 5 exploding genes
# # dominate, so every OTHER treated gene is scaled down and appears
# # DOWN-regulated: the median non-DE logFC shifts away from 0. Median-of-ratios
# # and upper-quartile are robust to a few enormous genes and keep it near 0.
# # This is exactly why DESeq2/edgeR do not use CPM for testing, and why
# # "normalise to total reads" is wrong whenever composition changes - the same
# # closure problem as microbiome data (Topic 26).

# %% [markdown]
# ### Q2: Dispersion
#
# Estimate the NB dispersion per gene by method of moments, then shrink it
# towards a fitted mean-dispersion trend (eq. 20.4). Show numerically why the
# raw per-gene estimate must not be used at n = 6.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# sf = sf_median_ratio(counts)
# norm = counts / sf
# keep = norm.mean(1) >= 1     # keep almost everything; Q3 studies filtering
# nrm, dtrue = norm[keep], D["disp_true"][keep]
# mu_hat = nrm.mean(1)
# var_hat = nrm.var(1, ddof=1)
# # NB: Var = mu + disp*mu^2  ->  disp = (Var - mu)/mu^2   (eq. 20.2)
# disp_mom = np.clip((var_hat - mu_hat) / mu_hat**2, 1e-4, None)
# # Trend: dispersion as a smooth function of mean (eq. 20.3). Fit in log space.
# o = np.argsort(mu_hat)
# logmu, logd = np.log(mu_hat[o]), np.log(disp_mom[o])
# w = 151
# trend = np.convolve(logd, np.ones(w) / w, mode="same")
# trend[:w] = trend[w]; trend[-w:] = trend[-w - 1]
# disp_trend = np.empty_like(disp_mom); disp_trend[o] = np.exp(trend)
# # Empirical-Bayes shrinkage towards the trend (eq. 20.4). The weight depends
# # on the residual degrees of freedom, n - 2 = 10 here.
# dfres = counts.shape[1] - 2
# logres = np.log(disp_mom) - np.log(disp_trend)
# prior_var = max(np.var(logres, ddof=1) - polygamma(1, dfres / 2), 0.01)
# wt = prior_var / (prior_var + polygamma(1, dfres / 2))
# disp_shrunk = np.exp(np.log(disp_trend) + wt * logres)
# print(f"  {keep.sum()} genes kept (mean normalised count >= 1)")
# print(f"  residual df = {dfres}, shrinkage weight towards the trend = {1-wt:.2f}")
# print(f"\n  {'estimator':<26}{'corr with truth':>18}{'median |error|':>18}")
# for name, est in [("per-gene MoM", disp_mom), ("trend only", disp_trend),
#                   ("shrunk (empirical Bayes)", disp_shrunk)]:
#     print(f"  {name:<26}{np.corrcoef(np.log(est), np.log(dtrue))[0,1]:>18.3f}"
#           f"{np.median(np.abs(np.log(est) - np.log(dtrue))):>18.3f}")
#
# # With 10 residual degrees of freedom a variance estimate has a relative
# # standard error of about sqrt(2/10) = 45%, so the per-gene dispersion is
# # extremely noisy - and the error is worst exactly where it hurts, in the
# # low-count genes. Shrinking each gene towards a trend fitted across
# # thousands of genes borrows strength (Topic 33) and is the single reason
# # DESeq2/edgeR work at n = 3-6. Note the trend ALONE is already better than
# # the per-gene estimate; the shrunk version is better than either.

# %% [markdown]
# ### Q3: Testing and FDR
#
# Fit a negative-binomial GLM per gene with an offset for the size factor,
# apply BH, and then show what **independent filtering** buys. Report the
# realised false discovery proportion: you know the truth here.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# kept_idx = np.where(keep)[0]
# truth_kept = D["is_de"][kept_idx]
# in_mod = np.isin(kept_idx, np.concatenate(list(D["modules"].values())))
# offset = np.log(sf)
#
# def fit_all(design_cols, test="lrt"):
#     """Per-gene NB GLM. `design_cols` are the columns BESIDES the intercept;
#     the first one is the one we test."""
#     X1 = sm.add_constant(np.column_stack(design_cols).astype(float))
#     rest = design_cols[1:]
#     X0 = (sm.add_constant(np.column_stack(rest).astype(float)) if rest
#           else np.ones((len(group), 1)))
#     p = np.ones(len(kept_idx)); lfc = np.zeros(len(kept_idx))
#     for i, gi in enumerate(kept_idx):
#         fam = sm.families.NegativeBinomial(alpha=disp_shrunk[i])
#         try:
#             f1 = sm.GLM(counts[gi], X1, family=fam, offset=offset).fit()
#             lfc[i] = f1.params[1] / np.log(2)
#             if test == "wald":
#                 p[i] = f1.pvalues[1]
#             else:
#                 f0 = sm.GLM(counts[gi], X0, family=fam, offset=offset).fit()
#                 p[i] = st.chi2.sf(2 * (f1.llf - f0.llf), 1)
#         except Exception:
#             p[i] = 1.0
#     return p, lfc
#
# # --- (a) Is the TEST calibrated? Measure the null p-value rate directly. ---
# print("  Rate of p < 0.05 among genes with NO true effect (should be 0.05):\n")
# print(f"  {'design':<20}{'test':<8}{'all nulls':>12}{'in module':>12}{'elsewhere':>12}")
# cal = {}
# for dname, cols in [("~ group", [group]), ("~ group + batch", [group, batch])]:
#     for tname in ("wald", "lrt"):
#         pv, lf = fit_all(cols, tname)
#         cal[(dname, tname)] = (pv, lf)
#         nul = ~truth_kept
#         print(f"  {dname:<20}{tname.upper():<8}"
#               f"{(pv[nul] < 0.05).mean():>12.4f}"
#               f"{(pv[nul & in_mod] < 0.05).mean():>12.4f}"
#               f"{(pv[nul & ~in_mod] < 0.05).mean():>12.4f}")
#
# # Read that table carefully - one result is expected and one is not.
# #
# #   * WALD is anticonservative and the LRT is closer to nominal, under both
# #     designs. A Wald test uses the curvature of the likelihood AT THE
# #     ESTIMATE; with 12 samples that quadratic approximation is poor and it
# #     errs towards small p-values (the extreme case is the Hauck-Donner
# #     effect, Topic 13). Prefer the LRT at small n - it is what edgeR's
# #     `glmLRT` does.
# #
# #   * Adding `batch` to the design lowers the IN-MODULE null rate, which is
# #     the column it should affect, and leaves the others essentially alone.
# #     Modest, but in the right place.
# #
# #   * The surprise: in-module genes are NOT worse than the rest, even in the
# #     `~ group` design that ignores batch entirely. Check why -
# #
# #         ratio of ESTIMATED to TRUE dispersion, per-gene method of moments:
# eb_in = np.median((np.clip((var_hat - mu_hat) / mu_hat**2, 1e-4, None))[in_mod]
#                   / dtrue[in_mod])
# eb_out = np.median((np.clip((var_hat - mu_hat) / mu_hat**2, 1e-4, None))[~in_mod]
#                    / dtrue[~in_mod])
# print(f"\n  estimated/true dispersion: module genes {eb_in:.2f}, "
#       f"other genes {eb_out:.2f}")
# #
# #     Module genes come out with ~1.3x their true dispersion while other genes
# #     come out at ~0.9x. The dispersion was estimated FROM THESE DATA, so the
# #     batch-driven spread inside each module was absorbed into a larger
# #     per-gene dispersion. Unmodelled structure gets LAUNDERED into the
# #     variance estimate.
# #
# #     That protects the type I error rate - and it is why the FDP damage here
# #     is limited - but it pays for the protection in POWER: those genes are
# #     now tested against an inflated variance, so real effects in them are
# #     harder to detect. You do not see the cost in a null-rate table; you see
# #     it as a study that found less than it should have. Modelling the batch
# #     explicitly, or estimating surrogate variables when it is unmeasured
# #     (Module 19), recovers that power instead of paying it away.
#
# # --- (b) Independent filtering, using the best-calibrated test. ---
# pvals, lfc_hat = cal[("~ group + batch", "lrt")]
#
# def bh(p):
#     n = len(p); o = np.argsort(p); q = np.empty(n)
#     q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
#     return np.clip(q, 0, 1)
#
# def report(name, mask):
#     """BH applied only to genes passing `mask`; the rest cannot be called."""
#     q = np.full(len(pvals), 1.0); q[mask] = bh(pvals[mask])
#     disc = q < 0.05
#     fdp = (disc & ~truth_kept).sum() / max(disc.sum(), 1)
#     tpr = (disc & truth_kept).sum() / truth_kept.sum()
#     print(f"  {name:<36}{mask.sum():>8}{disc.sum():>9}{fdp:>8.3f}{tpr:>8.3f}")
#
# print(f"\n  {'analysis (LRT, ~ group + batch)':<36}{'tested':>8}{'discov.':>9}"
#       f"{'FDP':>8}{'power':>8}")
# report("no filtering", np.ones(len(pvals), bool))
# for thresh in (5, 30, 100):
#     report(f"independent filter: mean >= {thresh}", mu_hat >= thresh)
#
# # Filtering barely changes anything here, and the reason is worth knowing:
# # this simulation has very few genes near zero - about 1% below a mean count
# # of 5 - so there is almost nothing to remove. In real RNA-seq 30-50% of
# # annotated genes are effectively unexpressed in any given tissue, and
# # dropping them removes a third or more of the multiplicity burden for free.
# # That is where independent filtering earns its keep.
# #
# # What the table DOES show is that filtering does not break anything: FDP and
# # power stay flat as the threshold rises, until it climbs high enough to
# # start discarding real signal and power falls.
# #
# # It is legitimate only because mean count is independent of the p-value
# # under the null. Filter on the observed fold change instead and FDR control
# # collapses - the adjective in "INDEPENDENT filtering" is load-bearing.
#
# # --- (c) Be honest about the realised FDP. ---
# print(f"""
#   The realised FDP sits ABOVE the nominal 0.05, for three compounding
#   reasons, all of which you just measured:
#
#     1. the test is mildly liberal at n = 6 per group (the 'elsewhere' column
#        above is ~0.06, not 0.05), so more nulls enter the ranking than BH
#        assumes;
#     2. correlated modules drift as a block, which both breaks independence
#        and creates genes that really are differential - just not because of
#        the treatment;
#     3. BH controls the EXPECTED FDP over repeated experiments. A single
#        experiment with only ~{int((bh(pvals) < 0.05).sum())} discoveries has a
#        very noisy realised FDP; one bad module can dominate it.
#
#   None of this means BH is broken. It means 'FDR = 5%' is a statement about
#   a procedure in the long run, not a guarantee about your one gene list -
#   and that calibration is something to CHECK, not assume.""")

# %% [markdown]
# ### Q4: Enrichment on your own result
#
# Test each of the 12 gene modules for enrichment, both naively and with a
# CAMERA-style variance inflation factor (eq. 29.5). Two modules are genuinely
# up-regulated. How many does each method report?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # Work on the moderated z-scale: turn each gene's p-value and sign into a z.
# z = np.sign(lfc_hat) * st.norm.isf(np.clip(pvals, 1e-300, 1) / 2)
# gene_pos = {g: i for i, g in enumerate(kept_idx)}
# logn = np.log2(norm[:, :] + 1)
#
# def set_tests(members):
#     idx = np.array([gene_pos[m] for m in members if m in gene_pos])
#     if len(idx) < 10:
#         return np.nan, np.nan, np.nan
#     m = len(idx)
#     zs = z[idx]
#     # Naive: assumes the m genes are independent.
#     t_naive = zs.mean() / (z.std(ddof=1) / np.sqrt(m))
#     # CAMERA: estimate the mean inter-gene correlation from the residuals and
#     # inflate the variance by VIF = 1 + (m-1)*rhobar (eq. 29.5).
#     sub = logn[[kept_idx[i] for i in idx]]
#     resid = sub - sub.mean(1, keepdims=True)
#     resid = resid / (resid.std(1, ddof=1, keepdims=True) + 1e-9)
#     C = np.corrcoef(resid)
#     rho_bar = (C.sum() - m) / (m * (m - 1))
#     vif = 1 + (m - 1) * rho_bar
#     t_cam = t_naive / np.sqrt(max(vif, 1e-6))
#     return (2 * st.norm.sf(abs(t_naive)), 2 * st.norm.sf(abs(t_cam)), rho_bar)
#
# print(f"  {'module':<12}{'size':>6}{'rho-bar':>10}{'VIF':>8}"
#       f"{'naive p':>12}{'CAMERA p':>12}{'truth':>8}")
# rows = []
# for name, members in D["modules"].items():
#     p_n, p_c, rb = set_tests(members)
#     if np.isnan(p_n):
#         continue
#     m = len([x for x in members if x in gene_pos])
#     truth = "UP" if name in ("module_00", "module_01") else "-"
#     rows.append((name, p_n, p_c, truth))
#     print(f"  {name:<12}{m:>6}{rb:>10.3f}{1+(m-1)*rb:>8.1f}"
#           f"{p_n:>12.2e}{p_c:>12.2e}{truth:>8}")
# n_naive = sum(p < 0.05 / len(rows) for _, p, _, _ in rows)
# n_cam = sum(p < 0.05 / len(rows) for _, _, p, _ in rows)
# fp_naive = sum(p < 0.05 / len(rows) and t == "-" for _, p, _, t in rows)
# fp_cam = sum(p < 0.05 / len(rows) and t == "-" for _, _, p, t in rows)
# print(f"\n  Bonferroni over {len(rows)} sets:")
# print(f"    naive : {n_naive} significant, {fp_naive} of them FALSE")
# print(f"    CAMERA: {n_cam} significant, {fp_cam} of them FALSE")
#
# # Every module is internally correlated by construction, so the naive test -
# # which divides by sqrt(m) as if the genes were independent - understates the
# # variance of the set mean and fires on modules with no true signal. Dividing
# # by sqrt(VIF) restores calibration. Note this is the SAME correction as the
# # design effect in E1: correlated genes and correlated cells are one problem.

# %% [markdown]
# ### Q5: The negative control
#
# Run your entire pipeline on data where the group labels have been
# **permuted**. A correct pipeline finds essentially nothing. Do it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # Permutation is expensive: every run refits two GLMs per gene. Use a random
# # subset of genes - the discovery COUNT scales with it, but the comparison
# # between permutation schemes, which is what we care about, does not.
# perm_genes = np.sort(np.random.default_rng(3).choice(len(kept_idx), 1500,
#                                                      replace=False))
#
# def pipeline(labels, use_batch):
#     """Q1-Q3 as one function, run with whatever group labels it is given."""
#     cols = [labels, batch] if use_batch else [labels]
#     X1 = sm.add_constant(np.column_stack(cols).astype(float))
#     X0 = (sm.add_constant(batch.astype(float)) if use_batch
#           else np.ones((len(labels), 1)))
#     p = np.ones(len(perm_genes))
#     for j, i in enumerate(perm_genes):
#         gi = kept_idx[i]
#         fam = sm.families.NegativeBinomial(alpha=disp_shrunk[i])
#         try:
#             f1 = sm.GLM(counts[gi], X1, family=fam, offset=offset).fit()
#             f0 = sm.GLM(counts[gi], X0, family=fam, offset=offset).fit()
#             p[j] = st.chi2.sf(2 * (f1.llf - f0.llf), 1)
#         except Exception:
#             p[j] = 1.0
#     return (bh(p) < 0.05).sum(), p
#
# rng = np.random.default_rng(99)
#
# def permute_free(r):
#     """Shuffle the labels across all samples, ignoring batch."""
#     return r.permutation(group)
#
# def permute_within_batch(r):
#     """Shuffle labels WITHIN each batch, so the permuted design stays
#     balanced with respect to batch - the only valid null here."""
#     lab = group.copy()
#     for b in np.unique(batch):
#         idx = np.where(batch == b)[0]
#         lab[idx] = r.permutation(group[idx])
#     return lab
#
# n_real, _ = pipeline(group, use_batch=True)
# print(f"  real labels, design ~ group + batch : {n_real:4d} discoveries "
#       f"among {len(perm_genes)} genes tested\n")
# print(f"  {'permutation scheme':<24}{'design':<18}{'mean':>7}{'max':>6}  runs")
# results = {}
# for pname, pfun in [("free (ignores batch)", permute_free),
#                     ("within batch", permute_within_batch)]:
#     for dname, ub in [("~ group", False), ("~ group + batch", True)]:
#         hits = [int(pipeline(pfun(rng), use_batch=ub)[0]) for _ in range(8)]
#         results[(pname, dname)] = hits
#         print(f"  {pname:<24}{dname:<18}{np.mean(hits):>7.1f}"
#               f"{np.max(hits):>6}  {hits}")
#
# print(f"""
#   Read this against the {n_real} discoveries the REAL labels produce. Every
#   permutation scheme collapses to a handful of genes out of {len(perm_genes)},
#   so the pipeline passes its negative control: it is not manufacturing
#   signal out of nothing. That is the first thing to establish, and most
#   published pipelines never establish it.
#
#   Now read the four rows against each other. One row is clearly the worst:
#   permuting freely AND leaving batch out of the design. Fixing EITHER of
#   those - permuting within batch, or putting batch in the design - brings it
#   back down, and doing both is no better than doing one. The nuisance
#   structure has to be accounted for somewhere; it does not much matter
#   where. The differences are modest here only because batch was balanced
#   within group by design. Under a confounded design (E1) they would not be.
#
#   The most useful column is the MAXIMUM, not the mean. Most permutations
#   return zero, but the occasional one returns a handful - those are the runs
#   where the shuffled labels happened to line up with batch, reproducing part
#   of the module structure. This is why a negative control must be run MANY
#   times: one permutation returning 0 proves nothing, and one returning 9 is
#   not evidence of a broken pipeline either. Look at the distribution.
#
#   Two rules worth keeping:
#     * permute within the strata that structure the data (batch, subject,
#       litter, plate) - the exchangeability rule from Module 07;
#     * put known nuisance variables in the DESIGN, so the test is not trying
#       to explain batch variation with the group coefficient.""")
#
# # The p-value histogram under a VALID permutation should be flat.
# _, pp = pipeline(permute_within_batch(rng), use_batch=True)
# hist, edges = np.histogram(pp, bins=10, range=(0, 1))
# print(f"\n  p-value histogram under a within-batch permutation (should be flat):")
# for i in range(10):
#     bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
#     print(f"    [{edges[i]:.1f},{edges[i+1]:.1f})  {hist[i]:5d} {bar}")

# %% [markdown]
# ## Debrief

# %%
header("The generative truth")
print(f"""  {D['is_de'].sum()} of {len(genes)} genes are truly differential.
  Dispersion follows 0.10 + 18/(mean + 12): low-count genes are far noisier.
  Modules 00 and 01 (60 genes each) are genuinely up-regulated; modules 02-11
  are correlated but NOT differential - they exist to catch a naive set test.
  All 12 modules respond to the processing BATCH, which is recorded in the
  sample sheet and balanced within each arm - so it biases nothing, but it
  correlates genes and must be in the design and in any permutation scheme.
  5 very high-expression genes are multiplied 60x in the treated group, which
  is real biology but creates composition bias in the total-count normaliser.

  The pipeline in order, and what each step is actually protecting you from:

    normalisation  -> composition bias (a median-based factor, never CPM)
    dispersion     -> the impossibility of estimating variance at n = 6
    testing        -> the mean-variance relationship of counts (NB, not t)
    filtering      -> multiplicity burden from genes with no power
    enrichment     -> inter-gene correlation (eq. 29.5 = eq. 1.8)
    permutation    -> everything you forgot""")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
lib = counts.sum(0)
ax[0].bar(range(len(lib)), lib / 1e6,
          color=["steelblue" if g == 0 else "darkorange" for g in group])
ax[0].set_xlabel("sample"); ax[0].set_ylabel("library size (millions)")
ax[0].set_title("Library sizes vary 4-fold")
mu_plot = counts.mean(1) + 1
var_plot = counts.var(1, ddof=1) + 1
ax[1].scatter(mu_plot, var_plot, s=3, alpha=0.3, color="grey")
gridx = np.logspace(0, 5, 50)
ax[1].plot(gridx, gridx, "k--", lw=1, label="Poisson (Var = mean)")
ax[1].plot(gridx, gridx + 0.3 * gridx**2, color="firebrick", lw=2,
           label="NB, disp = 0.3")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("mean count"); ax[1].set_ylabel("variance")
ax[1].set_title("Counts are overdispersed (eq. 20.2)"); ax[1].legend(fontsize=7)
frac = np.sort(counts, axis=0)[-5:].sum(0) / counts.sum(0)
ax[2].bar(range(len(frac)), frac * 100,
          color=["steelblue" if g == 0 else "darkorange" for g in group])
ax[2].set_xlabel("sample"); ax[2].set_ylabel("% of library, top 5 genes")
ax[2].set_title("Composition bias")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rnaseq_pipeline.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'rnaseq_pipeline.png')}")

# %% [markdown]
# ## What to take away
#
# 1. Normalise with a **robust** size factor. Total-count normalisation breaks
#    exactly when the biology is interesting.
# 2. You cannot estimate a per-gene variance at $n=6$. Shrink towards a trend.
# 3. Independent filtering raises power for free; filtering on anything
#    correlated with the test statistic destroys FDR control.
# 4. Set-level tests must account for inter-gene correlation.
# 5. **Permute the labels and re-run.** If that returns discoveries, stop.
#
# **Next:** `E3_prediction_audit.py`
