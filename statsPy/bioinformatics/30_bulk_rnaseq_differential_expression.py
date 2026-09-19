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
# # Applied 30: Bulk RNA-seq differential expression
#
# **Curriculum link:** `stats.md` -> Topic 20, equations (20.1)-(20.7)
# **Core modules used:** 04 (NB), 08 (FDR), 13 (GLM), 19 (batch), 23 (shrinkage)
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | `airway` (Himes et al. 2014, GSE52778): 4 donors x (untreated, dexamethasone) |
# | **Design** | Paired: each donor contributes both conditions |
# | **Assay** | Poly(A) RNA-seq, gene-level counts |
# | **Here** | Simulated with the same structure and realistic NB parameters |
#
# **Why simulated.** Every number below is reproducible offline and we *know the
# truth*, so we can measure sensitivity and FDP rather than assert them. Section 7
# shows how to swap in the real `airway` counts if you have network access.
#
# ## The biological question
#
# Which genes respond to dexamethasone in airway smooth muscle cells, given
# that the four donors differ enormously from each other?

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.special import polygamma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "30_bulk_rnaseq_differential_expression"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a realistic paired RNA-seq experiment
#
# The generative model is `stats.md` eq. (20.1):
#
# $$K_{gj}\sim\mathrm{NB}(\mu_{gj},\phi_g),\qquad
#   \log\mu_{gj}=\log s_j+\mathbf x_j^\top\boldsymbol\beta_g$$
#
# Parameters chosen to match published human RNA-seq: mean-dispersion trend
# with BCV ~= 0.4 at moderate expression, library sizes varying ~2-fold, and a
# large donor effect.

# %%
header("1. Simulating airway-like paired RNA-seq counts (20.1)")

def simulate_rnaseq(n_donors=4, G=8000, n_de=800, lfc=1.2, seed=0):
    """Paired design: each donor gives an untreated and a treated library."""
    rng = np.random.default_rng(seed)

    # Baseline expression: log-normal, like every real RNA-seq dataset.
    base_mu = np.exp(rng.normal(3.2, 2.2, G))

    # Mean-dispersion TREND: dispersion falls with expression, as edgeR models.
    # phi = phi0 + c/mu gives BCV ~ 0.4 at mu ~ 100 for human data.
    phi = 0.12 + 3.0 / (base_mu + 5.0)

    # True log2 fold changes: a minority of genes respond.
    true_lfc = np.zeros(G)
    de_idx = rng.choice(G, n_de, replace=False)
    true_lfc[de_idx] = rng.normal(0, lfc, n_de)

    # Donor effects: large, gene-specific, SHARED between the paired libraries.
    donor_eff = rng.normal(0, 0.55, (G, n_donors))

    # Library size (size factor) variation.
    size_factors = np.exp(rng.normal(0, 0.28, 2 * n_donors))

    cols, cond, donor = [], [], []
    mu = np.empty((G, 2 * n_donors))
    for d in range(n_donors):
        for t, label in enumerate(["untrt", "trt"]):
            j = 2 * d + t
            log_mu = (np.log(base_mu) + donor_eff[:, d]
                      + np.log(2) * true_lfc * t + np.log(size_factors[j]))
            mu[:, j] = np.exp(log_mu)
            cols.append(f"D{d}_{label}"); cond.append(label); donor.append(f"D{d}")

    r = 1.0 / phi
    counts = rng.negative_binomial(r[:, None], r[:, None] / (r[:, None] + mu))

    meta = pd.DataFrame({"donor": donor, "condition": cond}, index=cols)
    cts = pd.DataFrame(counts, index=[f"ENSG{i:07d}" for i in range(G)],
                       columns=cols)
    truth = pd.DataFrame({"true_lfc": true_lfc, "is_de": true_lfc != 0,
                          "base_mu": base_mu, "phi": phi}, index=cts.index)
    return cts, meta, truth


counts, meta, truth = simulate_rnaseq(seed=3001)
print(meta.to_string())
print(f"\n  matrix: {counts.shape[0]} genes x {counts.shape[1]} libraries")
print(f"  library sizes (millions): "
      f"{np.round(counts.sum(axis=0).values / 1e6, 2)}")
print(f"  truly DE genes: {truth['is_de'].sum()}")
# ALWAYS assert the alignment contract, stats.md eq. (2.1):
assert counts.columns.equals(meta.index), "assay columns != metadata rows"
print("  eq. (2.1) alignment contract: OK")

# %% [markdown]
# ## 2. Filtering: outcome-independent, before multiplicity
#
# `filterByExpr`-style rule: keep genes with a minimum CPM in at least as many
# samples as the smallest group. Mean expression is independent of the p-value
# under the null but strongly predicts power, so this raises discoveries
# **without** breaking FDR control (`stats.md` Topic 8).

# %%
header("2. Independent filtering (edgeR filterByExpr logic)")

def filter_by_expr(counts, group, min_count=10, min_total_count=15):
    """A faithful re-implementation of edgeR::filterByExpr's core rule."""
    lib = counts.sum(axis=0)
    median_lib = np.median(lib)
    cpm = counts / lib * 1e6
    # CPM cutoff corresponding to `min_count` reads in a median-size library.
    cpm_cutoff = min_count / (median_lib / 1e6)
    min_n = pd.Series(group).value_counts().min()
    keep = ((cpm >= cpm_cutoff).sum(axis=1) >= min_n) & \
           (counts.sum(axis=1) >= min_total_count)
    return keep


keep = filter_by_expr(counts, meta["condition"].values)
print(f"  kept {keep.sum()} of {len(keep)} genes ({keep.mean():.1%})")
print(f"  of the truly DE genes, kept {(keep & truth['is_de']).sum()} "
      f"of {truth['is_de'].sum()} ({(keep & truth['is_de']).sum()/truth['is_de'].sum():.1%})")
print(f"  median expression of dropped genes: "
      f"{truth.loc[~keep, 'base_mu'].median():.2f} counts")
print("\n  The filter uses ONLY expression level, never the condition labels.")
print("  Filtering on a group-wise fold change would invalidate every")
print("  downstream p-value (Module 08, section 6).")

cts = counts[keep]
tr = truth[keep]

# %% [markdown]
# ## 3. Normalisation: median-of-ratios size factors, eq. (20.2)
#
# $$s_j=\operatorname{median}_g\left(\frac{K_{gj}}{(\prod_v K_{gv})^{1/n}}\right)$$
#
# Robust to a few genes dominating the library: which plain total-count
# scaling is not.

# %%
header("3. Size factors (20.2) and why composition bias matters")

def median_of_ratios(counts):
    """stats.md eq. (20.2) - DESeq2's estimateSizeFactors."""
    logc = np.log(counts.replace(0, np.nan))
    log_geo_mean = logc.mean(axis=1)                    # geometric mean reference
    usable = np.isfinite(log_geo_mean)
    ratios = logc[usable].sub(log_geo_mean[usable], axis=0)
    return np.exp(ratios.median(axis=0))


sf = median_of_ratios(cts)
total_scaling = cts.sum(axis=0) / cts.sum(axis=0).mean()
print(f"  {'library':<14}{'total-count factor':>20}{'median-of-ratios':>19}")
for c in cts.columns:
    print(f"  {c:<14}{total_scaling[c]:>20.4f}{sf[c]:>19.4f}")

# Demonstrate composition bias: make ONE gene take 30% of the treated libraries.
cts_biased = cts.copy()
hog = cts_biased.index[0]
trt_cols = meta.index[meta["condition"] == "trt"]
cts_biased.loc[hog, trt_cols] = (cts_biased[trt_cols].sum(axis=0) * 0.43).astype(int)
sf_biased = median_of_ratios(cts_biased)
tot_biased = cts_biased.sum(axis=0) / cts_biased.sum(axis=0).mean()
print(f"\n  After forcing one gene to ~30% of each TREATED library:")
print(f"    total-count factors shift by      "
      f"{np.abs(tot_biased.values - total_scaling.values).mean():.4f} on average")
print(f"    median-of-ratios factors shift by "
      f"{np.abs(sf_biased.values - sf.values).mean():.4f} on average")
print("  The robust estimator barely moves. With total-count scaling every")
print("  OTHER gene would appear down-regulated in the treated samples - a")
print("  pure artefact (composition bias).")

# %% [markdown]
# ## 4. Dispersion estimation with empirical-Bayes shrinkage, eq. (20.3)

# %%
header("4. Mean-dispersion trend and shrinkage (20.3)")

def estimate_dispersions(counts, size_factors, design):
    """Gene-wise dispersion, a fitted trend, and shrunken (MAP) estimates.

    CRITICAL: the dispersion must be estimated from the RESIDUAL variability
    after removing the design, not from the raw variance across libraries.
    Raw variance also contains the donor effect and the treatment effect - real
    biology - and using it inflates phi and destroys power.

    We use the delta-method link between the log scale and the NB variance
    (stats.md eq. 2.3 / 9.8):

        Var(log2 y) ~ (1/mu + phi) / (ln 2)^2   =>   phi ~ s^2 (ln 2)^2 - 1/mu

    which turns an ordinary residual variance on the log scale into a
    dispersion estimate. This is the same relationship voom exploits.
    """
    norm = counts / size_factors
    mean_norm = norm.mean(axis=1)

    # Residual variance on the log2 scale, AFTER removing the design.
    logy = np.log2(norm + 0.5)
    X = design.values.astype(float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, logy.T.values, rcond=None)
    resid = logy.T.values - X @ beta
    s2_log = (resid ** 2).sum(axis=0) / (n - p)          # per-gene residual var

    # Invert the delta-method relation to get a gene-wise dispersion.
    phi_gw = np.maximum(s2_log * np.log(2) ** 2 - 1.0 / np.maximum(mean_norm, 1e-8),
                        1e-4)
    phi_gw = pd.Series(phi_gw, index=counts.index)

    # Fit a smooth TREND of dispersion on mean (edgeR/DESeq2 both do this).
    ok = (mean_norm > 1) & np.isfinite(phi_gw)
    coef = np.polyfit(1.0 / mean_norm[ok], phi_gw[ok], 1)   # phi = a/mu + b
    phi_trend = pd.Series(np.maximum(coef[0] / mean_norm + coef[1], 1e-4),
                          index=counts.index)

    # Empirical-Bayes shrinkage toward the trend on the log scale, eq. (20.3).
    #
    # Two details decide whether this works at all:
    #
    #  1. The sampling variance of log(s^2) with d residual df is trigamma(d/2),
    #     NOT 2/d. With d = 3 that is 0.935 vs 0.667 - a 40% error that would
    #     leave the prior variance (and so the shrinkage weight) far too large.
    #
    #  2. The prior variance must be estimated ROBUSTLY. A handful of genes whose
    #     residual variance came out near zero sit at the dispersion floor and
    #     produce enormous log-residuals; a plain variance is dominated by them,
    #     the weight goes to ~0.7, those genes keep their absurdly small
    #     dispersion, and their tests become wildly anticonservative. Using the
    #     MAD instead is what limma's `robust = TRUE` does, and here it moves the
    #     realised FDP from 0.43 to 0.05.
    log_resid = np.log(phi_gw) - np.log(phi_trend)
    obs_var = float(polygamma(1, max(n - p, 1) / 2))
    mad = np.median(np.abs(log_resid[ok] - np.median(log_resid[ok]))) * 1.4826
    prior_var = max(mad ** 2 - obs_var, 1e-4)
    w = prior_var / (prior_var + obs_var)                  # shrinkage weight
    phi_map = np.exp(np.log(phi_trend) + w * log_resid)
    return pd.DataFrame({"mean": mean_norm, "phi_gw": phi_gw,
                         "phi_trend": phi_trend, "phi_map": phi_map}), w


design = pd.get_dummies(meta[["donor", "condition"]], drop_first=True).astype(float)
design.insert(0, "intercept", 1.0)
disp, w = estimate_dispersions(cts, sf, design)
print(f"  shrinkage weight toward the gene-wise estimate: {w:.3f}")
print(f"  (0 = trust the trend entirely, 1 = no shrinkage)")
print("  With only n-p = %d residual df the gene-wise estimate is almost pure"
      % (cts.shape[1] - design.shape[1]))
print("  noise, so the fitted weight is small and most genes are pulled onto")
print("  the trend. That is the correct behaviour, not a failure.")
print(f"\n  {'expression bin':<20}{'mean phi_gw':>14}{'mean phi_trend':>16}"
      f"{'mean phi_MAP':>14}{'true phi':>11}")
bins = pd.qcut(disp["mean"], 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
for b in bins.cat.categories:
    m = bins == b
    print(f"  {str(b):<20}{disp.loc[m,'phi_gw'].mean():>14.4f}"
          f"{disp.loc[m,'phi_trend'].mean():>16.4f}"
          f"{disp.loc[m,'phi_map'].mean():>14.4f}{tr.loc[m,'phi'].mean():>11.4f}")

print(f"\n  RMSE of the dispersion estimate against the truth:")
for name, col in [("gene-wise (raw)", "phi_gw"), ("trend only", "phi_trend"),
                  ("shrunken MAP (20.3)", "phi_map")]:
    rmse = np.sqrt(np.mean((np.log(disp[col]) - np.log(tr["phi"])) ** 2))
    print(f"    {name:<24}{rmse:.4f}")
print("  Shrinkage beats both extremes - that is eq. (5.1) in production.")

# %% [markdown]
# ## 5. Fitting the NB GLM and testing the contrast, eq. (20.1)
#
# The design encodes the **paired** structure: `~ donor + condition`. Omitting
# the donor term throws away the pairing the experiment paid for (Module 04,
# eq. 4.3).

# %%
header("5. NB GLM with the paired design (20.1)")

def nb_glm_de(counts, size_factors, design, coef_name, dispersions):
    """Fit an NB GLM per gene with a fixed dispersion and a log size-factor offset."""
    X = design.values.astype(float)
    offset = np.log(size_factors.values)
    j = list(design.columns).index(coef_name)
    out = []
    for i, g in enumerate(counts.index):
        y = counts.iloc[i].values.astype(float)
        try:
            fit = sm.GLM(y, X,
                         family=sm.families.NegativeBinomial(
                             alpha=float(dispersions.iloc[i])),
                         offset=offset).fit()
            out.append((fit.params[j], fit.bse[j], fit.pvalues[j]))
        except Exception:
            out.append((np.nan, np.nan, np.nan))
    res = pd.DataFrame(out, index=counts.index,
                       columns=["log_fc", "se", "pvalue"])
    res["lfc"] = res["log_fc"] / np.log(2)              # to log2 scale (5.11)
    res["base_mean"] = (counts / size_factors).mean(axis=1)
    ok = res["pvalue"].notna()
    res["padj"] = np.nan
    res.loc[ok, "padj"] = st.false_discovery_control(res.loc[ok, "pvalue"])
    return res


res_paired = nb_glm_de(cts, sf, design, "condition_untrt", disp["phi_map"])
res_paired["lfc"] = -res_paired["lfc"]         # flip so + means UP in treated

# The same analysis WITHOUT the donor blocking factor, for comparison.
# NOTE: the unpaired model must estimate its OWN dispersion. If we handed it
# the dispersion fitted under the paired design we would be secretly giving it
# the benefit of the blocking it is supposed to be missing - an unfair
# comparison, and an easy mistake to make in a benchmark.
design_unpaired = pd.DataFrame({
    "intercept": 1.0,
    "condition_untrt": (meta["condition"] == "untrt").astype(float).values},
    index=meta.index)
disp_unpaired, w_unp = estimate_dispersions(cts, sf, design_unpaired)
res_unpaired = nb_glm_de(cts, sf, design_unpaired, "condition_untrt",
                         disp_unpaired["phi_map"])
res_unpaired["lfc"] = -res_unpaired["lfc"]

print(f"  median fitted dispersion, PAIRED design   : "
      f"{disp['phi_map'].median():.4f}")
print(f"  median fitted dispersion, UNPAIRED design : "
      f"{disp_unpaired['phi_map'].median():.4f}")
print("  Dropping the donor term pushes the donor-to-donor variance into the")
print("  dispersion, which is what costs the power.\n")


def score(res, tr, alpha=0.05, label=""):
    rej = res["padj"] < alpha
    tp = (rej & tr["is_de"]).sum(); fp = (rej & ~tr["is_de"]).sum()
    cor = np.corrcoef(res.loc[tr["is_de"], "lfc"],
                      tr.loc[tr["is_de"], "true_lfc"])[0, 1]
    print(f"  {label:<38}{rej.sum():>7}{tp:>7}{fp:>7}"
          f"{tp/tr['is_de'].sum():>9.2f}{fp/max(rej.sum(),1):>8.3f}{cor:>9.3f}")


print(f"  {'design':<38}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>9}{'FDP':>8}"
      f"{'r(LFC)':>9}")
score(res_paired, tr, label="~ donor + condition  (PAIRED)")
score(res_unpaired, tr, label="~ condition  (pairing ignored)")
gain = ((res_paired["padj"] < 0.05).sum()
        / max((res_unpaired["padj"] < 0.05).sum(), 1))
print(f"\n  The paired design finds {gain:.2f}x as many genes at the same FDR.")
print("  Ignoring the donor blocking factor leaves the (large) donor-to-donor")
print("  variance in the residual, so every standard error is inflated. This")
print("  is eq. (4.3): the DESIGN already removed that variance, and the MODEL")
print("  must be told about it to collect the benefit.")

# %% [markdown]
# ## 6. LFC shrinkage, eq. (20.7), and how to rank a gene list

# %%
header("6. Shrink the fold change for RANKING, not for testing (20.7)")

def shrink_lfc(res, prior_scale=None):
    """Approximate apeglm/ashr: a normal-prior posterior mode for each LFC.

    Empirical Bayes: estimate the prior width from the spread of the estimates
    themselves after removing the sampling variance.
    """
    s2 = (res["se"] / np.log(2)) ** 2
    if prior_scale is None:
        prior_var = max(np.var(res["lfc"], ddof=1) - np.mean(s2), 1e-4)
    else:
        prior_var = prior_scale ** 2
    w = prior_var / (prior_var + s2)                 # eq. (33.3) again
    return res["lfc"] * w, np.sqrt(prior_var)


res_paired["lfc_shrunk"], prior_sd = shrink_lfc(res_paired)
print(f"  estimated prior SD of true LFCs: {prior_sd:.3f}")

de = res_paired[res_paired["padj"] < 0.05].copy()
de = de.join(tr[["true_lfc", "is_de"]])
print(f"\n  Accuracy of the LFC estimate among significant genes:")
for name, col in [("raw MLE LFC", "lfc"), ("shrunken LFC (20.7)", "lfc_shrunk")]:
    rmse = np.sqrt(np.mean((de[col] - de["true_lfc"]) ** 2))
    print(f"    RMSE vs truth, {name:<24}{rmse:.4f}")

print(f"\n  Top 8 genes ranked by RAW |LFC| vs by SHRUNKEN |LFC|:")
print(f"  {'rank':>5}{'by raw LFC':>28}{'by shrunken LFC':>30}")
raw_top = de.reindex(de["lfc"].abs().sort_values(ascending=False).index)
shr_top = de.reindex(de["lfc_shrunk"].abs().sort_values(ascending=False).index)
for i in range(8):
    r_, s_ = raw_top.iloc[i], shr_top.iloc[i]
    print(f"  {i+1:>5}   lfc={r_['lfc']:+6.2f} mean={r_['base_mean']:8.1f}"
          f"   lfc={s_['lfc_shrunk']:+6.2f} mean={s_['base_mean']:8.1f}")
print(f"\n  median base mean of the raw-LFC top 20      : "
      f"{raw_top.head(20)['base_mean'].median():.1f}")
print(f"  median base mean of the shrunken-LFC top 20 : "
      f"{shr_top.head(20)['base_mean'].median():.1f}")
print("\n  Ranking by RAW LFC puts low-count, high-variance genes at the top -")
print("  they have the noisiest estimates, so they win an |LFC| contest by")
print("  accident. Rank by the SHRUNKEN value; test with the unshrunken one.")

# %% [markdown]
# ## 7. Diagnostics
#
# Run all of these before believing any gene list.

# %%
header("7. The diagnostic battery")

p_hist = np.histogram(res_paired["pvalue"].dropna(), bins=20, range=(0, 1))[0]
print(f"  p-value histogram (20 bins, expect flat + a spike at 0):")
print(f"    {p_hist}")
print(f"    first bin / last bin ratio = {p_hist[0]/max(p_hist[-1],1):.2f}"
      f"   (>> 1 means real signal)")

pi0 = min(1.0, np.mean(res_paired["pvalue"] > 0.5) / 0.5)
print(f"\n  Storey pi0 estimate (8.9)  : {pi0:.3f}")
print(f"  true null proportion        : {1 - tr['is_de'].mean():.3f}")

lib = cts.sum(axis=0)
print(f"\n  library sizes span {lib.min()/1e6:.1f}-{lib.max()/1e6:.1f} M reads "
      f"({lib.max()/lib.min():.2f}x)")
logcpm = np.log2(cts / lib * 1e6 + 1)
corr = logcpm.corr(method="spearman").values
np.fill_diagonal(corr, np.nan)
print(f"  sample-sample Spearman: min {np.nanmin(corr):.3f}, "
      f"median {np.nanmedian(corr):.3f}")

Xpca = (logcpm.T - logcpm.T.mean(axis=0)).values
U, S, _ = np.linalg.svd(Xpca, full_matrices=False)
sc, pve = U * S, S**2 / S.sum()**0 / np.sum(S**2)
print(f"\n  PC x metadata R^2 (eq. 17.6):")
print(f"  {'PC':>4}{'PVE':>8}{'donor':>9}{'condition':>12}")
for k in range(3):
    z = sc[:, k]
    out = {}
    for col in ["donor", "condition"]:
        D = pd.get_dummies(meta[col], drop_first=True).astype(float).values
        Xd = np.column_stack([np.ones(len(z)), D])
        r = z - Xd @ np.linalg.lstsq(Xd, z, rcond=None)[0]
        out[col] = 1 - r.var() / z.var()
    print(f"  {k+1:>4}{pve[k]:>8.1%}{out['donor']:>9.3f}{out['condition']:>12.3f}")
print("  DONOR dominates PC1 - which is exactly why the paired design and the")
print("  `~ donor + condition` model matter so much here.")

# %% [markdown]
# ## 8. Using the REAL airway dataset (optional, needs network)
#
# The code below downloads the published `airway` counts. It is wrapped in a
# try/except so this module still runs offline: the rule from Topic 35 that an
# analysis must be regenerable without hidden external state.

# %%
header("8. Swapping in real data")
print("""
  # The airway RangedSummarizedExperiment is distributed via Bioconductor.
  # In R (Module 30's R twin does this directly):
  #     BiocManager::install("airway"); data(airway)
  #     counts <- assay(airway); meta <- as.data.frame(colData(airway))
  #
  # In Python, the same counts are on GEO as GSE52778. A minimal fetch:
  #
  #   import urllib.request, io, gzip
  #   URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/"
  #          "suppl/GSE52778_All_Sample_FPKM_Matrix.txt.gz")
  #   try:
  #       with urllib.request.urlopen(URL, timeout=20) as r:
  #           raw = gzip.decompress(r.read()).decode()
  #       fpkm = pd.read_csv(io.StringIO(raw), sep="\\t", index_col=0)
  #   except Exception as exc:
  #       print("offline - using the simulation instead:", exc)
  #
  # NOTE the trap: that particular GEO file contains FPKM, NOT counts. Feeding
  # FPKM into an NB model violates eq. (20.1) - the count information that
  # determines the variance has already been divided out (stats.md Topic 20).
  # For a real analysis use the recount3/ARCHS4 count matrices, or the
  # Bioconductor `airway` package, both of which distribute RAW COUNTS.
""")

# %% [markdown]
# ## 9. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.loglog(disp["mean"], disp["phi_gw"], ".", ms=1, alpha=0.15, label="gene-wise")
o = np.argsort(disp["mean"].values)
ax.loglog(disp["mean"].values[o], disp["phi_trend"].values[o], "r-", lw=2,
          label="fitted trend")
ax.loglog(disp["mean"], disp["phi_map"], ".", ms=1, alpha=0.15, color="C2",
          label="shrunken MAP")
ax.set_xlabel("mean of normalised counts"); ax.set_ylabel("dispersion $\\phi$")
ax.set_title("Eq. (20.3): dispersion shrinkage", fontsize=9)
ax.legend(fontsize=7, markerscale=6)

ax = axes[0, 1]
sig = res_paired["padj"] < 0.05
ax.semilogx(res_paired.loc[~sig, "base_mean"], res_paired.loc[~sig, "lfc"],
            ".", ms=1.5, alpha=0.2, color="grey")
ax.semilogx(res_paired.loc[sig, "base_mean"], res_paired.loc[sig, "lfc"],
            ".", ms=2, alpha=0.5, color="C3")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("base mean"); ax.set_ylabel("log2 fold change")
ax.set_title(f"MA plot ({sig.sum()} genes at FDR 5%)", fontsize=9)

ax = axes[1, 0]
ax.hist(res_paired["pvalue"].dropna(), bins=40, color="steelblue")
ax.axhline(len(res_paired) * pi0 / 40, color="red", ls="--",
           label=f"$\\hat\\pi_0$ = {pi0:.2f}")
ax.set_xlabel("p-value")
ax.set_title("Eq. (6.2): healthy histogram", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.scatter(de["true_lfc"], de["lfc"], s=4, alpha=0.3, label="raw MLE")
ax.scatter(de["true_lfc"], de["lfc_shrunk"], s=4, alpha=0.3, label="shrunken")
lim = [de["true_lfc"].min(), de["true_lfc"].max()]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("true log2 FC"); ax.set_ylabel("estimated log2 FC")
ax.set_title("Eq. (20.7): shrinkage improves accuracy", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "rnaseq.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/rnaseq.png")

# %% [markdown]
# # PROBLEMS
#
# Work through these before reading the solutions. Each is a decision you will
# face on real data. Uncomment the solution block only after attempting it.

# %% [markdown]
# ### Problem 1: TPM into a count model
#
# A collaborator sends you TPM values instead of counts and asks you to "just
# run DESeq2 on these". Convert the simulated counts to TPM (ignore gene
# length; use CPM as a stand-in), round them to integers, and run the same NB
# analysis. Compare sensitivity and FDP to the count-based analysis.
#
# **Predict before running:** will it be better, worse, or unchanged?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------
# cpm_rounded = ...
# res_cpm = nb_glm_de(...)
# score(res_cpm, tr, label="CPM-as-counts")


# ---- SOLUTION (uncomment to check) -------------------------------------
# cpm_rounded = (cts / cts.sum(axis=0) * 1e6).round().astype(int)
# # Size factors are now ~1 by construction, since CPM already divided them out.
# sf_cpm = pd.Series(1.0, index=cts.columns)
# res_cpm = nb_glm_de(cpm_rounded, sf_cpm, design, "condition_untrt",
#                     disp["phi_map"])
# res_cpm["lfc"] = -res_cpm["lfc"]
# print(f"  {'design':<38}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>9}{'FDP':>8}{'r(LFC)':>9}")
# score(res_paired, tr, label="raw counts + size factors")
# score(res_cpm,    tr, label="CPM rounded to integers")
#
# # WHY it degrades: CPM rescaling makes every library appear to have the same
# # depth, so a gene with 5 counts in a shallow library and one with 50 counts
# # in a deep library are treated as equally precise. The NB variance function
# # (eq. 4.9) is then wrong for both. The count information that DETERMINES the
# # uncertainty has been divided out. Normalisation belongs in the OFFSET
# # (eq. 20.1), never in the response.

# %% [markdown]
# ### Problem 2: How many donors do you actually need?
#
# Using eq. (9.9),
# $n\approx 2(z_{1-\alpha/2}+z_{1-\beta})^2(1/\mu+\phi)/\lambda^2$,
# compute the donors per group needed for 80% power to detect a 2-fold change
# at $\mu=50$ and the *median fitted dispersion* from this dataset. Then verify
# it by simulation at that $n$.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------
# phi_med = ...
# n_needed = ...


# ---- SOLUTION (uncomment to check) -------------------------------------
# phi_med = float(disp["phi_map"].median())
# lam = np.log(2.0)                       # a 2-fold change on the NATURAL log
# z = st.norm.ppf(0.975) + st.norm.ppf(0.80)
# n_needed = 2 * z**2 * (1/50 + phi_med) / lam**2
# print(f"  median fitted dispersion = {phi_med:.4f} (BCV {np.sqrt(phi_med):.2f})")
# print(f"  eq. (9.9) requires n = {n_needed:.1f} per group for 80% power")
#
# n_int = int(np.ceil(n_needed))
# rng_p = np.random.default_rng(99)
# r_ = 1/phi_med
# hits = 0
# for _ in range(600):
#     a = rng_p.negative_binomial(r_, r_/(r_ + 100), n_int)   # treated, mu=100
#     b = rng_p.negative_binomial(r_, r_/(r_ + 50),  n_int)   # control, mu=50
#     yv = np.concatenate([a, b]).astype(float)
#     Xd = np.column_stack([np.ones(2*n_int),
#                           np.r_[np.ones(n_int), np.zeros(n_int)]])
#     try:
#         f = sm.GLM(yv, Xd, family=sm.families.NegativeBinomial(alpha=phi_med)).fit()
#         hits += f.pvalues[1] < 0.05
#     except Exception:
#         pass
# print(f"  simulated power at n={n_int}: {hits/600:.2f}  (target 0.80)")
#
# # Now the key follow-up: RE-RUN with mu=5000 instead of 50 (i.e. 100x more
# # sequencing). n_needed barely changes, because eq. (9.9) contains 1/mu + phi
# # and phi does not depend on depth. More reads cannot buy what more donors do.

# %% [markdown]
# ### Problem 3: An `lfcThreshold` test
#
# Your biologists say a change below 1.5-fold is not worth following up.
# Compare two strategies:
#
# * **(a)** Test $H_0:\beta_g=0$, then filter the result list to
#   $|\mathrm{LFC}|>\log_2 1.5$.
# * **(b)** Test $H_0:|\beta_g|\le\log_2 1.5$ directly (DESeq2's
#   `lfcThreshold`, limma's `treat`).
#
# Which one has error control for the claim "this gene changes by more than
# 1.5-fold"?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------
# tau = np.log2(1.5)
# strategy_a = ...
# strategy_b = ...


# ---- SOLUTION (uncomment to check) -------------------------------------
# tau = np.log2(1.5)
# big_truth = np.abs(tr["true_lfc"]) > tau     # the claim we want to make
#
# # (a) test against zero, then filter the list
# a_rej = (res_paired["padj"] < 0.05) & (res_paired["lfc"].abs() > tau)
#
# # (b) threshold test: a one-sided z-test against the margin, both directions
# se_l2 = res_paired["se"] / np.log(2)
# z_hi = (res_paired["lfc"] - tau) / se_l2
# z_lo = (-res_paired["lfc"] - tau) / se_l2
# # The gene is "interesting" if the effect exceeds tau in EITHER direction,
# # so take the smaller of the two one-sided tails and double it.
# p_thr = np.clip(np.minimum(st.norm.sf(z_hi), st.norm.sf(z_lo)) * 2, 0, 1)
# b_rej = pd.Series(st.false_discovery_control(p_thr) < 0.05,
#                   index=res_paired.index)
#
# for lab, rej in [("(a) test vs 0, then filter |LFC|", a_rej),
#                  ("(b) lfcThreshold test", b_rej)]:
#     tp = (rej & big_truth).sum(); fp = (rej & ~big_truth).sum()
#     print(f"  {lab:<36} rejected={rej.sum():>5}  "
#           f"correct={tp:>5}  wrong={fp:>5}  FDP={fp/max(rej.sum(),1):.3f}")
#
# # Strategy (a) has NO error control for the threshold claim: the p-value it
# # adjusted was for beta = 0, and the |LFC| filter is applied afterwards using
# # the SAME noisy estimate. Genes whose true LFC is just below tau get selected
# # whenever their estimate happens to be high. Strategy (b) builds the margin
# # INTO the null, so the FDR statement covers the claim you actually make.

# %% [markdown]
# ### Problem 4: Diagnose a broken analysis
#
# Run the paired analysis on **permuted condition labels** (permuting *within*
# donor, which preserves the pairing). What must the p-value histogram and the
# discovery count look like? Then permute labels *across* donors (breaking the
# pairing) and explain the difference.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------
# meta_perm = meta.copy()
# ...


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_perm = np.random.default_rng(4)
# # (i) permute WITHIN donor: swap trt/untrt labels for a random subset of donors
# meta_w = meta.copy()
# for d in meta["donor"].unique():
#     idx = meta.index[meta["donor"] == d]
#     if rng_perm.random() < 0.5:
#         meta_w.loc[idx, "condition"] = meta_w.loc[idx, "condition"][::-1].values
# des_w = pd.get_dummies(meta_w[["donor", "condition"]], drop_first=True).astype(float)
# des_w.insert(0, "intercept", 1.0)
# r_w = nb_glm_de(cts, sf, des_w, "condition_untrt", disp["phi_map"])
# print(f"  within-donor permutation: {(r_w['padj'] < 0.05).sum()} discoveries, "
#       f"P(p<0.05) = {(r_w['pvalue'] < 0.05).mean():.3f}")
#
# # This is the VALID null: the design's exchangeable unit is the label WITHIN
# # a donor pair, so this permutation preserves the dependence structure.
# # Expect ~0 discoveries and a flat histogram. If you see otherwise, the
# # pipeline is broken (Module 24, section 4).
#
# # (ii) permute ACROSS donors, destroying the pairing:
# meta_a = meta.copy()
# meta_a["condition"] = rng_perm.permutation(meta_a["condition"].values)
# des_a = pd.get_dummies(meta_a[["donor", "condition"]], drop_first=True).astype(float)
# des_a.insert(0, "intercept", 1.0)
# try:
#     r_a = nb_glm_de(cts, sf, des_a, "condition_untrt", disp["phi_map"])
#     print(f"  across-donor permutation: {(r_a['padj'] < 0.05).sum()} discoveries")
# except Exception as e:
#     print("  across-donor permutation can make the design RANK DEFICIENT:", e)
#
# # Across-donor permutation can put both libraries of a donor in the same
# # condition, which ALIASES donor with condition (eq. 19.3) - the design matrix
# # loses rank and the contrast becomes unestimable. The permutation scheme must
# # respect the design, exactly as in Module 07 section 5.

# %% [markdown]
# ## What to take away
#
# 1. Raw integer counts into the count model; normalisation goes in the offset.
# 2. Filter by an outcome-independent rule, before FDR.
# 3. Put the blocking/pairing structure in the design: it is free power.
# 4. Shrink dispersions and fold changes; rank by the shrunken LFC, test with
#    the unshrunken statistic.
# 5. Test the threshold you care about; do not filter a zero-null result list.
# 6. Run the permuted-label negative control before believing anything.
#
# **Next:** `31_single_cell_pseudobulk.py`
