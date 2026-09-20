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
# # Applied 35: Proteomics: missing values, batch structure, and inference
#
# **Curriculum link:** `stats.md` -> Topic 25, equations (25.1)-(25.4)
# **Core modules used:** 02, 05, 15 (missingness), 19 (batch), 23 (moderation)
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A TMT label-based LFQ experiment: 3 plexes x 2 conditions |
# | **Assay** | Peptide-level intensities -> protein summarisation |
# | **Key threats** | 20-50% missing values of **mixed mechanism**, plex effects, run-order drift |
# | **Here** | Simulated with an explicit MNAR/MAR mixture and known truth |
#
# ## The central decision
#
# Missingness in proteomics is a **mixture**: abundance-dependent censoring
# (MNAR) plus stochastic identification failure (MAR). Applying one imputer to
# both is the standard mistake: and it fails in opposite directions.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "35_proteomics_missing_values"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a peptide-level TMT experiment

# %%
header("1. Peptide intensities with plex effects and run-order drift")

def simulate_proteomics(n_proteins=800, n_per_group=9, peptides_per_protein=4,
                        n_de=120, effect=0.9, seed=0):
    """Returns PEPTIDE-level log2 intensities with realistic missingness."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    cond = np.array(["ctrl"] * n_per_group + ["case"] * n_per_group)
    # 3 TMT plexes, each BALANCED across conditions (good design).
    plex = np.array([f"plex{(i % 3) + 1}" for i in range(n)])
    run_order = rng.permutation(n)

    prot_abundance = rng.normal(22.0, 2.3, n_proteins)
    true_eff = np.zeros(n_proteins)
    de = rng.choice(n_proteins, n_de, replace=False)
    true_eff[de] = rng.normal(0, effect, n_de)

    plex_shift = {"plex1": 0.0, "plex2": 0.45, "plex3": -0.30}
    drift = 0.02 * (run_order - run_order.mean())        # run-order drift

    rows, meta_rows = [], []
    for p in range(n_proteins):
        # Peptides of one protein differ hugely in ionisation efficiency.
        pep_offsets = rng.normal(0, 1.6, peptides_per_protein)
        for k in range(peptides_per_protein):
            vals = np.empty(n)
            for j in range(n):
                vals[j] = (prot_abundance[p] + pep_offsets[k]
                           + true_eff[p] * (cond[j] == "case")
                           + plex_shift[plex[j]] + drift[j]
                           + rng.normal(0, 0.35))
            rows.append(vals)
            meta_rows.append({"protein": f"P{p:04d}", "peptide": f"P{p:04d}_{k}"})

    Y = pd.DataFrame(rows, columns=[f"S{i:02d}" for i in range(n)])
    pep_meta = pd.DataFrame(meta_rows)
    Y.index = pep_meta["peptide"]

    # --- MISSINGNESS: an explicit MIXTURE ---------------------------------
    # (a) MNAR: probability of detection rises with intensity (censoring).
    lod_center = np.quantile(Y.values, 0.22)
    p_mnar = 1 / (1 + np.exp((Y.values - lod_center) / 0.55))
    # (b) MAR: a flat stochastic identification failure.
    p_mar = 0.08
    missing = (rng.random(Y.shape) < p_mnar) | (rng.random(Y.shape) < p_mar)
    Yobs = Y.mask(missing)

    sample_meta = pd.DataFrame({"condition": cond, "plex": plex,
                                "run_order": run_order}, index=Y.columns)
    truth = pd.DataFrame({"true_eff": true_eff, "is_de": true_eff != 0},
                         index=[f"P{p:04d}" for p in range(n_proteins)])
    return Yobs, Y, pep_meta, sample_meta, truth, lod_center


Yobs, Ytrue, pep_meta, smeta, truth, LOD = simulate_proteomics(seed=3501)
print(f"  {Yobs.shape[0]} peptides ({truth.shape[0]} proteins) x "
      f"{Yobs.shape[1]} samples")
print(f"  overall missingness: {Yobs.isna().mean().mean():.1%}")
print(f"\n  missingness per sample: "
      f"{Yobs.isna().mean().min():.1%} - {Yobs.isna().mean().max():.1%}")
print(smeta.groupby(["plex", "condition"]).size().to_string())
print(f"\n  Plexes are BALANCED across conditions - the design of Module 19.")

# %% [markdown]
# ## 2. Diagnose the missingness mechanism (the plot every report needs)

# %%
header("2. Missingness vs abundance: the MNAR signature (Topic 15)")

miss_rate = Yobs.isna().mean(axis=1)
mean_obs = Yobs.mean(axis=1)
ok = mean_obs.notna()
r = st.spearmanr(mean_obs[ok], miss_rate[ok])
print(f"  Spearman(mean observed intensity, missingness rate) = "
      f"{r.statistic:+.3f}  (p = {r.pvalue:.1e})")
print(f"\n  {'abundance quintile':<24}{'mean missingness':>18}"
      f"{'mean observed log2':>21}")
q = pd.qcut(mean_obs[ok], 5, labels=[f"Q{i+1}" for i in range(5)])
for lvl in q.cat.categories:
    m = (q == lvl)
    print(f"  {str(lvl):<24}{miss_rate[ok][m].mean():>18.3f}"
          f"{mean_obs[ok][m].mean():>21.2f}")
print("\n  A strong NEGATIVE relationship is the signature of abundance-")
print("  dependent (MNAR / left-censored) missingness. A flat cloud would")
print("  indicate MAR. Here both are present, by construction.")

# Classify each peptide into a likely mechanism - the practical step.
cond_ctrl = (smeta["condition"] == "ctrl").values
cond_case = (smeta["condition"] == "case").values
miss_ctrl = Yobs.loc[:, cond_ctrl].isna().mean(axis=1)
miss_case = Yobs.loc[:, cond_case].isna().mean(axis=1)
# MNAR-like: present in one condition, largely absent in the other.
mnar_like = ((miss_ctrl > 0.7) & (miss_case < 0.3)) | \
            ((miss_case > 0.7) & (miss_ctrl < 0.3))
sporadic = (~mnar_like) & (miss_rate > 0) & (miss_rate < 0.6)
print(f"\n  peptides missing systematically in ONE condition (MNAR-like): "
      f"{int(mnar_like.sum())}")
print(f"  peptides with sporadic gaps (MAR-like)                     : "
      f"{int(sporadic.sum())}")
print("  These two classes need DIFFERENT treatment (Topic 25).")

# %% [markdown]
# ## 3. Normalisation and run-order drift, eq. (25.2)

# %%
header("3. Median normalisation and drift correction (25.2)")

def median_normalise(Y):
    """stats.md eq. (25.2)."""
    col_med = Y.median(axis=0)
    return Y.sub(col_med, axis=1).add(col_med.median())


Ynorm = median_normalise(Yobs)
print(f"  {'sample':<8}{'median before':>15}{'median after':>14}{'plex':>8}"
      f"{'run order':>11}")
for s in Yobs.columns[:6]:
    print(f"  {s:<8}{Yobs[s].median():>15.3f}{Ynorm[s].median():>14.3f}"
          f"{smeta.loc[s,'plex']:>8}{smeta.loc[s,'run_order']:>11}")

before = st.spearmanr(smeta["run_order"], Yobs.median(axis=0)).statistic
print(f"\n  Spearman(run order, sample median) BEFORE: {before:+.3f}")
print(f"  Spearman(run order, sample median) AFTER : undefined")
print("""
  That second line is not a bug, and it is worth understanding. Median
  normalisation subtracts each column's median and adds one common constant,
  so after it every sample median is IDENTICAL by construction. A correlation
  against a constant has zero variance in one argument and is undefined.

  The trap is to read that as "the drift is gone". It is not gone; this
  particular diagnostic has simply been blinded to it. Median normalisation
  removes a GLOBAL shift per sample. Drift that pushes different features by
  different amounts survives untouched, and a per-feature check still sees
  it:""")

def drift_fraction(Y):
    """Fraction of features whose intensity still tracks run order."""
    ro = smeta["run_order"].to_numpy()
    hits = 0
    for _, row in Y.iterrows():
        v = row.to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 8:
            continue
        if st.spearmanr(ro[ok], v[ok]).pvalue < 0.05:
            hits += 1
    return hits / len(Y)

print(f"    features correlated with run order, before: {drift_fraction(Yobs):.1%}")
print(f"    features correlated with run order, after : {drift_fraction(Ynorm):.1%}")
print("""
  Always plot intensity against INJECTION ORDER before and after, per feature
  and not just per sample. With pooled QC samples injected throughout the run,
  fit a LOESS curve per feature against run order and subtract it, which is
  the metabolomics standard.""")

# %% [markdown]
# ## 4. Peptide -> protein summarisation, eq. (25.3)
#
# $$y_{pij}=\mu_i+\text{peptide}_p+\text{sample}_j+\varepsilon$$
#
# **Summing intensities is wrong.** Peptides differ by orders of magnitude in
# ionisation efficiency, so a missing high-intensity peptide looks exactly like
# protein down-regulation.

# %%
header("4. Median polish vs summing (25.3)")

def median_polish(mat, n_iter=12):
    """Tukey's median polish: additive row (peptide) + column (sample) effects."""
    m = mat.copy()
    overall = 0.0
    row_eff = np.zeros(m.shape[0])
    col_eff = np.zeros(m.shape[1])
    for _ in range(n_iter):
        rmed = np.nanmedian(m, axis=1)
        rmed = np.nan_to_num(rmed)
        m -= rmed[:, None]; row_eff += rmed
        d = np.nanmedian(row_eff); row_eff -= d; overall += d
        cmed = np.nanmedian(m, axis=0)
        cmed = np.nan_to_num(cmed)
        m -= cmed[None, :]; col_eff += cmed
        d = np.nanmedian(col_eff); col_eff -= d; overall += d
    return overall + col_eff                     # protein-level per-sample value


def summarise_proteins(Y, pep_meta, method="median_polish"):
    out = {}
    for prot, idx in pep_meta.groupby("protein")["peptide"]:
        # Some peptides may have been dropped upstream (e.g. all-NaN rows
        # removed before imputation), so intersect rather than index blindly.
        present = [p_ for p_ in idx.values if p_ in Y.index]
        if not present:
            continue
        sub = Y.loc[present].values
        if np.all(np.isnan(sub)):
            continue
        if method == "median_polish":
            out[prot] = median_polish(sub)
        elif method == "sum":
            # Sum on the LINEAR scale, then back to log2 - what naive
            # pipelines do.
            lin = np.nansum(2.0 ** sub, axis=0)
            out[prot] = np.log2(np.where(lin > 0, lin, np.nan))
        else:
            out[prot] = np.nanmean(sub, axis=0)
    return pd.DataFrame(out, index=Y.columns).T


prot_mp = summarise_proteins(Ynorm, pep_meta, "median_polish")
prot_sum = summarise_proteins(Ynorm, pep_meta, "sum")

# How much does each summary depend on how many peptides were observed?
n_obs_pep = (Ynorm.notna().groupby(pep_meta["protein"].values).sum())
print(f"  Correlation between the summarised value and the NUMBER OF")
print(f"  OBSERVED PEPTIDES (a pure artefact - it should be near zero):")
for name, P in [("median polish (25.3)", prot_mp), ("sum of intensities", prot_sum)]:
    common = P.index.intersection(n_obs_pep.index)
    v = P.loc[common].mean(axis=1)
    k = n_obs_pep.loc[common].mean(axis=1)
    m = v.notna() & k.notna()
    print(f"    {name:<24} Spearman = "
          f"{st.spearmanr(k[m], v[m]).statistic:+.3f}")
print("\n  Summing intensities confounds 'how much protein' with 'how many")
print("  peptides happened to be identified'. Median polish estimates an")
print("  additive peptide effect and is robust to individual peptides dropping")
print("  out - which is exactly the situation MS creates.")

# %% [markdown]
# ## 5. Imputation: one imputer for two mechanisms is the standard mistake

# %%
header("5. MNAR vs MAR imputation (Topic 15, eq. 15.8)")

def impute_mnar(P, shift=1.8, width=0.3, seed=0):
    """MinProb / Perseus-style: draw from a DOWN-SHIFTED normal."""
    rng = np.random.default_rng(seed)
    out = P.copy()
    # A column with fewer than two observed values has no usable SD, and a
    # subset of rows (as impute_mixed passes) can easily produce one. Fall back
    # to the matrix-wide spread so every cell is imputed; leaving it NaN would
    # silently drop the protein from the comparison below.
    g_sd = float(np.nanstd(P.to_numpy(dtype=float), ddof=1))
    g_mu = float(np.nanmean(P.to_numpy(dtype=float)))
    if not np.isfinite(g_sd) or g_sd == 0:
        g_sd = 1.0
    if not np.isfinite(g_mu):
        g_mu = 0.0
    for s in P.columns:
        col = P[s]
        sd_ = col.std()
        if not np.isfinite(sd_) or sd_ == 0:
            sd_ = g_sd
        mu_ = col.mean()
        if not np.isfinite(mu_):
            mu_ = g_mu
        draws = pd.Series(rng.normal(mu_ - shift * sd_, width * sd_, len(col)),
                          index=col.index)
        out[s] = col.fillna(draws)
    return out


def impute_mar_knn(P, k=6):
    """A simple kNN imputer over SAMPLES (features as the space)."""
    from sklearn.impute import KNNImputer
    return pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(P.values),
                        index=P.index, columns=P.columns)


def impute_mixed(P, mnar_mask, seed=0):
    """Route each protein to the appropriate imputer - the recommended practice."""
    out = P.copy()
    mn = impute_mnar(P.loc[mnar_mask], seed=seed) if mnar_mask.any() else None
    rest = P.loc[~mnar_mask]
    mr = impute_mar_knn(rest) if len(rest) else None
    if mn is not None:
        out.loc[mnar_mask] = mn
    if mr is not None:
        out.loc[~mnar_mask] = mr
    return out


# IMPORTANT: imputation happens at the PEPTIDE level, before summarisation.
# That is where the missing values actually are - with 4 peptides per protein,
# a protein is only missing outright when every one of its peptides failed,
# which is rare. Imputing the protein matrix would barely change anything and
# would hide the decision that matters.
pep_mnar_like = mnar_like.reindex(Ynorm.index).fillna(False)
print(f"  peptides routed to MNAR imputation: {int(pep_mnar_like.sum())}")
print(f"  peptides routed to MAR imputation : "
      f"{int((Ynorm.isna().any(axis=1) & ~pep_mnar_like).sum())}")


def test_proteins(P, smeta, truth, label, min_obs=6):
    """Protein-level test with plex in the design, stats.md eq. (25.4)."""
    keep = P.notna().sum(axis=1) >= min_obs
    P = P[keep]
    res = []
    for p_ in P.index:
        d = smeta.copy()
        d["y"] = P.loc[p_].values
        d = d.dropna(subset=["y"])
        if d["condition"].nunique() < 2 or len(d) < min_obs:
            res.append((p_, np.nan, 1.0)); continue
        try:
            f = smf.ols("y ~ C(condition) + C(plex)", data=d).fit()
            key = [c for c in f.params.index if c.startswith("C(condition)")][0]
            res.append((p_, f.params[key], f.pvalues[key]))
        except Exception:
            res.append((p_, np.nan, 1.0))
    r = pd.DataFrame(res, columns=["protein", "effect", "pvalue"]).set_index("protein")
    r["padj"] = st.false_discovery_control(np.nan_to_num(r["pvalue"], nan=1.0))
    t = truth.reindex(r.index)
    rej = r["padj"] < 0.05
    tp = int((rej & t["is_de"]).sum()); fp = int((rej & ~t["is_de"]).sum())
    print(f"  {label:<40}{len(r):>8}{rej.sum():>7}{tp:>6}{fp:>6}"
          f"{tp/max(t['is_de'].sum(),1):>9.2f}{fp/max(rej.sum(),1):>8.3f}")
    return r


print(f"\n  {'strategy (applied to PEPTIDES)':<40}{'tested':>8}{'rej':>7}"
      f"{'TP':>6}{'FP':>6}{'sens':>9}{'FDP':>8}")

# (a) No imputation: median polish tolerates NaNs directly.
test_proteins(summarise_proteins(Ynorm, pep_meta), smeta, truth,
              "no imputation (median polish handles NaN)")

# (b) Down-shift EVERYTHING.
test_proteins(summarise_proteins(impute_mnar(Ynorm, seed=1), pep_meta),
              smeta, truth, "ALL peptides MNAR-imputed (down-shift)")

# (c) kNN EVERYTHING.
Yknn = Ynorm.copy()
Yknn = Yknn.dropna(how="all")
test_proteins(summarise_proteins(impute_mar_knn(Yknn), pep_meta), smeta, truth,
              "ALL peptides MAR-imputed (kNN)")

# (d) Route by mechanism.
test_proteins(summarise_proteins(impute_mixed(Ynorm.dropna(how="all"),
                                              pep_mnar_like.reindex(
                                                  Ynorm.dropna(how="all").index
                                              ).fillna(False), seed=1),
                                 pep_meta),
              smeta, truth, "MIXED: route each peptide by mechanism")

print("\n  READ THIS TABLE CAREFULLY - it does not say what most tutorials say.")
print("\n  1. NOT IMPUTING wins. Median polish estimates an additive peptide")
print("     effect from whatever peptides ARE observed, and the linear model")
print("     simply drops missing samples. Neither step needs a filled-in value,")
print("     so nothing has to be invented. Modern guidance increasingly says")
print("     the same: prefer methods that TOLERATE missingness over imputing.")
print("\n  2. Down-shifting EVERYTHING is the most damaging choice here: it")
print("     costs ~a third of the sensitivity. Every sporadically-missing")
print("     peptide gets a value ~1.8 SD below the mean regardless of WHY it")
print("     was missing, which adds variance to proteins that were fine.")
print("\n  3. kNN-everything sits in between, and the MIXED strategy matches it")
print(f"     almost exactly - because only {int(pep_mnar_like.sum())} peptides met the strict")
print("     'present in one condition, absent in the other' MNAR criterion.")
print("     When almost nothing is classified MNAR, routing by mechanism")
print("     reduces to the MAR branch. That is worth checking rather than")
print("     assuming: report how many features each branch received.")
print("\n  4. The 'tested' column differs between rows. Imputation changes WHICH")
print("     proteins are analysable at all, so these FDPs are not on perfectly")
print("     equal footing. Always report how the tested set was defined.")
print("\n  The general rule survives: diagnose the mechanism, and do not apply")
print("  one imputer to features that went missing for different reasons. But")
print("  the first question should be whether you need to impute at all.")

# %% [markdown]
# ## 6. Censoring, not missingness, eq. (15.8)
#
# A value below the detection limit is **known to be below it**. The censored
# likelihood uses that information; substitution rules do not.

# %%
header("6. Treating the LOD as censoring (15.8)")

def tobit_group_test(y_obs, censored, lod, group):
    """Two-group comparison under LEFT-CENSORING at `lod`."""
    def nll(par):
        mu0, delta, log_s = par
        s = np.exp(log_s)
        mu = mu0 + delta * group
        ll = 0.0
        unc = ~censored
        if unc.any():
            ll += np.sum(st.norm.logpdf(y_obs[unc], mu[unc], s))
        if censored.any():
            ll += np.sum(st.norm.logcdf((lod - mu[censored]) / s))
        return -ll
    start = [np.nanmean(y_obs[~censored]) if (~censored).any() else lod,
             0.0, np.log(max(np.nanstd(y_obs[~censored]), 0.1))]
    fit = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 3000, "fatol": 1e-9})
    # Likelihood-ratio test for delta = 0 (eq. 13.11 - prefer the LRT).
    def nll0(par):
        return nll([par[0], 0.0, par[1]])
    fit0 = minimize(nll0, [start[0], start[2]], method="Nelder-Mead",
                    options={"maxiter": 3000})
    lr = 2 * (fit0.fun - fit.fun)
    return fit.x[1], st.chi2.sf(max(lr, 0), 1)


rng = np.random.default_rng(3502)
TRUE_DELTA, SD, N = 0.8, 1.0, 40
lod = 21.0
res_rows = []
for rep in range(250):
    g = np.r_[np.zeros(N // 2), np.ones(N // 2)]
    z = rng.normal(21.5 + TRUE_DELTA * g, SD)
    cen = z < lod
    y_sub_half = np.where(cen, lod / 2, z)
    y_sub_lod = np.where(cen, lod, z)
    d_drop = (np.nanmean(np.where(cen, np.nan, z)[g == 1])
              - np.nanmean(np.where(cen, np.nan, z)[g == 0]))
    d_half = y_sub_half[g == 1].mean() - y_sub_half[g == 0].mean()
    d_lod = y_sub_lod[g == 1].mean() - y_sub_lod[g == 0].mean()
    d_tob, _ = tobit_group_test(np.where(cen, lod, z), cen, lod, g)
    res_rows.append({"drop": d_drop, "half": d_half, "lod": d_lod, "tobit": d_tob})
rr = pd.DataFrame(res_rows)
print(f"  TRUE group difference = {TRUE_DELTA}; ~{np.mean(cen):.0%} censored\n")
print(f"  {'strategy':<32}{'mean estimate':>15}{'bias':>9}{'SD':>9}")
for k, lab in [("drop", "drop censored values"),
               ("half", "substitute LOD/2"),
               ("lod", "substitute LOD"),
               ("tobit", "censored MLE (15.8)")]:
    print(f"  {lab:<32}{rr[k].mean():>15.4f}{rr[k].mean()-TRUE_DELTA:>9.4f}"
          f"{rr[k].std(ddof=1):>9.4f}")
print("\n  Every substitution rule is biased. The censored likelihood uses the")
print("  information that the value was BELOW the limit, which is real data.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.scatter(mean_obs[ok], miss_rate[ok], s=3, alpha=0.2)
ax.set_xlabel("mean observed log2 intensity"); ax.set_ylabel("missingness rate")
ax.set_title(f"MNAR signature (Spearman {r.statistic:+.2f})", fontsize=9)

ax = axes[0, 1]
ax.hist(Ytrue.values.ravel(), bins=80, alpha=0.5, label="true (complete)")
ax.hist(Yobs.values[~np.isnan(Yobs.values)], bins=80, alpha=0.7, label="observed")
ax.axvline(LOD, color="red", lw=2, label="detection region")
ax.set_xlabel("log2 intensity")
ax.set_title("Eq. (15.8): the left tail is censored", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.scatter(smeta["run_order"], Yobs.median(axis=0), s=40,
           c=[{"plex1": "C0", "plex2": "C1", "plex3": "C2"}[p]
              for p in smeta["plex"]], label=None)
ax.set_xlabel("injection order"); ax.set_ylabel("sample median log2")
ax.set_title("Run-order drift, coloured by plex", fontsize=9)

ax = axes[1, 1]
for k, lab in [("drop", "drop"), ("half", "LOD/2"), ("lod", "LOD"),
               ("tobit", "censored MLE")]:
    ax.hist(rr[k], bins=30, alpha=0.5, label=lab)
ax.axvline(TRUE_DELTA, color="k", lw=2, label="truth")
ax.set_xlabel("estimated group difference")
ax.set_title("Eq. (15.8) beats every substitution rule", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "proteomics.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/proteomics.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Ignoring the plex
#
# Re-run the protein-level test WITHOUT `C(plex)` in the design. What happens
# to power, and why? Then simulate a CONFOUNDED design (all controls in plex1,
# all cases in plex2) and check the design rank.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # Impute at the PEPTIDE level and then summarise, which is the order this
# # module argues for above. Imputing the protein matrix directly would
# # barely change it, because a protein is only missing when all 4 of its
# # peptides failed.
# Ypep = Ynorm.dropna(how="all")
# Yimp = impute_mixed(Ypep, pep_mnar_like.reindex(Ypep.index).fillna(False),
#                     seed=1)
# P = summarise_proteins(Yimp, pep_meta, "median_polish")
# rows = []
# for form, lab in [("y ~ C(condition)", "no plex term"),
#                   ("y ~ C(condition) + C(plex)", "plex as a block")]:
#     res = []
#     for p_ in P.index:
#         d = smeta.copy(); d["y"] = P.loc[p_].values; d = d.dropna(subset=["y"])
#         try:
#             f = smf.ols(form, data=d).fit()
#             key = [c for c in f.params.index if c.startswith("C(condition)")][0]
#             res.append(f.pvalues[key])
#         except Exception:
#             res.append(1.0)
#     q = st.false_discovery_control(np.nan_to_num(res, nan=1.0))
#     t = truth.reindex(P.index)
#     rej = q < 0.05
#     print(f"  {lab:<20} rej={rej.sum():>4} TP={int((rej & t['is_de']).sum()):>4} "
#           f"FDP={(rej & ~t['is_de']).sum()/max(rej.sum(),1):.3f}")
#
# # Blocking on plex removes the plex offsets from the residual, exactly as
# # donor blocking did in Module 30 - free power from a term the design already
# # provides.
#
# # Now the confounded version:
# smeta_bad = smeta.copy()
# smeta_bad["plex"] = np.where(smeta_bad["condition"] == "ctrl", "plex1", "plex2")
# Xd = np.column_stack([np.ones(len(smeta_bad)),
#                       (smeta_bad["condition"] == "case").astype(float),
#                       (smeta_bad["plex"] == "plex2").astype(float)])
# print(f"  confounded design: {Xd.shape[1]} columns, rank "
#       f"{np.linalg.matrix_rank(Xd)} -> condition and plex are ALIASED")
# # No software can separate them (eq. 19.3). Design plexes to be BALANCED, and
# # include a common reference channel in every plex.

# %% [markdown]
# ### Problem 2: How much does the imputation shift matter?
#
# Vary the MNAR down-shift from 0 to 3 SDs and record the number of
# "significant" proteins and the FDP. This is a sensitivity analysis of the
# kind Topic 15 requires.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'down-shift (SDs)':>18}{'rejected':>11}{'TP':>6}{'FP':>6}{'FDP':>8}")
# for shift in [0.0, 0.8, 1.8, 3.0]:
#     P = impute_mnar(prot_mp, shift=shift, seed=2)
#     res = []
#     for p_ in P.index:
#         d = smeta.copy(); d["y"] = P.loc[p_].values; d = d.dropna(subset=["y"])
#         try:
#             f = smf.ols("y ~ C(condition) + C(plex)", data=d).fit()
#             key = [c for c in f.params.index if c.startswith("C(condition)")][0]
#             res.append(f.pvalues[key])
#         except Exception:
#             res.append(1.0)
#     q = st.false_discovery_control(np.nan_to_num(res, nan=1.0))
#     t = truth.reindex(P.index); rej = q < 0.05
#     print(f"  {shift:>18.1f}{rej.sum():>11}{int((rej & t['is_de']).sum()):>6}"
#           f"{int((rej & ~t['is_de']).sum()):>6}"
#           f"{(rej & ~t['is_de']).sum()/max(rej.sum(),1):>8.3f}")
#
# # The down-shift is an ASSUMPTION about how far below the limit the missing
# # values sit, and it is untestable from the observed data. If your conclusion
# # changes across this range, say so - that IS the result (Module 24).

# %% [markdown]
# ### Problem 3: A protein seen in 3 of 18 samples
#
# Every protein in this dataset has a value in all 18 samples, because a
# protein survives summarisation if even one of its 4 peptides does. So the
# question is not "is it missing?" but **how much measurement is behind each
# number?** Count the peptide-level observations backing each protein (out of
# 4 peptides x 18 samples = 72), bin the test results by that, and decide
# where you would set a reporting threshold.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # Depth of measurement behind each protein, out of 4 peptides x 18 samples.
# depth = (Ynorm.notna().groupby(pep_meta["protein"].values).sum().sum(axis=1)
#          .reindex(prot_mp.index))
# print(f"  every protein has a value in all "
#       f"{int(prot_mp.notna().sum(axis=1).max())} samples, but the peptide")
# print(f"  evidence behind them ranges from {int(depth.min())} to "
#       f"{int(depth.max())} observations out of 72.\n")
#
# P = impute_mnar(prot_mp, seed=3)
# res = []
# for p_ in P.index:
#     d = smeta.copy(); d["y"] = P.loc[p_].values; d = d.dropna(subset=["y"])
#     try:
#         f = smf.ols("y ~ C(condition) + C(plex)", data=d).fit()
#         key = [c for c in f.params.index if c.startswith("C(condition)")][0]
#         pv = f.pvalues[key]
#         res.append((p_, pv if np.isfinite(pv) else 1.0))
#     except Exception:
#         res.append((p_, 1.0))
# rr2 = pd.DataFrame(res, columns=["protein", "p"]).set_index("protein")
# rr2["padj"] = st.false_discovery_control(rr2["p"].to_numpy())
# rr2["depth"] = depth.reindex(rr2.index)
# rr2["is_de"] = truth["is_de"].reindex(rr2.index)
#
# print(f"  {'peptide obs':>13}{'proteins':>10}{'called':>9}{'of which TRUE':>16}"
#       f"{'FDP':>8}")
# for lo, hi in [(0, 35), (36, 50), (51, 65), (66, 72)]:
#     m = (rr2["depth"] >= lo) & (rr2["depth"] <= hi)
#     called = m & (rr2["padj"] < 0.05)
#     tp = int((called & rr2["is_de"]).sum())
#     print(f"  {f'{lo}-{hi}':>13}{int(m.sum()):>10}{int(called.sum()):>9}{tp:>16}"
#           f"{(called.sum() - tp) / max(called.sum(), 1):>8.2f}")
#
# # Read the FDP column down the bins. Proteins backed by few peptide
# # observations produce calls that are driven by the imputed values rather
# # than by measurement, and their false discovery proportion is the worst of
# # any bin. The protein matrix looks complete, which is exactly what makes
# # this dangerous: the missingness has been hidden by summarisation.
# #
# # Set a minimum-evidence rule BEFORE testing - a common default is "at least
# # 2 peptides quantified in at least 70% of one condition" - and report how
# # many proteins it removed. A protein whose value rests on three peptide
# # measurements is a QC finding, not a biological one.

# %% [markdown]
# ## What to take away
#
# 1. Log-transform, normalise, then summarise to protein level: in that order,
#    inspecting after each step.
# 2. **Ask whether you need to impute at all**: median polish plus a model
#    that tolerates `NaN` outperformed every imputer here. If you must impute,
#    diagnose the mechanism first and route MNAR and MAR features differently.
# 3. Model detection limits as **censoring**, not missingness.
# 4. Model plex/batch/run-order explicitly; design them balanced.
# 5. Report how many proteins were quantified in how many samples.
#
# **Next:** `36_microbiome_compositional.py`
