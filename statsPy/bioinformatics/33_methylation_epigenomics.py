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
# # Applied 33: DNA methylation and epigenomics
#
# **Curriculum link:** `stats.md` -> Topic 23, equations (23.1)-(23.3)
# **Core modules used:** 02 (M-values), 13 (beta-binomial), 22 (mediation), 23
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | Illumina EPIC whole-blood case/control cohort (e.g. GSE40279-style ageing studies) |
# | **Assay** | Array beta values + a bisulfite-sequencing counterpart |
# | **Key confounder** | **Blood cell composition**: usually the dominant signal |
# | **Here** | Simulated with realistic probe behaviour and a known truth |
#
# ## The three decisions that dominate a methylation analysis
#
# 1. **Scale**: model on M, report on beta (23.1).
# 2. **Cell composition**: confounder, mediator, or outcome? (23.3)
# 3. **Region vs position**: neighbouring CpGs are correlated.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import nnls
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "33_methylation_epigenomics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Beta and M-values, eq. (23.1)
#
# $$\beta=\frac{M_{\text{int}}}{M_{\text{int}}+U_{\text{int}}+\alpha},\qquad
#   M=\log_2\frac{\beta}{1-\beta}$$

# %%
header("1. Beta values are heteroscedastic; M-values are not (23.1)")

def beta_to_m(beta, eps=1e-6):
    beta = np.clip(beta, eps, 1 - eps)
    return np.log2(beta / (1 - beta))


def m_to_beta(m):
    return 2.0**m / (2.0**m + 1.0)


rng = np.random.default_rng(3301)
print(f"  {'mean beta':>11}{'SD(beta)':>11}{'SD(M)':>9}"
      f"{'beta variance vs beta=0.5':>28}")
ref = None
for target in [0.02, 0.10, 0.30, 0.50, 0.75, 0.95]:
    m_true = np.log2(target / (1 - target))
    m_draws = rng.normal(m_true, 0.5, 40000)
    b_draws = m_to_beta(m_draws)
    if target == 0.50:
        ref = b_draws.var(ddof=1)
    print(f"  {target:>11.2f}{b_draws.std(ddof=1):>11.4f}"
          f"{m_draws.std(ddof=1):>9.4f}", end="")
    print(f"{b_draws.var(ddof=1)/ref if ref else float('nan'):>27.2f}x"
          if ref else "")
print("\n  Probes near 0 or 1 are structurally less variable. A linear model on")
print("  beta gives them standard errors that are too small, so they dominate")
print("  a hit list for a purely mathematical reason. MODEL ON M, REPORT BETA.")

# %% [markdown]
# ## 2. Simulate an array cohort with cell-composition confounding

# %%
header("2. A whole-blood EPIC-like cohort")

CELL_TYPES = ["CD4T", "CD8T", "NK", "Bcell", "Mono", "Gran"]


def simulate_methylation(n=120, n_probes=3000, n_dmp=150, seed=0):
    """Case/control cohort. Cases have BOTH a direct effect and a shift in
    granulocyte proportion, so cell composition is a genuine confounder."""
    rng = np.random.default_rng(seed)

    # Reference methylomes: each cell type has a characteristic profile.
    ref_M = rng.normal(0, 2.0, (n_probes, len(CELL_TYPES)))
    ref_M += rng.normal(0, 1.5, (n_probes, 1))          # shared probe baseline

    case = rng.binomial(1, 0.5, n)
    age = rng.normal(55, 12, n)
    sex = rng.binomial(1, 0.5, n)

    # Cell composition: cases have MORE granulocytes and fewer CD4T.
    base_logit = np.array([1.1, 0.5, 0.2, 0.3, 0.8, 1.9])
    shift = np.array([-0.35, 0.0, 0.0, 0.0, 0.10, 0.30])
    W = np.empty((n, len(CELL_TYPES)))
    for i in range(n):
        lg = base_logit + case[i] * shift + rng.normal(0, 0.22, len(CELL_TYPES))
        W[i] = np.exp(lg) / np.exp(lg).sum()

    # TRUE direct effect of case status, independent of composition.
    true_eff = np.zeros(n_probes)
    dmp = rng.choice(n_probes, n_dmp, replace=False)
    true_eff[dmp] = rng.normal(0, 0.85, n_dmp)

    # Observed M = composition mixture + direct effect + age + noise.
    M = (ref_M @ W.T
         + np.outer(true_eff, case)
         + np.outer(rng.normal(0, 0.012, n_probes), age - age.mean())
         + rng.normal(0, 0.45, (n_probes, n)))

    beta = m_to_beta(M)
    meta = pd.DataFrame({"case": case, "age": age, "sex": sex},
                        index=[f"S{i:03d}" for i in range(n)])
    for k, ct in enumerate(CELL_TYPES):
        meta[ct] = W[:, k]
    probes = [f"cg{i:07d}" for i in range(n_probes)]
    truth = pd.DataFrame({"true_eff": true_eff, "is_dmp": true_eff != 0},
                         index=probes)
    return (pd.DataFrame(beta, index=probes, columns=meta.index),
            pd.DataFrame(M, index=probes, columns=meta.index),
            meta, truth, pd.DataFrame(ref_M, index=probes, columns=CELL_TYPES))


beta_df, M_df, meta, truth, ref_M = simulate_methylation(seed=3302)
assert beta_df.columns.equals(meta.index)
print(f"  {beta_df.shape[0]} probes x {beta_df.shape[1]} samples")
print(f"  {truth['is_dmp'].sum()} probes have a TRUE direct effect of case status")
print(f"\n  mean cell proportions by group:")
print(meta.groupby("case")[CELL_TYPES].mean().round(3).to_string())
print(f"\n  Granulocytes differ by "
      f"{meta.groupby('case')['Gran'].mean().diff().iloc[-1]:+.3f} between groups")
print("  -> cell composition is associated with the outcome, i.e. a CONFOUNDER")
print("     for any probe whose methylation differs between cell types.")

# %% [markdown]
# ## 3. Probe-wise testing on M vs beta

# %%
header("3. Probe-wise linear models: what the M scale actually buys you")

def probewise(Y, meta, rhs_cols, coef_index=1):
    """Fit `probe ~ 1 + rhs_cols` for every probe. Returns effects and p-values.

    `rhs_cols` is a list of metadata column names; the coefficient of interest
    is the first one (`case`).
    """
    X = np.column_stack([np.ones(len(meta))] +
                        [meta[c].astype(float).values for c in rhs_cols])
    beta_hat, *_ = np.linalg.lstsq(X, Y.T.values, rcond=None)
    resid = Y.T.values - X @ beta_hat
    dof = X.shape[0] - X.shape[1]
    s2 = np.maximum((resid ** 2).sum(axis=0) / dof, 1e-14)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[coef_index, coef_index])
    t = beta_hat[coef_index] / se
    p = 2 * st.t.sf(np.abs(t), dof)
    return pd.DataFrame({"effect": beta_hat[coef_index], "se": se, "s2": s2,
                         "pvalue": p, "padj": st.false_discovery_control(p)},
                        index=Y.index)


def moderated_probewise(Y, meta, rhs_cols, coef_index=1):
    """The same fit, then limma-style empirical-Bayes moderation (eq. 20.6)."""
    from scipy.special import digamma, polygamma
    from scipy.optimize import brentq
    base = probewise(Y, meta, rhs_cols, coef_index)
    X = np.column_stack([np.ones(len(meta))] +
                        [meta[c].astype(float).values for c in rhs_cols])
    dof = X.shape[0] - X.shape[1]
    s2 = base["s2"].values
    e = np.log(s2) - digamma(dof / 2) + np.log(dof / 2)
    mad = np.median(np.abs(e - np.median(e))) * 1.4826
    target = mad ** 2 - polygamma(1, dof / 2)
    if target > 0:
        try:
            d0 = brentq(lambda d: polygamma(1, d / 2) - target, 1e-4, 1e4)
        except ValueError:
            d0 = 50.0
    else:
        d0 = 50.0
    s0_2 = float(np.exp(np.mean(e) + digamma(d0 / 2) - np.log(d0 / 2)))
    s2t = (d0 * s0_2 + dof * s2) / (d0 + dof)
    se = np.sqrt(s2t * np.linalg.inv(X.T @ X)[coef_index, coef_index])
    t = base["effect"].values / se
    p = 2 * st.t.sf(np.abs(t), dof + d0)
    out = base.copy()
    out["pvalue"] = p
    out["padj"] = st.false_discovery_control(p)
    return out, d0


def score(res, truth, label):
    rej = res["padj"] < 0.05
    tp = int((rej & truth["is_dmp"]).sum()); fp = int((rej & ~truth["is_dmp"]).sum())
    print(f"  {label:<44}{rej.sum():>7}{tp:>7}{fp:>7}"
          f"{tp/truth['is_dmp'].sum():>9.2f}{fp/max(rej.sum(),1):>8.3f}")


mean_beta = beta_df.mean(axis=1)
dist_from_edge = np.minimum(mean_beta, 1 - mean_beta)
res_beta = probewise(beta_df, meta, ["case", "age", "sex"])
res_M = probewise(M_df, meta, ["case", "age", "sex"])

# THE diagnostic: does residual variance depend on where the probe sits?
r_beta = st.spearmanr(dist_from_edge, res_beta["s2"]).statistic
r_M = st.spearmanr(dist_from_edge, res_M["s2"]).statistic
print(f"  Spearman(distance from the 0/1 boundary, residual variance):")
print(f"    beta scale : {r_beta:+.3f}   <- strong dependence")
print(f"    M scale    : {r_M:+.3f}   <- essentially none")
print("\n  That is the heteroscedasticity of eq. (23.1), measured directly.")

print(f"\n  {'probes binned by mean beta':<30}{'mean resid. var (beta)':>24}"
      f"{'mean resid. var (M)':>21}")
bins = pd.cut(mean_beta, [0, .1, .3, .7, .9, 1.0])
for b in bins.cat.categories:
    m = (bins == b).values
    print(f"  {str(b):<30}{res_beta.loc[m,'s2'].mean():>24.5f}"
          f"{res_M.loc[m,'s2'].mean():>21.5f}")

print("\n  WHY IT MATTERS - and it is NOT what most people assume. An ordinary")
print("  probe-wise t-test on beta is largely self-correcting: an extreme probe")
print("  has both a smaller effect AND a smaller standard error, and the two")
print("  mostly cancel. The damage appears once you MODERATE the variances,")
print("  which every standard pipeline does (limma, eq. 20.6). Empirical Bayes")
print("  assumes the residual variances are EXCHANGEABLE across probes. On the")
print("  beta scale they are not, so the estimated prior degrees of freedom")
print("  collapse and the method can barely borrow strength:")

mod_beta, d0_beta = moderated_probewise(beta_df, meta, ["case", "age", "sex"])
mod_M, d0_M = moderated_probewise(M_df, meta, ["case", "age", "sex"])
print(f"\n    prior df d0, beta scale : {d0_beta:8.1f}")
print(f"    prior df d0, M scale    : {d0_M:8.1f}")
print(f"    -> moderation on M adds the equivalent of ~{d0_M:.0f} extra df per")
print(f"       probe; on beta it adds ~{d0_beta:.0f}. With n = 4 or 6 samples per")
print("       group - the usual EWAS pilot - that difference is decisive.")

print(f"\n  {'analysis':<44}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>9}{'FDP':>8}")
score(res_beta, truth, "beta, ordinary t, adj. age+sex")
score(res_M, truth, "M, ordinary t, adj. age+sex")
score(mod_beta, truth, "beta, MODERATED t, adj. age+sex")
score(mod_M, truth, "M, MODERATED t, adj. age+sex")
print("\n  Note the FDP in all four rows is far above 0.05. None of these models")
print("  adjusts for CELL COMPOSITION, which is the real driver here - that is")
print("  section 4, and it matters far more than the scale choice.")

# %% [markdown]
# ## 4. Cell composition: confounder, mediator, or outcome? eq. (23.3)
#
# This is a **causal** decision (Module 22), not a statistical one.
# Reference-based deconvolution (Houseman) solves
# $\min_{\mathbf w\ge0,\sum w=1}\|\mathbf y-\mathbf R\mathbf w\|^2$.

# %%
header("4. Reference-based deconvolution and the adjustment decision (23.3)")

def houseman_deconvolve(Y, reference):
    """stats.md eq. (23.3): non-negative, sum-to-one constrained regression.

    Implemented as NNLS followed by renormalisation - the standard practical
    approximation to the constrained least-squares problem.
    """
    R = reference.values
    out = np.empty((Y.shape[1], R.shape[1]))
    for i in range(Y.shape[1]):
        w, _ = nnls(R, Y.iloc[:, i].values)
        out[i] = w / w.sum() if w.sum() > 0 else np.full(R.shape[1], 1/R.shape[1])
    return pd.DataFrame(out, index=Y.columns, columns=reference.columns)


# Use the most cell-type-discriminating probes as the reference panel.
disc = ref_M.std(axis=1).sort_values(ascending=False).head(400).index
W_hat = houseman_deconvolve(M_df.loc[disc], ref_M.loc[disc])
print(f"  Estimated vs true cell proportions (correlation across samples):")
for ct in CELL_TYPES:
    r = np.corrcoef(W_hat[ct], meta[ct])[0, 1]
    print(f"    {ct:<8} r = {r:+.3f}   "
          f"(true mean {meta[ct].mean():.3f}, est mean {W_hat[ct].mean():.3f})")

meta_w = meta.copy()
for ct in CELL_TYPES[:-1]:                 # drop one - they sum to 1
    meta_w[f"est_{ct}"] = W_hat[ct].values

res_adj = probewise(M_df, meta_w,
                    ["case", "age", "sex"] + [f"est_{c}" for c in CELL_TYPES[:-1]])
print(f"\n  {'analysis':<46}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>9}{'FDP':>8}")
score(res_M, truth, "M scale, NOT adjusted for composition")
score(res_adj, truth, "M scale, adjusted for estimated composition")

print("\n  Adjusting removes composition-driven false positives. But BEFORE")
print("  adjusting, decide the causal role (Module 22):")
print("    CONFOUNDER - composition differs for reasons unrelated to disease")
print("                 (age, smoking, sampling) -> ADJUST.")
print("    MEDIATOR   - the disease CAUSES the composition change, which then")
print("                 changes methylation -> adjusting removes the effect you")
print("                 wanted. Do a mediation analysis instead (eq. 32.6).")
print("    OUTCOME    - the composition change IS the finding -> model it")
print("                 directly and do not adjust methylation for it.")

# Quantify the mediation split for the composition path.
med_frac = []
for p_ in truth.index[truth["is_dmp"]][:60]:
    d = meta.copy(); d["y"] = M_df.loc[p_].values
    tot = smf.ols("y ~ case + age + sex", data=d).fit().params["case"]
    dir_ = smf.ols("y ~ case + age + sex + " +
                   " + ".join(CELL_TYPES[:-1]), data=d).fit().params["case"]
    if abs(tot) > 1e-6:
        med_frac.append(1 - dir_ / tot)
print(f"\n  Median proportion of the case effect 'explained' by composition")
print(f"  across 60 true DMPs: {np.median(med_frac):.1%}")
print("  If composition is a MEDIATOR, that fraction is a real biological")
print("  pathway and adjusting it away is a mistake.")

# %% [markdown]
# ## 5. Bisulfite sequencing: beta-binomial counts, eq. (23.2)

# %%
header("5. Bisulfite counts need a beta-binomial (23.2)")

rng = np.random.default_rng(3303)
n_s, n_sites = 24, 500
group = rng.binomial(1, 0.5, n_s)
true_pi = rng.beta(2, 2, n_sites)
delta = np.zeros(n_sites); delta[:50] = 0.12          # 50 truly differential
RHO = 0.03                                            # overdispersion

# Coverage varies enormously across sites - the defining feature of WGBS.
coverage = rng.negative_binomial(2, 2 / (2 + 30), (n_sites, n_s)) + 1

y = np.empty((n_sites, n_s), dtype=int)
for i in range(n_sites):
    for j in range(n_s):
        pi_ij = np.clip(true_pi[i] + delta[i] * group[j], 0.01, 0.99)
        a = pi_ij * (1 / RHO - 1); b = (1 - pi_ij) * (1 / RHO - 1)
        p_draw = rng.beta(a, b)
        y[i, j] = rng.binomial(coverage[i, j], p_draw)

def binomial_test_sites(y, cov, group):
    """WRONG: plain binomial GLM, ignoring biological overdispersion."""
    p = np.empty(len(y))
    for i in range(len(y)):
        d = pd.DataFrame({"m": y[i], "u": cov[i] - y[i], "g": group.astype(float)})
        try:
            f = smf.glm("m + u ~ g", data=d, family=sm.families.Binomial()).fit()
            p[i] = f.pvalues["g"]
        except Exception:
            p[i] = 1.0
    return p


def quasibinomial_test_sites(y, cov, group):
    """RIGHT-ish: quasi-binomial absorbs the extra-binomial variability."""
    p = np.empty(len(y))
    for i in range(len(y)):
        d = pd.DataFrame({"m": y[i], "u": cov[i] - y[i], "g": group.astype(float)})
        try:
            f = smf.glm("m + u ~ g", data=d, family=sm.families.Binomial()).fit()
            chi2 = np.sum(f.resid_pearson ** 2)
            phi = max(chi2 / f.df_resid, 1.0)
            t = f.params["g"] / (f.bse["g"] * np.sqrt(phi))
            p[i] = 2 * st.t.sf(abs(t), f.df_resid)
        except Exception:
            p[i] = 1.0
    return p


is_de = np.zeros(n_sites, bool); is_de[:50] = True
print(f"  {'model':<34}{'rej':>7}{'TP':>7}{'FP':>7}{'FDP':>8}")
for lab, fn in [("binomial (ignores overdispersion)", binomial_test_sites),
                ("quasi-binomial (eq. 23.2)", quasibinomial_test_sites)]:
    p = fn(y, coverage, group)
    q = st.false_discovery_control(np.nan_to_num(p, nan=1.0))
    rej = q < 0.05
    print(f"  {lab:<34}{rej.sum():>7}{(rej & is_de).sum():>7}"
          f"{(rej & ~is_de).sum():>7}{(rej & ~is_de).sum()/max(rej.sum(),1):>8.3f}")
print(f"\n  Coverage ranges {coverage.min()}-{coverage.max()}x. A plain binomial")
print("  treats a 200x site as 100x more certain than a 2x site, ignoring the")
print("  BIOLOGICAL variability between samples entirely (eq. 23.2).")

# %% [markdown]
# ## 6. Regions beat positions: neighbouring CpGs are correlated

# %%
header("6. Region-level inference")

# Give probes genomic positions; DMPs cluster into regions.
pos = np.sort(rng.integers(0, 10_000_000, len(truth)))
probe_pos = pd.Series(pos, index=truth.index)
z = st.norm.isf(res_adj["pvalue"] / 2) * np.sign(res_adj["effect"])

def region_scan(z, positions, window=2000, min_probes=3):
    """Combine adjacent z-scores in a sliding genomic window (Stouffer)."""
    order = np.argsort(positions.values)
    zs = z.values[order]; ps = positions.values[order]; ids = z.index[order]
    out = []
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] - ps[i] <= window:
            j += 1
        k = j - i + 1
        if k >= min_probes:
            # Stouffer (eq. 6.7). NOTE: assumes independence, which adjacent
            # CpGs violate - so the p-value needs a permutation calibration.
            stat = np.sum(zs[i:j+1]) / np.sqrt(k)
            out.append({"start": ps[i], "end": ps[j], "n_probes": k,
                        "z": stat, "probes": list(ids[i:j+1])})
        i = j + 1
    return pd.DataFrame(out)


regions = region_scan(z, probe_pos)
print(f"  {len(regions)} candidate regions with >= 3 probes")
print(f"  largest |region z|: {regions['z'].abs().max():.2f}")
print("\n  CRITICAL: the Stouffer combination assumes INDEPENDENT probes.")
print("  Adjacent CpGs are spatially correlated, so this p-value is")
print("  anticonservative. Calibrate by permuting SAMPLE LABELS and rerunning")
print("  the whole scan - which preserves the spatial correlation structure:")

null_max = []
for b in range(60):
    perm = rng.permutation(meta.index)
    mp = meta.copy(); mp["case"] = meta.loc[perm, "case"].values
    rp = probewise(M_df, mp, ["case", "age", "sex"])
    zp = st.norm.isf(rp["pvalue"] / 2) * np.sign(rp["effect"])
    rg = region_scan(zp, probe_pos)
    null_max.append(rg["z"].abs().max() if len(rg) else 0.0)
thresh = np.quantile(null_max, 0.95)
print(f"    95th percentile of the max |region z| under permutation: {thresh:.2f}")
print(f"    regions exceeding it: {(regions['z'].abs() > thresh).sum()}")
print(f"    naive |z| > 1.96 would have called: {(regions['z'].abs() > 1.96).sum()}")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
bb = np.linspace(0.02, 0.98, 60)
sds = [m_to_beta(rng.normal(np.log2(b/(1-b)), 0.5, 3000)).std(ddof=1) for b in bb]
ax.plot(bb, sds, lw=2, label="SD on the beta scale")
ax.axhline(0.5, color="C1", ls="--", label="SD on the M scale (constant)")
ax.set_xlabel("mean methylation $\\beta$"); ax.set_ylabel("SD")
ax.set_title("Eq. (23.1): beta is heteroscedastic", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
for ct, c in zip(CELL_TYPES, [f"C{i}" for i in range(6)]):
    ax.scatter(meta[ct], W_hat[ct], s=10, alpha=0.6, color=c, label=ct)
ax.plot([0, 0.6], [0, 0.6], "k--", lw=1)
ax.set_xlabel("true proportion"); ax.set_ylabel("deconvolved (eq. 23.3)")
ax.set_title("Reference-based deconvolution", fontsize=9)
ax.legend(fontsize=6)

ax = axes[1, 0]
ax.scatter(mean_beta[~truth["is_dmp"]], -np.log10(res_adj.loc[~truth["is_dmp"], "pvalue"]),
           s=3, alpha=0.2, color="grey", label="null probe")
ax.scatter(mean_beta[truth["is_dmp"]], -np.log10(res_adj.loc[truth["is_dmp"], "pvalue"]),
           s=8, alpha=0.6, color="C3", label="true DMP")
ax.set_xlabel("mean beta"); ax.set_ylabel("$-\\log_{10}p$ (M scale, adjusted)")
ax.set_title("Hits are not concentrated at the boundaries", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.hist(null_max, bins=25, alpha=0.7, label="permutation null (max |z|)")
ax.axvline(thresh, color="red", ls="--", label="95th percentile")
ax.axvline(regions["z"].abs().max(), color="k", lw=2, label="observed max")
ax.set_xlabel("max |region z|")
ax.set_title("Region significance by label permutation", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "methylation.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/methylation.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Report an M-scale effect as a beta difference
#
# Take the top 5 hits on the M scale and convert their effects into an
# interpretable "percentage points of methylation" difference. Why can you not
# just exponentiate the M coefficient?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# top = res_adj.nsmallest(5, "pvalue")
# print(f"  {'probe':>12}{'M effect':>10}{'mean beta':>11}"
#       f"{'beta at ctrl':>14}{'beta at case':>14}{'difference':>12}")
# for pr in top.index:
#     eff = res_adj.loc[pr, "effect"]
#     m_ctrl = M_df.loc[pr, (meta["case"] == 0).values].mean()
#     b_ctrl, b_case = m_to_beta(m_ctrl), m_to_beta(m_ctrl + eff)
#     print(f"  {pr:>12}{eff:>10.3f}{beta_df.loc[pr].mean():>11.3f}"
#           f"{b_ctrl:>14.3f}{b_case:>14.3f}{b_case-b_ctrl:>+12.3f}")
#
# # The map from M to beta is NONLINEAR (a logistic), so the SAME M-scale effect
# # corresponds to a different beta difference depending on where you start.
# # Near beta = 0.5 the map is steepest; near 0 or 1 a large M change barely
# # moves beta. You must evaluate m_to_beta at the actual baseline, which is why
# # you report "beta went from 0.42 to 0.55", not "the effect was 0.5 M-units".

# %% [markdown]
# ### Problem 2: Confounder or mediator?
#
# Simulate a variant where cell composition is a pure MEDIATOR (disease -> cells
# -> methylation, with no direct effect). Show that adjusting for composition
# destroys the total effect, and compute the mediated proportion with eq. (32.6).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_m = np.random.default_rng(77)
# n = 200
# case = rng_m.binomial(1, 0.5, n)
# gran = 0.30 + 0.10 * case + rng_m.normal(0, 0.04, n)   # disease -> composition
# meth = 2.0 * gran + rng_m.normal(0, 0.15, n)           # composition -> methylation
# d = pd.DataFrame({"case": case, "gran": gran, "meth": meth})
# tot = smf.ols("meth ~ case", data=d).fit()
# dir_ = smf.ols("meth ~ case + gran", data=d).fit()
# a_path = smf.ols("gran ~ case", data=d).fit().params["case"]
# print(f"  total effect (eq. 32.6 TE)     = {tot.params['case']:+.4f}  "
#       f"p = {tot.pvalues['case']:.2e}")
# print(f"  direct effect after adjusting  = {dir_.params['case']:+.4f}  "
#       f"p = {dir_.pvalues['case']:.3f}")
# print(f"  indirect (NIE) = a*b            = "
#       f"{a_path * dir_.params['gran']:+.4f}")
# print(f"  proportion mediated             = "
#       f"{a_path*dir_.params['gran']/tot.params['case']:.1%}")
#
# # There IS a real, total effect of disease on methylation - it just runs
# # THROUGH cell composition. Adjusting for the mediator makes it vanish and you
# # would wrongly conclude "no methylation difference". The causal question
# # decides the model; the data cannot tell you which role composition plays.

# %% [markdown]
# ### Problem 3: Probe-number bias in gene-set testing
#
# Assign each probe to a gene, with gene "length" drawn so that some genes have
# many probes. Show that a naive hypergeometric enrichment test (eq. 29.1) on
# "genes with >= 1 significant probe" is biased toward probe-rich genes even
# when probes are assigned significance at random.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_g = np.random.default_rng(88)
# n_genes = 400
# probes_per_gene = rng_g.negative_binomial(1.2, 1.2/(1.2+7)) + 1
# gene_of_probe = np.repeat(np.arange(n_genes), probes_per_gene)
# gene_of_probe = gene_of_probe[:len(truth)]
# n_genes_used = gene_of_probe.max() + 1
# # Assign significance to probes COMPLETELY AT RANDOM (5%).
# sig_probe = rng_g.random(len(gene_of_probe)) < 0.05
# gene_hit = pd.Series(sig_probe).groupby(gene_of_probe).any()
# n_probes = pd.Series(1).repeat(len(gene_of_probe)).groupby(gene_of_probe).sum()
# print(f"  {'probes per gene':>17}{'P(gene called)':>17}")
# for lo, hi in [(1, 1), (2, 4), (5, 9), (10, 100)]:
#     m = (n_probes >= lo) & (n_probes <= hi)
#     print(f"  {f'{lo}-{hi}':>17}{gene_hit[m].mean():>17.1%}")
#
# # Probe-rich genes are called far more often, purely because they have more
# # chances to contain a random hit. Any pathway enriched for probe-rich genes
# # (promoter CpG islands, long genes) will look "significant" in every study.
# # missMethyl::gometh corrects exactly this bias; a plain hypergeometric test
# # on the gene list does not.

# %% [markdown]
# ## What to take away
#
# 1. **Model M, report beta.** Model bisulfite counts as beta-binomial.
# 2. Decide the **causal role** of cell composition before adjusting.
# 3. Filter probes by outcome-independent quality criteria first.
# 4. Use region-level inference, calibrated by **label permutation**.
# 5. Correct gene-set tests for probe-number bias.
#
# **Next:** `34_gwas_association.py`
