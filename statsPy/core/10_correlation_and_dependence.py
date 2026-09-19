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
# # Module 10: Correlation and dependence
#
# **Curriculum link:** `stats.md` -> Topic 10, equations (10.1)-(10.10)
#
# ## What you will learn
#
# 1. Pearson (10.2) measures **linear** association only, and is not invariant
#    to the log transforms omics uses everywhere.
# 2. Fisher's $z$ (10.3) for correct intervals.
# 3. Partial correlation (10.5) and the precision matrix (10.6): marginal vs
#    conditional independence.
# 4. Simpson's paradox and repeated-measures correlation (10.7).
# 5. Correlation is **not agreement**: Bland-Altman (10.8), Lin's CCC (10.9).
# 6. Correlation matrices under multiplicity, and shrinkage (10.10).

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "10_correlation_and_dependence"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. What Pearson does and does not detect

# %%
header("1. r = 0 does not mean independent (10.1)-(10.2)")

rng = np.random.default_rng(1001)
n = 2000
x = rng.uniform(-3, 3, n)
cases = {
    "linear":        (x, 1.5 * x + rng.normal(0, 1, n)),
    "quadratic":     (x, x**2 + rng.normal(0, 1, n)),
    "monotone (exp)": (x, np.exp(x) + rng.normal(0, 1, n)),
    "independent":   (x, rng.normal(0, 1, n)),
}
print(f"  {'relationship':<18}{'Pearson r':>11}{'Spearman':>10}{'Kendall':>10}")
for name, (a, b) in cases.items():
    print(f"  {name:<18}{st.pearsonr(a, b).statistic:>11.3f}"
          f"{st.spearmanr(a, b).statistic:>10.3f}"
          f"{st.kendalltau(a, b).statistic:>10.3f}")
print("\n  'quadratic': r ~ 0 despite a perfect deterministic relationship.")
print("  'monotone' : Spearman/Kendall detect it far better than Pearson.")

# Scale dependence: critical for omics, where log is everywhere.
counts_a = rng.lognormal(3, 1, 500)
counts_b = counts_a ** 1.3 * rng.lognormal(0, 0.4, 500)
print(f"\n  Same data, different scale:")
print(f"    r on raw scale      = {st.pearsonr(counts_a, counts_b).statistic:.3f}")
print(f"    r on log scale      = "
      f"{st.pearsonr(np.log(counts_a), np.log(counts_b)).statistic:.3f}")
print(f"    Spearman (invariant)= {st.spearmanr(counts_a, counts_b).statistic:.3f}")
print("  ALWAYS state the scale on which a correlation was computed.")

# %% [markdown]
# ## 2. Fisher's z-transform, eq. (10.3)
#
# $$z=\operatorname{artanh}(r)\ \dot\sim\ \mathcal N\!\Big(\operatorname{artanh}(\rho),\tfrac1{n-3}\Big)$$

# %%
header("2. Correct confidence intervals for a correlation (10.3)")

def fisher_ci(r, n, alpha=0.05):
    """stats.md eq. (10.3): CI built on the z scale, mapped back with tanh."""
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = st.norm.ppf(1 - alpha / 2)
    return np.tanh([z - zc * se, z + zc * se])


for r, n_ in [(0.9, 10), (0.9, 100), (0.3, 20), (-0.6, 15)]:
    lo, hi = fisher_ci(r, n_)
    naive = np.array([r - 1.96 / np.sqrt(n_), r + 1.96 / np.sqrt(n_)])
    print(f"  r={r:+.2f}, n={n_:>4}: Fisher CI [{lo:+.3f}, {hi:+.3f}]   "
          f"naive symmetric [{naive[0]:+.3f}, {naive[1]:+.3f}]")
print("\n  The Fisher interval is asymmetric and stays inside [-1,1]; the naive")
print("  one escapes the parameter space. Coverage check:")

rng = np.random.default_rng(1002)
RHO, n_ = 0.7, 15
cov_f = cov_n = 0
for _ in range(20000):
    xy = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], n_)
    r = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
    lo, hi = fisher_ci(r, n_)
    cov_f += lo <= RHO <= hi
    cov_n += (r - 1.96/np.sqrt(n_)) <= RHO <= (r + 1.96/np.sqrt(n_))
print(f"    Fisher coverage = {cov_f/20000:.1%}   naive coverage = "
      f"{cov_n/20000:.1%}   (nominal 95%)")

# %% [markdown]
# ## 3. Partial correlation and the precision matrix, eq. (10.5)-(10.6)
#
# A zero in the **covariance** matrix means *marginal* independence; a zero in
# the **precision** matrix $\boldsymbol\Sigma^{-1}$ means *conditional*
# independence. Very different claims.

# %%
header("3. Marginal vs conditional independence (10.5)-(10.6)")

rng = np.random.default_rng(1003)
n = 3000
# Chain structure: Z -> X and Z -> Y. X and Y are marginally correlated but
# CONDITIONALLY INDEPENDENT given Z. This is the classic confounding pattern.
z = rng.normal(0, 1, n)
x = 1.2 * z + rng.normal(0, 1, n)
y = 1.5 * z + rng.normal(0, 1, n)


def partial_corr(a, b, c):
    """stats.md eq. (10.5)."""
    rab, rac, rbc = (st.pearsonr(a, b).statistic, st.pearsonr(a, c).statistic,
                     st.pearsonr(b, c).statistic)
    return (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))


print(f"  marginal r(X,Y)        = {st.pearsonr(x, y).statistic:+.4f}")
print(f"  partial  r(X,Y | Z)    = {partial_corr(x, y, z):+.4f}  (eq. 10.5)")

# The precision-matrix route (eq. 10.6) gives the same answer.
M = np.column_stack([x, y, z])
S = np.cov(M, rowvar=False)
Omega = np.linalg.inv(S)
pc_from_precision = -Omega[0, 1] / np.sqrt(Omega[0, 0] * Omega[1, 1])
print(f"  from precision matrix  = {pc_from_precision:+.4f}  (eq. 10.6) <- same")
print("\n  X and Y look strongly associated but are conditionally independent")
print("  given Z. A co-expression network built on MARGINAL correlations fills")
print("  with such indirect edges (stats.md Topic 30).")

# %% [markdown]
# ## 4. Simpson's paradox and repeated measures, eq. (10.7)

# %%
header("4. Pooling repeated measures can REVERSE the association (10.7)")

rng = np.random.default_rng(1004)
n_subj, n_visits = 12, 8
records = []
for s in range(n_subj):
    # Subjects with high baseline biomarker tend to have LOW outcome (between-
    # subject effect), but WITHIN each subject the relationship is POSITIVE.
    base_x = rng.normal(s * 1.0, 0.3)
    base_y = 30 - 2.0 * s + rng.normal(0, 0.8)
    for v in range(n_visits):
        xv = base_x + rng.normal(0, 0.5)
        yv = base_y + 1.5 * (xv - base_x) + rng.normal(0, 0.4)
        records.append({"subject": f"S{s:02d}", "x": xv, "y": yv})
d = pd.DataFrame(records)

pooled_r = st.pearsonr(d["x"], d["y"]).statistic
# Repeated-measures correlation (eq. 10.7): common within-subject slope.
fit = smf.ols("y ~ x + C(subject)", data=d).fit()
slope = fit.params["x"]
# Sign * sqrt(partial R^2) for the x term.
fit0 = smf.ols("y ~ C(subject)", data=d).fit()
ss_model = fit0.ssr - fit.ssr
r_rm = np.sign(slope) * np.sqrt(ss_model / (ss_model + fit.ssr))

print(f"  naive pooled correlation      r = {pooled_r:+.4f}")
print(f"  repeated-measures correlation r = {r_rm:+.4f}   (eq. 10.7)")
print(f"  common within-subject slope     = {slope:+.4f}")
print("\n  The pooled correlation has the OPPOSITE SIGN to the within-subject")
print("  relationship. Handle donor structure before interpreting correlation.")

# %% [markdown]
# ## 5. Correlation is not agreement, eq. (10.8)-(10.9)

# %%
header("5. Bland-Altman and Lin's CCC (10.8)-(10.9)")

rng = np.random.default_rng(1005)
truth = rng.normal(100, 15, 120)
assay1 = truth + rng.normal(0, 2, 120)
assay2 = 1.9 * truth + 8 + rng.normal(0, 2, 120)    # perfectly correlated, wrong scale

def lins_ccc(a, b):
    """stats.md eq. (10.9)."""
    va, vb = a.var(ddof=0), b.var(ddof=0)
    cov = np.cov(a, b, ddof=0)[0, 1]
    return 2 * cov / (va + vb + (a.mean() - b.mean()) ** 2)


def bland_altman(a, b):
    """stats.md eq. (10.8): bias and 95% limits of agreement."""
    diff = a - b
    return {"bias": diff.mean(),
            "loa": (diff.mean() - 1.96 * diff.std(ddof=1),
                    diff.mean() + 1.96 * diff.std(ddof=1))}


print(f"  Pearson r(assay1, assay2) = {st.pearsonr(assay1, assay2).statistic:.5f}"
      f"   <- 'excellent agreement'?")
print(f"  Lin's CCC                 = {lins_ccc(assay1, assay2):.5f}"
      f"   <- no, they disagree badly")
ba = bland_altman(assay1, assay2)
print(f"  Bland-Altman bias         = {ba['bias']:+.2f}")
print(f"  95% limits of agreement   = [{ba['loa'][0]:+.2f}, {ba['loa'][1]:+.2f}]")
print("\n  r near 1 with CCC near 0: the assays rank samples identically but")
print("  report entirely different numbers. For method comparison always use")
print("  (10.8)-(10.9), never r.")

# %% [markdown]
# ## 6. Correlation matrices: multiplicity and shrinkage, eq. (10.10)

# %%
header("6. Co-expression under multiplicity, and shrinkage (10.10)")

rng = np.random.default_rng(1006)
G, n_samp = 200, 20
# COMPLETELY independent features: every correlation we find is noise.
X = rng.normal(0, 1, (n_samp, G))
R = np.corrcoef(X, rowvar=False)
iu = np.triu_indices(G, 1)
rs = R[iu]
# t-test for each correlation: t = r sqrt((n-2)/(1-r^2)), df = n-2
tvals = rs * np.sqrt((n_samp - 2) / (1 - rs**2))
pvals = 2 * st.t.sf(np.abs(tvals), n_samp - 2)

print(f"  G = {G} features, n = {n_samp} samples, ALL truly independent")
print(f"  number of pairwise tests = G(G-1)/2 = {len(rs):,}")
print(f"  'significant' at p < 0.05 (unadjusted): {np.sum(pvals < 0.05):,} "
      f"({np.mean(pvals < 0.05):.1%})")
print(f"  |r| > 0.5 purely by chance             : {np.sum(np.abs(rs) > 0.5):,}")
from statsmodels.stats.multitest import multipletests
print(f"  surviving BH at 0.05                   : "
      f"{np.sum(multipletests(pvals, method='fdr_bh')[0]):,}")
print("\n  Co-expression networks built from unadjusted correlations are mostly")
print("  noise, especially at the small n typical of omics.")

def ledoit_wolf_shrinkage(X):
    """stats.md eq. (10.10): shrink the sample covariance to a scaled identity."""
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    T = mu * np.eye(p)                              # structured target
    d2 = np.sum((S - T) ** 2) / p
    b2_bar = np.mean([np.sum((np.outer(Xc[i], Xc[i]) - S) ** 2) / p
                      for i in range(n)]) / n
    b2 = min(b2_bar, d2)
    lam = b2 / d2
    return (1 - lam) * S + lam * T, lam


S_shrunk, lam = ledoit_wolf_shrinkage(X)
S_raw = np.cov(X, rowvar=False)
print(f"\n  p={G} > n={n_samp}, so the sample covariance is SINGULAR:")
print(f"    rank(S_sample) = {np.linalg.matrix_rank(S_raw)} (needs {G})")
print(f"    Ledoit-Wolf shrinkage intensity lambda = {lam:.4f}")
print(f"    rank(S_shrunk) = {np.linalg.matrix_rank(S_shrunk)}  -> invertible,")
print("    which is what makes the precision matrix of eq. (10.6) computable.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4))

a, b = cases["quadratic"]
axes[0].scatter(a, b, s=3, alpha=0.3)
axes[0].set_title(f"r = {st.pearsonr(a,b).statistic:.3f}, yet deterministic",
                  fontsize=9)

for s, grp in d.groupby("subject"):
    axes[1].scatter(grp["x"], grp["y"], s=12)
xs = np.linspace(d["x"].min(), d["x"].max(), 10)
axes[1].plot(xs, np.polyval(np.polyfit(d["x"], d["y"], 1), xs), "k--", lw=2,
             label=f"pooled r={pooled_r:+.2f}")
axes[1].set_title("Eq. (10.7): Simpson's paradox", fontsize=9)
axes[1].legend(fontsize=7)

diff = assay1 - assay2
avg = (assay1 + assay2) / 2
axes[2].scatter(avg, diff, s=10)
axes[2].axhline(ba["bias"], color="k")
for v in ba["loa"]:
    axes[2].axhline(v, color="red", ls="--")
axes[2].set_xlabel("mean of the two assays"); axes[2].set_ylabel("difference")
axes[2].set_title("Eq. (10.8): Bland-Altman", fontsize=9)

axes[3].hist(rs, bins=60, color="steelblue")
axes[3].axvline(0.5, color="red", ls="--"); axes[3].axvline(-0.5, color="red", ls="--")
axes[3].set_xlabel("pairwise r among INDEPENDENT features")
axes[3].set_title(f"n={n_samp}: |r|>0.5 happens {np.mean(np.abs(rs)>0.5):.1%} "
                  f"of the time", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "correlation.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/correlation.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 10)
#
# 1. Pearson for linear association on a justified scale; Spearman/Kendall for
#    monotone association or ordinal data.
# 2. Handle donor structure explicitly (10.7) before interpreting correlation
#    across repeated measurements.
# 3. Use Fisher's $z$ (10.3) for intervals and tests.
# 4. For method comparison, report (10.8)-(10.9), not $r$.
# 5. Apply log-ratio transformation before correlating compositions (Topic 26).
#
# **Next:** `11_linear_models.py`
