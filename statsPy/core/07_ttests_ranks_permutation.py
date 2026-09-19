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
# # Module 07: t-tests, rank tests, and permutation tests
#
# **Curriculum link:** `stats.md` -> Topic 7, equations (7.1)-(7.7)
#
# ## What you will learn
#
# 1. Pooled vs Welch (7.2)-(7.4), and **why Welch should be your default**.
# 2. That a t-test *is* a linear model (7.5): verified to machine precision.
# 3. What Mann-Whitney actually tests (7.6)-(7.7): a probabilistic index, which
#    equals the ROC AUC: **not** a median difference.
# 4. Permutation tests as the exact reference (4.12), and when exchangeability
#    fails.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "07_ttests_ranks_permutation"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The three t-tests, from scratch, eq. (7.1)-(7.4)

# %%
header("1. Implementing the t-tests from their formulas (7.1)-(7.4)")

def t_pooled(x, y):
    """stats.md eq. (7.2): equal-variance two-sample t."""
    n1, n2 = len(x), len(y)
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    t = (x.mean() - y.mean()) / se
    df = n1 + n2 - 2
    return t, df, 2 * st.t.sf(abs(t), df)


def t_welch(x, y):
    """stats.md eq. (7.3)-(7.4): Welch-Satterthwaite."""
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    se = np.sqrt(v1 + v2)
    t = (x.mean() - y.mean()) / se
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return t, df, 2 * st.t.sf(abs(t), df)


rng = np.random.default_rng(701)
x = rng.normal(0.0, 1.0, 8)
y = rng.normal(0.8, 2.5, 20)           # different mean AND different variance

tp, dfp, pp = t_pooled(x, y)
tw, dfw, pw = t_welch(x, y)
print(f"  pooled (7.2): t={tp:+.4f}  df={dfp:.0f}      p={pp:.4f}")
print(f"  Welch  (7.3): t={tw:+.4f}  df={dfw:.2f}   p={pw:.4f}")
print(f"  scipy  check: t={st.ttest_ind(x,y,equal_var=False).statistic:+.4f}  "
      f"p={st.ttest_ind(x,y,equal_var=False).pvalue:.4f}")
print(f"\n  Welch df lies between min(n)-1={min(len(x),len(y))-1} and "
      f"n1+n2-2={len(x)+len(y)-2}, as the theory requires.")

# %% [markdown]
# ### Why Welch should be the default
#
# The pooled test is badly anticonservative when the **smaller** group has the
# **larger** variance: a common situation (a small treated arm that also
# responds heterogeneously).

# %%
header("Calibration under unequal variances and unequal n")

def fpr(n1, n2, sd1, sd2, n_sim=6000, seed=0):
    """False-positive rate of each test under a TRUE null (equal means)."""
    rng = np.random.default_rng(seed)
    pooled_hits = welch_hits = 0
    for _ in range(n_sim):
        a = rng.normal(0, sd1, n1)
        b = rng.normal(0, sd2, n2)
        pooled_hits += t_pooled(a, b)[2] < 0.05
        welch_hits += t_welch(a, b)[2] < 0.05
    return pooled_hits / n_sim, welch_hits / n_sim


print(f"{'n1':>4}{'n2':>4}{'sd1':>6}{'sd2':>6}{'pooled FPR':>13}{'Welch FPR':>12}")
for (n1, n2, s1, s2) in [(10, 10, 1, 1), (10, 10, 1, 3),
                         (5, 25, 3, 1), (5, 25, 1, 3), (25, 5, 1, 3)]:
    fp, fw = fpr(n1, n2, s1, s2, seed=n1 * 100 + n2 + int(s1 * 10))
    flag = "  <-- BROKEN" if fp > 0.08 else ""
    print(f"{n1:>4}{n2:>4}{s1:>6}{s2:>6}{fp:>12.1%}{fw:>12.1%}{flag}")

print("\n  Welch holds ~5% everywhere. Pooled fails badly when the SMALL group")
print("  has the LARGE variance. Cost of using Welch when variances are equal:")
print("  essentially nothing. This is why R's t.test() defaults to Welch.")
print("\n  DO NOT pre-test variances and then choose: the two-stage procedure")
print("  has its own distorted error rate.")

# %% [markdown]
# ## 2. A t-test IS a linear model, eq. (7.5)
#
# $$y_i=\beta_0+\beta_1 x_i+\varepsilon_i, \qquad \hat\beta_1=\bar y_2-\bar y_1$$
#
# Once you see this, ANOVA, covariate adjustment, blocking, interactions and
# `limma` are all the same object.

# %%
header("2. t-test == lm == ANOVA, to machine precision (7.5)")

rng = np.random.default_rng(702)
g1 = rng.normal(5.0, 1.0, 15)
g2 = rng.normal(6.2, 1.0, 15)
df = pd.DataFrame({"y": np.concatenate([g1, g2]),
                   "g": ["A"] * 15 + ["B"] * 15})

fit = smf.ols("y ~ g", data=df).fit()
tt = st.ttest_ind(g1, g2, equal_var=True)
anova_f = st.f_oneway(g1, g2)

print(f"  lm coefficient beta1        = {fit.params['g[T.B]']:.10f}")
print(f"  difference of means         = {g2.mean() - g1.mean():.10f}")
print(f"  lm t-statistic              = {fit.tvalues['g[T.B]']:.10f}")
print(f"  pooled t-test statistic     = {-tt.statistic:.10f}")
print(f"  lm p-value                  = {fit.pvalues['g[T.B]']:.10e}")
print(f"  t-test p-value              = {tt.pvalue:.10e}")
print(f"\n  F from one-way ANOVA        = {anova_f.statistic:.10f}")
print(f"  t^2                         = {tt.statistic**2:.10f}   <- eq. (B.6)")
assert np.isclose(fit.tvalues["g[T.B]"], -tt.statistic)
assert np.isclose(anova_f.statistic, tt.statistic ** 2)
print("\n  Assertions passed: they are the same computation.")

# A paired t-test is a one-sample t-test on the differences (eq. 7.1):
rng = np.random.default_rng(703)
subj = rng.normal(0, 2, 14)
pre = subj + rng.normal(0, 0.7, 14)
post = subj + rng.normal(0, 0.7, 14) + 0.9
print(f"\n  paired t-test p             = {st.ttest_rel(post, pre).pvalue:.6f}")
print(f"  one-sample t on differences = {st.ttest_1samp(post - pre, 0).pvalue:.6f}")

# %% [markdown]
# ## 3. What Mann-Whitney actually tests, eq. (7.6)-(7.7)
#
# $$\frac{U_1}{n_1n_2}=\widehat{\Pr}(Y_1>Y_2)+\tfrac12\widehat{\Pr}(Y_1=Y_2)=\widehat{\mathrm{AUC}}$$
#
# It estimates a **probabilistic index**, identical to the ROC AUC. It tests
# medians *only* under the extra assumption of a pure location shift with
# identical shapes.

# %%
header("3. Mann-Whitney = AUC, not a median test (7.6)-(7.7)")

def mann_whitney_from_scratch(x, y):
    """stats.md eq. (7.6)-(7.7) computed directly from ranks."""
    n1, n2 = len(x), len(y)
    pooled = np.concatenate([x, y])
    ranks = st.rankdata(pooled)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    return {"U1": U1, "AUC": U1 / (n1 * n2),
            "E_U": n1 * n2 / 2,
            "Var_U": n1 * n2 * (n1 + n2 + 1) / 12,
            "rank_biserial": 2 * U1 / (n1 * n2) - 1}


rng = np.random.default_rng(704)
a = rng.normal(0, 1, 40)
b = rng.normal(0.7, 1, 50)
res = mann_whitney_from_scratch(a, b)
scipy_u = st.mannwhitneyu(a, b, alternative="two-sided")
# AUC computed the direct way: proportion of (a,b) pairs with a > b.
direct_auc = np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :])

print(f"  U1 from ranks (7.6)       = {res['U1']:.1f}")
print(f"  U from scipy              = {scipy_u.statistic:.1f}")
print(f"  AUC = U1/(n1*n2)  (7.7)   = {res['AUC']:.6f}")
print(f"  direct P(a > b)           = {direct_auc:.6f}   <- identical")
print(f"  rank-biserial correlation = {res['rank_biserial']:+.4f}")

# %% [markdown]
# ### The counterexample: identical medians, tiny Wilcoxon p-value

# %%
print("\n--- Two distributions with the SAME median but different shape ---")
rng = np.random.default_rng(705)
# Group 1: symmetric. Group 2: same median, but heavily right-skewed.
g1 = rng.normal(0, 1, 300)
g2 = np.concatenate([rng.normal(-0.55, 0.30, 150), rng.normal(1.8, 1.2, 150)])
g2 = g2 - np.median(g2) + np.median(g1)          # force medians to match exactly

print(f"  median group 1 = {np.median(g1):+.4f}")
print(f"  median group 2 = {np.median(g2):+.4f}   (matched by construction)")
print(f"  mean   group 1 = {g1.mean():+.4f}")
print(f"  mean   group 2 = {g2.mean():+.4f}")
print(f"\n  Mann-Whitney p = {st.mannwhitneyu(g1, g2).pvalue:.2e}"
      f"   <- 'significant' despite identical medians")
print(f"  AUC            = {mann_whitney_from_scratch(g1, g2)['AUC']:.4f}")
print(f"  Welch t-test p = {st.ttest_ind(g1, g2, equal_var=False).pvalue:.4f}")
print("\n  CONCLUSION: 'use Wilcoxon to compare medians' is FALSE in general.")
print("  Report what it estimates: P(Y1 > Y2), i.e. the AUC.")

# %% [markdown]
# ## 4. Choosing a test: a power comparison
#
# Neither test dominates. The right choice follows the estimand and the shape.

# %%
header("4. Relative power under three data-generating processes")

def compare_power(sampler_a, sampler_b, n=15, n_sim=3000, seed=0):
    rng = np.random.default_rng(seed)
    t_hits = w_hits = p_hits = 0
    for _ in range(n_sim):
        a, b = sampler_a(rng, n), sampler_b(rng, n)
        t_hits += st.ttest_ind(a, b, equal_var=False).pvalue < 0.05
        w_hits += st.mannwhitneyu(a, b).pvalue < 0.05
        # Permutation on the difference of means (exact under exchangeability)
        pooled = np.concatenate([a, b])
        obs = abs(a.mean() - b.mean())
        cnt = sum(abs(np.mean(pr[:n]) - np.mean(pr[n:])) >= obs
                  for pr in (rng.permutation(pooled) for _ in range(199)))
        p_hits += (1 + cnt) / 200 < 0.05
    return t_hits / n_sim, w_hits / n_sim, p_hits / n_sim


cases = {
    "Normal, shift 0.8": (lambda r, k: r.normal(0, 1, k),
                          lambda r, k: r.normal(0.8, 1, k)),
    "Log-normal, ratio 2": (lambda r, k: r.lognormal(0, 1, k),
                            lambda r, k: 2 * r.lognormal(0, 1, k)),
    "Heavy tails (t3), shift 0.8": (lambda r, k: r.standard_t(3, k),
                                    lambda r, k: r.standard_t(3, k) + 0.8),
    "TRUE NULL (calibration)": (lambda r, k: r.normal(0, 1, k),
                                lambda r, k: r.normal(0, 1, k)),
}
print(f"{'scenario':<30}{'Welch t':>10}{'Wilcoxon':>11}{'permutation':>13}")
for name, (fa, fb) in cases.items():
    a_, b_, c_ = compare_power(fa, fb, n=15, n_sim=1200, seed=abs(hash(name)) % 997)
    print(f"{name:<30}{a_:>10.1%}{b_:>11.1%}{c_:>13.1%}")

print("\n  Normal data      -> t wins slightly.")
print("  Log-normal ratio -> Wilcoxon wins substantially (it is scale-free,")
print("                      and the estimand is a stochastic ordering).")
print("  Heavy tails      -> Wilcoxon wins (ranks are robust).")
print("  TRUE NULL        -> all three hold ~5%: all are valid, just differently")
print("                      powerful. Choose by ESTIMAND, not by a pre-test.")

# %% [markdown]
# ## 5. Permutation and the limits of exchangeability
#
# Permutation tests are exact for Type I error under exchangeability: but
# exchangeability is destroyed by clustering and batch effects.

# %%
header("5. Permutation is exact ONLY under exchangeability")

rng = np.random.default_rng(706)

def perm_p(a, b, B=499, rng=None):
    pooled = np.concatenate([a, b])
    n = len(a)
    obs = abs(a.mean() - b.mean())
    cnt = sum(abs(np.mean(p[:n]) - np.mean(p[n:])) >= obs
              for p in (rng.permutation(pooled) for _ in range(B)))
    return (1 + cnt) / (B + 1)


# (a) Valid: independent observations under a true null.
hits_valid = 0
for _ in range(800):
    hits_valid += perm_p(rng.normal(0, 1, 8), rng.normal(0, 1, 8), rng=rng) < 0.05

# (b) Invalid: observations are CELLS from clustered donors; the permutation
#     null destroys the donor structure that is actually present.
hits_clustered = 0
for _ in range(800):
    b_eff = rng.normal(0, 1.0, 8)                       # 8 donors, 4 per arm
    cells = b_eff[:, None] + rng.normal(0, 1.0, (8, 30))
    hits_clustered += perm_p(cells[:4].ravel(), cells[4:].ravel(), B=199,
                             rng=rng) < 0.05

print(f"  permuting independent observations : FPR = {hits_valid/800:.1%}  (valid)")
print(f"  permuting CELLS from clustered data: FPR = {hits_clustered/800:.1%}  "
      f"(broken)")
print("\n  Permutation does not rescue pseudoreplication. To stay valid you must")
print("  permute at the level that was randomised - here, permute DONOR labels")
print("  and carry all of a donor's cells with them.")

# Correct version: permute donor labels.
hits_fixed = 0
for _ in range(800):
    b_eff = rng.normal(0, 1.0, 8)
    cells = b_eff[:, None] + rng.normal(0, 1.0, (8, 30))
    donor_means = cells.mean(axis=1)
    obs = abs(donor_means[:4].mean() - donor_means[4:].mean())
    cnt = sum(abs(pm[:4].mean() - pm[4:].mean()) >= obs
              for pm in (rng.permutation(donor_means) for _ in range(199)))
    hits_fixed += (1 + cnt) / 200 < 0.05
print(f"  permuting DONOR labels             : FPR = {hits_fixed/800:.1%}  (valid)")

# %% [markdown]
# ## 6. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(g1, bins=50, alpha=0.55, density=True, label="group 1")
axes[0].hist(g2, bins=50, alpha=0.55, density=True, label="group 2")
axes[0].axvline(np.median(g1), color="C0", ls="--")
axes[0].axvline(np.median(g2), color="C1", ls=":")
axes[0].set_title("Identical medians, tiny Wilcoxon p", fontsize=9)
axes[0].legend(fontsize=7)

# ROC curve = the Mann-Whitney statistic, visualised
scores = np.concatenate([a, b])
labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
order = np.argsort(-scores)
tpr = np.cumsum(labels[order]) / labels.sum()
fpr_ = np.cumsum(1 - labels[order]) / (1 - labels).sum()
axes[1].plot(fpr_, tpr, lw=2)
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
auc = np.trapezoid(tpr, fpr_) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr_)
axes[1].set_title(f"Eq. (7.7): AUC = {auc:.3f} = U/(n1 n2)", fontsize=9)
axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")

# Welch df as a function of the variance ratio
ratios = np.logspace(-1.5, 1.5, 100)
for (n1, n2) in [(5, 25), (15, 15), (25, 5)]:
    dfs = [(1/n1 + r**2/n2)**2 / ((1/n1)**2/(n1-1) + (r**2/n2)**2/(n2-1))
           for r in ratios]
    axes[2].semilogx(ratios, dfs, label=f"n1={n1}, n2={n2}")
axes[2].set_xlabel("sd2 / sd1"); axes[2].set_ylabel("Welch df")
axes[2].set_title("Eq. (7.4): Welch-Satterthwaite df", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "two_group_tests.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/two_group_tests.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 7)
#
# 1. Pairing comes from the design, never from a normality test.
# 2. Default to Welch for independent two-group comparisons.
# 3. Use a rank test when the *estimand* is rank-based, and then report
#    $\Pr(Y_1>Y_2)$ (7.7), not a median difference.
# 4. Normality applies to the sampling distribution of the estimate, not the raw
#    values.
#
# **Next:** `08_multiple_testing.py`
