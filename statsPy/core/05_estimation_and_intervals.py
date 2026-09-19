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
# # Module 05: Estimation, effect sizes, and confidence intervals
#
# **Curriculum link:** `stats.md` -> Topic 5, equations (5.1)-(5.15)
#
# Estimation, not dichotomised testing, is the primary scientific task.
#
# ## What you will learn
#
# 1. Bias-variance (5.1): why accepting bias can *reduce* total error.
# 2. What a confidence interval actually guarantees, by simulation (5.3)-(5.4).
# 3. Standardised effect sizes and the Hedges correction (5.5)-(5.6).
# 4. Ratio measures on the log scale, and the delta method (5.7)-(5.11).
# 5. Bootstrap intervals: percentile, basic, BCa (5.12)-(5.13) - and the
#    **resample-the-right-unit** rule.
# 6. Robust (sandwich) standard errors (5.14).
# 7. Equivalence testing (TOST, 5.15): the only valid way to argue "no effect".

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "05_estimation_and_intervals"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Bias-variance decomposition, eq. (5.1)
#
# $$\mathrm{MSE}(\hat\theta)=\underbrace{\big(\mathbb E[\hat\theta]-\theta\big)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}$$
#
# This licenses every shrinkage method in omics. We demonstrate it with a
# James-Stein-style shrinkage estimator: deliberately biased, yet lower MSE.

# %%
header("1. Bias-variance: why a biased estimator can win (5.1)")

rng = np.random.default_rng(501)
G, SIGMA = 60, 1.0                     # 60 "genes", each with one noisy estimate
theta_true = rng.normal(0.0, 0.7, G)   # true effects, mostly small

def shrink(y, lam):
    """Shrink every estimate toward the grand mean by factor (1 - lam)."""
    return y.mean() + (1 - lam) * (y - y.mean())


rows = []
for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    ests = np.array([shrink(theta_true + rng.normal(0, SIGMA, G), lam)
                     for _ in range(4000)])
    bias2 = np.mean((ests.mean(axis=0) - theta_true) ** 2)
    var = np.mean(ests.var(axis=0, ddof=1))
    mse = np.mean((ests - theta_true) ** 2)
    rows.append({"shrinkage": lam, "bias^2": bias2, "variance": var,
                 "MSE": mse, "bias^2+var": bias2 + var})
tab = pd.DataFrame(rows)
print(tab.round(4).to_string(index=False))
best = tab.loc[tab["MSE"].idxmin(), "shrinkage"]
print(f"\nMinimum MSE at shrinkage = {best}, NOT at 0 (the unbiased estimator).")
print("'bias^2+var' reproduces 'MSE' exactly - that is eq. (5.1).")

# %% [markdown]
# ## 2. What a confidence interval guarantees, eq. (5.2)-(5.4)
#
# A 95% CI is a **procedure** that covers the truth 95% of the time across
# repetitions. It is *not* a 95% probability statement about $\theta$ given this
# dataset. Eq. (5.4): the duality - says the CI is the set of null values that
# would not be rejected.

# %%
header("2. Coverage, and the CI/test duality (5.3)-(5.4)")

rng = np.random.default_rng(502)
MU, SD, n = 5.0, 2.0, 12
hits = 0
widths = []
for _ in range(20000):
    x = rng.normal(MU, SD, n)
    half = st.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)   # eq. (5.3)
    widths.append(2 * half)
    hits += abs(x.mean() - MU) <= half
print(f"  nominal coverage 95%, empirical = {hits/20000:.2%}")
print(f"  mean interval width = {np.mean(widths):.3f}")
print("  Intervals vary in width from sample to sample; ~1 in 20 misses entirely.")

# Duality: invert the test.
rng = np.random.default_rng(503)
x = rng.normal(MU, SD, n)
grid = np.linspace(x.mean() - 4, x.mean() + 4, 4001)
pvals = np.array([st.ttest_1samp(x, m).pvalue for m in grid])
not_rejected = grid[pvals > 0.05]
lo_t, hi_t = st.t.interval(0.95, n - 1, loc=x.mean(),
                           scale=st.sem(x))
print(f"\n  CI from the t formula        : [{lo_t:.4f}, {hi_t:.4f}]")
print(f"  Set of mu with p > 0.05      : [{not_rejected.min():.4f}, "
      f"{not_rejected.max():.4f}]   <- eq. (5.4), identical")

# %% [markdown]
# ## 3. Standardised effect sizes, eq. (5.5)-(5.6)
#
# $$d=\frac{\bar y_1-\bar y_2}{s_p}, \qquad g=J(\nu)\,d, \qquad
#   J(\nu)=\frac{\Gamma(\nu/2)}{\sqrt{\nu/2}\,\Gamma((\nu-1)/2)}\approx 1-\frac{3}{4\nu-1}$$

# %%
header("3. Cohen's d and the Hedges correction (5.5)-(5.6)")

from scipy.special import gammaln

def cohens_d(x, y):
    """stats.md eq. (5.5)."""
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1))
                 / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp


def hedges_J(nu):
    """Exact correction factor of stats.md eq. (5.6)."""
    return np.exp(gammaln(nu / 2) - 0.5 * np.log(nu / 2) - gammaln((nu - 1) / 2))


print(f"{'n per group':>12} {'nu':>5} {'exact J':>10} {'1-3/(4nu-1)':>13} "
      f"{'d overstated by':>17}")
for n in [3, 4, 6, 10, 20, 50]:
    nu = 2 * n - 2
    J = hedges_J(nu)
    print(f"{n:>12} {nu:>5} {J:>10.4f} {1 - 3/(4*nu - 1):>13.4f} "
          f"{100*(1/J - 1):>16.1f}%")

# Verify the bias is real, by simulation.
rng = np.random.default_rng(504)
TRUE_D = 0.8
n = 4
ds, gs = [], []
for _ in range(20000):
    a = rng.normal(TRUE_D, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    d = cohens_d(a, b)
    ds.append(d)
    gs.append(hedges_J(2 * n - 2) * d)
print(f"\n  true d = {TRUE_D},  n = {n} per group")
print(f"  mean Cohen's d  = {np.mean(ds):.4f}   (biased upward)")
print(f"  mean Hedges' g  = {np.mean(gs):.4f}   (corrected)")
print("\n  CAUTION: d is a RATIO. A large d can mean a tiny denominator.")
print("  Always report the raw difference and its units alongside.")

# %% [markdown]
# ## 4. Ratio measures and the delta method, eq. (5.7)-(5.11)
#
# Odds ratios, risk ratios and fold changes must be built on the **log scale**,
# where the sampling distribution is far closer to normal, then exponentiated.

# %%
header("4. Odds ratio, risk ratio, delta method (5.7)-(5.10)")

# A 2x2 table: exposed/unexposed x event/no event
a, b, c, d_ = 30, 70, 15, 85     # a,b = exposed(event, no); c,d = unexposed

or_hat = (a * d_) / (b * c)                                   # eq. (5.7)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d_)
or_ci = np.exp(np.log(or_hat) + np.array([-1, 1]) * 1.96 * se_log_or)

risk1, risk0 = a / (a + b), c / (c + d_)
rr_hat = risk1 / risk0                                        # eq. (5.8)
se_log_rr = np.sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d_))
rr_ci = np.exp(np.log(rr_hat) + np.array([-1, 1]) * 1.96 * se_log_rr)

print(f"  baseline risk (unexposed) = {risk0:.3f}")
print(f"  odds ratio  = {or_hat:.3f}   95% CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}]")
print(f"  risk ratio  = {rr_hat:.3f}   95% CI [{rr_ci[0]:.3f}, {rr_ci[1]:.3f}]")
print(f"  risk difference = {risk1 - risk0:+.3f}")
print("\n  OR > RR always (when risk > 0). They coincide only for RARE outcomes.")

# Conversion formula: RR = OR / (1 - p0 + p0*OR)
for p0 in [0.01, 0.10, 0.40]:
    OR = 2.0
    rr = OR / (1 - p0 + p0 * OR)
    print(f"    baseline risk {p0:>5.2f}:  OR = 2.00  corresponds to RR = {rr:.3f}")

# Delta method check (eq. 5.9) by simulation.
rng = np.random.default_rng(505)
sims = []
for _ in range(20000):
    ea = rng.binomial(100, 0.30)
    ec = rng.binomial(100, 0.15)
    if 0 < ea < 100 and 0 < ec < 100:
        sims.append(np.log((ea * (100 - ec)) / ((100 - ea) * ec)))
print(f"\n  delta-method SE(log OR) = {se_log_or:.4f}")
print(f"  simulated  SD(log OR)   = {np.std(sims, ddof=1):.4f}   <- eq. (5.9) works")

# %% [markdown]
# ## 5. Bootstrap intervals, eq. (5.12)-(5.13), and the unit rule

# %%
header("5. Bootstrap: percentile, basic, BCa (5.12)-(5.13)")

def bootstrap_ci(x, statistic, B=4000, alpha=0.05, seed=0):
    """Percentile, basic (5.12) and BCa (5.13) intervals for a 1-D sample."""
    rng = np.random.default_rng(seed)
    n = len(x)
    theta_hat = statistic(x)
    boot = np.array([statistic(x[rng.integers(0, n, n)]) for _ in range(B)])

    pct = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    basic = np.array([2 * theta_hat - pct[1], 2 * theta_hat - pct[0]])

    # --- BCa: bias correction z0 and acceleration a from the jackknife -------
    z0 = st.norm.ppf(np.mean(boot < theta_hat))
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    acc = num / den if den != 0 else 0.0
    zl, zu = st.norm.ppf(alpha / 2), st.norm.ppf(1 - alpha / 2)
    a1 = st.norm.cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
    a2 = st.norm.cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))
    bca = np.percentile(boot, [100 * a1, 100 * a2])
    return {"percentile": pct, "basic": basic, "BCa": bca,
            "estimate": theta_hat}


rng = np.random.default_rng(506)
skewed = rng.lognormal(mean=1.0, sigma=0.8, size=25)   # e.g. cytokine conc.
res = bootstrap_ci(skewed, np.mean, seed=1)
print(f"  sample (n=25, log-normal), mean = {res['estimate']:.3f}")
for k in ["percentile", "basic", "BCa"]:
    print(f"    {k:<11}: [{res[k][0]:.3f}, {res[k][1]:.3f}]")
lo, hi = st.t.interval(0.95, 24, loc=skewed.mean(), scale=st.sem(skewed))
print(f"    {'t-interval':<11}: [{lo:.3f}, {hi:.3f}]")
print("\n  BCa is asymmetric here, correctly reflecting the skew. Percentile and")
print("  basic differ because the bootstrap distribution is not symmetric.")

# %% [markdown]
# ### The rule that matters most: resample the EXPERIMENTAL UNIT

# %%
print("\n--- Resampling the wrong level reproduces pseudoreplication ---")
rng = np.random.default_rng(507)
n_donors, m_cells = 6, 300
b = rng.normal(0, 1.0, n_donors)
cells = b[:, None] + rng.normal(0, 1.0, (n_donors, m_cells))

# WRONG: resample cells, ignoring donor structure.
flat = cells.ravel()
wrong = np.array([flat[rng.integers(0, len(flat), len(flat))].mean()
                  for _ in range(3000)])
# RIGHT: resample DONORS (a cluster bootstrap), carrying all their cells.
right = np.array([cells[rng.integers(0, n_donors, n_donors)].mean()
                  for _ in range(3000)])

print(f"  bootstrap SE resampling cells  : {wrong.std(ddof=1):.4f}")
print(f"  bootstrap SE resampling donors : {right.std(ddof=1):.4f}")
print(f"  truth sqrt(sigma_b^2/n + ...)  : "
      f"{np.sqrt(1.0/n_donors + 1.0/(n_donors*m_cells)):.4f}   (eq. 1.5)")
print("  The cell bootstrap is ~{:.0f}x too optimistic.".format(
    right.std(ddof=1) / wrong.std(ddof=1)))

# %% [markdown]
# ## 6. Robust (sandwich) standard errors, eq. (5.14)
#
# When the mean model is right but the variance model is wrong, HC standard
# errors stay valid. **Use HC3 when $n<250$**: HC0 is badly anticonservative in
# the small samples typical of bioinformatics.

# %%
header("6. Heteroscedasticity-consistent SEs (5.14)")

rng = np.random.default_rng(508)
n = 40
x = rng.uniform(0, 3, n)
# Variance grows strongly with x: classic heteroscedasticity.
y = 1.0 + 0.0 * x + rng.normal(0, 0.3 + 1.2 * x, n)   # TRUE slope is 0
df = pd.DataFrame({"x": x, "y": y})
fit = smf.ols("y ~ x", data=df).fit()
print(f"  {'method':<16}{'SE(slope)':>12}{'p-value':>10}")
print(f"  {'classical OLS':<16}{fit.bse['x']:>12.4f}{fit.pvalues['x']:>10.4f}")
for cov in ["HC0", "HC1", "HC2", "HC3"]:
    r = fit.get_robustcov_results(cov_type=cov)
    print(f"  {cov:<16}{r.bse[1]:>12.4f}{r.pvalues[1]:>10.4f}")

# Calibration: which one actually holds its 5% error rate here?
counts = {k: 0 for k in ["classical", "HC0", "HC3"]}
for _ in range(1500):
    xx = rng.uniform(0, 3, n)
    yy = 1.0 + rng.normal(0, 0.3 + 1.2 * xx, n)      # null: slope = 0
    f = smf.ols("y ~ x", data=pd.DataFrame({"x": xx, "y": yy})).fit()
    counts["classical"] += f.pvalues["x"] < 0.05
    counts["HC0"] += f.get_robustcov_results("HC0").pvalues[1] < 0.05
    counts["HC3"] += f.get_robustcov_results("HC3").pvalues[1] < 0.05
print("\n  False-positive rate under a TRUE null (nominal 5%), n = 40:")
for k, v in counts.items():
    print(f"    {k:<11}: {v/1500:.1%}")
print("  HC3 is the right default at bioinformatics sample sizes.")

# %% [markdown]
# ## 7. Equivalence testing (TOST), eq. (5.15)
#
# $p>0.05$ is **not** evidence of no effect. To claim equivalence you must
# pre-specify a margin $\Delta$ and reject both one-sided nulls. Equivalently:
# the $100(1-2\alpha)\%$ CI must lie entirely inside $(-\Delta,\Delta)$.

# %%
header("7. TOST: the only valid way to claim 'no meaningful difference' (5.15)")

def tost(x, y, delta, alpha=0.05):
    """Two one-sided tests for equivalence within +/- delta."""
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
    dfree = (se**4) / ((x.var(ddof=1)/len(x))**2/(len(x)-1)
                       + (y.var(ddof=1)/len(y))**2/(len(y)-1))
    p_lower = st.t.sf((diff + delta) / se, dfree)     # H01: theta <= -delta
    p_upper = st.t.cdf((diff - delta) / se, dfree)    # H02: theta >= +delta
    p_tost = max(p_lower, p_upper)
    ci = diff + np.array([-1, 1]) * st.t.ppf(1 - alpha, dfree) * se  # 90% CI
    return {"diff": diff, "p_tost": p_tost, "ci_90": ci,
            "equivalent": p_tost < alpha}


rng = np.random.default_rng(509)
delta = 0.5    # pre-specified smallest biologically meaningful difference

print("Case A - small n, truly no effect (the ambiguous case):")
a, b_ = rng.normal(0, 1, 10), rng.normal(0, 1, 10)
r = tost(a, b_, delta)
print(f"  difference = {r['diff']:+.3f}, NHST p = "
      f"{st.ttest_ind(a, b_, equal_var=False).pvalue:.3f} (not significant)")
print(f"  TOST p = {r['p_tost']:.3f}, 90% CI [{r['ci_90'][0]:+.3f}, "
      f"{r['ci_90'][1]:+.3f}] -> equivalent? {r['equivalent']}")
print("  Not significant AND not equivalent = the study was simply inconclusive.")

print("\nCase B - large n, truly no effect:")
a, b_ = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
r = tost(a, b_, delta)
print(f"  difference = {r['diff']:+.3f}, NHST p = "
      f"{st.ttest_ind(a, b_, equal_var=False).pvalue:.3f}")
print(f"  TOST p = {r['p_tost']:.4f}, 90% CI [{r['ci_90'][0]:+.3f}, "
      f"{r['ci_90'][1]:+.3f}] -> equivalent? {r['equivalent']}")
print("  Now we can positively claim the difference is smaller than delta.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(tab["shrinkage"], tab["bias^2"], "o-", label="bias$^2$")
axes[0].plot(tab["shrinkage"], tab["variance"], "s-", label="variance")
axes[0].plot(tab["shrinkage"], tab["MSE"], "^-", lw=2, label="MSE")
axes[0].set_xlabel("shrinkage toward the mean")
axes[0].set_title("Eq. (5.1): bias-variance trade-off")
axes[0].legend(fontsize=7)

rng = np.random.default_rng(510)
for i in range(40):
    xx = rng.normal(MU, SD, n)
    h = st.t.ppf(0.975, n-1) * xx.std(ddof=1)/np.sqrt(n)
    covered = abs(xx.mean() - MU) <= h
    axes[1].plot([xx.mean()-h, xx.mean()+h], [i, i],
                 color="C0" if covered else "red", lw=1.5)
axes[1].axvline(MU, color="k", ls="--")
axes[1].set_title("Eq. (5.3): 95% CIs across repetitions")
axes[1].set_xlabel("estimate")
axes[1].set_yticks([])

boot = np.array([np.mean(skewed[np.random.default_rng(i).integers(0, 25, 25)])
                 for i in range(4000)])
axes[2].hist(boot, bins=60, color="lightsteelblue")
for k, c in [("percentile", "C1"), ("BCa", "C2")]:
    for v in res[k]:
        axes[2].axvline(v, color=c, ls="--", lw=1.5)
axes[2].axvline(res["estimate"], color="k", lw=2)
axes[2].set_title("Eq. (5.12)-(5.13): bootstrap distribution")
axes[2].set_xlabel("bootstrap mean")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "estimation.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/estimation.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 5)
#
# 1. Report direction, magnitude, interval and units. "Upregulated, p = 0.003"
#    is not a result.
# 2. Build intervals on the scale where the sampling distribution is symmetric
#    (log for ratios, Fisher z for correlations, logit for proportions), then
#    back-transform.
# 3. To claim equivalence, pre-specify $\Delta$ and use TOST (5.15).
# 4. Bootstrap the experimental unit, and keep learned preprocessing inside the
#    resample.
#
# **Next:** `06_hypothesis_tests_and_pvalues.py`
