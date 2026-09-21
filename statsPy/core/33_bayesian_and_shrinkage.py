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
# # Module 33: Bayesian reasoning and hierarchical shrinkage
#
# **Curriculum link:** `stats.md` -> Topic 33, equations (33.1)-(33.7)
#
# With 3 replicates per group and 20,000 genes, each gene alone is
# uninformative: but the **ensemble** is not.
#
# ## What you will learn
#
# 1. Conjugate updating (33.2)-(33.3), and that the posterior mean is a
#    **precision-weighted average**.
# 2. **The James-Stein theorem (33.5)**: shrinkage is a frequentist result, not
#    a Bayesian assumption.
# 3. Empirical Bayes in practice: a from-scratch **moderated t-test** (limma's
#    eq. 20.6) that beats the ordinary t-test at small $n$.
# 4. Posterior predictive checks (33.6).
# 5. A Metropolis sampler with convergence diagnostics.
# 6. Why Bayes factors (33.7) are prior-sensitive.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import digamma, polygamma
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "33_bayesian_and_shrinkage"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Conjugate updating, eq. (33.2)-(33.3)

# %%
header("1. Beta-binomial and normal-normal conjugacy (33.2)-(33.3)")

# Beta-binomial: an allele frequency / response rate.
a0, b0 = 2.0, 8.0                       # prior: ~20%, worth a+b = 10 observations
y_obs, n_obs = 14, 40
a1, b1 = a0 + y_obs, b0 + n_obs - y_obs
print(f"  prior Beta({a0}, {b0}): mean {a0/(a0+b0):.3f}, "
      f"equivalent to {a0+b0:.0f} pseudo-observations")
print(f"  data: {y_obs}/{n_obs} = {y_obs/n_obs:.3f}")
print(f"  posterior Beta({a1}, {b1}): mean {a1/(a1+b1):.3f}")
lo, hi = st.beta.ppf([0.025, 0.975], a1, b1)
print(f"  95% CREDIBLE interval [{lo:.3f}, {hi:.3f}]")
print(f"  P(theta > 0.30 | data) = {st.beta.sf(0.30, a1, b1):.4f}")
print("  A credible interval IS a probability statement about theta given the")
print("  data - which is what people wrongly believe a confidence interval to be.")

# Normal-normal: the shrinkage formula, eq. (33.3).
print("\n  Normal-normal (33.3): posterior mean = lambda*y + (1-lambda)*mu0")
mu0, tau2, sigma2 = 0.0, 1.0, 1.0
print(f"  {'sigma^2 (data noise)':>22}{'lambda':>10}{'weight on data':>17}")
for s2 in [0.1, 0.5, 1.0, 4.0, 20.0]:
    lam = tau2 / (tau2 + s2)
    print(f"  {s2:>22.1f}{lam:>10.3f}{lam:>16.1%}")
print("  Noisier data -> more shrinkage toward the prior. Compare with the")
print("  mixed-model BLUP, eq. (14.6): it is the SAME formula.")

# %% [markdown]
# ## 2. The James-Stein theorem, eq. (33.5)
#
# For $G\ge3$ independent normal means, the shrunken estimator has **uniformly
# lower total risk** than the obvious one: a frequentist theorem.

# %%
header("2. James-Stein: shrinkage wins, provably (33.5)")

def james_stein(y, sigma2=1.0):
    """stats.md eq. (33.5). Positive-part version to avoid sign flips."""
    G = len(y)
    shrink = 1 - (G - 2) * sigma2 / np.sum(y ** 2)
    return max(shrink, 0.0) * y


rng = np.random.default_rng(2301)
print(f"  {'scenario':<34}{'MLE risk':>11}{'JS risk':>10}{'JS better?':>12}")
for label, theta in [
        ("all means zero", np.zeros(20)),
        ("means near zero (typical omics)", rng.normal(0, 0.4, 20)),
        ("means well spread", rng.normal(0, 2.0, 20)),
        ("one huge outlier", np.r_[np.zeros(19), 10.0])]:
    risk_mle, risk_js = [], []
    for _ in range(4000):
        y = theta + rng.normal(0, 1, len(theta))
        risk_mle.append(np.sum((y - theta) ** 2))
        risk_js.append(np.sum((james_stein(y) - theta) ** 2))
    print(f"  {label:<34}{np.mean(risk_mle):>11.2f}{np.mean(risk_js):>10.2f}"
          f"{str(np.mean(risk_js) < np.mean(risk_mle)):>12}")
print("\n  JS wins in every case (that is the theorem), and wins MOST when the")
print("  true effects are clustered near zero - exactly the omics situation,")
print("  where the vast majority of genes are not differentially expressed.")

# %% [markdown]
# ## 3. Empirical Bayes in practice: a moderated t-test, eq. (20.6)
#
# $$\tilde s_g^2=\frac{d_0s_0^2+d_gs_g^2}{d_0+d_g}, \qquad
#   \tilde t_g=\frac{\hat\beta_g}{\tilde s_g\sqrt{\ldots}}\sim t_{d_g+d_0}$$
#
# This is what `limma` does, and it is the single biggest practical win in
# small-$n$ omics. We implement it from scratch.

# %%
header("3. From-scratch moderated t-test (20.6)")

def fit_prior_variance(s2, df):
    """Estimate (d0, s0^2) by matching moments of log(s2), as limma does.

    Under the model s2_g ~ s0^2 * chi2_{d0} / d0, the variance of log(s2)
    exceeds trigamma(df/2) by exactly trigamma(d0/2). Solve for d0.
    """
    z = np.log(s2)
    e = z - digamma(df / 2) + np.log(df / 2)
    target = np.var(e, ddof=1) - polygamma(1, df / 2)
    if target <= 0:
        return np.inf, float(np.exp(np.mean(e)))           # infinite prior df
    f = lambda d0: polygamma(1, d0 / 2) - target
    lo, hi = 1e-4, 1e4
    if f(lo) * f(hi) > 0:
        return np.inf, float(np.exp(np.mean(e)))
    d0 = brentq(f, lo, hi)
    s0_2 = float(np.exp(np.mean(e) + digamma(d0 / 2) - np.log(d0 / 2)))
    return d0, s0_2


def moderated_t(A, B):
    """Two-group moderated t-test. A, B are G x n matrices."""
    n1, n2 = A.shape[1], B.shape[1]
    df = n1 + n2 - 2
    diff = A.mean(axis=1) - B.mean(axis=1)
    s2 = (((n1 - 1) * A.var(axis=1, ddof=1)
           + (n2 - 1) * B.var(axis=1, ddof=1)) / df)
    d0, s0_2 = fit_prior_variance(s2, df)
    if np.isinf(d0):
        s2_tilde, total_df = np.full_like(s2, s0_2), np.inf
    else:
        s2_tilde = (d0 * s0_2 + df * s2) / (d0 + df)        # eq. (20.6)
        total_df = df + d0
    se = np.sqrt(s2_tilde * (1 / n1 + 1 / n2))
    t = diff / se
    p = 2 * st.t.sf(np.abs(t), total_df)
    return {"t": t, "p": p, "s2": s2, "s2_tilde": s2_tilde,
            "d0": d0, "s0_2": s0_2, "df_total": total_df, "diff": diff}


rng = np.random.default_rng(2302)
G, n_rep, n_de = 6000, 3, 400
true_sd = np.sqrt(1 / rng.gamma(5, 1 / 5, G))     # gene-specific SDs
is_de = np.zeros(G, bool); is_de[:n_de] = True
delta = np.where(is_de, 1.6 * rng.choice([-1, 1], G), 0.0)
A = rng.normal(delta[:, None], true_sd[:, None], (G, n_rep))
B = rng.normal(0.0, true_sd[:, None], (G, n_rep))

mod = moderated_t(A, B)
ord_p = st.ttest_ind(A, B, axis=1).pvalue

print(f"  G = {G}, n = {n_rep} per group, {n_de} truly differential")
print(f"  estimated prior df d0 = {mod['d0']:.2f}   "
      f"(the ordinary test has only {2*n_rep-2} df)")
print(f"  moderated tests therefore use {mod['df_total']:.2f} df - the")
print(f"  equivalent of several extra samples, for free.")

print(f"\n  {'method':<22}{'rejected':>10}{'true pos':>10}{'false pos':>11}"
       f"{'sens':>8}{'FDP':>8}")
for name, pv in [("ordinary t", ord_p), ("moderated t (20.6)", mod["p"])]:
    q = st.false_discovery_control(pv)
    rej = q < 0.05
    tp, fp = np.sum(rej & is_de), np.sum(rej & ~is_de)
    print(f"  {name:<22}{rej.sum():>10}{tp:>10}{fp:>11}"
          f"{tp/n_de:>8.2f}{fp/max(rej.sum(),1):>8.3f}")

# The pathology moderation removes: a gene with an accidentally tiny variance.
worst = np.argsort(mod["s2"])[:5]
print(f"\n  The pathology this removes - genes with an accidentally tiny s_g:")
print(f"  {'gene':>8}{'s_g^2':>10}{'ordinary |t|':>14}{'moderated |t|':>15}"
      f"{'truly DE?':>11}")
for g in worst:
    print(f"  {g:>8}{mod['s2'][g]:>10.5f}"
          f"{abs(st.ttest_ind(A[g], B[g]).statistic):>14.2f}"
          f"{abs(mod['t'][g]):>15.2f}{str(bool(is_de[g])):>11}")
print("  Without moderation these dominate the top of the gene list purely")
print("  because their variance estimate was unluckily small.")

# %% [markdown]
# ## 4. Posterior predictive checks, eq. (33.6)

# %%
header("4. Posterior predictive check: can the model reproduce the data? (33.6)")

rng = np.random.default_rng(2303)
# Real data: negative binomial (overdispersed). Fitted model: Poisson.
y_real = rng.negative_binomial(2.0, 2.0 / (2.0 + 15.0), size=300)
lam_hat = y_real.mean()

def ppc(y, simulate, stat, n_sim=4000):
    """stats.md eq. (33.6): compare an observed statistic to replicates."""
    obs = stat(y)
    reps = np.array([stat(simulate(len(y))) for _ in range(n_sim)])
    p = np.mean(reps >= obs)
    return obs, reps, p


stats_to_check = {
    "variance": np.var,
    "proportion of zeros": lambda v: np.mean(v == 0),
    "maximum": np.max,
}
rng_p = np.random.default_rng(11)
print(f"  Fitting a POISSON model to negative-binomial data (lambda_hat = "
      f"{lam_hat:.2f})")
print(f"  {'statistic':<24}{'observed':>11}{'mean replicate':>17}"
      f"{'ppc p-value':>13}")
for name, f in stats_to_check.items():
    obs, reps, p = ppc(y_real, lambda k: rng_p.poisson(lam_hat, k), f)
    print(f"  {name:<24}{obs:>11.3f}{reps.mean():>17.3f}{p:>13.4f}")

print("\n  Now the CORRECT negative-binomial model:")
phi_hat = max((y_real.var(ddof=1) - lam_hat) / lam_hat**2, 1e-6)
r_hat = 1 / phi_hat
for name, f in stats_to_check.items():
    obs, reps, p = ppc(y_real,
                       lambda k: rng_p.negative_binomial(
                           r_hat, r_hat / (r_hat + lam_hat), k), f)
    print(f"  {name:<24}{obs:>11.3f}{reps.mean():>17.3f}{p:>13.4f}")

print("\n  ppc p-values near 0 or 1 mean the model CANNOT reproduce that")
print("  feature of the data. This is the Bayesian analogue of residual")
print("  diagnostics and is far more informative than any single fit statistic.")

# %% [markdown]
# ## 5. A Metropolis sampler with convergence diagnostics

# %%
header("5. MCMC and the diagnostics you must report")

def metropolis(log_post, init, n_iter=8000, step=0.4, seed=0):
    """Random-walk Metropolis. Returns the chain and the acceptance rate."""
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(init, float))
    chain = np.empty((n_iter, len(x)))
    lp = log_post(x)
    acc = 0
    for i in range(n_iter):
        prop = x + rng.normal(0, step, len(x))
        lp_prop = log_post(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            acc += 1
        chain[i] = x
    return chain, acc / n_iter


# Logistic regression with a weakly informative N(0, 2.5^2) prior.
rng = np.random.default_rng(2304)
n = 200
xv = rng.normal(0, 1, n)
yv = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.2 * xv))))

def log_posterior(beta):
    eta = beta[0] + beta[1] * xv
    ll = np.sum(yv * eta - np.log1p(np.exp(eta)))
    lprior = np.sum(st.norm.logpdf(beta, 0, 2.5))     # regularising prior
    return ll + lprior


chains = [metropolis(log_posterior, [0, 0], seed=s)[0] for s in range(4)]
_, acc = metropolis(log_posterior, [0, 0], seed=0)
burn = 2000
post = np.concatenate([c[burn:] for c in chains])

def r_hat(chains, burn=2000):
    """Split-Rhat: between- vs within-chain variance. Want < 1.01."""
    x = np.array([c[burn:] for c in chains])              # (m, n, p)
    m, nn, p = x.shape
    W = x.var(axis=1, ddof=1).mean(axis=0)
    B = nn * x.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (nn - 1) / nn * W + B / nn
    return np.sqrt(var_hat / W)


def ess(x):
    """Effective sample size from the autocorrelation sum. Want > 400."""
    x = x - x.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n-1:] / (np.arange(n, 0, -1) * x.var())
    s = 0.0
    for k in range(1, min(n, 500)):
        if ac[k] < 0.05:
            break
        s += ac[k]
    return n / (1 + 2 * s)


print(f"  acceptance rate = {acc:.3f}   (0.2-0.4 is healthy for random walk)")
print(f"  {'parameter':<14}{'post. mean':>12}{'95% credible':>24}"
      f"{'R-hat':>8}{'ESS':>9}")
rh = r_hat(chains, burn)
for j, nm in enumerate(["intercept", "slope"]):
    lo, hi = np.percentile(post[:, j], [2.5, 97.5])
    print(f"  {nm:<14}{post[:, j].mean():>12.4f}"
          f"   [{lo:>7.4f}, {hi:>7.4f}]{rh[j]:>8.4f}"
          f"{ess(chains[0][burn:, j]):>9.0f}")
print(f"  truth: intercept -0.400, slope 1.200")
print("\n  NEVER report a posterior without R-hat < 1.01 and ESS > 400.")

# Compare with the frequentist MLE - they should agree with a weak prior.
import statsmodels.api as sm
mle = sm.GLM(yv, sm.add_constant(xv), family=sm.families.Binomial()).fit()
print(f"\n  frequentist MLE: intercept {mle.params[0]:+.4f}, "
      f"slope {mle.params[1]:+.4f}")
print("  With a weakly informative prior and n=200, Bayesian and frequentist")
print("  answers essentially coincide. The prior matters at SMALL n - which")
print("  is precisely when a regularising prior is most valuable.")

# %% [markdown]
# ## 6. Bayes factors are prior-sensitive, eq. (33.7)

# %%
header("6. Lindley's paradox (33.7)")

# A z-test of H0: mu = 0 with a N(0, tau^2) prior under H1.
z_obs, n_bf = 2.5, 100
sigma = 1.0
se = sigma / np.sqrt(n_bf)
y_bar = z_obs * se
print(f"  data: y_bar = {y_bar:.4f}, SE = {se:.4f}, z = {z_obs}, "
      f"p = {2*st.norm.sf(z_obs):.4f}")
print(f"\n  {'prior SD tau under H1':>24}{'BF_10':>10}  interpretation")
for tau in [0.05, 0.1, 0.5, 1.0, 5.0, 50.0]:
    # Marginal likelihood under H1 is N(0, tau^2 + se^2); under H0 is N(0, se^2)
    bf = (st.norm.pdf(y_bar, 0, np.sqrt(tau**2 + se**2))
          / st.norm.pdf(y_bar, 0, se))
    note = "favours H1" if bf > 3 else ("favours H0" if bf < 1/3 else "ambiguous")
    print(f"  {tau:>24.2f}{bf:>10.3f}  {note}")
print("\n  The SAME data give BF from strongly favouring H1 to favouring H0,")
print("  purely by changing the prior width. The p-value did not move.")
print("  If you report a Bayes factor, report its sensitivity across prior")
print("  scales - the marginal likelihood depends on the prior even where the")
print("  posterior does not.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
gx = np.linspace(0, 1, 400)
ax.plot(gx, st.beta.pdf(gx, a0, b0), label=f"prior Beta({a0:.0f},{b0:.0f})")
ax.plot(gx, st.beta.pdf(gx, a1, b1), lw=2, label=f"posterior Beta({a1:.0f},{b1:.0f})")
ax.axvline(y_obs / n_obs, color="k", ls="--", label="data proportion")
ax.set_title("Eq. (33.2): conjugate updating", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.scatter(np.sqrt(mod["s2"]), np.sqrt(mod["s2_tilde"]), s=2, alpha=0.2)
lim = [0, np.percentile(np.sqrt(mod["s2"]), 99.5)]
ax.plot(lim, lim, "k--", lw=1, label="no shrinkage")
ax.axhline(np.sqrt(mod["s0_2"]), color="red", ls=":", label="prior $s_0$")
ax.set_xlabel("ordinary $s_g$"); ax.set_ylabel("moderated $\\tilde s_g$")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title("Eq. (20.6): variance shrinkage", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
for j, c in enumerate(chains):
    ax.plot(c[:, 1], lw=0.5, alpha=0.7, label=f"chain {j}" if j < 2 else None)
ax.axvline(burn, color="red", ls="--", label="burn-in")
ax.set_xlabel("iteration"); ax.set_ylabel("slope")
ax.set_title(f"MCMC trace (R-hat = {rh[1]:.4f})", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
obs, reps, _ = ppc(y_real, lambda k: rng_p.poisson(lam_hat, k), np.var)
ax.hist(reps, bins=50, alpha=0.7, label="Poisson replicates")
ax.axvline(obs, color="red", lw=2, label="observed variance")
ax.set_title("Eq. (33.6): posterior predictive check", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "bayes.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/bayes.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 33)
#
# 1. Use empirical Bayes shrinkage whenever you have many parallel features and
#    few replicates: the single biggest practical win in small-$n$ omics.
# 2. Report credible intervals **as** probability statements, and say so.
# 3. Make priors explicit, weakly informative, and checked.
# 4. Check convergence (R-hat, ESS) before interpreting anything.
#
# **Next:** `34_diagnostics_and_reproducibility.py`
