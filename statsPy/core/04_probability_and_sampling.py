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
# # Module 04: Probability models and sampling distributions
#
# **Curriculum link:** `stats.md` -> Topic 4, equations (4.1)-(4.12)
#
# ## What you will learn
#
# 1. Variance algebra (4.1)-(4.2), and why **pairing** is so powerful (4.3).
# 2. The law of total variance (4.4): the formal parent of Module 01.
# 3. The CLT (4.5) and its convergence rate (4.6): it is about the *mean*, not
#    the data.
# 4. Poisson (4.7) vs negative binomial (4.8)-(4.9), derived as a gamma-Poisson
#    mixture and verified numerically.
# 5. Likelihood, score, Fisher information (4.10)-(4.11): where every standard
#    error printed by `glm`/`lm`/`coxph` actually comes from.
# 6. Permutation p-values (4.12), and why the `+1` is mandatory.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "04_probability_and_sampling"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. Variance algebra and the power of pairing, eq. (4.1)-(4.3)
#
# $$\operatorname{Var}(aX+bY)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y)+2ab\operatorname{Cov}(X,Y)$$
#
# $$\operatorname{Var}(D)=2\sigma^2(1-r) \quad\text{for } D=Y_{\text{post}}-Y_{\text{pre}}$$
#
# With $r=0.8$, pairing cuts the variance of the estimated effect **fivefold**
# compared with an unpaired comparison using the same measurements.

# %%
header("1. Why pairing works (4.2)-(4.3)")

rng = np.random.default_rng(401)
n_pairs, sigma = 12, 1.0
TRUE_EFFECT = 0.4

print(f"{'r':>6} {'Var(D) theory':>15} {'Var(D) sim':>12} "
      f"{'paired power':>13} {'unpaired power':>15}")
for r in [0.0, 0.3, 0.6, 0.8, 0.95]:
    theory = 2 * sigma**2 * (1 - r)              # eq. (4.3)

    diffs, p_paired, p_unpaired = [], [], []
    for _ in range(3000):
        # Bivariate normal pre/post with correlation r
        subject = rng.normal(0, np.sqrt(r) * sigma, n_pairs)
        pre = subject + rng.normal(0, np.sqrt(1 - r) * sigma, n_pairs)
        post = subject + rng.normal(0, np.sqrt(1 - r) * sigma, n_pairs) + TRUE_EFFECT
        d = post - pre
        diffs.append(d.var(ddof=1))
        p_paired.append(st.ttest_rel(post, pre).pvalue)
        # The WRONG analysis: ignore the pairing the design provided.
        p_unpaired.append(st.ttest_ind(post, pre, equal_var=False).pvalue)

    print(f"{r:>6.2f} {theory:>15.4f} {np.mean(diffs):>12.4f} "
          f"{np.mean(np.array(p_paired) < 0.05):>12.1%} "
          f"{np.mean(np.array(p_unpaired) < 0.05):>14.1%}")

print("\nSame data, same n. Analysing a paired design as unpaired throws away")
print("the precision you paid for. Pairing comes from the DESIGN (Topic 7).")

# %% [markdown]
# ## 2. Law of total variance, eq. (4.4)
#
# $$\operatorname{Var}(Y)=\mathbb{E}\big[\operatorname{Var}(Y\mid Z)\big]
#   +\operatorname{Var}\big(\mathbb{E}[Y\mid Z]\big)$$
#
# This is the formal statement behind Module 01's eq. (1.4)-(1.5) and behind
# every variance component in a mixed model (Topic 14).

# %%
header("2. Law of total variance, verified (4.4)")

rng = np.random.default_rng(402)
n_donors, m_cells = 200, 60
SIGMA_B, SIGMA_E = 1.3, 0.8

b = rng.normal(0, SIGMA_B, n_donors)
y = b[:, None] + rng.normal(0, SIGMA_E, (n_donors, m_cells))

total_var = y.ravel().var(ddof=1)
within = y.var(axis=1, ddof=1).mean()            # E[Var(Y|Z)]
between = y.mean(axis=1).var(ddof=1)             # Var(E[Y|Z])

print(f"  Var(Y)                     = {total_var:.4f}")
print(f"  E[Var(Y|donor)]  (within)  = {within:.4f}   (truth sigma_e^2 = {SIGMA_E**2:.4f})")
print(f"  Var(E[Y|donor])  (between) = {between:.4f}   (truth sigma_b^2 = {SIGMA_B**2:.4f})")
print(f"  within + between           = {within + between:.4f}  <- matches Var(Y)")
print(f"\n  ICC (eq. 1.7) = sigma_b^2/(sigma_b^2+sigma_e^2) = "
      f"{SIGMA_B**2/(SIGMA_B**2+SIGMA_E**2):.4f}")

# %% [markdown]
# ## 3. The central limit theorem, eq. (4.5)-(4.6)
#
# $$\sqrt n\,\frac{\bar Y_n-\mu}{\sigma}\xrightarrow{d}\mathcal N(0,1)$$
#
# **The CLT is a statement about the sampling distribution of the mean, not
# about the data.** Berry-Esseen (4.6) says the error is $O(n^{-1/2})$ and gets
# worse with skewness: which is exactly the situation for low-count genes,
# cytokine concentrations and rare taxa.

# %%
header("3. CLT convergence depends on skewness (4.5)-(4.6)")

def ci_coverage(sampler, n, n_sim=6000, alpha=0.05, seed=0):
    """Empirical coverage of the nominal 95% t-interval (eq. 5.3)."""
    rng = np.random.default_rng(seed)
    mu_true = sampler(rng, 400_000).mean()       # high-precision reference
    tcrit = st.t.ppf(1 - alpha / 2, n - 1)
    hits = 0
    for _ in range(n_sim):
        x = sampler(rng, n)
        half = tcrit * x.std(ddof=1) / np.sqrt(n)
        hits += abs(x.mean() - mu_true) <= half
    return hits / n_sim


samplers = {
    "Normal (skew 0)":        lambda r, k: r.normal(0, 1, k),
    "Exponential (skew 2)":   lambda r, k: r.exponential(1, k),
    "Log-normal (skew ~6)":   lambda r, k: r.lognormal(0, 1, k),
    "Poisson(0.5) (skew 1.4)": lambda r, k: r.poisson(0.5, k).astype(float),
}

print(f"{'distribution':<26}" + "".join(f"{f'n={n}':>9}" for n in [5, 10, 30, 100]))
for name, f in samplers.items():
    row = "".join(f"{ci_coverage(f, n, 2500, seed=hash(name) % 1000 + n):>8.1%} "
                  for n in [5, 10, 30, 100])
    print(f"{name:<26}{row}")
print("\n(nominal coverage is 95.0%)")
print("\nSymmetric data: the interval is fine even at n=5. Heavily skewed data:")
print("coverage is still short of nominal at n=30. This is Berry-Esseen (4.6):")
print("the approximation error scales with E|Y-mu|^3 / sigma^3.")

# %% [markdown]
# ## 4. Poisson vs negative binomial, eq. (4.7)-(4.9)
#
# The NB arises as a **gamma-Poisson mixture**: if $Y\mid\Lambda\sim\text{Poisson}(\Lambda)$
# and $\Lambda\sim\text{Gamma}(1/\phi,\ \mu\phi)$, then marginally
# $\mathbb{E}[Y]=\mu$ and $\operatorname{Var}(Y)=\mu+\phi\mu^2$.
#
# The one-line derivation uses (4.4):
# $\operatorname{Var}(Y)=\mathbb{E}[\Lambda]+\operatorname{Var}(\Lambda)=\mu+\phi\mu^2$.

# %%
header("4. NB as a gamma-Poisson mixture (4.8)-(4.9)")

rng = np.random.default_rng(404)
MU, PHI, N = 40.0, 0.16, 400_000

# Build the NB by explicitly compounding, exactly as the derivation says.
lam = rng.gamma(shape=1.0 / PHI, scale=MU * PHI, size=N)   # Lambda ~ Gamma
y_mix = rng.poisson(lam)                                    # Y | Lambda ~ Poisson

# Draw the same thing from numpy's NB parameterisation as a cross-check.
size = 1.0 / PHI
y_nb = rng.negative_binomial(size, size / (size + MU), N)

print(f"  target mean     = {MU}")
print(f"  target variance = mu + phi*mu^2 = {MU + PHI*MU**2:.2f}   (eq. 4.9)")
print(f"\n  gamma-Poisson compound : mean {y_mix.mean():7.3f}  var {y_mix.var(ddof=1):9.3f}")
print(f"  numpy negative_binomial: mean {y_nb.mean():7.3f}  var {y_nb.var(ddof=1):9.3f}")
print(f"  pure Poisson(mu)       : mean {MU:7.3f}  var {MU:9.3f}   <- far too small")

# Verify the pmf formula of eq. (4.8) against scipy.
k = np.arange(0, 12)
from scipy.special import gammaln
log_pmf_manual = (gammaln(k + 1/PHI) - gammaln(1/PHI) - gammaln(k + 1)
                  + (1/PHI) * np.log(1 / (1 + MU*PHI))
                  + k * np.log(MU*PHI / (1 + MU*PHI)))
pmf_scipy = st.nbinom.pmf(k, size, size / (size + MU))
print(f"\n  max |eq.(4.8) pmf - scipy nbinom pmf| = "
      f"{np.max(np.abs(np.exp(log_pmf_manual) - pmf_scipy)):.3e}")

# The practical consequence: a Poisson test on overdispersed data.
print("\nConsequence - testing overdispersed counts with a Poisson model:")
rng = np.random.default_rng(405)
false_pos = 0
for _ in range(2000):
    a = rng.negative_binomial(size, size / (size + MU), 5)
    b = rng.negative_binomial(size, size / (size + MU), 5)   # same distribution!
    # Poisson assumption: Var = mean, so SE of the difference of totals:
    tot_a, tot_b = a.sum(), b.sum()
    se_poisson = np.sqrt(tot_a + tot_b)
    z = (tot_a - tot_b) / se_poisson
    false_pos += abs(z) > 1.96
print(f"  Poisson-based false-positive rate on NB data: {false_pos/2000:.1%}"
      f"   (should be 5%)")
print("  This is why RNA-seq needs NB, not Poisson (stats.md Topic 13, 20).")

# %% [markdown]
# ## 5. Likelihood, score, and Fisher information, eq. (4.10)-(4.11)
#
# $$U(\theta)=\frac{\partial\ell}{\partial\theta}, \qquad
#   \mathcal I(\theta)=-\mathbb E\!\left[\frac{\partial^2\ell}{\partial\theta^2}\right],
#   \qquad \widehat{\operatorname{Var}}(\hat\theta)\approx\mathcal I(\hat\theta)^{-1}$$
#
# Every Wald standard error you have ever read off a model summary is (4.11).
# Here we compute one from scratch and compare it to `scipy`'s answer.

# %%
header("5. Fisher information gives the standard error (4.10)-(4.11)")

rng = np.random.default_rng(406)
LAMBDA_TRUE, n = 4.0, 200
y = rng.poisson(LAMBDA_TRUE, n)

# For Poisson: l(lambda) = sum(y)*log(lambda) - n*lambda - sum(log(y!))
#   score      U(lambda)  = sum(y)/lambda - n            -> MLE = ybar
#   observed I I(lambda)  = sum(y)/lambda^2
#   expected I            = n/lambda
lam_hat = y.mean()
observed_info = y.sum() / lam_hat**2
expected_info = n / lam_hat

se_observed = 1 / np.sqrt(observed_info)
se_expected = 1 / np.sqrt(expected_info)
se_textbook = np.sqrt(lam_hat / n)               # the familiar formula

print(f"  MLE lambda_hat              = {lam_hat:.4f}  (truth {LAMBDA_TRUE})")
print(f"  SE from observed info (4.11)= {se_observed:.5f}")
print(f"  SE from expected info       = {se_expected:.5f}")
print(f"  SE = sqrt(lambda_hat/n)     = {se_textbook:.5f}   <- all identical")

# And confirm numerically by maximising the log-likelihood.
def neg_loglik(params):
    lam = np.exp(params[0])                      # log-parameterise for stability
    return -np.sum(st.poisson.logpmf(y, lam))

opt = minimize(neg_loglik, x0=[np.log(1.0)], method="BFGS")
print(f"\n  numerical MLE (BFGS)        = {np.exp(opt.x[0]):.4f}")
print(f"  Cramer-Rao bound: no unbiased estimator beats SE = {se_expected:.5f}")

# %% [markdown]
# ## 6. Permutation tests and the mandatory `+1`, eq. (4.12)
#
# $$\hat p=\frac{1+\#\{T(\pi_b(\mathbf y))\ge T_{\text{obs}}\}}{B+1}$$
#
# The `+1` in numerator and denominator includes the observed labelling in the
# reference set. Without it, $\hat p$ can be exactly 0: which is never a valid
# p-value: and the test is anticonservative.

# %%
header("6. Permutation p-values: the +1 is not a fudge (4.12)")

def permutation_test(x, y, B=999, seed=0, add_one=True):
    """Two-sided permutation test for a difference in means (eq. 4.12)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    t_obs = abs(x.mean() - y.mean())
    count = 0
    for _ in range(B):
        perm = rng.permutation(pooled)
        count += abs(perm[:n_x].mean() - perm[n_x:].mean()) >= t_obs
    if add_one:
        return (1 + count) / (B + 1)             # valid
    return count / B                             # anticonservative


# Calibration check: under a TRUE null a valid p-value rejects at exactly alpha.
rng = np.random.default_rng(407)
p_correct, p_wrong = [], []
for _ in range(1200):
    a = rng.normal(0, 1, 6)
    b = rng.normal(0, 1, 6)
    s = int(rng.integers(0, 10**6))
    p_correct.append(permutation_test(a, b, B=199, seed=s, add_one=True))
    p_wrong.append(permutation_test(a, b, B=199, seed=s, add_one=False))

p_correct, p_wrong = np.array(p_correct), np.array(p_wrong)
print(f"  smallest p WITH the +1    : {p_correct.min():.4f}  = 1/(B+1)")
print(f"  smallest p WITHOUT the +1 : {p_wrong.min():.4f}  <- 0 is not a p-value")
print("\n  A valid p-value satisfies P(p <= alpha) <= alpha for EVERY alpha.")
print(f"  {'alpha':>8} {'with +1':>12} {'without +1':>12}")
for alpha in [0.05, 0.01, 0.005, 0.001]:
    print(f"  {alpha:>8.3f} {np.mean(p_correct <= alpha):>12.4f} "
          f"{np.mean(p_wrong <= alpha):>12.4f}")
print("\n  At alpha >= 1/(B+1) the two agree. Below it the uncorrected version")
print("  keeps rejecting (whenever the count is 0) while the true attainable")
print("  resolution has run out - so it is anticonservative exactly in the")
print("  extreme tail, which is where genome-wide thresholds live (Topic 8).")
print(f"\n  Minimum attainable p with B=199 permutations: 1/(B+1) = {1/200:.4f}")
print("  So B >= 999 if you need to reach p = 0.001.")

# Exactness: with n small enough, enumerate ALL permutations.
from itertools import combinations
a = np.array([5.1, 4.8, 6.2, 5.5])
b = np.array([6.9, 7.3, 6.5, 7.8])
pooled = np.concatenate([a, b])
t_obs = abs(a.mean() - b.mean())
all_stats = []
for idx in combinations(range(8), 4):
    mask = np.zeros(8, bool)
    mask[list(idx)] = True
    all_stats.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
exact_p = np.mean(np.array(all_stats) >= t_obs)
print(f"\n  EXACT permutation p (all C(8,4)={len(all_stats)} splits) = {exact_p:.4f}")
print(f"  Welch t-test p for comparison                    = "
      f"{st.ttest_ind(a, b, equal_var=False).pvalue:.4f}")
print("  With 4 vs 4 the smallest achievable p is 2/70 = 0.0286 - a hard floor")
print("  imposed by the design, not by the effect size.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) CLT in action for a skewed variable
rng = np.random.default_rng(408)
for n, c in [(1, "C0"), (5, "C1"), (30, "C2")]:
    means = rng.lognormal(0, 1, (20000, n)).mean(axis=1)
    z = (means - means.mean()) / means.std(ddof=1)
    axes[0].hist(z, bins=80, range=(-4, 4), density=True, histtype="step",
                 color=c, label=f"n={n}")
grid = np.linspace(-4, 4, 200)
axes[0].plot(grid, st.norm.pdf(grid), "k--", lw=1, label="N(0,1)")
axes[0].set_title("Eq. (4.5): CLT for log-normal data")
axes[0].legend(fontsize=7)

# (b) Poisson vs NB
k = np.arange(0, 120)
axes[1].plot(k, st.poisson.pmf(k, 40), "o-", ms=2, label="Poisson(40)")
axes[1].plot(k, st.nbinom.pmf(k, 1/0.16, (1/0.16)/((1/0.16)+40)), "s-", ms=2,
             label="NB(mu=40, phi=0.16)")
axes[1].set_xlabel("count")
axes[1].set_ylabel("probability")
axes[1].set_title("Eq. (4.7)-(4.9): overdispersion")
axes[1].legend(fontsize=7)

# (c) permutation null distribution
rng = np.random.default_rng(409)
x0 = rng.normal(0, 1, 10)
y0 = rng.normal(1.2, 1, 10)
pooled = np.concatenate([x0, y0])
null_stats = []
for _ in range(5000):
    perm = rng.permutation(pooled)
    null_stats.append(perm[:10].mean() - perm[10:].mean())
axes[2].hist(null_stats, bins=60, color="lightsteelblue")
axes[2].axvline(x0.mean() - y0.mean(), color="red", lw=2, label="observed")
axes[2].set_title("Eq. (4.12): permutation null")
axes[2].set_xlabel("difference in means under random labelling")
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "probability.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/probability.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 4)
#
# 1. Name the distribution and the source of randomness before computing a
#    standard error.
# 2. Check overdispersion: if $s^2\gg\bar y$ for counts, Poisson inference is
#    anticonservative: often dramatically (section 4).
# 3. Prefer permutation when the design justifies exchangeability and $n$ is too
#    small for the CLT; prefer likelihood when a parametric mean-variance model
#    is credible and $G$ is large enough to borrow strength (Topic 33).
#
# ## Self-check
#
# * Pre/post measurements correlate at $r=0.9$. How much does pairing reduce
#   $\operatorname{Var}(\hat\delta)$? *(From $2\sigma^2$ to $0.2\sigma^2$: 10x.)*
# * You run 999 permutations and none exceeds the observed statistic. What do
#   you report? *(p = 1/1000 = 0.001, not p = 0.)*
#
# **Next:** `05_estimation_and_intervals.py`
