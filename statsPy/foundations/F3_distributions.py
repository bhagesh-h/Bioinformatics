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
# # Foundations 3: Distributions
#
# **Curriculum link:** `stats.md` -> Part 0, §0.5, equation (0.10)
# **Assumes:** `F1`, `F2`.
#
# ## What a distribution is
#
# A **distribution** answers: *which values does this variable take, and how
# often?* The mean and standard deviation are two summaries of it. The
# distribution is the whole picture.
#
# You do not need to memorise any formulas here. You need to recognise **which
# situation produces which shape**, because that is what decides which method
# is valid. Every one of the distributions below corresponds to a *mechanism*,
# and naming the mechanism usually names the method.

# %%
import os

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "F3_distributions"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


def sparkline(values, bins=28, width=56, height=7):
    """Draw a small text histogram, so you can see shapes without a viewer."""
    counts, edges = np.histogram(values, bins=bins)
    counts = counts / counts.max()
    rows = []
    for level in range(height, 0, -1):
        row = "".join("#" if c * height >= level - 0.5 else " " for c in counts)
        rows.append("    |" + row)
    rows.append("    +" + "-" * bins)
    rows.append(f"     {edges[0]:<{bins//2}.1f}{edges[-1]:>{bins - bins//2}.1f}")
    return "\n".join(rows)


rng = np.random.default_rng(2)

# %% [markdown]
# ## 1. The normal distribution: what averages look like
#
# Symmetric, centred at $\mu$, with spread $\sigma$. Written
# $X \sim N(\mu, \sigma^2)$, where "$\sim$" means "is distributed as".

# %%
header("1. Normal: the bell curve")

MU, SIGMA = 100, 15
x = rng.normal(MU, SIGMA, 200_000)
print(f"  X ~ Normal(mu = {MU}, sigma = {SIGMA})")
print(sparkline(x))
print(f"\n  {'within':<12}{'theory':>10}{'observed':>12}")
for k, theory in ((1, 0.6827), (2, 0.9545), (3, 0.9973)):
    print(f"  {f'{k} sigma':<12}{theory:>10.4f}"
          f"{np.mean(np.abs(x - MU) < k * SIGMA):>12.4f}")

print("""
  The 68 / 95 / 99.7 rule. Worth memorising - it turns any normal-ish number
  into an instant sanity check.

  The normal distribution is everywhere in statistics NOT because biological
  measurements are naturally bell-shaped (mostly they are not) but because
  AVERAGES are, whatever the raw data looks like. That is the central limit
  theorem, and it is the subject of F4.""")

# %% [markdown]
# ## 2. Binomial: counting successes out of $n$ tries
#
# The number of "yes" results in $n$ independent yes/no trials, each with
# probability $p$. How many of 50 patients respond; how many of 200 cells are
# positive.

# %%
header("2. Binomial: how many out of n?")

n_trials, p_success = 50, 0.30
b = rng.binomial(n_trials, p_success, 200_000)
print(f"  X ~ Binomial(n = {n_trials}, p = {p_success})")
print(sparkline(b))
print(f"\n  {'quantity':<28}{'formula':<16}{'theory':>10}{'observed':>12}")
print(f"  {'mean':<28}{'n*p':<16}{n_trials*p_success:>10.3f}{b.mean():>12.3f}")
print(f"  {'variance':<28}{'n*p*(1-p)':<16}"
      f"{n_trials*p_success*(1-p_success):>10.3f}{b.var():>12.3f}")

print("""
  Note the variance formula: it is LARGEST at p = 0.5 and shrinks towards 0 as
  p approaches 0 or 1. A proportion near 0 or 1 is intrinsically less variable
  than one near a half - which is why the uncertainty on a proportion is not
  the same at every level, and why proportions get their own model (Topic 13)
  rather than being fed to a t-test.""")

for p_ in (0.02, 0.5, 0.98):
    v = n_trials * p_ * (1 - p_)
    print(f"    p = {p_:<5}  variance = {v:>6.2f}")

# %% [markdown]
# ## 3. Poisson: counting events in a window, eq. (0.10)
#
# Events happening independently at a constant rate: reads mapping to a gene,
# mutations per genome, colonies on a plate. Its defining property is very
# restrictive.

# %%
header("3. Poisson: mean equals variance (0.10)")

for lam in (2, 10, 100):
    po = rng.poisson(lam, 200_000)
    print(f"  lambda = {lam:<5} mean = {po.mean():>8.3f}   variance = {po.var():>8.3f}"
          f"   ratio = {po.var()/po.mean():.3f}")

print(sparkline(rng.poisson(10, 200_000)))
print("""
  mean = variance is the Poisson's signature, and its weakness. It leaves NO
  freedom: tell me the mean and I have told you the spread.

  Notice also that as lambda grows the shape becomes symmetric and bell-like.
  A Poisson with a large mean is well approximated by a normal - which is why
  high-count genes can be treated more casually than low-count ones.""")

# %% [markdown]
# ## 4. Negative binomial: counts in the real world
#
# Real biological counts are almost always **more** variable than Poisson
# allows, because the units genuinely differ: mice, patients and cultures are
# not identical. That extra spread is called **overdispersion**.

# %%
header("4. Negative binomial: Poisson plus biological variability")

MEAN = 100
print(f"  All of these have the same mean ({MEAN}). Only the spread differs.\n")
print(f"  {'model':<40}{'mean':>9}{'variance':>11}{'var/mean':>10}")

pois = rng.poisson(MEAN, 200_000)
print(f"  {'Poisson (technical noise only)':<40}{pois.mean():>9.1f}"
      f"{pois.var():>11.1f}{pois.var()/pois.mean():>10.2f}")

for disp in (0.05, 0.2, 0.5):
    # A negative binomial is a Poisson whose RATE varies between samples.
    # Simulate it exactly that way, so the mechanism is visible:
    rate = rng.gamma(shape=1 / disp, scale=MEAN * disp, size=200_000)
    nb = rng.poisson(rate)
    print(f"  {f'Neg. binomial, dispersion = {disp}':<40}{nb.mean():>9.1f}"
          f"{nb.var():>11.1f}{nb.var()/nb.mean():>10.2f}")

print(f"""
  Var = mu + dispersion * mu^2. The first term is counting noise; the second
  is biological variability between samples, and it dominates at high counts.

  This is THE reason RNA-seq is analysed with DESeq2 or edgeR rather than a
  t-test or a Poisson model. Use Poisson on data that is really negative
  binomial and you understate the variance badly - at mean 100 with
  dispersion 0.2, by a factor of {(MEAN + 0.2*MEAN**2)/MEAN:.0f}. Every p-value inherits that error.

  Topic 20 is about estimating that dispersion when you have 3 samples, which
  is the central technical difficulty of differential expression.""")

# %% [markdown]
# ## 5. Uniform: and the most useful diagnostic in this course
#
# Every value equally likely. Dull in itself, but it underpins a check you
# should run constantly.

# %%
header("5. Uniform: p-values under the null are flat")

# Run a t-test 20,000 times on data where NOTHING is going on.
null_p = np.array([st.ttest_ind(rng.normal(0, 1, 10), rng.normal(0, 1, 10)).pvalue
                   for _ in range(20_000)])
print("  20,000 t-tests on data with NO real difference:")
print(sparkline(null_p))
print(f"\n  {'p < 0.05':<18}{np.mean(null_p < 0.05):>8.4f}   (should be 0.05)")
print(f"  {'p < 0.10':<18}{np.mean(null_p < 0.10):>8.4f}   (should be 0.10)")
print(f"  {'p < 0.50':<18}{np.mean(null_p < 0.50):>8.4f}   (should be 0.50)")

# Now the same thing when there IS an effect.
real_p = np.array([st.ttest_ind(rng.normal(0.9, 1, 10), rng.normal(0, 1, 10)).pvalue
                   for _ in range(20_000)])
print("\n  The same test when there IS a real difference:")
print(sparkline(real_p))

print("""
  FLAT means the test is behaving. SPIKED AT ZERO means there is signal.

  That gives you a diagnostic you can use on any analysis: plot a histogram of
  all your p-values. If you expect mostly nulls and it is NOT flat, something
  is wrong - unmodelled batch, non-independence, a broken assumption. If it
  slopes upward towards 1, your test is conservative. A p-value histogram is
  the cheapest and most informative plot in genomics (Topic 8).""")

# %% [markdown]
# ## 6. Distributions of *statistics*, not measurements
#
# The $t$, chi-squared and $F$ distributions do not describe things you
# measure. They describe numbers you **compute** from a sample. You meet them
# when testing, never when collecting data.

# %%
header("6. t, chi-squared and F describe test statistics")

print("  The t distribution is the normal's heavier-tailed cousin. It is what")
print("  you get when you must ESTIMATE sigma instead of knowing it.\n")
print(f"  {'df':<8}{'P(|t| > 2)':>14}{'P(|t| > 2) if normal':>24}")
for df in (2, 5, 10, 30, 1000):
    print(f"  {df:<8}{2*st.t.sf(2, df):>14.4f}{2*st.norm.sf(2):>24.4f}")

print("""
  With few degrees of freedom the tails are FATTER - extreme values are more
  likely than a normal would suggest. That is exactly right: when you have
  little data, your estimate of sigma is itself unreliable, and the t
  distribution builds in that extra doubt.

  As df grows, t converges on the normal. By df = 30 the difference barely
  matters, which is the origin of the folklore that "n = 30 is enough".

  The practical consequence: with a small sample, the critical value for a 95%
  interval is not 1.96 but something larger, so your interval is WIDER. Small
  studies are automatically made more humble - as long as you use the right
  distribution.""")

# %% [markdown]
# ## 7. Skew, and why biology uses logs

# %%
header("7. Skewed data, and what the log transform does")

expr = rng.lognormal(mean=np.log(50), sigma=1.1, size=200_000)
print("  A typical gene expression variable (right-skewed):")
print(sparkline(expr))
print(f"\n  mean   = {expr.mean():>8.1f}")
print(f"  median = {np.median(expr):>8.1f}   <- less than half the mean")
print(f"  {np.mean(expr < expr.mean()):.1%} of values are BELOW the mean")

print("\n  The same data after taking logs:")
print(sparkline(np.log2(expr)))
print(f"\n  mean   = {np.log2(expr).mean():>8.2f}")
print(f"  median = {np.median(np.log2(expr)):>8.2f}   <- now they agree")
print(f"  {np.mean(np.log2(expr) < np.log2(expr).mean()):.1%} of values are below the mean")

print("""
  For the raw variable, "the mean" is not a typical value - a minority of
  large values drags it above most of the data. Reporting mean +/- SD here
  would be actively misleading.

  The log fixes it, and not by accident. Biological quantities tend to vary
  MULTIPLICATIVELY - a treatment doubles expression, it does not add 40 units
  - and the log turns multiplication into addition. That converts skewed,
  proportional variation into symmetric, additive variation, which is exactly
  what the standard toolkit is built for.

  This is why expression is analysed as log2 fold change, why concentrations
  are logged, and why a "2-fold change" is a more natural unit than "+50".""")

# %% [markdown]
# ## 8. Figure

# %%
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))

ax[0, 0].hist(x, bins=70, color="steelblue", density=True)
for k in (1, 2):
    for sgn in (-1, 1):
        ax[0, 0].axvline(MU + sgn * k * SIGMA, color="firebrick", ls="--", lw=1)
ax[0, 0].set_title("Normal: 68 / 95 rule")

ax[0, 1].hist(b, bins=range(0, 35), color="seagreen", density=True)
ax[0, 1].set_title(f"Binomial(n={n_trials}, p={p_success})")

for lam, colour in ((2, "steelblue"), (10, "darkorange"), (40, "firebrick")):
    ax[0, 2].hist(rng.poisson(lam, 60_000), bins=60, alpha=0.55, density=True,
                  color=colour, label=f"$\\lambda$ = {lam}")
ax[0, 2].set_title("Poisson: mean = variance"); ax[0, 2].legend(fontsize=8)

grid = np.arange(0, 320)
ax[1, 0].plot(grid, st.poisson.pmf(grid, MEAN), lw=2, label="Poisson")
for disp, colour in ((0.05, "darkorange"), (0.3, "firebrick")):
    ax[1, 0].plot(grid, st.nbinom.pmf(grid, 1 / disp, 1 / (1 + disp * MEAN)),
                  lw=2, color=colour, label=f"NB, disp = {disp}")
ax[1, 0].set_title("Same mean, very different spread"); ax[1, 0].legend(fontsize=8)

ax[1, 1].hist(null_p, bins=40, color="grey", density=True, label="no effect")
ax[1, 1].hist(real_p, bins=40, color="firebrick", alpha=0.6, density=True,
              label="real effect")
ax[1, 1].axhline(1.0, color="black", ls="--", lw=1)
ax[1, 1].set_title("p-values: flat under the null")
ax[1, 1].set_xlabel("p-value"); ax[1, 1].legend(fontsize=8)

ax[1, 2].hist(np.log2(expr), bins=70, color="seagreen", density=True)
ax[1, 2].set_title("Skewed data, after $\\log_2$")
ax[1, 2].set_xlabel("$\\log_2$(expression)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "distributions.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'distributions.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: How wrong is Poisson on real count data?
#
# Simulate counts that are truly negative binomial, then test them with a
# method that assumes Poisson. Measure the false positive rate.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def poisson_test(a, b):
#     """Compare two groups of counts ASSUMING Poisson (variance = mean)."""
#     ma, mb = a.mean(), b.mean()
#     se = np.sqrt(ma / len(a) + mb / len(b))       # Poisson SE
#     return 2 * st.norm.sf(abs(ma - mb) / max(se, 1e-9))
#
# print(f"  {'true dispersion':<20}{'var/mean':>10}{'false positive rate':>22}")
# for disp in (0.0, 0.05, 0.2, 0.5):
#     hits, ratios = 0, []
#     for _ in range(3000):
#         if disp == 0:
#             a, b_ = rng.poisson(100, 6), rng.poisson(100, 6)
#         else:
#             a = rng.poisson(rng.gamma(1/disp, 100*disp, 6))
#             b_ = rng.poisson(rng.gamma(1/disp, 100*disp, 6))
#         ratios.append(np.var(np.concatenate([a, b_])) /
#                       np.mean(np.concatenate([a, b_])))
#         hits += poisson_test(a, b_) < 0.05
#     print(f"  {disp:<20.2f}{np.mean(ratios):>10.1f}{hits/3000:>21.1%}")
#
# # With genuinely Poisson data the test is correct - about 5%. Add even modest
# # overdispersion and the false positive rate climbs steeply, because the test
# # is dividing by a standard error that is far too small.
# #
# # THE GROUPS ARE IDENTICAL IN EVERY ROW. Every "discovery" is false. This is
# # what happens when you analyse RNA-seq with a model that assumes counting
# # noise is the only noise, and it is why DESeq2 and edgeR spend most of their
# # effort estimating the dispersion rather than testing.

# %% [markdown]
# ### Problem 2: When does the normal approximation start working?
#
# The central limit theorem says averages become normal. Find out how fast,
# starting from a very non-normal variable.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # A brutally skewed source: most values tiny, a few enormous.
# def draw(n, reps):
#     return rng.lognormal(0, 1.8, (reps, n)).mean(axis=1)
#
# print(f"  {'n averaged':<14}{'skewness':>12}{'P(within 2 SD)':>18}"
#       f"{'normal says':>14}")
# for n in (1, 2, 5, 15, 50, 200):
#     m = draw(n, 40_000)
#     z = (m - m.mean()) / m.std()
#     print(f"  {n:<14}{st.skew(m):>12.2f}{np.mean(np.abs(z) < 2):>18.3f}"
#           f"{0.9545:>14.3f}")
#
# # Skewness measures asymmetry; 0 is symmetric. It falls steadily towards 0 as
# # n grows, and the coverage climbs towards the normal's 95.45%.
# #
# # Two lessons. First, the CLT really works - even this source becomes
# # bell-shaped. Second, "n = 30 is enough" is FOLKLORE, not a theorem: for a
# # source this skewed, n = 50 is still visibly off. The nastier the source
# # distribution, the larger n must be. When in doubt, use a method that does
# # not rely on the approximation at all - a permutation test or a bootstrap
# # (Topic 7, Topic 5).

# %% [markdown]
# ### Problem 3: Reading a p-value histogram
#
# Below are four analyses. Plot the p-value histogram for each and say what it
# tells you. This is a skill worth more than any single test.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def tt(a, b):
#     return st.ttest_ind(a, b).pvalue
#
# cases = {}
# # (a) everything null, test correct
# cases["all null, correct test"] = [tt(rng.normal(0,1,8), rng.normal(0,1,8))
#                                    for _ in range(4000)]
# # (b) 10% of features have a real effect
# cases["10% real effects"] = [tt(rng.normal(1.5 if i % 10 == 0 else 0, 1, 8),
#                                 rng.normal(0, 1, 8)) for i in range(4000)]
# # (c) null, but observations are CLUSTERED (pseudoreplication)
# def clustered():
#     ga, gb = rng.normal(0, 1, 2), rng.normal(0, 1, 2)
#     return tt(np.repeat(ga, 4) + rng.normal(0, .3, 8),
#               np.repeat(gb, 4) + rng.normal(0, .3, 8))
# cases["null, but clustered data"] = [clustered() for _ in range(4000)]
# # (d) null, but a conservative (over-cautious) test
# cases["null, conservative test"] = [min(1.0, 2.2 * tt(rng.normal(0,1,8),
#                                     rng.normal(0,1,8))) for _ in range(4000)]
#
# print(f"  {'analysis':<28}{'p<0.05':>9}{'shape':>34}")
# for name, ps in cases.items():
#     ps = np.array(ps)
#     lo, hi = np.mean(ps < 0.2), np.mean(ps > 0.8)
#     shape = ("spike at 0 (signal, or a bug)" if lo > 2.5 * hi
#              else "sloping up (conservative)" if hi > 1.5 * lo else "flat (healthy)")
#     print(f"  {name:<28}{np.mean(ps < 0.05):>9.3f}{shape:>34}")
#
# # (a) FLAT, 5% below 0.05. The test is calibrated. This is the baseline you
# #     compare everything else against.
# # (b) FLAT with a SPIKE at zero. Healthy, with real signal. The height of the
# #     flat part estimates what fraction of your features are null - which is
# #     exactly how Storey's q-value works (Topic 8).
# # (c) A spike at zero WITH NO REAL EFFECTS ANYWHERE. Indistinguishable from
# #     (b) by eye, which is what makes it dangerous. If you know most features
# #     should be null and you see this, suspect non-independence or an
# #     unmodelled batch before you celebrate.
# # (d) SLOPING UP towards 1. The test is conservative - you are losing power
# #     but not making false claims. Common with rank tests on tiny samples and
# #     with over-corrected multiple testing.
# #
# # Plot this histogram for every genome-scale analysis you run. It costs
# # nothing and catches errors that no amount of staring at the top hits will.

# %% [markdown]
# ## What to take away
#
# 1. A distribution is a **mechanism**, not just a shape. Naming the mechanism
#    usually names the method.
# 2. **Poisson** means counting noise only; **negative binomial** adds the
#    biological variability that real data always has.
# 3. **p-values are uniform under the null.** A p-value histogram is the
#    cheapest diagnostic you own.
# 4. $t$, $\chi^2$ and $F$ describe **computed statistics**, not measurements;
#    the $t$'s fat tails are what keep small studies honest.
# 5. Biology varies **multiplicatively**, so logs turn skewed data into the
#    symmetric, additive form every standard method expects.
#
# **Next:** `F4_estimates_and_standard_errors.py`
