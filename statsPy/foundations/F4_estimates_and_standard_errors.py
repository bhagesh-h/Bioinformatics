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
# # Foundations 4: Estimates, standard errors, and the central limit theorem
#
# **Curriculum link:** `stats.md` -> Part 0, §0.6, equations (0.11)-(0.12)
# **Assumes:** `F1`, `F2`, `F3`.
#
# ## The question this module answers
#
# You computed an average from your data. **How much should you trust it?**
#
# The answer comes from a thought experiment you can never perform but can
# always simulate: *if I ran this whole experiment again, how different would
# the answer be?* That spread is the **standard error**, and remarkably, it can
# be worked out from the single sample you actually have.

# %%
import os

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "F4_estimates_and_standard_errors"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(3)

# %% [markdown]
# ## 1. Estimator vs estimate
#
# A small distinction that clears up a lot of confusion.

# %%
header("1. The recipe and the number it produced")

TRUE_MU, TRUE_SIGMA = 20.0, 4.0
sample = rng.normal(TRUE_MU, TRUE_SIGMA, 12)

print(f"  ESTIMATOR: 'take the average of the sample'   <- a recipe")
print(f"  ESTIMATE : {sample.mean():.4f}                        <- what it gave, this time")
print(f"  TRUTH    : {TRUE_MU:.4f}                        <- unknown in real life")

print("""
  The distinction matters because PROPERTIES BELONG TO THE RECIPE, not to the
  number. It is meaningless to ask whether 19.87 is unbiased. It is very
  meaningful to ask whether "take the average" is unbiased - and it is.

  So when someone says "the sample mean is unbiased with standard error
  sigma/sqrt(n)", they are describing the recipe's behaviour across all the
  experiments you might have run, not the number on your screen.""")

# %% [markdown]
# ## 2. The sampling distribution
#
# Imagine repeating the entire experiment thousands of times. The resulting
# estimates form a distribution. You only ever see **one draw** from it: but
# its shape is what "how much should I trust this?" means.

# %%
header("2. The distribution of an estimate across repeat experiments")

N = 12
estimates = np.array([rng.normal(TRUE_MU, TRUE_SIGMA, N).mean()
                      for _ in range(50_000)])

print(f"  50,000 repeats of a {N}-sample experiment:\n")
print(f"  {'centre of the estimates':<34}{estimates.mean():>10.4f}"
      f"   (truth = {TRUE_MU})")
print(f"  {'spread of the estimates (SD)':<34}{estimates.std():>10.4f}")
print(f"  {'sigma / sqrt(n), eq. (0.11)':<34}{TRUE_SIGMA/np.sqrt(N):>10.4f}   <- matches")
print(f"\n  {'how often within 1 SE of truth':<34}"
      f"{np.mean(np.abs(estimates - TRUE_MU) < TRUE_SIGMA/np.sqrt(N)):>10.3f}")
print(f"  {'how often within 2 SE of truth':<34}"
      f"{np.mean(np.abs(estimates - TRUE_MU) < 2*TRUE_SIGMA/np.sqrt(N)):>10.3f}")

print("""
  THE SPREAD OF THIS DISTRIBUTION IS THE STANDARD ERROR. That is the whole
  definition. Everything else is machinery for estimating it without being
  able to run the experiment 50,000 times.""")

# %% [markdown]
# ## 3. SD and SE are different things
#
# They are confused constantly, including in published figures. The difference
# is not subtle and it changes what a claim means.

# %%
header("3. Standard DEVIATION vs standard ERROR")

print(f"  {'n':<8}{'SD of the data (s)':>22}{'SE of the mean':>18}{'ratio':>10}")
for n in (5, 20, 100, 500, 2000):
    s = rng.normal(TRUE_MU, TRUE_SIGMA, (4000, n))
    print(f"  {n:<8}{s.std(axis=1, ddof=1).mean():>22.4f}"
          f"{s.mean(axis=1).std():>18.4f}{s.std(axis=1, ddof=1).mean()/s.mean(axis=1).std():>10.2f}")

print(f"""
  Read DOWN the two columns. The SD column does not move - it is estimating
  sigma = {TRUE_SIGMA}, a fact about how much individual observations vary, and no
  amount of data changes that. The SE column shrinks steadily, because it
  describes how well you know the MEAN.

    s  (standard deviation) : how spread out the OBSERVATIONS are.
                              Does not shrink with n. Use it to DESCRIBE.

    SE (standard error)     : how spread out your ESTIMATE would be across
                              repeat experiments. Shrinks as sqrt(n).
                              Use it to say how PRECISELY you know something.

  An error bar in a figure is uninterpretable unless the caption says which
  one it is. SE bars are always shorter, which is presumably why they are
  more popular.""")

# %% [markdown]
# ## 4. You only have one sample: and that is enough
#
# The magic step: eq. (0.12) estimates the spread of the sampling distribution
# using nothing but the one sample you have.

# %%
header("4. Estimating the standard error from a single sample")

print(f"  {'trial':<8}{'sample mean':>14}{'sample s':>11}"
      f"{'estimated SE':>15}{'true SE':>10}")
true_se = TRUE_SIGMA / np.sqrt(N)
for trial in range(6):
    s_ = rng.normal(TRUE_MU, TRUE_SIGMA, N)
    print(f"  {trial+1:<8}{s_.mean():>14.3f}{s_.std(ddof=1):>11.3f}"
          f"{s_.std(ddof=1)/np.sqrt(N):>15.3f}{true_se:>10.3f}")

print(f"""
  Each trial's estimated SE is close to the truth ({true_se:.3f}) without ever
  repeating the experiment. That is what makes statistics practical: the
  variability WITHIN your single sample tells you how much your ESTIMATE
  would vary BETWEEN samples.

  Why does that work? Because of eq. (0.9) - variances of independent things
  add. Nothing else. And that is also its weak point: if your observations
  are not independent, s/sqrt(n) estimates the wrong thing entirely, and no
  amount of care elsewhere repairs it (F2, section 3).""")

# %% [markdown]
# ## 5. The central limit theorem
#
# The result that makes the normal distribution unavoidable: **averages become
# normal even when the data is not.**

# %%
header("5. Averages become bell-shaped whatever the source")

sources = {
    "uniform (flat)":       lambda k, r: rng.uniform(0, 1, (r, k)),
    "exponential (skewed)": lambda k, r: rng.exponential(1, (r, k)),
    "binary coin flips":    lambda k, r: rng.integers(0, 2, (r, k)).astype(float),
    "lognormal (very skewed)": lambda k, r: rng.lognormal(0, 1.5, (r, k)),
}

print(f"  Skewness of the AVERAGE (0 = perfectly symmetric):\n")
print(f"  {'source distribution':<26}" + "".join(f"{f'n={k}':>9}" for k in (1, 2, 5, 30, 200)))
for name, draw in sources.items():
    row = "".join(f"{st.skew(draw(k, 30_000).mean(axis=1)):>9.2f}" for k in (1, 2, 5, 30, 200))
    print(f"  {name:<26}{row}")

print("""
  Every row marches towards zero. Flat data, skewed data, data that can only
  be 0 or 1 - average enough of it and you get a bell curve.

  This is why methods built on the normal distribution work far more often
  than they have any right to: they are not assuming your DATA is normal, they
  are assuming your ESTIMATE is, and that is a much weaker requirement.

  But read the table again. The lognormal row is still skewed at n = 30. "n =
  30 is enough" is folklore. The nastier the source, the larger n must be, and
  with small samples of skewed data you should use a permutation test or a
  bootstrap instead of trusting the approximation (Topics 5 and 7).""")

# %% [markdown]
# ## 6. The tyranny of the square root

# %%
header("6. Why precision is expensive")

print(f"  {'n':<10}{'SE':>10}{'to halve the SE from here':>30}")
base = TRUE_SIGMA / np.sqrt(10)
for n in (10, 40, 160, 640, 2560):
    print(f"  {n:<10}{TRUE_SIGMA/np.sqrt(n):>10.4f}{f'n = {4*n}':>30}")

print("""
  Each row costs four times the previous one and buys a halving. To improve
  precision tenfold you need a hundred times the data.

  Two practical consequences:

    * Diminishing returns are severe. Going from n = 10 to n = 40 is
      transformative; from n = 1000 to n = 4000 is usually not worth it.

    * BETTER DESIGN BEATS MORE DATA. Removing a source of noise - pairing,
      blocking, a better assay, adjusting for a known covariate - reduces
      sigma directly rather than fighting the square root. That is why
      Topic 9 (power) is about design, not about sample size tables.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].hist(estimates, bins=70, color="steelblue", density=True)
ax[0].axvline(TRUE_MU, color="firebrick", lw=2)
for sgn in (-1, 1):
    ax[0].axvline(TRUE_MU + sgn * true_se, color="black", ls="--", lw=1)
ax[0].set_title(f"Sampling distribution (n = {N})")
ax[0].set_xlabel("sample mean")

ns = np.array([2, 5, 10, 25, 50, 100, 250, 500, 1000])
ax[1].plot(ns, [rng.normal(TRUE_MU, TRUE_SIGMA, (3000, n)).std(axis=1, ddof=1).mean()
                for n in ns], "o-", color="darkorange", label="SD of the data")
ax[1].plot(ns, [rng.normal(TRUE_MU, TRUE_SIGMA, (3000, n)).mean(axis=1).std()
                for n in ns], "o-", color="steelblue", label="SE of the mean")
ax[1].plot(ns, TRUE_SIGMA / np.sqrt(ns), "--", color="black",
           label=r"$\sigma/\sqrt{n}$")
ax[1].set_xscale("log"); ax[1].set_xlabel("n")
ax[1].set_title("SD stays put; SE shrinks"); ax[1].legend(fontsize=8)

for k, colour in ((1, "firebrick"), (2, "darkorange"), (30, "steelblue")):
    v = rng.exponential(1, (40_000, k)).mean(axis=1)
    ax[2].hist((v - v.mean()) / v.std(), bins=70, alpha=0.5, density=True,
               color=colour, label=f"n = {k}")
grid = np.linspace(-4, 4, 200)
ax[2].plot(grid, st.norm.pdf(grid), "k--", lw=1.5, label="normal")
ax[2].set_xlim(-4, 4); ax[2].set_title("CLT: averaging skewed data")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "standard_errors.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'standard_errors.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Why divide by $n-1$?
#
# The sample variance divides by $n-1$, not $n$. Find out what happens if you
# use $n$.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  true variance = {TRUE_SIGMA**2:.2f}\n")
# print(f"  {'n':<8}{'divide by n':>16}{'divide by n-1':>18}{'which is right?':>20}")
# for n in (2, 3, 5, 10, 50):
#     samples = rng.normal(TRUE_MU, TRUE_SIGMA, (60_000, n))
#     v_n = samples.var(axis=1, ddof=0).mean()      # ddof=0 -> divide by n
#     v_n1 = samples.var(axis=1, ddof=1).mean()     # ddof=1 -> divide by n-1
#     print(f"  {n:<8}{v_n:>16.3f}{v_n1:>18.3f}"
#           f"{'n-1':>20}")
#
# # Dividing by n gives a variance that is too SMALL, badly so at small n
# # (at n = 2 it is half the truth). Dividing by n-1 is right on average.
# #
# # The reason: you measure spread around the SAMPLE mean, not the true mean,
# # and the sample mean sits in the middle of your own data by construction -
# # closer to it than the true mean would be. So distances from it are
# # systematically too small, and n-1 corrects exactly for that.
# #
# # The n-1 is the "degrees of freedom": you had n numbers, but once the mean
# # is fixed, only n-1 of them are free to vary. That same idea reappears every
# # time you fit a model - a model with p parameters leaves n-p degrees of
# # freedom, which is why fitting many parameters to little data leaves you
# # unable to estimate the noise at all.

# %% [markdown]
# ### Problem 2: What does non-independence do to the standard error?
#
# Build samples where the observations are correlated, and check whether
# $s/\sqrt{n}$ still works.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# n_units, per_unit = 5, 10           # 50 rows, but only 5 independent units
# print(f"  50 measurements arranged as {n_units} units x {per_unit} sub-samples\n")
# print(f"  {'ICC':<8}{'design effect':>15}{'TRUE SE':>10}{'s/sqrt(50) claims':>20}"
#       f"{'too small by':>14}")
# for icc in (0.0, 0.1, 0.3, 0.6):
#     sd_u = np.sqrt(icc); sd_e = np.sqrt(1 - icc)
#     means, claimed = [], []
#     for _ in range(6000):
#         u = rng.normal(0, sd_u, n_units)
#         y = np.repeat(u, per_unit) + rng.normal(0, sd_e, n_units * per_unit)
#         means.append(y.mean()); claimed.append(y.std(ddof=1) / np.sqrt(len(y)))
#     de = 1 + (per_unit - 1) * icc
#     print(f"  {icc:<8.1f}{de:>15.2f}{np.std(means):>10.4f}"
#           f"{np.mean(claimed):>20.4f}{np.std(means)/np.mean(claimed):>13.2f}x")
#
# # At ICC = 0 the formula is right. As the correlation grows it understates
# # the uncertainty by roughly sqrt(design effect), where
# #
# #     design effect = 1 + (m - 1) x ICC          (eq. 1.8)
# #
# # and m is the number of sub-samples per unit. Note that m matters as much as
# # the ICC: 20 cells per mouse with ICC 0.1 is as damaging as 3 cells per
# # mouse with ICC 0.9.
# #
# # The number of rows in your table is not n. The number of INDEPENDENT UNITS
# # is n, and that is the most valuable sentence in this course.

# %% [markdown]
# ### Problem 3: Bootstrap: a standard error without a formula
#
# $s/\sqrt{n}$ only works for a mean. What if you want the SE of a **median**,
# which has no simple formula? Resample your own data and watch.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# data = rng.lognormal(3, 0.8, 60)      # one skewed sample, as in real life
#
# def bootstrap_se(x, statistic, B=4000):
#     """Resample x WITH REPLACEMENT B times; the spread of the statistic
#     across those resamples estimates its standard error."""
#     n = len(x)
#     vals = [statistic(rng.choice(x, n, replace=True)) for _ in range(B)]
#     return np.std(vals)
#
# # The truth, obtained the way we never can in real life: repeat the whole
# # experiment many times.
# truth_mean = np.std([np.mean(rng.lognormal(3, 0.8, 60)) for _ in range(4000)])
# truth_med = np.std([np.median(rng.lognormal(3, 0.8, 60)) for _ in range(4000)])
#
# print(f"  {'statistic':<14}{'formula SE':>14}{'bootstrap SE':>16}{'TRUE SE':>12}")
# print(f"  {'mean':<14}{data.std(ddof=1)/np.sqrt(len(data)):>14.3f}"
#       f"{bootstrap_se(data, np.mean):>16.3f}{truth_mean:>12.3f}")
# print(f"  {'median':<14}{'none exists':>14}"
#       f"{bootstrap_se(data, np.median):>16.3f}{truth_med:>12.3f}")
#
# # For the mean, the bootstrap reproduces what the formula gives - a useful
# # sanity check that the method is sound.
# #
# # For the median there IS no simple formula, and the bootstrap gets it
# # anyway. That is the point: the bootstrap gives you a standard error for
# # almost any statistic you can compute, by using your sample as a stand-in
# # for the population.
# #
# # It is not magic. It assumes your sample is representative and that
# # observations are INDEPENDENT - resampling rows of clustered data
# # reproduces exactly the error from Problem 2. For grouped data you must
# # resample whole GROUPS (the "cluster bootstrap"). Topic 5 covers this.

# %% [markdown]
# ## What to take away
#
# 1. **Properties belong to the estimator** (the recipe), not to the estimate
#    (the number).
# 2. The **standard error** is the spread of your estimate across repeat
#    experiments: and you can estimate it from one sample.
# 3. **SD describes the data; SE describes your knowledge.** SD does not shrink
#    with $n$; SE does.
# 4. The **central limit theorem** makes averages normal whatever the source -
#    but not instantly, and not for free.
# 5. Precision costs $n$ **squared**. Better design beats more samples.
# 6. All of it rests on **independence**. Without it, $s/\sqrt{n}$ is wrong.
#
# **Next:** `F5_confidence_intervals_and_pvalues.py`
