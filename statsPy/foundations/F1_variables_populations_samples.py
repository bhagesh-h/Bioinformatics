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
# # Foundations 1: Variables, populations, and samples
#
# **Curriculum link:** `stats.md` -> Part 0, §0.1-§0.3
# **Assumes:** nothing. If you can read a table, you can read this.
#
# ## The one idea
#
# You measure a few things. You want to say something about all the things.
#
# That gap: between the handful you measured and the many you care about - is
# what statistics is for. This module makes the gap visible by doing something
# you can never do in real life: we will *invent* an entire population, so we
# know the true answer, and then take samples from it and see how close we get.
#
# Everything else in this course is a more sophisticated version of this
# module.

# %%
# These three lines appear at the top of nearly every module.
#   numpy      - arrays and random numbers ("np" is the conventional shorthand)
#   matplotlib - plotting
#   os         - to find out where to save figures
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")   # draw to a file, not to a window (we are in a container)
import matplotlib.pyplot as plt

MODULE_NAME = "F1_variables_populations_samples"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    """Print a banner so the output is easy to read."""
    print("\n" + "=" * 72); print(t); print("=" * 72)


# A "random number generator" with a fixed starting point (a *seed*). Fixing the
# seed means you get the same "random" numbers every time you run this file,
# which is what makes the results reproducible. Change 0 to 1 and every number
# below changes - but every CONCLUSION stays the same. That is the point.
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A variable is something that varies
#
# Here is a small dataset: one row per mouse, one column per variable. This is
# how essentially all data is laid out.

# %%
header("1. Variables: one row per unit, one column per measurement")

n_mice = 8
data = {
    "mouse":     [f"m{i+1}" for i in range(n_mice)],
    "treatment": ["control"] * 4 + ["drug"] * 4,
    "weight_g":  np.round(rng.normal(24, 2, n_mice), 1),
    "marker":    np.round(rng.normal(12, 1.5, n_mice), 1),
    "responded": rng.integers(0, 2, n_mice),
    "grade":     rng.choice(["I", "II", "III"], n_mice),
}

print(f"  {'mouse':<8}{'treatment':<12}{'weight_g':>10}{'marker':>9}"
      f"{'responded':>11}{'grade':>7}")
for i in range(n_mice):
    print(f"  {data['mouse'][i]:<8}{data['treatment'][i]:<12}"
          f"{data['weight_g'][i]:>10}{data['marker'][i]:>9}"
          f"{data['responded'][i]:>11}{data['grade'][i]:>7}")

print(f"""
  The UNIT here is the mouse. There are {n_mice} of them, so n = {n_mice}.

  Deciding what the unit is turns out to be the hardest and most important
  question in experimental biology. If we had measured 20 cells from each
  mouse, we would have 160 rows - but still only {n_mice} units. Counting those
  160 rows as 160 independent observations is the most common fatal error in
  published biology. That is Topic 1, and exercise E1.""")

# %% [markdown]
# ## 2. The type of a variable decides what arithmetic is allowed
#
# This is not pedantry. The type determines which methods are valid, and
# getting it wrong is a real and common source of nonsense.

# %%
header("2. Variable types, and what you may do with each")

print(f"  {'variable':<12}{'type':<16}{'average it?':<14}{'why'}")
rows = [
    ("weight_g",  "continuous",  "yes",
     "any value in a range; differences are meaningful"),
    ("marker",    "continuous",  "yes",
     "same"),
    ("responded", "binary",      "yes (!)",
     "0/1, so the average IS the proportion who responded"),
    ("grade",     "ordinal",     "careful",
     "ordered, but is III minus II the same as II minus I?"),
    ("treatment", "categorical", "no",
     "'control' and 'drug' are labels, not quantities"),
]
for v, t, a, why in rows:
    print(f"  {v:<12}{t:<16}{a:<14}{why}")

prop = data["responded"].mean()
print(f"""
  The binary case is worth pausing on:
    responded = {list(data['responded'])}
    average   = {prop:.3f}  <- this is the PROPORTION that responded

  Coding yes/no as 1/0 makes the mean and the proportion the same number. That
  small trick is why binary outcomes slot into the same machinery as
  continuous ones (Topic 13).

  And the categorical case is where software betrays you: if `treatment` were
  coded 1 and 2 instead of 'control' and 'drug', every function here would
  happily compute an average of 1.5 and tell you nothing is wrong.""")

# %% [markdown]
# ## 3. Population vs sample: the whole problem, made visible
#
# In real research you only ever see a sample, and the population is unknown
# forever. Here we cheat: we build the population ourselves. That lets us
# *check* whether our sample-based answers are any good.

# %%
header("3. Inventing a population so we can see how well sampling works")

# THE POPULATION: 100,000 mice. In reality this is unknowable. Here it is ours.
POP_SIZE = 100_000
TRUE_MEAN = 24.0        # this is mu  - the parameter we are trying to learn
TRUE_SD = 2.0           # this is sigma
population = rng.normal(TRUE_MEAN, TRUE_SD, POP_SIZE)

print(f"  The population: {POP_SIZE:,} mice")
print(f"    true mean   mu    = {population.mean():.4f}   <- Greek letter = TRUTH")
print(f"    true SD     sigma = {population.std():.4f}")
print("\n  Now take samples from it, as a real experiment would:\n")
print(f"  {'sample size n':<16}{'sample mean x-bar':>20}{'error':>12}")

for n in (3, 10, 30, 100, 1000):
    sample = rng.choice(population, n, replace=False)
    print(f"  {n:<16}{sample.mean():>20.3f}{sample.mean() - TRUE_MEAN:>+12.3f}")

print(f"""
  Two things to notice, and they are the whole of §0.3:

    1. Every sample gives a DIFFERENT answer. None equals {TRUE_MEAN}. The
       sample mean is not the truth - it is a noisy glimpse of the truth.

    2. Bigger samples are closer ON AVERAGE - but look at the n = 3 row. It
       happened to land almost exactly on the truth this time, closer than
       n = 100 did. That is luck, not reliability, and it is exactly why you
       cannot judge a method from one run. Section 4 repeats each experiment
       1,000 times, which is the only way to tell lucky from reliable.

  Notation, which you will see everywhere:
    mu, sigma  (Greek)      = population values. Fixed. Unknown. What you want.
    x-bar, s   (Latin/hats) = sample values.     Known. They vary.

  Greek is truth; Latin is your guess.""")

# %% [markdown]
# ## 4. The same experiment, run 1,000 times
#
# A single sample tells you almost nothing about how *reliable* a sample is.
# So let us do what no one can do in real life: repeat the experiment.

# %%
header("4. Repeating the experiment 1,000 times")

N_REPEATS = 1000
print(f"  {'n':<8}{'mean of the 1000 estimates':>28}{'spread of them (SD)':>22}")
spreads = {}
for n in (3, 10, 30, 100):
    means = np.array([rng.choice(population, n, replace=False).mean()
                      for _ in range(N_REPEATS)])
    spreads[n] = means
    print(f"  {n:<8}{means.mean():>28.3f}{means.std():>22.3f}")

print(f"""
  Read the two columns separately - they say different things.

  COLUMN 1 is about BIAS: every row lands on {TRUE_MEAN}. The sample mean is
  RIGHT ON AVERAGE, even with n = 3. A single 3-mouse experiment can be far
  off, but it is not systematically too high or too low.

  COLUMN 2 is about PRECISION: the spread shrinks as n grows. This spread has
  a name - the STANDARD ERROR - and it is what "how much should I trust this
  number?" actually means.

  Bias and precision are independent. An estimate can be unbiased and useless
  (huge spread), or precise and wrong (small spread, systematically off). A
  confounded experiment (E1, E4) is precise and wrong, which is the dangerous
  combination, because precision looks like credibility.""")

# %% [markdown]
# ## 5. The surprising part: the fraction sampled does not matter
#
# Almost everyone's intuition gets this wrong.

# %%
header("5. Sample SIZE matters; sample FRACTION does not")

small_pop = rng.normal(TRUE_MEAN, TRUE_SD, 2_000)
big_pop = rng.normal(TRUE_MEAN, TRUE_SD, 2_000_000)

print(f"  {'population':<16}{'pop size':>12}{'n':>8}{'fraction':>12}"
      f"{'SD of estimate':>18}")
for name, pop in (("small", small_pop), ("big", big_pop)):
    for n in (100,):
        means = np.array([rng.choice(pop, n, replace=False).mean()
                          for _ in range(600)])
        print(f"  {name:<16}{len(pop):>12,}{n:>8}{n/len(pop):>12.5%}"
              f"{means.std():>18.4f}")

print("""
  One population is a THOUSAND times bigger than the other. A sample of 100
  covers 5% of the first and 0.005% of the second. The precision is the same.

  This is why a poll of 1,000 people works about equally well for a small
  country and a huge one, and why "we sequenced 90% of the cells" is not the
  reassurance it sounds like. What buys precision is n - the number of
  INDEPENDENT units - and nothing else.""")

# %% [markdown]
# ## 6. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].hist(population, bins=60, color="grey", edgecolor="none")
ax[0].axvline(TRUE_MEAN, color="firebrick", lw=2, label=f"true mean $\\mu$ = {TRUE_MEAN}")
ax[0].set_title("The population (100,000 mice)")
ax[0].set_xlabel("weight (g)"); ax[0].set_ylabel("how many mice")
ax[0].legend(fontsize=8)

one_sample = rng.choice(population, 10, replace=False)
ax[1].hist(population, bins=60, color="lightgrey", edgecolor="none", density=True)
ax[1].plot(one_sample, np.full(10, 0.02), "o", color="steelblue", ms=7,
           label="one sample, n = 10")
ax[1].axvline(TRUE_MEAN, color="firebrick", lw=2)
ax[1].axvline(one_sample.mean(), color="steelblue", lw=2, ls="--",
              label=f"$\\bar{{x}}$ = {one_sample.mean():.2f}")
ax[1].set_title("One sample is a noisy glimpse")
ax[1].set_xlabel("weight (g)"); ax[1].legend(fontsize=8)

for n, colour in zip((3, 10, 30, 100), ("firebrick", "darkorange", "seagreen", "steelblue")):
    ax[2].hist(spreads[n], bins=40, alpha=0.55, color=colour, density=True,
               label=f"n = {n}")
ax[2].axvline(TRUE_MEAN, color="black", lw=1.5)
ax[2].set_title("Estimates from 1,000 repeat experiments")
ax[2].set_xlabel("sample mean"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "populations_and_samples.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'populations_and_samples.png')}")

# %% [markdown]
# # PROBLEMS
#
# Write your attempt in the `YOUR CODE HERE` block, run the file, then
# uncomment the solution and compare. The prose at the end of each solution is
# the real answer: the code just shows how to get there.
#
# ### Problem 1: Does a bigger sample fix a biased one?
#
# Suppose your mice are not randomly chosen: you happen to pick the heaviest
# ones. Simulate that, and see whether a large $n$ rescues you.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'n':<8}{'random sample':>16}{'heaviest-60% sample':>24}")
# heavy_half = population[population > np.quantile(population, 0.40)]
# for n in (10, 100, 1000, 10000):
#     fair = rng.choice(population, n, replace=False).mean()
#     biased = rng.choice(heavy_half, n, replace=False).mean()
#     print(f"  {n:<8}{fair:>16.3f}{biased:>24.3f}")
# print(f"  {'TRUTH':<8}{TRUE_MEAN:>16.3f}{TRUE_MEAN:>24.3f}")
#
# # The random column converges on the truth. The biased column converges just
# # as tightly - on the WRONG NUMBER. More data makes a biased estimate more
# # precisely wrong, and gives you more confidence in a false answer.
# #
# # This is the most important thing to understand about sample size. n fixes
# # NOISE. It does nothing about BIAS. No amount of data repairs a sample that
# # was collected in a way related to what you are measuring - which is why
# # HOW you select units matters more than how many you select, and why
# # randomisation is the most valuable single step in an experiment.

# %% [markdown]
# ### Problem 2: The median and the outlier
#
# Add one absurd value (a mis-entered weight of 240 g) to a sample of 20 and
# see what happens to the mean and the median.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# clean = rng.choice(population, 20, replace=False)
# dirty = np.append(clean, 240.0)          # a decimal point in the wrong place
# print(f"  {'':<22}{'mean':>10}{'median':>10}{'SD':>10}")
# print(f"  {'clean (n=20)':<22}{clean.mean():>10.2f}{np.median(clean):>10.2f}"
#       f"{clean.std(ddof=1):>10.2f}")
# print(f"  {'with one typo (n=21)':<22}{dirty.mean():>10.2f}{np.median(dirty):>10.2f}"
#       f"{dirty.std(ddof=1):>10.2f}")
# print(f"  {'shift':<22}{dirty.mean()-clean.mean():>+10.2f}"
#       f"{np.median(dirty)-np.median(clean):>+10.2f}"
#       f"{dirty.std(ddof=1)-clean.std(ddof=1):>+10.2f}")
#
# # One bad value moves the mean by about 10 g and the median by almost
# # nothing. The median is ROBUST: it depends on the ORDER of the values, not
# # their magnitudes, so a single extreme point cannot drag it.
# #
# # The SD is hit even harder than the mean, because it squares distances - so
# # outliers affect your uncertainty estimate more than your estimate.
# #
# # This is why you always PLOT your data before analysing it (Topic 3), and
# # why skewed biological measurements are usually analysed on a log scale.
# # It is not that the median is "better" - it answers a different question.
# # Use the mean when you want a total or an average effect; use the median
# # when you want a typical value and the tail is not the point.

# %% [markdown]
# ### Problem 3: What is the unit?
#
# For each study below, decide what one row of the analysis table should be,
# and therefore what $n$ is. Then check yourself against the solution.
#
# 1. 6 mice per group, one blood sample each, one measurement per sample.
# 2. 6 mice per group, 20 cells imaged per mouse.
# 3. 3 flasks of cells per condition, each split into 3 wells, each measured.
# 4. 200 patients, one tumour biopsy each, 20,000 genes measured per biopsy.
# 5. 4 patients, 5,000 single cells sequenced per patient.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# cases = [
#     ("6 mice/group, 1 measurement each", "mouse", 6,
#      "nothing to collapse; rows already independent"),
#     ("6 mice/group, 20 cells each", "mouse", 6,
#      "120 cells, but cells share a mouse -> average to 6 mouse values"),
#     ("3 flasks/condition, 3 wells each", "flask", 3,
#      "wells are technical replicates OF the flask"),
#     ("200 patients, 20,000 genes each", "patient", 200,
#      "genes are VARIABLES, not units - 20,000 tests of n=200"),
#     ("4 patients, 5,000 cells each", "patient", 4,
#      "20,000 cells, n = 4. This is the single-cell trap (Module 31)"),
# ]
# print(f"  {'study':<38}{'unit':<10}{'n':>4}   why")
# for study, unit, n, why in cases:
#     print(f"  {study:<38}{unit:<10}{n:>4}   {why}")
#
# # The rule: THE UNIT IS WHAT YOU RANDOMISED, or what varies independently.
# # Everything measured inside a unit is a sub-sample of it, and averaging
# # those sub-samples loses nothing about the treatment effect.
# #
# # Case 4 is the one people get right by accident and case 5 is the one they
# # get wrong. In both, the big number (20,000) is tempting, and in both it is
# # not n. In case 4 the 20,000 genes create a MULTIPLE TESTING problem
# # (Topic 8); in case 5 the 20,000 cells create a PSEUDOREPLICATION problem
# # (Topic 1). Different problems, opposite fixes, same tempting big number.

# %% [markdown]
# ## What to take away
#
# 1. A **unit** is the thing that varies independently. Counting sub-samples as
#    units is the most common fatal error in biology.
# 2. **Greek letters are the truth** ($\mu$, $\sigma$); Latin letters and hats
#    are your estimates ($\bar{x}$, $s$).
# 3. A sample mean is **unbiased but noisy**. Every experiment gives a
#    different answer, and that is normal, not a mistake.
# 4. Larger $n$ reduces **noise**. It does nothing about **bias**.
# 5. What matters is the **size** of the sample, not the fraction of the
#    population it covers.
#
# **Next:** `F2_probability_basics.py`
