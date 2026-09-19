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
# # Foundations 2: Probability, independence, and conditional probability
#
# **Curriculum link:** `stats.md` -> Part 0, §0.4, equations (0.1)-(0.9)
# **Assumes:** `F1`.
#
# ## Why you need this
#
# Your sample could have come out otherwise. Probability is the mathematics of
# "could have come out otherwise", which is why it sits underneath every
# p-value, every confidence interval, and every error rate in this course.
#
# You need surprisingly little of it. This module covers all of it:
#
# * probability as a long-run frequency,
# * the two combining rules, and the assumption one of them needs,
# * **independence**: the assumption that fails most often in biology,
# * **conditional probability**, and why $P(A \mid B) \neq P(B \mid A)$,
# * expectation and variance, and the one variance rule that explains why
#   averaging works at all.

# %%
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "F2_probability_basics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(1)

# %% [markdown]
# ## 1. A probability is a long-run frequency
#
# "The probability of heads is 0.5" is a claim about what happens if you keep
# flipping: not a claim about the next flip. Watch it converge.

# %%
header("1. Probability is what happens in the long run")

flips = rng.integers(0, 2, 20_000)          # 0 = tails, 1 = heads
print(f"  {'after this many flips':<26}{'proportion heads':>20}")
for n in (10, 100, 1_000, 10_000, 20_000):
    print(f"  {n:<26,}{flips[:n].mean():>20.4f}")

print("""
  The proportion wanders a lot early on and settles down later. That is all a
  probability is: the value the proportion settles to.

  Note what it does NOT tell you - whether the next flip is heads. A
  probability is a statement about a PROCESS repeated, never about a single
  outcome. This is exactly the distinction that makes confidence intervals
  and p-values so easy to misread (F5).""")

# %% [markdown]
# ## 2. The two combining rules, eq. (0.2)-(0.3)
#
# Everything in elementary probability is these two rules.

# %%
header("2. 'or' adds; 'and' multiplies (but only if independent)")

die = rng.integers(1, 7, 200_000)
p_1 = np.mean(die == 1)
p_2 = np.mean(die == 2)
p_1or2 = np.mean((die == 1) | (die == 2))
print("  OR rule, eq. (0.2) - for outcomes that cannot both happen:")
print(f"    P(1) = {p_1:.4f},  P(2) = {p_2:.4f}")
print(f"    P(1) + P(2)   = {p_1 + p_2:.4f}   <- the rule")
print(f"    P(1 or 2)     = {p_1or2:.4f}   <- simulated")

a = rng.integers(0, 2, 200_000)             # coin A
b = rng.integers(0, 2, 200_000)             # coin B, flipped separately
print("\n  AND rule, eq. (0.3) - for INDEPENDENT events:")
print(f"    P(A heads) = {a.mean():.4f},  P(B heads) = {b.mean():.4f}")
print(f"    product       = {a.mean() * b.mean():.4f}   <- the rule")
print(f"    P(both heads) = {np.mean((a == 1) & (b == 1)):.4f}   <- simulated")

# %% [markdown]
# ## 3. Independence, and what happens when it fails
#
# Two events are **independent** when knowing one happened tells you nothing
# about the other. This is the assumption behind eq. (0.3): and the one that
# biology breaks constantly.

# %%
header("3. When independence fails, multiplying gives the wrong answer")

# Two coins welded together: they always land the same way. Each is still a
# perfectly fair coin - P(heads) = 0.5 for each, separately.
welded = rng.integers(0, 2, 200_000)
c, d = welded, welded

print(f"  Two coins that are NOT independent (they always match):")
print(f"    P(C heads) = {c.mean():.4f}   <- still a fair coin")
print(f"    P(D heads) = {d.mean():.4f}   <- still a fair coin")
print(f"    product (what eq. 0.3 would give) = {c.mean() * d.mean():.4f}")
print(f"    P(both heads), actually           = {np.mean((c == 1) & (d == 1)):.4f}")
print(f"    the rule is wrong by a factor of  "
      f"{np.mean((c == 1) & (d == 1)) / (c.mean() * d.mean()):.1f}x")

print("""
  Each coin is individually fair. The rule still fails, because the rule was
  never about the individual coins - it was about their INDEPENDENCE.

  Now scale that up. This is the whole of pseudoreplication:""")

# Now the same point at the scale of a real experiment. Two designs, both
# producing 120 numbers:
#   (a) 120 genuinely independent measurements
#   (b) 6 mice x 20 cells, where the 20 cells from one mouse share that
#       mouse's own level - so they are NOT independent of each other.
N_UNITS, N_SUB, SD_UNIT, SD_CELL = 6, 20, 1.0, 1.0
REPS = 4000

indep_means, clust_means, clust_naive_se = [], [], []
for _ in range(REPS):
    # (a) 120 independent draws
    indep_means.append(rng.normal(0, SD_CELL, N_UNITS * N_SUB).mean())
    # (b) 6 mouse levels, then 20 cells around each mouse level
    mouse_level = rng.normal(0, SD_UNIT, N_UNITS)
    cells = np.repeat(mouse_level, N_SUB) + rng.normal(0, SD_CELL, N_UNITS * N_SUB)
    clust_means.append(cells.mean())
    # what the naive formula would CLAIM the uncertainty is, for design (b)
    clust_naive_se.append(cells.std(ddof=1) / np.sqrt(N_UNITS * N_SUB))

true_sd_indep = np.std(indep_means)
true_sd_clust = np.std(clust_means)
naive_se_clust = np.mean(clust_naive_se)

print(f"\n  {'design':<40}{'rows':>6}{'TRUE SD of the mean':>22}")
print(f"  {'(a) 120 independent measurements':<40}{N_UNITS*N_SUB:>6}"
      f"{true_sd_indep:>22.4f}")
print(f"  {'(b) 6 mice x 20 cells':<40}{N_UNITS*N_SUB:>6}"
      f"{true_sd_clust:>22.4f}")
print(f"\n  For design (b), what s/sqrt(120) CLAIMS : {naive_se_clust:>8.4f}")
print(f"  What the uncertainty actually IS       : {true_sd_clust:>8.4f}")
print(f"  The naive formula is too small by      : "
      f"{true_sd_clust / naive_se_clust:>8.1f}x")
print(f"  For comparison, s/sqrt(6) would give   : "
      f"{np.mean([np.std(rng.normal(0, np.sqrt(SD_UNIT**2 + SD_CELL**2/N_SUB), N_UNITS), ddof=1) for _ in range(2000)]) / np.sqrt(N_UNITS):>8.4f}")

print("""
  Both designs produce 120 numbers. Design (b) is far less informative,
  because its 120 rows contain only 6 independent facts about the population.

  The naive formula divides by sqrt(120) as if there were 120. The last line
  shows what you get by treating the mouse as the unit and dividing by
  sqrt(6) instead - and that number matches the truth.

  This is not a subtle effect. In exercise E1 it turns p = 0.06 into
  p = 2e-12. Independence is not a technicality - it is the load-bearing
  assumption of applied statistics, and eq. (0.3) is where it enters.""")

# %% [markdown]
# ## 4. Conditional probability, eq. (0.4)
#
# $P(A \mid B)$: "the probability of $A$ **given** $B$" - is the probability of
# $A$ in the restricted world where you already know $B$ happened.

# %%
header("4. P(A given B) is not P(B given A)")

# A screening test for a disease that 1% of people have.
N = 1_000_000
PREVALENCE = 0.01
SENSITIVITY = 0.99      # P(test positive | diseased)
SPECIFICITY = 0.95      # P(test negative | healthy)

diseased = rng.random(N) < PREVALENCE
positive = np.where(diseased,
                    rng.random(N) < SENSITIVITY,        # true positives
                    rng.random(N) < (1 - SPECIFICITY))  # false positives

p_pos_given_dis = positive[diseased].mean()
p_dis_given_pos = diseased[positive].mean()

print(f"  A disease {PREVALENCE:.0%} of people have.")
print(f"  A test that is {SENSITIVITY:.0%} sensitive and {SPECIFICITY:.0%} specific.\n")
print(f"  {'quantity':<46}{'value':>10}")
print(f"  {'P(test positive | you have it)':<46}{p_pos_given_dis:>10.3f}")
print(f"  {'P(you have it | test positive)':<46}{p_dis_given_pos:>10.3f}   <- !")
print(f"\n  of {positive.sum():,} positive tests, only "
      f"{(positive & diseased).sum():,} are real cases")

print("""
  The test is 99% accurate at detecting the disease, and yet a positive result
  means you probably do NOT have it. Nothing is wrong with the test. The
  reason is that healthy people vastly outnumber sick ones, so even a small
  false-positive RATE produces a large NUMBER of false positives.

  This asymmetry - P(A|B) is not P(B|A) - is the most misread idea in
  statistics, and it is exactly the mistake people make with p-values:

    a p-value is   P(data this extreme | no real effect)
    people read it P(no real effect | data this extreme)

  Those are as different as the two rows above. F5 returns to this.""")

# %% [markdown]
# ## 5. Expectation and variance, eq. (0.5)-(0.7)
#
# Two numbers summarise a distribution: where it sits, and how spread out it
# is.

# %%
header("5. Expectation (where it sits) and variance (how spread out)")

MU, SIGMA = 10.0, 3.0
x = rng.normal(MU, SIGMA, 500_000)

print(f"  {'quantity':<40}{'formula':<22}{'value':>10}")
print(f"  {'expected value E[X] = mu':<40}{'mean of X':<22}{x.mean():>10.4f}")
print(f"  {'variance = E[(X - mu)^2]':<40}{'mean squared distance':<22}"
      f"{np.mean((x - MU) ** 2):>10.4f}")
print(f"  {'standard deviation = sqrt(variance)':<40}{'back to original units':<22}"
      f"{np.sqrt(np.mean((x - MU) ** 2)):>10.4f}")

print(f"""
  Why square the distances (eq. 0.6)? Because without squaring, the positives
  and negatives cancel exactly:

    mean of (X - mu)          = {np.mean(x - MU):>9.5f}   <- always ~0, useless
    mean of |X - mu|          = {np.mean(np.abs(x - MU)):>9.5f}   <- works, but awkward algebra
    mean of (X - mu)^2        = {np.mean((x - MU)**2):>9.5f}   <- the variance

  Squaring makes the units wrong (squared grams), so we take the square root
  at the end and get the STANDARD DEVIATION - a spread in the original units.

  Rule of thumb for a bell-shaped variable: about 68% of values fall within
  1 SD of the mean and 95% within 2 SD. Check it:
    within 1 SD: {np.mean(np.abs(x - MU) < SIGMA):.1%}
    within 2 SD: {np.mean(np.abs(x - MU) < 2 * SIGMA):.1%}""")

# %% [markdown]
# ## 6. The variance rule that explains everything, eq. (0.9)
#
# $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ -
# **but only when $X$ and $Y$ are independent.** This one line is why averaging
# reduces noise, and therefore why larger samples are more precise.

# %%
header("6. Variances add - for independent things (0.9)")

n = 200_000
X = rng.normal(0, 1, n)
Y = rng.normal(0, 1, n)                 # independent of X
Z = X.copy()                            # perfectly dependent on X

print(f"  {'':<34}{'Var(X)+Var(Y)':>16}{'Var(X+Y) actual':>18}")
print(f"  {'X, Y independent':<34}{X.var() + Y.var():>16.4f}{(X + Y).var():>18.4f}")
print(f"  {'X, Z identical (dependent)':<34}{X.var() + Z.var():>16.4f}"
      f"{(X + Z).var():>18.4f}   <- rule fails")

print("\n  Now the consequence. Average n independent values and the variance")
print("  of the average is divided by n, so its SD is divided by sqrt(n):\n")
print(f"  {'n':<8}{'SD of one value':>18}{'SD of the average':>20}{'sigma/sqrt(n)':>16}")
for k in (1, 4, 16, 64, 256):
    avgs = rng.normal(0, 1, (20_000, k)).mean(axis=1)
    print(f"  {k:<8}{1.0:>18.4f}{avgs.std():>20.4f}{1 / np.sqrt(k):>16.4f}")

print("""
  The last two columns match. That is eq. (0.11), the STANDARD ERROR, and it
  falls straight out of eq. (0.9).

  Two consequences worth carrying with you:

    * To halve your uncertainty you must QUADRUPLE your sample. Precision is
      expensive, and gets more expensive the more you have.

    * The rule required INDEPENDENCE. If your n values are 20 cells from each
      of 6 mice, the variance does not divide by 120, and dividing by 120
      anyway is exactly the error in section 3.

  Everything in F4 is this table.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

running = np.cumsum(flips) / np.arange(1, len(flips) + 1)
ax[0].plot(running, lw=0.8, color="steelblue")
ax[0].axhline(0.5, color="firebrick", ls="--", lw=1.5)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("number of flips"); ax[0].set_ylabel("proportion heads")
ax[0].set_title("A probability is a long-run frequency")

labels = ["true\npositives", "false\npositives"]
vals = [(positive & diseased).sum(), (positive & ~diseased).sum()]
ax[1].bar(labels, vals, color=["seagreen", "firebrick"])
ax[1].set_ylabel("people testing positive")
ax[1].set_title(f"P(disease | positive) = {p_dis_given_pos:.2f}")

ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
sds = [rng.normal(0, 1, (8000, k)).mean(axis=1).std() for k in ks]
ax[2].plot(ks, sds, "o-", color="steelblue", label="simulated")
ax[2].plot(ks, 1 / np.sqrt(ks), "--", color="firebrick",
           label=r"$\sigma/\sqrt{n}$")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("n averaged"); ax[2].set_ylabel("SD of the average")
ax[2].set_title("Eq. (0.9) $\\Rightarrow$ eq. (0.11)")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "probability_basics.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'probability_basics.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: The base rate
#
# Re-run the screening example with a disease that 30% of people have (a test
# used in a high-risk clinic rather than the general population). What happens
# to $P(\text{disease} \mid \text{positive})$, and why?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'prevalence':<14}{'P(pos | disease)':>18}{'P(disease | pos)':>18}")
# for prev in (0.001, 0.01, 0.10, 0.30, 0.60):
#     dis = rng.random(400_000) < prev
#     pos = np.where(dis, rng.random(400_000) < 0.99, rng.random(400_000) < 0.05)
#     print(f"  {prev:<14.1%}{pos[dis].mean():>18.3f}{dis[pos].mean():>18.3f}")
#
# # The middle column never moves: the TEST has not changed. The right column
# # swings from almost 0 to almost 1, purely because of how common the disease
# # is in the population being tested.
# #
# # The same test is nearly useless for screening the general population and
# # very informative in a high-risk clinic. This is why screening programmes
# # target risk groups, and why "the test is 99% accurate" is not a meaningful
# # statement on its own.
# #
# # The statistical version: your interpretation of a positive RESULT depends
# # on how plausible the thing was BEFORE you tested. That is Bayes' theorem,
# # and it is why a p-value of 0.04 means something quite different for a
# # pre-registered hypothesis than for the 500th gene you looked at.

# %% [markdown]
# ### Problem 2: How much does dependence cost you?
#
# Generate 120 measurements as 6 groups of 20 that share a group effect. Vary
# how strong the group effect is, and compare the true SD of the mean with what
# the naive $s/\sqrt{n}$ formula claims.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# n_groups, per_group = 6, 20
# print(f"  {'group SD':<12}{'ICC':>8}{'true SD of mean':>18}"
#       f"{'naive s/sqrt(120)':>20}{'too small by':>14}")
# for sd_group in (0.0, 0.3, 0.7, 1.5):
#     truth, naive = [], []
#     for _ in range(2000):
#         g = rng.normal(0, sd_group, n_groups)
#         y = np.repeat(g, per_group) + rng.normal(0, 1.0, n_groups * per_group)
#         truth.append(y.mean())
#         naive.append(y.std(ddof=1) / np.sqrt(len(y)))
#     icc = sd_group**2 / (sd_group**2 + 1.0**2)
#     print(f"  {sd_group:<12.1f}{icc:>8.2f}{np.std(truth):>18.4f}"
#           f"{np.mean(naive):>20.4f}{np.std(truth)/np.mean(naive):>13.1f}x")
#
# # With no group effect the two columns agree - the 120 values really are
# # independent. As the group effect grows they diverge fast, and the naive
# # formula understates the uncertainty several-fold.
# #
# # The ICC column is the fraction of total variance that lives BETWEEN groups.
# # Even an ICC of 0.1 - which sounds negligible - does real damage, because
# # the penalty depends on the GROUP SIZE too: it is 1 + (m-1) x ICC, the
# # "design effect" of eq. (1.8). With m = 20 cells per mouse, an ICC of 0.1
# # means each mouse contributes about a third of what you thought.
# #
# # The fix is never a better formula. It is to analyse at the level of the
# # unit you randomised (F1, Problem 3) or to model the grouping explicitly
# # with a mixed model (Topic 14).

# %% [markdown]
# ### Problem 3: Does the rule of thumb survive a skewed variable?
#
# The "68% within 1 SD, 95% within 2 SD" rule assumes a bell shape. Check it on
# a strongly right-skewed variable, like a gene expression level.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# normal_x = rng.normal(100, 20, 200_000)
# skewed_x = rng.lognormal(mean=np.log(100), sigma=0.9, size=200_000)
# print(f"  {'variable':<22}{'mean':>9}{'median':>9}{'within 1 SD':>14}"
#       f"{'within 2 SD':>14}{'below the mean':>16}")
# for name, v in (("normal", normal_x), ("skewed (lognormal)", skewed_x)):
#     m, sd = v.mean(), v.std()
#     print(f"  {name:<22}{m:>9.1f}{np.median(v):>9.1f}"
#           f"{np.mean(np.abs(v - m) < sd):>14.1%}"
#           f"{np.mean(np.abs(v - m) < 2 * sd):>14.1%}"
#           f"{np.mean(v < m):>16.1%}")
# print(f"\n  after taking logs of the skewed variable:")
# lv = np.log(skewed_x); m, sd = lv.mean(), lv.std()
# print(f"  {'log(skewed)':<22}{m:>9.1f}{np.median(lv):>9.1f}"
#       f"{np.mean(np.abs(lv - m) < sd):>14.1%}"
#       f"{np.mean(np.abs(lv - m) < 2 * sd):>14.1%}"
#       f"{np.mean(lv < m):>16.1%}")
#
# # For the skewed variable the rule of thumb breaks, and note the last column:
# # far MORE than half the values sit below the mean. A few large values drag
# # the mean above the typical value, so "the average gene expression" is not a
# # description of a typical gene.
# #
# # Taking logs restores the bell shape, and with it the rule of thumb and the
# # entire standard toolkit. This is why expression, concentrations and
# # survival times are analysed on a log scale - not tradition, but because
# # the log turns multiplicative, skewed variation into additive, symmetric
# # variation. Topic 2 covers when a transformation is appropriate and what it
# # does to the interpretation of your effect size.

# %% [markdown]
# ## What to take away
#
# 1. A probability is a **long-run frequency**, a statement about a repeated
#    process: never about one outcome.
# 2. "Or" adds; "and" multiplies **only under independence**.
# 3. **Independence is the assumption biology breaks**, and eq. (0.3) is where
#    the damage begins.
# 4. $P(A \mid B) \neq P(B \mid A)$. This is the p-value misreading, and the
#    reason a 99%-accurate test can still mostly produce false positives.
# 5. Variance is the average **squared** distance from the mean; its square
#    root, the SD, is the spread in original units.
# 6. Variances **add for independent things**: and that single fact is why
#    averaging works, and why $\mathrm{SE} = \sigma/\sqrt{n}$.
#
# **Next:** `F3_distributions.py`
