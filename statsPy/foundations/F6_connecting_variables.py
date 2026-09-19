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
# # Foundations 6: How statistics connects variables
#
# **Curriculum link:** `stats.md` -> Part 0, §0.9, equation (0.16)
# **Assumes:** `F1`-`F5`.
#
# ## The last foundation
#
# Most real questions are not "what is the average?" but "**does this relate to
# that?**" Does the drug change the marker; does expression predict survival.
#
# This module covers two things, and they are the two things that matter most
# in the rest of the course:
#
# 1. **One equation**: $Y = \beta_0 + \beta_1 X_1 + \cdots + \varepsilon$ -
#    which turns out to be nearly every method you have heard of, in disguise.
# 2. **Four roles a variable can play**: confounder, mediator, collider,
#    precision variable: which look *identical in the data* and demand
#    opposite treatment.
#
# The second is where careers go wrong. "Adjust for more variables to be safe"
# is one of the most damaging pieces of folk wisdom in applied science.

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

MODULE_NAME = "F6_connecting_variables"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(5)

# %% [markdown]
# ## 1. Reading eq. (0.16)
#
# $$Y = \beta_0 + \beta_1 X + \varepsilon$$
#
# *The outcome equals a baseline, plus a contribution from the predictor, plus
# what is left over.* Let us build data where we KNOW the answer, then recover
# it.

# %%
header("1. Fitting a line, and what the numbers mean")

TRUE_B0, TRUE_B1, NOISE = 5.0, 2.5, 3.0
n = 200
dose = rng.uniform(0, 10, n)
response = TRUE_B0 + TRUE_B1 * dose + rng.normal(0, NOISE, n)

fit = smf.ols("response ~ dose", data=pd.DataFrame(dict(dose=dose,
                                                        response=response))).fit()
b0, b1 = fit.params["Intercept"], fit.params["dose"]
se1 = fit.bse["dose"]

print(f"  {'quantity':<36}{'truth':>10}{'estimate':>11}{'SE':>9}")
print(f"  {'intercept  beta_0':<36}{TRUE_B0:>10.2f}{b0:>11.3f}"
      f"{fit.bse['Intercept']:>9.3f}")
print(f"  {'slope      beta_1':<36}{TRUE_B1:>10.2f}{b1:>11.3f}{se1:>9.3f}")
print(f"  {'residual SD (the noise)':<36}{NOISE:>10.2f}"
      f"{np.sqrt(fit.scale):>11.3f}{'':>9}")

print(f"""
  How to read them:

    beta_0 = {b0:.2f}  the predicted response when dose = 0. Often meaningless on
                  its own (nobody gives a dose of 0), but the line has to
                  start somewhere.

    beta_1 = {b1:.2f}  THE ANSWER. Increase the dose by one unit and the response
                  goes up by about {b1:.2f}. This is the slope, and in almost every
                  analysis in this course, the estimate you care about is a
                  slope like this one.

    residual   what the model failed to explain. Every observation sits some
               distance off the line; those distances are the epsilon in
               eq. (0.16), and their spread is what all your standard errors
               are built from.

  And the interval, exactly as in F5:
    95% CI for the slope = {b1:.3f} +/- 1.96 x {se1:.3f}
                         = ({b1 - 1.96*se1:.3f}, {b1 + 1.96*se1:.3f})""")

# %% [markdown]
# ## 2. A t-test IS a linear model
#
# Most of the tests you have heard of are eq. (0.16) with a particular kind of
# predictor. Here is the proof, not the assertion.

# %%
header("2. The same answer, three different names")

g0 = rng.normal(10.0, 2.0, 30)
g1 = rng.normal(12.0, 2.0, 30)
df = pd.DataFrame({"y": np.concatenate([g0, g1]),
                   "group": ["control"] * 30 + ["drug"] * 30})

tt = st.ttest_ind(g1, g0, equal_var=True)
lm = smf.ols("y ~ group", data=df).fit()
av = sm.stats.anova_lm(lm)

print(f"  {'method':<34}{'estimate':>12}{'p-value':>14}")
print(f"  {'two-sample t-test':<34}{g1.mean()-g0.mean():>12.4f}{tt.pvalue:>14.6f}")
print(f"  {'linear model  y ~ group':<34}{lm.params['group[T.drug]']:>12.4f}"
      f"{lm.pvalues['group[T.drug]']:>14.6f}")
print(f"  {'one-way ANOVA':<34}{'':>12}{av['PR(>F)'].iloc[0]:>14.6f}")
print(f"\n  t statistic from the t-test : {tt.statistic:.6f}")
print(f"  t statistic from the model  : {lm.tvalues['group[T.drug]']:.6f}")

print("""
  Identical to six decimal places, because they are the same calculation.

  The trick is how a categorical predictor enters the equation: "drug" becomes
  a 0/1 column, so beta_1 is the difference between the group means, and
  testing beta_1 = 0 IS the t-test.

  This is why Topic 11 calls the linear model "the core framework". Learning
  twenty named tests is memorisation. Learning eq. (0.16) and how each kind of
  predictor and outcome plugs into it is understanding:

    binary predictor       -> t-test
    categorical predictor  -> ANOVA
    continuous predictor   -> regression
    several predictors     -> multiple regression / ANCOVA
    binary OUTCOME         -> logistic regression      (Topic 13)
    count OUTCOME          -> Poisson / neg. binomial  (Topic 13)
    grouped observations   -> mixed model              (Topic 14)
    time-to-event OUTCOME  -> Cox regression           (Topic 28)""")

# %% [markdown]
# ## 3. What "adjusting for" actually does
#
# Adding a variable to eq. (0.16) changes the meaning of every other
# coefficient. This is the mechanism behind confounder control.

# %%
header("3. Holding other things fixed")

n = 400
age = rng.normal(60, 10, n)
# Treatment is given more often to older patients - so age is a CONFOUNDER.
treated = (rng.random(n) < 1 / (1 + np.exp(-(age - 60) / 5))).astype(float)
TRUE_TREAT_EFFECT = -2.0
outcome = 30 + TRUE_TREAT_EFFECT * treated + 0.5 * age + rng.normal(0, 3, n)
d = pd.DataFrame(dict(age=age, treated=treated, outcome=outcome))

m_crude = smf.ols("outcome ~ treated", data=d).fit()
m_adj = smf.ols("outcome ~ treated + age", data=d).fit()

print(f"  TRUE treatment effect = {TRUE_TREAT_EFFECT:+.2f}\n")
print(f"  {'model':<36}{'treatment coefficient':>24}{'bias':>10}")
print(f"  {'outcome ~ treated':<36}{m_crude.params['treated']:>24.3f}"
      f"{m_crude.params['treated']-TRUE_TREAT_EFFECT:>+10.3f}")
print(f"  {'outcome ~ treated + age':<36}{m_adj.params['treated']:>24.3f}"
      f"{m_adj.params['treated']-TRUE_TREAT_EFFECT:>+10.3f}")

print(f"""
  The crude comparison has the WRONG SIGN: a drug that helps appears to harm,
  because the people who got it were older, and age drives the outcome.

  Adding age to the model recovers the truth. "beta_treated, adjusting for
  age" means: among patients OF THE SAME AGE, what is the difference? That is
  what "holding other predictors fixed" buys you, and it is the entire
  mechanism of confounder control.

  The crude comparison is not badly ESTIMATED - its standard error is small
  and its p-value is tiny. It is precisely and confidently wrong, which is the
  dangerous kind of wrong (F1, section 4).""")

# %% [markdown]
# ## 4. The four roles: and why "adjust for everything" is wrong
#
# Now the part that matters. We simulate four different causal structures.
# **In every one, the extra variable is measured, is strongly associated with
# the outcome, and improves the model fit.** Only the truth differs.

# %%
header("4. Confounder, mediator, collider, precision variable")

N = 4000
TRUE_EFFECT = 1.0
results = []

# --- CONFOUNDER: Z causes both X and Y --------------------------------------
z = rng.normal(0, 1, N)
x = 0.9 * z + rng.normal(0, 1, N)
y = TRUE_EFFECT * x + 1.2 * z + rng.normal(0, 1, N)
results.append(("confounder", "Z -> X and Z -> Y", x, y, z, "ADJUST"))

# --- MEDIATOR: X causes M, M causes Y (M is on the causal path) --------------
x = rng.normal(0, 1, N)
m = 1.1 * x + rng.normal(0, 1, N)
y = 0.4 * x + 1.0 * m + rng.normal(0, 1, N)     # total effect = 0.4 + 1.1*1.0
results.append(("mediator", "X -> M -> Y", x, y, m, "DO NOT"))

# --- COLLIDER: X and Y both cause C ------------------------------------------
x = rng.normal(0, 1, N)
y = TRUE_EFFECT * x + rng.normal(0, 1, N)
c = 1.0 * x + 1.0 * y + rng.normal(0, 1, N)
results.append(("collider", "X -> C <- Y", x, y, c, "DO NOT"))

# --- PRECISION VARIABLE: W causes Y only -------------------------------------
x = rng.normal(0, 1, N)
w = rng.normal(0, 1, N)
y = TRUE_EFFECT * x + 2.0 * w + rng.normal(0, 1, N)
results.append(("precision var.", "W -> Y only", x, y, w, "ADJUST"))

print(f"  {'role':<16}{'structure':<20}{'unadj.':>9}{'(SE)':>9}"
      f"{'adjusted':>10}{'(SE)':>9}{'verdict':>9}")
for role, structure, xv, yv, zv, verdict in results:
    dd = pd.DataFrame(dict(x=xv, y=yv, z=zv))
    un = smf.ols("y ~ x", data=dd).fit()
    ad = smf.ols("y ~ x + z", data=dd).fit()
    print(f"  {role:<16}{structure:<20}{un.params['x']:>9.3f}{un.bse['x']:>9.3f}"
          f"{ad.params['x']:>10.3f}{ad.bse['x']:>9.3f}{verdict:>9}")

print(f"""
  The true X -> Y effect is {TRUE_EFFECT:.1f} in the confounder, collider and
  precision rows (the mediator row is deliberately different - see below).

  Read each row:

    CONFOUNDER      unadjusted is badly wrong; adjusting FIXES it.
                    -> you MUST adjust.

    MEDIATOR        unadjusted gives the TOTAL effect of X on Y (about 1.5:
                    the direct 0.4 plus the 1.1 that flows through M).
                    Adjusting gives only the DIRECT effect (0.4), because you
                    have blocked the pathway through which the treatment
                    works. Neither is wrong - they answer different questions.
                    If you want "does the drug help?", do NOT adjust.

    COLLIDER        unadjusted is CORRECT; adjusting BREAKS it, pushing the
                    estimate far from the truth. Conditioning on a common
                    effect makes its causes appear related when they are not.
                    -> you must NOT adjust.

    PRECISION VAR.  the estimate barely moves - no bias either way - but
                    compare the two SE columns: adjusting roughly HALVES the
                    standard error, because W explained a big chunk of the
                    noise in Y. Free precision. -> adjust.

  NOW THE POINT. Every one of those Z variables is strongly associated with Y.
  Every one improves the model fit. NOTHING IN THE DATA distinguishes them.
  A confounder and a collider are statistically indistinguishable.

  The distinction comes from knowing HOW THE DATA WERE GENERATED - from
  biology, and from the time order of events. That is why you draw the causal
  diagram BEFORE you fit, and why automatic variable selection cannot be a
  substitute for thinking.

  The one rule that is nearly always safe:
      NEVER adjust for anything measured AFTER the treatment,
  because such a variable is a mediator, or a collider, or both.""")

# %% [markdown]
# ## 5. Fit statistics cannot save you

# %%
header("5. Every fit statistic says to include the collider")

x = rng.normal(0, 1, N)
y = TRUE_EFFECT * x + rng.normal(0, 1, N)
c = 1.0 * x + 1.0 * y + rng.normal(0, 1, N)
dd = pd.DataFrame(dict(x=x, y=y, c=c))
m1 = smf.ols("y ~ x", data=dd).fit()
m2 = smf.ols("y ~ x + c", data=dd).fit()

print(f"  {'model':<22}{'R-squared':>12}{'AIC':>12}{'resid SD':>11}"
      f"{'beta_x':>10}{'bias':>9}")
for name, m in (("y ~ x", m1), ("y ~ x + c", m2)):
    print(f"  {name:<22}{m.rsquared:>12.4f}{m.aic:>12.1f}"
          f"{np.sqrt(m.scale):>11.4f}{m.params['x']:>10.3f}"
          f"{m.params['x']-TRUE_EFFECT:>+9.3f}")
print(f"\n  p-value for the collider term: {m2.pvalues['c']:.3e}")

print("""
  The model with the collider has a higher R-squared, a much better AIC, a
  smaller residual SD, and an overwhelmingly significant coefficient.

  It is also the wrong answer.

  Every automatic model-selection procedure - stepwise regression, "include
  anything with p < 0.1", minimise AIC - would choose it. Model fit measures
  how well you PREDICT y. It says nothing about whether beta_x means what you
  want it to mean. Prediction and causal estimation are different jobs
  (Topics 31 and 32), and a procedure tuned for one can be actively harmful
  for the other.""")

# %% [markdown]
# ## 6. Association is not causation: but randomisation fixes it

# %%
header("6. Why randomisation is worth more than any analysis")

def study(randomised, reps=800):
    """Estimate the X->Y effect when X is assigned at random vs when it is not."""
    ests = []
    for _ in range(reps):
        z_ = rng.normal(0, 1, 300)                      # a hidden common cause
        if randomised:
            x_ = rng.normal(0, 1, 300)                  # YOU assign it
        else:
            x_ = 0.9 * z_ + rng.normal(0, 1, 300)       # it happens for reasons
        y_ = TRUE_EFFECT * x_ + 1.2 * z_ + rng.normal(0, 1, 300)
        ests.append(smf.ols("y_ ~ x_", data=pd.DataFrame(dict(x_=x_, y_=y_)))
                    .fit().params["x_"])
    return np.mean(ests), np.std(ests)

for label, rand in (("randomised experiment", True),
                    ("observational study", False)):
    m, s = study(rand)
    print(f"  {label:<28}estimate = {m:>6.3f}  (truth {TRUE_EFFECT:.1f}), "
          f"SD across studies = {s:.3f}")

print("""
  In the randomised version, nothing can have caused both X and Y, because YOU
  caused X - with a coin flip. Confounding is eliminated BY DESIGN, and the
  crude comparison is already correct. No adjustment needed, no assumptions
  about which variables matter.

  In the observational version the estimate is badly biased, and no amount of
  data fixes it (F1, Problem 1). You can adjust for confounders you measured
  and thought of. You cannot adjust for the ones you did not.

  This is why one randomised experiment is worth a great deal more than a
  larger observational study, and why Topic 32 is mostly about what you must
  ASSUME when randomisation is impossible.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

ax[0].scatter(dose, response, s=8, alpha=0.4, color="steelblue")
grid = np.linspace(0, 10, 50)
ax[0].plot(grid, b0 + b1 * grid, color="firebrick", lw=2,
           label=f"$\\hat{{y}} = {b0:.1f} + {b1:.2f}x$")
ax[0].set_xlabel("dose"); ax[0].set_ylabel("response")
ax[0].set_title("Eq. (0.16): a line through data"); ax[0].legend(fontsize=8)

older = age > np.median(age)
for mask, colour, lab in ((~older, "steelblue", "younger"), (older, "firebrick", "older")):
    ax[1].scatter(d.treated[mask] + rng.normal(0, 0.04, mask.sum()),
                  d.outcome[mask], s=7, alpha=0.4, color=colour, label=lab)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["untreated", "treated"])
ax[1].set_ylabel("outcome"); ax[1].legend(fontsize=8)
ax[1].set_title("Confounding: who got treated matters")

roles = [r[0] for r in results]
unadj, adj = [], []
for role, structure, xv, yv, zv, verdict in results:
    ddd = pd.DataFrame(dict(x=xv, y=yv, z=zv))
    unadj.append(smf.ols("y ~ x", data=ddd).fit().params["x"])
    adj.append(smf.ols("y ~ x + z", data=ddd).fit().params["x"])
pos = np.arange(len(roles))
ax[2].bar(pos - 0.2, unadj, 0.4, label="unadjusted", color="steelblue")
ax[2].bar(pos + 0.2, adj, 0.4, label="adjusted", color="darkorange")
ax[2].axhline(TRUE_EFFECT, color="black", ls="--", lw=1.5, label="truth")
ax[2].set_xticks(pos); ax[2].set_xticklabels(roles, fontsize=7.5)
ax[2].set_ylabel(r"estimated $\beta_x$")
ax[2].set_title("Adjusting helps, harms, or does nothing")
ax[2].legend(fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "connecting_variables.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'connecting_variables.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Correlation, slope, and $R^2$
#
# These three numbers describe the same relationship. Work out how they relate,
# and which one answers which question.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'noise SD':<12}{'slope':>9}{'SE':>8}{'correlation r':>15}"
#       f"{'R-squared':>12}{'p':>11}")
# for noise in (0.5, 2.0, 6.0, 20.0):
#     xx = rng.uniform(0, 10, 300)
#     yy = 5 + 2.5 * xx + rng.normal(0, noise, 300)
#     f = smf.ols("yy ~ xx", data=pd.DataFrame(dict(xx=xx, yy=yy))).fit()
#     r = np.corrcoef(xx, yy)[0, 1]
#     print(f"  {noise:<12.1f}{f.params['xx']:>9.3f}{f.bse['xx']:>8.3f}"
#           f"{r:>15.3f}{f.rsquared:>12.3f}{f.pvalues['xx']:>11.2e}")
# print(f"\n  check: R-squared equals r^2 -> {r:.4f}^2 = {r**2:.4f}")
#
# # The SLOPE stays near 2.5 in every row. It is the EFFECT SIZE: how much y
# # changes per unit of x, in the units of the measurement. It does not care
# # how noisy the data is.
# #
# # The CORRELATION and R-SQUARED collapse as noise grows. They measure how
# # TIGHTLY the points hug the line - a mixture of the effect and the noise.
# #
# # So: report the SLOPE when you want to know how big the effect is, and the
# # correlation when you want to know how predictable y is from x. A huge
# # slope with r = 0.1 is a real, large effect that explains almost nothing;
# # r = 0.9 with a tiny slope is a very predictable, biologically trivial
# # relationship. Papers confuse these constantly. (Topic 10.)

# %% [markdown]
# ### Problem 2: Simpson's paradox
#
# Construct data where the overall trend points one way and the trend within
# every subgroup points the other way.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# groups, xs, ys = [], [], []
# for g, base in enumerate([0, 6, 12, 18]):
#     xg = rng.normal(base, 1.2, 150)
#     yg = 30 - 1.5 * base + (-0.8) * (xg - base) + rng.normal(0, 1.0, 150)
#     groups += [f"g{g}"] * 150; xs.append(xg); ys.append(yg)
# dp = pd.DataFrame(dict(x=np.concatenate(xs), y=np.concatenate(ys), g=groups))
#
# overall = smf.ols("y ~ x", data=dp).fit().params["x"]
# within = smf.ols("y ~ x + C(g)", data=dp).fit().params["x"]
# print(f"  ignoring the group      : slope = {overall:+.3f}")
# print(f"  adjusting for the group : slope = {within:+.3f}")
# print("\n  slope computed separately inside each group:")
# for g, sub in dp.groupby("g"):
#     print(f"    {g}: {smf.ols('y ~ x', data=sub).fit().params['x']:+.3f}")
#
# # The overall slope is POSITIVE. Every within-group slope is NEGATIVE. Both
# # are correct descriptions of the same data - they answer different
# # questions, and only one of them is usually the question you meant.
# #
# # This is Simpson's paradox, and it is confounding in its purest form: the
# # group drives both x and y, so comparing across groups mixes the two.
# #
# # It is not a curiosity. It appears whenever you pool across batches,
# # centres, cell types or donors - which in bioinformatics is always. It is
# # why Topic 19 insists on modelling the batch, and why "we combined the two
# # cohorts to increase power" deserves a hard look.

# %% [markdown]
# ### Problem 3: Build your own collider bias
#
# The classic example: among hospitalised patients, two unrelated diseases
# appear to be associated, because having either one raises your chance of
# being admitted. Simulate it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# N2 = 200_000
# # Two completely unrelated conditions in the general population.
# disease_a = rng.random(N2) < 0.10
# disease_b = rng.random(N2) < 0.10
# # Either one makes admission more likely. Admission is the COLLIDER.
# p_admit = 0.02 + 0.30 * disease_a + 0.30 * disease_b
# admitted = rng.random(N2) < p_admit
#
# def assoc(a, b):
#     """Odds ratio between two binary variables."""
#     n11 = np.sum(a & b); n10 = np.sum(a & ~b)
#     n01 = np.sum(~a & b); n00 = np.sum(~a & ~b)
#     return (n11 * n00) / max(n10 * n01, 1)
#
# print(f"  {'population studied':<34}{'n':>10}{'odds ratio A vs B':>20}")
# print(f"  {'everyone (the truth)':<34}{N2:>10,}"
#       f"{assoc(disease_a, disease_b):>20.3f}")
# print(f"  {'only hospitalised patients':<34}{admitted.sum():>10,}"
#       f"{assoc(disease_a[admitted], disease_b[admitted]):>20.3f}")
# print(f"  {'only NON-hospitalised people':<34}{(~admitted).sum():>10,}"
#       f"{assoc(disease_a[~admitted], disease_b[~admitted]):>20.3f}")
#
# # In the whole population the odds ratio is 1.0 - the diseases are unrelated,
# # exactly as constructed. Among hospitalised patients it is well below 1,
# # implying one disease PROTECTS against the other.
# #
# # The reason: if you are in hospital and you do not have disease A, you
# # probably have B - that is why you are there. Conditioning on the common
# # effect creates the association.
# #
# # This is Berkson's paradox, and it is not hypothetical. It contaminates any
# # analysis restricted to a selected group: hospital records, patients who
# # survived long enough to be biopsied, cells that passed a QC filter, samples
# # with enough RNA to sequence. Whenever your sample was SELECTED on something
# # your variables affect, ask whether you have conditioned on a collider.

# %% [markdown]
# ## What to take away
#
# 1. **Eq. (0.16) is the spine of the course.** t-test, ANOVA, regression,
#    logistic, Poisson, mixed and Cox models are all the same idea with a
#    different outcome type or a different predictor structure.
# 2. A coefficient means "change in $Y$ per unit of $X$, **holding the other
#    predictors fixed**": and that clause is the whole of confounder control.
# 3. **Four roles, opposite treatments:** adjust for confounders and precision
#    variables; do not adjust for mediators or colliders.
# 4. **The data cannot tell you which is which.** Only knowledge of how the
#    data were generated can: so draw the diagram before you fit.
# 5. **Fit statistics actively mislead here.** $R^2$, AIC and p-values all
#    endorse including a collider.
# 6. **Randomisation beats adjustment**, because it removes confounding you
#    never thought of.
#
#
# ## You have finished the foundations
#
# You now have everything you need for the main course:
#
# * what a unit, a population and a sample are (`F1`)
# * probability, independence, and conditional probability (`F2`)
# * the distributions and what mechanisms produce them (`F3`)
# * estimates, standard errors and the central limit theorem (`F4`)
# * confidence intervals and every kind of p-value (`F5`)
# * the linear model and the four causal roles (`F6`)
#
# **Go to `statsPy/core/01_study_design_and_estimands.py`.** Topic 1 is where
# the beginner material and the professional material meet, and it is the
# single most important topic in the course.
