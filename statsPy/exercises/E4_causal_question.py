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
# # Exercise 4: Answer a causal question from observational data
#
# **Curriculum link:** `stats.md` -> Topics 11, 15, 32, 34
# **Core modules used:** 11, 32, 34, 28b
#
# ## The brief
#
# A registry contains 4,000 patients. Some received drug A; the rest did not.
# You are asked: **does drug A improve the biomarker response?**
#
# You are given, for each patient: `severity` at baseline, `biomarker_0` at
# baseline, `age`, `treated`, a post-treatment `toxicity` flag, and the
# outcome `biomarker_1`.
#
# The temptation is to regress the outcome on everything available. Resist it.
# Which variables you adjust for is a **causal** decision, not a statistical
# one, and the data cannot make it for you. Adjusting for more variables is
# not safer: one of the variables above will make your answer *worse*.
#
# Work through Q1-Q5. The true treatment effect is printed in the debrief.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

MODULE_NAME = "E4_causal_question"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


TRUE_ATE = -1.50    # the estimand: average treatment effect on biomarker_1

# %% [markdown]
# ## The data
#
# The causal structure, written as a DAG:
#
# ```
#        severity -------------+
#           |                  |
#           v                  v
#        treated ---------> biomarker_1
#           |                  ^
#           v                  |
#        toxicity <---- frailty (UNMEASURED)
#
#        age --> severity      (age also affects the outcome directly)
#        biomarker_0 --> biomarker_1,  biomarker_0 --> treated
# ```
#
# * `severity` and `biomarker_0` are **confounders**: they cause both treatment
#   and outcome. Adjusting for them is required.
# * `toxicity` is a **collider**: treatment and the unmeasured `frailty` both
#   cause it. Conditioning on it opens the path
#   `treated -> toxicity <- frailty -> biomarker_1`, which was closed.

# %%
def make_registry(seed=3, n=4000, ate=TRUE_ATE, unmeasured=0.0):
    """An observational registry with confounding by indication.

    `unmeasured` adds a frailty variable that affects both treatment and
    outcome but is NOT returned - used in Q5 for sensitivity analysis.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(64, 11, n)
    # Frailty is NEVER returned. It always affects the outcome and the
    # toxicity risk; `unmeasured` additionally makes it affect TREATMENT,
    # which is what turns it into a confounder (Q5).
    frailty = rng.normal(0, 1, n)
    severity = 0.045 * (age - 64) + rng.normal(0, 1, n) + unmeasured * frailty
    biomarker_0 = 10 + 1.4 * severity + rng.normal(0, 1.2, n)
    # CONFOUNDING BY INDICATION: sicker patients are more likely to be treated.
    lp_t = -0.4 + 0.95 * severity + 0.22 * (biomarker_0 - 10) + unmeasured * frailty
    treated = rng.binomial(1, 1 / (1 + np.exp(-lp_t)))
    # The outcome depends on treatment (the estimand), the confounders, and
    # the unmeasured frailty.
    biomarker_1 = (8 + ate * treated + 1.7 * severity + 0.55 * (biomarker_0 - 10)
                   + 0.02 * (age - 64) + 1.6 * frailty
                   + rng.normal(0, 1.2, n))
    # COLLIDER: toxicity is a common effect of TREATMENT and FRAILTY, and
    # frailty also drives the outcome. Conditioning on toxicity therefore
    # opens the path  treated -> toxicity <- frailty -> biomarker_1,
    # which was closed before.
    lp_x = -1.0 + 2.0 * treated + 2.2 * frailty
    toxicity = rng.binomial(1, 1 / (1 + np.exp(-lp_x)))
    return pd.DataFrame(dict(age=age, severity=severity, biomarker_0=biomarker_0,
                             treated=treated, toxicity=toxicity,
                             biomarker_1=biomarker_1))


header("The registry")
reg = make_registry()
print(f"  n = {len(reg)}, treated = {reg.treated.sum()} "
      f"({reg.treated.mean():.1%})")
print(f"\n  {'variable':<16}{'untreated':>12}{'treated':>12}{'std. diff':>12}")
for v in ["age", "severity", "biomarker_0", "toxicity", "biomarker_1"]:
    a, b = reg.loc[reg.treated == 0, v], reg.loc[reg.treated == 1, v]
    sd = np.sqrt((a.var() + b.var()) / 2)
    print(f"  {v:<16}{a.mean():>12.3f}{b.mean():>12.3f}"
          f"{(b.mean() - a.mean()) / sd:>12.3f}")
print("\n  A standardised difference above ~0.1 marks an imbalanced covariate.")
print("  The treated patients are SICKER at baseline - that is the whole")
print("  problem, and it is visible before any modelling.")

# %% [markdown]
# ### Q1: The naive answer, and why it has the wrong sign
#
# Compute the unadjusted difference in `biomarker_1` between arms. Compare it
# to the truth (printed in the debrief; it is negative: the drug helps).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# naive = (reg.loc[reg.treated == 1, "biomarker_1"].mean()
#          - reg.loc[reg.treated == 0, "biomarker_1"].mean())
# print(f"  naive difference in means : {naive:+.3f}")
# print(f"  true ATE                  : {TRUE_ATE:+.3f}")
# print(f"  bias                      : {naive - TRUE_ATE:+.3f}")
# print(f"\n  The naive estimate has the WRONG SIGN. A drug that genuinely")
# print(f"  lowers the biomarker appears to raise it, because the patients who")
# print(f"  received it were sicker to begin with.")
#
# # This is confounding by indication, and it is the default state of every
# # observational treatment comparison in medicine. Nothing about the sample
# # size helps: with n = 4,000 the naive estimate is precisely and confidently
# # wrong. Precision is not accuracy.

# %% [markdown]
# ### Q2: Adjust for the confounders, two ways
#
# Estimate the ATE by (a) outcome regression adjusting for `severity`,
# `biomarker_0` and `age`, and (b) inverse probability of treatment weighting
# (eq. 32.4). Check covariate balance after weighting.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# CONF = ["severity", "biomarker_0", "age"]
#
# # (a) Outcome regression: the coefficient on `treated` is the ATE provided
# #     the model is right and all confounders are in it.
# m_out = smf.ols("biomarker_1 ~ treated + severity + biomarker_0 + age",
#                 data=reg).fit()
# ate_reg = m_out.params["treated"]
#
# # (b) IPTW: model treatment, then weight each patient by 1/P(their own
# #     treatment). This reconstructs a pseudo-population in which treatment is
# #     independent of the measured confounders.
# ps_model = smf.logit("treated ~ severity + biomarker_0 + age", data=reg).fit(disp=0)
# ps = ps_model.predict(reg)
# w = np.where(reg.treated == 1, 1 / ps, 1 / (1 - ps))
# # Stabilised weights: multiply by the marginal probability of the treatment
# # actually received. Same estimand, much lower variance.
# p_t = reg.treated.mean()
# w_stab = np.where(reg.treated == 1, p_t / ps, (1 - p_t) / (1 - ps))
# ate_iptw = (np.average(reg.biomarker_1[reg.treated == 1],
#                        weights=w_stab[reg.treated == 1])
#             - np.average(reg.biomarker_1[reg.treated == 0],
#                          weights=w_stab[reg.treated == 0]))
#
# print(f"  {'estimator':<34}{'ATE':>10}{'bias':>10}")
# print(f"  {'naive difference':<34}"
#       f"{reg.loc[reg.treated==1,'biomarker_1'].mean()-reg.loc[reg.treated==0,'biomarker_1'].mean():>10.3f}"
#       f"{reg.loc[reg.treated==1,'biomarker_1'].mean()-reg.loc[reg.treated==0,'biomarker_1'].mean()-TRUE_ATE:>10.3f}")
# print(f"  {'outcome regression':<34}{ate_reg:>10.3f}{ate_reg-TRUE_ATE:>10.3f}")
# print(f"  {'IPTW (stabilised)':<34}{ate_iptw:>10.3f}{ate_iptw-TRUE_ATE:>10.3f}")
# print(f"  {'TRUTH':<34}{TRUE_ATE:>10.3f}{0.0:>10.3f}")
#
# # Balance check: the standardised difference should collapse after weighting.
# print(f"\n  {'covariate':<16}{'std diff before':>18}{'std diff after IPTW':>22}")
# for v in CONF:
#     a, b = reg.loc[reg.treated == 0, v], reg.loc[reg.treated == 1, v]
#     sd = np.sqrt((a.var() + b.var()) / 2)
#     before = (b.mean() - a.mean()) / sd
#     wa = np.average(a, weights=w_stab[reg.treated == 0])
#     wb = np.average(b, weights=w_stab[reg.treated == 1])
#     print(f"  {v:<16}{before:>18.3f}{(wb - wa) / sd:>22.3f}")
# print(f"\n  weight diagnostics: max = {w_stab.max():.2f}, "
#       f"effective n = {w_stab.sum()**2 / (w_stab**2).sum():.0f} of {len(reg)}")
#
# # Both estimators recover the truth, because between them they use the SAME
# # assumption: all confounders are measured and correctly modelled. They are
# # not independent checks of each other - they are two ways of imposing one
# # assumption. What IPTW adds is the BALANCE TABLE, which is checkable: if the
# # standardised differences do not collapse, your propensity model is wrong
# # and you know it before looking at the outcome.

# %% [markdown]
# ### Q3: The collider
#
# Now add `toxicity` to the outcome regression: it is measured, it is
# strongly associated with the outcome, and adding it "controls for more".
# What happens?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# m_coll = smf.ols("biomarker_1 ~ treated + severity + biomarker_0 + age + toxicity",
#                  data=reg).fit()
# ate_coll = m_coll.params["treated"]
# # Stratifying on the collider is the same mistake in another costume.
# sub = reg[reg.toxicity == 0]
# m_strat = smf.ols("biomarker_1 ~ treated + severity + biomarker_0 + age",
#                   data=sub).fit()
# print(f"  {'model':<46}{'ATE':>10}{'bias':>10}")
# print(f"  {'correct: confounders only':<46}{ate_reg:>10.3f}{ate_reg-TRUE_ATE:>10.3f}")
# print(f"  {'+ toxicity (a COLLIDER)':<46}{ate_coll:>10.3f}{ate_coll-TRUE_ATE:>10.3f}")
# print(f"  {'restricted to toxicity == 0':<46}{m_strat.params['treated']:>10.3f}"
#       f"{m_strat.params['treated']-TRUE_ATE:>10.3f}")
# print(f"  {'TRUTH':<46}{TRUE_ATE:>10.3f}{0.0:>10.3f}")
# print(f"\n  toxicity is strongly 'significant': p = "
#       f"{m_coll.pvalues['toxicity']:.2e}, and adding it RAISES R^2 from "
#       f"{m_out.rsquared:.3f} to {m_coll.rsquared:.3f}.")
#
# # Every statistical signal says to include toxicity: it is highly
# # significant, it raises R^2, it lowers the residual variance. And it makes
# # the answer WRONG - here it overstates the benefit by about a third.
# #
# # Toxicity is a common effect of treatment and of frailty, and frailty also
# # causes the outcome. Conditioning on a common effect induces an association
# # between its causes (eq. 32.2), so the closed path
# #     treated -> toxicity <- frailty -> biomarker_1
# # is opened and its bias lands on the treatment coefficient. Restricting to
# # non-toxic patients does the same damage - slightly worse, in fact - which
# # is why "we excluded patients who experienced toxicity" is a red flag in a
# # methods section.
# #
# # NOTHING IN THE DATA tells you toxicity is a collider rather than a
# # confounder. The variable's statistics look identical either way. Only the
# # knowledge that it happened AFTER treatment, and is affected by the outcome,
# # settles it. That knowledge is causal and comes from the biology - which is
# # why you draw the DAG before you fit anything.

# %% [markdown]
# ### Q4: Doubly robust estimation
#
# AIPW (eq. 32.6) combines the outcome model and the propensity model, and is
# consistent if **either** is correct. Demonstrate that by deliberately
# misspecifying each one in turn.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def estimators(df, out_formula, ps_formula):
#     """Return (outcome regression, IPTW, AIPW) for the given specifications."""
#     mo = smf.ols(out_formula, data=df).fit()
#     d1 = df.assign(treated=1); d0 = df.assign(treated=0)
#     mu1, mu0 = mo.predict(d1), mo.predict(d0)
#     mps = smf.logit(ps_formula, data=df).fit(disp=0)
#     e = np.clip(mps.predict(df), 0.02, 0.98)
#     a, yv = df.treated.values, df.biomarker_1.values
#     ate_g = (mu1 - mu0).mean()
#     pt = a.mean()
#     ws = np.where(a == 1, pt / e, (1 - pt) / (1 - e))
#     ate_ipw = (np.average(yv[a == 1], weights=ws[a == 1])
#                - np.average(yv[a == 0], weights=ws[a == 0]))
#     # AIPW: the g-formula estimate plus a weighted residual correction.
#     aipw = ((a * (yv - mu1) / e - (1 - a) * (yv - mu0) / (1 - e)) + mu1 - mu0).mean()
#     return ate_g, ate_ipw, aipw
#
# GOOD_OUT = "biomarker_1 ~ treated + severity + biomarker_0 + age"
# BAD_OUT = "biomarker_1 ~ treated"                     # omits all confounders
# GOOD_PS = "treated ~ severity + biomarker_0 + age"
# BAD_PS = "treated ~ age"                              # omits the real drivers
#
# print(f"  {'outcome model':<16}{'PS model':<14}{'g-form':>10}{'IPTW':>10}{'AIPW':>10}")
# for on, of_ in [("correct", GOOD_OUT), ("WRONG", BAD_OUT)]:
#     for pn, pf in [("correct", GOOD_PS), ("WRONG", BAD_PS)]:
#         g_, i_, a_ = estimators(reg, of_, pf)
#         print(f"  {on:<16}{pn:<14}{g_:>10.3f}{i_:>10.3f}{a_:>10.3f}")
# print(f"  {'':<30}{'':>10}{'':>10}{'':>10}")
# print(f"  {'TRUTH':<30}{TRUE_ATE:>10.3f}{TRUE_ATE:>10.3f}{TRUE_ATE:>10.3f}")
#
# # Read the AIPW column: it lands near the truth in the first three rows and
# # fails only when BOTH models are wrong. That is the double robustness
# # property - two chances to be right instead of one.
# #
# # It is not magic. AIPW does not protect against an UNMEASURED confounder,
# # because neither model can contain a variable you do not have. It protects
# # against getting the functional form wrong, not against having the wrong
# # variable list. Q5 is about the other failure.

# %% [markdown]
# ### Q5: What would it take to explain the result away?
#
# Introduce an unmeasured confounder of increasing strength and find the point
# at which the adjusted estimate becomes meaningless. Report an E-value.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'unmeasured':<14}{'naive':>10}{'adjusted':>11}{'AIPW':>10}{'bias':>10}")
# for u in (0.0, 0.3, 0.6, 1.0, 1.6):
#     r = make_registry(seed=3, unmeasured=u)
#     nv = (r.loc[r.treated == 1, "biomarker_1"].mean()
#           - r.loc[r.treated == 0, "biomarker_1"].mean())
#     g_, i_, a_ = estimators(r, GOOD_OUT, GOOD_PS)
#     print(f"  {u:<14.1f}{nv:>10.3f}{g_:>11.3f}{a_:>10.3f}{a_-TRUE_ATE:>10.3f}")
# print(f"  {'TRUTH':<14}{'':>10}{TRUE_ATE:>11.3f}{TRUE_ATE:>10.3f}{0.0:>10.3f}")
#
# # Every method above adjusts only for what it can see. As the unmeasured
# # frailty strengthens, ALL of them drift together - the doubly robust
# # estimator no better than the others. No amount of methodological
# # sophistication substitutes for measuring the confounder.
# #
# # Since you can never rule this out, report how strong the unmeasured
# # confounding would have to be. For a risk ratio RR the E-value is
# #
# #     E = RR + sqrt(RR * (RR - 1))                       (eq. 32.8)
# #
# # the minimum association (with both treatment and outcome) an unmeasured
# # confounder would need to explain the estimate away.
# for rr in (1.2, 1.5, 2.0, 3.0):
#     print(f"    RR = {rr:.1f}  ->  E-value = {rr + np.sqrt(rr*(rr-1)):.2f}")
# print("\n  An E-value of 1.7 means a confounder would need ~1.7-fold")
# print("  associations with both treatment and outcome - plausible, so the")
# print("  finding is fragile. An E-value of 5 would be hard to dismiss.")
# print("  Report it alongside every observational effect estimate.")

# %% [markdown]
# ## Debrief

# %%
header("The generative truth")
print(f"""  TRUE average treatment effect = {TRUE_ATE:+.2f} on biomarker_1.

  Variable roles - the only thing that matters, and the only thing the data
  cannot tell you:

    severity      CONFOUNDER  causes treatment and outcome  -> ADJUST
    biomarker_0   CONFOUNDER  causes treatment and outcome  -> ADJUST
    age           CONFOUNDER  via severity, plus direct     -> ADJUST
    toxicity      COLLIDER    caused by treatment AND frailty -> DO NOT ADJUST
    frailty       UNMEASURED  causes the outcome and toxicity -> DO NOT ADJUST
                              (and, in Q5, treatment too -> then a confounder)

  The four lessons, in order of how often they are violated:

    1. The naive comparison had the WRONG SIGN. Confounding by indication is
       the default, not the exception.
    2. Adjusting for MORE variables is not safer. Adding the collider made a
       correct analysis wrong, while improving every fit statistic.
    3. Doubly robust estimation gives two chances at the functional form. It
       gives no protection at all against a variable you did not measure.
    4. Because you can never prove no unmeasured confounding, quantify how
       much would be needed. That is what an E-value is for.

  Which variables to adjust for is decided by the DAG - by what you know
  about how the data were generated - and never by stepwise selection, by
  p-values, or by which model fits best.""")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a_, lbl, col in [(0, "untreated", "steelblue"), (1, "treated", "darkorange")]:
    s = reg[reg.treated == a_]
    ax[0].scatter(s.severity, s.biomarker_1, s=5, alpha=0.3, color=col, label=lbl)
ax[0].set_xlabel("severity (confounder)"); ax[0].set_ylabel("biomarker_1")
ax[0].legend(fontsize=8); ax[0].set_title("Treated patients start sicker")
ps_m = smf.logit("treated ~ severity + biomarker_0 + age", data=reg).fit(disp=0)
psv = ps_m.predict(reg)
ax[1].hist(psv[reg.treated == 0], bins=30, alpha=0.6, color="steelblue",
           label="untreated", density=True)
ax[1].hist(psv[reg.treated == 1], bins=30, alpha=0.6, color="darkorange",
           label="treated", density=True)
ax[1].set_xlabel("propensity score"); ax[1].set_ylabel("density")
ax[1].legend(fontsize=8); ax[1].set_title("Overlap (positivity) looks adequate")
rrs = np.linspace(1.01, 4, 100)
ax[2].plot(rrs, rrs + np.sqrt(rrs * (rrs - 1)), lw=2, color="firebrick")
ax[2].set_xlabel("observed risk ratio"); ax[2].set_ylabel("E-value")
ax[2].set_title("Eq. (32.8): how much confounding would it take?")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "causal_question.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'causal_question.png')}")

# %% [markdown]
# ## What to take away
#
# 1. Draw the DAG **before** fitting. Adjustment sets come from the DAG.
# 2. Confounding by indication routinely flips the sign.
# 3. A collider is statistically indistinguishable from a confounder. Only
#    knowledge of the data-generating process separates them.
# 4. Never adjust for anything measured **after** treatment.
# 5. Doubly robust != robust to unmeasured confounding.
# 6. Report an E-value with every observational estimate.
#
# **This is the final exercise.** Return to `stats.md` for the theory, or to
# `statsPy/bioinformatics/` to apply it.
