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
# # Applied 45: Longitudinal and time-varying causal inference
#
# **Curriculum link:** `stats.md` -> Topic 45, equations (45.1)-(45.4)
# **Core modules used:** 14, 28a, 32, 28b
#
# ## The question
#
# Treatment is given repeatedly, and the thing that drives the next treatment
# decision is itself affected by the last one. Standard regression fails here,
# and it fails in a way no additional covariate repairs.
#
# ## The structure that breaks everything
#
# ```
#    A0 -----> L1 -----> A1 -----> Y
#     \         \                 ^
#      \         \________________/
#       \________________________/
# ```
#
# `L1` is a **confounder** of `A1 -> Y` and a **mediator** of `A0 -> Y` at the
# same time. Adjusting for it blocks part of the effect of `A0`; not adjusting
# leaves `A1` confounded. One regression cannot do both.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "45_longitudinal_causal"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(45)
TRUE_PER_PERIOD = -1.0        # effect of one period of treatment on Y

# %% [markdown]
# ## 1. A two-period study with time-varying confounding, eq. (45.1)

# %%
header("1. Simulating the structure in eq. (45.1)")


def simulate(n=6000, seed=0, effect=TRUE_PER_PERIOD):
    r = np.random.default_rng(seed)
    U = r.normal(0, 1, n)                       # stable frailty, unobserved
    L0 = r.normal(0, 1, n) + 0.8 * U            # baseline biomarker
    A0 = (r.random(n) < 1 / (1 + np.exp(-(0.9 * L0)))).astype(float)
    # L1 responds to treatment AND to frailty: mediator and confounder at once.
    L1 = 0.7 * L0 + 0.9 * U - 1.4 * A0 + r.normal(0, 1, n)
    A1 = (r.random(n) < 1 / (1 + np.exp(-(0.9 * L1)))).astype(float)
    Y = effect * (A0 + A1) + 1.1 * U + 0.6 * L1 + r.normal(0, 1, n)
    return pd.DataFrame(dict(U=U, L0=L0, A0=A0, L1=L1, A1=A1, Y=Y,
                             cum=A0 + A1))


def true_mean(a0, a1, n=400_000, seed=99, effect=TRUE_PER_PERIOD):
    """E[Y] if we INTERVENED to set A0=a0 and A1=a1 for everyone.

    This is the estimand. Note it is not simply 2 x the per-period effect:
    A0 also acts on Y through L1, so its TOTAL effect is larger than its
    direct one. Computing the truth by intervention rather than by arithmetic
    is the only way to avoid comparing an estimator against the wrong number.
    """
    r = np.random.default_rng(seed)
    U = r.normal(0, 1, n)
    L0 = r.normal(0, 1, n) + 0.8 * U
    L1 = 0.7 * L0 + 0.9 * U - 1.4 * a0 + r.normal(0, 1, n)
    Y = effect * (a0 + a1) + 1.1 * U + 0.6 * L1 + r.normal(0, 1, n)
    return Y.mean()


BASE = true_mean(0, 0)
TRUE_A0 = true_mean(1, 0) - BASE
TRUE_A1 = true_mean(0, 1) - BASE
TRUE_BOTH = true_mean(1, 1) - BASE

d = simulate(seed=1)
print(f"  n = {len(d)}, direct effect per period = {TRUE_PER_PERIOD}")
print(f"\n  TRUE effects, obtained by intervening on the simulation:")
print(f"    treat at t0 only : {TRUE_A0:+.3f}   "
      f"(direct {TRUE_PER_PERIOD:+.2f}, plus {-1.4*0.6:+.2f} through L1)")
print(f"    treat at t1 only : {TRUE_A1:+.3f}   (direct only; nothing downstream)")
print(f"    treat in both    : {TRUE_BOTH:+.3f}")
print(f"\n  The two periods do NOT have the same total effect, because only A0")
print(f"  has a mediated path. An MSM with a single 'cumulative treatment'")
print(f"  coefficient would be misspecified here.")
print(f"  treated at t0: {d.A0.mean():.1%}; treated at t1: {d.A1.mean():.1%}")
print(f"\n  L1 is caused by A0 (coefficient -1.4) and predicts A1:")
print(f"    mean L1 | A0=0 : {d.L1[d.A0 == 0].mean():+.3f}")
print(f"    mean L1 | A0=1 : {d.L1[d.A0 == 1].mean():+.3f}")
print(f"    corr(L1, A1)   : {np.corrcoef(d.L1, d.A1)[0,1]:+.3f}")
print("""
  Treatment at t0 lowers the biomarker, and a low biomarker makes treatment at
  t1 less likely. So L1 sits on the causal path from A0 to Y and simultaneously
  confounds A1.""")

# %% [markdown]
# ## 2. Why no single regression works

# %%
header("2. Adjust for L1, or do not: both are wrong")

models = {
    "Y ~ cum (no adjustment)": "Y ~ cum",
    "Y ~ cum + L0": "Y ~ cum + L0",
    "Y ~ cum + L0 + L1 (adjust the mediator)": "Y ~ cum + L0 + L1",
    "Y ~ cum + L0 + L1 + U (impossible)": "Y ~ cum + L0 + L1 + U",
}
print(f"  true effect of treatment in BOTH periods = {TRUE_BOTH:.3f}\n")
print(f"  {'model':<44}{'estimate':>11}{'bias':>9}")
for lab, f in models.items():
    b = smf.ols(f, data=d).fit().params["cum"] * 2
    print(f"  {lab:<44}{b:>11.3f}{b - TRUE_BOTH:>9.3f}")

print("""
  Not adjusting for L1 leaves A1 confounded by the biomarker. Adjusting for L1
  blocks the pathway through which A0 acts, and additionally conditions on a
  collider (L1 is caused by both A0 and U), which opens a new bias.

  Even the impossible model, which adjusts for the unobserved frailty U, does
  not recover the total effect, because it still blocks the mediated path.

  There is no covariate set that fixes this. The problem is not a missing
  variable, it is that one variable has two incompatible roles.""")

# %% [markdown]
# ## 3. Marginal structural model by IPTW, eq. (45.2)-(45.3)

# %%
header("3. Stabilised weights and the MSM (45.2)-(45.3)")


def msm_weights(d, stabilise=True, truncate=None):
    """Eq. (45.3): the product over time of P(own treatment | past) ratios."""
    # Denominator: treatment model given the full measured history.
    m0_d = smf.logit("A0 ~ L0", data=d).fit(disp=0)
    m1_d = smf.logit("A1 ~ L1 + L0 + A0", data=d).fit(disp=0)
    p0_d = m0_d.predict(d); p1_d = m1_d.predict(d)
    den = (np.where(d.A0 == 1, p0_d, 1 - p0_d) *
           np.where(d.A1 == 1, p1_d, 1 - p1_d))
    if stabilise:
        m0_n = smf.logit("A0 ~ 1", data=d).fit(disp=0)
        m1_n = smf.logit("A1 ~ A0", data=d).fit(disp=0)
        p0_n = m0_n.predict(d); p1_n = m1_n.predict(d)
        num = (np.where(d.A0 == 1, p0_n, 1 - p0_n) *
               np.where(d.A1 == 1, p1_n, 1 - p1_n))
    else:
        num = 1.0
    w = num / den
    if truncate:
        w = np.clip(w, *np.percentile(w, [truncate, 100 - truncate]))
    return w


# A single study of 6,000 is noisy, so average over replicate studies: the
# question here is whether each estimator is BIASED, which is a property of
# the recipe rather than of one dataset (Module 40).
acc = {k: [] for k in ("noadj_a0", "noadj_a1", "adj_a0", "adj_a1",
                       "msm_a0", "msm_a1")}
for i in range(25):
    di = simulate(n=6000, seed=1200 + i)
    f1 = smf.ols("Y ~ A0 + A1 + L0", data=di).fit()
    f2 = smf.ols("Y ~ A0 + A1 + L0 + L1", data=di).fit()
    wi = msm_weights(di)
    fm = sm.WLS(di.Y, sm.add_constant(di[["A0", "A1"]].to_numpy()),
                weights=wi).fit()
    acc["noadj_a0"].append(f1.params["A0"]); acc["noadj_a1"].append(f1.params["A1"])
    acc["adj_a0"].append(f2.params["A0"]);   acc["adj_a1"].append(f2.params["A1"])
    acc["msm_a0"].append(fm.params.iloc[1]); acc["msm_a1"].append(fm.params.iloc[2])
m = {k: np.mean(v) for k, v in acc.items()}

print(f"  mean over 25 replicate studies of n = 6,000\n")
print(f"  {'method':<40}{'A0 effect':>11}{'A1 effect':>11}{'both':>9}{'bias':>9}")
for lab, k in (("regression adjusting L0 only", "noadj"),
               ("regression adjusting L0 and L1", "adj"),
               ("MSM by stabilised IPTW (45.3)", "msm")):
    tot = m[f"{k}_a0"] + m[f"{k}_a1"]
    print(f"  {lab:<40}{m[f'{k}_a0']:>11.3f}{m[f'{k}_a1']:>11.3f}{tot:>9.3f}"
          f"{tot - TRUE_BOTH:>9.3f}")
print(f"  {'TRUTH (by intervention)':<40}{TRUE_A0:>11.3f}{TRUE_A1:>11.3f}"
      f"{TRUE_BOTH:>9.3f}{0.0:>9.3f}")
w = msm_weights(d)

print("""
  The MSM recovers both effects, including the larger total effect of A0 that
  runs partly through L1. The weights build a pseudo-population in which L1 no
  longer predicts A1, so the confounding is removed WITHOUT conditioning on
  the mediator, which is why the mediated path survives in the estimate.

  The regression adjusting for L1 does the opposite: it recovers roughly the
  DIRECT effect of A0 and loses the mediated part entirely.

  Note the MSM still assumes no unmeasured confounding of treatment, which
  here means it would fail if U affected treatment directly. Weighting fixes
  the structural problem, not the ignorability assumption.""")

# %% [markdown]
# ## 4. The weights are the weak point

# %%
header("4. Weight diagnostics, and why stabilisation matters")

w_un = msm_weights(d, stabilise=False)
print(f"  {'weights':<26}{'mean':>8}{'SD':>9}{'max':>10}{'ESS':>10}{'ESS %':>9}")
for lab, ww in (("unstabilised", w_un), ("stabilised (45.3)", w),
                ("stabilised + 1% trim", msm_weights(d, truncate=1))):
    ess = ww.sum() ** 2 / (ww ** 2).sum()
    print(f"  {lab:<26}{ww.mean():>8.3f}{ww.std():>9.3f}{ww.max():>10.2f}"
          f"{ess:>10.0f}{100*ess/len(d):>9.1f}")

print("""
  Stabilised weights have mean near 1 and a far smaller spread, which is why
  they are the default: same estimand, much lower variance.

  The effective sample size is the number to report. A study of 6,000 people
  whose weights leave an ESS of 2,000 has the precision of a study of 2,000,
  and if the ESS collapses the estimate is being driven by a handful of
  individuals with extreme weights.

  Truncation buys stability and introduces bias. State the rule you used and
  show the estimate with and without it.""")

# %% [markdown]
# ## 5. Positivity: the assumption that fails silently

# %%
header("5. When some people could never have been treated")

print(f"  {'treatment model strength':<30}{'min P(treat)':>14}{'max weight':>13}"
      f"{'MSM estimate':>15}")
for strength in (0.9, 2.0, 4.0):
    r = np.random.default_rng(9)
    n = 6000
    U = r.normal(0, 1, n); L0 = r.normal(0, 1, n) + 0.8 * U
    A0 = (r.random(n) < 1 / (1 + np.exp(-(strength * L0)))).astype(float)
    L1 = 0.7 * L0 + 0.9 * U - 1.4 * A0 + r.normal(0, 1, n)
    A1 = (r.random(n) < 1 / (1 + np.exp(-(strength * L1)))).astype(float)
    Y = TRUE_PER_PERIOD * (A0 + A1) + 1.1 * U + 0.6 * L1 + r.normal(0, 1, n)
    dd = pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y, cum=A0 + A1))
    ww = msm_weights(dd)
    p1 = smf.logit("A1 ~ L1 + L0 + A0", data=dd).fit(disp=0).predict(dd)
    est = sm.WLS(dd.Y, sm.add_constant(dd.cum),
                 weights=ww).fit().params.iloc[1] * 2
    print(f"  {strength:<30.1f}{min(p1.min(), 1-p1.max()):>14.4f}"
          f"{ww.max():>13.1f}{est:>15.3f}")
print(f"  {'TRUTH':<30}{'':>14}{'':>13}{TRUE_BOTH:>15.3f}")

print("""
  As the treatment becomes more deterministic, some people have almost no
  chance of receiving the treatment they did not receive. Their weights
  explode and the estimate degrades.

  This is a POSITIVITY violation, and it is the reason to inspect the
  predicted probability distribution before weighting. If a subgroup never
  receives one of the treatments, no method can tell you what would have
  happened to them; the honest move is to restrict the population and say so.""")

# %% [markdown]
# ## 6. Multi-state models, eq. (45.4)

# %%
header("6. Illness-death: more than one thing can happen (45.4)")

r = np.random.default_rng(12)
n = 4000
# Healthy -> Ill -> Dead, with direct Healthy -> Dead too.
rate_ill, rate_death_h, rate_death_i = 0.12, 0.05, 0.30
t_ill = r.exponential(1 / rate_ill, n)
t_death_h = r.exponential(1 / rate_death_h, n)
becomes_ill = t_ill < t_death_h
t_entry_ill = np.where(becomes_ill, t_ill, np.inf)
t_death = np.where(becomes_ill, t_ill + r.exponential(1 / rate_death_i, n), t_death_h)
cens = r.exponential(8.0, n)
obs = np.minimum(t_death, cens); died = t_death <= cens

print(f"  transition intensities (45.4): healthy->ill {rate_ill}, "
      f"healthy->dead {rate_death_h}, ill->dead {rate_death_i}\n")
print(f"  {'quantity':<44}{'value':>10}")
print(f"  {'became ill before dying':<44}{becomes_ill.mean():>10.1%}")
print(f"  {'died during follow-up':<44}{died.mean():>10.1%}")

# The mistake: treat illness as a fixed baseline covariate.
ever_ill = becomes_ill & (t_entry_ill <= obs)
naive = st.ttest_ind(obs[ever_ill], obs[~ever_ill])
print(f"\n  Naive comparison, 'ever ill' as a baseline group:")
print(f"    mean follow-up if ever ill    : {obs[ever_ill].mean():.2f}")
print(f"    mean follow-up if never ill   : {obs[~ever_ill].mean():.2f}")
print(f"    p = {naive.pvalue:.2e}")
print("""
  To be classified as 'ever ill' you must first survive long enough to become
  ill, so the comparison is contaminated by immortal time, exactly as in
  Module 28b. Illness is a STATE ENTERED AT A TIME, not a baseline attribute.

  The multi-state formulation keeps each transition separate: each intensity
  in (45.4) is estimated on the people actually at risk of that transition at
  that moment. Competing risks (Module 28b) is the special case where the
  states are absorbing and there is no recovery.""")

# Show the fix: time-dependent covariate.
rows = []
for i in range(n):
    if becomes_ill[i] and t_entry_ill[i] < obs[i]:
        rows.append((0, t_entry_ill[i], 0, 0))
        rows.append((t_entry_ill[i], obs[i], int(died[i]), 1))
    else:
        rows.append((0, obs[i], int(died[i]), 0))
tv = pd.DataFrame(rows, columns=["start", "stop", "event", "ill"])
tv = tv[tv.stop > tv.start]
at_risk_ill = tv.ill.sum()
ev_ill = tv.event[tv.ill == 1].sum(); ev_h = tv.event[tv.ill == 0].sum()
py_ill = (tv.stop - tv.start)[tv.ill == 1].sum()
py_h = (tv.stop - tv.start)[tv.ill == 0].sum()
print(f"  Splitting follow-up at the moment of illness:")
print(f"    death rate while healthy : {ev_h/py_h:.3f} per unit time "
      f"(true {rate_death_h})")
print(f"    death rate while ill     : {ev_ill/py_ill:.3f} per unit time "
      f"(true {rate_death_i})")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].hist(w_un, bins=60, alpha=0.6, color="firebrick", label="unstabilised")
ax[0].hist(w, bins=60, alpha=0.6, color="steelblue", label="stabilised")
ax[0].set_yscale("log"); ax[0].set_xlabel("IPTW weight")
ax[0].set_title("Eq. (45.3)"); ax[0].legend(fontsize=8)

ests = {"no adj": [], "adj L1": [], "MSM": []}
for i in range(40):
    di = simulate(n=3000, seed=2000 + i)
    ests["no adj"].append(smf.ols("Y ~ cum + L0", data=di).fit().params["cum"] * 2)
    ests["adj L1"].append(smf.ols("Y ~ cum + L0 + L1", data=di).fit().params["cum"] * 2)
    wi = msm_weights(di)
    mi = sm.WLS(di.Y, sm.add_constant(di[["A0", "A1"]].to_numpy()),
                weights=wi).fit()
    ests["MSM"].append(mi.params.iloc[1] + mi.params.iloc[2])
ax[1].boxplot([ests["no adj"], ests["adj L1"], ests["MSM"]],
              tick_labels=["no adj", "adj L1", "MSM"])
ax[1].axhline(TRUE_BOTH, color="firebrick", ls="--", lw=1.5)
ax[1].set_ylabel("estimated total effect"); ax[1].set_title("40 replicate studies")

ax[2].bar(["healthy", "ill"], [ev_h / py_h, ev_ill / py_ill], color="steelblue")
ax[2].plot([0, 1], [rate_death_h, rate_death_i], "o", color="firebrick", ms=9,
           label="truth")
ax[2].set_ylabel("death rate"); ax[2].set_title("Eq. (45.4) transition intensities")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "longitudinal_causal.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'longitudinal_causal.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: How much does the bias depend on the feedback?
#
# Vary how strongly treatment affects the biomarker, and watch the two naive
# regressions diverge from the MSM.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def sim_fb(fb, seed=0, n=5000):
#     r = np.random.default_rng(seed)
#     U = r.normal(0, 1, n); L0 = r.normal(0, 1, n) + 0.8 * U
#     A0 = (r.random(n) < 1/(1+np.exp(-0.9*L0))).astype(float)
#     L1 = 0.7*L0 + 0.9*U - fb*A0 + r.normal(0, 1, n)
#     A1 = (r.random(n) < 1/(1+np.exp(-0.9*L1))).astype(float)
#     Y = TRUE_PER_PERIOD*(A0+A1) + 1.1*U + 0.6*L1 + r.normal(0, 1, n)
#     return pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y, cum=A0+A1))
#
# print(f"  true total effect = {TRUE_BOTH}\n")
# print(f"  {'A0 -> L1 feedback':<22}{'no adj':>10}{'adj L1':>10}{'MSM':>10}")
# for fb in (0.0, 0.5, 1.4, 2.5):
#     dd = sim_fb(fb, seed=31)
#     a = smf.ols("Y ~ cum + L0", data=dd).fit().params["cum"] * 2
#     b = smf.ols("Y ~ cum + L0 + L1", data=dd).fit().params["cum"] * 2
#     c = sm.WLS(dd.Y, sm.add_constant(dd.cum),
#                weights=msm_weights(dd)).fit().params[1] * 2
#     print(f"  {fb:<22.1f}{a:>10.3f}{b:>10.3f}{c:>10.3f}")
#
# # With NO feedback (fb = 0) the problem disappears: L1 is a plain confounder,
# # adjusting for it is correct, and all three methods roughly agree. That is
# # worth seeing, because it shows g-methods are not needed everywhere.
# #
# # As the feedback grows, the two regressions move in OPPOSITE directions from
# # the truth while the MSM stays put. Opposite directions is the diagnostic
# # signature: if adjusting and not adjusting for a time-varying covariate give
# # answers that straddle rather than bracket, you have this structure.
# #
# # The condition for needing g-methods is precise: a covariate that both
# # predicts later treatment AND is affected by earlier treatment.

# %% [markdown]
# ### Problem 2: Does the MSM survive unmeasured confounding?
#
# Let the frailty U influence treatment directly and see what the MSM does.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def sim_u(u_on_a, seed=0, n=6000):
#     r = np.random.default_rng(seed)
#     U = r.normal(0, 1, n); L0 = r.normal(0, 1, n) + 0.8*U
#     A0 = (r.random(n) < 1/(1+np.exp(-(0.9*L0 + u_on_a*U)))).astype(float)
#     L1 = 0.7*L0 + 0.9*U - 1.4*A0 + r.normal(0, 1, n)
#     A1 = (r.random(n) < 1/(1+np.exp(-(0.9*L1 + u_on_a*U)))).astype(float)
#     Y = TRUE_PER_PERIOD*(A0+A1) + 1.1*U + 0.6*L1 + r.normal(0, 1, n)
#     return pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y, cum=A0+A1))
#
# print(f"  true total effect = {TRUE_BOTH}\n")
# print(f"  {'U -> treatment strength':<28}{'MSM estimate':>14}{'bias':>9}")
# for uo in (0.0, 0.4, 0.8, 1.5):
#     dd = sim_u(uo, seed=41)
#     est = sm.WLS(dd.Y, sm.add_constant(dd.cum),
#                  weights=msm_weights(dd)).fit().params[1] * 2
#     print(f"  {uo:<28.1f}{est:>14.3f}{est - TRUE_BOTH:>9.3f}")
#
# # The MSM is unbiased only in the first row. As soon as an unmeasured
# # variable influences treatment, the weights are computed from the wrong
# # model and the bias returns.
# #
# # This is the honest limitation and it mirrors Module 32 and exercise E4:
# # g-methods solve the STRUCTURAL problem of a mediator-confounder, they do
# # not solve ignorability. No weighting scheme can adjust for something you
# # did not measure.
# #
# # Report a sensitivity analysis: how strong would U have to be to move the
# # estimate to null? That is the E-value logic applied to a longitudinal
# # setting.

# %% [markdown]
# ### Problem 3: Immortal time in a multi-state setting
#
# Quantify the bias from treating a state entered during follow-up as a
# baseline attribute, and confirm the time-dependent analysis removes it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# from lifelines import CoxPHFitter
# # Naive: 'ever ill' as a baseline covariate.
# naive_df = pd.DataFrame(dict(T=obs, E=died.astype(int),
#                              ill=ever_ill.astype(int)))
# cn = CoxPHFitter().fit(naive_df, "T", "E")
# # Correct: split follow-up at the moment of illness (already built as `tv`).
# tv2 = tv.copy()
# cph = CoxPHFitter()
# cph.fit(tv2, duration_col="stop", event_col="event",
#         entry_col="start", formula="ill")
# true_hr = rate_death_i / rate_death_h
# print(f"  true hazard ratio for illness = {true_hr:.2f}\n")
# print(f"  {'analysis':<42}{'HR':>9}{'bias':>9}")
# print(f"  {'ill as a BASELINE covariate':<42}"
#       f"{np.exp(cn.params_['ill']):>9.2f}"
#       f"{np.exp(cn.params_['ill']) - true_hr:>9.2f}")
# print(f"  {'ill as a TIME-DEPENDENT covariate':<42}"
#       f"{np.exp(cph.params_['ill']):>9.2f}"
#       f"{np.exp(cph.params_['ill']) - true_hr:>9.2f}")
#
# # The baseline analysis understates the hazard dramatically, and can even
# # reverse it, because everyone in the 'ill' group is guaranteed to have
# # survived until they became ill. That immortal person-time is credited to
# # the illness state.
# #
# # Splitting the follow-up assigns each person-day to the state they were
# # actually in, and recovers the true ratio.
# #
# # The general rule, which covers Module 28b's landmark analysis and this
# # module's multi-state models: a variable measured AFTER baseline must enter
# # the model as time-dependent, or the analysis must start the clock at the
# # moment the variable is known.

# %% [markdown]
# ## What to take away
#
# 1. A covariate that is **both** affected by prior treatment and predictive of
#    later treatment (45.1) cannot be handled by any single regression.
# 2. **IPTW with stabilised weights** (45.3) removes the confounding without
#    conditioning on the mediator.
# 3. Report the **weight distribution and effective sample size**, and state
#    any truncation rule.
# 4. **Positivity** fails silently. Inspect predicted treatment probabilities.
# 5. g-methods fix structure, not ignorability. Unmeasured confounding
#    defeats them exactly as it defeats everything else.
# 6. States entered during follow-up are **time-dependent**, never baseline
#    attributes (45.4).
#
# **Next:** `46_interpretation_and_foundation_models.py`
