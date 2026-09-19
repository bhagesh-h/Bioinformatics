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
# # Exercise 1: Audit a confounded, pseudoreplicated experiment
#
# **Curriculum link:** `stats.md` -> Topics 1, 7, 9, 14, 19
# **Core modules used:** 01, 07, 09, 14, 19
#
# ## The brief
#
# A collaborator sends you this result and asks you to "just check the stats":
#
# > *"We measured marker X by imaging in primary hepatocytes. Control n = 60
# > cells, treated n = 60 cells, p = 5e-9 by t-test. The effect is highly
# > significant."*
#
# On asking, you learn:
#
# * the 60 control cells came from **3 control mice**, 20 cells each;
# * the 60 treated cells came from **3 treated mice**, 20 cells each;
# * all control mice were imaged on **Monday**, all treated mice on **Tuesday**.
#
# Your job is to work out what, if anything, this experiment shows. Work
# through Q1-Q5; each has a full worked solution commented out underneath.
#
# **Do not read the solution until you have written something.** The point of
# the exercise is the reasoning, and the reasoning is what the solution explains.

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

MODULE_NAME = "E1_design_audit"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## The data
#
# Generated from a known truth so you can check your answers. The generative
# parameters are printed at the end of the file: resist looking early.

# %%
def make_experiment(seed=1, n_mice_per_arm=3, cells_per_mouse=20,
                    true_effect=0.0, day_effect=1.2, sd_mouse=0.8, sd_cell=1.0):
    """One imaging experiment with mouse-level and cell-level variation.

    `true_effect` is the real biological difference between arms; `day_effect`
    is a purely technical shift applied to whichever mice were imaged on the
    second day. In the confounded design these two are indistinguishable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for arm, label in [(0, "control"), (1, "treated")]:
        for m in range(n_mice_per_arm):
            mouse_id = f"{label[:4]}_{m + 1}"
            # Mouse-level random intercept: the biological replicate.
            mouse_mean = rng.normal(0, sd_mouse)
            day = arm            # CONFOUNDED: arm 0 -> Monday, arm 1 -> Tuesday
            for c in range(cells_per_mouse):
                y = (true_effect * arm + day_effect * day + mouse_mean
                     + rng.normal(0, sd_cell))
                rows.append(dict(mouse=mouse_id, arm=arm, condition=label,
                                 day=["Mon", "Tue"][day], cell=c, y=y))
    return pd.DataFrame(rows)


header("The data as reported")
dat = make_experiment(seed=1)
print(dat.groupby("condition").agg(cells=("y", "size"), mice=("mouse", "nunique"),
                                   mean=("y", "mean"), sd=("y", "std")).round(3))
print("\n  per mouse:")
print(dat.groupby(["condition", "day", "mouse"])["y"]
      .agg(["size", "mean"]).round(3).to_string())

naive = st.ttest_ind(dat.loc[dat.arm == 1, "y"], dat.loc[dat.arm == 0, "y"])
print(f"\n  The collaborator's test: t = {naive.statistic:.2f}, "
      f"p = {naive.pvalue:.2e}  (n = {len(dat)} cells)")

# %% [markdown]
# ### Q1: How much of that p-value is real?
#
# The cells within a mouse are not independent. Estimate the intraclass
# correlation $\rho$ (eq. 1.7), the design effect $1+(m-1)\rho$ (eq. 1.8) and
# the effective sample size (eq. 1.9). How many independent observations does
# this experiment really contain?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # Variance components from a random-intercept model fitted WITHIN arm, so the
# # between-arm difference does not leak into the mouse variance.
# vc = smf.mixedlm("y ~ C(arm)", dat, groups=dat["mouse"]).fit(reml=True)
# s2_mouse = float(vc.cov_re.iloc[0, 0])
# s2_cell = float(vc.scale)
# rho = s2_mouse / (s2_mouse + s2_cell)
# m = dat.groupby("mouse").size().mean()
# de = 1 + (m - 1) * rho
# n_eff = len(dat) / de
# print(f"  sigma^2_mouse = {s2_mouse:.3f}")
# print(f"  sigma^2_cell  = {s2_cell:.3f}")
# print(f"  ICC rho (eq. 1.7)          = {rho:.3f}")
# print(f"  design effect (eq. 1.8)    = {de:.2f}   with m = {m:.0f} cells/mouse")
# print(f"  effective n (eq. 1.9)      = {n_eff:.1f}  (not {len(dat)})")
# print(f"  independent units actually randomised = {dat.mouse.nunique()} mice")
#
# # The 120 cells carry the information of roughly n_eff independent
# # observations - and even that is optimistic, because the quantity that was
# # RANDOMISED is the mouse. There are 6 experimental units, 3 per arm. The
# # p = 3e-8 is an artefact of treating 20 measurements of the same mouse as 20
# # independent facts about the treatment.

# %% [markdown]
# ### Q2: Analyse it correctly
#
# Produce (a) a mouse-level summary t-test and (b) a mixed model with a random
# intercept for mouse. Compare the p-values and the degrees of freedom to the
# naive test. Do (a) and (b) agree, and should they?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# per_mouse = dat.groupby(["mouse", "arm"])["y"].mean().reset_index()
# tt = st.ttest_ind(per_mouse.loc[per_mouse.arm == 1, "y"],
#                   per_mouse.loc[per_mouse.arm == 0, "y"])
# mm = smf.mixedlm("y ~ C(arm)", dat, groups=dat["mouse"]).fit(reml=True)
# print(f"  {'analysis':<40}{'estimate':>10}{'p':>12}{'n units':>10}")
# print(f"  {'naive cell-level t-test (WRONG)':<40}"
#       f"{dat[dat.arm==1].y.mean()-dat[dat.arm==0].y.mean():>10.3f}"
#       f"{naive.pvalue:>12.2e}{len(dat):>10}")
# print(f"  {'mouse-level summary t-test':<40}"
#       f"{per_mouse[per_mouse.arm==1].y.mean()-per_mouse[per_mouse.arm==0].y.mean():>10.3f}"
#       f"{tt.pvalue:>12.3f}{len(per_mouse):>10}")
# print(f"  {'mixed model, random intercept':<40}"
#       f"{mm.params.iloc[1]:>10.3f}{mm.pvalues.iloc[1]:>12.3f}"
#       f"{dat.mouse.nunique():>10}")
#
# # Both give the SAME point estimate (1.252) - with a balanced design and only
# # a random intercept, the mixed model reduces to the analysis of mouse means.
# # But their p-values differ: ~0.09 for the summary t-test, ~0.03 for the mixed
# # model. The summary test is the trustworthy one.
# #
# # statsmodels' `mixedlm` reports a WALD z-test, referring the estimate to a
# # normal distribution. That is an asymptotic result valid as the number of
# # CLUSTERS grows, and here there are 6. The summary t-test correctly uses 4
# # degrees of freedom and a t reference, which has much heavier tails. With so
# # few clusters the mixed model's p-value is anticonservative - the same issue
# # Module 14 addresses with Satterthwaite or Kenward-Roger denominator degrees
# # of freedom (R's `lmerTest`; statsmodels has no equivalent).
# #
# # Rule of thumb: below ~8 clusters per arm, trust the summary analysis or use
# # KR degrees of freedom. A mixed model is not automatically the safer choice.
# #
# # Either way the honest p-value is around 0.03-0.09, not 5e-9, and it is a
# # p-value for "arm OR day", which Q3 shows cannot be separated. Averaging away
# # 20 cells per mouse loses nothing: the cells were never independent evidence
# # about the treatment.

# %% [markdown]
# ### Q3: The confound
#
# Every control mouse was imaged on Monday and every treated mouse on Tuesday.
# Fit a model containing both `arm` and `day`. What happens, and why?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# X = pd.get_dummies(dat[["arm", "day"]], columns=["day"], drop_first=True).astype(float)
# X = sm.add_constant(X)
# print("  design matrix columns:", list(X.columns))
# print(f"  rank of design matrix = {np.linalg.matrix_rank(X.values)}, "
#       f"columns = {X.shape[1]}")
# print(f"  corr(arm, day) = {np.corrcoef(dat.arm, (dat.day=='Tue').astype(int))[0,1]:.3f}")
# fit = sm.OLS(dat.y, X).fit()
# print(f"\n  OLS coefficient for day_Tue: {fit.params.get('day_Tue', float('nan'))}")
# print(f"  OLS coefficient for arm    : {fit.params['arm']:.3f}")
#
# # The design matrix is RANK DEFICIENT: `arm` and `day_Tue` are the same
# # column. statsmodels silently drops one (or returns a pseudo-inverse
# # solution) and reports a coefficient for the survivor - which is the SUM of
# # the biological and technical effects, with no way to split them.
# #
# # This is not a modelling problem with a modelling solution. No amount of
# # ComBat, no covariate, no random effect can separate two variables that are
# # perfectly collinear. The information is not in the data. The only honest
# # report is: "the arm effect and the day effect are completely confounded;
# # this experiment cannot estimate the treatment effect."

# %% [markdown]
# ### Q4: Design it properly
#
# Re-simulate with the same number of mice and cells, but with the two arms
# **balanced across both days**. Show that the confound is now separable and
# that the analysis recovers the truth. Use `true_effect=0.0`: a design that
# cannot control its false positive rate is worse than useless.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def make_balanced(seed, true_effect=0.0, day_effect=1.2,
#                   n_mice_per_arm=4, cells_per_mouse=20,
#                   sd_mouse=0.8, sd_cell=1.0):
#     """Same budget, but each arm is split across BOTH imaging days."""
#     rng = np.random.default_rng(seed)
#     rows = []
#     for arm in (0, 1):
#         for m in range(n_mice_per_arm):
#             mouse_mean = rng.normal(0, sd_mouse)
#             day = m % 2                     # arms balanced across days
#             for c in range(cells_per_mouse):
#                 rows.append(dict(mouse=f"{arm}_{m}", arm=arm, day=day,
#                                  y=true_effect*arm + day_effect*day
#                                    + mouse_mean + rng.normal(0, sd_cell)))
#     return pd.DataFrame(rows)
#
# d_bal = make_balanced(seed=2)
# Xb = sm.add_constant(d_bal[["arm", "day"]].astype(float))
# print(f"  rank = {np.linalg.matrix_rank(Xb.values)} of {Xb.shape[1]} columns "
#       f"-> identifiable")
# print(f"  corr(arm, day) = {np.corrcoef(d_bal.arm, d_bal.day)[0,1]:+.3f}")
#
# # A SINGLE experiment estimates each effect with large uncertainty (8 mice),
# # so judge the design by averaging over many replicates: an identifiable
# # design is one whose estimates are UNBIASED, not one that is right once.
# est = []
# for i in range(400):
#     db = make_balanced(seed=5000 + i)
#     pm = db.groupby(["mouse", "arm", "day"])["y"].mean().reset_index()
#     f = sm.OLS(pm.y, sm.add_constant(pm[["arm", "day"]].astype(float))).fit()
#     est.append((f.params["arm"], f.params["day"]))
# est = np.array(est)
# print(f"\n  over 400 balanced experiments:")
# print(f"    arm estimate: mean {est[:,0].mean():+.3f} (truth  0.0), "
#       f"SD {est[:,0].std():.3f}")
# print(f"    day estimate: mean {est[:,1].mean():+.3f} (truth +1.2), "
#       f"SD {est[:,1].std():.3f}")
#
# # False positive rate over many balanced experiments with NO true effect:
# fp_bal = fp_adj = fp_conf = 0
# R = 500
# for i in range(R):
#     db = make_balanced(seed=1000 + i)
#     pm = db.groupby(["mouse", "arm", "day"])["y"].mean().reset_index()
#     # (i) ignore day: unbiased, but the day effect inflates the residual SD
#     fp_bal += st.ttest_ind(pm[pm.arm == 1].y, pm[pm.arm == 0].y).pvalue < 0.05
#     # (ii) adjust for day: the nuisance variance is removed
#     fp_adj += sm.OLS(pm.y, sm.add_constant(pm[["arm", "day"]].astype(float))
#                      ).fit().pvalues["arm"] < 0.05
#     dc = make_experiment(seed=1000 + i, true_effect=0.0)
#     pc = dc.groupby(["mouse", "arm"])["y"].mean().reset_index()
#     fp_conf += st.ttest_ind(pc[pc.arm == 1].y, pc[pc.arm == 0].y).pvalue < 0.05
# print(f"\n  false positive rate, {R} experiments with NO true effect (nominal 5%):")
# print(f"    balanced, mouse-level t-test, day ignored : {fp_bal/R:5.1%}")
# print(f"    balanced, mouse-level model adjusting day : {fp_adj/R:5.1%}")
# print(f"    CONFOUNDED design, same t-test            : {fp_conf/R:5.1%}  <- day read as biology")
#
# # Balancing costs NOTHING - same mice, same cells, same money - and turns an
# # uninterpretable experiment into an interpretable one.
# #
# # Note the two balanced rows. Ignoring day is already VALID: with the arms
# # balanced across days the day effect cancels, so there is no bias and the
# # error rate is controlled (if anything conservatively, because the unmodelled
# # day effect inflates the residual variance). Adjusting for day removes that
# # nuisance variance and buys back POWER. So: balance protects validity,
# # adjustment recovers efficiency. The confounded design has neither.

# %% [markdown]
# ### Q5: What would have been enough?
#
# Given the ICC you estimated in Q1, how many **mice** per arm are needed for
# 80% power to detect a true effect of 0.8 SD? Does adding cells help?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def power_mice(n_mice, cells, effect=0.8, sd_mouse=0.8, sd_cell=1.0,
#                R=1500, seed=0):
#     """Simulated power for a mouse-level t-test - the analysis we would run."""
#     rng = np.random.default_rng(seed)
#     hits = 0
#     for _ in range(R):
#         # Variance of a mouse MEAN of `cells` cells (eq. 1.5).
#         se_mouse = np.sqrt(sd_mouse**2 + sd_cell**2 / cells)
#         a = rng.normal(0, se_mouse, n_mice)
#         b = rng.normal(effect, se_mouse, n_mice)
#         hits += st.ttest_ind(b, a).pvalue < 0.05
#     return hits / R
#
# print(f"  {'mice/arm':<12}" + "".join(f"{c:>12}" for c in (5, 20, 100, 1000)))
# print(f"  {'':<12}" + "".join(f"{'cells='+str(c):>12}" for c in (5, 20, 100, 1000)))
# for nm in (3, 5, 8, 12, 20):
#     row = "".join(f"{power_mice(nm, c, seed=nm*100+c):>11.0%} "
#                   for c in (5, 20, 100, 1000))
#     print(f"  {nm:<12}{row}")
#
# # Read ACROSS a row: multiplying cells per mouse by 200 barely moves the
# # power, because Var(mouse mean) = sigma_b^2 + sigma_e^2/m has a FLOOR at
# # sigma_b^2 = 0.64 that no number of cells can cross (eq. 1.6). Read DOWN a
# # column: mice buy power directly.
# #
# # The original experiment had 3 mice per arm and therefore about 15% power
# # for a large (0.8 SD) effect - it would miss a real effect of that size five
# # times out of six. Reaching 80% needs roughly 20 mice per arm. The study was
# # under-resourced by a factor of ~6 in the only dimension that buys power,
# # while over-sampling 20x in a dimension that buys almost none.

# %% [markdown]
# ## Debrief: the generative truth
#
# Now that you have answered, here is what actually produced the data.

# %%
header("The generative truth")
print("""  true_effect = 0.0      <- there is NO biological difference between arms
  day_effect  = 1.2      <- a purely technical shift on the second imaging day
  sd_mouse    = 0.8      <- biological variation between mice
  sd_cell     = 1.0      <- measurement/cell variation within a mouse

  The collaborator's p = 5e-9 is entirely manufactured by two errors:

    1. PSEUDOREPLICATION. 120 cells were treated as 120 independent
       observations. There were 6 independent units. (Topic 1)

    2. CONFOUNDING. The arms differ by imaging day as well as treatment, and
       the two are perfectly collinear, so the 1.2-unit day effect is read off
       as a biological effect. (Topic 19)

  Either error alone would be serious; together they turn pure noise into a
  headline result. Neither is detectable from the summary statistics the
  collaborator originally sent - you had to ask how the samples were
  collected. That question is the single highest-value thing a statistician
  does.""")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, lbl in [(0, "control (Mon)"), (1, "treated (Tue)")]:
    sub = dat[dat.arm == a]
    for j, (mouse, g) in enumerate(sub.groupby("mouse")):
        ax[0].scatter(np.full(len(g), a) + (j - 1) * 0.12,
                      g.y, s=8, alpha=0.5,
                      color=["steelblue", "darkorange"][a])
pm = dat.groupby(["mouse", "arm"])["y"].mean().reset_index()
ax[0].scatter(pm.arm, pm.y, s=110, marker="_", color="black", zorder=5,
              label="mouse means")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["control\n(Monday)", "treated\n(Tuesday)"])
ax[0].set_ylabel("marker X"); ax[0].legend(fontsize=8)
ax[0].set_title("120 cells, but 6 independent units")
cells = np.array([1, 2, 5, 10, 20, 50, 100, 500, 1000])
ax[1].plot(cells, np.sqrt(0.8**2 + 1.0**2 / cells), lw=2, color="firebrick",
           label="SE of a mouse mean")
ax[1].axhline(0.8, ls="--", color="grey", label=r"floor $\sigma_b$ (eq. 1.6)")
ax[1].set_xscale("log"); ax[1].set_xlabel("cells per mouse")
ax[1].set_ylabel("SD"); ax[1].legend(fontsize=8)
ax[1].set_title("More cells cannot cross the floor")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "design_audit.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'design_audit.png')}")

# %% [markdown]
# ## What to take away
#
# 1. Ask **what was randomised**. That is $n$. Everything else is a
#    measurement of it.
# 2. The design effect $1+(m-1)\rho$ converts a big-looking dataset into the
#    handful of independent observations it really contains.
# 3. Perfect confounding is not fixable by analysis. Balance at design time.
# 4. Replicates buy power; sub-samples buy precision on each replicate, and
#    hit a floor almost immediately.
#
# **Next:** `E2_rnaseq_end_to_end.py`
