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
# # Module 28a: Time-course, longitudinal, and survival data
#
# **Curriculum link:** `stats.md` -> Topic 28, equations (28.1)-(28.13)
#
# ## What you will learn
#
# **Part A: trajectories:** random slopes (28.1)-(28.2), the treatmentxtime
# interaction as the estimand, and separating between- from within-person
# effects (28.3).
#
# **Part B: survival:** hazard and survival functions (28.4)-(28.5),
# Kaplan-Meier with Greenwood (28.6)-(28.7), the log-rank test (28.8), Cox
# partial likelihood (28.9)-(28.10), PH diagnostics (28.11), RMST (28.12),
# competing risks (28.13), and two named biases.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "28a_survival_and_longitudinal"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# # Part A: Longitudinal trajectories
#
# ## A1. The treatment x time interaction is the estimand, eq. (28.1)-(28.2)

# %%
header("A1. Repeated visits: the treatment x time interaction (28.1)-(28.2)")

rng = np.random.default_rng(2001)
n_subj, n_visit = 30, 5
rows = []
for s in range(n_subj):
    arm = s % 2
    b0 = rng.normal(0, 1.5)           # subject intercept
    b1 = rng.normal(0, 0.25)          # subject SLOPE
    for v in range(n_visit):
        y = (12 + b0) + (0.15 + 0.35 * arm + b1) * v + rng.normal(0, 0.6)
        rows.append({"y": y, "time": v, "arm": arm, "subject": f"S{s:02d}"})
dl = pd.DataFrame(rows)

# WRONG: pool all visits as if independent.
m_wrong = smf.ols("y ~ time * arm", data=dl).fit()
# RIGHT: random intercept + random slope.
m_rs = smf.mixedlm("y ~ time * arm", data=dl, groups=dl["subject"],
                   re_formula="~time").fit()

print(f"  TRUE treatment x time interaction = 0.350")
print(f"  {'analysis':<34}{'beta(time:arm)':>16}{'SE':>9}{'p':>11}")
print(f"  {'OLS, visits pooled (WRONG)':<34}{m_wrong.params['time:arm']:>16.4f}"
      f"{m_wrong.bse['time:arm']:>9.4f}{m_wrong.pvalues['time:arm']:>11.4f}")
print(f"  {'mixed model, random slope':<34}{m_rs.params['time:arm']:>16.4f}"
      f"{m_rs.bse['time:arm']:>9.4f}{m_rs.pvalues['time:arm']:>11.4f}")
print(f"\n  A main effect of `arm` describes only the BASELINE difference")
print(f"  (here {m_rs.params['arm']:+.3f}). The scientific question 'do the")
print("  groups diverge over time?' is the INTERACTION.")

# %% [markdown]
# ## A2. Cross-sectional $\ne$ longitudinal, eq. (28.3)
#
# Separate the person-mean (between) from the within-person deviation.

# %%
header("A2. Between- vs within-person effects (28.3)")

rng = np.random.default_rng(2002)
rows = []
for s in range(60):
    # Older subjects entered the cohort healthier (a selection/cohort effect),
    # so the BETWEEN-person age slope is POSITIVE...
    base_age = rng.uniform(40, 80)
    level = 5.0 + 0.06 * base_age + rng.normal(0, 0.5)
    for v in range(4):
        age = base_age + v
        # ...while WITHIN each person the biomarker DECLINES with age.
        rows.append({"y": level - 0.10 * v + rng.normal(0, 0.25),
                     "age": age, "subject": f"S{s:02d}"})
da = pd.DataFrame(rows)
da["age_mean"] = da.groupby("subject")["age"].transform("mean")
da["age_dev"] = da["age"] - da["age_mean"]

m_naive = smf.ols("y ~ age", data=da).fit()
m_sep = smf.mixedlm("y ~ age_mean + age_dev", data=da,
                    groups=da["subject"]).fit()
print(f"  naive pooled age slope          = {m_naive.params['age']:+.4f}")
print(f"  between-person (age_mean) slope = {m_sep.params['age_mean']:+.4f}  "
      f"(truth +0.06)")
print(f"  within-person  (age_dev)  slope = {m_sep.params['age_dev']:+.4f}  "
      f"(truth -0.10)")
print("\n  The naive analysis reports a POSITIVE age effect; the biology is a")
print("  NEGATIVE within-person decline. Always decompose (28.3) in cohorts.")

# %% [markdown]
# ## A3. Nonlinear trajectories with splines

# %%
header("A3. Comparing smooth trajectories (11.13) + (11.6)")

rng = np.random.default_rng(2003)
t = np.tile(np.linspace(0, 24, 9), 24)
subj = np.repeat([f"P{i:02d}" for i in range(24)], 9)
arm = np.repeat(rng.permutation([0, 1] * 12), 9)
base = 3 + 2 * np.sin(t / 24 * 2 * np.pi)
y = base + 0.9 * arm * np.sin(t / 24 * 2 * np.pi) + \
    np.repeat(rng.normal(0, 0.4, 24), 9) + rng.normal(0, 0.3, len(t))
dt = pd.DataFrame({"y": y, "t": t, "arm": arm, "subject": subj})

m0 = smf.mixedlm("y ~ bs(t, df=5)", data=dt, groups=dt["subject"]).fit(reml=False)
m1 = smf.mixedlm("y ~ bs(t, df=5) * C(arm)", data=dt,
                 groups=dt["subject"]).fit(reml=False)
lr = 2 * (m1.llf - m0.llf)
ddf = int(m1.df_modelwc - m0.df_modelwc)
print(f"  LRT for 'different smooth trajectories': chi2 = {lr:.3f}, "
      f"df = {ddf}, p = {st.chi2.sf(lr, ddf):.3e}")
print("  NOTE: ML (reml=False) is required to compare models with DIFFERENT")
print("  FIXED EFFECTS - REML likelihoods are not comparable (Module 14).")

# %% [markdown]
# # Part B: Survival analysis
#
# ## B1. Censoring, and why it cannot be ignored

# %%
header("B1. Right-censoring: three wrong answers and one right one")

def simulate_survival(n=300, hr=1.8, censor_rate=0.9, seed=0):
    """Exponential survival with a treatment hazard ratio, plus censoring."""
    rng = np.random.default_rng(seed)
    arm = rng.binomial(1, 0.5, n)
    base_hazard = 0.08
    T = rng.exponential(1 / (base_hazard * hr**arm))
    # size=n matters here: `1/(base_hazard*censor_rate)` is a SCALAR, so
    # without it numpy returns one number and every patient would share the
    # same censoring time. (The T line is safe because hr**arm is an array.)
    C = rng.exponential(1 / (base_hazard * censor_rate), size=n)
    return pd.DataFrame({"time": np.minimum(T, C),
                         "event": (T <= C).astype(int),
                         "arm": arm, "true_T": T})


d = simulate_survival(seed=2004)
print(f"  n = {len(d)}, events = {d['event'].sum()} "
      f"({d['event'].mean():.0%}), censored = {(1-d['event']).sum()}")
print(f"\n  TRUE median survival (uncensored truth) = "
      f"{np.median(d['true_T']):.2f}")
print(f"  (a) mean observed time, ignoring censoring = {d['time'].mean():.2f}"
      f"   <- biased DOWN")
print(f"  (b) mean among events only                 = "
      f"{d.loc[d['event']==1,'time'].mean():.2f}   <- biased DOWN more")
km_all = KaplanMeierFitter().fit(d["time"], d["event"])
print(f"  (c) Kaplan-Meier median (28.6)             = "
      f"{km_all.median_survival_time_:.2f}   <- correct")
print(f"      true median of T                        = "
      f"{np.median(d['true_T']):.2f}")
print("\n  Censored subjects are NOT failures and NOT missing: they contribute")
print("  information up to their censoring time. Requires NON-INFORMATIVE")
print("  censoring: T independent of C given covariates.")

# %% [markdown]
# ## B2. Kaplan-Meier from scratch, eq. (28.6)-(28.7)

# %%
header("B2. KM estimator and Greenwood's variance (28.6)-(28.7)")

def kaplan_meier(time, event):
    """stats.md eq. (28.6)-(28.7), implemented directly."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    ts = np.unique(time[event == 1])
    S, var_sum, out = 1.0, 0.0, []
    for t_i in ts:
        n_i = np.sum(time >= t_i)                 # at risk
        d_i = np.sum((time == t_i) & (event == 1))
        S *= (1 - d_i / n_i)                      # eq. (28.6)
        var_sum += d_i / (n_i * (n_i - d_i)) if n_i > d_i else 0.0
        out.append({"t": t_i, "n_risk": n_i, "d": d_i, "S": S,
                    "se": S * np.sqrt(var_sum)})   # eq. (28.7)
    return pd.DataFrame(out)


mine = kaplan_meier(d["time"], d["event"])
lf = KaplanMeierFitter().fit(d["time"], d["event"])
# Pick three event times spread through the follow-up, whatever the length.
idx = [int(f * (len(mine) - 1)) for f in (0.1, 0.5, 0.9)]
check_t = mine["t"].iloc[idx]
print(f"  {'time':>8}{'n at risk':>11}{'S (mine)':>11}{'S (lifelines)':>15}"
      f"{'Greenwood SE':>14}")
for t_i in check_t:
    row = mine[mine["t"] == t_i].iloc[0]
    s_lf = float(lf.survival_function_at_times(t_i).iloc[0])
    print(f"  {t_i:>8.3f}{row['n_risk']:>11.0f}{row['S']:>11.5f}"
          f"{s_lf:>15.5f}{row['se']:>14.5f}")
print(f"  max |mine - lifelines| over all event times: "
      f"{np.max([abs(r['S'] - float(lf.survival_function_at_times(r['t']).iloc[0])) for _, r in mine.iterrows()]):.2e}")

print("\n  ALWAYS show the number-at-risk table. The right tail of a KM curve")
print("  is estimated from very few subjects and looks far more precise than")
print("  it is:")
for q in [0.25, 0.5, 0.75, 0.95]:
    row = mine.iloc[int(q * (len(mine) - 1))]
    print(f"    at t = {row['t']:6.2f}: {row['n_risk']:3.0f} at risk, "
          f"S = {row['S']:.3f} +/- {1.96*row['se']:.3f}")

# %% [markdown]
# ## B3. Log-rank test, eq. (28.8)

# %%
header("B3. Log-rank from scratch (28.8)")

def logrank_from_scratch(time, event, group):
    """stats.md eq. (28.8): O-E over the risk sets, hypergeometric variance."""
    time, event, group = (np.asarray(time, float), np.asarray(event, int),
                          np.asarray(group, int))
    O_minus_E, V = 0.0, 0.0
    for t_i in np.unique(time[event == 1]):
        at_risk = time >= t_i
        n_i = at_risk.sum()
        n_1i = (at_risk & (group == 1)).sum()
        d_i = ((time == t_i) & (event == 1)).sum()
        d_1i = ((time == t_i) & (event == 1) & (group == 1)).sum()
        if n_i > 1:
            O_minus_E += d_1i - d_i * n_1i / n_i
            V += (d_i * (n_i - d_i) * n_1i * (n_i - n_1i)
                  / (n_i**2 * (n_i - 1)))
    Z = O_minus_E / np.sqrt(V)
    return Z, 2 * st.norm.sf(abs(Z)), Z**2


Z, p_mine, chi2 = logrank_from_scratch(d["time"], d["event"], d["arm"])
lr_lf = logrank_test(d.loc[d["arm"] == 0, "time"], d.loc[d["arm"] == 1, "time"],
                     d.loc[d["arm"] == 0, "event"], d.loc[d["arm"] == 1, "event"])
print(f"  mine      : Z = {Z:+.4f}, chi2 = {chi2:.4f}, p = {p_mine:.3e}")
print(f"  lifelines : chi2 = {lr_lf.test_statistic:.4f}, "
      f"p = {lr_lf.p_value:.3e}")
print("\n  The log-rank is most powerful under PROPORTIONAL HAZARDS and can have")
print("  near-zero power when survival curves CROSS.")

# %% [markdown]
# ## B4. Cox model and the PH assumption, eq. (28.9)-(28.11)

# %%
header("B4. Cox proportional hazards and its diagnostic (28.9)-(28.11)")

d2 = d.copy()
d2["age"] = np.random.default_rng(7).normal(60, 10, len(d2))
cph = CoxPHFitter().fit(d2[["time", "event", "arm", "age"]],
                        duration_col="time", event_col="event")
print(cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].round(4).to_string())
print(f"\n  TRUE hazard ratio for arm = 1.80; estimated = "
      f"{np.exp(cph.params_['arm']):.3f}")
print("\n  exp(beta) is a HAZARD RATIO: a ratio of instantaneous event rates")
print("  among those still at risk. It is NOT a risk ratio and NOT a ratio of")
print("  survival times.")
print(f"\n  Effective sample size is the NUMBER OF EVENTS, not subjects:")
print(f"    subjects = {len(d2)}, events = {d2['event'].sum()}, "
       f"covariates = 2 -> {d2['event'].sum()/2:.0f} events per covariate "
       f"(guidance: >= 10)")

# PH check via scaled Schoenfeld residuals.
print("\n  Proportional-hazards check (28.11), scaled Schoenfeld residuals:")
from lifelines.statistics import proportional_hazard_test
zph = proportional_hazard_test(cph, d2[["time", "event", "arm", "age"]],
                               time_transform="rank")
print(zph.summary.round(4).to_string())
print("  Small p => the coefficient varies with time => PH is violated.")

# Now a deliberate PH violation: a time-varying effect.
print("\n  A dataset with a genuinely NON-proportional (crossing) effect:")
rng = np.random.default_rng(2005)
n = 400
arm2 = rng.binomial(1, 0.5, n)
# Treated do WORSE early (surgery-like), BETTER later.
T = np.where(arm2 == 1,
             np.where(rng.random(n) < 0.25, rng.exponential(2, n),
                      rng.exponential(25, n)),
             rng.exponential(10, n))
C = rng.exponential(30, n)
d3 = pd.DataFrame({"time": np.minimum(T, C), "event": (T <= C).astype(int),
                   "arm": arm2})
cph3 = CoxPHFitter().fit(d3, duration_col="time", event_col="event")
zph3 = proportional_hazard_test(cph3, d3, time_transform="rank")
lr3 = logrank_test(d3.loc[d3["arm"] == 0, "time"], d3.loc[d3["arm"] == 1, "time"],
                   d3.loc[d3["arm"] == 0, "event"], d3.loc[d3["arm"] == 1, "event"])
print(f"    Cox HR = {np.exp(cph3.params_['arm']):.3f}, "
      f"p = {cph3.summary.loc['arm','p']:.3f}")
print(f"    log-rank p = {lr3.p_value:.3f}   <- little power against crossing")
print(f"    PH test p  = {zph3.summary['p'].iloc[0]:.4f}   <- PH is violated")

# %% [markdown]
# ## B5. RMST: an estimand that needs no PH assumption, eq. (28.12)

# %%
header("B5. Restricted mean survival time (28.12)")

def rmst(time, event, tau):
    """stats.md eq. (28.12): integral of S(t) up to tau, from the KM curve."""
    km = kaplan_meier(time, event)
    km = km[km["t"] <= tau]
    ts = np.r_[0.0, km["t"].values, tau]
    Ss = np.r_[1.0, km["S"].values]
    widths = np.diff(ts)
    return float(np.sum(Ss * widths))


TAU = 20.0
for label, dd in [("proportional-hazards data", d), ("crossing-hazards data", d3)]:
    r0 = rmst(dd.loc[dd["arm"] == 0, "time"], dd.loc[dd["arm"] == 0, "event"], TAU)
    r1 = rmst(dd.loc[dd["arm"] == 1, "time"], dd.loc[dd["arm"] == 1, "event"], TAU)
    print(f"  {label:<28} RMST(tau={TAU:.0f}): arm0 = {r0:5.2f}, "
          f"arm1 = {r1:5.2f}, difference = {r1-r0:+6.2f}")
print("\n  RMST differences are interpretable as 'time gained within tau' and")
print("  require NO proportional-hazards assumption - increasingly preferred")
print("  when hazards are non-proportional.")

# %% [markdown]
# ## B6. Competing risks: $1-\mathrm{KM}$ overestimates, eq. (28.13)

# %%
header("B6. Competing risks (28.13)")

rng = np.random.default_rng(2006)
n = 800
T1 = rng.exponential(20, n)          # event of interest
T2 = rng.exponential(12, n)          # competing event (e.g. other-cause death)
Cc = rng.exponential(40, n)
obs = np.minimum(np.minimum(T1, T2), Cc)
cause = np.where((T1 <= T2) & (T1 <= Cc), 1,
                 np.where((T2 < T1) & (T2 <= Cc), 2, 0))

# Naive: treat the competing event as censoring, then use 1 - KM.
km_naive = KaplanMeierFitter().fit(obs, (cause == 1).astype(int))
t_eval = 15.0
naive_risk = 1 - float(km_naive.survival_function_at_times(t_eval).iloc[0])

# Correct: the cumulative incidence function (eq. 28.13).
def cif(time, cause, k, t_eval):
    """CIF_k(t) = sum over event times of S(u-) * (d_k / n_at_risk)."""
    order = np.argsort(time)
    time, cause = time[order], cause[order]
    S, out = 1.0, 0.0
    for t_i in np.unique(time[cause > 0]):
        n_i = np.sum(time >= t_i)
        d_k = np.sum((time == t_i) & (cause == k))
        d_all = np.sum((time == t_i) & (cause > 0))
        if t_i <= t_eval:
            out += S * d_k / n_i
        S *= (1 - d_all / n_i)
    return out


true_risk = np.mean((T1 <= t_eval) & (T1 <= T2))
print(f"  at t = {t_eval}:")
print(f"    TRUE probability of the event of interest = {true_risk:.4f}")
print(f"    1 - Kaplan-Meier (naive)                  = {naive_risk:.4f}"
      f"   <- OVERESTIMATES")
print(f"    cumulative incidence function (28.13)     = "
      f"{cif(obs, cause, 1, t_eval):.4f}   <- correct")
print("\n  Treating a competing event as censoring asks 'what would happen in a")
print("  world where the competing event cannot occur?' - usually not the")
print("  question. Use CIF for absolute risk; cause-specific hazards for")
print("  aetiology; Fine-Gray for subdistribution modelling.")

# %% [markdown]
# ## B7. Two named biases

# %%
header("B7. Immortal time bias and optimal cut-point bias")

# --- Immortal time bias -----------------------------------------------------
rng = np.random.default_rng(2007)
n = 600
T = rng.exponential(10, n)
Cc = rng.exponential(25, n)
time = np.minimum(T, Cc)
event = (T <= Cc).astype(int)
# "Responder" status can only be assessed at 3 months - so to be classified a
# responder you must first SURVIVE 3 months. The treatment has NO real effect.
responder = (time > 3) & (rng.random(n) < 0.5)
db = pd.DataFrame({"time": time, "event": event,
                   "responder": responder.astype(int)})
cph_bad = CoxPHFitter().fit(db, duration_col="time", event_col="event")
print(f"  TRUE effect of 'responder' status = NONE")
print(f"  naive Cox HR                      = "
      f"{np.exp(cph_bad.params_['responder']):.3f}  "
      f"(p = {cph_bad.summary.loc['responder','p']:.2e})   <- spurious")

# Landmark analysis: restrict to survivors at the landmark, then compare.
land = db[db["time"] > 3].copy()
land["time"] = land["time"] - 3
cph_land = CoxPHFitter().fit(land, duration_col="time", event_col="event")
print(f"  landmark analysis at 3 months, HR = "
      f"{np.exp(cph_land.params_['responder']):.3f}  "
      f"(p = {cph_land.summary.loc['responder','p']:.3f})   <- corrected")

# --- Optimal cut-point bias -------------------------------------------------
print("\n  Optimal cut-point bias: dichotomising a biomarker at the")
print("  'most significant' threshold, when the biomarker is PURE NOISE:")
rng = np.random.default_rng(2008)
min_ps, honest_ps = [], []
for _ in range(300):
    n = 200
    T = rng.exponential(10, n); Cc = rng.exponential(20, n)
    dd = pd.DataFrame({"time": np.minimum(T, Cc),
                       "event": (T <= Cc).astype(int),
                       "bm": rng.normal(0, 1, n)})       # NO real association
    ps = []
    for q in np.arange(0.2, 0.85, 0.05):
        dd["hi"] = (dd["bm"] > dd["bm"].quantile(q)).astype(int)
        if 5 < dd["hi"].sum() < n - 5:
            lr_ = logrank_test(dd.loc[dd["hi"] == 0, "time"],
                               dd.loc[dd["hi"] == 1, "time"],
                               dd.loc[dd["hi"] == 0, "event"],
                               dd.loc[dd["hi"] == 1, "event"])
            ps.append(lr_.p_value)
    min_ps.append(min(ps))
    honest_ps.append(CoxPHFitter().fit(dd[["time", "event", "bm"]],
                                       duration_col="time", event_col="event"
                                       ).summary.loc["bm", "p"])
print(f"    'best' cut-point p < 0.05 : {np.mean(np.array(min_ps) < 0.05):.1%}"
      f"   <- should be 5%")
print(f"    continuous biomarker p<0.05: "
      f"{np.mean(np.array(honest_ps) < 0.05):.1%}   <- calibrated")
print("  Keep the biomarker CONTINUOUS (with splines if needed), or pre-specify")
print("  the cut-point and validate it externally.")

# %% [markdown]
# ## Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
for s, grp in dl.groupby("subject"):
    ax.plot(grp["time"], grp["y"], "-", lw=0.8, alpha=0.6,
            color="C0" if grp["arm"].iloc[0] == 0 else "C1")
ax.set_xlabel("visit"); ax.set_ylabel("y")
ax.set_title("Eq. (28.1): individual trajectories", fontsize=9)

ax = axes[0, 1]
for a, c in [(0, "C0"), (1, "C1")]:
    sub = d[d["arm"] == a]
    k = KaplanMeierFitter().fit(sub["time"], sub["event"], label=f"arm {a}")
    k.plot_survival_function(ax=ax, color=c, ci_show=True)
ax.set_title(f"Eq. (28.6): KM, log-rank p = {lr_lf.p_value:.1e}", fontsize=9)
ax.set_xlabel("time"); ax.set_ylabel("S(t)")

ax = axes[1, 0]
for a, c in [(0, "C0"), (1, "C1")]:
    sub = d3[d3["arm"] == a]
    k = KaplanMeierFitter().fit(sub["time"], sub["event"], label=f"arm {a}")
    k.plot_survival_function(ax=ax, color=c, ci_show=False)
ax.set_title("Crossing curves: PH violated, log-rank powerless", fontsize=9)
ax.set_xlabel("time"); ax.set_ylabel("S(t)")

ax = axes[1, 1]
ax.hist(min_ps, bins=30, alpha=0.7, label="'optimal' cut-point")
ax.hist(honest_ps, bins=30, alpha=0.7, label="continuous biomarker")
ax.axvline(0.05, color="red", ls="--")
ax.set_xlabel("p-value under a TRUE null")
ax.set_title("Optimal cut-point bias", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "survival.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/survival.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 28)
#
# 1. Repeated measures -> mixed model with treatment x time; report the
#    interaction and separate between- from within-person effects.
# 2. Time-to-event -> Kaplan-Meier + Cox, with a PH check and an at-risk table.
# 3. Count **events**, not subjects, when judging power.
# 4. Competing risks -> CIF (28.13), never $1-\mathrm{KM}$.
# 5. Never dichotomise a biomarker at a data-chosen optimal cut-point.
#
# **Next:** `31_prediction_and_validation.py`
