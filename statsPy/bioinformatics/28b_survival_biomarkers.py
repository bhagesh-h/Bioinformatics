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
# # Applied 28b: Survival analysis with molecular biomarkers
#
# **Curriculum link:** `stats.md` -> Topic 28, equations (28.4)-(28.13)
# **Core modules used:** 28a (survival), 31 (validation), 32 (causal), 08
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A TCGA-style cohort: expression + clinical outcome |
# | **Assay** | Gene expression (log2 CPM) + time-to-event with censoring |
# | **Key threats** | Optimal cut-point bias, immortal time, overfitting, competing risks |
# | **Here** | Simulated with a known prognostic gene and known null genes |
#
# ## The question
#
# Is a gene **prognostic**, and would a risk score built from expression
# generalise to the next cohort?

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "28b_survival_biomarkers"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a molecular survival cohort

# %%
header("1. A TCGA-like cohort with one truly prognostic gene")

def simulate_survival_cohort(n=400, G=300, n_prog=5, beta=0.55,
                             censor_scale=9.0, seed=0):
    """Exponential survival driven by a handful of genes, plus stage and age."""
    rng = np.random.default_rng(seed)
    expr = rng.normal(0, 1, (n, G))
    # Genes are correlated in modules, as real expression is.
    for b in range(0, G, 25):
        latent = rng.normal(0, 1, n)
        expr[:, b:b+25] = 0.6 * latent[:, None] + 0.8 * expr[:, b:b+25]
    expr = (expr - expr.mean(0)) / expr.std(0)

    age = rng.normal(62, 11, n)
    stage = rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])

    prog = np.arange(n_prog)
    lp = (expr[:, prog] @ np.full(n_prog, beta)
          + 0.02 * (age - 62) + 0.45 * (stage - 1))
    baseline = 0.06
    T = rng.exponential(1 / (baseline * np.exp(lp)))
    C = rng.exponential(censor_scale, n)   # size=n is essential: without it
    #                                        numpy returns a SCALAR and every
    #                                        patient shares one censoring time.
    df = pd.DataFrame(expr, columns=[f"G{g:03d}" for g in range(G)])
    df["time"] = np.minimum(T, C)
    df["event"] = (T <= C).astype(int)
    df["age"] = age
    df["stage"] = stage
    truth = np.zeros(G, bool); truth[prog] = True
    return df, pd.Series(truth, index=[f"G{g:03d}" for g in range(G)])


d, is_prog = simulate_survival_cohort(seed=3801)
genes = [c for c in d.columns if c.startswith("G")]
print(f"  n = {len(d)}, events = {d['event'].sum()} "
      f"({d['event'].mean():.0%}), censored = {(1-d['event']).sum()}")
print(f"  {len(genes)} genes, {int(is_prog.sum())} truly prognostic: "
      f"{list(is_prog.index[is_prog])}")
print(f"  median follow-up (censored): "
      f"{d.loc[d['event']==0,'time'].median():.2f}")
print(f"\n  Effective sample size for a Cox model is the NUMBER OF EVENTS")
print(f"  ({d['event'].sum()}), not the number of patients. Guidance: >= 10")
print(f"  events per covariate -> at most {d['event'].sum()//10} covariates.")

# %% [markdown]
# ## 2. Kaplan-Meier and the at-risk table, eq. (28.6)-(28.7)

# %%
header("2. Kaplan-Meier with the at-risk table (28.6)-(28.7)")

km = KaplanMeierFitter().fit(d["time"], d["event"])
print(f"  median survival = {km.median_survival_time_:.2f}")
print(f"\n  {'time':>8}{'at risk':>10}{'S(t)':>9}{'95% CI':>22}")
for t in [1, 2, 4, 8, 12, 16]:
    n_risk = int((d["time"] >= t).sum())
    if n_risk == 0:
        continue
    s = float(km.survival_function_at_times(t).iloc[0])
    ci = km.confidence_interval_survival_function_
    idx = ci.index[ci.index <= t]
    if len(idx):
        lo, hi = ci.loc[idx[-1]].values
    else:
        lo = hi = np.nan
    print(f"  {t:>8}{n_risk:>10}{s:>9.3f}   [{lo:.3f}, {hi:.3f}]")
print("\n  Watch the at-risk column collapse. The right tail of a KM curve is")
print("  estimated from a handful of patients and looks far more precise than")
print("  it is. ALWAYS publish the number-at-risk table under the curve.")

# %% [markdown]
# ## 3. Gene-wise Cox screening, eq. (28.9)-(28.10)

# %%
header("3. Screening genes with Cox models (28.9)-(28.10)")

def cox_screen(d, genes, adjust=("age", "stage")):
    out = []
    for g in genes:
        cols = ["time", "event", g] + list(adjust)
        try:
            f = CoxPHFitter().fit(d[cols], duration_col="time", event_col="event")
            out.append((g, f.params_[g], f.summary.loc[g, "se(coef)"],
                        f.summary.loc[g, "p"]))
        except Exception:
            out.append((g, np.nan, np.nan, 1.0))
    r = pd.DataFrame(out, columns=["gene", "logHR", "se", "p"]).set_index("gene")
    r["HR"] = np.exp(r["logHR"])
    r["padj"] = st.false_discovery_control(np.nan_to_num(r["p"], nan=1.0))
    return r


res = cox_screen(d, genes)
rej = res["padj"] < 0.05
print(f"  {rej.sum()} genes at FDR 5%; "
      f"{int((rej & is_prog.reindex(res.index)).sum())} of them truly prognostic")
print(f"\n  Top 8 by p-value:")
print(f"  {'gene':>8}{'HR':>8}{'logHR':>9}{'p':>12}{'padj':>11}{'truth':>9}")
for g in res.nsmallest(8, "p").index:
    print(f"  {g:>8}{res.loc[g,'HR']:>8.3f}{res.loc[g,'logHR']:>9.3f}"
          f"{res.loc[g,'p']:>12.2e}{res.loc[g,'padj']:>11.2e}"
          f"{'PROG' if is_prog[g] else '-':>9}")

print("\n  exp(beta) is a HAZARD RATIO: a ratio of instantaneous event rates")
print("  among those still at risk. It is NOT a risk ratio, and NOT a ratio of")
print("  survival times. A HR of 1.7 does not mean patients die 1.7x sooner.")

# %% [markdown]
# ## 4. The optimal cut-point trap
#
# Dichotomising a continuous biomarker at the "most significant" threshold is a
# maximally-selected statistic with a badly inflated Type I error rate.

# %%
header("4. Optimal cut-point bias, measured")

def best_cutpoint_p(d, gene, qs=np.arange(0.2, 0.81, 0.05)):
    best_p, best_q = 1.0, None
    for q in qs:
        thr = d[gene].quantile(q)
        hi = d[gene] > thr
        if hi.sum() < 10 or (~hi).sum() < 10:
            continue
        lr = logrank_test(d.loc[~hi, "time"], d.loc[hi, "time"],
                          d.loc[~hi, "event"], d.loc[hi, "event"])
        if lr.p_value < best_p:
            best_p, best_q = lr.p_value, q
    return best_p, best_q


# IMPORTANT: pick null genes from a DIFFERENT correlation module. Genes 0-24
# form one module and contain the 5 truly prognostic genes, so their
# module-mates are genuinely (indirectly) associated with survival - they are
# not nulls. Using them would measure correlation, not the cut-point bias.
null_genes = [g for g in genes[50:] if not is_prog[g]][:120]
p_best = []
p_cont = []
for g in null_genes:
    bp, _ = best_cutpoint_p(d, g)
    p_best.append(bp)
    p_cont.append(res.loc[g, "p"])
p_best, p_cont = np.array(p_best), np.array(p_cont)
print(f"  Tested {len(null_genes)} genes from correlation modules containing NO")
print(f"  prognostic gene (so they are true nulls, directly and indirectly):")
print(f"    'best cut-point' p < 0.05      : {np.mean(p_best < 0.05):.1%}"
      f"   <- should be 5%")
print(f"    continuous Cox p < 0.05        : {np.mean(p_cont < 0.05):.1%}")
print(f"    'best cut-point' p < 0.01      : {np.mean(p_best < 0.01):.1%}")
print("\n  Searching over cut-points and reporting the best one inflates the")
print("  error rate several-fold. Keep the biomarker CONTINUOUS (with a spline")
print("  if the effect may be nonlinear), or pre-specify the cut-point and")
print("  validate it in an independent cohort.")

# The honest alternative when you must dichotomise: report the p-value from a
# permutation null of the WHOLE cut-point search.
g0 = null_genes[0]
obs_p, obs_q = best_cutpoint_p(d, g0)
rng = np.random.default_rng(5)
null_best = []
for _ in range(99):
    dp = d.copy()
    dp[g0] = rng.permutation(dp[g0].values)
    null_best.append(best_cutpoint_p(dp, g0)[0])
print(f"\n  For gene {g0}: naive best-cutpoint p = {obs_p:.4f}")
print(f"  permutation-calibrated p (search included) = "
      f"{(1 + np.sum(np.array(null_best) <= obs_p)) / 100:.3f}")

# %% [markdown]
# ## 5. Proportional hazards and RMST, eq. (28.11)-(28.12)

# %%
header("5. Checking PH, and an estimand that does not need it (28.11)-(28.12)")

top_gene = res.nsmallest(1, "p").index[0]
cph = CoxPHFitter().fit(d[["time", "event", top_gene, "age", "stage"]],
                        duration_col="time", event_col="event")
zph = proportional_hazard_test(cph, d[["time", "event", top_gene, "age", "stage"]],
                               time_transform="rank")
print(f"  PH test (scaled Schoenfeld residuals, eq. 28.11):")
print(zph.summary.round(4).to_string())

def rmst(time, event, tau):
    """stats.md eq. (28.12): area under the KM curve up to tau."""
    k = KaplanMeierFitter().fit(time, event)
    sf = k.survival_function_
    ts = sf.index.values
    ss = sf.iloc[:, 0].values
    ts = np.r_[ts[ts <= tau], tau]
    ss = np.r_[ss[:len(ts) - 1], ss[len(ts) - 2] if len(ts) > 1 else 1.0]
    return float(np.sum(ss[:-1] * np.diff(ts)) + ss[0] * ts[0])


TAU = 10.0
hi = d[top_gene] > d[top_gene].median()
r_hi = rmst(d.loc[hi, "time"], d.loc[hi, "event"], TAU)
r_lo = rmst(d.loc[~hi, "time"], d.loc[~hi, "event"], TAU)
print(f"\n  RMST up to tau = {TAU} (eq. 28.12):")
print(f"    low  {top_gene}: {r_lo:.3f}")
print(f"    high {top_gene}: {r_hi:.3f}")
print(f"    difference     : {r_lo - r_hi:+.3f} time units within {TAU}")
print("  RMST differences are directly interpretable as 'time gained' and need")
print("  NO proportional-hazards assumption - increasingly preferred when the")
print("  PH test fails or the curves cross.")

# %% [markdown]
# ## 6. Building and validating a risk score, eq. (31.3)-(31.4)

# %%
header("6. A multi-gene risk score, honestly validated")

def build_score(train, genes, p_thresh=0.01, adjust=("age", "stage")):
    """Screen on the TRAINING fold only, then fit a Cox model on the survivors."""
    r = cox_screen(train, genes, adjust=adjust)
    sel = list(r.index[r["p"] < p_thresh])
    if not sel:
        sel = list(r.nsmallest(3, "p").index)
    f = CoxPHFitter(penalizer=0.1).fit(
        train[["time", "event"] + sel + list(adjust)],
        duration_col="time", event_col="event")
    return f, sel


# (a) HONEST: nested selection + fitting inside each CV fold.
kf = KFold(5, shuffle=True, random_state=0)
c_honest, sizes = [], []
for tr, te in kf.split(d):
    f, sel = build_score(d.iloc[tr], genes)
    risk = f.predict_partial_hazard(d.iloc[te][sel + ["age", "stage"]])
    c_honest.append(concordance_index(d.iloc[te]["time"], -risk,
                                      d.iloc[te]["event"]))
    sizes.append(len(sel))

# (b) LEAKY: select genes using ALL the data, then cross-validate the fit.
r_all = cox_screen(d, genes)
sel_all = list(r_all.index[r_all["p"] < 0.01])
c_leaky = []
for tr, te in kf.split(d):
    f = CoxPHFitter(penalizer=0.1).fit(
        d.iloc[tr][["time", "event"] + sel_all + ["age", "stage"]],
        duration_col="time", event_col="event")
    risk = f.predict_partial_hazard(d.iloc[te][sel_all + ["age", "stage"]])
    c_leaky.append(concordance_index(d.iloc[te]["time"], -risk,
                                     d.iloc[te]["event"]))

# (c) Apparent performance: fit and evaluate on the same data.
f_all, _ = build_score(d, genes)
risk_all = f_all.predict_partial_hazard(d[sel_all + ["age", "stage"]]
                                        if sel_all else d[["age", "stage"]])
c_apparent = concordance_index(d["time"], -risk_all, d["event"])

# (d) Clinical-only baseline - the comparison that is usually missing.
f_clin = CoxPHFitter().fit(d[["time", "event", "age", "stage"]],
                           duration_col="time", event_col="event")
c_clin = []
for tr, te in kf.split(d):
    fc = CoxPHFitter().fit(d.iloc[tr][["time", "event", "age", "stage"]],
                           duration_col="time", event_col="event")
    rr = fc.predict_partial_hazard(d.iloc[te][["age", "stage"]])
    c_clin.append(concordance_index(d.iloc[te]["time"], -rr, d.iloc[te]["event"]))

print(f"  {'evaluation':<46}{'C-index':>10}")
print(f"  {'apparent (fit and test on all data)':<46}{c_apparent:>10.3f}")
print(f"  {'CV, genes selected on ALL data (LEAKY)':<46}{np.mean(c_leaky):>10.3f}")
print(f"  {'CV, selection INSIDE each fold (honest)':<46}{np.mean(c_honest):>10.3f}")
print(f"  {'CV, age + stage only (clinical baseline)':<46}{np.mean(c_clin):>10.3f}")
print(f"\n  genes selected per fold: {sizes}")
print(f"  optimism from the leak: "
      f"{np.mean(c_leaky) - np.mean(c_honest):+.3f}")
print(f"  gain over the clinical baseline: "
      f"{np.mean(c_honest) - np.mean(c_clin):+.3f}")
print("""
  Read the leak line carefully: the optimism is tiny. That is NOT evidence the
  leak is harmless. The 5 true genes here have a large effect and are selected
  in EVERY fold, so 'selecting on all the data' picks the same genes anyway.
  A strong signal hides the leak. Section 6b turns the signal off.

  The other lesson stands on its own: ALWAYS report the clinical-only
  baseline. A molecular score that does not beat age and stage is not a useful
  biomarker, however significant its p-value.""")

# %% [markdown]
# ## 6b. The same leak, measured where it is visible

# %%
header("6b. Measuring the selection leak on cohorts with NO prognostic gene")

# Turn the signal off (beta = 0). Any apparent skill from the gene score is
# now pure leakage, and the reference is the CLINICAL baseline rather than
# 0.5, because age and stage stay genuinely prognostic.
REP = 12
cl, ch, cc = [], [], []
for i in range(REP):
    d0, _ = simulate_survival_cohort(n=150, G=200, n_prog=0, beta=0.0,
                                     seed=3800 + i)
    g0 = [c for c in d0.columns if c.startswith("G")]
    kf0 = KFold(5, shuffle=True, random_state=100 + i)
    r0 = cox_screen(d0, g0)
    sel0 = list(r0.index[r0["p"] < 0.01]) or list(r0.nsmallest(3, "p").index)
    a, b, c = [], [], []
    for tr, te in kf0.split(d0):
        trd, ted = d0.iloc[tr], d0.iloc[te]
        # honest: reselect inside the fold
        fh, sh = build_score(trd, g0)
        a.append(concordance_index(ted["time"],
                 -fh.predict_partial_hazard(ted[sh + ["age", "stage"]]),
                 ted["event"]))
        # leaky: genes chosen on all the data
        fl = CoxPHFitter(penalizer=0.1).fit(
            trd[["time", "event"] + sel0 + ["age", "stage"]],
            duration_col="time", event_col="event")
        b.append(concordance_index(ted["time"],
                 -fl.predict_partial_hazard(ted[sel0 + ["age", "stage"]]),
                 ted["event"]))
        # the true ceiling: clinical variables only
        fc = CoxPHFitter().fit(trd[["time", "event", "age", "stage"]],
                               duration_col="time", event_col="event")
        c.append(concordance_index(ted["time"],
                 -fc.predict_partial_hazard(ted[["age", "stage"]]),
                 ted["event"]))
    ch.append(np.mean(a)); cl.append(np.mean(b)); cc.append(np.mean(c))
ch, cl, cc = np.array(ch), np.array(cl), np.array(cc)

print(f"  {REP} null cohorts (n = 150, 200 genes, NO true prognostic gene):")
print(f"  {'evaluation':<42}{'mean C':>10}{'beats clinical':>16}")
print(f"  {'CV, genes selected on ALL data (LEAKY)':<42}{cl.mean():>10.3f}"
      f"{np.mean(cl > cc):>16.0%}")
print(f"  {'CV, selection INSIDE each fold (honest)':<42}{ch.mean():>10.3f}"
      f"{np.mean(ch > cc):>16.0%}")
print(f"  {'CV, age + stage only (the true ceiling)':<42}{cc.mean():>10.3f}"
      f"{'-':>16}")
print(f"\n  leaky  - clinical baseline: {np.mean(cl - cc):+.3f}"
      f"   <- entirely spurious")
print(f"  honest - clinical baseline: {np.mean(ch - cc):+.3f}")
print("""
  Note the reference is the CLINICAL baseline, not 0.5: age and stage stay
  prognostic here, so even a gene score built from pure noise inherits their
  skill. Read the two gaps against it:

    honest CV lands BELOW the clinical baseline. Correct - the noise genes add
    parameters and no signal, so they cost precision and degrade the model. An
    honest evaluation is able to say 'these genes make it worse'.

    leaky CV lands ABOVE it. The genes were chosen while looking at the test
    fold, so the score is scored on the very data that defined it. Select
    features INSIDE the fold - the same leak as Module 31.""")

# %% [markdown]
# ## 7. Immortal time bias

# %%
header("7. Immortal time bias and the landmark fix")

rng = np.random.default_rng(3802)
d2 = d.copy()
LANDMARK = 2.0
# "Responders" can only be assessed at the landmark - so a responder must
# first SURVIVE to it. The label has NO real effect on survival.
d2["responder"] = ((d2["time"] > LANDMARK) &
                   (rng.random(len(d2)) < 0.5)).astype(int)
# A small penalizer keeps the fit stable: `responder` is 0 for everyone who
# died before the landmark, which is a near-degenerate design.
cph_bad = CoxPHFitter(penalizer=0.01).fit(
    d2[["time", "event", "responder"]], duration_col="time", event_col="event")
land = d2[d2["time"] > LANDMARK].copy()
land["time"] = land["time"] - LANDMARK
cph_land = CoxPHFitter(penalizer=0.01).fit(
    land[["time", "event", "responder"]], duration_col="time", event_col="event")
print(f"  TRUE effect of 'responder' = NONE (assigned at random)")
print(f"  naive Cox HR            = {np.exp(cph_bad.params_['responder']):.3f}"
      f"  (p = {cph_bad.summary.loc['responder','p']:.2e})   <- spurious")
print(f"  landmark analysis at t={LANDMARK}: HR = "
      f"{np.exp(cph_land.params_['responder']):.3f}"
      f"  (p = {cph_land.summary.loc['responder','p']:.3f})   <- corrected")
print("\n  The bias is structural: to be classified you must survive long")
print("  enough to be classified. Fix with a landmark analysis or a")
print("  time-dependent covariate. This is the same 'time zero' problem that")
print("  target-trial emulation is designed to prevent (Module 32).")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
for lab, m, c in [("low", ~hi, "C0"), ("high", hi, "C1")]:
    k = KaplanMeierFitter().fit(d.loc[m, "time"], d.loc[m, "event"],
                                label=f"{top_gene} {lab}")
    k.plot_survival_function(ax=ax, color=c)
lr = logrank_test(d.loc[~hi, "time"], d.loc[hi, "time"],
                  d.loc[~hi, "event"], d.loc[hi, "event"])
ax.set_title(f"KM by median split (log-rank p = {lr.p_value:.1e})", fontsize=9)
ax.set_xlabel("time"); ax.set_ylabel("S(t)")

ax = axes[0, 1]
ax.hist(p_best, bins=25, alpha=0.7, label="'best' cut-point")
ax.hist(p_cont, bins=25, alpha=0.7, label="continuous Cox")
ax.axvline(0.05, color="red", ls="--")
ax.set_xlabel("p-value (genes with NO true effect)")
ax.set_title("Optimal cut-point bias", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
vals = [c_apparent, np.mean(c_leaky), np.mean(c_honest), np.mean(c_clin)]
labs = ["apparent", "leaky CV", "honest CV", "clinical only"]
ax.bar(range(4), vals, color=["C3", "C1", "C0", "grey"])
ax.axhline(0.5, color="k", ls="--", label="chance")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7, rotation=15)
ax.set_ylabel("C-index"); ax.set_ylim(0.4, 1.0)
ax.set_title("Eq. (31.4): apparent vs validated", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ok = res["p"].notna()
ax.scatter(res.loc[ok & ~is_prog.reindex(res.index), "logHR"],
           -np.log10(res.loc[ok & ~is_prog.reindex(res.index), "p"]),
           s=8, alpha=0.4, color="grey", label="null gene")
ax.scatter(res.loc[is_prog.reindex(res.index).fillna(False), "logHR"],
           -np.log10(res.loc[is_prog.reindex(res.index).fillna(False), "p"]),
           s=40, color="C3", label="truly prognostic")
ax.set_xlabel("log hazard ratio"); ax.set_ylabel("$-\\log_{10}p$")
ax.set_title("Cox screen (eq. 28.10)", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "survival_biomarkers.png"), dpi=110,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/survival_biomarkers.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Events, not patients
#
# Re-run the Cox screen on subsets with 400, 200 and 100 patients, and separately
# on subsets chosen to have 100, 60 and 30 EVENTS. Which quantity predicts the
# power better?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_e = np.random.default_rng(9)
# print(f"  {'subset':<28}{'patients':>10}{'events':>9}{'prognostic found':>19}")
# for n_sub in [400, 200, 100]:
#     idx = rng_e.choice(len(d), n_sub, replace=False)
#     sub = d.iloc[idx]
#     r = cox_screen(sub, list(is_prog.index[is_prog]))
#     print(f"  {f'{n_sub} random patients':<28}{n_sub:>10}"
#           f"{int(sub['event'].sum()):>9}{int((r['p']<0.05).sum()):>19}")
# # Now hold PATIENTS roughly constant but vary CENSORING so events change.
# for cs in [20.0, 6.0, 2.5]:
#     dd, ip = simulate_survival_cohort(n=300, censor_scale=cs, seed=42)
#     r = cox_screen(dd, list(ip.index[ip]))
#     print(f"  {f'n=300, censor scale {cs}':<28}{300:>10}"
#           f"{int(dd['event'].sum()):>9}{int((r['p']<0.05).sum()):>19}")
#
# # Power tracks EVENTS, not patients. A large cohort with few events is a small
# # study. This is why survival trials are powered on the number of events and
# # why 'events per variable >= 10' is the standard rule of thumb.

# %% [markdown]
# ### Problem 2: Competing risks
#
# Add a competing cause of death. Compare $1-\mathrm{KM}$ for the event of
# interest against the cumulative incidence function (eq. 28.13).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_cr = np.random.default_rng(13)
# n = 900
# T1 = rng_cr.exponential(14, n)         # relapse (the event of interest)
# T2 = rng_cr.exponential(9, n)          # death from another cause
# Cc = rng_cr.exponential(30, n)
# obs = np.minimum(np.minimum(T1, T2), Cc)
# cause = np.where((T1 <= T2) & (T1 <= Cc), 1,
#                  np.where((T2 < T1) & (T2 <= Cc), 2, 0))
# t_eval = 10.0
# km_n = KaplanMeierFitter().fit(obs, (cause == 1).astype(int))
# naive = 1 - float(km_n.survival_function_at_times(t_eval).iloc[0])
# def cif(time, cause, k, t_eval):
#     o = np.argsort(time); time, cause = time[o], cause[o]
#     S, out = 1.0, 0.0
#     for ti in np.unique(time[cause > 0]):
#         n_i = np.sum(time >= ti)
#         d_k = np.sum((time == ti) & (cause == k))
#         d_all = np.sum((time == ti) & (cause > 0))
#         if ti <= t_eval:
#             out += S * d_k / n_i
#         S *= (1 - d_all / n_i)
#     return out
# truth_risk = np.mean((T1 <= t_eval) & (T1 <= T2))
# print(f"  at t = {t_eval}:")
# print(f"    TRUE probability of relapse   = {truth_risk:.4f}")
# print(f"    1 - Kaplan-Meier (naive)      = {naive:.4f}   <- overestimates")
# print(f"    cumulative incidence (28.13)  = {cif(obs, cause, 1, t_eval):.4f}")
#
# # Treating the competing death as CENSORING asks 'what would the relapse risk
# # be in a world where nobody died of anything else?' - rarely the clinical
# # question. Use the CIF for absolute risk, cause-specific hazards for
# # aetiology, and Fine-Gray for subdistribution modelling.

# %% [markdown]
# ### Problem 3: Nonlinear biomarker effects
#
# Fit the top gene with a natural spline in the Cox model and compare to the
# linear fit. Is the linear assumption adequate?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def ortho_poly(x, degree):
#     """Orthogonal polynomial basis, the equivalent of R's poly().
#
#     A raw basis (x, x^2, x^3) is severely collinear, and a spline basis such
#     as patsy's cr() or bs() sums to a constant - which a Cox model cannot
#     absorb, because it has no intercept. Either one makes the Newton-Raphson
#     step singular and lifelines raises a ConvergenceError.
#
#     Orthogonalising fixes both problems, and has a second benefit: the FIRST
#     column is the linear term, so the polynomial model strictly NESTS the
#     linear one and a likelihood-ratio test is valid.
#     """
#     z = (x - x.mean()) / x.std()
#     M = np.column_stack([z ** k for k in range(1, degree + 1)])
#     Q, _ = np.linalg.qr(M - M.mean(0))
#     return Q
#
# DEGREE = 3
# P = ortho_poly(d[top_gene].values, DEGREE)
# dp = pd.concat(
#     [d[["time", "event", "age", "stage"]].reset_index(drop=True),
#      pd.DataFrame({f"poly{k+1}": P[:, k] for k in range(DEGREE)})], axis=1)
#
# # Nested pair: the linear model keeps only poly1.
# f_lin = CoxPHFitter().fit(dp[["time", "event", "poly1", "age", "stage"]],
#                           duration_col="time", event_col="event")
# f_pol = CoxPHFitter().fit(dp, duration_col="time", event_col="event")
# lr = 2 * (f_pol.log_likelihood_ - f_lin.log_likelihood_)
# ddf = DEGREE - 1
# print(f"  linear      partial log-lik = {f_lin.log_likelihood_:.3f}")
# print(f"  polynomial  partial log-lik = {f_pol.log_likelihood_:.3f}")
# print(f"  LRT chi2 = {lr:.3f} on {ddf} df, p = {st.chi2.sf(max(lr, 0), ddf):.4f}")
#
# # The truth in this simulation IS linear, so the extra curvature terms should
# # NOT improve the fit significantly - and they do not. That is a negative
# # control for the method: a test for nonlinearity that fired on linear data
# # would be useless.
# #
# # On real data a significant LRT means the linear hazard assumption is
# # inadequate. The response is to KEEP the flexible fit and plot it, not to
# # fall back on dichotomising - which section 4 showed is far worse. A spline
# # or polynomial says "the risk rises steeply above the median and flattens
# # again"; a median split says "high versus low" and throws that away.

# %% [markdown]
# ## What to take away
#
# 1. Count **events**, not patients.
# 2. Publish the at-risk table with every KM curve.
# 3. Never dichotomise a biomarker at a data-chosen optimal cut-point.
# 4. Check PH; switch estimand to RMST when it fails.
# 5. Select features **inside** the fold, and always report the clinical-only
#    baseline.
# 6. Competing risks need the CIF, never $1-\mathrm{KM}$.
#
# **Next:** `29_gene_set_enrichment.py`
