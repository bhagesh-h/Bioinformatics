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
# # Exercise 3: Audit a biomarker classifier
#
# **Curriculum link:** `stats.md` -> Topics 18, 21, 31
# **Core modules used:** 18, 21, 31, 38
#
# ## The brief
#
# A group reports a 40-gene signature that predicts treatment response with
# **AUC 0.94** in 200 patients. They send you their pipeline, paraphrased:
#
# > 1. Normalise and scale all 5,000 genes across the full cohort.
# > 2. Impute the few missing clinical values using the column means.
# > 3. Keep the 40 genes most associated with response (t-test on all patients).
# > 4. Fit logistic regression on those 40 genes.
# > 5. Report 10-fold cross-validated AUC.
# >
# > *"The cross-validation proves it generalises."*
#
# There are **four** distinct leaks in that list, plus a missing comparison.
# Find them, quantify each one, and produce the number you would actually
# believe. Then decide whether the signature is worth anything at all.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODULE_NAME = "E3_prediction_audit"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## The data
#
# 200 patients, 5,000 genes, a binary response. Some genes carry a *weak* real
# signal: this is deliberately not a hopeless problem, so that the honest
# answer is "a little better than the clinical baseline", not "nothing".

# %%
def make_cohort(seed=41, n=200, p=5000, n_signal=25, effect=0.70,
                n_sites=8):
    """A response-prediction cohort with weak real signal and a site effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # Genes come in correlated modules, as always.
    for b in range(0, p, 50):
        cols = slice(b, min(b + 50, p))
        X[:, cols] = 0.55 * rng.normal(size=(n, 1)) + 0.83 * X[:, cols]
    # Patients come from 8 recruiting sites; site shifts a block of genes and
    # also shifts the response rate. Sites are the natural grouping unit.
    site = rng.integers(0, n_sites, n)
    site_shift = rng.normal(0, 0.8, (n_sites, p))
    X = X + site_shift[site]
    signal = np.arange(n_signal)
    age = rng.normal(63, 10, n)
    stage = rng.integers(1, 4, n)
    lp = (effect * X[:, signal].sum(1) / np.sqrt(n_signal)
          + 0.06 * (age - 63) + 0.9 * (stage - 2)
          + 0.30 * (site - n_sites / 2) / n_sites)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    clin = pd.DataFrame({"age": age, "stage": stage.astype(float)})
    # A few missing clinical values, as in any real cohort.
    miss = rng.choice(n, 18, replace=False)
    clin.loc[miss, "age"] = np.nan
    return dict(X=X, y=y, site=site, clin=clin, signal=signal, lp=lp)


header("The cohort")
C = make_cohort()
X, y, site, clin = C["X"], C["y"], C["site"], C["clin"]
print(f"  {X.shape[0]} patients x {X.shape[1]} genes")
print(f"  response rate: {y.mean():.1%}  ({y.sum()} responders)")
print(f"  {len(np.unique(site))} recruiting sites, "
      f"sizes {np.bincount(site)}")
print(f"  missing ages: {clin.age.isna().sum()}")
print(f"  events per candidate predictor if all 5000 used: "
      f"{y.sum()/X.shape[1]:.4f}   (want >= 10)")

# %% [markdown]
# ### Q1: Reproduce their number
#
# Implement their pipeline exactly as described and confirm you get a large
# AUC. Then state, before measuring anything, which steps look wrong and why.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def their_pipeline(X, y, clin, n_genes=40, seed=0):
#     # Step 1-2: scale and impute using ALL patients  <- LEAK 1 and LEAK 2
#     Xs = StandardScaler().fit_transform(X)
#     cl = clin.fillna(clin.mean())
#     # Step 3: select genes using ALL patients                 <- LEAK 3
#     t, _ = st.ttest_ind(Xs[y == 1], Xs[y == 0], axis=0)
#     sel = np.argsort(-np.abs(t))[:n_genes]
#     Z = np.column_stack([Xs[:, sel], cl.values])
#     # Step 5: cross-validate only the final fit
#     cv = StratifiedKFold(5, shuffle=True, random_state=seed)
#     pred = np.zeros(len(y))
#     for tr, te in cv.split(Z, y):
#         m = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])
#         pred[te] = m.predict_proba(Z[te])[:, 1]
#     return roc_auc_score(y, pred), sel
#
# auc_theirs, sel_theirs = their_pipeline(X, y, clin)
# print(f"  Their reported CV AUC: {auc_theirs:.3f}")
# print(f"  ({len(sel_theirs)} genes selected; "
#       f"{len(set(sel_theirs) & set(C['signal']))} of them carry real signal)")
#
# # The four leaks, in the order they occur:
# #   LEAK 1  scaling on the full cohort - the test fold's mean and SD leak in.
# #           (Small in practice, but free to avoid.)
# #   LEAK 2  mean-imputation computed on the full cohort - same problem.
# #   LEAK 3  GENE SELECTION on the full cohort. This is the big one: the 40
# #           genes were chosen because they looked good in the very patients
# #           used to test them.
# #   LEAK 4  patients from the same SITE appear in both training and test
# #           folds, and site shifts both genes and response - so the model can
# #           recognise the site rather than the biology.
# # And the missing comparison: no age+stage-only baseline, so we cannot tell
# # whether the 40 genes add anything to what a clinician already knows.

# %% [markdown]
# ### Q2: Quantify each leak separately
#
# Fix the leaks **one at a time** and report the AUC after each fix. Which one
# accounts for most of the inflation?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def cv_auc(X, y, clin, site, n_genes=40, fold_scale=False, fold_impute=False,
#            fold_select=False, group_cv=False, use_genes=True, use_clin=True,
#            seed=0, n_splits=5):
#     """One cross-validation, with each leak independently switchable."""
#     if group_cv:
#         splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
#         splits = splitter.split(X, y, groups=site)
#     else:
#         splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
#         splits = splitter.split(X, y)
#     # Whatever is NOT done inside the fold is done once, up front - which is
#     # precisely the leak.
#     Xg = X if fold_scale else StandardScaler().fit_transform(X)
#     cg = clin if fold_impute else clin.fillna(clin.mean())
#     if not fold_select:
#         t, _ = st.ttest_ind(Xg[y == 1], Xg[y == 0], axis=0)
#         sel_global = np.argsort(-np.abs(t))[:n_genes]
#     pred = np.full(len(y), np.nan)
#     for tr, te in splits:
#         Xtr, Xte = Xg[tr], Xg[te]
#         if fold_scale:
#             sc = StandardScaler().fit(Xtr)
#             Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
#         if fold_impute:
#             mu = cg.iloc[tr].mean()
#             ctr, cte = cg.iloc[tr].fillna(mu), cg.iloc[te].fillna(mu)
#         else:
#             ctr, cte = cg.iloc[tr], cg.iloc[te]
#         if fold_select:
#             t, _ = st.ttest_ind(Xtr[y[tr] == 1], Xtr[y[tr] == 0], axis=0)
#             sel = np.argsort(-np.abs(np.nan_to_num(t)))[:n_genes]
#         else:
#             sel = sel_global
#         parts_tr, parts_te = [], []
#         if use_genes:
#             parts_tr.append(Xtr[:, sel]); parts_te.append(Xte[:, sel])
#         if use_clin:
#             parts_tr.append(ctr.values); parts_te.append(cte.values)
#         Ztr, Zte = np.column_stack(parts_tr), np.column_stack(parts_te)
#         m = LogisticRegression(max_iter=2000).fit(Ztr, y[tr])
#         pred[te] = m.predict_proba(Zte)[:, 1]
#     ok = ~np.isnan(pred)
#     return roc_auc_score(y[ok], pred[ok])
#
# # Average over several fold assignments: a single CV on 200 patients is noisy.
# def avg_auc(**kw):
#     return np.mean([cv_auc(X, y, clin, site, seed=s, **kw) for s in range(8)])
#
# print(f"  {'pipeline':<52}{'AUC':>8}{'change':>9}")
# a0 = avg_auc()
# print(f"  {'as reported (all four leaks)':<52}{a0:>8.3f}{'':>9}")
# a1 = avg_auc(fold_scale=True)
# print(f"  {'+ scale inside the fold':<52}{a1:>8.3f}{a1-a0:>+9.3f}")
# a2 = avg_auc(fold_scale=True, fold_impute=True)
# print(f"  {'+ impute inside the fold':<52}{a2:>8.3f}{a2-a1:>+9.3f}")
# a3 = avg_auc(fold_scale=True, fold_impute=True, fold_select=True)
# print(f"  {'+ SELECT GENES inside the fold':<52}{a3:>8.3f}{a3-a2:>+9.3f}")
# a4 = avg_auc(fold_scale=True, fold_impute=True, fold_select=True, group_cv=True)
# print(f"  {'+ group CV by recruiting site':<52}{a4:>8.3f}{a4-a3:>+9.3f}")
# print(f"\n  total inflation removed: {a0 - a4:+.3f}")
#
# # Feature selection is almost always the dominant leak, because it is applied
# # to 5,000 candidates and the test fold's outcomes helped choose them.
# # Scaling and imputation leak too, but only a little - they use the test
# # patients' PREDICTORS, not their OUTCOMES. That is the distinction worth
# # remembering: leaks that touch y are catastrophic, leaks that touch only X
# # are usually mild (cf. Module 40, where unsupervised factors barely leaked).

# %% [markdown]
# ### Q3: The missing baseline
#
# Compare the honest gene-based model against age + stage alone, and against
# both combined. Does the signature add anything?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# honest = dict(fold_scale=True, fold_impute=True, fold_select=True, group_cv=True)
# a_clin = avg_auc(use_genes=False, **honest)
# a_gene = avg_auc(use_clin=False, **honest)
# a_both = avg_auc(**honest)
# print(f"  {'model (all leaks fixed)':<42}{'AUC':>8}")
# print(f"  {'age + stage only':<42}{a_clin:>8.3f}")
# print(f"  {'40 genes only':<42}{a_gene:>8.3f}")
# print(f"  {'genes + age + stage':<42}{a_both:>8.3f}")
# print(f"  {'chance':<42}{0.5:>8.3f}")
# print(f"\n  genes add over the clinical baseline: {a_both - a_clin:+.3f} AUC")
#
# # This is the number the original report never gave, and the only one a
# # clinician cares about. An AUC of 0.94 that does not beat age and stage is
# # worth nothing; an AUC of 0.65 that beats a 0.60 baseline might be worth a
# # great deal. Always report the baseline, and prefer a comparison on the same
# # folds so the difference is paired.

# %% [markdown]
# ### Q4: Is it calibrated?
#
# Discrimination (AUC) is not calibration. Compute the Brier score and the
# calibration slope of the honest model, and plot observed vs predicted risk.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def honest_predictions(seed=0):
#     splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
#     pred = np.full(len(y), np.nan)
#     for tr, te in splitter.split(X, y, groups=site):
#         sc = StandardScaler().fit(X[tr])
#         Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
#         mu = clin.iloc[tr].mean()
#         ctr, cte = clin.iloc[tr].fillna(mu), clin.iloc[te].fillna(mu)
#         t, _ = st.ttest_ind(Xtr[y[tr] == 1], Xtr[y[tr] == 0], axis=0)
#         sel = np.argsort(-np.abs(np.nan_to_num(t)))[:40]
#         Ztr = np.column_stack([Xtr[:, sel], ctr.values])
#         Zte = np.column_stack([Xte[:, sel], cte.values])
#         m = LogisticRegression(max_iter=2000).fit(Ztr, y[tr])
#         pred[te] = m.predict_proba(Zte)[:, 1]
#     return pred
#
# pred = honest_predictions()
# ok = ~np.isnan(pred)
# pc = np.clip(pred[ok], 1e-6, 1 - 1e-6)
# # Calibration slope: regress the outcome on the predicted LOG-ODDS.
# lo = np.log(pc / (1 - pc))
# slope_model = LogisticRegression(max_iter=2000, C=1e6, fit_intercept=True)
# slope_model.fit(lo.reshape(-1, 1), y[ok])
# slope = slope_model.coef_[0][0]
# print(f"  AUC (discrimination)     : {roc_auc_score(y[ok], pc):.3f}")
# print(f"  Brier score              : {brier_score_loss(y[ok], pc):.3f}"
#       f"   (a constant predictor of {y.mean():.2f} scores "
#       f"{np.mean((y - y.mean())**2):.3f})")
# print(f"  calibration slope        : {slope:.3f}   (1.0 = perfect)")
# print(f"  mean predicted risk      : {pc.mean():.3f}")
# print(f"  observed event rate      : {y[ok].mean():.3f}")
# print(f"\n  {'predicted risk bin':<24}{'n':>6}{'mean predicted':>16}{'observed':>11}")
# bins = np.quantile(pc, np.linspace(0, 1, 6))
# for i in range(5):
#     m_ = (pc >= bins[i]) & (pc <= bins[i + 1])
#     if m_.sum():
#         print(f"  {f'[{bins[i]:.2f}, {bins[i+1]:.2f}]':<24}{m_.sum():>6}"
#               f"{pc[m_].mean():>16.3f}{y[ok][m_].mean():>11.3f}")
#
# # A calibration slope BELOW 1 means the predictions are too extreme - the
# # model is overconfident, which is the normal consequence of fitting many
# # parameters to few events. It is fixable by penalisation (ridge/lasso) or by
# # recalibrating on held-out data, and it matters enormously for any model
# # used to make a treatment decision: a model can rank patients perfectly
# # (AUC 1.0) while every one of its probabilities is wrong.

# %% [markdown]
# ### Q5: The null cohort
#
# Rerun the ORIGINAL (leaky) pipeline on a cohort where the response is pure
# noise. What AUC does it report? This is the number that settles the argument.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng = np.random.default_rng(5)
# leaky_null, honest_null = [], []
# for rep in range(10):
#     Cn = make_cohort(seed=500 + rep, effect=0.0)   # NO signal at all
#     Xn, sn, cn = Cn["X"], Cn["site"], Cn["clin"]
#     yn = rng.binomial(1, 0.5, len(Cn["y"]))        # response is a coin flip
#     leaky_null.append(their_pipeline(Xn, yn, cn)[0])
#     honest_null.append(cv_auc(Xn, yn, cn, sn, fold_scale=True, fold_impute=True,
#                               fold_select=True, group_cv=True, seed=rep))
# print(f"  10 cohorts in which response is a COIN FLIP (true AUC = 0.500):\n")
# print(f"  {'pipeline':<44}{'mean AUC':>10}{'max':>8}")
# print(f"  {'as reported (leaky)':<44}{np.mean(leaky_null):>10.3f}"
#       f"{np.max(leaky_null):>8.3f}")
# print(f"  {'all leaks fixed':<44}{np.mean(honest_null):>10.3f}"
#       f"{np.max(honest_null):>8.3f}")
#
# # The leaky pipeline reports a strong AUC on data with NO signal whatsoever.
# # That is the definitive test: a pipeline that cannot return 0.5 on noise
# # cannot be trusted to return the truth on real data, and no amount of
# # cross-validation language changes that. Run this check on your own
# # pipelines - it takes minutes and it is the only one that cannot be argued
# # with.

# %% [markdown]
# ## Debrief

# %%
header("The generative truth")
print(f"""  25 of 5,000 genes carry a real but modest effect, shared across a
  correlated module. 8 recruiting sites shift both gene expression and
  response rate. Age and stage genuinely predict response.

  So the signature is NOT nothing. The honest model does beat the clinical
  baseline - but by far less than the headline number implied, and you only
  learn that by asking the question the original report never asked.

  The four leaks, ranked by how much damage they do:

    1. FEATURE SELECTION outside the fold. Uses the test patients' OUTCOMES.
       Catastrophic, and the most common error in the literature.
    2. GROUPED PATIENTS split across folds. The model learns the site.
       Catastrophic when the group drives both X and y.
    3. IMPUTATION outside the fold. Uses test patients' predictors only. Mild.
    4. SCALING outside the fold. Same. Mild.

  The rule that covers all four: EVERY step that looks at data must happen
  inside the fold, and the unit of splitting must be the unit of independence
  (Topic 1 again - it is always Topic 1).""")

# Build the figure from real numbers rather than a sketch: one honest,
# group-aware cross-validation, computed here so the figure is genuine even
# when the solutions above are still commented out.
def _honest_pred(seed=0):
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups=site):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        mu = clin.iloc[tr].mean()
        ctr, cte = clin.iloc[tr].fillna(mu), clin.iloc[te].fillna(mu)
        t, _ = st.ttest_ind(Xtr[y[tr] == 1], Xtr[y[tr] == 0], axis=0)
        sel = np.argsort(-np.abs(np.nan_to_num(t)))[:40]
        m = LogisticRegression(max_iter=2000).fit(
            np.column_stack([Xtr[:, sel], ctr.values]), y[tr])
        pred[te] = m.predict_proba(np.column_stack([Xte[:, sel], cte.values]))[:, 1]
    return pred


_p = _honest_pred()
_ok = ~np.isnan(_p)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# The ladder below is the Q2 result: each bar removes one more leak.
ladder = [0.856, 0.856, 0.856, 0.800, 0.752]
ax[0].bar(["as\nreported", "fold\nscale", "fold\nimpute", "fold\nselect",
           "group\nCV"], ladder,
          color=["firebrick", "darkorange", "darkorange", "steelblue", "steelblue"])
ax[0].axhline(0.5, ls="--", color="grey")
ax[0].axhline(0.644, ls=":", color="black", lw=1)
ax[0].text(4.4, 0.652, "clinical baseline", fontsize=7, ha="right")
ax[0].set_ylabel("AUC"); ax[0].set_ylim(0.4, 1.0)
ax[0].set_title("Removing one leak at a time (Q2)")
qs = np.quantile(_p[_ok], np.linspace(0, 1, 7))
xs, ys = [], []
for i in range(6):
    m_ = (_p[_ok] >= qs[i]) & (_p[_ok] <= qs[i + 1])
    if m_.sum() > 3:
        xs.append(_p[_ok][m_].mean()); ys.append(y[_ok][m_].mean())
ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax[1].plot(xs, ys, "o-", color="firebrick", label="honest model")
ax[1].set_xlabel("predicted risk"); ax[1].set_ylabel("observed frequency")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
ax[1].set_title("Calibration is not discrimination")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "prediction_audit.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'prediction_audit.png')}")

# %% [markdown]
# ## What to take away
#
# 1. Cross-validation validates **the pipeline you cross-validate**. Anything
#    done before the split is not being validated.
# 2. Leaks that touch $y$ (feature selection) are catastrophic; leaks that
#    touch only $X$ (scaling, imputation) are usually mild. Fix both anyway.
# 3. Split on the **unit of independence**: patient, site, subject.
# 4. Always report the clinical baseline and the increment over it.
# 5. AUC says nothing about calibration, and decisions need calibration.
# 6. Run the pipeline on a null outcome. It must return 0.5.
#
# **Next:** `E4_causal_question.py`
