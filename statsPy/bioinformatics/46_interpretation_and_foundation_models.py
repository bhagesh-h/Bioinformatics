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
# # Applied 46: Model interpretation and pretrained models
#
# **Curriculum link:** `stats.md` -> Topic 46, equations (46.1)-(46.3)
# **Core modules used:** 19, 21, 28, 31
#
# ## The question
#
# The model predicts well. What is it using, can you believe the explanation,
# and what changes when the model was pretrained on data you did not control?
#
# ## The claim to keep in mind
#
# An explanation is a statement about the **model**, not about biology. A
# feature can be important to a model because it proxies batch.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

MODULE_NAME = "46_interpretation_and_foundation_models"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(46)

# %% [markdown]
# ## 1. Permutation importance, eq. (46.1)

# %%
header("1. Permutation importance on independent features (46.1)")


def perm_importance(model, X, y, metric, n_rep=8, seed=0):
    """Eq. (46.1): the loss increase when feature j is shuffled."""
    r = np.random.default_rng(seed)
    base = metric(y, model.predict(X))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        vals = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = r.permutation(Xp[:, j])
            vals.append(metric(y, model.predict(Xp)))
        out[j] = np.mean(vals) - base
    return out


n, p = 1200, 8
X = rng.normal(size=(n, p))
beta = np.array([2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
y = X @ beta + rng.normal(0, 1, n)
rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
pi = perm_importance(rf, X, y, mean_squared_error)

print(f"  {'feature':<10}{'true beta':>11}{'perm importance':>18}")
for j in range(p):
    print(f"  {j:<10}{beta[j]:>11.1f}{pi[j]:>18.4f}")
print("""
  With independent features, permutation importance recovers the truth: the
  ranking matches the coefficients and the null features sit near zero.

  Everything from here is about what happens when that independence fails,
  which in omics data it always does.""")

# %% [markdown]
# ## 2. Correlated features split the credit

# %%
header("2. Correlated features split the credit")

print("""  Case A: x1 CAUSES y, and x2 is a noisy copy of x1 with no effect of its own.
""")
print(f"  {'corr(x1,x2)':<14}{'x1 (causal)':>13}{'x2 (a copy)':>13}{'x2 share':>11}")
for rho in (0.0, 0.5, 0.8, 0.95, 0.99):
    r = np.random.default_rng(7)
    x1 = r.normal(size=n)
    x2 = rho * x1 + np.sqrt(max(1 - rho ** 2, 1e-9)) * r.normal(size=n)
    Xc = np.column_stack([x1, x2, r.normal(size=(n, 3))])
    yc = 2.0 * x1 + r.normal(0, 1, n)
    m = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                              random_state=1).fit(Xc, yc)
    p_ = perm_importance(m, Xc, yc, mean_squared_error, seed=2)
    print(f"  {rho:<14.2f}{p_[0]:>13.3f}{p_[1]:>13.3f}"
          f"{p_[1]/(p_[0]+p_[1]):>11.1%}")
print("""
  Here the leak is mild. Even at correlation 0.99, x1 keeps most of the credit,
  because it is the genuine cause and therefore splits the data slightly better
  than its copy does.

  Case B is the one that matters in omics. Neither measured feature is the
  cause: both are noisy readouts of a latent driver z, the way two co-expressed
  genes are both readouts of one transcription factor.
""")
print(f"  {'proxy noise':<14}{'corr(x1,x2)':>13}{'x1':>9}{'x2':>9}{'x1 share':>11}")
for tau in (2.0, 1.0, 0.5, 0.2, 0.05):
    r = np.random.default_rng(11)
    z = r.normal(size=n)
    x1 = z + r.normal(0, tau, n)
    x2 = z + r.normal(0, tau, n)
    Xc = np.column_stack([x1, x2, r.normal(size=(n, 3))])
    yc = 2.0 * z + r.normal(0, 1, n)
    m = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                              random_state=1).fit(Xc, yc)
    p_ = perm_importance(m, Xc, yc, mean_squared_error, seed=2)
    print(f"  {tau:<14.2f}{np.corrcoef(x1, x2)[0,1]:>13.2f}{p_[0]:>9.3f}"
          f"{p_[1]:>9.3f}{p_[0]/(p_[0]+p_[1]):>11.1%}")
print("""
  The credit splits almost exactly in half at every noise level, and each
  feature alone looks about half as important as the pair really is.

  Two consequences. A gene downstream of a real driver can look unimportant
  because a co-expressed gene absorbs half its credit. And "gene X was the top
  feature" is close to a coin flip between the two: Problem 2 measures how
  close.

  Report importance for CORRELATED GROUPS, not individual features, or cluster
  the features first and permute whole clusters.""")

header("2b. The split also depends on a hyperparameter, not on biology")

print(f"  {'max_features':<16}{'x1 (causal)':>13}{'x2 (a copy)':>13}{'x2 share':>11}")
for mf in (1.0, "sqrt"):
    r = np.random.default_rng(7)
    x1 = r.normal(size=n)
    x2 = 0.95 * x1 + np.sqrt(1 - 0.95 ** 2) * r.normal(size=n)
    Xc = np.column_stack([x1, x2, r.normal(size=(n, 3))])
    yc = 2.0 * x1 + r.normal(0, 1, n)
    m = RandomForestRegressor(n_estimators=300, max_features=mf,
                              random_state=1).fit(Xc, yc)
    p_ = perm_importance(m, Xc, yc, mean_squared_error, seed=2)
    print(f"  {str(mf):<16}{p_[0]:>13.3f}{p_[1]:>13.3f}"
          f"{p_[1]/(p_[0]+p_[1]):>11.1%}")
print("""
  Same data, same correlation, same true effect. When every split considers all
  features the forest keeps choosing x1, so x2 looks unimportant. When each
  split sees a random subset, x2 gets used whenever x1 is absent and its
  importance rises several-fold.

  The share of credit given to a feature with no effect is therefore partly a
  function of a tuning parameter. Always state which model and which settings
  produced an importance ranking.""")

# %% [markdown]
# ## 3. Permutation evaluates the model off the data manifold

# %%
header("3. Shuffling creates samples that could never exist")

rho = 0.95
z = rng.normal(size=n)
x1 = z
x2 = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
Xm = np.column_stack([x1, x2])
ym = (x1 + x2 > 0).astype(int)
clf = RandomForestClassifier(n_estimators=250, random_state=2).fit(Xm, ym)

r2 = np.random.default_rng(5)
Xperm = Xm.copy(); Xperm[:, 1] = r2.permutation(Xperm[:, 1])
print(f"  corr(x1, x2) in the real data      : {np.corrcoef(Xm[:,0], Xm[:,1])[0,1]:.3f}")
print(f"  corr(x1, x2) after permuting x2    : {np.corrcoef(Xperm[:,0], Xperm[:,1])[0,1]:.3f}")
d_real = np.abs(Xm[:, 0] - Xm[:, 1])
d_perm = np.abs(Xperm[:, 0] - Xperm[:, 1])
print(f"\n  |x1 - x2| in real data  : median {np.median(d_real):.2f}, "
      f"99th pct {np.quantile(d_real, 0.99):.2f}")
print(f"  |x1 - x2| after permuting: median {np.median(d_perm):.2f}, "
      f"99th pct {np.quantile(d_perm, 0.99):.2f}")
print(f"\n  fraction of permuted rows outside the real data's 99th percentile: "
      f"{np.mean(d_perm > np.quantile(d_real, 0.99)):.1%}")

print("""
  Permuting breaks the correlation, so a large share of the rows the model is
  asked to score are combinations that never occur in nature: a high x1 with a
  low x2 when the two are 95% correlated.

  The importance you get is therefore partly a measurement of the model's
  behaviour in a region where it was never trained and where its output is
  arbitrary. Conditional or grouped permutation schemes exist for this reason.""")

# %% [markdown]
# ## 4. Shapley values, eq. (46.2)-(46.3)

# %%
header("4. Exact Shapley values on a small model (46.2)-(46.3)")


def shapley_exact(predict, x, background, features):
    """Eq. (46.2) by full enumeration. Feasible only for a handful of
    features, which is why real implementations approximate."""
    from itertools import permutations
    import math
    k = len(features)
    phi = np.zeros(k)
    for order in permutations(range(k)):
        present = []
        prev = predict(background.copy())
        for j in order:
            present.append(j)
            xx = background.copy()
            xx[:, present] = x[present]
            cur = predict(xx)
            phi[j] += cur - prev
            prev = cur
    return phi / math.factorial(k)


Xs = rng.normal(size=(800, 4))
ys = 1.5 * Xs[:, 0] - 1.0 * Xs[:, 1] + 0.0 * Xs[:, 2] + 0.5 * Xs[:, 3] + \
     rng.normal(0, 0.5, 800)
ridge = Ridge(alpha=1.0).fit(Xs, ys)
bg = Xs.mean(0, keepdims=True)
target = Xs[0]
phi = shapley_exact(lambda A: ridge.predict(A)[0], target, bg, range(4))

print(f"  {'feature':<10}{'coefficient':>13}{'x value':>10}{'Shapley phi':>14}")
for j in range(4):
    print(f"  {j:<10}{ridge.coef_[j]:>13.3f}{target[j]:>10.3f}{phi[j]:>14.4f}")
print(f"\n  additivity check, eq. (46.3):")
print(f"    baseline + sum(phi) = {ridge.predict(bg)[0] + phi.sum():.4f}")
print(f"    model prediction    = {ridge.predict(target.reshape(1, -1))[0]:.4f}")
print("""
  For a linear model the Shapley value of feature j is simply
  coefficient x (value minus baseline), and the additivity property (46.3)
  holds exactly.

  That transparency is why Shapley values are attractive. It is also why they
  are over-trusted: additivity is a mathematical property of the attribution,
  not evidence that the model captured a real mechanism.""")

# %% [markdown]
# ## 5. An explanation can be entirely about batch

# %%
header("5. The most important feature is a batch proxy")

n_s, n_g = 600, 40
r5 = np.random.default_rng(21)
y_out = r5.integers(0, 2, n_s)
# A confounded design: 85% of cases were processed in batch 1, 85% of
# controls in batch 0. Nobody planned it; it is how the samples arrived.
FRAC = 0.85
batch = np.where(r5.random(n_s) < np.where(y_out == 1, FRAC, 1 - FRAC), 1, 0)
bio = r5.normal(0, 1, n_s) + 1.0 * y_out       # a genuinely causal gene
G = r5.normal(0, 1, (n_s, n_g))
G[:, 0] = bio
G[:, 1] = 3.0 * batch + r5.normal(0, 0.3, n_s)  # a purely batch-driven gene
Xg = G

clf2 = RandomForestClassifier(n_estimators=300, random_state=3).fit(Xg, y_out)
pi2 = perm_importance(clf2, Xg, y_out,
                      lambda a, b: -roc_auc_score(a, b), seed=7)
top = np.argsort(-pi2)[:4]
print(f"  confounding: corr(batch, outcome) = "
      f"{np.corrcoef(batch, y_out)[0,1]:.2f}\n")
print(f"  {'rank':<7}{'feature':<10}{'importance':>13}{'what it really is':>26}")
labels = {0: "real biology", 1: "batch proxy"}
for i, j in enumerate(top):
    print(f"  {i+1:<7}{j:<10}{pi2[j]:>13.4f}{labels.get(j, 'noise'):>26}")

print(f"\n  gene 1 vs batch   : r = {np.corrcoef(Xg[:, 1], batch)[0,1]:.3f}")
print(f"  gene 1 vs outcome : r = {np.corrcoef(Xg[:, 1], y_out)[0,1]:.3f}")
print("""
  The batch-driven gene is the TOP feature, ahead of the one gene that actually
  causes the outcome. Its explanation is stable, reproducible, and has a
  plausible story attached to it as soon as you look up what the gene does.

  Nothing inside the importance calculation can detect this. The defences are
  all outside the model: check top features against technical metadata
  (Module 19), design so batch is not confounded with the outcome (Module 01),
  and validate any claimed mechanism experimentally.""")

# %% [markdown]
# ## 6. Evaluating a pretrained model

# %%
header("6. Pretraining breaks the usual meaning of a test split")

n_pat, cells_per = 24, 60
pid = np.repeat(np.arange(n_pat), cells_per)
pat_effect = rng.normal(0, 1.0, n_pat)
label = (pat_effect > 0).astype(int)[pid]
sig = pat_effect[pid] + rng.normal(0, 0.8, len(pid))
noise = rng.normal(0, 1, (len(pid), 20))
Xp = np.column_stack([sig, noise])

# A "pretrained embedding" that saw every patient, including the test ones.
emb_leaky = np.column_stack([pat_effect[pid] + rng.normal(0, 0.15, len(pid)), noise])
# An embedding built without the held-out patients would not carry this.
emb_clean = np.column_stack([sig, noise])

def grouped_auc(Xd, seed=0):
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    pred = np.full(len(label), np.nan)
    for tr, te in cv.split(Xd, label, groups=pid):
        m = LogisticRegression(max_iter=2000).fit(Xd[tr], label[tr])
        pred[te] = m.predict_proba(Xd[te])[:, 1]
    ok = ~np.isnan(pred)
    return roc_auc_score(label[ok], pred[ok])

print(f"  {'representation':<46}{'grouped CV AUC':>16}")
print(f"  {'raw features':<46}{grouped_auc(Xp):>16.3f}")
print(f"  {'embedding built WITHOUT held-out patients':<46}"
      f"{grouped_auc(emb_clean):>16.3f}")
print(f"  {'embedding that saw every patient (leaky)':<46}"
      f"{grouped_auc(emb_leaky):>16.3f}")

print("""
  The clean embedding is deliberately identical to the raw features here. An
  embedding built without the held-out patients cannot carry information about
  them beyond what the features already contain, so its row SHOULD match. That
  is the baseline the leaky row is being compared against.

  The leaky embedding wins, and the cross-validation is grouped correctly by
  patient. The split is not the problem: the REPRESENTATION already encodes
  information about the held-out patients, so no downstream split can undo it.

  This is the structural issue when evaluating pretrained models. If your
  evaluation cohort was part of the pretraining corpus, your held-out AUC is
  not held out. The questions to ask about any foundation model result:

    1. Was the evaluation data in the pretraining corpus?
    2. Does it beat a simple baseline on the SAME splits?
    3. Does the embedding separate batch, or biology?""")

header("6b. Always report the simple baseline")

for lab, Xd in (("logistic regression on raw features", Xp),
                ("random forest on raw features", Xp)):
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=1)
    pred = np.full(len(label), np.nan)
    for tr, te in cv.split(Xd, label, groups=pid):
        m = (LogisticRegression(max_iter=2000) if "logistic" in lab
             else RandomForestClassifier(n_estimators=200, random_state=0))
        m.fit(Xd[tr], label[tr])
        pred[te] = m.predict_proba(Xd[te])[:, 1]
    ok = ~np.isnan(pred)
    print(f"  {lab:<46}{roc_auc_score(label[ok], pred[ok]):>16.3f}")
# Pseudobulk baseline: one row per patient.
pb = np.array([Xp[pid == i].mean(0) for i in range(n_pat)])
pl = np.array([label[pid == i][0] for i in range(n_pat)])
aucs = []
for s in range(5):
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=s)
    pr = np.full(n_pat, np.nan)
    for tr, te in cv.split(pb, pl, groups=np.arange(n_pat)):
        m = LogisticRegression(max_iter=2000).fit(pb[tr], pl[tr])
        pr[te] = m.predict_proba(pb[te])[:, 1]
    aucs.append(roc_auc_score(pl, pr))
print(f"  {'pseudobulk, one row per patient':<46}{np.mean(aucs):>16.3f}")
print("""
  The pseudobulk baseline beats every per-cell model here, including the leaky
  embedding, because the label is a property of the PATIENT and averaging cells
  removes per-cell noise without discarding anything relevant. Running the
  comparison per cell instead of per patient also inflates the apparent sample
  size from 24 to 1440 (Module 01).

  A large model that does not beat a logistic regression on pseudobulk has not
  been shown to be useful. Report the baseline on the same splits, every
  time.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

grid = [0.2, 0.5, 0.8, 0.95, 0.99]
share_copy, share_proxy = [], []
for rr in grid:
    r = np.random.default_rng(7)
    a = r.normal(size=n)
    b = rr * a + np.sqrt(max(1 - rr ** 2, 1e-9)) * r.normal(size=n)
    Xc = np.column_stack([a, b, r.normal(size=(n, 3))])
    yc = 2.0 * a + r.normal(0, 1, n)
    mm = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                               random_state=1).fit(Xc, yc)
    pp = perm_importance(mm, Xc, yc, mean_squared_error, n_rep=5, seed=3)
    share_copy.append(pp[1] / (pp[0] + pp[1]))

    tau = np.sqrt(max(1.0 / rr - 1.0, 1e-9))     # corr(x1,x2) = 1/(1+tau^2)
    r = np.random.default_rng(11)
    z = r.normal(size=n)
    a = z + r.normal(0, tau, n); b = z + r.normal(0, tau, n)
    Xc = np.column_stack([a, b, r.normal(size=(n, 3))])
    yc = 2.0 * z + r.normal(0, 1, n)
    mm = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                               random_state=1).fit(Xc, yc)
    pp = perm_importance(mm, Xc, yc, mean_squared_error, n_rep=5, seed=3)
    share_proxy.append(pp[1] / (pp[0] + pp[1]))
ax[0].plot(grid, share_copy, "o-", color="steelblue",
           label="B is a copy of the cause")
ax[0].plot(grid, share_proxy, "o-", color="firebrick",
           label="both are proxies of a\nlatent cause")
ax[0].axhline(0.5, ls="--", color="black", lw=1)
ax[0].set_ylim(0, 0.7)
ax[0].set_ylabel("share of credit given to feature B")
ax[0].set_xlabel("correlation between x1 and x2")
ax[0].set_ylabel("permutation importance")
ax[0].set_title("Correlated features split credit"); ax[0].legend(fontsize=8)

o = np.argsort(-pi2)[:8]
ax[1].barh(range(8), pi2[o][::-1],
           color=["firebrick" if j == 1 else ("steelblue" if j == 0 else "grey")
                  for j in o][::-1])
ax[1].set_yticks(range(8)); ax[1].set_yticklabels([f"gene {j}" for j in o][::-1],
                                                  fontsize=7)
ax[1].set_xlabel("importance")
ax[1].set_title("Top feature is a pure batch proxy")

ax[2].bar(["raw", "clean\nembedding", "leaky\nembedding"],
          [grouped_auc(Xp), grouped_auc(emb_clean), grouped_auc(emb_leaky)],
          color=["grey", "steelblue", "firebrick"])
ax[2].axhline(0.5, ls="--", color="black", lw=1)
ax[2].set_ylabel("grouped CV AUC"); ax[2].set_ylim(0.4, 1.0)
ax[2].set_title("Pretraining leakage survives a correct split")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "interpretation.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'interpretation.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Do the importance methods agree?
#
# Compare permutation importance, impurity importance and a linear model's
# coefficients on the same correlated data.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# z = rng.normal(size=n)
# Xa = np.column_stack([z, 0.9 * z + 0.44 * rng.normal(size=n),
#                       rng.normal(size=(n, 3))])
# ya = 2.0 * Xa[:, 0] + rng.normal(0, 1, n)
# rf_a = RandomForestRegressor(n_estimators=300, random_state=4).fit(Xa, ya)
# ri = Ridge(alpha=1.0).fit(Xa, ya)
# pim = perm_importance(rf_a, Xa, ya, mean_squared_error, seed=8)
# print(f"  {'feature':<10}{'truth':>8}{'permutation':>14}{'impurity':>11}"
#       f"{'ridge coef':>13}")
# for j in range(5):
#     truth = 2.0 if j == 0 else 0.0
#     print(f"  {j:<10}{truth:>8.1f}{pim[j]:>14.4f}"
#           f"{rf_a.feature_importances_[j]:>11.4f}{ri.coef_[j]:>13.3f}")
# print(f"\n  rank correlation permutation vs impurity: "
#       f"{st.spearmanr(pim, rf_a.feature_importances_).statistic:.3f}")
#
# # The three methods rank the correlated pair differently, and none of them
# # recovers "feature 0 matters, feature 1 does not", because the data cannot
# # distinguish them. Impurity importance is additionally biased toward
# # high-cardinality and continuous features, which is a property of the
# # splitting criterion rather than of the data.
# #
# # The practical rule: state which method you used, because the answer depends
# # on it, and never present a single importance ranking as though it were a
# # property of the biology. Where methods disagree, that disagreement is the
# # finding.

# %% [markdown]
# ### Problem 2: How stable is the top feature?
#
# Resample the data and record how often the same feature comes top.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def top_counts(kind, par, reps=40, m=600):
#     tops = []
#     for b in range(reps):
#         rb = np.random.default_rng(300 + b)
#         zz = rb.normal(size=m)
#         if kind == "copy":     # x2 is a copy of the causal x1
#             a = zz
#             bb = par * zz + np.sqrt(max(1 - par ** 2, 1e-9)) * rb.normal(size=m)
#             tgt = a
#         else:                  # both are proxies of a latent cause
#             a = zz + rb.normal(0, par, m)
#             bb = zz + rb.normal(0, par, m)
#             tgt = zz
#         xb = np.column_stack([a, bb, rb.normal(size=(m, 3))])
#         yb = 2.0 * tgt + rb.normal(0, 1, m)
#         mb = RandomForestRegressor(n_estimators=120, max_features="sqrt",
#                                    random_state=b).fit(xb, yb)
#         pb_ = perm_importance(mb, xb, yb, mean_squared_error, n_rep=3, seed=b)
#         tops.append(int(np.argmax(pb_)))
#     t = np.array(tops)
#     return np.mean(t == 0), np.mean(t == 1), np.mean(t > 1)
#
# print(f"  {'setting':<40}{'A top':>9}{'B top':>9}{'other':>9}")
# for lab_, k, p_ in (("copy of the cause, r = 0.00", "copy", 0.0),
#                     ("copy of the cause, r = 0.95", "copy", 0.95),
#                     ("copy of the cause, r = 0.99", "copy", 0.99),
#                     ("two proxies, low noise (r = 0.96)", "proxy", 0.2),
#                     ("two proxies, high noise (r = 0.51)", "proxy", 1.0)):
#     a_, b_, o_ = top_counts(k, p_)
#     print(f"  {lab_:<40}{a_:>9.0%}{b_:>9.0%}{o_:>9.0%}")
#
# # When one feature is the true cause, it comes top nearly always, even when a
# # near-perfect copy sits beside it. When neither feature is the cause and both
# # are proxies, which one comes top is close to a coin flip.
# #
# # So "the top predictor was gene X" is a draw from this distribution, and how
# # wide the distribution is depends on something you cannot see: whether the
# # feature is the driver or a readout of one.
# #
# # The honest output is a stability measure. Resample, refit, and report how
# # often each feature appears in the top k. It costs one loop and is almost
# # never done.

# %% [markdown]
# ### Problem 3: Detect pretraining leakage without seeing the corpus
#
# You cannot inspect a foundation model's training data. Devise a check that
# still reveals contamination.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # The signature of leakage: the embedding predicts patient IDENTITY far
# # better than raw features do, even for patients it should not know.
# def identity_auc(Xd, seed=0):
#     """Can we recover WHICH patient a cell came from, from the embedding?"""
#     r = np.random.default_rng(seed)
#     target = r.integers(0, n_pat)
#     lab = (pid == target).astype(int)
#     cv = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
#     # Group by patient so the model cannot simply memorise this patient.
#     grp = pid.copy()
#     pred = np.full(len(lab), np.nan)
#     for tr, te in cv.split(Xd, lab, groups=grp):
#         if lab[tr].sum() == 0: continue
#         m = LogisticRegression(max_iter=1000).fit(Xd[tr], lab[tr])
#         pred[te] = m.predict_proba(Xd[te])[:, 1]
#     ok = ~np.isnan(pred)
#     return roc_auc_score(lab[ok], pred[ok]) if len(set(lab[ok])) > 1 else np.nan
#
# print(f"  {'representation':<44}{'predicts outcome':>18}{'encodes patient':>18}")
# for lab_, Xd in (("raw features", Xp), ("clean embedding", emb_clean),
#                  ("leaky embedding", emb_leaky)):
#     ids = [identity_auc(Xd, s) for s in range(6)]
#     print(f"  {lab_:<44}{grouped_auc(Xd):>18.3f}{np.nanmean(ids):>18.3f}")
#
# # The leaky embedding is unusual on BOTH axes: it predicts the outcome better
# # and it carries more patient-specific information. That second number is the
# # diagnostic, because it can be computed without any access to the
# # pretraining corpus.
# #
# # Other checks in the same spirit: compare performance on cohorts collected
# # AFTER the model's training cutoff, which cannot have been included; and
# # test whether the embedding clusters by processing site. None of these
# # proves contamination, but a model that is unremarkable on all of them is
# # much easier to believe.

# %% [markdown]
# ## What to take away
#
# 1. Permutation importance (46.1) works on independent features and misleads
#    on correlated ones, where credit **splits** between copies.
# 2. Permuting evaluates the model **off the data manifold**, in a region where
#    its behaviour is arbitrary.
# 3. Shapley values (46.2) are exactly additive (46.3). Additivity is a
#    property of the attribution, not evidence of a mechanism.
# 4. An explanation describes the **model**. The top feature can be a pure
#    batch proxy, and no importance method will tell you.
# 5. Report **stability**: how often does the same feature come top under
#    resampling?
# 6. For pretrained models, a correct split does not undo contamination in the
#    representation. Ask what was in the corpus, and always report a simple
#    baseline on the same splits.
#
# **This is the final applied module.** See `statsPy/exercises/` for the
# integrative problem sets.
