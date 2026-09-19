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
# # Module 21: Statistical learning and biomarker prediction
#
# **Curriculum link:** `stats.md` -> Topic 31, equations (31.1)-(31.12)
#
# Prediction is a **different objective** from association, with different
# validation standards.
#
# ## What you will learn
#
# 1. The four leaks that destroy biomarker studies: each demonstrated on data
#    containing **no real signal at all**.
# 2. Group-aware splitting: donors, not cells.
# 3. Nested cross-validation (31.4).
# 4. Discrimination (31.8) vs **calibration** (31.9)-(31.11) vs clinical
#    utility (31.12).
# 5. Why lasso-selected features are unstable (31.6).

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import (KFold, GroupKFold, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve, precision_recall_curve)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "21_prediction_and_validation"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Leak #1: feature selection outside the fold
#
# The classic demonstration: select the features most correlated with the
# outcome using **all** samples, then cross-validate. The outcome is random.

# %%
header("1. Feature selection outside the fold (the classic leak)")

rng = np.random.default_rng(2101)
n, p = 100, 5000
X = rng.normal(0, 1, (n, p))
y = rng.integers(0, 2, n)                    # PURE NOISE - no signal whatsoever

# --- WRONG: select the top-100 features using ALL the data, then CV ---------
F, _ = f_classif(X, y)
top = np.argsort(-F)[:100]
auc_leaky = cross_val_score(LogisticRegression(max_iter=1000),
                            X[:, top], y, cv=StratifiedKFold(5, shuffle=True,
                                                             random_state=0),
                            scoring="roc_auc").mean()

# --- RIGHT: put the selection INSIDE the pipeline, so it is refit per fold --
pipe = Pipeline([("sel", SelectKBest(f_classif, k=100)),
                 ("clf", LogisticRegression(max_iter=1000))])
auc_honest = cross_val_score(pipe, X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             scoring="roc_auc").mean()

print(f"  Data: n={n}, p={p}, outcome is a COIN FLIP. Honest AUC must be ~0.5.")
print(f"    select on all data, then CV  : AUC = {auc_leaky:.3f}   <- FANTASY")
print(f"    selection inside the pipeline: AUC = {auc_honest:.3f}   <- honest")
print("\n  The leaky number is what gets published as a '90% accurate")
print("  biomarker signature'. Nothing in the data supports it.")

# %% [markdown]
# ## 2. Leak #2: scaling/normalisation outside the fold

# %%
header("2. Unsupervised preprocessing fitted on all data")

from sklearn.decomposition import PCA

# A single 5-fold CV at n=60 has a standard error of roughly 0.06 on the AUC,
# which is larger than the effect we are trying to measure. So we REPLICATE
# over many independent null datasets - the same discipline section 7 of
# Module 09 applies to power.
from sklearn.decomposition import PCA


def null_auc(p_feat, n=60, n_rep=60, k=20):
    """Mean CV-AUC on pure-noise data, with PCA fitted leakily vs honestly."""
    leaky, honest = [], []
    for rep in range(n_rep):
        r = np.random.default_rng(10_000 + rep)
        X = r.normal(0, 1, (n, p_feat))
        y = r.integers(0, 2, n)
        cv = StratifiedKFold(5, shuffle=True, random_state=rep)
        # LEAKY: PCA sees every row, including the held-out ones.
        Xall = PCA(n_components=k, random_state=0).fit_transform(
            StandardScaler().fit_transform(X))
        leaky.append(cross_val_score(LogisticRegression(max_iter=1000),
                                     Xall, y, cv=cv, scoring="roc_auc").mean())
        # HONEST: PCA refitted inside each fold.
        honest.append(cross_val_score(
            Pipeline([("sc", StandardScaler()),
                      ("pca", PCA(n_components=k, random_state=0)),
                      ("clf", LogisticRegression(max_iter=1000))]),
            X, y, cv=cv, scoring="roc_auc").mean())
    return np.mean(leaky), np.mean(honest), np.std(leaky, ddof=1) / np.sqrt(n_rep)


print("  Pure-noise data (outcome is a coin flip), so the honest AUC is 0.5.")
print("  Averaged over 60 independent datasets to beat the CV noise:\n")
print(f"  {'p (features)':>14}{'PCA on all rows':>18}{'PCA in the fold':>18}"
      f"{'optimism':>11}{'MC SE':>8}")
for p_feat in [200, 800, 3000]:
    a_leak, a_ok, se = null_auc(p_feat)
    print(f"  {p_feat:>14,}{a_leak:>18.4f}{a_ok:>18.4f}"
          f"{a_leak - a_ok:>+11.4f}{se:>8.4f}")

print("\n  READ THE RESULT HONESTLY: the optimism is within Monte-Carlo error")
print("  of ZERO at every p. Fitting an UNSUPERVISED PCA on all rows did not")
print("  measurably inflate performance here. That is the correct conclusion")
print("  from this experiment, and it is worth stating plainly rather than")
print("  asserting a leak that the data do not show.")
print("\n  So distinguish two classes of preprocessing:")
print("\n  (a) OUTCOME-BLIND steps - scaling, PCA, quantile normalisation,")
print("      unsupervised batch correction. They never see y, so the only")
print("      thing they leak is the test FEATURE distribution. As measured")
print("      above, that is a small effect. It is still worth avoiding: it")
print("      costs nothing to put them in a Pipeline, and steps that borrow")
print("      across ROWS (kNN/iterative imputation, sample-sample quantile")
print("      normalisation) leak considerably more than PCA does.")
print("\n  (b) OUTCOME-AWARE steps - feature selection by association with y,")
print("      normalising cases and controls separately, choosing a cut-point")
print("      to maximise separation. These are CATASTROPHIC, as section 1")
print("      showed: AUC 1.000 on a coin flip. They can also destroy real")
print("      signal, since a 'correction' that centres each outcome group on")
print("      its own mean removes exactly the difference you wanted to detect.")
print("\n  RULE: if a step LEARNS anything from data, put it in the Pipeline.")
print("  Then the fold boundary is structural rather than a matter of memory,")
print("  and you never have to reason about which class a step falls into.")

# %% [markdown]
# ## 3. Leak #3: splitting a donor's cells across folds
#
# **Group-aware splitting is mandatory.**

# %%
header("3. Donor leakage: the most common bioinformatics leak")

from sklearn.model_selection import StratifiedGroupKFold

rng = np.random.default_rng(2103)
n_donors, cells_per = 40, 25
donor = np.repeat(np.arange(n_donors), cells_per)
# Each donor has a strong individual signature; the OUTCOME is assigned per
# donor and is pure noise with respect to any real biology.
donor_sig = rng.normal(0, 2.0, (n_donors, 50))
Xc = donor_sig[donor] + rng.normal(0, 1.0, (n_donors * cells_per, 50))
y_donor = np.repeat([0, 1], n_donors // 2)          # balanced by construction
rng.shuffle(y_donor)
yc = y_donor[donor]

auc_random = cross_val_score(LogisticRegression(max_iter=1000), Xc, yc,
                             cv=KFold(5, shuffle=True, random_state=0),
                             scoring="roc_auc").mean()
# StratifiedGroupKFold keeps donors intact AND keeps both classes in each fold.
sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
auc_group = cross_val_score(LogisticRegression(max_iter=1000), Xc, yc,
                            cv=sgkf, groups=donor, scoring="roc_auc").mean()

# Prove the splits are clean.
leak = any(set(donor[tr]) & set(donor[te]) for tr, te in sgkf.split(Xc, yc, donor))
print(f"  {n_donors} donors x {cells_per} cells; outcome assigned per donor at random.")
print(f"    random 5-fold CV over CELLS        : AUC = {auc_random:.3f}"
      f"   <- memorises donors")
print(f"    StratifiedGroupKFold by DONOR      : AUC = {auc_group:.3f}"
      f"   <- honest")
print(f"    any donor in both train and test?  : {leak}")
print("\n  Random splitting puts near-duplicate cells from the same donor in")
print("  both train and test. The model learns 'which donor is this?', which")
print("  perfectly predicts the label - and generalises to nobody.")
print("  The same applies to repeated samples, technical replicates, and")
print("  multiple tissue sections from one patient.")
print("\n  Use StratifiedGroupKFold, not plain GroupKFold: with few donors,")
print("  ungrouped-but-stratified or grouped-but-unstratified splits can")
print("  produce single-class test folds, and AUC is then undefined.")

# %% [markdown]
# ## 4. Leak #4: tuning and reporting on the same CV, eq. (31.4)

# %%
header("4. Nested cross-validation (31.4)")

rng = np.random.default_rng(2104)
n, p = 120, 200
X = rng.normal(0, 1, (n, p))
beta = np.zeros(p); beta[:5] = 0.35              # a small REAL signal
y = (X @ beta + rng.normal(0, 1, n) > 0).astype(int)

grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
base = Pipeline([("sc", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000))])

gs = GridSearchCV(base, grid, cv=inner, scoring="roc_auc").fit(X, y)
auc_flat = gs.best_score_                      # the score we TUNED on
auc_nested = cross_val_score(GridSearchCV(base, grid, cv=inner,
                                          scoring="roc_auc"),
                             X, y, cv=outer, scoring="roc_auc").mean()
print(f"  best inner-CV score (used for tuning) : AUC = {auc_flat:.3f}"
      f"   <- optimistic")
print(f"  nested CV (outer loop)                : AUC = {auc_nested:.3f}"
      f"   <- honest")
print(f"  optimism = {auc_flat - auc_nested:+.3f}")
print("\n  Reporting the CV score you selected on is selection bias. The outer")
print("  loop estimates performance; the inner loop tunes.")

# %% [markdown]
# ## 5. Discrimination vs calibration vs utility, eq. (31.8)-(31.12)

# %%
header("5. AUC is not enough (31.8)-(31.12)")

rng = np.random.default_rng(2105)
# Use distinct names: later sections rebind `n`, `X` and `y`, and the figure
# at the bottom still needs this cohort. Shadowing is the commonest
# source of "it ran yesterday" bugs in analysis scripts.
n_cal = 4000
z = rng.normal(0, 1, n_cal)
p_true = 1 / (1 + np.exp(-(-3.0 + 1.5 * z)))     # ~6% prevalence
y_cal = rng.binomial(1, p_true)
print(f"  simulated cohort: n = {n_cal}, prevalence = {y_cal.mean():.3f}")

# Three "models": well calibrated, over-confident, and shifted.
models = {
    "well calibrated": p_true,
    "over-confident (2x log-odds)": 1 / (1 + np.exp(-2 * np.log(p_true/(1-p_true)))),
    "shifted (+1 on log-odds)": 1 / (1 + np.exp(-(np.log(p_true/(1-p_true)) + 1))),
}

def calibration_slope(y, p):
    """stats.md eq. (31.11): regress the outcome on the predicted log-odds."""
    lo = np.log(np.clip(p, 1e-9, 1-1e-9) / np.clip(1-p, 1e-9, 1-1e-9))
    import statsmodels.api as sm
    fit = sm.GLM(y, sm.add_constant(lo), family=sm.families.Binomial()).fit()
    return fit.params[1], fit.params[0]


def net_benefit(y, p, pt):
    """stats.md eq. (31.12)."""
    pred = p >= pt
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    return tp / len(y) - (fp / len(y)) * (pt / (1 - pt))


print(f"\n  {'model':<30}{'AUC':>8}{'PR-AUC':>9}{'Brier':>9}"
      f"{'calib slope':>13}{'NB@0.10':>10}")
for name, p in models.items():
    slope, intercept = calibration_slope(y_cal, p)
    print(f"  {name:<30}{roc_auc_score(y_cal, p):>8.3f}"
          f"{average_precision_score(y_cal, p):>9.3f}"
          f"{brier_score_loss(y_cal, p):>9.4f}{slope:>13.3f}"
          f"{net_benefit(y_cal, p, 0.10):>10.4f}")

print("\n  All three have IDENTICAL AUC - ranking is unchanged by a monotone")
print("  transformation of the probabilities. Only Brier and the calibration")
print("  slope reveal that two of them produce wrong PROBABILITIES.")
print("  Calibration slope: 1 = perfect, <1 = predictions too extreme")
print("  (the usual overfitting signature, and the shrinkage factor to apply).")

print(f"\n  Prevalence-dependence of AUC vs PPV at a fixed threshold:")
print(f"  {'prevalence':>12}{'AUC':>8}{'PPV at p>=0.10':>17}")
for prev_shift in [-3.0, -1.5, 0.0]:
    pt_ = 1 / (1 + np.exp(-(prev_shift + 1.5 * z)))
    yy = rng.binomial(1, pt_)   # z and pt_ both have length n_cal
    sel = pt_ >= 0.10
    ppv = yy[sel].mean() if sel.sum() else np.nan
    print(f"  {yy.mean():>12.3f}{roc_auc_score(yy, pt_):>8.3f}{ppv:>17.3f}")
print("  AUC is PREVALENCE-INDEPENDENT - a virtue for comparing models and a")
print("  trap for judging usefulness. Report precision-recall for imbalanced")
print("  problems; its baseline IS the prevalence.")

# %% [markdown]
# ## 6. Lasso selects unstable feature sets, eq. (31.6)-(31.7)

# %%
header("6. Correlated features make lasso selection unstable (31.6)-(31.7)")

rng = np.random.default_rng(2106)
n, p = 120, 60
# Build 5 groups of 6 highly correlated features (like a co-expression module).
latent = rng.normal(0, 1, (n, 10))
X = np.column_stack([latent[:, k // 6] + rng.normal(0, 0.35, n)
                     for k in range(30)] +
                    [rng.normal(0, 1, n) for _ in range(30)])
beta = np.zeros(p); beta[[0, 6, 12]] = 1.5       # one member per correlated group
yv = X @ beta + rng.normal(0, 1, n)

def selection_frequency(model_cls, n_boot=60, **kw):
    counts = np.zeros(p)
    for b in range(n_boot):
        idx = np.random.default_rng(b).integers(0, n, n)
        m = Pipeline([("sc", StandardScaler()),
                      ("m", model_cls(**kw))]).fit(X[idx], yv[idx])
        counts += np.abs(m.named_steps["m"].coef_) > 1e-8
    return counts / n_boot


f_lasso = selection_frequency(Lasso, alpha=0.15, max_iter=5000)
f_enet = selection_frequency(ElasticNet, alpha=0.15, l1_ratio=0.3, max_iter=5000)
grp0 = list(range(6))
print(f"  Features 0-5 are one correlated group; only feature 0 is truly causal.")
print(f"  {'feature':>9}{'lasso sel. freq':>18}{'elastic-net sel. freq':>24}")
for j in grp0:
    star = " *causal*" if j == 0 else ""
    print(f"  {j:>9}{f_lasso[j]:>18.2f}{f_enet[j]:>24.2f}{star}")
print(f"\n  mean selection frequency of the true features "
      f"(0, 6, 12): lasso {f_lasso[[0,6,12]].mean():.2f}, "
      f"elastic net {f_enet[[0,6,12]].mean():.2f}")
print("\n  Lasso picks ONE arbitrary member of each correlated group, and which")
print("  one changes across resamples. Elastic net's ridge component spreads")
print("  weight across the group (the 'grouping effect'), which is more stable")
print("  and more biologically sensible. NEVER present lasso-selected features")
print("  as 'the important genes' without a stability analysis.")

# %% [markdown]
# ## 7. A correct pipeline, end to end

# %%
header("7. Template: a leak-proof biomarker pipeline")

rng = np.random.default_rng(2107)
n_donors, cells_per, p = 40, 10, 300
donor = np.repeat(np.arange(n_donors), cells_per)
sig = rng.normal(0, 1.0, (n_donors, p))
y_don = rng.integers(0, 2, n_donors)
sig[y_don == 1, :10] += 0.9                      # a REAL, modest signal
Xf = sig[donor] + rng.normal(0, 1.0, (n_donors * cells_per, p))
yf = y_don[donor]

# Everything learned lives inside the Pipeline; splitting is by donor.
pipeline = Pipeline([
    ("scale", StandardScaler()),                 # fitted per fold
    ("select", SelectKBest(f_classif, k=20)),    # fitted per fold
    ("clf", LogisticRegression(max_iter=2000, C=0.1)),
])
scores = cross_val_score(pipeline, Xf, yf, cv=GroupKFold(5), groups=donor,
                         scoring="roc_auc")
print(f"  GroupKFold AUC per fold: {np.round(scores, 3)}")
print(f"  mean = {scores.mean():.3f}  (SD {scores.std(ddof=1):.3f})")
print("\n  Checklist encoded in this snippet:")
print("    [x] every learned step inside Pipeline -> refit per fold")
print("    [x] GroupKFold(groups=donor)          -> no donor in both sides")
print("    [x] metric matches the decision       -> AUC + calibration below")
print("    [x] fold-to-fold spread reported      -> not just the mean")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for name, p_ in models.items():
    fpr, tpr, _ = roc_curve(y_cal, p_)
    axes[0].plot(fpr, tpr,
                 label=f"{name.split(' (')[0]} (AUC {roc_auc_score(y_cal, p_):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("Eq. (31.8): identical AUCs", fontsize=9)
axes[0].legend(fontsize=6)

for name, p_ in models.items():
    bins = np.quantile(p_, np.linspace(0, 1, 11))
    idx = np.digitize(p_, bins[1:-1])
    xs = [p_[idx == b].mean() for b in range(10)]
    ys = [y_cal[idx == b].mean() for b in range(10)]
    axes[1].plot(xs, ys, "o-", ms=4, label=name.split(" (")[0])
axes[1].plot([0, 0.6], [0, 0.6], "k--", lw=1, label="perfect")
axes[1].set_xlabel("mean predicted probability"); axes[1].set_ylabel("observed")
axes[1].set_title("Eq. (31.11): calibration plot", fontsize=9)
axes[1].legend(fontsize=6)

pts = np.linspace(0.01, 0.4, 60)
for name, p_ in models.items():
    axes[2].plot(pts, [net_benefit(y_cal, p_, t) for t in pts],
                 label=name.split(" (")[0])
axes[2].plot(pts, [net_benefit(y_cal, np.ones(n_cal), t) for t in pts], "k:",
             label="treat all")
axes[2].axhline(0, color="k", lw=1, label="treat none")
axes[2].set_xlabel("threshold probability $p_t$"); axes[2].set_ylabel("net benefit")
axes[2].set_title("Eq. (31.12): decision curve", fontsize=9)
axes[2].legend(fontsize=6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "prediction.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/prediction.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 31)
#
# 1. **Split by donor/patient, always.** Group-aware CV is not optional.
# 2. Put every learned step inside the fold: build a `Pipeline` so this is
#    enforced structurally rather than by discipline.
# 3. Use nested CV whenever you tune.
# 4. Report discrimination **and** calibration **and** (for clinical models)
#    net benefit.
# 5. Distinguish apparent, internally validated, and externally validated
#    performance. Only the third predicts the next cohort.
#
# **Next:** `22_causal_inference.py`
