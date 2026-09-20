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
# # Module 28: Conformal prediction and distribution-free uncertainty
#
# **Curriculum link:** `stats.md` -> Topic 39, equations (39.1)-(39.5)
# **Assumes:** Module 21.
#
# ## What you will learn
#
# 1. **Split conformal** (39.1)-(39.3): a finite-sample coverage guarantee for
#    any model, with no distributional assumption.
# 2. Why the $n+1$ and the ceiling in (39.2) are load-bearing.
# 3. Coverage is **marginal**, not conditional, and what that costs you.
# 4. **CQR** (39.4) gives intervals that widen where the model is uncertain.
# 5. **Prediction sets** for classification (39.5), including the empty set.
# 6. Exchangeability is the assumption, and grouped data breaks it exactly as
#    it breaks cross-validation.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

MODULE_NAME = "28_conformal_prediction"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(28)
ALPHA = 0.10                       # target 90% coverage


def conformal_quantile(scores, alpha=ALPHA):
    """Eq. (39.2). The ceiling and the n+1 are what make it exact."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf              # too few calibration points to promise this
    return np.sort(scores)[k - 1]


# %% [markdown]
# ## 1. A regression problem where the model is wrong

# %%
header("1. Split conformal on a deliberately misspecified model")


def make_data(n, seed=0):
    """Heteroscedastic and nonlinear, so a linear model is genuinely wrong."""
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, n)
    noise = 0.3 + 0.9 * np.abs(x)              # spread grows with |x|
    y = np.sin(1.5 * x) * 2.0 + 0.5 * x + r.normal(0, noise)
    return x.reshape(-1, 1), y


Xtr, ytr = make_data(500, 1)
Xcal, ycal = make_data(500, 2)
Xte, yte = make_data(4000, 3)

model = LinearRegression().fit(Xtr, ytr)       # the wrong model, on purpose
s_cal = np.abs(ycal - model.predict(Xcal))     # eq. (39.1)
q = conformal_quantile(s_cal)                  # eq. (39.2)
lo, hi = model.predict(Xte) - q, model.predict(Xte) + q
cov = np.mean((yte >= lo) & (yte <= hi))

# The textbook alternative: a Gaussian interval from the residual SD.
sd = np.std(ytr - model.predict(Xtr), ddof=2)
z = st.norm.ppf(1 - ALPHA / 2)
cov_g = np.mean(np.abs(yte - model.predict(Xte)) <= z * sd)

print(f"  model: linear. truth: sinusoidal with heteroscedastic noise.")
print(f"  target coverage = {1-ALPHA:.0%}\n")
print(f"  {'interval':<34}{'coverage':>11}{'mean width':>13}")
print(f"  {'Gaussian, residual SD':<34}{cov_g:>11.1%}{2*z*sd:>13.3f}")
print(f"  {'split conformal (39.1)-(39.3)':<34}{cov:>11.1%}{2*q:>13.3f}")
print("""
  The model is badly misspecified and the conformal interval still covers at
  the advertised rate. It does not need the model to be right; it only needs
  the calibration and test data to be exchangeable.

  The Gaussian interval happens to be close here, but its validity rests on
  normal, homoscedastic residuals, neither of which holds. It is right by
  luck, and it has no guarantee to fall back on.""")

# %% [markdown]
# ## 2. The finite-sample correction is not cosmetic

# %%
header("2. Why (39.2) uses ceil((n+1)(1-alpha)) and not a plain quantile")

print(f"  {'calibration n':<16}{'naive quantile':>17}{'conformal (39.2)':>19}")
for n_cal in (10, 20, 50, 200, 1000):
    c_naive, c_conf = [], []
    for i in range(400):
        Xc, yc = make_data(n_cal, 500 + i)
        Xt, yt = make_data(600, 9000 + i)
        s = np.abs(yc - model.predict(Xc))
        qn = np.quantile(s, 1 - ALPHA)                     # naive
        qc = conformal_quantile(s)                         # eq. (39.2)
        pr = model.predict(Xt)
        c_naive.append(np.mean(np.abs(yt - pr) <= qn))
        c_conf.append(np.mean(np.abs(yt - pr) <= qc))
    print(f"  {n_cal:<16}{np.mean(c_naive):>17.1%}{np.mean(c_conf):>19.1%}")

print(f"""
  Target is {1-ALPHA:.0%}. The naive empirical quantile undercovers at small n, and
  the shortfall is worst exactly when you have least data to spare. The
  correction in (39.2) fixes it at every sample size.

  Note the top row. With n = 10, ceil(11 x 0.9) = 10, so the rule uses the
  LARGEST of the ten calibration scores and has nothing in reserve. Ask for
  95% from those same ten points and ceil(11 x 0.95) = 11 > 10: there is no
  score extreme enough, and the function returns infinity rather than
  pretending.""")
print(f"  smallest calibration set that can promise a given level:")
for a_ in (0.20, 0.10, 0.05, 0.01):
    need = int(np.ceil(1 / a_)) - 1
    print(f"    {1-a_:.0%} coverage needs n >= {need}")
print("""
  A method that says 'I cannot do this with the data you gave me' is more
  useful than one that quietly undercovers. Note the practical consequence:
  a 99% conformal interval needs at least 99 calibration points, whatever the
  model.""")

# %% [markdown]
# ## 3. Marginal is not conditional

# %%
header("3. The guarantee is marginal, and that matters")

bins = np.quantile(Xte.ravel(), np.linspace(0, 1, 6))
print(f"  Overall coverage {cov:.1%}. By region of the predictor:\n")
print(f"  {'x range':<20}{'coverage':>11}{'width':>9}")
for i in range(5):
    m = (Xte.ravel() >= bins[i]) & (Xte.ravel() <= bins[i + 1])
    print(f"  {f'[{bins[i]:+.1f}, {bins[i+1]:+.1f}]':<20}"
          f"{np.mean((yte[m] >= lo[m]) & (yte[m] <= hi[m])):>11.1%}"
          f"{2*q:>9.3f}")

print("""
  The average is right and the parts are not. Where the noise is small the
  interval is far too wide; where it is large the interval undercovers badly.

  This is exactly what eq. (39.3) promises and no more: coverage averaged over
  the population. If your decision is made per patient, an interval that is
  90% correct on average but 60% correct for the patients who matter is not
  fit for purpose. The next section fixes it.""")

# %% [markdown]
# ## 4. CQR: let the width follow the data, eq. (39.4)

# %%
header("4. Conformalised quantile regression (39.4)")

lo_m = GradientBoostingRegressor(loss="quantile", alpha=ALPHA / 2,
                                 n_estimators=200, max_depth=3,
                                 random_state=0).fit(Xtr, ytr)
hi_m = GradientBoostingRegressor(loss="quantile", alpha=1 - ALPHA / 2,
                                 n_estimators=200, max_depth=3,
                                 random_state=0).fit(Xtr, ytr)
s_cqr = np.maximum(lo_m.predict(Xcal) - ycal, ycal - hi_m.predict(Xcal))  # (39.4)
q_cqr = conformal_quantile(s_cqr)
lo_c, hi_c = lo_m.predict(Xte) - q_cqr, hi_m.predict(Xte) + q_cqr
cov_c = np.mean((yte >= lo_c) & (yte <= hi_c))

print(f"  {'method':<26}{'coverage':>11}{'mean width':>13}{'width at |x|<1':>16}"
      f"{'width at |x|>2':>16}")
near = np.abs(Xte.ravel()) < 1
far = np.abs(Xte.ravel()) > 2
print(f"  {'split conformal':<26}{cov:>11.1%}{2*q:>13.3f}{2*q:>16.3f}{2*q:>16.3f}")
print(f"  {'CQR (39.4)':<26}{cov_c:>11.1%}{np.mean(hi_c-lo_c):>13.3f}"
      f"{np.mean((hi_c-lo_c)[near]):>16.3f}{np.mean((hi_c-lo_c)[far]):>16.3f}")

print(f"\n  CQR coverage by region:")
for i in range(5):
    m = (Xte.ravel() >= bins[i]) & (Xte.ravel() <= bins[i + 1])
    print(f"    [{bins[i]:+.1f}, {bins[i+1]:+.1f}]  "
          f"{np.mean((yte[m] >= lo_c[m]) & (yte[m] <= hi_c[m])):.1%}")

print("""
  Same marginal guarantee, far better behaviour within regions, and a narrower
  interval on average. The width now reports where the model is genuinely
  uncertain, which is the information a reader wants.

  Conformal prediction never makes a model better. It makes the model's
  uncertainty honest, and a better model gives tighter honest intervals.""")

# %% [markdown]
# ## 5. Classification: prediction sets, eq. (39.5)

# %%
header("5. Prediction sets, including the empty one (39.5)")

SEP = 3.5                     # well separated, so the model is confident


def make_cls(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    X = r.normal(0, 1, (n, 4)) + shift
    X[:, 0] += y * SEP                      # classes 0/1/2 separated on x0
    X[:, 1] += (y == 2) * 1.0
    return X, y


Xa, ya = make_cls(1200, 11)
Xb, yb = make_cls(1200, 12)
Xc, yc = make_cls(3000, 13)
clf = RandomForestClassifier(n_estimators=250, random_state=0).fit(Xa, ya)
s_cls = 1 - clf.predict_proba(Xb)[np.arange(len(yb)), yb]
q_cls = conformal_quantile(s_cls)
sets = (1 - clf.predict_proba(Xc)) <= q_cls                # eq. (39.5)
sizes = sets.sum(1)
covered = sets[np.arange(len(yc)), yc]

print(f"  target {1-ALPHA:.0%} coverage, 3 classes\n")
print(f"  coverage of the true label : {covered.mean():.1%}")
print(f"  mean set size              : {sizes.mean():.2f}")
print(f"\n  {'set size':<12}{'share':>10}{'coverage in that group':>26}")
for k in range(4):
    m = sizes == k
    if m.sum():
        print(f"  {k:<12}{m.mean():>10.1%}"
              f"{(covered[m].mean() if m.sum() else float('nan')):>26.1%}")

print(f"\n  the calibrated threshold is: include class k when p_k >= "
      f"{1 - q_cls:.3f}")
print("""
  A set of size 1 is a confident call. Size 2 or 3 would mean the model cannot
  separate those classes for this sample, which a single predicted label would
  have concealed.

  Size 0 is the interesting one: no class reaches the threshold, so the sample
  does not look like anything the model was calibrated on. That is an outlier
  flag you get for free, and a point prediction can never produce it.""")

# A point deliberately placed midway between two class centres is exactly the
# case where no label should be asserted.
amb = rng.normal(0, 1, (500, 4))
amb[:, 0] = SEP * 0.5                       # halfway between class 0 and 1
p_amb = clf.predict_proba(amb)
s_amb = (1 - p_amb) <= q_cls
print(f"\n  500 samples placed halfway between two class centres:")
print(f"    mean set size {s_amb.sum(1).mean():.2f}, "
      f"empty sets {np.mean(s_amb.sum(1) == 0):.0%}")
print(f"    a plain classifier would label every one of them, with mean "
      f"confidence {p_amb.max(1).mean():.2f}")
print("""
  The conformal rule abstains on nearly all of them. The plain classifier
  cannot:
  its class probabilities are forced to sum to 1, so SOMETHING always wins,
  however implausible the sample.""")

# %% [markdown]
# ## 6. Exchangeability is the assumption that breaks

# %%
header("6. Distribution shift and grouped data")

# (a) covariate shift: the test set is drawn from a shifted distribution
for sh in (0.0, 0.5, 1.5):
    Xs, ys = make_cls(3000, 14, shift=sh)
    st_ = (1 - clf.predict_proba(Xs)) <= q_cls
    print(f"  covariate shift {sh:<4} -> coverage {st_[np.arange(len(ys)), ys].mean():.1%}")

# (b) grouped data calibrated the wrong way
header("6b. Calibrating across groups you will not see again")


def grouped(n_groups, per, seed):
    """Each group (donor, batch, site) has its own offset AND occupies its own
    region of the predictor, which is what lets a flexible model memorise the
    offsets of groups it has seen."""
    r = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    centre = r.permutation(np.linspace(-3, 3, n_groups))[g]
    offset = r.normal(0, 2.0, n_groups)[g]          # a per-group shift
    x = r.normal(centre, 0.06)
    y = np.sin(1.5 * x) * 2 + offset + r.normal(0, 0.5, n_groups * per)
    return x.reshape(-1, 1), y, g


NG, PER, N_FIT, N_CAL = 80, 40, 30, 15
Xg, yg, gg = grouped(NG, PER, 21)
# ONE model, fit on groups 0..N_FIT-1. Only the CALIBRATION differs between the
# two rows below, so nothing else can explain the gap.
fit_rows = np.nonzero(gg < N_FIT)[0]
held = np.random.default_rng(7).choice(fit_rows, 300, replace=False)
fitset = np.setdiff1d(fit_rows, held)
mod = GradientBoostingRegressor(n_estimators=400, max_depth=4,
                                random_state=0).fit(Xg[fitset], yg[fitset])
# WRONG: calibrate on held-out ROWS, whose groups the model trained on.
q_row = conformal_quantile(np.abs(yg[held] - mod.predict(Xg[held])))
# RIGHT: calibrate on whole held-out GROUPS, matching how the model is used.
cal_g = (gg >= N_FIT) & (gg < N_FIT + N_CAL)
q_grp = conformal_quantile(np.abs(yg[cal_g] - mod.predict(Xg[cal_g])))
new_g = gg >= N_FIT + N_CAL                        # groups never seen before
err = np.abs(yg[new_g] - mod.predict(Xg[new_g]))

print(f"  target {1-ALPHA:.0%}, evaluated on groups never seen in training\n")
print(f"  {'calibration set':<42}{'coverage':>11}{'width':>9}")
print(f"  {'held-out ROWS from the training groups':<42}"
      f"{np.mean(err <= q_row):>11.1%}{2*q_row:>9.2f}")
print(f"  {'whole held-out GROUPS':<42}"
      f"{np.mean(err <= q_grp):>11.1%}{2*q_grp:>9.2f}")

print("""
  Both rows use the SAME model, so the model is not what differs. The
  row-calibrated interval is several times too narrow and covers around half
  the time instead of 90%.

  The reason: those calibration rows came from groups the model had already
  fitted, so their residuals measure how well it INTERPOLATES within a known
  group, not how well it handles a new one. Calibrating on whole held-out
  groups measures the right thing and restores coverage.

  This is the same error as leaking donors across cross-validation folds
  (Module 21), and it has the same fix. Note also what supplies the precision
  of the calibration: the number of GROUPS, not the number of rows. Three
  calibration groups give an unreliable quantile however many cells each one
  contains.

  Conformal prediction is only as valid as the exchangeability you arrange,
  and an interval that is too narrow is worse than no interval, because it
  looks authoritative.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

o = np.argsort(Xte.ravel())
xs = Xte.ravel()[o]
ax[0].scatter(Xte.ravel(), yte, s=3, alpha=0.12, color="grey")
ax[0].fill_between(xs, lo[o], hi[o], alpha=0.25, color="steelblue",
                   label="split conformal")
ax[0].plot(xs, model.predict(Xte)[o], color="firebrick", lw=1.5)
ax[0].set_title(f"Constant width, {cov:.0%} marginal")
ax[0].set_xlabel("x"); ax[0].legend(fontsize=8)

ax[1].scatter(Xte.ravel(), yte, s=3, alpha=0.12, color="grey")
ax[1].fill_between(xs, lo_c[o], hi_c[o], alpha=0.3, color="seagreen", label="CQR")
ax[1].set_title(f"CQR width follows the noise, {cov_c:.0%}")
ax[1].set_xlabel("x"); ax[1].legend(fontsize=8)

ns = [10, 20, 30, 50, 100, 200, 500, 1000]
naive_c, conf_c = [], []
for n_cal in ns:
    a_, b_ = [], []
    for i in range(250):
        Xc, yc2 = make_data(n_cal, 700 + i)
        Xt, yt2 = make_data(400, 7700 + i)
        s = np.abs(yc2 - model.predict(Xc))
        pr = model.predict(Xt)
        a_.append(np.mean(np.abs(yt2 - pr) <= np.quantile(s, 1 - ALPHA)))
        b_.append(np.mean(np.abs(yt2 - pr) <= conformal_quantile(s)))
    naive_c.append(np.mean(a_)); conf_c.append(np.mean(b_))
ax[2].plot(ns, naive_c, "o-", color="firebrick", label="naive quantile")
ax[2].plot(ns, conf_c, "o-", color="steelblue", label="eq. (39.2)")
ax[2].axhline(1 - ALPHA, color="black", ls="--", lw=1)
ax[2].set_xscale("log"); ax[2].set_xlabel("calibration set size")
ax[2].set_ylabel("coverage"); ax[2].set_title("The finite-sample correction")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "conformal.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'conformal.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Does a better model give better guarantees?
#
# Compare conformal intervals from a badly wrong model, a reasonable model and
# a nearly perfect one. What changes and what does not?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# from sklearn.neighbors import KNeighborsRegressor
# print(f"  target coverage {1-ALPHA:.0%}\n")
# print(f"  {'model':<34}{'train R2':>10}{'coverage':>11}{'mean width':>13}")
# for lab, m in (("linear (wrong)", LinearRegression()),
#                ("gradient boosting", GradientBoostingRegressor(random_state=0)),
#                ("k-NN, k=3 (overfits)", KNeighborsRegressor(n_neighbors=3))):
#     m.fit(Xtr, ytr)
#     s = np.abs(ycal - m.predict(Xcal))
#     qq = conformal_quantile(s)
#     pr = m.predict(Xte)
#     c_ = np.mean(np.abs(yte - pr) <= qq)
#     print(f"  {lab:<34}{m.score(Xtr, ytr):>10.3f}{c_:>11.1%}{2*qq:>13.3f}")
#
# # The coverage column is essentially constant. That is the guarantee doing
# # its job: it does not care how good the model is, only that calibration and
# # test data are exchangeable.
# #
# # The width column is where model quality shows up. A better model produces
# # smaller residuals on the calibration set, so the interval is tighter at the
# # same coverage.
# #
# # This is the right way to read conformal prediction. It converts model
# # quality into interval WIDTH rather than into a coverage claim you cannot
# # check. Note the k-NN row in particular: it overfits the training data, so
# # its train R2 looks excellent while its calibration residuals, and therefore
# # its interval, are not.

# %% [markdown]
# ### Problem 2: How bad does shift have to be?
#
# Conformal coverage assumes exchangeability. Quantify how quickly it degrades
# under covariate shift, and check whether a wider nominal level rescues it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def shifted(n, seed, shift):
#     r = np.random.default_rng(seed)
#     x = r.uniform(-3 + shift, 3 + shift, n)
#     noise = 0.3 + 0.9 * np.abs(x)
#     y = np.sin(1.5 * x) * 2.0 + 0.5 * x + r.normal(0, noise)
#     return x.reshape(-1, 1), y
#
# print(f"  {'shift':<10}" + "".join(f"{f'alpha={a}':>13}" for a in (0.10, 0.05, 0.01)))
# for sh in (0.0, 0.25, 0.5, 1.0, 2.0):
#     row = ""
#     for a_ in (0.10, 0.05, 0.01):
#         qs = conformal_quantile(np.abs(ycal - model.predict(Xcal)), a_)
#         Xs, ys = shifted(3000, 31, sh)
#         row += f"{np.mean(np.abs(ys - model.predict(Xs)) <= qs):>13.1%}"
#     print(f"  {sh:<10.2f}{row}")
#
# # Coverage falls steadily as the test distribution moves away from the
# # calibration distribution, and asking for a stricter nominal level does not
# # rescue it: every column degrades together. The guarantee was conditional on
# # exchangeability, and no choice of alpha restores an assumption that is
# # false.
# #
# # This is the honest limitation. Conformal prediction is not robust to
# # distribution shift, it is exact under exchangeability. In practice that
# # means recalibrating on data from the new site, batch or platform, which
# # requires labelled examples from it. Weighted conformal methods exist for
# # known covariate shift, but they need the shift to be estimable.

# %% [markdown]
# ### Problem 3: Conformal risk for a decision
#
# A prediction set is only useful if it changes what you do. Build a rule that
# treats a patient when the set contains only "responder", defers when the set
# is ambiguous, and measure the error rate among the acted-on cases.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# Xa2, ya2 = make_cls(1500, 41); Xb2, yb2 = make_cls(1500, 42)
# Xc2, yc2 = make_cls(4000, 43)
# clf2 = RandomForestClassifier(n_estimators=300, random_state=1).fit(Xa2, ya2)
# print(f"  {'alpha':<9}{'acted on':>11}{'deferred':>11}{'error | acted on':>19}"
#       f"{'error if forced':>18}")
# for a_ in (0.20, 0.10, 0.05, 0.01):
#     s2 = 1 - clf2.predict_proba(Xb2)[np.arange(len(yb2)), yb2]
#     qq = conformal_quantile(s2, a_)
#     sets2 = (1 - clf2.predict_proba(Xc2)) <= qq
#     singleton = sets2.sum(1) == 1
#     pred_single = sets2[singleton].argmax(1)
#     err_act = np.mean(pred_single != yc2[singleton]) if singleton.any() else np.nan
#     err_all = np.mean(clf2.predict(Xc2) != yc2)
#     print(f"  {a_:<9.2f}{singleton.mean():>11.1%}{1-singleton.mean():>11.1%}"
#           f"{err_act:>19.1%}{err_all:>18.1%}")
#
# # The last column is the error rate if you force a single label for every
# # sample, which is what a normal classifier does. The fourth column is the
# # error rate among the cases the conformal rule was willing to act on.
# #
# # Acting only on singleton sets buys a much lower error rate, paid for by
# # deferring the hard cases. As alpha shrinks, fewer cases get a singleton set
# # and the error among those falls further.
# #
# # That trade is the practical value of conformal prediction in a clinical or
# # screening setting: it identifies WHICH predictions are trustworthy rather
# # than reporting one confidence number for all of them. The deferred cases
# # are not a failure, they are the cases that should go to a human.

# %% [markdown]
# ## What to take away
#
# 1. Split conformal gives **finite-sample, distribution-free, model-agnostic**
#    coverage (39.3). The only requirement is exchangeability.
# 2. The $n+1$ and the ceiling in (39.2) are what make it exact. A plain
#    empirical quantile undercovers.
# 3. Coverage is **marginal**. Use CQR (39.4) when you need the width to
#    reflect local uncertainty.
# 4. **Prediction sets** (39.5) express ambiguity, and the empty set flags a
#    sample unlike anything you calibrated on.
# 5. A better model does not improve coverage, it narrows the interval.
# 6. Grouped data and distribution shift break exchangeability, and no choice
#    of $\alpha$ repairs it.
#
# **Next:** `29_simulation_and_benchmarking.py`
